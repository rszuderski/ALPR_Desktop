"""One approved I/O-only continuation. Never rewrites a historical receipt.

The signed experiment, exact writer SHA pair, evidence bundle, and exact runner
wiring are all required. Other runs and any other code delta remain blocked.
"""
import ast
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import zipfile

from .eval396 import canonical, contained, digest, require

RUN_NAME = "EVAL396_E2E_run_20261010_130727_74a41238"
RUN_RELATIVE = "Workspace/E-MZ-DN-01/evaluation_runs/" + RUN_NAME
PROTOCOL_SHA = "9ddf557a7f23b2a497f08ba65edb46b125dff9e619f3d8efcd7f0d18c937e39a"
APPROVAL_SHA = "df3eb7559e4d2593520fd93c2a594ab799a3ae5768665a1737e5a7c43b549cb6"
OLD_MT_SHA = "dfcd0d637ca04f21875bebc4ad38fa504faaac385044f32e0d6587f41747642f"
NEW_MT_SHA = "c0e9ec16951381c6e4274e055153cfd4663f660afe104faefb439431060e11b1"
OLD_E2E_SHA = "86d5d9ce18ff7d4ce9ffc1ff0e0362d3fa3457d3a22041c0b077b5ed8946150e"
MT_FILE = "auto_annotation_tool/ranking/mt_eval_runner.py"
E2E_FILE = "auto_annotation_tool/ranking/e2e_eval_runner.py"
HELPER_FILE = "auto_annotation_tool/ranking/e2e_io_compat.py"
REPORT_FILE = "auto_annotation_tool/ranking/e2e_eval_report.py"
BUNDLE_RELATIVE = "auto_annotation_tool/ranking/protocols/eval396_e2e_io_compat_20261010"
DOCUMENT_RELATIVE = BUNDLE_RELATIVE + "/compatibility.json"
APPLIED_FILE = "io_compatibility_applied.json"
ERROR = "E2E_RUNTIME_CODE_CHANGED"

# These substitutions are the entire permitted integration in the E2E runner.
# Reconstructing it from the pinned original rejects even unrelated comments.
WIRING = (
    ('from . import e2e_artifacts as storage\n',
     'from . import e2e_artifacts as storage\nfrom . import e2e_io_compat as io_compat\n'),
    ('MIN_FREE_BYTES = 2 * 1024**3\n',
     'MIN_FREE_BYTES = 2 * 1024**3\nIO_COMPAT_DOCUMENT_SHA = "__DOCUMENT_SHA__"\n'),
    ('        "ranking/e2e_eval_runner.py", "ranking/e2e_artifacts.py", "ranking/e2e_eval_metrics.py",\n',
     '        "ranking/e2e_eval_runner.py", "ranking/e2e_artifacts.py", "ranking/e2e_eval_metrics.py",\n'
     '        "ranking/e2e_io_compat.py", "ranking/e2e_eval_report.py",\n'
     '        "ranking/protocols/eval396_e2e_io_compat_20261010/compatibility.json",\n'),
    ('def _verify_prepared(prepared, *, runtime=True):\n',
     'def _verify_prepared(prepared, *, runtime=True, resume_dir=None):\n'),
    ('        require(_code_receipt() == p["code_sha256"], "E2E_RUNTIME_CODE_CHANGED")\n',
     '        context["io_compatibility"] = io_compat.verify_runtime_code(\n'
     '            p, _code_receipt(), resume_dir, IO_COMPAT_DOCUMENT_SHA)\n'),
    ('    _verify_prepared(prepared)\n    p, inputs = prepared["protocol"], prepared["inputs"]\n',
     '    context = _verify_prepared(prepared, resume_dir=resume_dir)\n    p, inputs = prepared["protocol"], prepared["inputs"]\n'),
    ('            ledger = storage.recover(path, existing["ledger"])\n',
     '            io_compat.record_application(path, context.get("io_compatibility"))\n'
     '            ledger = storage.recover(path, existing["ledger"])\n'),
    ('            context = _verify_prepared(prepared)\n            inspected = storage.inspect_ledger(path, allowed)\n',
     '            context = _verify_prepared(prepared, resume_dir=path)\n            inspected = storage.inspect_ledger(path, allowed)\n'),
    ('    return {name: mt._sha(contained(path, name)) for name in sorted(names)}\n',
     '    if (path / io_compat.APPLIED_FILE).exists():\n        names.add(io_compat.APPLIED_FILE)\n'
     '    return {name: mt._sha(contained(path, name)) for name in sorted(names)}\n'),
    ('    prepared, approval, manifest = _read_snapshot(path)\n    context = _verify_prepared(prepared, runtime=False)\n',
     '    prepared, approval, manifest = _read_snapshot(path)\n'
     '    io_application = io_compat.read_application(path, prepared["protocol"], IO_COMPAT_DOCUMENT_SHA)\n'
     '    context = _verify_prepared(prepared, runtime=False)\n'),
    ('        "ledger": inspected, "pairs": assembled, "scoring": scoring,\n',
     '        "ledger": inspected, "pairs": assembled, "scoring": scoring, "io_compatibility": io_application,\n'),
)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def patched_runner_bytes(original, document_sha):
    require(sha_bytes(original) == OLD_E2E_SHA, ERROR + ": integration baseline")
    require(len(document_sha) == 64 and all(c in "0123456789abcdef" for c in document_sha), ERROR + ": document SHA")
    value = original
    for before, after in WIRING:
        before = before.encode("utf-8")
        after = after.replace("__DOCUMENT_SHA__", document_sha).encode("utf-8")
        require(value.count(before) == 1, ERROR + ": integration scope")
        value = value.replace(before, after, 1)
    return value


def verify_writer_scope(original, corrected):
    require(sha_bytes(original) == OLD_MT_SHA and sha_bytes(corrected) == NEW_MT_SHA, ERROR + ": writer SHA pair")
    trees = [ast.parse(value) for value in (original, corrected)]
    for tree in trees:
        writers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_write_json"]
        require(len(writers) == 1, ERROR + ": writer scope")
        tree.body.remove(writers[0])
    require(ast.dump(trees[0], include_attributes=False) == ast.dump(trees[1], include_attributes=False),
            ERROR + ": non-I/O AST difference")


def _certificate(protocol, run_dir, document_sha):
    root = Path(__file__).resolve().parents[2]
    require(run_dir is not None and Path(run_dir).resolve() == (root / RUN_RELATIVE).resolve(), ERROR + ": experiment scope")
    run_dir = Path(run_dir).resolve()
    require(protocol.get("protocol_sha256") == PROTOCOL_SHA
            and digest({k: v for k, v in protocol.items() if k != "protocol_sha256"}) == PROTOCOL_SHA,
            ERROR + ": protocol binding")
    document_file = root / DOCUMENT_RELATIVE
    raw = document_file.read_bytes()
    require(sha_bytes(raw) == document_sha, ERROR + ": compatibility document")
    document = json.loads(raw)
    require(document["schema"] == "alpr.e2e.io_compatibility.v1"
            and document["run_relative_path"] == RUN_RELATIVE
            and document["protocol_sha256"] == PROTOCOL_SHA
            and document["approval_contract_sha256"] == APPROVAL_SHA
            and document["writer_sha256"] == {"original": OLD_MT_SHA, "corrected": NEW_MT_SHA}, ERROR + ": certificate scope")
    evidence = {}
    for name, sha in document["evidence_files_sha256"].items():
        data = contained(document_file.parent, name).read_bytes()
        require(sha_bytes(data) == sha, ERROR + ": compatibility evidence " + name)
        evidence[name] = data
    verify_writer_scope(evidence["original_mt_eval_runner.py.txt"], evidence["corrected_mt_eval_runner.py.txt"])
    expected_runner = patched_runner_bytes(evidence["original_e2e_eval_runner.py.txt"], document_sha)
    for name, sha in document["immutable_run_files_sha256"].items():
        require(sha_bytes(contained(run_dir, name).read_bytes()) == sha, ERROR + ": immutable run binding " + name)
    stored_protocol = json.loads((run_dir / "protocol.json").read_bytes())
    approval = json.loads((run_dir / "approval.json").read_bytes())
    require(stored_protocol == protocol and approval["contract_sha256"] == APPROVAL_SHA
            and approval["operator_signature"] == "rs" and approval["protocol_sha256"] == PROTOCOL_SHA
            and digest({k: v for k, v in approval.items() if k != "contract_sha256"}) == APPROVAL_SHA,
            ERROR + ": original approval")
    old = protocol["code_sha256"]
    mt_path, e2e_path = str((root / MT_FILE).resolve()), str((root / E2E_FILE).resolve())
    require(old.get(mt_path) == OLD_MT_SHA and old.get(e2e_path) == OLD_E2E_SHA, ERROR + ": historical code binding")
    expected = {**old, mt_path: NEW_MT_SHA, e2e_path: sha_bytes(expected_runner),
                str((root / HELPER_FILE).resolve()): document["compatibility_checker_sha256"],
                str((root / REPORT_FILE).resolve()): document["provenance_exporter_sha256"],
                str(document_file.resolve()): document_sha}
    result = {"schema": "alpr.e2e.io_compatibility_applied.v1", "run_dir": str(run_dir),
              "protocol_sha256": PROTOCOL_SHA, "approval_contract_sha256": APPROVAL_SHA,
              "document_sha256": document_sha, "document": document,
              "historical_code_sha256": deepcopy(old), "effective_code_sha256": expected}
    result["certificate_sha256"] = digest(result)
    return result


def _check_initial_recovery_state(path, certificate):
    baseline = certificate["document"]["recovery_baseline"]
    for name, sha in baseline["files_sha256"].items():
        require(sha_bytes(contained(path, name).read_bytes()) == sha, ERROR + ": recovery baseline " + name)
    ledger = json.loads((path / "progress.json").read_bytes())
    require(len(ledger["records"]) == baseline["ledger_records"] == 588, ERROR + ": recovery ledger")


def _check_retained_baseline(path, certificate):
    """Continuation must retain every originally committed forward/scene."""
    root = Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(io.BytesIO((root / BUNDLE_RELATIVE / "original_run_binding.zip").read_bytes())) as archive:
        original = json.loads(archive.read("progress.json"))["records"]
        pending = json.loads(archive.read("pending.json"))
    records = json.loads((path / "progress.json").read_bytes())["records"]
    require(all(records.get(name) == sha for name, sha in original.items()), ERROR + ": original results missing or changed")
    if pending["file"] in records:
        require(records[pending["file"]] == pending["sha256"], ERROR + ": recovered result changed")
    else:
        expected = certificate["document"]["recovery_baseline"]["files_sha256"]["pending.json"]
        require(sha_bytes((path / "pending.json").read_bytes()) == expected, ERROR + ": original pending missing")


def verify_runtime_code(protocol, actual, resume_dir, document_sha):
    """Strict equality remains the default; only one signed run has an exception."""
    if actual == protocol["code_sha256"]:
        return None
    try:
        certificate = _certificate(protocol, resume_dir, document_sha)
        require(actual == certificate["effective_code_sha256"], ERROR + ": code outside approved patch")
        path = Path(resume_dir).resolve()
        applied = path / APPLIED_FILE
        if applied.exists():
            require(json.loads(applied.read_bytes()) == certificate, ERROR + ": applied certificate changed")
            _check_retained_baseline(path, certificate)
        else:
            _check_initial_recovery_state(path, certificate)
        return certificate
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if str(exc).startswith(ERROR):
            raise
        raise ValueError(ERROR + ": compatibility verification failed: " + str(exc)) from exc


def record_application(path, certificate):
    """Explicit Resume only, under execution lock, before native recovery."""
    if certificate is None:
        return
    path = Path(path).resolve()
    require(str(path) == certificate["run_dir"], ERROR + ": application path")
    if not (path / APPLIED_FILE).exists():
        _check_initial_recovery_state(path, certificate)
    from .e2e_artifacts import immutable_json
    immutable_json(path / APPLIED_FILE, certificate)


def read_application(path, protocol, document_sha):
    """Historical read: verify a saved addendum without requiring current runtime."""
    file = Path(path) / APPLIED_FILE
    if not file.exists():
        return None
    try:
        expected = _certificate(protocol, path, document_sha)
        require(json.loads(file.read_bytes()) == expected, ERROR + ": applied certificate changed")
        _check_retained_baseline(Path(path), expected)
        return expected
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if str(exc).startswith(ERROR):
            raise
        raise ValueError(ERROR + ": applied certificate verification failed: " + str(exc)) from exc
