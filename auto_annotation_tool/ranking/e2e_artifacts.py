"""Atomic E2E ledger. An interrupted, journaled forward is never repeated."""
from copy import deepcopy
from pathlib import Path

from .eval396 import canonical, contained, digest, require
from .mt_eval_runner import _sha, _write_json


def detection_path(pair_id, scene_sha, index):
    return f"raw/{pair_id}/detections/{scene_sha}/{index:04d}.json"


def scene_path(pair_id, scene_sha):
    return f"raw/{pair_id}/scenes/{scene_sha}.json"


def expected_paths(inputs, pairs):
    paths = set()
    for pair in pairs:
        for scene in inputs[pair["mt"]]:
            paths.add(scene_path(pair["id"], scene["scene_sha256"]))
            paths.update(detection_path(pair["id"], scene["scene_sha256"], d["index"]) for d in scene["predictions"])
    return paths


def signed_record(payload):
    require("record_sha256" not in payload, "E2E_RECORD_ALREADY_SIGNED")
    return {**payload, "record_sha256": digest(payload)}


def check_record(row):
    require(row.get("record_sha256") == digest({k: v for k, v in row.items() if k != "record_sha256"}),
            "E2E_RAW_RECORD_CHANGED")


def read_json(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inspect_ledger(path, allowed):
    """Read only; do not adopt orphan records or clean journals here."""
    path = Path(path)
    ledger = read_json(path / "progress.json")
    require(ledger.get("schema") == "alpr.e2e.progress.v1" and isinstance(ledger.get("records"), dict), "E2E_LEDGER_INVALID")
    require(set(ledger["records"]) <= allowed, "E2E_UNKNOWN_INDEXED_RECORD")
    disk = {p.relative_to(path).as_posix() for p in (path / "raw").rglob("*.json")}
    require(set(ledger["records"]) <= disk, "E2E_MISSING_INDEXED_RECORD")
    require(disk <= allowed, "E2E_UNKNOWN_RAW_RECORD")
    rows = {}
    for name, sha in ledger["records"].items():
        file = contained(path, name)
        require(_sha(file) == sha, "E2E_RAW_FILE_CHANGED: " + name)
        row = read_json(file)
        check_record(row)
        rows[name] = row
    journal = path / "pending.json"
    pending = None
    if journal.exists():
        pending = read_json(journal)
        require(pending.get("schema") == "alpr.e2e.pending.v1"
                and pending.get("transaction_sha256") == digest({k: v for k, v in pending.items() if k != "transaction_sha256"}),
                "E2E_PENDING_CHANGED")
        name, row = pending["file"], pending["record"]
        require(name in allowed, "E2E_UNKNOWN_PENDING_RECORD")
        check_record(row)
        import hashlib
        require(hashlib.sha256(canonical(row) + b"\n").hexdigest() == pending["sha256"], "E2E_PENDING_RAW_SHA_CHANGED")
        if name in disk:
            require(_sha(contained(path, name)) == pending["sha256"], "E2E_PENDING_DISK_MISMATCH")
        if name in rows:
            require(rows[name] == row and ledger["records"][name] == pending["sha256"], "E2E_PENDING_LEDGER_MISMATCH")
        rows[name] = row
    orphan = disk - set(ledger["records"])
    require(not orphan or (pending is not None and orphan == {pending["file"]}), "E2E_UNJOURNALED_ORPHAN")
    return {"ledger": ledger, "rows": rows, "pending": pending,
            "pending_file_sha256": _sha(journal) if pending else None,
            "recovery_required": pending is not None}


def publish(path, ledger, name, row):
    path = Path(path)
    require(name not in ledger["records"], "E2E_RECORD_ALREADY_INDEXED")
    require(not (path / "prediction_manifest.json").exists(), "E2E_PREDICTIONS_SEALED")
    require(not (path / "pending.json").exists(), "E2E_RECOVERY_REQUIRED")
    check_record(row)
    import hashlib
    sha = hashlib.sha256(canonical(row) + b"\n").hexdigest()
    pending = {"schema": "alpr.e2e.pending.v1", "file": name, "record": row, "sha256": sha}
    pending["transaction_sha256"] = digest(pending)
    _write_json(path / "pending.json", pending, exclusive=True)
    require(_write_json(contained(path, name), row, exclusive=True) == sha, "E2E_RAW_PUBLICATION_CHANGED")
    updated = deepcopy(ledger)
    updated["records"][name] = sha
    _write_json(path / "progress.json", updated)
    ledger.clear()
    ledger.update(updated)
    (path / "pending.json").unlink()


def recover(path, inspected):
    """Only explicit Resume calls this, after all rows were semantically checked."""
    path = Path(path)
    pending = inspected["pending"]
    if not pending:
        return inspected["ledger"]
    require(not (path / "prediction_manifest.json").exists(), "E2E_SEALED_JOURNAL_CONFLICT")
    require(_sha(path / "pending.json") == inspected["pending_file_sha256"], "E2E_PENDING_CHANGED_DURING_RECOVERY")
    ledger = deepcopy(inspected["ledger"])
    name, sha = pending["file"], pending["sha256"]
    file = contained(path, name)
    if file.exists():
        require(_sha(file) == sha, "E2E_RECOVERY_RAW_CHANGED")
    else:
        require(_write_json(file, pending["record"], exclusive=True) == sha, "E2E_RECOVERY_RAW_CHANGED")
    ledger["records"][name] = sha
    _write_json(path / "progress.json", ledger)
    (path / "pending.json").unlink()
    return ledger


def immutable_json(path, value):
    """Idempotent finalization; never overwrite an existing finished artifact."""
    path = Path(path)
    if path.exists():
        require(read_json(path) == value, "E2E_IMMUTABLE_FILE_CONFLICT: " + path.name)
        return _sha(path)
    return _write_json(path, value, exclusive=True)


def seal_predictions(path, protocol_sha, ledger, allowed):
    require(set(ledger["records"]) == allowed and not (Path(path) / "pending.json").exists(), "E2E_PREDICTIONS_INCOMPLETE")
    payload = {"schema": "alpr.e2e.prediction_manifest.v1", "protocol_sha256": protocol_sha,
               "files": deepcopy(ledger["records"])}
    immutable_json(Path(path) / "prediction_manifest.json", payload)
    return payload
