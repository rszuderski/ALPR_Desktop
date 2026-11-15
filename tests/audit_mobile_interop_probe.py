"""Read-only application audit; generated test archives stay in output."""
from pathlib import Path
import sys
import hashlib
import json
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from test_mobile_model_package_contract import export_fixture_model
from auto_annotation_tool.exporters.mobile_model_exporter import MobileAlprPackageExporter, MobileAlprPackageRequest
from auto_annotation_tool.mobile_acquisition import registration_key
from auto_annotation_tool.ranking.mobile_human_review import normalize_registration
from auto_annotation_tool.ranking.mobile_mt_invocations import group_mt_invocations
from auto_annotation_tool.ranking.mobile_package_experiments import read_mobile_report_bundle


def rewrite(source, target, edit):
    with zipfile.ZipFile(source) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    edit(manifest, entries)
    entries["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(target, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return target


result = {"normalization": {"raw": "wx 123ab", "crop_key": registration_key("wx 123ab"),
                            "research_key": normalize_registration("wx 123ab")}}
(ROOT / "output").mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="handoff-audit-", dir=ROOT / "output") as temporary:
    root = Path(temporary)
    models = {role: export_fixture_model(root / role, role)[0] for role in ("plate", "character")}
    exporter = MobileAlprPackageExporter()
    package = exporter.export(MobileAlprPackageRequest(destination=root / "valid.alprmodel",
                         plate_package=models["plate"], character_package=models["character"]))

    def check_package(name, edit):
        invalid = rewrite(package, root / (name + ".alprmodel"), edit)
        try:
            exporter.validate_package(invalid)
        except Exception as error:
            result[name] = {"accepted": False, "error": str(error)}
        else:
            result[name] = {"accepted": True}

    def wrong_id(manifest, entries):
        manifest["models"]["plate"]["model_id"] = "different-model-id"

    def wrong_sidecar(manifest, entries):
        item = manifest["models"]["plate"]
        name = item["manifest_file"]
        sidecar = json.loads(entries[name])
        sidecar["model_id"] = "different-sidecar-id"
        entries[name] = json.dumps(sidecar).encode()
        item["sha256"][name] = hashlib.sha256(entries[name]).hexdigest()

    check_package("package_child_id_mismatch", wrong_id)
    check_package("package_sidecar_mismatch", wrong_sidecar)
    for name, schema in (("unknown_bundle_schema", "unrelated.schema.v99"),
                         ("known_bundle_schema", "alpr.mobile_research_bundle.v1")):
        path = root / (name + ".alprsession")
        report = json.dumps({"schema": "alpr.mobile_benchmark_report.v1", "report_id": "audit"}).encode()
        manifest = {"schema": schema, "entry_sha256": {"report.json": hashlib.sha256(report).hexdigest()}}
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("report.json", report)
            archive.writestr("manifest.json", json.dumps(manifest))
        bundle = read_mobile_report_bundle(path)
        result[name] = {"accepted": bundle.validation.ok, "kind": bundle.bundle_kind,
                        "warnings": bundle.validation.warnings}

    for name, overrides in (
        ("not_run_with_invocation_and_zero", {"mt_status": "NOT_RUN", "mt_executed": False,
                                            "mt_invocation_id": "invocation-1", "mt_detection_count": 0}),
        ("executed_not_run", {"mt_status": "NOT_RUN", "mt_executed": True,
                              "mt_invocation_id": "invocation-1", "mt_detection_count": 0}),
        ("executed_without_invocation", {"mt_status": "NO_DETECTION", "mt_executed": True,
                                         "mt_detection_count": 0}),
    ):
        row = {"id": "attempt-1", "subject_key": "subject-1", "session_id": "audit", **overrides}
        try:
            groups = group_mt_invocations({row["id"]: row}, "audit")
        except Exception as error:
            result[name] = {"accepted": False, "error": str(error)}
        else:
            group = next(iter(groups.values()))
            result[name] = {"accepted": True, "executed": group.executed, "legacy_identity": group.legacy_identity}

print(json.dumps(result, ensure_ascii=True, indent=2))
