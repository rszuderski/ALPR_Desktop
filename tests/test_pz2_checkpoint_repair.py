from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import repair_pz2_az_checkpoints as repair
from auto_annotation_tool.gui.z3_review_runtime import build_review_reference_snapshot
from auto_annotation_tool.registry.az_registry import AZRegistry
from auto_annotation_tool.registry.az_revision_store import AZRevisionStore, compute_az_payload_sha256, canonicalize_az_payload
from auto_annotation_tool.registry.pz2_az_adapter import pz2_metadata_to_az_payload
from auto_annotation_tool.registry.crop_identity import build_pz1_crop_identity, compute_crop_identity_sha256


@pytest.fixture
def scenario(tmp_path):
    workspace = tmp_path / "Workspace"
    preview = workspace / "9_projects/Target/3_cropped_characters/run_001"
    images = preview / "images"
    images.mkdir(parents=True)
    registry = AZRegistry.for_workspace(workspace)
    registry.initialize()
    project = "PRJ-TEST"
    with registry.database.transaction() as con:
        con.execute("INSERT INTO projects(project_id,display_name) VALUES (?,?)", (project, "Target"))
    store = AZRevisionStore(registry.database)
    metadata = {}
    initial_revisions = {}
    for index, name in enumerate(["missing", "imported", "correct", "pending", "excluded"]):
        source_sha = hashlib.sha256(name.encode()).hexdigest()
        identity = build_pz1_crop_identity(source_image_id="img-sha256-" + source_sha,
            source_annotation_id="ann-" + name, source_geometry_hash=source_sha,
            interpolation="lanczos4", output_width=64, output_height=32)
        identity_sha = compute_crop_identity_sha256(identity)
        image = images / (name + ".jpg")
        image.write_bytes(name.encode())
        package = name in {"imported", "correct"}
        registered = registry.register_crop_artifact(crop_identity_sha256=identity_sha,
            identity_mode=identity["identity_mode"], source_file_sha256=source_sha,
            artifact_path=image, artifact_sha256=repair.file_sha(image), size_bytes=image.stat().st_size,
            width=64, height=32, source_annotation_id="ann-" + name, source_geometry_hash=source_sha,
            project_id=project, iteration_num=1, source_mode="fixture", source_plate_key=name,
            source_at_ref="azpkg:PKG-TEST" if package else "fixture")
        row = {"crop_id": registered.crop_id, "crop_identity_sha256": identity_sha,
            "ground_truth_text": "A", "ground_truth_source": "manual_z2",
            "source_annotation_id": "ann-" + name, "status": "perfect",
            "characters": [{"character": "A", "bbox": [4, 4, 20, 28], "confidence": 1.0,
                            "method": "manual", "source_kind": "local_manual"}],
            "plate_layout": "single_row", "plate_layout_override": "single_row",
            "gold_state": {"approved": True, "excluded": False, "candidate": True},
            "source_info": {"bucket": "local_manual", "origin": "preview_editor"},
            "review_state": {"status": "approved", "approved_by": "human", "approved_at": "2026-09-24T12:00:00+02:00"}}
        row["review_state"]["approved_reference"] = build_review_reference_snapshot(row)
        if package:
            row["append_provenance"] = {"package_id": "PKG-TEST", "target_project_id": project,
                "iteration_num": 1, "source_at_ref": "azpkg:PKG-TEST", "source_plate_key": name,
                "artifact_id": registered.artifact_id, "entry_id": name}
        if name == "pending":
            row.pop("review_state")
            row["gold_state"]["approved"] = False
        if name == "excluded":
            row["gold_state"]["excluded"] = True
        metadata[name] = row
        if name != "missing":
            payload = pz2_metadata_to_az_payload(row, image_width=64, image_height=32)
            status = "approved"
            if name == "imported":
                payload["gold_state"]["approved"] = False
                payload["status"] = "needs_fix"
                status = "imported_pending_review"
            elif name == "pending":
                status = "needs_fix"
            elif name == "excluded":
                payload["characters"][0]["character"] = "X"
                status = "excluded"
            initial_revisions[name] = store.save_revision(crop_id=registered.crop_id, payload=payload,
                source_kind="fixture", trust_state="local_manual", bind_project_id=project, effective_status=status)
    (preview / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (workspace / "campaigns_registry.json").write_text('{"active_project":"Target"}', encoding="utf-8")

    def audit_row(name):
        row = metadata[name]
        revision = initial_revisions.get(name)
        return {"plate_id": name, "crop_id": row["crop_id"], "identity_sha256": row["crop_identity_sha256"],
                "metadata_payload_sha256": compute_az_payload_sha256(pz2_metadata_to_az_payload(row, image_width=64, image_height=32)),
                "package_id": "PKG-TEST" if name in {"imported", "correct"} else None,
                "az_revision_id": revision.az_revision_id if revision else None,
                "stored_payload_sha256": revision.payload_sha256 if revision else None}

    audit = {"project_id": project, "iteration": 1, "counts": {"records": len(metadata)},
             "missing_bindings": [audit_row("missing")],
             "package_rows": [audit_row("imported"), audit_row("correct")],
             "pending_rows": [audit_row("pending")],
             "differences": [{"local_decision": "excluded", "plate_id": "excluded"}]}
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    context = repair.make_context(workspace, audit_path, project_id=project, preview_dir=preview)
    return {"context": context, "metadata": metadata, "registry": registry, "revisions": initial_revisions,
            "audit_path": audit_path, "workspace": workspace, "tmp": tmp_path}


def fingerprint(context):
    with repair.closing(repair.read_only(context.database)) as connection:
        return repair.sql_fingerprints(connection)


def files_snapshot(workspace):
    return {str(path.relative_to(workspace)): (repair.file_sha(path), repair.file_signature(path))
            for path in workspace.rglob("*") if path.is_file() and not path.name.endswith(("-wal", "-shm"))}


def test_dry_run_lists_exact_scope_and_changes_no_source_files(scenario):
    context = scenario["context"]
    before = files_snapshot(context.workspace)
    result = repair.run_repair(context)
    assert result["dry_run"] == {"missing_bindings_to_create": 1, "package_bindings_to_promote": 1,
        "already_correct": 1, "human_pending_to_skip": 1, "excluded_differences_skip": 1, "sets_disjoint": True}
    assert {item["crop_id"] for item in result["plan"] if item["action"] != "skip"} == {
        scenario["metadata"][name]["crop_id"] for name in ["missing", "imported"]}
    assert result["backup"] is None and not result["applied"]
    assert files_snapshot(context.workspace) == before
    assert result["changes"]["changed_tables"] == []


def test_apply_preserves_metadata_membership_history_provenance_and_exempt_rows(scenario):
    context = scenario["context"]
    metadata_bytes = context.metadata_path.read_bytes()
    metadata_before = deepcopy(scenario["metadata"])
    initial = fingerprint(context)
    with repair.closing(repair.read_only(context.database)) as con:
        bindings_before = {row["crop_id"]: dict(row) for row in con.execute("SELECT * FROM project_crop_az")}
        old_revisions = {row["az_revision_id"]: dict(row) for row in con.execute("SELECT * FROM az_revisions")}
    result = repair.run_repair(context, apply=True)
    assert result["changes"]["created_bindings"] == result["changes"]["changed_bindings"] == 1
    assert result["changes"]["created_revisions"] == 2
    assert set(result["changes"]["changed_tables"]) == {"az_revisions", "project_crop_az"}
    assert result["after"]["project_crop_az"] == 5
    assert result["after"]["package_approved"] == 2
    assert result["after"]["package_imported_pending_review"] == 0
    assert result["after"]["pending"] == 1
    assert result["integrity_check"] == "ok" and result["foreign_key_errors"] == []
    assert context.metadata_path.read_bytes() == metadata_bytes
    assert json.loads(context.metadata_path.read_text(encoding="utf-8")) == metadata_before
    after = fingerprint(context)
    assert all(after[name] == value for name, value in initial.items() if name not in {"az_revisions", "project_crop_az"})
    with repair.closing(repair.read_only(context.database)) as con:
        for name in ["correct", "pending", "excluded"]:
            crop_id = scenario["metadata"][name]["crop_id"]
            assert dict(con.execute("SELECT * FROM project_crop_az WHERE crop_id=?", (crop_id,)).fetchone()) == bindings_before[crop_id]
        for revision_id, row in old_revisions.items():
            assert dict(con.execute("SELECT * FROM az_revisions WHERE az_revision_id=?", (revision_id,)).fetchone()) == row
        for name in ["missing", "imported"]:
            crop_id = scenario["metadata"][name]["crop_id"]
            revision = con.execute("SELECT ar.*,pca.effective_status FROM az_revisions ar JOIN project_crop_az pca ON pca.az_revision_id=ar.az_revision_id WHERE pca.crop_id=?", (crop_id,)).fetchone()
            expected = canonicalize_az_payload(pz2_metadata_to_az_payload(metadata_before[name], image_width=64, image_height=32))
            assert revision["effective_status"] == "approved"
            assert json.loads(revision["payload_json"]) == expected
            if name == "imported":
                assert revision["parent_revision_id"] == scenario["revisions"][name].az_revision_id
    backup = Path(result["backup"])
    assert (backup / "metadata.json").read_bytes() == metadata_bytes
    with repair.closing(repair.read_only(backup / "alpr_registry.sqlite3")) as con:
        assert repair.sql_fingerprints(con) == initial
    manifest = json.loads((backup / "backup_manifest.json").read_text(encoding="utf-8"))
    assert manifest["database_source"] == str(context.database)


def test_second_apply_is_physical_and_logical_noop_without_writer_or_backup(scenario, monkeypatch):
    context = scenario["context"]
    repair.run_repair(context, apply=True)
    before = files_snapshot(context.workspace)
    tables = fingerprint(context)
    writer = Mock(side_effect=AssertionError("Writer called for a no-op"))
    monkeypatch.setattr(AZRevisionStore, "save_revision", writer)
    repeated = repair.run_repair(context, apply=True)
    assert repeated["changes"] == {"created_revisions": 0, "created_bindings": 0, "changed_bindings": 0,
        "changed_timestamps": 0, "changed_tables": [], "metadata_changed": False}
    assert not repeated["applied"] and repeated["backup"] is None
    writer.assert_not_called()
    assert fingerprint(context) == tables and files_snapshot(context.workspace) == before


@pytest.mark.parametrize("change,reason", [
    ("payload", "payload_changed_since_audit"), ("reviewer", "no_current_human_approval"),
    ("excluded", "no_current_human_approval"), ("reference", "no_current_human_approval"),
    ("identity", "identity_changed"),
])
def test_changed_audit_preconditions_skip_without_guessing(scenario, change, reason):
    row = scenario["metadata"]["missing"]
    if change == "payload": row["characters"][0]["bbox"][0] += 1
    elif change == "reviewer": row["review_state"]["approved_by"] = "system"
    elif change == "excluded": row["gold_state"]["excluded"] = True
    elif change == "reference": row["review_state"]["approved_reference"]["ground_truth_text"] = "B"
    elif change == "identity": row["crop_identity_sha256"] = "f" * 64
    scenario["context"].metadata_path.write_text(json.dumps(scenario["metadata"]), encoding="utf-8")
    result = repair.run_repair(scenario["context"])
    item = next(item for item in result["plan"] if item["plate_id"] == "missing")
    assert item["action"] == "skip" and item["reason"] == reason


def test_package_provenance_change_is_not_overwritten(scenario):
    scenario["metadata"]["imported"]["append_provenance"]["package_id"] = "OTHER-PACKAGE"
    scenario["context"].metadata_path.write_text(json.dumps(scenario["metadata"]), encoding="utf-8")
    result = repair.run_repair(scenario["context"], apply=True)
    assert result["changes"]["changed_bindings"] == 0
    assert next(item for item in result["skipped"] if item["plate_id"] == "imported")["reason"] == "package_provenance_changed"


def test_preconditions_are_rechecked_after_backup_under_write_transaction(scenario, monkeypatch):
    context = scenario["context"]
    backup = repair.create_backup
    def changed_after_backup(*args, **kwargs):
        result = backup(*args, **kwargs)
        with scenario["registry"].database.transaction() as con:
            con.execute("DELETE FROM iteration_crop_members WHERE crop_id=?", (scenario["metadata"]["missing"]["crop_id"],))
        return result
    monkeypatch.setattr(repair, "create_backup", changed_after_backup)
    result = repair.run_repair(context, apply=True)
    assert result["changes"]["created_bindings"] == 0
    assert next(item for item in result["skipped"] if item["plate_id"] == "missing")["reason"] == "missing_crop_or_membership"


def test_writer_failure_rolls_back_entire_repair(scenario, monkeypatch):
    context = scenario["context"]
    before = fingerprint(context)
    original = AZRevisionStore.save_revision
    calls = []
    def fail_second(self, **kwargs):
        calls.append(kwargs["crop_id"])
        if len(calls) == 2: raise RuntimeError("Injected SQL failure")
        return original(self, **kwargs)
    monkeypatch.setattr(AZRevisionStore, "save_revision", fail_second)
    with pytest.raises(RuntimeError, match="Injected SQL failure"):
        repair.run_repair(context, apply=True)
    assert fingerprint(context) == before


def test_metadata_change_during_apply_rolls_back_sql(scenario, monkeypatch):
    context = scenario["context"]
    before = fingerprint(context)
    original = AZRevisionStore.save_revision
    def external_editor(self, **kwargs):
        saved = original(self, **kwargs)
        context.metadata_path.write_bytes(context.metadata_path.read_bytes() + b" ")
        return saved
    monkeypatch.setattr(AZRevisionStore, "save_revision", external_editor)
    with pytest.raises(repair.RepairError, match="Metadata zmieniło"):
        repair.run_repair(context, apply=True)
    assert fingerprint(context) == before


def test_external_transaction_writer_requires_active_transaction(scenario):
    with repair.closing(repair.read_only(scenario["context"].database)) as con:
        with pytest.raises(ValueError, match="aktywnej transakcji"):
            AZRevisionStore(scenario["registry"].database).save_revision(
                crop_id="unused", payload={}, source_kind="unused", trust_state="unused", connection=con)


def test_duplicate_audit_scope_and_wrong_project_are_rejected_before_write(scenario):
    audit = json.loads(scenario["audit_path"].read_text(encoding="utf-8"))
    audit["package_rows"].append(audit["missing_bindings"][0])
    scenario["audit_path"].write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(repair.RepairError, match="rozłączne"):
        repair.make_context(scenario["workspace"], scenario["audit_path"], project_id="PRJ-TEST", preview_dir=scenario["context"].preview)
    with pytest.raises(repair.RepairError, match="Projekt/iteracja"):
        repair.make_context(scenario["workspace"], scenario["audit_path"], project_id="OTHER", preview_dir=scenario["context"].preview)


def test_backup_cannot_write_into_images(scenario):
    before = fingerprint(scenario["context"])
    with pytest.raises(repair.RepairError, match="_registry/backups"):
        repair.run_repair(scenario["context"], apply=True, backup_directory=scenario["context"].preview / "images")
    assert fingerprint(scenario["context"]) == before
