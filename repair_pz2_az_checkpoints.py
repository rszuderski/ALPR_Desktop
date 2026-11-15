"""Audited PZ2/AZ checkpoints only; explicit workspace, no metadata writes.

Example (read only):
  python repair_pz2_az_checkpoints.py --dry-run --workspace C:\\path\\Workspace \
      --audit output/live_checkpoint_preflight_20261004/audit_live.json
Live --apply requires separate operator authorization. Rehearse on a copy first.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid

from auto_annotation_tool.gui.z3_review_runtime import review_approval_is_current
from auto_annotation_tool.registry.az_revision_store import AZRevisionStore, compute_az_payload_sha256
from auto_annotation_tool.registry.database import RegistryDatabase
from auto_annotation_tool.registry.pz2_az_adapter import pz2_metadata_to_az_payload, derive_pz2_revision_context

PROJECT_ID = "PRJ-6A69AEDEBBABDA3590C1"
PREVIEW_REL = "9_projects/MZ_finalny_2026_17596F/3_cropped_characters/run_001_20260923_013511"


class RepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepairContext:
    workspace: Path
    preview: Path
    database: Path
    audit: dict
    project_id: str
    iteration: int

    @property
    def metadata_path(self):
        return self.preview / "metadata.json"


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_signature(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def read_only(path):
    connection = sqlite3.connect(Path(path).absolute().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def sql_fingerprints(connection):
    result = {}
    for table, in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        identifier = '"' + table.replace('"', '""') + '"'
        rows = sorted(json.dumps(tuple(row), ensure_ascii=True, default=str)
                      for row in connection.execute(f"SELECT * FROM {identifier}"))
        result[table] = {"count": len(rows), "sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest()}
    return result


def make_context(workspace, audit_path, *, project_id=PROJECT_ID, iteration=1, preview_dir=PREVIEW_REL):
    # Preserve the caller's logical junction alias; resolve only for containment.
    workspace = Path(workspace).absolute()
    preview = Path(preview_dir)
    preview = preview.absolute() if preview.is_absolute() else workspace / preview
    try:
        preview.resolve().relative_to(workspace.resolve())
    except ValueError as exc:
        raise RepairError("Preview znajduje się poza podanym Workspace.") from exc
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8-sig"))
    if audit.get("project_id") != project_id or audit.get("iteration") != iteration:
        raise RepairError("Projekt/iteracja nie odpowiada audytowi.")
    missing = audit.get("missing_bindings", [])
    package = audit.get("package_rows", [])
    ids = [row["crop_id"] for row in missing + package]
    plates = [row["plate_id"] for row in missing + package]
    if len(ids) != len(set(ids)) or len(plates) != len(set(plates)):
        raise RepairError("Zakresy audytu zawierają duplikaty albo nie są rozłączne.")
    protected = {row.get("crop_id") for row in audit.get("pending_rows", [])}
    protected.update(row.get("crop_id") for row in audit.get("differences", [])
                     if row.get("local_decision") == "excluded")
    if set(ids) & protected:
        raise RepairError("Zakres naprawy obejmuje pending lub excluded poza zakresem.")
    for row in missing + package:
        if not row.get("metadata_payload_sha256") or not row.get("identity_sha256"):
            raise RepairError("Niekompletny audyt payloadu lub tożsamości cropa.")
    database = workspace / "_registry/alpr_registry.sqlite3"
    if not database.is_file() or not (preview / "metadata.json").is_file():
        raise RepairError("Brak istniejącej bazy lub metadata.json w wybranym Workspace.")
    return RepairContext(workspace, preview, database, audit, project_id, iteration)


def current_crop(connection, context, crop_id):
    return connection.execute("""
        SELECT pc.crop_id, pc.identity_sha256, pc.width, pc.height,
               pcm.crop_id AS project_member, icm.crop_id AS iteration_member,
               icm.source_at_ref, icm.source_plate_key, icm.artifact_id,
               pca.az_revision_id, pca.effective_status, ar.payload_sha256,
               (SELECT COUNT(*) FROM az_revisions history WHERE history.crop_id=pc.crop_id) AS revision_count
        FROM plate_crops pc
        LEFT JOIN project_crop_members pcm ON pcm.crop_id=pc.crop_id AND pcm.project_id=?
        LEFT JOIN iteration_crop_members icm ON icm.crop_id=pc.crop_id AND icm.project_id=? AND icm.iteration_num=?
        LEFT JOIN project_crop_az pca ON pca.crop_id=pc.crop_id AND pca.project_id=?
        LEFT JOIN az_revisions ar ON ar.az_revision_id=pca.az_revision_id
        WHERE pc.crop_id=?
    """, (context.project_id, context.project_id, context.iteration, context.project_id, crop_id)).fetchone()


def inspect_record(connection, context, metadata, audited, group):
    item = {"plate_id": audited["plate_id"], "crop_id": audited["crop_id"], "group": group,
            "action": "skip", "reason": None, "revision_id_before": None}

    def skip(reason):
        item["reason"] = reason
        return item

    data = metadata.get(audited["plate_id"])
    if not isinstance(data, dict) or data.get("crop_id") != audited["crop_id"]:
        return skip("metadata_crop_changed")
    crop = current_crop(connection, context, audited["crop_id"])
    if crop is None or not crop["project_member"] or not crop["iteration_member"]:
        return skip("missing_crop_or_membership")
    item["revision_id_before"] = crop["az_revision_id"]
    if crop["identity_sha256"] != audited["identity_sha256"] or data.get("crop_identity_sha256") != audited["identity_sha256"]:
        return skip("identity_changed")
    gold = data.get("gold_state") or {}
    state = data.get("review_state") or {}
    if (not gold.get("approved") or gold.get("excluded") or state.get("approved_by") != "human"
            or not state.get("approved_at") or not review_approval_is_current(data)):
        return skip("no_current_human_approval")
    try:
        payload = pz2_metadata_to_az_payload(data, image_width=crop["width"], image_height=crop["height"])
        payload_sha = compute_az_payload_sha256(payload)
        revision_context = derive_pz2_revision_context(data)
    except (ValueError, TypeError, KeyError) as exc:
        return skip("invalid_payload: " + str(exc))
    if payload_sha != audited["metadata_payload_sha256"]:
        return skip("payload_changed_since_audit")
    if group == "missing" and revision_context.source_kind != "local_manual":
        return skip("local_manual_context_changed")
    if group == "package":
        provenance = data.get("append_provenance") or {}
        expected_package = audited.get("package_id")
        if (not expected_package or provenance.get("package_id") != expected_package
                or provenance.get("target_project_id") != context.project_id
                or provenance.get("iteration_num") != context.iteration
                or provenance.get("source_at_ref") != "azpkg:" + expected_package
                or crop["source_at_ref"] != provenance.get("source_at_ref")
                or crop["source_plate_key"] != provenance.get("source_plate_key")
                or crop["artifact_id"] != provenance.get("artifact_id")):
            return skip("package_provenance_changed")
    if crop["effective_status"] == "approved" and crop["payload_sha256"] == payload_sha:
        return skip("already_correct")
    if group == "missing":
        if crop["az_revision_id"] or crop["revision_count"]:
            return skip("binding_or_revision_appeared")
        item["action"] = "create_missing_binding"
    else:
        if crop["effective_status"] != "imported_pending_review":
            return skip("binding_status_changed")
        if crop["az_revision_id"] != audited["az_revision_id"] or crop["payload_sha256"] != audited["stored_payload_sha256"]:
            return skip("binding_revision_changed")
        item["action"] = "promote_package_binding"
    item.update(payload=payload, payload_sha256=payload_sha, source_kind=revision_context.source_kind,
                trust_state=revision_context.trust_state, source_status=revision_context.source_status)
    return item


def inspect_scope(connection, context, metadata):
    return [inspect_record(connection, context, metadata, row, group)
            for group, rows in [("missing", context.audit["missing_bindings"]), ("package", context.audit["package_rows"])]
            for row in rows]


def state_counts(connection, context, metadata):
    package_ids = [row["crop_id"] for row in context.audit["package_rows"]]
    package_statuses = {}
    if package_ids:
        placeholders = ",".join("?" for _ in package_ids)
        package_statuses = dict(connection.execute(
            f"SELECT effective_status, COUNT(*) FROM project_crop_az WHERE project_id=? AND crop_id IN ({placeholders}) GROUP BY effective_status",
            (context.project_id, *package_ids)))
    approved = sum(bool((row.get("gold_state") or {}).get("approved") and not (row.get("gold_state") or {}).get("excluded")) for row in metadata.values())
    excluded = sum(bool((row.get("gold_state") or {}).get("excluded")) for row in metadata.values())
    return {"records": len(metadata), "approved": approved, "excluded": excluded,
            "pending": len(metadata) - approved - excluded,
            "project_crop_az": connection.execute("SELECT COUNT(*) FROM project_crop_az WHERE project_id=?", (context.project_id,)).fetchone()[0],
            "project_crop_members": connection.execute("SELECT COUNT(*) FROM project_crop_members WHERE project_id=?", (context.project_id,)).fetchone()[0],
            "iteration_crop_members": connection.execute("SELECT COUNT(*) FROM iteration_crop_members WHERE project_id=? AND iteration_num=?", (context.project_id, context.iteration)).fetchone()[0],
            "az_revisions": connection.execute("SELECT COUNT(*) FROM az_revisions").fetchone()[0],
            "package_approved": package_statuses.get("approved", 0),
            "package_imported_pending_review": package_statuses.get("imported_pending_review", 0)}


def create_backup(context, metadata_sha, directory=None):
    base = Path(directory).absolute() if directory else context.workspace / "_registry/backups"
    try:
        relative = base.resolve().relative_to(context.workspace.resolve())
    except ValueError:
        pass
    else:
        if relative.parts[:2] != ("_registry", "backups"):
            raise RepairError("Backup wewnątrz Workspace musi należeć do _registry/backups.")
    backup = base / ("pz2_az_repair_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    backup.mkdir(parents=True, exist_ok=False)
    with closing(read_only(context.database)) as source, closing(sqlite3.connect(backup / "alpr_registry.sqlite3")) as target:
        source.backup(target)
    shutil.copy2(context.metadata_path, backup / "metadata.json")
    if file_sha(backup / "metadata.json") != metadata_sha or file_sha(context.metadata_path) != metadata_sha:
        raise RepairError("Metadata zmieniło się podczas tworzenia backupu; apply przerwany.")
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "workspace_logical": str(context.workspace),
                "workspace_physical": str(context.workspace.resolve()), "database_source": str(context.database),
                "metadata_source": str(context.metadata_path), "metadata_sha256": metadata_sha,
                "backup_database_sha256": file_sha(backup / "alpr_registry.sqlite3"),
                "project_id": context.project_id, "iteration": context.iteration}
    (backup / "backup_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(backup)


def run_repair(context, *, apply=False, backup_directory=None):
    metadata_signature = file_signature(context.metadata_path)
    metadata_sha = file_sha(context.metadata_path)
    metadata = json.loads(context.metadata_path.read_text(encoding="utf-8-sig"))
    if not isinstance(metadata, dict) or file_signature(context.metadata_path) != metadata_signature:
        raise RepairError("Niestabilne lub nieprawidłowe metadata; apply przerwany.")
    if len(metadata) != context.audit["counts"]["records"]:
        raise RepairError("Liczba rekordów zmieniła się od audytu; wymagany nowy audyt.")
    with closing(read_only(context.database)) as connection:
        if connection.execute("PRAGMA user_version").fetchone()[0] != 6:
            raise RepairError("Repair wymaga istniejącego registry v6; nie wykonuje migracji.")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or connection.execute("PRAGMA foreign_key_check").fetchall():
            raise RepairError("Integralność registry nie jest poprawna.")
        before = state_counts(connection, context, metadata)
        before_tables = sql_fingerprints(connection)
        plan = inspect_scope(connection, context, metadata)
    actions = [item for item in plan if item["action"] != "skip"]
    result = {"mode": "apply" if apply else "dry-run", "workspace_logical": str(context.workspace),
              "workspace_physical": str(context.workspace.resolve()), "project_id": context.project_id,
              "iteration": context.iteration, "preview_dir": str(context.preview), "before": before,
              "dry_run": {"missing_bindings_to_create": sum(item["action"] == "create_missing_binding" for item in plan),
                          "package_bindings_to_promote": sum(item["action"] == "promote_package_binding" for item in plan),
                          "already_correct": sum(item["group"] == "package" and item["reason"] == "already_correct" for item in plan),
                          "human_pending_to_skip": len(context.audit.get("pending_rows", [])),
                          "excluded_differences_skip": sum(row["local_decision"] == "excluded" for row in context.audit.get("differences", [])),
                          "sets_disjoint": True},
              "backup": None, "applied": [], "skipped": [item for item in plan if item["action"] == "skip"]}
    if apply and actions:
        result["backup"] = create_backup(context, metadata_sha, backup_directory)
        store = AZRevisionStore(RegistryDatabase(context.database))
        with closing(sqlite3.connect(context.database, timeout=30)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("BEGIN IMMEDIATE")
            try:
                audited_by_crop = {row["crop_id"]: row for row in context.audit["missing_bindings"] + context.audit["package_rows"]}
                for planned in actions:
                    if file_signature(context.metadata_path) != metadata_signature:
                        raise RepairError("Metadata zmieniło się podczas apply; transakcja cofnięta.")
                    item = inspect_record(connection, context, metadata, audited_by_crop[planned["crop_id"]], planned["group"])
                    if item["action"] == "skip":
                        result["skipped"].append(item)
                        continue
                    saved = store.save_revision(crop_id=item["crop_id"], payload=item["payload"],
                        source_kind=item["source_kind"], trust_state=item["trust_state"], source_status=item["source_status"],
                        origin_project_id=context.project_id, origin_iteration=context.iteration,
                        bind_project_id=context.project_id, effective_status="approved", connection=connection)
                    result["applied"].append({key: value for key, value in item.items() if key != "payload"} |
                        {"revision_id_after": saved.az_revision_id, "revision_created": saved.created})
                if file_signature(context.metadata_path) != metadata_signature or file_sha(context.metadata_path) != metadata_sha:
                    raise RepairError("Metadata zmieniło się przed commit; transakcja cofnięta.")
                if connection.execute("PRAGMA foreign_key_check").fetchall():
                    raise RepairError("Błąd FK; transakcja cofnięta.")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
    with closing(read_only(context.database)) as connection:
        result["after"] = state_counts(connection, context, metadata)
        after_tables = sql_fingerprints(connection)
        result["integrity_check"] = connection.execute("PRAGMA integrity_check").fetchone()[0]
        result["foreign_key_errors"] = [tuple(row) for row in connection.execute("PRAGMA foreign_key_check")]
    result["changes"] = {
        "created_revisions": sum(item["revision_created"] for item in result["applied"]),
        "created_bindings": sum(item["group"] == "missing" for item in result["applied"]),
        "changed_bindings": sum(item["group"] == "package" for item in result["applied"]),
        "changed_timestamps": sum(item["group"] == "package" for item in result["applied"]),
        "changed_tables": [name for name in before_tables if before_tables[name] != after_tables[name]],
        "metadata_changed": file_sha(context.metadata_path) != metadata_sha,
    }
    result["plan"] = [{key: value for key, value in item.items() if key != "payload"} for item in plan]
    if (not apply or not actions) and (result["changes"]["changed_tables"] or result["changes"]["metadata_changed"]):
        raise RepairError("Dane zmieniły się podczas odczytu; wymagany stabilny dry-run.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audytowany repair checkpointów PZ2/AZ, bez zmian metadata.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--workspace", required=True, help="Jawny logiczny Workspace docelowy.")
    parser.add_argument("--audit", required=True, help="audit_live.json potwierdzający zakres naprawy.")
    parser.add_argument("--project-id", default=PROJECT_ID)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--preview-dir", default=PREVIEW_REL, help="Ścieżka względem podanego Workspace.")
    parser.add_argument("--backup-dir")
    parser.add_argument("--report", help="Raport JSON poza Workspace.")
    args = parser.parse_args(argv)
    try:
        context = make_context(args.workspace, args.audit, project_id=args.project_id,
                               iteration=args.iteration, preview_dir=args.preview_dir)
        if args.report:
            try:
                Path(args.report).resolve().relative_to(context.workspace.resolve())
            except ValueError:
                pass
            else:
                raise RepairError("Raport musi być zapisany poza Workspace.")
        result = run_repair(context, apply=args.apply, backup_directory=args.backup_dir)
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: result[key] for key in ["mode", "workspace_logical", "before", "dry_run", "after", "changes", "backup", "integrity_check"]}, ensure_ascii=True))
        for item in result["plan"]:
            if item["action"] != "skip":
                print(json.dumps({key: item[key] for key in ["action", "plate_id", "crop_id", "revision_id_before"]}))
        return 0
    except (RepairError, ValueError, KeyError, OSError, sqlite3.Error) as exc:
        print("Repair przerwany: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
