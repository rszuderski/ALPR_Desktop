import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from auto_annotation_tool.registry import RegistryDatabase
from auto_annotation_tool.registry.az_package_transport import (
    AZ_PACKAGE_SCHEMA,
    analyze_az_package,
    load_az_package,
)
from auto_annotation_tool.registry.az_registry import (
    AZRegistry,
    crop_contract_sha256,
)
from auto_annotation_tool.registry.az_revision_store import (
    compute_az_payload_sha256,
)
from auto_annotation_tool.registry.crop_identity import (
    build_pz1_crop_identity,
    compute_crop_identity_sha256,
)


class AZPackageTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "Workspace"
        self.workspace.mkdir()

        self.db = RegistryDatabase(
            self.workspace / "_registry" / "alpr_registry.sqlite3"
        )
        self.registry = AZRegistry(
            self.db,
            workspace_dir=self.workspace,
        )
        self.registry.initialize()

        with self.db.transaction() as con:
            con.execute(
                """
                INSERT INTO projects(
                    project_id, display_name, folder_name
                ) VALUES (?, ?, ?)
                """,
                ("PRJ-TARGET", "Target", "Target_F00"),
            )

        self.package_dir = self.root / "package"
        (self.package_dir / "crops").mkdir(parents=True)
        self.crop_bytes = b"portable-crop-A"
        self.crop_path = self.package_dir / "crops" / "plate_A.jpg"
        self.crop_path.write_bytes(self.crop_bytes)

        self.source_sha = "a" * 64
        self.geometry_sha = "b" * 64
        self.identity = build_pz1_crop_identity(
            source_image_id="img-sha256-" + self.source_sha,
            source_annotation_id="plate-ann-A",
            source_geometry_hash=self.geometry_sha,
            interpolation="lanczos4",
            output_width=256,
            output_height=64,
        )
        self.identity_sha = compute_crop_identity_sha256(self.identity)
        self.contract_sha = crop_contract_sha256(
            self.identity["crop_contract"]
        )

    def tearDown(self):
        self.tmp.cleanup()

    def az_payload(self, char="A"):
        return {
            "crop_identity_sha256": self.identity_sha,
            "characters": [{
                "character": char,
                "bbox": [0.10, 0.20, 0.20, 0.80],
                "row": 0,
                "method": "manual",
                "source_kind": "local_manual",
                "confidence": 1.0,
            }],
            "layout": {"kind": "1R", "confirmed": True},
            "gold_state": {
                "approved": True,
                "excluded": False,
                "candidate": True,
            },
            "status": "perfect",
        }

    def crop_entry(self, *, source_sha=None, with_az=True):
        payload = self.az_payload()
        entry = {
            "entry_id": "donor-A",
            "crop_identity": self.identity,
            "crop_identity_sha256": self.identity_sha,
            "crop_contract_sha256": self.contract_sha,
            "source_file_sha256": source_sha or self.source_sha,
            "artifact": {
                "path": "crops/plate_A.jpg",
                "sha256": hashlib.sha256(self.crop_bytes).hexdigest(),
                "size_bytes": len(self.crop_bytes),
                "width": 256,
                "height": 64,
            },
        }
        if with_az:
            entry["az"] = {
                "payload": payload,
                "payload_sha256": compute_az_payload_sha256(payload),
                "source_kind": "project_export",
                "source_status": "perfect",
                "trust_state": "source_local",
                "origin_project_id": "PRJ-SOURCE",
                "origin_iteration": 1,
                "source_az_revision_id": "AZR-SOURCE-1",
                "created_at": "2026-09-29T12:00:00+00:00",
            }
        return entry

    def write_package(self, crops):
        manifest = {
            "schema": AZ_PACKAGE_SCHEMA,
            "package_id": "AZPKG-TEST",
            "created_at": "2026-09-29T12:00:00+00:00",
            "source": {
                "project_id": "PRJ-SOURCE",
                "iteration": 1,
            },
            "crops": crops,
        }
        (self.package_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def register_global_crop(self):
        artifact = self.workspace / "existing" / "plate_A.jpg"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(self.crop_bytes)
        result = self.registry.register_crop_artifact(
            crop_identity_sha256=self.identity_sha,
            identity_mode=self.identity["identity_mode"],
            source_file_sha256=self.source_sha,
            artifact_path=artifact,
            artifact_sha256=hashlib.sha256(self.crop_bytes).hexdigest(),
            size_bytes=len(self.crop_bytes),
            width=256,
            height=64,
            source_annotation_id="plate-ann-A",
            source_geometry_hash=self.geometry_sha,
            crop_contract_sha256=self.contract_sha,
        )
        return result

    def test_new_crop_is_planned_without_writes(self):
        self.write_package([self.crop_entry()])
        package = load_az_package(self.package_dir)

        with self.db.read_connection() as con:
            before = {
                "crops": con.execute(
                    "SELECT COUNT(*) FROM plate_crops"
                ).fetchone()[0],
                "members": con.execute(
                    "SELECT COUNT(*) FROM project_crop_members"
                ).fetchone()[0],
                "az": con.execute(
                    "SELECT COUNT(*) FROM az_revisions"
                ).fetchone()[0],
            }

        plan = analyze_az_package(
            self.registry,
            package,
            target_project_id="PRJ-TARGET",
        )

        self.assertEqual(plan.new_count, 1)
        self.assertEqual(plan.already_present_count, 0)
        self.assertEqual(plan.conflict_count, 0)
        self.assertEqual(plan.invalid_count, 0)
        item = plan.items[0]
        self.assertTrue(item.needs_new_crop)
        self.assertTrue(item.needs_project_attach)
        self.assertTrue(item.package_has_az)
        self.assertEqual(item.az_relation, "package_only")

        with self.db.read_connection() as con:
            after = {
                "crops": con.execute(
                    "SELECT COUNT(*) FROM plate_crops"
                ).fetchone()[0],
                "members": con.execute(
                    "SELECT COUNT(*) FROM project_crop_members"
                ).fetchone()[0],
                "az": con.execute(
                    "SELECT COUNT(*) FROM az_revisions"
                ).fetchone()[0],
            }

        self.assertEqual(before, after)

    def test_existing_global_crop_is_new_for_target_but_not_recreated(self):
        existing = self.register_global_crop()
        self.write_package([self.crop_entry()])

        plan = analyze_az_package(
            self.registry,
            load_az_package(self.package_dir),
            target_project_id="PRJ-TARGET",
        )

        item = plan.items[0]
        self.assertEqual(item.status, "new")
        self.assertFalse(item.needs_new_crop)
        self.assertTrue(item.needs_project_attach)
        self.assertEqual(item.registry_crop_id, existing.crop_id)
        self.assertTrue(item.artifact_already_linked)

    def test_target_member_is_already_present(self):
        existing = self.register_global_crop()
        with self.db.transaction() as con:
            con.execute(
                """
                INSERT INTO project_crop_members(
                    project_id,
                    crop_id,
                    first_seen_iteration,
                    last_seen_iteration,
                    first_seen_at,
                    last_seen_at,
                    source_mode
                ) VALUES (?, ?, 1, 1, ?, ?, ?)
                """,
                (
                    "PRJ-TARGET",
                    existing.crop_id,
                    "2026-09-29T12:00:00+00:00",
                    "2026-09-29T12:00:00+00:00",
                    "package_test",
                ),
            )
        self.write_package([self.crop_entry()])

        plan = analyze_az_package(
            self.registry,
            load_az_package(self.package_dir),
            target_project_id="PRJ-TARGET",
        )

        item = plan.items[0]
        self.assertEqual(item.status, "already_present")
        self.assertFalse(item.needs_project_attach)
        self.assertEqual(item.registry_crop_id, existing.crop_id)

    def test_existing_identity_with_other_source_sha_is_conflict(self):
        existing = self.register_global_crop()
        self.assertTrue(existing.crop_id)
        self.write_package([
            self.crop_entry(source_sha="c" * 64)
        ])

        plan = analyze_az_package(
            self.registry,
            load_az_package(self.package_dir),
            target_project_id="PRJ-TARGET",
        )

        self.assertEqual(plan.conflict_count, 1)
        item = plan.items[0]
        self.assertEqual(item.status, "conflict")
        self.assertIn("source_file_sha256", item.reason)

    def test_bad_artifact_hash_is_reported_as_invalid_item(self):
        entry = self.crop_entry()
        entry["artifact"]["sha256"] = "f" * 64
        self.write_package([entry])

        package = load_az_package(self.package_dir)

        self.assertEqual(len(package.crops), 0)
        self.assertEqual(len(package.invalid_items), 1)
        self.assertIn(
            "artifact.sha256",
            package.invalid_items[0].reason,
        )

    def test_bad_az_payload_hash_is_invalid(self):
        entry = self.crop_entry()
        entry["az"]["payload_sha256"] = "e" * 64
        self.write_package([entry])

        package = load_az_package(self.package_dir)

        self.assertEqual(len(package.crops), 0)
        self.assertEqual(len(package.invalid_items), 1)
        self.assertIn(
            "az.payload_sha256",
            package.invalid_items[0].reason,
        )

    def test_package_without_az_is_valid(self):
        self.write_package([
            self.crop_entry(with_az=False)
        ])

        plan = analyze_az_package(
            self.registry,
            load_az_package(self.package_dir),
            target_project_id="PRJ-TARGET",
        )

        self.assertEqual(plan.new_count, 1)
        self.assertEqual(plan.with_az_count, 0)
        self.assertEqual(plan.without_az_count, 1)
        self.assertFalse(plan.items[0].package_has_az)

    def test_path_traversal_is_invalid(self):
        entry = self.crop_entry()
        entry["artifact"]["path"] = "../plate_A.jpg"
        self.write_package([entry])

        package = load_az_package(self.package_dir)

        self.assertEqual(len(package.crops), 0)
        self.assertEqual(len(package.invalid_items), 1)
        self.assertIn(
            "niedozwolony segment",
            package.invalid_items[0].reason,
        )


class AZPackageAppendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "Workspace"
        self.workspace.mkdir()

        self.db = RegistryDatabase(
            self.workspace / "_registry" / "alpr_registry.sqlite3"
        )
        self.registry = AZRegistry(
            self.db,
            workspace_dir=self.workspace,
        )
        self.registry.initialize()

        with self.db.transaction() as con:
            con.execute(
                """
                INSERT INTO projects(
                    project_id, display_name, folder_name
                ) VALUES (?, ?, ?)
                """,
                ("PRJ-TARGET", "Target", "Target_F00"),
            )

        self.package_dir = self.root / "package"
        (self.package_dir / "crops").mkdir(parents=True)
        self.crop_bytes = b"portable-crop-A"
        self.crop_path = self.package_dir / "crops" / "plate_A.jpg"
        self.crop_path.write_bytes(self.crop_bytes)

        self.source_sha = "a" * 64
        self.geometry_sha = "b" * 64
        self.identity = build_pz1_crop_identity(
            source_image_id="img-sha256-" + self.source_sha,
            source_annotation_id="plate-ann-A",
            source_geometry_hash=self.geometry_sha,
            interpolation="lanczos4",
            output_width=256,
            output_height=64,
        )
        self.identity_sha = compute_crop_identity_sha256(self.identity)
        self.contract_sha = crop_contract_sha256(
            self.identity["crop_contract"]
        )

    def tearDown(self):
        self.tmp.cleanup()

    def az_payload(self, char="A"):
        return {
            "crop_identity_sha256": self.identity_sha,
            "characters": [{
                "character": char,
                "bbox": [0.10, 0.20, 0.20, 0.80],
                "row": 0,
                "method": "manual",
                "source_kind": "local_manual",
                "confidence": 1.0,
            }],
            "layout": {"kind": "1R", "confirmed": True},
            "gold_state": {
                "approved": True,
                "excluded": False,
                "candidate": True,
            },
            "status": "perfect",
        }

    def crop_entry(self, *, source_sha=None, with_az=True):
        payload = self.az_payload()
        entry = {
            "entry_id": "donor-A",
            "crop_identity": self.identity,
            "crop_identity_sha256": self.identity_sha,
            "crop_contract_sha256": self.contract_sha,
            "source_file_sha256": source_sha or self.source_sha,
            "artifact": {
                "path": "crops/plate_A.jpg",
                "sha256": hashlib.sha256(self.crop_bytes).hexdigest(),
                "size_bytes": len(self.crop_bytes),
                "width": 256,
                "height": 64,
            },
        }
        if with_az:
            entry["az"] = {
                "payload": payload,
                "payload_sha256": compute_az_payload_sha256(payload),
                "source_kind": "project_export",
                "source_status": "perfect",
                "trust_state": "source_local",
                "origin_project_id": "PRJ-SOURCE",
                "origin_iteration": 1,
                "source_az_revision_id": "AZR-SOURCE-1",
                "created_at": "2026-09-29T12:00:00+00:00",
            }
        return entry

    def write_package(self, crops):
        manifest = {
            "schema": AZ_PACKAGE_SCHEMA,
            "package_id": "AZPKG-TEST",
            "created_at": "2026-09-29T12:00:00+00:00",
            "source": {
                "project_id": "PRJ-SOURCE",
                "iteration": 1,
            },
            "crops": crops,
        }
        (self.package_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def test_append_new_crop_with_az_is_project_local_pending_review(self):
        from auto_annotation_tool.registry.az_package_transport import (
            append_az_package,
        )

        self.write_package([self.crop_entry()])
        package = load_az_package(self.package_dir)
        target_dir = (
            self.workspace
            / "9_projects"
            / "Target_F00"
            / "3_cropped_characters"
            / "_az_append"
        )

        result = append_az_package(
            self.registry,
            package,
            target_project_id="PRJ-TARGET",
            iteration_num=1,
            target_artifact_dir=target_dir,
        )

        self.assertEqual(result.planned_new, 1)
        self.assertEqual(result.appended, 1)
        self.assertEqual(result.az_bound, 1)
        self.assertEqual(result.conflicts, 0)
        self.assertEqual(result.invalid, 0)
        self.assertEqual(result.new_crops, 1)
        self.assertTrue(Path(result.items[0].artifact_path).is_file())

        crop_id = result.items[0].crop_id
        with self.db.read_connection() as con:
            self.assertIsNotNone(
                con.execute(
                    """
                    SELECT 1 FROM project_crop_members
                    WHERE project_id = ? AND crop_id = ?
                    """,
                    ("PRJ-TARGET", crop_id),
                ).fetchone()
            )
            iteration = con.execute(
                """
                SELECT source_plate_key, source_at_ref
                FROM iteration_crop_members
                WHERE project_id = ?
                  AND iteration_num = 1
                  AND crop_id = ?
                """,
                ("PRJ-TARGET", crop_id),
            ).fetchone()
            self.assertIsNotNone(iteration)
            self.assertEqual(iteration["source_plate_key"], "donor-A")
            self.assertEqual(
                iteration["source_at_ref"],
                "azpkg:AZPKG-TEST",
            )

            binding = con.execute(
                """
                SELECT pca.effective_status, ar.trust_state,
                       ar.origin_project_id, ar.source_kind
                FROM project_crop_az pca
                JOIN az_revisions ar
                  ON ar.az_revision_id = pca.az_revision_id
                WHERE pca.project_id = ? AND pca.crop_id = ?
                """,
                ("PRJ-TARGET", crop_id),
            ).fetchone()
            self.assertIsNotNone(binding)
            self.assertEqual(
                binding["effective_status"],
                "imported_pending_review",
            )
            self.assertEqual(
                binding["trust_state"],
                "external_pending_review",
            )
            self.assertEqual(
                binding["origin_project_id"],
                "PRJ-SOURCE",
            )
            self.assertEqual(
                binding["source_kind"],
                "package_import:project_export",
            )

    def test_append_without_az_attaches_crop_but_creates_no_az_binding(self):
        from auto_annotation_tool.registry.az_package_transport import (
            append_az_package,
        )

        self.write_package([self.crop_entry(with_az=False)])
        result = append_az_package(
            self.registry,
            load_az_package(self.package_dir),
            target_project_id="PRJ-TARGET",
            iteration_num=1,
            target_artifact_dir=self.workspace / "append_no_az",
        )

        self.assertEqual(result.appended, 1)
        self.assertEqual(result.az_bound, 0)
        self.assertEqual(result.without_az, 1)

        with self.db.read_connection() as con:
            self.assertEqual(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM project_crop_az
                    WHERE project_id = ?
                    """,
                    ("PRJ-TARGET",),
                ).fetchone()[0],
                0,
            )

    def test_second_append_is_noop_and_does_not_duplicate_target(self):
        from auto_annotation_tool.registry.az_package_transport import (
            append_az_package,
        )

        self.write_package([self.crop_entry()])
        package = load_az_package(self.package_dir)
        target_dir = self.workspace / "append_idempotent"

        first = append_az_package(
            self.registry,
            package,
            target_project_id="PRJ-TARGET",
            iteration_num=1,
            target_artifact_dir=target_dir,
        )
        second = append_az_package(
            self.registry,
            package,
            target_project_id="PRJ-TARGET",
            iteration_num=1,
            target_artifact_dir=target_dir,
        )

        self.assertEqual(first.appended, 1)
        self.assertEqual(second.appended, 0)
        self.assertEqual(second.already_present, 1)

        with self.db.read_connection() as con:
            self.assertEqual(
                con.execute(
                    """
                    SELECT COUNT(*) FROM project_crop_members
                    WHERE project_id = ?
                    """,
                    ("PRJ-TARGET",),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                con.execute(
                    """
                    SELECT COUNT(*) FROM iteration_crop_members
                    WHERE project_id = ? AND iteration_num = 1
                    """,
                    ("PRJ-TARGET",),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                con.execute(
                    """
                    SELECT COUNT(*) FROM project_crop_az
                    WHERE project_id = ?
                    """,
                    ("PRJ-TARGET",),
                ).fetchone()[0],
                1,
            )

    def test_sql_failure_rolls_back_registry_and_removes_new_artifact(self):
        from unittest.mock import patch
        from auto_annotation_tool.registry.az_package_transport import (
            append_az_package,
        )

        self.write_package([self.crop_entry()])
        package = load_az_package(self.package_dir)
        target_dir = self.workspace / "append_rollback"

        with self.db.read_connection() as con:
            before = {
                "crops": con.execute(
                    "SELECT COUNT(*) FROM plate_crops"
                ).fetchone()[0],
                "members": con.execute(
                    "SELECT COUNT(*) FROM project_crop_members"
                ).fetchone()[0],
                "iterations": con.execute(
                    "SELECT COUNT(*) FROM iteration_crop_members"
                ).fetchone()[0],
                "revisions": con.execute(
                    "SELECT COUNT(*) FROM az_revisions"
                ).fetchone()[0],
            }

        with patch.object(
            AZRegistry,
            "_attach_iteration",
            side_effect=RuntimeError("forced AZ009B rollback"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "forced AZ009B rollback",
            ):
                append_az_package(
                    self.registry,
                    package,
                    target_project_id="PRJ-TARGET",
                    iteration_num=1,
                    target_artifact_dir=target_dir,
                )

        with self.db.read_connection() as con:
            after = {
                "crops": con.execute(
                    "SELECT COUNT(*) FROM plate_crops"
                ).fetchone()[0],
                "members": con.execute(
                    "SELECT COUNT(*) FROM project_crop_members"
                ).fetchone()[0],
                "iterations": con.execute(
                    "SELECT COUNT(*) FROM iteration_crop_members"
                ).fetchone()[0],
                "revisions": con.execute(
                    "SELECT COUNT(*) FROM az_revisions"
                ).fetchone()[0],
            }
        self.assertEqual(before, after)
        self.assertFalse(
            any(target_dir.glob("crop_*"))
            if target_dir.exists()
            else False
        )


if __name__ == "__main__":
    unittest.main()
