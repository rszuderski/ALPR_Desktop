import hashlib
import tempfile
import unittest
from pathlib import Path

from auto_annotation_tool.registry import RegistryDatabase
from auto_annotation_tool.registry.az_project_import import (
    IMPORT_PENDING_REVIEW,
    analyze_project_az_import,
    import_project_az_bindings,
)
from auto_annotation_tool.registry.az_registry import AZRegistry
from auto_annotation_tool.registry.az_revision_store import AZRevisionStore
from auto_annotation_tool.registry.crop_identity import (
    build_pz1_crop_identity,
    compute_crop_identity_sha256,
)


class AZProjectImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name) / "Workspace"
        self.workspace.mkdir()

        self.db = RegistryDatabase(
            self.workspace / "_registry" / "alpr_registry.sqlite3"
        )
        self.registry = AZRegistry(
            self.db,
            workspace_dir=self.workspace,
        )
        self.registry.initialize()
        self.store = AZRevisionStore(self.db)

        with self.db.transaction() as con:
            for project_id in ("PRJ-A", "PRJ-B"):
                con.execute(
                    """
                    INSERT INTO projects(project_id, display_name)
                    VALUES (?, ?)
                    """,
                    (project_id, project_id),
                )

        self.crop_importable = self._create_crop("importable", attach_b=True)
        self.crop_missing = self._create_crop("missing", attach_b=False)
        self.crop_same = self._create_crop("same", attach_b=True)
        self.crop_conflict = self._create_crop("conflict", attach_b=True)

        self.rev_importable = self._bind_source(
            self.crop_importable, char="A"
        )
        self.rev_missing = self._bind_source(
            self.crop_missing, char="M"
        )
        self.rev_same = self._bind_source(
            self.crop_same, char="S"
        )
        self.rev_conflict = self._bind_source(
            self.crop_conflict, char="C"
        )

        # Target already has exactly the same revision.
        self.store.save_revision(
            crop_id=self.crop_same["crop_id"],
            payload=self._payload(self.crop_same, "S"),
            source_kind="project_import",
            trust_state="external_pending_review",
            bind_project_id="PRJ-B",
            effective_status=IMPORT_PENDING_REVIEW,
        )

        # Target conflict: another semantic revision is already bound.
        self.target_conflict_revision = self.store.save_revision(
            crop_id=self.crop_conflict["crop_id"],
            payload=self._payload(self.crop_conflict, "X"),
            source_kind="local_manual",
            trust_state="local_manual",
            origin_project_id="PRJ-B",
            origin_iteration=1,
            bind_project_id="PRJ-B",
            effective_status="approved",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _create_crop(self, label: str, *, attach_b: bool) -> dict:
        source_sha = hashlib.sha256(
            f"source:{label}".encode("utf-8")
        ).hexdigest()
        geometry_sha = hashlib.sha256(
            f"geometry:{label}".encode("utf-8")
        ).hexdigest()

        identity_payload = build_pz1_crop_identity(
            source_image_id="img-sha256-" + source_sha,
            source_annotation_id=f"plate-{label}",
            source_geometry_hash=geometry_sha,
            interpolation="lanczos4",
            output_width=256,
            output_height=64,
        )
        identity_sha = compute_crop_identity_sha256(identity_payload)

        crop_path = (
            self.workspace / "runs" / label / "images" / f"{label}.jpg"
        )
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop_path.write_bytes(f"crop:{label}".encode("utf-8"))
        artifact_sha = hashlib.sha256(crop_path.read_bytes()).hexdigest()

        first = self.registry.register_crop_artifact(
            crop_identity_sha256=identity_sha,
            identity_mode=identity_payload["identity_mode"],
            source_file_sha256=source_sha,
            artifact_path=crop_path,
            artifact_sha256=artifact_sha,
            size_bytes=crop_path.stat().st_size,
            width=256,
            height=64,
            source_annotation_id=f"plate-{label}",
            source_geometry_hash=geometry_sha,
            project_id="PRJ-A",
            iteration_num=1,
            source_mode="campaign",
        )

        if attach_b:
            second = self.registry.register_crop_artifact(
                crop_identity_sha256=identity_sha,
                identity_mode=identity_payload["identity_mode"],
                source_file_sha256=source_sha,
                artifact_path=crop_path,
                artifact_sha256=artifact_sha,
                size_bytes=crop_path.stat().st_size,
                width=256,
                height=64,
                source_annotation_id=f"plate-{label}",
                source_geometry_hash=geometry_sha,
                project_id="PRJ-B",
                iteration_num=1,
                source_mode="campaign",
            )
            self.assertEqual(first.crop_id, second.crop_id)

        return {
            "crop_id": first.crop_id,
            "identity_sha": identity_sha,
        }

    def _payload(self, crop: dict, char: str) -> dict:
        return {
            "crop_identity_sha256": crop["identity_sha"],
            "characters": [
                {
                    "character": char,
                    "bbox": [0.1, 0.2, 0.2, 0.8],
                    "row": 1,
                    "method": "manual",
                    "source_kind": "local_manual",
                    "confidence": 1.0,
                }
            ],
            "layout": {"kind": "single_row", "confirmed": True},
            "gold_state": {
                "approved": True,
                "excluded": False,
                "candidate": True,
            },
            "status": "perfect",
            "expected_text": char,
        }

    def _bind_source(self, crop: dict, *, char: str):
        return self.store.save_revision(
            crop_id=crop["crop_id"],
            payload=self._payload(crop, char),
            source_kind="local_manual",
            trust_state="local_manual",
            origin_project_id="PRJ-A",
            origin_iteration=1,
            bind_project_id="PRJ-A",
            effective_status="approved",
        )

    def test_plan_classifies_safe_overlap_and_conflicts(self):
        plan = analyze_project_az_import(
            self.registry,
            source_project_id="PRJ-A",
            target_project_id="PRJ-B",
        )

        by_crop = {item.crop_id: item for item in plan.items}
        self.assertEqual(
            by_crop[self.crop_importable["crop_id"]].status,
            "importable",
        )
        self.assertEqual(
            by_crop[self.crop_missing["crop_id"]].status,
            "target_missing_crop",
        )
        self.assertEqual(
            by_crop[self.crop_same["crop_id"]].status,
            "already_bound",
        )
        self.assertEqual(
            by_crop[self.crop_conflict["crop_id"]].status,
            "target_conflict",
        )

        self.assertEqual(plan.importable_count, 1)
        self.assertEqual(plan.target_missing_crop_count, 1)
        self.assertEqual(plan.already_bound_count, 1)
        self.assertEqual(plan.conflict_count, 1)

    def test_import_binds_only_safe_overlap_without_new_revision(self):
        with self.db.read_connection() as con:
            revisions_before = con.execute(
                "SELECT COUNT(*) FROM az_revisions"
            ).fetchone()[0]

        result = import_project_az_bindings(
            self.registry,
            source_project_id="PRJ-A",
            target_project_id="PRJ-B",
            updated_at="2026-09-25T22:00:00+00:00",
        )

        self.assertEqual(result.imported, 1)
        self.assertEqual(result.already_bound, 1)
        self.assertEqual(result.conflicts, 1)
        self.assertEqual(result.target_missing_crop, 1)
        self.assertEqual(result.effective_status, IMPORT_PENDING_REVIEW)

        target = self.store.get_project_az(
            project_id="PRJ-B",
            crop_id=self.crop_importable["crop_id"],
        )
        self.assertIsNotNone(target)
        self.assertEqual(
            target["az_revision_id"],
            self.rev_importable.az_revision_id,
        )
        self.assertEqual(
            target["effective_status"],
            IMPORT_PENDING_REVIEW,
        )
        self.assertEqual(target["origin_project_id"], "PRJ-A")

        with self.db.read_connection() as con:
            revisions_after = con.execute(
                "SELECT COUNT(*) FROM az_revisions"
            ).fetchone()[0]
        self.assertEqual(revisions_before, revisions_after)

    def test_import_does_not_overwrite_target_conflict(self):
        import_project_az_bindings(
            self.registry,
            source_project_id="PRJ-A",
            target_project_id="PRJ-B",
        )

        target = self.store.get_project_az(
            project_id="PRJ-B",
            crop_id=self.crop_conflict["crop_id"],
        )
        self.assertEqual(
            target["az_revision_id"],
            self.target_conflict_revision.az_revision_id,
        )
        self.assertNotEqual(
            target["az_revision_id"],
            self.rev_conflict.az_revision_id,
        )

    def test_second_import_is_idempotent(self):
        first = import_project_az_bindings(
            self.registry,
            source_project_id="PRJ-A",
            target_project_id="PRJ-B",
        )
        second = import_project_az_bindings(
            self.registry,
            source_project_id="PRJ-A",
            target_project_id="PRJ-B",
        )

        self.assertEqual(first.imported, 1)
        self.assertEqual(second.imported, 0)
        self.assertEqual(second.already_bound, 2)

    def test_crop_filter_limits_import_scope(self):
        result = import_project_az_bindings(
            self.registry,
            source_project_id="PRJ-A",
            target_project_id="PRJ-B",
            crop_ids=[self.crop_importable["crop_id"]],
        )
        self.assertEqual(result.imported, 1)
        self.assertEqual(result.conflicts, 0)
        self.assertEqual(result.target_missing_crop, 0)

    def test_source_and_target_must_differ(self):
        with self.assertRaises(ValueError):
            analyze_project_az_import(
                self.registry,
                source_project_id="PRJ-A",
                target_project_id="PRJ-A",
            )


    def test_list_import_sources_summarizes_source_against_target(self):
        from auto_annotation_tool.registry.az_project_import import (
            list_project_az_import_sources,
        )

        candidates = list_project_az_import_sources(
            self.registry,
            target_project_id="PRJ-B",
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.source_project_id, "PRJ-A")
        self.assertEqual(candidate.target_project_id, "PRJ-B")
        self.assertEqual(candidate.source_display_name, "PRJ-A")
        self.assertEqual(candidate.source_az_count, 4)
        self.assertEqual(candidate.importable_count, 1)
        self.assertEqual(candidate.already_bound_count, 1)
        self.assertEqual(candidate.conflict_count, 1)
        self.assertEqual(candidate.target_missing_crop_count, 1)
        self.assertEqual(candidate.invalid_source_count, 0)
        self.assertTrue(candidate.can_import)

    def test_list_import_sources_is_read_only(self):
        from auto_annotation_tool.registry.az_project_import import (
            list_project_az_import_sources,
        )

        with self.db.read_connection() as con:
            before = {
                "projects": con.execute(
                    "SELECT COUNT(*) FROM projects"
                ).fetchone()[0],
                "bindings": con.execute(
                    "SELECT COUNT(*) FROM project_crop_az"
                ).fetchone()[0],
                "revisions": con.execute(
                    "SELECT COUNT(*) FROM az_revisions"
                ).fetchone()[0],
            }

        candidates = list_project_az_import_sources(
            self.registry,
            target_project_id="PRJ-B",
        )
        self.assertTrue(candidates)

        with self.db.read_connection() as con:
            after = {
                "projects": con.execute(
                    "SELECT COUNT(*) FROM projects"
                ).fetchone()[0],
                "bindings": con.execute(
                    "SELECT COUNT(*) FROM project_crop_az"
                ).fetchone()[0],
                "revisions": con.execute(
                    "SELECT COUNT(*) FROM az_revisions"
                ).fetchone()[0],
            }

        self.assertEqual(before, after)

    def test_list_import_sources_reports_no_match_for_target_without_crops(self):
        from auto_annotation_tool.registry.az_project_import import (
            list_project_az_import_sources,
        )

        with self.db.transaction() as con:
            con.execute(
                """
                INSERT INTO projects(project_id, display_name)
                VALUES (?, ?)
                """,
                ("PRJ-C", "C"),
            )

        candidates = list_project_az_import_sources(
            self.registry,
            target_project_id="PRJ-C",
        )

        source_a = next(
            item for item in candidates
            if item.source_project_id == "PRJ-A"
        )
        self.assertEqual(source_a.importable_count, 0)
        self.assertEqual(source_a.target_missing_crop_count, 4)
        self.assertFalse(source_a.can_import)


if __name__ == "__main__":
    unittest.main()
