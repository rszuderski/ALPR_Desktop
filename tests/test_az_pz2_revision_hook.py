import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from auto_annotation_tool.registry import RegistryDatabase
from auto_annotation_tool.registry.az_registry import AZRegistry
from auto_annotation_tool.registry.az_revision_store import AZRevisionStore
from auto_annotation_tool.registry.crop_identity import (
    build_pz1_crop_identity,
    compute_crop_identity_sha256,
)
from auto_annotation_tool.registry.pz2_revision_registry import save_pz2_revision
from auto_annotation_tool.gui import z3_review_runtime
from auto_annotation_tool.gui import z3_list_review


SOURCE_SHA = "a" * 64
GEOMETRY_SHA = "b" * 64


class PZ2RevisionRegistryTests(unittest.TestCase):
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

        crop_path = (
            self.workspace
            / "3_cropped_characters"
            / "run_001"
            / "images"
            / "plate_000001.jpg"
        )
        crop_path.parent.mkdir(parents=True)
        crop_path.write_bytes(b"crop")

        identity_payload = build_pz1_crop_identity(
            source_image_id="img-sha256-" + SOURCE_SHA,
            source_annotation_id="plate-ann-1",
            source_geometry_hash=GEOMETRY_SHA,
            interpolation="lanczos4",
            output_width=256,
            output_height=64,
        )
        self.identity_sha = compute_crop_identity_sha256(identity_payload)

        registered = self.registry.register_crop_artifact(
            crop_identity_sha256=self.identity_sha,
            identity_mode=identity_payload["identity_mode"],
            source_file_sha256=SOURCE_SHA,
            artifact_path=crop_path,
            artifact_sha256=hashlib.sha256(
                crop_path.read_bytes()
            ).hexdigest(),
            size_bytes=crop_path.stat().st_size,
            width=256,
            height=64,
            source_annotation_id="plate-ann-1",
            source_geometry_hash=GEOMETRY_SHA,
        )
        self.crop_id = registered.crop_id

        with self.db.transaction() as con:
            con.execute(
                """
                INSERT INTO projects(project_id, display_name)
                VALUES (?, ?)
                """,
                ("PRJ-A", "A"),
            )

    def tearDown(self):
        self.tmp.cleanup()

    def metadata(self):
        return {
            "crop_id": self.crop_id,
            "crop_identity_sha256": self.identity_sha,
            "characters": [
                {
                    "character": "A",
                    "bbox": [20, 8, 60, 56],
                    "confidence": 1.0,
                    "method": "manual",
                    "source_kind": "local_manual",
                    "reading_row": 1,
                    "reading_col": 1,
                }
            ],
            "plate_layout": "single_row",
            "plate_layout_override": "single_row",
            "layout_source": "manual_override",
            "status": "perfect",
            "gold_state": {
                "approved": True,
                "excluded": False,
                "candidate": True,
            },
            "source_info": {
                "bucket": "local_manual",
                "origin": "preview_editor",
            },
        }

    def test_free_mode_creates_revision_without_project_binding(self):
        result = save_pz2_revision(
            self.registry,
            self.metadata(),
        )

        self.assertTrue(result.revision.created)
        self.assertIsNone(result.revision.project_id)

        with self.db.read_connection() as con:
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM az_revisions"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM project_crop_az"
                ).fetchone()[0],
                0,
            )

    def test_campaign_mode_binds_current_revision(self):
        result = save_pz2_revision(
            self.registry,
            self.metadata(),
            project_id="PRJ-A",
            iteration_num=2,
        )

        self.assertEqual(result.revision.project_id, "PRJ-A")
        self.assertEqual(result.effective_status, "approved")

        with self.db.read_connection() as con:
            row = con.execute(
                """
                SELECT az_revision_id, effective_status
                FROM project_crop_az
                WHERE project_id = ? AND crop_id = ?
                """,
                ("PRJ-A", self.crop_id),
            ).fetchone()
        self.assertEqual(
            row["az_revision_id"],
            result.revision.az_revision_id,
        )
        self.assertEqual(row["effective_status"], "approved")

    def test_same_semantic_state_is_deduplicated(self):
        first = save_pz2_revision(
            self.registry,
            self.metadata(),
            project_id="PRJ-A",
            iteration_num=2,
        )
        second = save_pz2_revision(
            self.registry,
            self.metadata(),
            project_id="PRJ-A",
            iteration_num=2,
        )

        self.assertTrue(first.revision.created)
        self.assertFalse(second.revision.created)
        self.assertEqual(
            first.revision.az_revision_id,
            second.revision.az_revision_id,
        )

    def test_exclusion_creates_new_child_revision(self):
        first = save_pz2_revision(
            self.registry,
            self.metadata(),
            project_id="PRJ-A",
            iteration_num=2,
        )
        excluded = self.metadata()
        excluded["gold_state"] = {
            "approved": True,
            "excluded": True,
            "candidate": False,
        }

        second = save_pz2_revision(
            self.registry,
            excluded,
            project_id="PRJ-A",
            iteration_num=2,
        )

        self.assertTrue(second.revision.created)
        self.assertEqual(
            second.revision.parent_revision_id,
            first.revision.az_revision_id,
        )
        self.assertEqual(second.effective_status, "excluded")

    def test_registry_dimensions_are_used_not_gui_runtime_dimensions(self):
        data = self.metadata()
        data["plate_image_width"] = 9999
        data["plate_image_height"] = 9999

        result = save_pz2_revision(self.registry, data)
        loaded = AZRevisionStore(self.db).get_revision(
            result.revision.az_revision_id
        )

        self.assertEqual(
            loaded["payload"]["characters"][0]["bbox"],
            [0.078125, 0.125, 0.234375, 0.875],
        )

    def test_group_approval_checkpoints_imported_az_and_reopening_updates_binding(self):
        data = self.metadata()
        data["ground_truth_text"] = "A"
        data["status"] = "needs_fix"
        data["gold_state"]["approved"] = False
        data["gold_state"]["candidate"] = False
        data["review_state"] = {
            "status": "in_progress", "source": "az_project_import",
        }
        imported = save_pz2_revision(
            self.registry, data, project_id="PRJ-A", iteration_num=1,
        )
        with self.db.transaction() as con:
            con.execute(
                "UPDATE project_crop_az SET effective_status = ? WHERE project_id = ?",
                ("imported_pending_review", "PRJ-A"),
            )

        host = Mock()
        host.preview_metadata = {"plate": data}
        host._characters_to_text.return_value = "A"
        host._get_plate_source_bucket.return_value = "local_manual"
        host._get_review_quality_status.return_value = "perfect"
        host._review_approval_is_current.return_value = True

        def checkpoint(_host, _plate_id, row, *, event):
            return save_pz2_revision(
                self.registry, row, project_id="PRJ-A", iteration_num=1,
            )

        with patch.object(z3_review_runtime, "get_review_quality_status", return_value="perfect"), \
                patch.object(z3_review_runtime, "_persist_review_az_revision_best_effort",
                             side_effect=checkpoint) as save_az, \
                patch.object(z3_review_runtime, "_refresh_after_change") as refresh:
            self.assertTrue(z3_list_review.change_plate_approval(host, "plate", True)["ok"])
            approved = AZRevisionStore(self.db).get_project_az(project_id="PRJ-A", crop_id=self.crop_id)
            self.assertEqual(approved["effective_status"], "approved")
            self.assertTrue(approved["payload"]["gold_state"]["approved"])
            self.assertNotEqual(approved["az_revision_id"], imported.revision.az_revision_id)
            save_az.assert_called_once_with(host, "plate", data, event="review_approved")
            refresh.assert_not_called()

            approved_at = data["review_state"]["approved_at"]
            with self.db.transaction() as con:
                con.execute(
                    "UPDATE project_crop_az SET effective_status = ? WHERE project_id = ?",
                    ("imported_pending_review", "PRJ-A"),
                )
            self.assertTrue(z3_list_review.change_plate_approval(host, "plate", True)["unchanged"])
            repeated = AZRevisionStore(self.db).get_project_az(project_id="PRJ-A", crop_id=self.crop_id)
            self.assertEqual(repeated["effective_status"], "approved")
            self.assertEqual(repeated["az_revision_id"], approved["az_revision_id"])
            self.assertEqual(data["review_state"]["approved_at"], approved_at)

            self.assertTrue(z3_list_review.change_plate_approval(host, "plate", False)["ok"])
            reopened = AZRevisionStore(self.db).get_project_az(project_id="PRJ-A", crop_id=self.crop_id)
            self.assertEqual(reopened["effective_status"], "needs_fix")
            self.assertFalse(reopened["payload"]["gold_state"]["approved"])
            self.assertEqual(save_az.call_count, 3)
            self.assertEqual(save_az.call_args.kwargs["event"], "approval_reverted")
            refresh.assert_not_called()


class PZ2ReviewHookTests(unittest.TestCase):
    def host_with_plate(self, data):
        host = Mock()
        host.preview_metadata = {"plate_000001": data}
        host._preview_active_pid = "plate_000001"
        return host

    def test_n_event_persists_az_revision_after_metadata_save(self):
        data = {
            "gold_state": {
                "approved": False,
                "excluded": False,
                "candidate": False,
            },
            "status": "needs_fix",
        }
        host = self.host_with_plate(data)

        with patch.object(
            z3_review_runtime,
            "_refresh_after_change",
        ) as refresh, patch.object(
            z3_review_runtime,
            "_persist_review_az_revision_best_effort",
        ) as save_az:
            result = z3_review_runtime.toggle_review_excluded(
                host,
                persist=True,
            )

        self.assertTrue(result["ok"])
        refresh.assert_called_once()
        save_az.assert_called_once_with(
            host,
            "plate_000001",
            data,
            event="exclude_toggle",
        )

    def test_n_event_with_persist_false_does_not_write_az(self):
        data = {
            "gold_state": {},
            "status": "needs_fix",
        }
        host = self.host_with_plate(data)

        with patch.object(
            z3_review_runtime,
            "_refresh_after_change",
        ), patch.object(
            z3_review_runtime,
            "_persist_review_az_revision_best_effort",
        ) as save_az:
            z3_review_runtime.toggle_review_excluded(
                host,
                persist=False,
            )

        save_az.assert_not_called()

    def test_o_event_persists_az_revision(self):
        data = {
            "review_state": {
                "status": z3_review_runtime.REVIEW_IN_PROGRESS,
            },
            "characters": [
                {
                    "character": "A",
                    "bbox": [1, 1, 10, 20],
                }
            ],
            "gold_state": {
                "excluded": False,
            },
            "status": "needs_fix",
        }
        host = self.host_with_plate(data)
        host._ensure_plate_source_metadata = Mock()
        host._get_plate_source_bucket.return_value = "local_manual"

        host._characters_to_text.return_value = "A"
        with patch.object(
            z3_review_runtime,
            "get_review_quality_status",
            return_value="perfect",
        ), patch.object(
            z3_review_runtime,
            "plate_gt",
            return_value="A",
        ), patch.object(
            z3_review_runtime,
            "_refresh_after_change",
        ), patch.object(
            z3_review_runtime,
            "_persist_review_az_revision_best_effort",
        ) as save_az:
            result = z3_review_runtime.confirm_review_gold(
                host,
                persist=True,
                quiet=True,
                refresh=True,
            )

        self.assertTrue(result["ok"])
        save_az.assert_called_once_with(
            host,
            "plate_000001",
            data,
            event="review_approved",
        )


if __name__ == "__main__":
    unittest.main()
