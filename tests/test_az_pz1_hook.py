import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from auto_annotation_tool.registry import RegistryDatabase
from auto_annotation_tool.registry.az_registry import (
    AZRegistry,
    ensure_campaign_project,
    project_id_from_folder_name,
)
from auto_annotation_tool.gui import z3_extraction_tab_ui as extraction


class AZCampaignProjectRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name) / "Workspace"
        self.workspace.mkdir()
        self.registry = AZRegistry(
            RegistryDatabase(
                self.workspace / "_registry" / "alpr_registry.sqlite3"
            ),
            workspace_dir=self.workspace,
        )
        self.registry.initialize()

    def tearDown(self):
        self.tmp.cleanup()

    def test_project_id_is_stable_and_case_insensitive(self):
        a = project_id_from_folder_name("Demo_ABC123")
        b = project_id_from_folder_name("demo_abc123")
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("PRJ-"))

    def test_ensure_campaign_project_is_idempotent(self):
        first = ensure_campaign_project(
            self.registry,
            project_name="Demo",
            folder_name="Demo_ABC123",
        )
        second = ensure_campaign_project(
            self.registry,
            project_name="Demo",
            folder_name="Demo_ABC123",
        )
        self.assertEqual(first, second)

        with self.registry.database.read_connection() as con:
            row = con.execute(
                """
                SELECT project_id, campaign_key, folder_name, display_name
                FROM projects
                WHERE project_id = ?
                """,
                (first,),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["campaign_key"], "Demo")
        self.assertEqual(row["folder_name"], "Demo_ABC123")
        self.assertEqual(row["display_name"], "Demo")


class AZPZ1HookTests(unittest.TestCase):
    def _host(self, linear=False):
        host = Mock()
        host._step3_linear_mode = linear
        return host

    def test_free_mode_registers_without_project_membership(self):
        report = Mock()
        report.to_dict.return_value = {
            "total": 2,
            "new_crops": 2,
            "reused_crops": 0,
            "new_artifacts": 2,
        }
        fake_registry = Mock()

        with patch.object(
            extraction.CONFIG,
            "WORKSPACE_DIR",
            Path("Workspace"),
        ), patch.object(
            extraction.CAMPAIGN,
            "get_active_project_name",
            return_value="",
        ), patch(
            "auto_annotation_tool.registry.az_registry.AZRegistry.for_workspace",
            return_value=fake_registry,
        ), patch(
            "auto_annotation_tool.registry.pz1_run_registry.register_pz1_preview_run",
            return_value=report,
        ) as register:
            result = extraction._register_completed_pz1_run_in_az_registry(
                self._host(False),
                Path("run"),
                interpolation="lanczos4",
                source_at_ref="annotations.xml",
            )

        self.assertTrue(result["ok"])
        kwargs = register.call_args.kwargs
        self.assertIsNone(kwargs["project_id"])
        self.assertIsNone(kwargs["iteration_num"])
        self.assertEqual(kwargs["source_mode"], "pz1")

    def test_campaign_ensures_project_and_passes_iteration(self):
        report = Mock()
        report.to_dict.return_value = {
            "total": 1,
            "new_crops": 1,
            "reused_crops": 0,
            "new_artifacts": 1,
        }
        fake_registry = Mock()

        project_state = {
            "projects": {
                "Demo": {
                    "folder_name": "Demo_ABC123",
                }
            }
        }

        with patch.object(
            extraction.CONFIG,
            "WORKSPACE_DIR",
            Path("Workspace"),
        ), patch.object(
            extraction.CAMPAIGN,
            "state",
            project_state,
        ), patch.object(
            extraction.CAMPAIGN,
            "get_active_project_name",
            return_value="Demo",
        ), patch.object(
            extraction.CAMPAIGN,
            "get_current_iteration_num",
            return_value=4,
        ), patch(
            "auto_annotation_tool.registry.az_registry.AZRegistry.for_workspace",
            return_value=fake_registry,
        ), patch(
            "auto_annotation_tool.registry.az_registry.ensure_campaign_project",
            return_value="PRJ-TEST",
        ) as ensure_project, patch(
            "auto_annotation_tool.registry.pz1_run_registry.register_pz1_preview_run",
            return_value=report,
        ) as register:
            result = extraction._register_completed_pz1_run_in_az_registry(
                self._host(True),
                Path("run"),
                interpolation="cubic",
                source_at_ref="source.xml",
            )

        self.assertTrue(result["ok"])
        ensure_project.assert_called_once_with(
            fake_registry,
            project_name="Demo",
            folder_name="Demo_ABC123",
        )
        kwargs = register.call_args.kwargs
        self.assertEqual(kwargs["project_id"], "PRJ-TEST")
        self.assertEqual(kwargs["iteration_num"], 4)
        self.assertEqual(kwargs["interpolation"], "cubic")
        self.assertEqual(kwargs["source_at_ref"], "source.xml")

    def test_registry_failure_is_non_fatal(self):
        with patch.object(
            extraction.CAMPAIGN,
            "get_active_project_name",
            return_value="",
        ), patch(
            "auto_annotation_tool.registry.az_registry.AZRegistry.for_workspace",
            side_effect=RuntimeError("registry down"),
        ):
            result = extraction._register_completed_pz1_run_in_az_registry(
                self._host(False),
                Path("run"),
                interpolation="lanczos4",
            )

        self.assertFalse(result["ok"])
        self.assertIn("registry down", result["error"])

    def test_real_hook_runs_after_reextract_merge(self):
        source = inspect.getsource(extraction.run_extraction)
        merge_pos = source.find(
            "_merge_reextract_seed_metadata_into_preview("
        )
        hook_pos = source.find(
            "_register_completed_pz1_run_in_az_registry("
        )
        processing_pos = source.find(
            "if self.is_processing and session_token == self._project_reset_token:"
        )

        self.assertGreaterEqual(merge_pos, 0)
        self.assertGreater(hook_pos, merge_pos)
        self.assertGreater(processing_pos, hook_pos)


if __name__ == "__main__":
    unittest.main()
