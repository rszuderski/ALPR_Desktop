import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from auto_annotation_tool.gui import campaign_step1_assets as assets
from auto_annotation_tool.campaign_transition_evaluator import CampaignTransitionEvalContext, is_transition_ready
from auto_annotation_tool.campaign_transition_specs import get_transition_specs_for_edge
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette


class EntryResourceSelectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.images = root / "images"
        self.source = root / "external"
        self.draft = root / "project" / "_staging" / "auto_annotations" / "import"
        for directory in (self.images, self.source, self.draft):
            directory.mkdir(parents=True)
        for directory in (self.source, self.draft):
            (directory / "annotations.xml").write_text("<annotations/>", encoding="utf-8")
        self.mt = root / "mt.pt"
        self.mz = root / "mz.pt"
        self.mt.touch()
        self.mz.touch()
        self.campaign = Mock()
        self.campaign.get_active_project_name.return_value = "demo"
        self.campaign.get_iteration_path.return_value = ""
        self.campaign.get_global_model.side_effect = lambda role: str(self.mt if role == "plate" else self.mz)
        patcher = patch.object(assets, "CAMPAIGN", self.campaign)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.owner = Mock()
        self.owner.STEP1_CHAR_MIN_IMAGES = 10
        self.owner.STEP3_CHAR_MIN_PLATES = 10
        self.owner._get_iteration_target.return_value = ""
        self.owner._get_step1_presentation_mode.return_value = "operational_assets"
        self.owner._get_project_start_mode.return_value = "assets"
        self.owner._get_project_start_effective_images_source.return_value = {
            "effective_dir": self.images, "effective_exists": True, "effective_count": 10,
        }
        self.owner._get_step1_char_route_preflight_state.return_value = {}
        self.owner._get_project_start_plate_source_info.return_value = {
            "run_path": str(self.draft), "xml_path": str(self.draft / "annotations.xml"),
            "images_path": str(self.images), "source_mode": "draft",
        }
        self.owner._get_project_start_effective_model_state.return_value = {}
        self.owner._validate_project_model_selection.return_value = (True, "")
        self.owner._get_model_identity_label.side_effect = lambda value: Path(value).stem
        self.owner._format_project_start_asset_source.side_effect = lambda value: str(value or "")
        self.owner._get_iteration_image_count.return_value = 10
        self.owner._count_images_in_dir.return_value = 10
        self.owner._get_project_start_adoptable_normalized_image_names.return_value = {"abc_001.jpg", "xyz_002.jpg"}
        self.owner._summarize_project_start_xml_match.return_value = {"total": 2}
        self.compatibility = {
            "checked": True, "ok": True, "matched": 2, "total": 2,
            "matched_plate_count": 3, "adoptable_matched": 2, "adoptable_matched_plate_count": 3,
            "images_dir": self.images,
        }
        self.owner._check_project_start_run_compatibility.return_value = self.compatibility
        self.annotation = Mock()
        self.annotation._get_run_plate_annotation_counts.return_value = (2, 3)
        self.annotation._import_external_annotation_run_to_workspace.return_value = (self.draft, "", False)
        self.owner.app.tabs = {"annotation": self.annotation}
        self.owner.app.palette = get_theme_palette()

    def import_at(self, **kwargs):
        return assets._import_project_start_plate_run(
            self.owner, self.source / "annotations.xml", refresh_dashboard_after_import=False, **kwargs,
        )

    def refresh_rows(self):
        self.owner._set_project_start_asset_row_state.reset_mock()
        assets._refresh_project_start_panel(self.owner)
        return {call.args[0]: call.kwargs for call in self.owner._set_project_start_asset_row_state.call_args_list}

    def test_import_without_route_creates_review_source_and_does_not_choose_path(self):
        self.owner._get_iteration_target.side_effect = AssertionError("Import must not query the route")
        progress = Mock()
        self.assertTrue(self.import_at(confirm_import=False, progress_callback=progress))
        source = self.campaign.set_project_start_plate_source.call_args.kwargs
        self.assertEqual(source["source_mode"], "draft")
        self.assertEqual(source["source_run_path"], str(self.draft.resolve()))
        self.assertEqual(source["source_input_path"], str(self.images.resolve()))
        self.campaign.set_iteration_path.assert_not_called()
        self.campaign.set_iteration_target.assert_not_called()
        self.assertEqual(progress.call_args.args[0], 100.0)
        args = self.annotation._import_external_annotation_run_to_workspace.call_args.kwargs
        self.assertFalse(args["copy_images"])
        self.assertEqual(args["allowed_normalized_names"], {"abc_001.jpg", "xyz_002.jpg"})

    def test_same_import_contract_with_either_route(self):
        for target in ("plate", "char"):
            self.owner._get_iteration_target.return_value = target
            self.assertTrue(self.import_at(confirm_import=False))
            self.assertEqual(self.campaign.set_project_start_plate_source.call_args.kwargs["source_mode"], "draft")
        self.owner._get_iteration_target.assert_not_called()

    def test_import_still_requires_images(self):
        self.owner._get_project_start_effective_images_source.return_value = {}
        self.assertFalse(self.import_at(confirm_import=False))
        self.annotation._import_external_annotation_run_to_workspace.assert_not_called()
        self.campaign.set_project_start_plate_source.assert_not_called()

    def test_incompatible_annotations_do_not_replace_project_source(self):
        self.owner._check_project_start_run_compatibility.return_value = {
            "checked": True, "ok": False, "matched": 0, "total": 2, "images_dir": self.images,
        }
        self.assertFalse(self.import_at(confirm_import=False))
        self.owner.app.themed_error.assert_called()
        self.campaign.set_project_start_plate_source.assert_not_called()

    def test_cancel_keeps_selected_source_and_route(self):
        self.owner._choose_project_start_annotation_import_mode.return_value = None
        self.assertFalse(self.import_at(confirm_import=True))
        self.campaign.set_project_start_plate_source.assert_not_called()
        self.campaign.set_iteration_path.assert_not_called()

    def test_resources_can_be_selected_before_route_and_counts_are_preserved(self):
        rows = self.refresh_rows()
        states = {call.args[0]: call.kwargs["enabled"] for call in self.owner._set_project_start_badge_button_state.call_args_list
                  if "enabled" in call.kwargs}
        self.assertTrue(states[self.owner.btn_ingest_import_plate_run])
        self.assertTrue(states[self.owner.btn_ingest_pick_plate_model])
        self.assertTrue(states[self.owner.btn_ingest_pick_char_model])
        self.assertEqual(rows["plate_run"]["counter_text"], "2 obrazów / 3 tablic")
        self.assertIn("AT do kontroli", rows["plate_run"]["validation_text"])
        self.assertFalse(rows["plate_run"]["meta"]["contract_ready"])
        self.assertEqual(rows["plate_model"]["tone"], "success")
        self.assertEqual(rows["char_model"]["tone"], "success")
        self.assertEqual(rows["images"]["tone"], "success")
        self.campaign.set_iteration_path.assert_not_called()

    def test_model_validation_is_not_hidden_by_missing_route(self):
        self.owner._validate_project_model_selection.return_value = (False, "Niepoprawny model")
        rows = self.refresh_rows()
        for key in ("plate_model", "char_model"):
            self.assertEqual(rows[key]["tone"], "error")
            self.assertEqual(rows[key]["validation_text"], "Niepoprawny model")

    def test_mz_is_preserved_if_plate_route_is_selected_later(self):
        self.owner._get_iteration_target.return_value = "plate"
        self.campaign.get_iteration_path.return_value = "plate_training"
        rows = self.refresh_rows()
        self.assertEqual(rows["char_model"]["source_path"], str(self.mz))
        self.assertEqual(rows["char_model"]["tone"], "success")
        self.assertIn("nie jest używany", rows["char_model"]["validation_text"])
        self.assertNotEqual(rows["char_model"]["requirement"], "disabled")

    def test_resource_refresh_does_not_persist_an_inferred_legacy_route(self):
        for target in ("plate", "char"):
            self.owner._get_iteration_target.return_value = target
            self.refresh_rows()
        self.campaign.set_iteration_path.assert_not_called()

    def test_az_is_honestly_planned_not_blocked_by_route(self):
        rows = self.refresh_rows()
        self.assertIn("Planowane", rows["char_run"]["validation_text"])
        self.assertNotIn("wybierz", rows["char_run"]["validation_text"].lower())
        self.assertFalse(rows["char_run"]["meta"]["contract_ready"])

    def test_model_picker_does_not_require_route(self):
        self.owner._open_project_start_model_candidate_browser.return_value = True
        assets._choose_project_start_model(self.owner, "char", parent=self.owner.frame)
        self.owner._open_project_start_model_candidate_browser.assert_called_once_with("char", parent=self.owner.frame)
        self.owner._get_iteration_target.assert_not_called()

    def test_resources_alone_do_not_approve_t01_without_explicit_work_choice(self):
        spec = get_transition_specs_for_edge("e1_to_e2")[0]
        self.assertFalse(is_transition_ready(spec, CampaignTransitionEvalContext(
            current_step=1, current_iteration=1, image_count=1000, material_ready=True, plate_material_count=50,
        )))


if __name__ == "__main__":
    unittest.main()
