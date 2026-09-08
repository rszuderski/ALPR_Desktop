from pathlib import Path
from types import MethodType, SimpleNamespace
import time
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from auto_annotation_tool.gui import z2_panel_workflow as workflow, z2_preview_state as state
from auto_annotation_tool.gui.tab_annotation import AnnotationTab
from auto_annotation_tool.gui import z2_preview_workflow as preview


def annotation(name, kind="empty", approved=False):
    attributes = {"manually_edited": "true"} if kind == "manual" else {}
    detections = [] if kind == "empty" else [SimpleNamespace(label="plate", attributes=attributes)]
    return SimpleNamespace(filename=name, detections=detections, _approved_for_training=approved)


class AutoResultListTests(unittest.TestCase):
    def owner(self, free):
        owner = Mock()
        owner._is_free_mode_session_context.return_value = free
        for key in ("_campaign_pending_batch_summary", "_campaign_reuse_manual_summary",
                    "_campaign_auto_manual_overlay_bundle", "_campaign_auto_pre_run_snapshot",
                    "_campaign_auto_pre_run_visible_state"):
            setattr(owner, key, {})
        owner._pending_preview_approved_filenames = set()
        owner._campaign_reuse_manual_filenames = set()
        owner._campaign_hidden_project_approved_filenames = set()
        owner._get_preview_approved_filenames_base.return_value = set()
        owner._get_campaign_hidden_project_approved_filenames_runtime.return_value = set()
        owner._get_campaign_manual_touched_filenames.return_value = set()
        owner._get_preview_list_render_state = MethodType(state._get_preview_list_render_state, owner)
        owner._mark_auto_plate_origin_for_annotations = AnnotationTab._mark_auto_plate_origin_for_annotations
        owner.current_preview_index = 2
        owner.current_annotations = [annotation("detected.jpg"), annotation("empty.jpg"),
                                     annotation("manual.jpg", "manual", True), annotation("auto_ok.jpg", "auto", True)]
        state._build_preview_list_render_state_cache(owner)
        owner.rows = [state._preview_annotation_status_tag(owner, ann) for ann in owner.current_annotations]
        owner._restore_preview_from_annotation_run.return_value = True

        def publish(**kwargs):
            self.assertTrue(kwargs["preserve_selection"])
            self.assertTrue(kwargs["rebuild_state_cache"])
            self.assertTrue(kwargs["recolor_rows"])
            state._build_preview_list_render_state_cache(owner)
            owner.rows = [state._preview_annotation_status_tag(owner, ann) for ann in owner.current_annotations]
        owner._populate_preview_list_async.side_effect = publish
        return owner

    def test_final_result_replaces_stale_rows_in_campaign_and_free_mode(self):
        for free in (False, True):
            for restored in (False, True):
                with self.subTest(free=free, restored=restored):
                    owner = self.owner(free)
                    owner._restore_preview_from_annotation_run.return_value = restored
                    owner.current_annotations[0].detections = annotation("x", "auto").detections
                    self.assertEqual(owner.rows[0], "--")
                    workflow._finalize_successful_annotation_run_ui(owner, Path("test-run"))
                    self.assertEqual(owner.rows, ["A", "--", "M|OK", "A|OK"])
                    owner._populate_preview_list_async.assert_called_once()
                    owner._restore_preview_from_annotation_run.assert_called_once_with(Path("test-run"), defer_ui_restore=True)
                    self.assertEqual(owner.current_preview_index, 2)

    def test_scope_merge_precedes_final_list_publication(self):
        owner = self.owner(False)
        def merge(tab, _run):
            tab.current_annotations.append(annotation("outside_scope.jpg", "manual"))
            tab.current_annotations[0] = annotation("detected.jpg", "auto")
            return 1
        with patch.object(workflow, "_merge_pre_run_visible_state_after_auto", side_effect=merge):
            workflow._finalize_successful_annotation_run_ui(owner, Path("test-run"))
        self.assertEqual(owner.rows, ["A", "--", "M|OK", "A|OK", "M"])
        owner._populate_preview_list_async.assert_called_once()

    def test_manual_template_does_not_become_auto(self):
        owner = self.owner(False)
        owner._open_existing_run_for_campaign_review.return_value = True
        workflow._finalize_successful_annotation_run_ui(owner, Path("test-run"), manual_template=True)
        owner._populate_preview_list_async.assert_not_called()
        self.assertEqual(owner.rows[0], "--")

    def test_real_listbox_cancels_old_batches_and_paints_new_statuses(self):
        root = tk.Tk()
        root.withdraw()
        owner = self.owner(True)
        owner.frame = tk.Frame(root)
        owner.preview_listbox = tk.Listbox(owner.frame)
        owner._preview_list_populate_token = 0
        owner._preview_list_populate_after_id = None
        owner._campaign_deferred_run_restore_in_progress = False
        owner._cancel_preview_list_population = MethodType(state._cancel_preview_list_population, owner)
        owner._build_preview_list_render_state_cache = MethodType(state._build_preview_list_render_state_cache, owner)
        owner._get_preview_list_entries.side_effect = lambda: list(enumerate(owner.current_annotations))
        owner._get_preview_display_index.side_effect = lambda index: index
        owner._preview_annotation_status_tag = MethodType(state._preview_annotation_status_tag, owner)
        owner._preview_list_item_text.side_effect = lambda ann, **kw: state._preview_list_item_text(owner, ann, **{**kw, "lightweight": True})
        owner._preview_list_color_plan.return_value = ("auto", "green")
        owner._preview_list_effective_color_bucket.side_effect = lambda ann: state._get_preview_list_render_state(owner, ann)["bucket"]
        owner._preview_list_color_for_bucket.return_value = "blue"
        owner.current_annotations = [annotation(f"image_{i}.jpg") for i in range(600)]
        owner._populate_preview_list_async.side_effect = lambda **kw: preview._populate_preview_list_async(owner, **kw)
        try:
            preview._populate_preview_list_async(owner, batch_size=10, render_current=False, preserve_selection=True)
            old_job = owner._preview_list_populate_after_id
            self.assertTrue(old_job)
            self.assertIn("[--]", owner.preview_listbox.get(0))
            owner.current_annotations = [annotation(f"image_{i}.jpg", "auto") for i in range(600)]
            workflow._finalize_successful_annotation_run_ui(owner, Path("test-run"))
            self.assertNotIn(old_job, root.tk.call("after", "info"))
            deadline = time.monotonic() + 3
            while owner._preview_list_populate_after_id and time.monotonic() < deadline:
                root.update()
                time.sleep(0.005)
            self.assertEqual(owner.preview_listbox.size(), 600)
            self.assertTrue(all("[A]" in row for row in owner.preview_listbox.get(0, "end")))
            self.assertEqual(owner.preview_listbox.curselection(), (2,))
        finally:
            owner._cancel_preview_list_population()
            root.destroy()
