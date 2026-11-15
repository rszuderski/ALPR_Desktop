from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_annotation_tool.data_models import Detection, ImageAnnotation
from auto_annotation_tool.gui import z2_canvas_interaction as interaction
from auto_annotation_tool.gui import z2_gt_readiness as readiness
from auto_annotation_tool.gui import z2_gt_review as review
from auto_annotation_tool.gui import z2_layout_ui_runtime as layout
from auto_annotation_tool.gui import z2_manifest_runtime as manifest
from auto_annotation_tool.gui import z2_plate_gt_inline as inline
from auto_annotation_tool.gui import z2_preview_editor as editor
from auto_annotation_tool.gui import z2_preview_state as state
from auto_annotation_tool.gui import z2_preview_workflow as workflow
from auto_annotation_tool.gui import z2_run_io_runtime as run_io


def annotation(name, *texts):
    return ImageAnnotation(name, 200, 100, [Detection(
        "plate", .95, (10, 10, 80, 30), polygon=[(10, 10), (80, 10), (80, 30), (10, 30)],
        attributes={"ground_truth_text": text, "manually_edited": "true", "plate_annotation_id": f"{name}-{i}"},
    ) for i, text in enumerate(texts)])


@pytest.fixture
def host(monkeypatch):
    manager = SimpleNamespace(get_iteration_target=lambda: "char")
    monkeypatch.setattr(readiness, "CAMPAIGN", manager)
    monkeypatch.setattr(inline, "CAMPAIGN", manager)
    monkeypatch.setattr(manifest, "_get_campaign_hidden_project_approved_filenames_runtime", lambda h: set())
    owner = SimpleNamespace(
        app=SimpleNamespace(palette={"error": "#ff0000", "accent": "#0000ff", "warning": "#ffff00"}),
        current_annotations=[], current_preview_index=0,
        _is_free_mode_session_context=lambda: False,
        _get_plate_detections=lambda ann: ann.plates if ann else [],
        _is_plate_detection_label=lambda value: str(value).lower() == "plate",
        _campaign_reuse_manual_badge=lambda: "REUSE",
        _preview_approved_filenames=set(), _campaign_pending_approved_filenames=set(),
        _preview_dirty_images=set(), _campaign_auto_manual_overlay_bundle={},
        _get_campaign_hidden_project_approved_filenames_runtime=lambda: set(),
        _get_campaign_manual_touched_filenames=lambda: set(),
        _resolve_preview_image_path=lambda ann: None,
        _get_preview_display_index=lambda i: i,
        _get_preview_metric_filter_thresholds=lambda: (.99, .99),
        _normalize_preview_list_sort_mode=lambda mode=None: mode or "Status: problem, ED, OK",
        _get_preview_annotation_quality_summary=Mock(),
        preview_listbox=Mock(), preview_missing_gt_only_check=Mock(),
        _preview_draw_mode=False,
        _preview_shortcuts_enabled=lambda *a, **kw: True,
        _preview_shortcut_is_duplicate=lambda *a: False,
    )
    for name in (
        "_update_preview_edit_status", "_update_preview_toolbar_state", "_mark_preview_user_interaction",
        "_schedule_preview_approved_persist", "_persist_preview_approved_filenames",
        "_refresh_preview_list_row_for_actual_index", "_refresh_preview_canvas_light",
        "_refresh_preview_list_summary", "_queue_free_mode_session_save",
        "_refresh_step2_action_states", "_schedule_preview_approval_followup_refresh",
        "_schedule_preview_autosave", "_clear_listbox_selection_fast", "_refresh_preview_list",
        "_update_preview_canvas_metrics_overlay", "_mark_preview_image_dirty",
    ):
        setattr(owner, name, Mock())
    for module, names in (
        (manifest, ("_get_preview_approved_filenames_base", "_get_preview_approved_filenames",
                    "_preview_annotation_can_be_approved_for_export", "_preview_annotation_is_explicitly_approved")),
        (state, ("_get_preview_list_render_state", "_preview_annotation_status_tag",
                 "_preview_annotation_origin_tag", "_preview_annotation_is_manually_corrected",
                 "_preview_annotation_has_auto_plate", "_preview_annotation_is_reused_from_previous_manual",
                 "_preview_annotation_sort_bucket", "_preview_annotation_sort_bucket_for_mode",
                 "_preview_list_status_priority", "_preview_list_item_text")),
    ):
        for name in names:
            setattr(owner, name, MethodType(getattr(module, name), owner))
    owner._get_preview_annotation = lambda: owner.current_annotations[owner.current_preview_index]
    monkeypatch.setattr(workflow, "_resolve_campaign_graph_gate_context", lambda h: ("T03", "", {}))
    monkeypatch.setattr(workflow, "_schedule_campaign_char_source_refresh_after_approval", Mock())
    monkeypatch.setattr(workflow, "_schedule_campaign_post_approval_refresh", Mock())
    monkeypatch.setattr(workflow, "_update_preview_approval_badge_fast", Mock())
    monkeypatch.setattr(workflow, "_try_update_campaign_right_panel_counts_after_approval", lambda h: True)
    return owner


@pytest.mark.parametrize("texts,can_approve", [((), False), (("",), True), (("AA123", ""), True),
                                              (("AA123", "BB234"), True), (("  - ",), True)])
def test_character_route_can_approve_geometry_without_gt(host, texts, can_approve):
    ann = annotation("AA123_BB234_001.jpg", *texts)
    assert host._preview_annotation_can_be_approved_for_export(ann) is can_approve


@pytest.mark.parametrize("free,target", [(True, "char"), (False, "plate")])
def test_other_routes_allow_geometry_approval_without_gt(host, monkeypatch, free, target):
    host._is_free_mode_session_context = lambda: free
    monkeypatch.setattr(readiness.CAMPAIGN, "get_iteration_target", lambda: target)
    assert host._preview_annotation_can_be_approved_for_export(annotation("plain.jpg", ""))


@pytest.mark.parametrize("cached", [False, True])
def test_old_geometry_approval_remains_valid_without_text_gt(host, cached):
    ann = annotation("old.jpg", "AA123", "")
    ann._approved_for_training = True
    host.current_annotations = [ann]
    host._preview_approved_filenames = {"old.jpg"}
    host._campaign_reuse_manual_filenames = {"old.jpg"}
    if cached:
        state._build_preview_list_render_state_cache(host)
        assert host._get_preview_list_render_state(ann)["approved"] is True
    assert host._preview_annotation_is_explicitly_approved(ann)
    assert host._get_preview_approved_filenames() == {"old.jpg"}
    assert host._preview_annotation_sort_bucket(ann) == "approved"
    assert state._preview_list_item_color(host, ann) != "#ff0000"
    assert "OK" in host._preview_list_item_text(ann, lightweight=True)
    host._get_preview_annotation_quality_summary.assert_not_called()


def test_missing_gt_does_not_move_approved_geometry_to_problems(host):
    complete = annotation("a.jpg", "AA123")
    missing = annotation("z.jpg", "")
    host.current_annotations = [complete, missing]
    host._preview_approved_filenames = {"a.jpg", "z.jpg"}
    state._build_preview_list_render_state_cache(host)
    rows = state._build_preview_list_sorted_entries(host, "Status: problem, ED, OK")
    assert [a.filename for _, a in rows] == ["a.jpg", "z.jpg"]


def test_missing_gt_filter_reveals_old_approved_rows_despite_quality_thresholds(host):
    host.preview_missing_gt_only_var = SimpleNamespace(get=lambda: True)
    host._get_campaign_hidden_project_approved_filenames_runtime = lambda: {"partial.jpg"}
    host.current_annotations = [annotation("ok.jpg", "AA123"), annotation("partial.jpg", "AA123", ""),
                                annotation("empty.jpg"), annotation("missing.jpg", "")]
    rows = layout._filter_preview_list_entries(host, list(enumerate(host.current_annotations)))
    assert [i for i, _ in rows] == [1, 3]
    host._get_preview_annotation_quality_summary.assert_not_called()
    review.refresh_missing_gt_count(host)
    host.preview_missing_gt_only_check.configure.assert_called_with(text="Brak GT: 2 zdjęć / 2 ramek")


@pytest.mark.parametrize("fast", [False, True])
def test_mixed_bulk_approval_accepts_geometry_but_rejects_empty_images(host, fast):
    host.current_annotations = [annotation("ok.jpg", "AA123"), annotation("partial.jpg", "AA123", ""),
                                annotation("empty.jpg")]
    workflow._set_selected_preview_images_approved(
        host, True, actual_indices=[0, 1, 2], show_warning_modal=False,
        persist_immediately=not fast, refresh_export_sources=not fast, schedule_followup_refresh=fast,
    )
    assert host._preview_approved_filenames == {"ok.jpg", "partial.jpg"}
    assert host._campaign_pending_approved_filenames == {"ok.jpg", "partial.jpg"}
    assert host.current_annotations[1]._approved_for_training
    assert not host.current_annotations[2]._approved_for_training


def test_space_allows_approving_geometry_with_partial_gt(host):
    host.current_annotations = [annotation("partial.jpg", "AA123", "", "")]
    host._set_selected_preview_images_approved = Mock()
    assert interaction._on_preview_toggle_image_approval_shortcut(host) == "break"
    host._set_selected_preview_images_approved.assert_called_once()


def test_clearing_gt_preserves_geometry_approval_and_refreshes_only_current_row(host, monkeypatch):
    ann = annotation("approved.jpg", "AA123")
    ann._approved_for_training = True
    host.current_annotations = [ann, annotation("next.jpg", "BB234")]
    host._preview_approved_filenames = {ann.filename}
    host._campaign_pending_approved_filenames = {ann.filename}
    host._save_preview_edits = Mock(return_value=True)
    monkeypatch.setattr(inline.z2_gt_pack_runtime, "sync_plate_gt_after_xml_save", Mock(return_value={"ok": True}))
    ok, text = inline.save_inline_plate_gt_value(host, ann, ann.plates[0], "", refresh_gate=False, lightweight_save=True)
    assert ok and text == ""
    assert ann._approved_for_training
    assert host._get_preview_approved_filenames() == {ann.filename}
    assert host._campaign_pending_approved_filenames == {ann.filename}
    assert host.current_preview_index == 0
    host._refresh_preview_list.assert_not_called()
    host._refresh_preview_list_row_for_actual_index.assert_called_once_with(0, refresh_summary=False, lightweight=True)
    host._schedule_preview_approved_persist.assert_not_called()


def test_filling_last_gt_removes_red_label_without_navigating(host, monkeypatch):
    ann = annotation("partial.jpg", "AA123", "")
    host.current_annotations = [ann]
    state._build_preview_list_render_state_cache(host)
    host._save_preview_edits = Mock(return_value=True)
    monkeypatch.setattr(inline.z2_gt_pack_runtime, "sync_plate_gt_after_xml_save", Mock(return_value={"ok": True}))
    ok, text = inline.save_inline_plate_gt_value(host, ann, ann.plates[1], "bb234", refresh_gate=False, lightweight_save=True)
    assert ok and text == "BB234"
    assert host._preview_annotation_can_be_approved_for_export(ann)
    assert "BRAK GT" not in host._preview_list_item_text(ann, lightweight=True)
    assert host.current_preview_index == 0
    host._refresh_preview_list.assert_not_called()


def test_failed_gt_clear_restores_text_and_approval(host):
    ann = annotation("approved.jpg", "AA123")
    ann._approved_for_training = True
    host.current_annotations = [ann]
    host._preview_approved_filenames = {ann.filename}
    host._save_preview_edits = Mock(return_value=False)
    ok, text = inline.save_inline_plate_gt_value(host, ann, ann.plates[0], "", refresh_gate=False, lightweight_save=True)
    assert not ok and text == "AA123"
    assert host._preview_annotation_is_explicitly_approved(ann)
    host._schedule_preview_approved_persist.assert_not_called()


def test_approved_geometry_counter_includes_rows_with_missing_gt(host):
    host.current_annotations = [annotation("ok.jpg", "AA123"), annotation("partial.jpg", "AA123", "")]
    host._preview_approved_filenames = {ann.filename for ann in host.current_annotations}
    counts = run_io._get_current_preview_plate_count_state(host)
    assert counts["approved_images"] == 2
    assert counts["approved_plates"] == 3
    assert counts["total_plates"] == 3


def test_tk_filter_and_row_color_update_after_completing_gt(host):
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        ann = annotation("partial.jpg", "AA123", "")
        host.current_annotations = [ann, annotation("ready.jpg", "BB234")]
        host._invalidate_preview_list_frozen_order = Mock()
        review.build_missing_gt_filter(host, root)
        review.refresh_missing_gt_count(host)
        assert "1 zdjęć / 1 ramek" in host.preview_missing_gt_only_check.cget("text")
        host.preview_filter_conf_var = tk.DoubleVar(master=root, value=.99)
        host.preview_filter_fit_var = tk.DoubleVar(master=root, value=.99)
        host.preview_missing_gt_only_check.invoke()
        assert review.missing_gt_filter_active(host)
        assert host.preview_filter_conf_var.get() == host.preview_filter_fit_var.get() == 0.0
        host._refresh_preview_list.assert_called_once_with(preserve_selection=True, render_current=True)
        assert [i for i, _ in layout._filter_preview_list_entries(host, list(enumerate(host.current_annotations)))] == [0]

        host.preview_listbox = tk.Listbox(root, exportselection=False)
        host.preview_listbox.insert(tk.END, "partial.jpg", "ready.jpg")
        host._preview_list_item_color = MethodType(state._preview_list_item_color, host)
        host._clear_listbox_selection_fast = lambda box: box.selection_clear(0, tk.END)
        host._refresh_preview_list_row_for_actual_index = MethodType(editor._refresh_preview_list_row_for_actual_index, host)
        review.refresh_annotation_gt_review(host, ann)
        assert host.preview_listbox.itemcget(0, "foreground") != "#ff0000"
        assert "BRAK GT" not in host.preview_listbox.get(0)

        ann.plates[1].attributes["ground_truth_text"] = "CC345"
        review.refresh_annotation_gt_review(host, ann)
        root.update()
        assert "BRAK GT" not in host.preview_listbox.get(0)
        assert host.preview_listbox.itemcget(0, "foreground") != "#ff0000"
        assert "0 zdjęć / 0 ramek" in host.preview_missing_gt_only_check.cget("text")
        assert host.preview_listbox.curselection() == (0,)
        assert host.current_preview_index == 0
    finally:
        root.destroy()
