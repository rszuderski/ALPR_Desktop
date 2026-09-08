import ast
import copy
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from unittest.mock import Mock, patch

import pytest

from auto_annotation_tool.gui.tab_character_annotation import CharacterAnnotationTab
from auto_annotation_tool.gui import z3_campaign_flow as flow
from auto_annotation_tool.gui import campaign_dashboard_ui as dashboard
from auto_annotation_tool.gui import campaign_ui_helpers
from auto_annotation_tool.gui.campaign_graph_presentation import close_graph_dialogs
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette


def two_rows(layout="two_row", override=None):
    data = {"plate_layout": layout, "characters": [
        {"character": str(i), "bbox": box, "reading_row": 2}
        for i, box in enumerate(([10, 10, 30, 40], [45, 10, 65, 40], [10, 70, 30, 100], [45, 70, 65, 100]))],
        "layout_separator": {"x1": 0, "x2": 1, "y1": 0.5, "y2": 0.5, "source": "auto_default_midline"}}
    if override:
        data["plate_layout_override"] = override
    return data


@pytest.mark.parametrize("layout,override", [("two_row", None), ("two_row_candidate", None), ("two_row", "two_row")])
def test_auto_separator_recovers_geometry_without_image_dimensions_or_trusting_stale_rows(layout, override):
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    data = two_rows(layout, override)
    before = copy.deepcopy(data["characters"])
    separator = host._ensure_preview_layout_separator(data)
    assert separator["x2"] == 65
    assert separator["y1"] == separator["y2"] == 55
    assert [host._get_preview_row_for_bbox(c["bbox"], data) for c in data["characters"]] == [1, 1, 2, 2]
    assert data["characters"] == before
    assert host._is_preview_layout_separator_interactive(data)


def test_manual_and_dragging_separator_control_rows_without_moving_character_boxes():
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    data = two_rows(override="two_row")
    manual = {"x1": 0, "x2": 80, "y1": 60, "y2": 45, "source": "manual"}
    data["layout_separator"] = copy.deepcopy(manual)
    assert host._ensure_preview_layout_separator(data, image_w=80, image_h=120) == manual
    before = copy.deepcopy(data)
    host._preview_layout_separator_drag_state = {"preview_separator": {**manual, "y1": 15, "y2": 15}}
    assert [host._get_preview_row_for_bbox(c["bbox"], data) for c in data["characters"]] == [2, 2, 2, 2]
    assert data == before
    host._preview_layout_separator_drag_state = None
    assert host._get_preview_row_for_bbox(data["characters"][0]["bbox"], data) == 1


def test_overlapping_rows_split_between_character_centres_not_image_midpoint():
    host = CharacterAnnotationTab.__new__(CharacterAnnotationTab)
    data = two_rows()
    data.update(plate_image_width=320, plate_image_height=400)
    for index, rec in enumerate(data["characters"]):
        rec["bbox"][1:4:2] = [10, 90] if index < 2 else [70, 150]
    separator = host._ensure_preview_layout_separator(data)
    assert separator["y1"] == separator["y2"] == 80
    assert separator["source"] == "auto_row_centers"


@pytest.fixture(scope="module")
def root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


def graph_callbacks(host, campaign):
    """Execute production graph callbacks, sharing the same closure cache."""
    source = Path(dashboard.__file__).read_text(encoding="utf-8")
    renderer = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_render_step1_route_actions")
    wanted = {"_t06_interrupted_work_state", "_open_t06_actions_modal"}
    nodes = [copy.deepcopy(n) for n in renderer.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    factory = ast.parse("def factory():\n    _t06_interrupted_work_cache = None\n    _t06_interrupted_work_cache_key = None\n").body[0]
    factory.body += nodes + ast.parse("return locals()").body
    palette = get_theme_palette()
    ns = {**vars(dashboard), "tk": tk, "self": host, "CAMPAIGN": campaign,
          "current_step": 3, "current_target": "char", "current_iteration": 2,
          "active_project_name": "test", "palette": palette, "close_graph_dialogs": close_graph_dialogs,
          "card_bg": palette["panel"], "field_bg": palette["field"], "fg": palette["fg"],
          "muted": palette["muted"], "success": palette["success"], "warning": palette["warning"], "error": palette["error"],
          "_t06_exported_char_dataset_state": lambda: {"reason": "missing_char_dataset", "ok": False},
          "_is_t06_resume_action": lambda label: "tablic" in label and "znak" not in label,
          "_is_t06_z3_action": lambda label: "PZ3" in label or "dataset" in label,
          "campaign_ui_helpers": campaign_ui_helpers}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])), "t05_callbacks", "exec"), ns)
    return ns["factory"]()


def all_text(widget):
    parts = []
    for child in widget.winfo_children():
        if "text" in child.keys():
            parts.append(str(child["text"]))
        parts.append(all_text(child))
    return "\n".join(parts)


def test_t05_resume_save_return_refreshes_cached_interruption_and_recommends_step_two(root):
    state = {"t06_work_session": {"active": True, "state": "interrupted", "work_area": "z3",
                                "working_gate_id": "T05", "substep": 2, "interrupted_at": "2026-09-07T10:00:00"},
             "t06_contracts": {"pz2_char_boxes": {"fulfilled": True, "iteration": 2, "reason": "pz2_detection_ready"}}}
    def upsert(*, updates):
        for key, value in updates.items():
            state[key] = {**state.get(key, {}), **copy.deepcopy(value)}
    campaign = Mock()
    campaign.get_iteration_state.side_effect = lambda: copy.deepcopy(state)
    campaign.upsert_iteration_state.side_effect = upsert
    campaign.get_current_iteration_num.return_value = 2
    campaign.get_plate_approved_set_iteration_stats.return_value = {}
    campaign.get_manual_plate_stage_images_state.return_value = {}
    frame = tk.Frame(root)
    wizard = SimpleNamespace(frame=frame, _get_char_route_ready_source=lambda: {},
                             _get_step2_disk_approval_fallback=lambda **kw: {},
                             _rebuild_roadmap_ui=Mock(), _refresh_dashboard=Mock(), request_wizard_stage_focus=Mock())
    wizard.app = SimpleNamespace(tabs={}, style_dialog_window=lambda dialog, **kw: dialog.title(kw["title"]))
    callbacks = graph_callbacks(wizard, campaign)
    pending = callbacks["_t06_interrupted_work_state"]
    assert pending()["interrupted_kind"] == "z3"
    host = SimpleNamespace(app=SimpleNamespace(tabs={"campaign": wizard}, open_controlled_tab=Mock(),
                                              update_campaign_tab_access=Mock(), update_status=Mock()),
        _hide_campaign_detect_splash=Mock(), _cancel_preview_char_label_interaction=Mock(),
        _flush_scheduled_preview_metadata_save=Mock(), _sync_step3_access_from_preview_state=Mock(),
        _campaign_step3_pz2_current_contract_ready=lambda: True, preview_metadata={})
    with patch.object(flow, "CAMPAIGN", campaign):
        flow._mark_t06_z3_work_session(host, state="active", substep=2, force=True)
        assert pending()["t06_work_session"]["state"] == "active"
        flow.return_to_wizard_from_step3_pz2(host)
    assert state["t06_work_session"]["state"] == "ready_for_pz3"
    assert not state["t06_work_session"]["active"]
    assert not state["t06_work_session"].get("interrupted_at")
    assert pending() == {}  # Same graph callback, no redraw or modal close required.
    saved = copy.deepcopy(state)
    callbacks["_open_t06_actions_modal"]("", [("Popraw anotacje znaków w PZ2", Mock(), "info"),
                                                   ("Utwórz wariant datasetu", Mock(), "info")])
    dialog = next(w for w in frame.winfo_children() if isinstance(w, tk.Toplevel))
    text = all_text(dialog)
    assert "WYKONANE IT2" in text
    assert "PRZERWANE" not in text
    assert "Następny krok to eksport datasetu AZ w PZ3." in text
    dialog.destroy()
    assert state == saved
    assert pending() == {}
    frame.destroy()


def test_editor_entry_closes_graph_snapshots_and_keeps_other_dialogs(root):
    owner = tk.Frame(root)
    info = tk.Toplevel(owner)
    info._campaign_graph_dialog = True
    work = tk.Toplevel(owner)
    work._campaign_graph_dialog = True
    other = tk.Toplevel(owner)
    close_graph_dialogs(owner)
    assert not info.winfo_exists() and not work.winfo_exists()
    assert other.winfo_exists()
    owner.destroy()
