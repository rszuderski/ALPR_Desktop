from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tkinter as tk

import pytest

from auto_annotation_tool.gui import z2_miniflow_runtime as flow
from auto_annotation_tool.gui import z2_context_runtime as context
from auto_annotation_tool.gui import z2_run_io_runtime as run_io
from auto_annotation_tool.gui import z2_layout_ui_runtime as layout
from auto_annotation_tool.gui import z2_free_mode_flow as free_flow
from auto_annotation_tool.gui import z2_export_workflow as exports
from auto_annotation_tool.gui import z2_run_lifecycle as lifecycle


class Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def run_owner(free=True):
    owner = Mock()
    owner._is_free_mode_session_context.return_value = free
    owner.current_annotation_run_dir = None
    owner.last_staging_run_dir = None
    owner.current_input_dir = Path("images")
    owner.plate_dataset_run_var = Value()
    owner._should_skip_annotation_run_lookup_for_current_input.return_value = False
    owner._resolve_safe_annotation_run_dir.side_effect = lambda value, **kw: Path(value) if value else None
    return owner


@pytest.mark.parametrize("screen,route,entry", [
    ("route_choice", "", "new"), ("workflow", "manual", "new"),
    ("workflow", "manual", "continue"), ("workflow", "manual", "import"),
    ("workflow", "auto", "new"), ("manual_review", "manual", "continue"),
    ("auto_summary", "auto", "new"), ("export", "manual", "new"),
])
def test_free_presentation_never_discovers_or_selects_an_unrelated_run(screen, route, entry):
    owner = run_owner()
    owner._get_free_mode_screen.return_value = screen
    owner._get_workflow_route.return_value = route
    owner._get_manual_entry_mode.return_value = entry
    assert flow._get_preferred_annotation_run_dir(owner, require_xml=True) is None
    owner._find_latest_annotation_run_for_input.assert_not_called()
    owner._find_latest_annotation_run_dir.assert_not_called()


@pytest.mark.parametrize("free", [True, False])
@pytest.mark.parametrize("source", ["current_annotation_run_dir", "last_staging_run_dir", "plate_dataset_run_var"])
def test_explicit_run_remains_available_in_both_modes(free, source):
    owner = run_owner(free)
    if source.endswith("var"):
        owner.plate_dataset_run_var.set("selected-run")
    else:
        setattr(owner, source, Path("selected-run"))
    assert flow._get_preferred_annotation_run_dir(owner, require_xml=True) == Path("selected-run")
    owner._find_latest_annotation_run_for_input.assert_not_called()


@pytest.mark.parametrize("gate", ["T02", "T03", "T04", "T05", "T06"])
@pytest.mark.parametrize("iteration", [1, 2])
def test_campaign_lookup_is_preserved(gate, iteration):
    owner = run_owner(False)
    owner._campaign_graph_entry_context = {"graph_gate_id": gate, "iteration": iteration}
    owner._find_latest_annotation_run_for_input.return_value = Path("campaign-run")
    assert flow._get_preferred_annotation_run_dir(owner, require_xml=True) == Path("campaign-run")
    owner._find_latest_annotation_run_for_input.assert_called_once()


def test_free_export_panel_does_not_select_latest_run_when_empty():
    owner = run_owner()
    owner.input_dir_var = Value()
    exports._refresh_plate_dataset_export_sources(owner)
    owner._find_latest_annotation_run_dir.assert_not_called()
    owner._load_plate_dataset_context_from_run.assert_not_called()


def test_free_actions_do_not_read_a_project_left_in_campaign_manager():
    from auto_annotation_tool.campaign_manager import CAMPAIGN
    owner = run_owner()
    owner.plate_custom_var = Value()
    owner.input_dir_var = Value()
    owner._get_preferred_annotation_run_dir.return_value = None
    owner._get_workflow_route.return_value = "auto"
    owner._get_auto_vehicle_choice.return_value = "skip"
    owner.is_processing = False
    with patch.object(CAMPAIGN, "get_active_project_name", return_value="previous-project"), \
         patch.object(CAMPAIGN, "get_global_model") as get_model:
        ctx = layout._build_z2_action_context(owner)
    get_model.assert_not_called()
    assert ctx.mode == "free"
    assert ctx.campaign_step == 0
    assert not ctx.has_plate_model


@pytest.mark.parametrize("route,manual_xml,from_auto,expected", [
    ("manual", True, False, "manual"), ("manual", False, True, "manual"),
    ("manual", False, False, "manual"), ("auto", False, False, "auto"),
    ("", True, False, "manual"), ("", False, False, "auto"),
])
def test_resume_preserves_manual_review_and_auto_origin(route, manual_xml, from_auto, expected):
    owner = run_owner()
    owner.is_processing = False
    owner.current_annotations = []
    owner.last_staging_run_dir = Path("saved-run")
    owner.workflow_route_var = Value(route)
    owner.free_mode_screen_var = Value()
    owner._get_workflow_route.side_effect = owner.workflow_route_var.get
    owner._manual_review_from_auto = from_auto
    owner._restore_preview_from_session_run.return_value = True
    owner._annotation_run_manifest_is_manual_template.return_value = manual_xml
    assert context.ensure_free_mode_session_preview_ready(owner)
    assert owner.workflow_route_var.get() == expected
    assert owner.free_mode_screen_var.get() == ("manual_review" if expected == "manual" else "auto_summary")
    if expected == "manual":
        assert owner._manual_review_from_auto == (from_auto and not manual_xml)


def test_free_resume_does_not_touch_campaign_context():
    owner = run_owner(False)
    assert not context.ensure_free_mode_session_preview_ready(owner)
    owner._restore_preview_from_session_run.assert_not_called()


def test_resume_does_not_replace_explicit_selection_with_newer_history():
    owner = run_owner()
    owner.current_annotation_run_dir = Path("selected-older-run")
    owner._get_preferred_annotation_run_dir.side_effect = lambda **kw: flow._get_preferred_annotation_run_dir(owner, **kw)
    owner._find_latest_annotation_run_dir.return_value = Path("newer-unrelated-run")
    assert lifecycle._resolve_best_free_mode_restore_run(owner) == Path("selected-older-run")
    owner._find_latest_annotation_run_dir.assert_not_called()
    owner._score_annotation_run_restore_candidate.assert_not_called()


def test_empty_free_session_does_not_restore_history_implicitly():
    owner = run_owner()
    owner._get_preferred_annotation_run_dir.side_effect = lambda **kw: flow._get_preferred_annotation_run_dir(owner, **kw)
    assert lifecycle._resolve_best_free_mode_restore_run(owner) is None
    owner._find_latest_annotation_run_dir.assert_not_called()


@pytest.mark.parametrize("free,entry,should_load", [(True, "new", False), (True, "import", False),
                                                    (True, "continue", True), (False, "new", True)])
def test_history_is_loaded_when_needed_without_changing_campaign(free, entry, should_load):
    owner = run_owner(free)
    owner._get_manual_entry_mode.return_value = entry
    owner._manual_review_active = False
    owner._manual_review_from_auto = False
    owner.manual_history_run_var = Value()
    with patch.object(free_flow, "get_manual_review_history_display_entries", return_value=[]) as load:
        free_flow.refresh_manual_review_history_ui(owner)
    assert bool(load.call_count) == should_load


@pytest.mark.parametrize("route,entry,load_history", [("", "new", False), ("auto", "new", False),
                                                     ("manual", "new", False), ("manual", "import", False),
                                                     ("manual", "continue", True)])
def test_copy_context_does_not_scan_hidden_history(route, entry, load_history):
    owner = run_owner()
    owner._get_workflow_route.return_value = route
    owner._get_z2_thematic_route.return_value = route
    owner._get_manual_entry_mode.return_value = entry
    owner._get_workflow_progress_display.return_value = (1, 3)
    owner._campaign_pending_batch_summary = {}
    ctx = flow._build_z2_left_panel_copy_context(owner)
    assert bool(owner._get_manual_review_history_display_entries.call_count) == load_history
    assert ctx.has_manual_history == load_history


def xml_owner():
    owner = SimpleNamespace()
    owner._resolve_safe_annotation_run_dir = lambda path, **kwargs: Path(path) if path else None
    return owner


def make_xml(root, index, name="A.jpg"):
    run = root / f"run_{index}"
    run.mkdir(exist_ok=True)
    (run / "annotations.xml").write_text(f'<annotations><image name="{name}" /></annotations>', encoding="utf-8")
    return run


def test_141_xmls_are_parsed_once_not_on_every_lookup(tmp_path):
    owner = xml_owner()
    runs = [make_xml(tmp_path, i, f"image_{i}.jpg") for i in range(141)]
    with patch.object(run_io.ET, "iterparse", wraps=run_io.ET.iterparse) as parse:
        for _ in range(3):
            for i, run in enumerate(runs):
                assert run_io._get_cached_run_xml_image_names(owner, run) == {f"image_{i}.jpg"}
    assert parse.call_count == 141


def test_xml_cache_invalidates_changed_file_and_protects_values(tmp_path):
    owner = xml_owner()
    run = make_xml(tmp_path, 0)
    first = run_io._get_cached_run_xml_image_names(owner, run)
    first.clear()
    assert run_io._get_cached_run_xml_image_names(owner, run) == {"a.jpg"}
    make_xml(tmp_path, 0, "changed.jpg")
    assert run_io._get_cached_run_xml_image_names(owner, run) == {"changed.jpg"}
    assert len(owner._z2_run_xml_image_name_set_cache) == 1


def test_xml_cache_is_bounded_and_keeps_recent_hits(tmp_path):
    owner = xml_owner()
    runs = [make_xml(tmp_path, i) for i in range(257)]
    for run in runs[:256]:
        run_io._get_cached_run_xml_image_names(owner, run)
    run_io._get_cached_run_xml_image_names(owner, runs[0])
    run_io._get_cached_run_xml_image_names(owner, runs[-1])
    cache = owner._z2_run_xml_image_name_set_cache
    assert len(cache) == 256
    paths = {key[1] for key in cache}
    assert str((runs[0] / "annotations.xml").resolve()).lower() in paths
    assert str((runs[1] / "annotations.xml").resolve()).lower() not in paths


def test_xml_cache_also_limits_retained_names(tmp_path):
    owner = xml_owner()
    owner._z2_run_xml_image_name_set_cache = {
        ("xml", "old1", 1, 1): frozenset(str(i) for i in range(100_000)),
        ("xml", "old2", 1, 1): frozenset(str(i) for i in range(100_000)),
    }
    run_io._get_cached_run_xml_image_names(owner, make_xml(tmp_path, 0))
    assert sum(map(len, owner._z2_run_xml_image_name_set_cache.values())) <= 200_000


def test_empty_source_does_not_read_xml(tmp_path):
    with patch.object(run_io, "_get_cached_input_dir_image_names", return_value=set()), \
         patch.object(run_io, "_get_cached_run_xml_image_names") as xml:
        assert not run_io._annotation_run_matches_expected_image_names(xml_owner(), tmp_path, tmp_path)
        xml.assert_not_called()


def test_xml_parse_failure_is_not_cached(tmp_path):
    owner = xml_owner()
    run = make_xml(tmp_path, 0)
    with patch.object(run_io.ET, "iterparse", side_effect=OSError("temporarily unreadable")):
        assert run_io._get_cached_run_xml_image_names(owner, run) == set()
    assert run_io._get_cached_run_xml_image_names(owner, run) == {"a.jpg"}


@pytest.fixture
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


def test_unchanged_canvas_width_does_not_enqueue_more_layout(tk_root):
    canvas = tk.Canvas(tk_root, width=400)
    content = tk.Frame(canvas)
    window = canvas.create_window(0, 0, window=content, width=400)
    owner = SimpleNamespace(left_settings_canvas=canvas, _left_settings_window_id=window,
                            _schedule_main_pane_layout_refresh=Mock())
    event = SimpleNamespace(width=400)
    for _ in range(10):
        layout._sync_left_panel_canvas_width(owner, event)
    owner._schedule_main_pane_layout_refresh.assert_not_called()
    layout._sync_left_panel_canvas_width(owner, SimpleNamespace(width=420))
    owner._schedule_main_pane_layout_refresh.assert_called_once()


def test_bound_labels_have_one_wrap_authority(tk_root):
    host = tk.Frame(tk_root)
    label = tk.Label(host, text="test", wraplength=300)
    label._wrap_container = host
    owner = SimpleNamespace(workflow_entry_shell_inner=host, route_summary_lbl=label,
                            _refresh_bound_label_wraplength=Mock())
    layout._sync_workflow_copy_wraplength(owner, SimpleNamespace(width=600))
    owner._refresh_bound_label_wraplength.assert_called_once_with(label)
    assert label.cget("wraplength") == 300


def test_splash_drawing_does_not_reenter_event_loop(tk_root):
    canvas = tk.Canvas(tk_root, width=500, height=16)
    owner = SimpleNamespace(_campaign_step2_splash_progress=canvas)
    with patch.object(canvas, "update_idletasks", side_effect=AssertionError("reentrant redraw")) as update:
        context._draw_campaign_step2_splash_bar(owner, progress_value=50,
                                               accent="green", trough_color="gray", border_color="black")
    update.assert_not_called()
    assert len(canvas.find_all()) == 2
    assert canvas.coords(canvas.find_all()[0])[2] >= 498


def test_scroll_waits_for_layout_and_keeps_only_latest_target(tk_root):
    canvas = Mock()
    canvas.bbox.return_value = (0, 0, 100, 1000)
    content = object()
    def target(y):
        return SimpleNamespace(winfo_y=lambda: y, winfo_parent=lambda: "content", nametowidget=lambda name: content)
    owner = SimpleNamespace(frame=tk_root, left_settings_canvas=canvas, left_settings_content=content)
    layout._scroll_left_panel_to_widget(owner, target(20))
    first_job = owner._left_panel_scroll_after_id
    layout._scroll_left_panel_to_widget(owner, target(70))
    assert first_job not in tk_root.tk.call("after", "info")
    canvas.yview_moveto.assert_not_called()
    tk_root.update_idletasks()
    canvas.yview_moveto.assert_called_once_with(0.07)
    canvas.update_idletasks.assert_not_called()
