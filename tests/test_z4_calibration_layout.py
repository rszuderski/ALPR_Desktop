"""Native Tk geometry on synthetic inputs; no production calibration/export."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tkinter as tk
from tkinter import ttk

import pytest

from auto_annotation_tool.gui.app import AutoAnnotationApp
from auto_annotation_tool.gui.app_theme_definitions import THEME_DEFINITIONS
from auto_annotation_tool.gui import z4_int8_calibration as ui, z4_model_export as export
from auto_annotation_tool.gui.z4_calibration_context import context_from_candidate
from test_z4_int8_calibration import root, calibration_tk_runtime, pump
from test_z4_calibration_context import candidate_fixture, bind_inputs


@pytest.fixture
def themed(root):
    app = AutoAnnotationApp.__new__(AutoAnnotationApp)
    app.root, app.style, app.themes = root, ttk.Style(root), THEME_DEFINITIONS
    original_scaling = float(root.tk.call("tk", "scaling"))
    errors = []
    root.report_callback_exception = lambda *exc: errors.append(exc)
    root.deiconify()
    yield app
    for child in list(root.winfo_children()):
        child.destroy()
    root.tk.call("tk", "scaling", original_scaling)
    assert not errors


def visible_in(widget, container):
    x = widget.winfo_rootx() - container.winfo_rootx()
    y = widget.winfo_rooty() - container.winfo_rooty()
    assert widget.winfo_ismapped()
    assert 0 <= x and x + widget.winfo_width() <= container.winfo_width()
    assert 0 <= y and y + widget.winfo_height() <= container.winfo_height()


@pytest.mark.parametrize("geometry,scaling,theme", [
    ("800x600", 1.3333, "light_visual_cs"),
    ("1080x800", 1.3333, "light_visual_cs"),
    ("800x600", 1.6667, "light_visual_cs"),
    ("1080x800", 1.6667, "dark_visual_cs"),
])
def test_pinned_layout_keeps_guidance_footer_and_focused_fields_visible(themed, tmp_path, geometry, scaling, theme):
    root = themed.root
    root.tk.call("tk", "scaling", scaling)
    themed._setup_style(theme)
    f, c = candidate_fixture(tmp_path, "character")
    d = ui.Int8CalibrationDialog(root, context=context_from_candidate(c))
    bind_inputs(d, f)
    d.geometry(geometry + "+0+0")
    root.update()
    assert not d.identity_frame.winfo_manager()
    assert not d.report_frame.winfo_manager() and not d.details.frame.winfo_manager()
    assert d._next_field == "final_inventory"
    for button in (d.next_button, d.prepare_button, d.save_button, d.close_button):
        visible_in(button, d)
        assert button.winfo_width() >= button.winfo_reqwidth()
    visible_in(d.next_label, d)
    assert d.next_label.winfo_height() >= d.next_label.winfo_reqheight()
    assert len({w.winfo_rooty() for w in (d.prepare_button, d.save_button, d.close_button)}) == 1
    assert d.entries["limit"].winfo_width() < d.canvas.winfo_width() / 4
    d.mode.set(ui.exploratory.LABEL)
    d.check_sources()
    pump(root, lambda: not d._busy)
    root.update()
    assert not d.field_widgets["final_inventory"][1].winfo_manager()
    if geometry == "1080x800" and scaling == 1.3333:
        visible_in(d.report_toggle, d.canvas)
    d.entries["limit"].focus_force()
    root.update()
    visible_in(d.entries["limit"], d.canvas)
    assert d.entries["limit"].tk_focusNext() == d.suggest_button
    d.entries["limit"].event_generate("<Tab>")
    root.update()
    assert d.focus_get() == d.suggest_button
    visible_in(d.suggest_button, d.canvas)
    d.entries["checkpoint_sha256"].focus_force()
    root.update()
    visible_in(d.entries["checkpoint_sha256"], d.canvas)
    assert not d.approval.get() and not hasattr(d, "research_check")


def test_disclosures_keep_identity_and_manifest_available_without_empty_report(themed, tmp_path):
    root = themed.root
    themed._setup_style("light_visual_cs")
    f, c = candidate_fixture(tmp_path)
    d = ui.Int8CalibrationDialog(root, context=context_from_candidate(c), mode="exploratory")
    bind_inputs(d, f)
    d.identity_toggle.invoke()
    d.report_toggle.invoke()
    root.update()
    assert d.identity_frame.winfo_manager() == "grid"
    assert d._verified["sha256"] in d.card_text.get()
    assert d.report_frame.winfo_manager() == "grid" and not d.details.frame.winfo_manager()
    d.values["limit"].set("2")
    assert d.values["approved_by"].get()=="" and not d.approval.get()
    d.prepare()
    pump(root, lambda: not d._busy)
    assert d.result["status"] == "EXPLORATORY_ONLY"
    assert "EXPLORATORY_ONLY" in d.report_summary.cget("text")
    assert d.details.frame.winfo_manager() == "grid"
    assert d.result["input_fingerprint"] in d.details.get("1.0", "end")
    d.values["limit"].set("1")
    assert d.result is None and not d.details.frame.winfo_manager()
    assert not d.approval.get() and not hasattr(d, "research_check")


@pytest.mark.parametrize("width,scaling", [(1100, 1.3333), (1920, 1.3333), (1100, 1.6667), (1920, 1.6667)])
def test_export_center_selection_actions_share_one_visible_row(themed, tmp_path, width, scaling):
    root = themed.root
    root.tk.call("tk", "scaling", scaling)
    themed._setup_style("light_visual_cs")
    _, c = candidate_fixture(tmp_path)
    host = SimpleNamespace(app=SimpleNamespace(root=root, palette=themed.palette, show_assistant_message=Mock()),
                           frame=root, _build_mobile_model_export_path=lambda *_: tmp_path / "NEVER_EXPORTED.alprmodel")
    with patch.object(export, "_collect_mobile_export_candidates", return_value=[c]), \
         patch.object(export.MobileModelExporter, "export", side_effect=AssertionError("export forbidden")):
        center = export._open_mobile_model_export_center(host)
        center.geometry(f"{width}x800+0+0")
        root.update()
        labels = ["Eksportuj zaznaczone", "Zamknij"]
        forbidden = {"Sprawdź precyzję TFLite", "Kalibracja INT8 — wskazany model",
                     "Wybierz trening ręcznie", "Raporty z telefonu"}
        assert not any(isinstance(w, ttk.Button) and w.cget("text") in forbidden
                       for w in ui.Int8CalibrationDialog._walk(center))
        buttons = [next(w for w in ui.Int8CalibrationDialog._walk(center)
                        if isinstance(w, ttk.Button) and w.cget("text") == label) for label in labels]
        assert len({w.winfo_rooty() for w in buttons}) == 1
        assert len({w.winfo_height() for w in buttons}) == 1
        for button in buttons:
            visible_in(button, center)
            assert button.winfo_width() >= button.winfo_reqwidth()
        for left, right in zip(buttons, buttons[1:]):
            assert left.winfo_rootx() + left.winfo_width() + 7 <= right.winfo_rootx()
