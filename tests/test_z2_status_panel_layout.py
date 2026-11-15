"""Native geometry checks for dynamic campaign status copy in a narrow panel."""
import tkinter as tk
from tkinter import ttk, font as tkfont
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_annotation_tool.gui import z2_layout_ui_runtime as layout
from auto_annotation_tool.gui import z2_panel_workflow as workflow
from auto_annotation_tool.gui import z2_status_ui_runtime as status
from auto_annotation_tool.gui.z2_right_panel_widgets import build_annotation_right_panel


@pytest.fixture
def panel():
    root = tk.Tk()
    root.geometry("280x460+40+40")
    errors = []
    root.report_callback_exception = lambda *error: errors.append(error)
    owner = SimpleNamespace(frame=root, app=SimpleNamespace(palette={}))
    for name in ("approve_context_var", "approve_hint_title_var", "approve_gate_hint_var", "approve_breakdown_title_var"):
        setattr(owner, name, tk.StringVar(root, ""))
    for module, names in (
        (layout, ("_sync_approve_hint_wraplength", "_sync_right_panel_canvas_width", "_sync_right_panel_scrollregion")),
        (status, ("_set_inline_label_state", "_set_approve_context_box_state", "_set_approve_hint_box_state")),
    ):
        for name in names:
            setattr(owner, name, getattr(module, name).__get__(owner))
    for name in ("_approve_annotation_stage", "_return_to_campaign_wizard", "_refresh_approval_breakdown_canvas",
                 "_schedule_main_pane_layout_refresh"):
        setattr(owner, name, Mock())
    build_annotation_right_panel(owner, root, panel_bg="#252526", panel_border="#3c3c3c", nav_button_width=18)
    owner.approve_btn_row.pack(fill="x")
    owner.approve_hint_box.pack(fill="x", before=owner.approve_btn_frame)
    owner.approve_hint_title_lbl.pack(fill="x")
    owner.approve_hint_table_frame.pack(fill="x")
    owner.return_to_campaign_right_btn.pack(fill="x")
    root.update()
    yield root, owner
    root.destroy()
    assert not errors


def assert_table_fits(owner):
    table = owner.approve_hint_table_frame
    for widgets in table._compact_table_cache["layout_rows"]:
        for widget in widgets:
            assert widget.winfo_reqwidth() <= widget.winfo_width()
            assert widget.winfo_reqheight() <= widget.winfo_height()
            assert widget.winfo_rootx() >= table.winfo_rootx()
            assert widget.winfo_rootx() + widget.winfo_width() <= table.winfo_rootx() + table.winfo_width()


def test_rebuilt_and_reused_campaign_tables_wrap_both_columns(panel):
    root, owner = panel
    table = owner.approve_hint_table_frame
    workflow._render_compact_info_table(owner, table, [("Status", "OK", "success")], show_header=False)
    root.update()
    rows = [("Po zamknięciu bramki T03", "1128 obrazów [OK] / 1284 tablic", "success"),
            ("GT numerów tablic", "GT opcjonalny: 1128 / 1284 tablic", "info")]
    workflow._render_compact_info_table(owner, table, rows, show_header=False)
    root.update()
    for width in (280, 380, 260, 300):
        root.geometry(f"{width}x460")
        root.update()
        assert_table_fits(owner)
    widgets_before = list(table._compact_table_cache["rows"])
    rows[0] = (rows[0][0], "Bardzo długa dynamiczna wartość po zmianie liczników w kampanii", "warning")
    workflow._render_compact_info_table(owner, table, rows, reuse_existing=True, show_header=False)
    root.update()
    assert_table_fits(owner)
    assert table._compact_table_cache["rows"] == widgets_before


def test_long_copy_and_campaign_cta_remain_accessible(panel):
    root, owner = panel
    owner.approve_context_box.pack(fill="x", before=owner.approve_hint_box)
    owner.approve_context_var.set("Zatwierdzone obrazy i tablice zostaną przekazane do kolejnej bramki kampanii.")
    owner.approve_hint_title_var.set("Podsumowanie zatwierdzonych tablic bieżącej iteracji")
    workflow._render_compact_info_table(owner, owner.approve_hint_table_frame,
        [(f"Długi opis pozycji {i}", "1128 obrazów [OK] / 1284 tablic", "success") for i in range(12)])
    owner.return_to_campaign_right_btn.configure(text="Przekaż [OK] do puli YOLO tablic i wróć do bramki T04", width=48)
    root.update()
    assert_table_fits(owner)
    for label in (owner.approve_context_lbl, owner.approve_hint_title_lbl):
        assert label.winfo_reqwidth() <= label.winfo_width()
        assert label.winfo_reqheight() <= label.winfo_height()
    canvas = owner.right_settings_canvas
    assert canvas.yview()[1] < 1
    canvas.yview_moveto(1)
    root.update()
    button = owner.return_to_campaign_right_btn
    assert button.winfo_rooty() >= canvas.winfo_rooty()
    assert button.winfo_rooty() + button.winfo_height() <= canvas.winfo_rooty() + canvas.winfo_height()
    assert "\n" in button.cget("text")
    font = tkfont.Font(root=root, font=ttk.Style().lookup("WorkflowCard.TButton", "font") or "TkDefaultFont")
    assert all(font.measure(line) <= button.winfo_width() - 28 for line in button.cget("text").splitlines())
    owner.return_to_campaign_right_btn.configure(text="Wróć do grafu")
    root.update()
    assert button.cget("text") == "Wróć do grafu"
