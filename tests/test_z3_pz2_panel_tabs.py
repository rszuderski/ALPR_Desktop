import tkinter as tk

from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette
from auto_annotation_tool.gui.campaign_sidebar import ProjectSidebarToggle


def test_vertical_sidebar_toggle_is_tall_thin_and_rotated():
    root = tk.Tk()
    root.withdraw()
    try:
        tab = ProjectSidebarToggle(
            root,
            get_theme_palette(),
            lambda: None,
            collapsed_text="Tablice",
            expanded_text="Tablice",
            orientation="vertical",
            side="left",
        )
        root.update_idletasks()
        assert tab._height > tab._width
        assert str(tab.itemcget("label", "angle")) in {"90.0", "90"}
        before = tuple(float(v) for v in tab.coords("direction"))
        tab.set_collapsed(True)
        after = tuple(float(v) for v in tab.coords("direction"))
        assert before != after
    finally:
        root.destroy()


def test_right_vertical_tab_uses_opposite_rotation():
    root = tk.Tk()
    root.withdraw()
    try:
        tab = ProjectSidebarToggle(
            root,
            get_theme_palette(),
            lambda: None,
            collapsed_text="Ustawienia",
            expanded_text="Ustawienia",
            orientation="vertical",
            side="right",
        )
        assert str(tab.itemcget("label", "angle")) in {
            "270.0", "270", "-90.0", "-90"
        }
    finally:
        root.destroy()


def test_status_panel_uses_dynamic_wrap():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "auto_annotation_tool" / "gui" / "z3_detection_tab_ui.py"
    ).read_text(encoding="utf-8-sig")
    assert "def _sync_preview_status_panel_wrap(event=None):" in source
    assert "width - 34" in source
    assert "preview_repair_progress_title_lbl" in source


def test_workspace_drawers_use_vertical_side_tabs():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "auto_annotation_tool" / "gui" / "z3_workspace_drawers.py"
    ).read_text(encoding="utf-8-sig")
    assert 'orientation="vertical"' in source
    assert "button.set_collapsed(not panel_visible_target)" in source


def test_tool_drawer_uses_vertical_edge_tab():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "auto_annotation_tool" / "gui" / "z2_drawer_slide.py"
    ).read_text(encoding="utf-8-sig")
    assert 'collapsed_text="Narzędzia"' in source
    assert 'orientation="vertical"' in source
    assert 'side="right"' in source
