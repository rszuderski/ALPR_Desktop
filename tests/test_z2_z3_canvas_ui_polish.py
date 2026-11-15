import inspect

from auto_annotation_tool.gui import z2_layout_ui_runtime
from auto_annotation_tool.gui import z2_main_widgets
from auto_annotation_tool.gui import z2_panel_workflow
from auto_annotation_tool.gui import z2_right_panel_widgets
from auto_annotation_tool.gui import z3_detection_tab_ui
from auto_annotation_tool.gui import z3_theme_ui


def test_z2_canvas_dark():
    src = inspect.getsource(z2_main_widgets.create_annotation_widgets)
    assert 'bg="#1e1e1e"' in src


def test_z2_theme_keeps_canvas_dark():
    src = inspect.getsource(z2_panel_workflow.apply_theme)
    assert 'background="#1e1e1e"' in src


def test_pz2_canvas_and_host_dark():
    src = inspect.getsource(z3_detection_tab_ui)
    assert src.count('bg="#1e1e1e"') >= 2


def test_pz2_theme_keeps_dark_background():
    src = inspect.getsource(z3_theme_ui.apply_character_annotation_theme)
    assert 'background="#1e1e1e"' in src
    assert 'preview_canvas_host.configure(bg="#1e1e1e")' in src


def test_z2_right_panel_limits_compact():
    src = inspect.getsource(z2_layout_ui_runtime._get_main_pane_width_limits)
    assert 'max(260, min(280,' in src


def test_z2_right_panel_default_compact():
    src = inspect.getsource(z2_panel_workflow)
    assert 'int(total_width * 0.16)' in src
    assert '), 280))' in src


def test_z2_right_panel_copy_wrap_compact():
    src = inspect.getsource(z2_right_panel_widgets)
    assert 'wraplength=320' not in src
    assert src.count('wraplength=260') >= 3
