"""Regression for zero wraplength returned as Tcl strings/objects.

Real Tk renders labels and the shipped configurator. Costly model operations
in the existing harness remain explicit doubles; no real export is run.
"""
from pathlib import Path
import ast
import tkinter as tk
import tkinter.font as tkfont

import pytest

from auto_annotation_tool.gui.z4_export_configurator_view import ExportConfiguratorView
from z4_export_configurator_support import create, pump

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'auto_annotation_tool/gui/z4_model_export.py'
TITLES = ('Wejście [px]', 'Próg conf', 'Próg IoU')


@pytest.fixture(scope='module')
def display():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip('Tk display unavailable: ' + str(exc))
    root.geometry('120x80+0+0')
    pump(root)
    yield root
    root.destroy()


@pytest.fixture
def clean_display(display):
    old_objects = display.tk.wantobjects()
    old_scale = display.tk.call('tk', 'scaling')
    old_handler = display.report_callback_exception
    errors = []
    existing = set(display.winfo_children())
    display.report_callback_exception = lambda *error: errors.append(error)
    yield display
    for child in display.winfo_children():
        if child not in existing:
            child.destroy()
    pump(display)
    display.tk.wantobjects(old_objects)
    display.tk.call('tk', 'scaling', old_scale)
    display.report_callback_exception = old_handler
    assert not errors, errors


def view_with_label(root, text, wrap):
    window = tk.Toplevel(root)
    window.geometry('940x600+0+0')
    view = ExportConfiguratorView(window, 'TEST — synthetic data')
    page = view.add_role('MZ')
    label = tk.Label(page, text=text, wraplength=wrap, font=view.font_bold)
    label.grid(row=0, column=0, sticky='w')
    return view, label


def assert_single_line(root, label):
    font = tkfont.Font(root=root, font=label.cget('font'))
    border = 2 * (label.winfo_pixels(label.cget('borderwidth')) +
                  label.winfo_pixels(label.cget('highlightthickness')))
    horizontal = 2 * label.winfo_pixels(label.cget('padx')) + border
    vertical = 2 * label.winfo_pixels(label.cget('pady')) + border
    assert label.winfo_fpixels(label.cget('wraplength')) <= 0
    assert not label.bind('<Configure>'), 'Short label received shrink/wrap callback'
    assert label.winfo_width() >= font.measure(label.cget('text')) + horizontal
    assert label.winfo_height() <= font.metrics('linespace') + vertical + 2


@pytest.mark.parametrize('objects', [True, False], ids=['typed_tcl', 'string_tcl'])
@pytest.mark.parametrize('wrap', [0, '0', '0.0', '0p', '0c', -1])
def test_zero_or_negative_wrap_is_not_enabled_by_truthiness(clean_display, objects, wrap):
    root = clean_display
    root.tk.wantobjects(objects)
    view, label = view_with_label(root, 'Wejście [px]', wrap)
    for _ in range(3):
        view.style_widgets(view.root)
        pump(root, 5)
    assert_single_line(root, label)


@pytest.mark.parametrize('objects', [True, False], ids=['typed_tcl', 'string_tcl'])
@pytest.mark.parametrize('size,scale', [('1040x660', 1.333), ('1280x720', 1.667), ('1649x870', 2.0)])
def test_all_role_parameter_labels_stay_readable_without_changing_values(clean_display, tmp_path, objects, size, scale):
    root = clean_display
    root.tk.wantobjects(objects)
    root.tk.call('tk', 'scaling', scale)
    ctx, env = create(root, SOURCE, tmp_path / 'packages', roles=('MT', 'MZ', 'MP'))
    window = ctx['executor_modal']
    window.geometry(size + '+0+0')
    view = ctx['presentation']
    before = [(s['marker'], s['imgsz'].get(), s['conf'].get(), s['iou'].get()) for s in ctx['model_states']]
    for role in ('MT', 'MZ', 'MP'):
        view.notebook.select(view.pages[role].shell)
        pump(root, 8)
        labels = [w for w in view.walk(view.pages[role].body)
                  if isinstance(w, tk.Label) and w.cget('text') in TITLES]
        assert {w.cget('text') for w in labels} == set(TITLES)
        for label in labels:
            assert_single_line(root, label)
            assert label.winfo_rootx() >= window.winfo_rootx()
            assert label.winfo_rootx() + label.winfo_width() <= window.winfo_rootx() + window.winfo_width()
    after = [(s['marker'], s['imgsz'].get(), s['conf'].get(), s['iou'].get()) for s in ctx['model_states']]
    assert before == after
    assert not ctx['exec_preflight_state']['ready']
    assert not env['worker_state']['running']
    assert ctx['exec_progress_var'].get() == 0
    assert 'eksport nieuruchomiony' in view.phase.get()


@pytest.mark.parametrize('objects', [True, False], ids=['typed_tcl', 'string_tcl'])
@pytest.mark.parametrize('wrap', [120, '1i'])
def test_explicit_positive_wrap_is_preserved(clean_display, objects, wrap):
    root = clean_display
    root.tk.wantobjects(objects)
    view, label = view_with_label(root, 'Opis wykonania', wrap)
    label.grid_configure(sticky='ew')
    view.style_widgets(view.root)
    pump(root, 6)
    assert label.winfo_fpixels(label.cget('wraplength')) > 0
    assert label.bind('<Configure>')


@pytest.mark.parametrize('objects', [True, False], ids=['typed_tcl', 'string_tcl'])
def test_long_help_still_wraps_and_retains_complete_text(clean_display, objects):
    root = clean_display
    root.tk.wantobjects(objects)
    text = 'Ten opis jest dłuższy niż sześćdziesiąt znaków i musi pozostać w całości w interfejsie. ' * 4
    view, label = view_with_label(root, text, 0)
    label.grid_configure(sticky='ew')
    view.style_widgets(view.root)
    pump(root, 6)
    assert label.cget('text') == text
    assert label.winfo_fpixels(label.cget('wraplength')) > 0
    assert label.bind('<Configure>')


def test_wrap_decision_uses_numeric_tk_distance_and_source_is_utf8():
    path = ROOT / 'auto_annotation_tool/gui/z4_export_configurator_view.py'
    source = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'style_widgets')
    comparisons = [n for n in ast.walk(method) if isinstance(n, ast.Compare)]
    assert any(isinstance(n.left, ast.Call) and isinstance(n.left.func, ast.Attribute)
               and n.left.func.attr == 'winfo_fpixels' for n in comparisons)
    assert 'if widget.cget("wraplength") or' not in source
