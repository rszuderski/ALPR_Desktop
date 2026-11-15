"""Z4MOBREP029: right scientific-dashboard scroll follows pointer, not focus.

No session reads/writes. Tests cover Tk bindtag precedence over Treeview class
wheel bindings as well as pure routing for source/no-selection conditions.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk

import pytest

SRC = Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui/z4_mobile_report_browser.py'
TREE = ast.parse(SRC.read_text(encoding='utf-8'))


def _extract_functions():
    helper = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_report_wheel_scroll_units')
    klass = next(n for n in TREE.body if isinstance(n, ast.ClassDef)
                 and n.name == 'MobileReportBrowser')
    names = ('_pointer_inside_widget', '_route_report_pointer_wheel', '_install_report_pointer_wheel')
    methods = [next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name)
               for name in names]
    # Staticmethod decoration belongs to the class, not this standalone AST exec.
    methods[0].decorator_list = []
    mod = ast.Module(body=[helper, *methods], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {'tk': tk}
    exec(compile(mod, str(SRC), 'exec'), ns)
    return ns

NS = _extract_functions()
wheel_units = NS['_report_wheel_scroll_units']


class FakeWidget:
    def __init__(self, x=0, y=0, w=100, h=100, *, visible=True):
        self.rect = (x, y, w, h)
        self.visible = visible
        self.scrolled = []
    def winfo_ismapped(self): return self.visible
    def winfo_rootx(self): return self.rect[0]
    def winfo_rooty(self): return self.rect[1]
    def winfo_width(self): return self.rect[2]
    def winfo_height(self): return self.rect[3]
    def yview_scroll(self, n, units): self.scrolled.append((n, units))


class FakeWindow:
    def __init__(self):
        self.pointer = (500, 100)
        self.other_window = False
    def winfo_pointerxy(self): return self.pointer
    def winfo_containing(self, x, y):
        return SimpleNamespace(winfo_toplevel=lambda: object() if self.other_window else self)


class Harness:
    _pointer_inside_widget = staticmethod(NS['_pointer_inside_widget'])
    _route_report_pointer_wheel = NS['_route_report_pointer_wheel']
    _install_report_pointer_wheel = NS['_install_report_pointer_wheel']
    def __init__(self):
        self.window = FakeWindow()
        dashboard = FakeWidget(400, 0, 500, 400)
        dashboard.canvas = FakeWidget(410, 10, 470, 380)
        self.science_panel = dashboard
        self.report_table = SimpleNamespace(shell=FakeWidget(0, 0, 300, 400), tree=FakeWidget())


def _event(**kwargs):
    return SimpleNamespace(delta=kwargs.get('delta', -120), num=kwargs.get('num', 0),
                           state=kwargs.get('state', 0), widget=kwargs.get('widget', object()))


@pytest.mark.parametrize('delta,expected', [(120,-3),(-120,3),(240,-6),(-240,6),(1,-1),(-1,1),(0,0)])
def test_windows_and_touchpad_directions(delta, expected):
    assert wheel_units(_event(delta=delta)) == expected


def test_linux_wheel_and_ctrl_wheel():
    assert wheel_units(_event(delta=0,num=4)) == -3
    assert wheel_units(_event(delta=0,num=5)) == 3
    assert wheel_units(_event(delta=-120,state=0x0004)) == 0


def test_right_panel_receives_wheel_even_if_left_tree_keeps_focus():
    h=Harness()
    ev=_event(widget=h.report_table.tree)
    assert h._route_report_pointer_wheel(ev) == 'break'
    assert h.science_panel.canvas.scrolled == [(3,'units')]
    assert h.report_table.tree.scrolled == []


def test_left_list_receives_wheel_when_pointer_is_over_list():
    h=Harness()
    h.window.pointer=(90,100)
    assert h._route_report_pointer_wheel(_event(widget=h.science_panel.canvas)) == 'break'
    assert h.report_table.tree.scrolled == [(3,'units')]
    assert h.science_panel.canvas.scrolled == []


def test_pointer_elsewhere_does_not_steal_wheel():
    h=Harness()
    h.window.pointer=(350,100)
    assert h._route_report_pointer_wheel(_event()) is None
    assert h.report_table.tree.scrolled == [] and h.science_panel.canvas.scrolled == []


def test_inactive_tab_and_foreground_dialog_do_not_scroll_report():
    h=Harness()
    h.science_panel.visible=False
    assert h._route_report_pointer_wheel(_event()) is None
    h.science_panel.visible=True
    h.window.other_window=True
    assert h._route_report_pointer_wheel(_event()) is None
    assert not h.science_panel.canvas.scrolled


def test_does_not_scroll_for_ctrl_wheel_or_missing_event_delta():
    h=Harness()
    assert h._route_report_pointer_wheel(_event(state=0x0004)) is None
    assert h._route_report_pointer_wheel(_event(delta=0)) is None
    assert not h.science_panel.canvas.scrolled


def test_integration_tk_bindtags_intercept_focused_treeview():
    """Actual Tk class wheel binding must not pre-empt right panel routing."""
    try:
        root=tk.Tk()
    except tk.TclError:
        pytest.skip('Tk requires a display')
    try:
        root.geometry('900x540+10+10')
        win=tk.Toplevel(root)
        win.geometry('840x420+20+20')
        left_frame=tk.Frame(win, width=220)
        left_frame.pack(side='left',fill='y')
        left=ttk.Treeview(left_frame,height=10)
        left.pack(fill='both',expand=True)
        for i in range(75):
            left.insert('','end',values=(str(i),),text=str(i))
        right=tk.Frame(win,width=550)
        right.pack(side='left',fill='both',expand=True)
        canvas=tk.Canvas(right,width=480,height=290)
        canvas.pack(fill='both',expand=True)
        content=tk.Frame(canvas)
        canvas.create_window((0,0),window=content,anchor='nw')
        for i in range(60):
            ttk.Label(content,text=f'Kafelek testowy {i}').pack(anchor='w',pady=3)
        root.update_idletasks();root.update()
        canvas.configure(scrollregion=canvas.bbox('all'))
        class TkHarness(Harness):
            def __init__(self):
                self.window=win
                self.science_panel=right
                self.science_panel.canvas=canvas
                self.report_table=SimpleNamespace(shell=left_frame,tree=left)
        h=TkHarness()
        h._install_report_pointer_wheel()
        tag=h._report_wheel_tag
        assert left.bindtags()[0] == tag
        assert canvas.bindtags()[0] == tag
        # Simulate pointer hovering right but keyboard focus still held by left.
        win.winfo_pointerxy=lambda:(right.winfo_rootx()+25,right.winfo_rooty()+35)
        win.winfo_containing=lambda x,y:SimpleNamespace(winfo_toplevel=lambda:win)
        left.focus_set()
        left_start=left.yview(); right_start=canvas.yview()
        left.event_generate('<MouseWheel>',delta=-120)
        root.update()
        assert canvas.yview()[0] > right_start[0]
        assert left.yview() == left_start
        # Reverse: right widget receives event although pointer is over left.
        win.winfo_pointerxy=lambda:(left_frame.winfo_rootx()+20,left_frame.winfo_rooty()+40)
        new_right=canvas.yview();old_left=left.yview()
        canvas.event_generate('<MouseWheel>',delta=-120)
        root.update()
        assert left.yview()[0] > old_left[0]
        assert canvas.yview() == new_right
    finally:
        try:
            for seq in ('<MouseWheel>','<Button-4>','<Button-5>'):
                win.unbind_class(tag,seq)
        except (NameError,tk.TclError):
            pass
        root.destroy()


def test_scope_no_global_bind_all_or_source_data_change():
    code=SRC.read_text(encoding='utf-8')
    assert 'bind_all(' not in code
    assert 'self._install_report_pointer_wheel()' in code
    assert 'self.window.unbind_class(tag, sequence)' in code
    assert 'self._show_empty()' in code
