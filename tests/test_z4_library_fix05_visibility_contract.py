# -*- coding: utf-8 -*-
"""FIX05: foreground setup never replaces real visibility with a geometry PASS.

Real Tk controls and isolated negative cases; no production changes. Synthetic
Win32 None results below exercise rejection/waiting, not a claim of Windows QA.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import tkinter as tk
import pytest

from test_z4_library_fix04_details_modal import gui, tk_root, choose, pump, state
from z4_tk_visibility_support import foreground_test_window, assert_native_widgets_visible


def opened(gui):
    root, lib, _ = gui
    tree, _ = choose(gui)
    dialog = lib.show_details()
    pump(root, 6)
    return root, lib, tree, dialog


@pytest.mark.parametrize('source', ['artifact','event'])
def test_application_stacks_modal_above_owner_without_test_topmost(gui,source):
    root,lib,_=gui
    choose(gui,source)
    d=lib.show_details();pump(root,6)
    # Check application's own ordering BEFORE the test foreground isolation.
    assert root.tk.getboolean(root.tk.call('wm','stackorder',d.window,'isabove',lib.window))
    assert not root.tk.getboolean(d.window.attributes('-topmost'))
    assert d.window.grab_current() is d.window and d.window.grab_status()=='local'


@pytest.mark.parametrize('finish',['success','error','destroy'])
def test_foreground_setup_restores_topmost_and_does_not_take_a_grab(gui,finish,monkeypatch):
    root,lib,tree,d=opened(gui)
    previous=d.window.attributes('-topmost')
    grab=root.grab_current()
    geometry=d.window.geometry()
    states=[b.state() for b in d.action_buttons]
    original=d.window.attributes
    calls=[]
    def attributes(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(d.window,'attributes',attributes)
    try:
        with foreground_test_window(d.window,focus=d.summary):
            assert root.grab_current() is grab
            assert [b.state() for b in d.action_buttons]==states
            assert d.window.geometry()==geometry
            if finish=='error':raise RuntimeError('controlled assertion failure')
            if finish=='destroy':d.close()
    except RuntimeError as exc:
        assert finish=='error' and str(exc)=='controlled assertion failure'
    assert ('-topmost',True) in calls
    if finish!='destroy':
        assert calls[-1]==('-topmost',previous)
        assert root.tk.getboolean(original('-topmost'))==root.tk.getboolean(previous)
        assert root.grab_current() is grab
    else:
        assert root.grab_current() is None and not d.is_open()


@pytest.mark.parametrize('hit',['none','owner','wrong_button'])
def test_no_native_hit_or_wrong_widget_remains_a_failure(gui,monkeypatch,hit):
    root,lib,_,d=opened(gui)
    target=d.path_button
    fake={'none':None,'owner':lib.window,'wrong_button':d.close_button}[hit]
    monkeypatch.setattr(root,'winfo_containing',lambda *args:fake)
    with foreground_test_window(d.window):
        with pytest.raises(AssertionError,match='NATIVE_VISIBILITY_NOT_CONFIRMED') as failed:
            assert_native_widgets_visible(root,d.window,[target],timeout=.08,stable_for=.02)
    data=json.loads(str(failed.value).split('\n',1)[1])
    assert data['controls'][0]['problem']=='native_hit_mismatch'
    assert data['controls'][0]['hit']==(str(fake) if fake is not None else None)
    assert data['tk'] and data['scaling'] and data['modal_geometry']
    assert 'screen' in data and 'grab' in data


def test_delayed_native_visibility_waits_for_real_target_not_geometry(gui,monkeypatch):
    root,lib,_,d=opened(gui)
    native=root.winfo_containing
    hits=[]
    def delayed(*args):
        hit=None if len(hits)<4 else native(*args)
        hits.append(hit)
        return hit
    monkeypatch.setattr(root,'winfo_containing',delayed)
    with foreground_test_window(d.window):
        assert_native_widgets_visible(root,d.window,[d.path_button],timeout=2,stable_for=.03)
    assert hits[:4]==[None]*4 and len(hits)>=6
    assert hits[-1] is d.path_button


def test_visibility_must_be_stable_for_a_continuous_interval(gui,monkeypatch):
    root,lib,_,d=opened(gui)
    native=root.winfo_containing;calls=[]
    def unstable(*args):
        index=len(calls)
        hit=None if index in (0,2,4) else native(*args)
        calls.append((time.monotonic(),hit))
        return hit
    monkeypatch.setattr(root,'winfo_containing',unstable)
    with foreground_test_window(d.window):
        assert_native_widgets_visible(root,d.window,[d.path_button],timeout=2,stable_for=.05)
    assert calls[4][1] is None
    assert calls[-1][0]-calls[5][0]>=.05
    assert all(item[1] is d.path_button for item in calls[5:])


@pytest.mark.parametrize('problem',['hidden','ancestor_clipped'])
def test_real_hidden_or_clipped_control_never_passes(gui,problem):
    root,lib,_,d=opened(gui)
    with foreground_test_window(d.window):
        if problem=='hidden':
            d.path_button.grid_remove();pump(root)
        else:
            # Actual widget and actual layout: tiny parent clips a wide button.
            frame=tk.Frame(d.shell,width=20,height=20)
            frame.place(x=30,y=140,width=20,height=20)
            bad=tk.Button(frame,text='clipped actual control')
            bad.place(x=0,y=0,width=140,height=24);pump(root)
        target=d.path_button if problem=='hidden' else bad
        with pytest.raises(AssertionError,match='NATIVE_VISIBILITY_NOT_CONFIRMED') as failed:
            assert_native_widgets_visible(root,d.window,[target],timeout=.08,stable_for=.02)
        diag=json.loads(str(failed.value).split('\n',1)[1])['controls'][0]
        assert diag['problem']==('not_viewable_or_empty' if problem=='hidden' else 'clipped_by_ancestor')


def test_real_overlapping_control_is_rejected_without_rearranging_it(gui):
    root,lib,_,d=opened(gui)
    with foreground_test_window(d.window):
        b=d.path_button
        cover=tk.Frame(b.master,background='black')
        cover.place(x=b.winfo_x(),y=b.winfo_y(),width=b.winfo_width(),height=b.winfo_height())
        cover.lift();pump(root)
        with pytest.raises(AssertionError,match='native_hit_mismatch'):
            assert_native_widgets_visible(root,d.window,[b],timeout=.10,stable_for=.02)
        assert cover.winfo_ismapped()  # Assertion never hides/reorders an obstruction.
        cover.destroy();pump(root)
        assert_native_widgets_visible(root,d.window,[b],timeout=2,stable_for=.02)


def test_foreground_setup_refuses_to_show_an_unmapped_dialog(gui):
    root,lib,_,d=opened(gui)
    d.window.withdraw();pump(root)
    with pytest.raises(AssertionError,match='already visible'):
        with foreground_test_window(d.window):pass
    assert d.window.state()=='withdrawn'


def test_disabled_sha_stays_disabled_during_visibility_test(gui):
    root,lib,_,d=opened(gui)
    assert d.sha_button.instate(['disabled'])
    with foreground_test_window(d.window):
        assert_native_widgets_visible(root,d.window,d.action_buttons+[d.close_button])
        assert d.sha_button.instate(['disabled'])
    assert d.sha_button.instate(['disabled'])


def test_test_isolation_preserves_library_state_snapshot_and_bytes(gui):
    root,lib,tree,d=opened(gui);store=gui[2]
    before={str(p):p.read_bytes() for p in store.rglob('*') if p.is_file()}
    selection=state(lib,tree);snapshot=d.snapshot
    with foreground_test_window(d.window,focus=d.summary):
        assert_native_widgets_visible(root,d.window,d.action_buttons+[d.close_button])
    assert state(lib,tree)==selection and d.snapshot is snapshot
    assert {str(p):p.read_bytes() for p in store.rglob('*') if p.is_file()}==before


@pytest.mark.parametrize('timeout,stable',[(-1,0),(.1,.1),(0,0),(11,0),(1,-.1)])
def test_invalid_wait_budgets_are_rejected(gui,timeout,stable):
    root,lib,_,d=opened(gui)
    with pytest.raises(ValueError):
        assert_native_widgets_visible(root,d.window,[d.path_button],timeout=timeout,stable_for=stable)


def test_empty_target_list_is_not_a_vacuous_pass(gui):
    root,lib,_,d=opened(gui)
    with pytest.raises(ValueError,match='No controls'):
        assert_native_widgets_visible(root,d.window,[])


def test_helpers_are_not_imported_by_production_and_original_exact_hit_assertion_remains():
    repo=Path(__file__).resolve().parents[1]
    for file in (repo/'auto_annotation_tool').rglob('*.py'):
        assert 'z4_tk_visibility_support' not in file.read_text(encoding='utf-8-sig')
    code=(Path(__file__).parent/'test_z4_library_fix04_details_modal.py').read_text(encoding='utf-8')
    assert 'assert root.winfo_containing(b.winfo_rootx()+b.winfo_width()//2,b.winfo_rooty()+b.winfo_height()//2)==b' in code
    assert "('940x570',1.333),('1280x720',1.667),('1649x870',2.0)" in code
