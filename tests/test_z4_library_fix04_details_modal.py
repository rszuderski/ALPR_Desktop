# -*- coding: utf-8 -*-
"""FIX04: real Tk tests of on-demand modal, pinned source, focus and full lists.

All packages are synthetic. No inference, Windows session or model is simulated
as an actual scientific measurement. Old split-pane requirements are superseded.
"""
from __future__ import annotations
import copy
import gc
import json
import time
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
import pytest

from auto_annotation_tool.gui.z4_mobile_artifact_library import MobileArtifactLibrary
from auto_annotation_tool.gui.z4_mobile_history_presentation import SOURCE_CHECK, SOURCE_FILE, SOURCE_EXPORT
from auto_annotation_tool.gui.z4_mobile_artifact_validation_history import read_check_events
from auto_annotation_tool.gui import z4_mobile_artifact_registry as registry
from test_z4lib_history_sources_20261020 import package, check
from z4_tk_visibility_support import foreground_test_window, assert_native_widgets_visible


def pump(root, n=3):
    for _ in range(n):
        root.update()
        time.sleep(.012)


def whole_rows(tree):
    result=[]
    for iid in tree.get_children():
        b=tree.bbox(iid)
        if b and b[1] >= 0 and b[1]+b[3] <= tree.winfo_height():
            result.append(iid)
    return result


@pytest.fixture(scope='module')
def tk_root():
    import tkinter as tk
    gc.collect()
    try:
        root=tk.Tk()
    except tk.TclError:
        pytest.skip('Tk display unavailable')
    root.withdraw()
    errors=[]
    root.report_callback_exception=lambda typ, exc, tb: errors.append((typ,exc))
    yield root, errors
    root.destroy()
    del root
    gc.collect()


@pytest.fixture
def gui(tmp_path,tk_root):
    root,errors=tk_root
    gc.collect()
    scaling=float(root.tk.call('tk','scaling'));root.tk.call('tk','scaling',1.5)
    errors.clear()
    store=tmp_path/'packages'
    for n in range(24):
        path=package(store,f'SYNTHETIC_{n:02d}.alprmodel')
        if n<2:check(store,path)
    path=store/'plate'/'SYNTHETIC_22.alprmodel'
    registry.append_event(store,operation_id='FIX04TEST_EXPORT',operation='SINGLE_MODEL_EXPORT',outcome='STARTED')
    registry.append_event(store,operation_id='FIX04TEST_EXPORT',operation='SINGLE_MODEL_EXPORT',outcome='SUCCEEDED',artifact_path=str(path))
    lib=MobileArtifactLibrary(root,packages_root=store,repo_root=tmp_path)
    deadline=time.monotonic()+8
    while lib.busy and time.monotonic()<deadline:pump(root,1)
    assert not lib.busy
    lib.window.geometry('1649x870+0+0');pump(root,4)
    yield root,lib,store
    lib.close();pump(root)
    root.tk.call('tk','scaling',scaling)
    del lib;gc.collect()
    assert not errors, errors


def choose(gui, source='artifact', origin=None, index=0):
    root,lib,_=gui
    lib.notebook.select(lib.events_tab if source=='event' else lib.packages_tab);pump(root)
    tree=lib.event_tree if source=='event' else lib.art_tree
    entries=lib.events_by_iid if source=='event' else lib.by_iid
    ids=[iid for iid,entry in entries.items() if origin is None or entry.get('_history_origin')==origin]
    iid=ids[index];tree.see(iid);tree.selection_set(iid);tree.focus(iid)
    tree.focus_force();pump(root)
    return tree,iid


def state(lib,tree):
    return (tree.selection(), tree.focus(), tree.yview(), tree.xview(),
            lib.filter_var.get(),copy.deepcopy(lib.sort_state),lib.history_source_filter.get())


@pytest.mark.parametrize('source', ['artifact','event'])
def test_single_selection_keeps_full_table_and_never_opens_modal(gui,source):
    root,lib,_=gui
    tree,iid=choose(gui,source)
    height=tree.winfo_height()
    assert not hasattr(lib,'split') and not hasattr(lib,'details_frame')
    assert lib._details_dialog is None and lib.details_button.instate(['!disabled'])
    for id2 in tree.get_children()[:5]:
        tree.selection_set(id2);pump(root)
        assert tree.winfo_height()==height and lib._details_dialog is None
    if lib.window.winfo_width()>=1600 and lib.window.winfo_height()>=860:
        assert len(whole_rows(tree))>=10


@pytest.mark.parametrize('source', ['artifact','event'])
def test_details_button_opens_local_modal_for_actual_selection(gui,source):
    root,lib,_=gui
    tree,iid=choose(gui,source)
    lib.details_button.invoke();pump(root)
    d=lib._details_dialog
    assert d is not None and d.is_open()
    assert d.snapshot.source==source
    assert str(d.window.transient())==str(lib.window)
    assert d.window.grab_current() is d.window and d.window.grab_status()=='local'
    assert d.window.resizable()==(1,1)
    assert d.summary.winfo_ismapped()
    assert not lib._closed


@pytest.mark.parametrize('source',['artifact','event'])
def test_enter_only_on_list_opens_modal_and_escape_returns_to_list(gui,source):
    root,lib,_=gui
    tree,iid=choose(gui,source);before=state(lib,tree)
    tree.event_generate('<KeyPress-Return>');pump(root)
    d=lib._details_dialog;assert d is not None
    d.summary.focus_force();d.summary.event_generate('<KeyPress-Escape>');pump(root)
    assert lib._details_dialog is None and lib.window.winfo_exists()
    assert lib.window.grab_current() is None
    assert lib.window.focus_get() is tree
    assert state(lib,tree)==before


@pytest.mark.parametrize('source',['artifact','event'])
def test_double_click_opens_pointed_row_not_previous_selection(gui,source):
    root,lib,_=gui
    tree,iid=choose(gui,source)
    wanted=tree.get_children()[1];tree.see(wanted);pump(root)
    x,y,w,h=tree.bbox(wanted)
    # Directly call the bound handler with the actual cell coordinates. Physical
    # event-dispatch sequence is covered separately (Tk forbids event_generate(Double)).
    assert lib._details_double_click(SimpleNamespace(widget=tree,x=x+25,y=y+h//2))=='break'
    pump(root);d=lib._details_dialog
    assert tree.selection()==(wanted,)
    target=lib.selected_event() if source=='event' else lib.selected_artifact()
    assert d.snapshot.path==str(target.get('artifact_path') if source=='event' else target.path)


@pytest.mark.parametrize('where',['header','blank','other_widget'])
def test_double_click_outside_real_row_does_not_open_stale_selection(gui,where):
    root,lib,_=gui
    tree,iid=choose(gui)
    if where=='header':ev=SimpleNamespace(widget=tree,x=30,y=3)
    elif where=='blank':
        lib.filter_var.set('SYNTHETIC_00');pump(root)
        tree.selection_set(tree.get_children()[0]);pump(root)
        ev=SimpleNamespace(widget=tree,x=30,y=tree.winfo_height()-5)
    else:ev=SimpleNamespace(widget=lib.window,x=10,y=10)
    lib._details_double_click(ev);pump(root)
    assert lib._details_dialog is None


def test_physical_two_click_sequence_opens_modal(gui):
    root,lib,_=gui
    tree,iid=choose(gui);x,y,w,h=tree.bbox(iid)
    for t in [1000,1080]:
        tree.event_generate('<ButtonPress-1>',x=x+30,y=y+h//2,time=t)
        tree.event_generate('<ButtonRelease-1>',x=x+30,y=y+h//2,time=t+15)
    pump(root)
    assert lib._details_dialog is not None


def test_enter_in_search_does_not_open_selected_item(gui):
    root,lib,_=gui
    choose(gui)
    import tkinter as tk
    entries=[]
    def visit(w):
        if w.winfo_class()=='TEntry':entries.append(w)
        for child in w.winfo_children():visit(child)
    visit(lib.window)
    entry=entries[0];entry.focus_force();entry.event_generate('<KeyPress-Return>');pump(root)
    assert lib._details_dialog is None


@pytest.mark.parametrize('source',['artifact','event'])
def test_no_selection_disables_details_and_programmatic_open_does_nothing(gui,source):
    root,lib,_=gui
    lib.notebook.select(lib.events_tab if source=='event' else lib.packages_tab);pump(root)
    assert lib.details_button.instate(['disabled'])
    assert lib.show_details() is None
    assert lib._details_dialog is None


@pytest.mark.parametrize('source',['artifact','event'])
def test_repeated_open_reuses_same_modal_and_keeps_immutable_snapshot(gui,source):
    root,lib,_=gui
    tree,iid=choose(gui,source)
    d=lib.show_details();pump(root)
    assert lib.show_details() is d
    with pytest.raises(FrozenInstanceError):d.snapshot.name='other'
    before=(d.snapshot,d.summary.get('1.0','end'),d.raw.get('1.0','end'))
    tree.selection_set(tree.get_children()[-1]);pump(root)
    lib.refresh_views();pump(root)
    assert (d.snapshot,d.summary.get('1.0','end'),d.raw.get('1.0','end'))==before


def test_open_modal_not_replaced_when_filter_removes_original_row(gui):
    root,lib,_=gui
    tree,iid=choose(gui)
    d=lib.show_details();pump(root);before=d.snapshot
    lib.filter_var.set('not-present');pump(root)
    assert not lib.art_tree.selection()
    assert d.snapshot is before and d.raw.get('1.0','end').strip()==before.raw_json.strip()
    d.path_button.invoke();assert lib.window.clipboard_get()==before.path
    d.close();pump(root)
    assert lib.details_button.instate(['disabled'])
    assert lib.show_details() is None


@pytest.mark.parametrize('origin,json_label,sha_label',[
    (SOURCE_CHECK,'Wpis źródłowy (JSON)','Kopiuj SHA pliku z wpisu'),
    (SOURCE_EXPORT,'Wpis źródłowy (JSON)','Kopiuj SHA pliku z wpisu'),
    (SOURCE_FILE,'Wpis źródłowy (JSON)','Kopiuj SHA pliku z wpisu'),
])
def test_each_history_source_retains_full_json_and_correct_sha_policy(gui,origin,json_label,sha_label):
    root,lib,_=gui
    tree,iid=choose(gui,'event',origin=origin)
    original=copy.deepcopy(lib.selected_event());d=lib.show_details();pump(root)
    assert json.loads(d.raw.get('1.0','end'))==original
    assert d.notebook.tab(1,'text')==json_label and d.sha_button.cget('text')==sha_label
    if origin==SOURCE_FILE:
        assert not d.snapshot.sha_value and d.sha_button.instate(['disabled'])
    else:
        d.sha_button.invoke();assert lib.window.clipboard_get()==original['artifact_sha256']
        assert len(lib.window.clipboard_get())==64
    assert lib.validate_button.instate(['disabled'])


def test_package_checkpoint_sha_is_not_file_sha(gui):
    root,lib,store=gui
    tree,iid=choose(gui)
    row=lib.selected_artifact();new=copy.deepcopy(row.manifest)
    new['source']={'checkpoint_sha256':'abc1'*16}
    lib.by_iid[iid]=replace(row,manifest=new)
    d=lib.show_details();pump(root)
    assert d.notebook.tab(1,'text')=='Manifest i źródła (JSON)'
    assert d.sha_button.cget('text')=='Kopiuj SHA checkpointu'
    d.sha_button.invoke();assert lib.window.clipboard_get()=='abc1'*16
    assert d.snapshot.sha_value!=registry.sha256_file(Path(row.path))
    assert json.loads(d.raw.get('1.0','end'))['manifest']==new


def test_no_checkpoint_sha_does_not_fall_back_to_artifact_hash(gui):
    root,lib,_=gui;tree,iid=choose(gui)
    d=lib.show_details();pump(root)
    assert d.sha_button.instate(['disabled']) and not d.snapshot.sha_value


@pytest.mark.parametrize('close_kind',['button','escape','wm','owner','direct_destroy','before_map'])
def test_all_close_routes_release_grab_and_remove_singleton(gui,close_kind):
    root,lib,_=gui;tree,iid=choose(gui)
    before=state(lib,tree);d=lib.show_details()
    if close_kind!='before_map':pump(root)
    if close_kind=='button':d.close_button.invoke()
    elif close_kind=='escape':d._escape()
    elif close_kind=='wm':d.window.tk.call(d.window.protocol('WM_DELETE_WINDOW'))
    elif close_kind=='owner':lib.close()
    elif close_kind=='direct_destroy':d.window.destroy()
    else:d.close()
    pump(root)
    assert lib._details_dialog is None and not d.is_open()
    assert root.grab_current() is None
    if close_kind!='owner':
        assert lib.window.winfo_exists() and state(lib,tree)==before


@pytest.mark.parametrize('source',['artifact','event'])
def test_modal_close_preserves_scroll_filter_order_and_choice(gui,source):
    root,lib,_=gui;tree,iid=choose(gui,source)
    lib.sort_column('events' if source=='event' else 'artifacts','date')
    tree.yview_moveto(.5);tree.xview_moveto(.25);pump(root)
    before=state(lib,tree);d=lib.show_details();pump(root)
    d.notebook.select(1);pump(root);d.close();pump(root)
    assert state(lib,tree)==before


def test_reopen_uses_new_source_not_last_closed_modal(gui):
    root,lib,_=gui;tree,iid=choose(gui)
    d=lib.show_details();pump(root);original=d.snapshot.path;d.close()
    tree,iid=choose(gui,'event',origin=SOURCE_CHECK,index=1)
    new=lib.show_details();pump(root)
    assert new is not d and new.snapshot.source=='event'
    assert new.snapshot.path==lib.selected_event()['artifact_path']
    assert new.notebook.tab(1,'text')=='Wpis źródłowy (JSON)'


@pytest.mark.parametrize('size,scale',[('940x570',1.333),('1280x720',1.667),('1649x870',2.0)])
def test_modal_footer_and_both_scrollable_texts_visible_at_dpi(gui,size,scale):
    root,lib,_=gui
    root.tk.call('tk','scaling',scale)
    lib._font_body.configure(size=10);lib._font_bold.configure(size=10);lib._font_title.configure(size=12)
    lib._apply_palette();lib.window.geometry(size+'+0+0');pump(root)
    choose(gui);d=lib.show_details();pump(root,6)
    d.window.geometry('760x510');pump(root,6)
    # FIX05: this is a native DESKTOP visibility test, not only Tk geometry.
    # Isolate its foreground on Windows (PowerShell/browser can otherwise cover
    # the point), then wait for stable actual hits after native resize. Never
    # accept None, skip the platform, enable a disabled action or mock the hit.
    with foreground_test_window(d.window, focus=d.summary):
        assert_native_widgets_visible(root,d.window,[*d.action_buttons,d.close_button])
        for b in [*d.action_buttons,d.close_button,d.status_label]:
            assert b.winfo_ismapped()
            x,y=b.winfo_rootx()-d.window.winfo_rootx(),b.winfo_rooty()-d.window.winfo_rooty()
            assert x>=0 and y>=0 and x+b.winfo_width()<=d.window.winfo_width()
            assert y+b.winfo_height()<=d.window.winfo_height()
        # Keep the original exact native hit-test assertions as well.
        for b in [*d.action_buttons,d.close_button]:
            assert root.winfo_containing(b.winfo_rootx()+b.winfo_width()//2,b.winfo_rooty()+b.winfo_height()//2)==b
        assert d.summary.winfo_height()>=3*lib._font_body.metrics('linespace')
        assert d.summary.cget('state')=='disabled' and d.raw.cget('state')=='disabled'
        d.notebook.select(1);pump(root)
        assert_native_widgets_visible(root,d.window,[*d.action_buttons,d.close_button])
        assert d.raw.winfo_ismapped() and d.raw.cget('wrap')=='none'
        assert d.raw.cget('xscrollcommand') and d.raw.cget('yscrollcommand')
        assert d.summary.cget('yscrollcommand')


def test_open_close_browse_copies_do_not_mutate_files_or_ledger(gui):
    root,lib,store=gui
    before={str(p):p.read_bytes() for p in store.rglob('*') if p.is_file()}
    events=copy.deepcopy(lib.history);checks=copy.deepcopy(lib._desktop_check_events)
    for source,origin in [('artifact',None),('event',SOURCE_CHECK),('event',SOURCE_EXPORT),('event',SOURCE_FILE)]:
        choose(gui,source,origin=origin);d=lib.show_details();pump(root)
        d.path_button.invoke();d.sha_button.invoke();d.close();pump(root)
    after={str(p):p.read_bytes() for p in store.rglob('*') if p.is_file()}
    assert before==after and events==lib.history and checks==lib._desktop_check_events


def test_background_validation_refresh_does_not_rewrite_open_modal(gui):
    root,lib,store=gui;tree,iid=choose(gui)
    row=lib.selected_artifact();d=lib.show_details();pump(root);before=d.snapshot
    entry=check(store,Path(row.path))
    lib._verification_active={'token':45,'path':row.path,'kind':row.kind};lib.busy=True
    lib.pipe.put(('verified',dict(token=45,path=row.path,kind=row.kind,ok=True,sha256=entry['sha256'],history_record=entry,history_warning='')))
    lib.poll();pump(root)
    assert d.snapshot is before and d.raw.get('1.0','end').strip()==before.raw_json.strip()
    d.close();again=lib.show_details();pump(root)
    assert 'Kontrola PC zaliczona' in again.summary.get('1.0','end')
    assert len(registry.load_history(store)[0])==2  # Existing export unchanged.


def test_dark_theme_changes_presentation_not_snapshot(gui):
    from tkinter import ttk
    root,lib,_=gui;choose(gui);d=lib.show_details();pump(root);before=d.snapshot
    style=ttk.Style(root);original=style.theme_use();name='FIX04_DARK_'+str(id(lib))
    style.theme_create(name,parent='clam',settings={'TFrame':{'configure':{'background':'#222831'}}})
    try:
        style.theme_use(name);pump(root,6)
        assert d.summary.cget('background')==lib.colors['surface']=='#29343E'
        assert d.snapshot is before
    finally:
        style.theme_use(original);pump(root)


def test_modal_size_retained_only_for_current_library_window(gui):
    root,lib,_=gui;choose(gui);d=lib.show_details();pump(root)
    d.window.geometry('950x640');pump(root);size=(d.window.winfo_width(),d.window.winfo_height())
    d.close();again=lib.show_details();pump(root)
    assert (again.window.winfo_width(),again.window.winfo_height())==size


@pytest.mark.parametrize('direct',[False,True])
def test_previous_local_grab_is_restored_without_becoming_global(gui,direct):
    root,lib,_=gui;choose(gui)
    lib.window.grab_set()
    try:
        d=lib.show_details();pump(root)
        assert root.grab_current() is d.window
        if direct:d.window.destroy()
        else:d.close()
        pump(root)
        assert root.grab_current() is lib.window
        assert lib.window.grab_status()=='local'
    finally:
        lib.window.grab_release()


def test_another_dialogs_newer_grab_is_not_stolen_on_close(gui):
    import tkinter as tk
    root,lib,_=gui;choose(gui)
    d=lib.show_details();pump(root)
    other=tk.Toplevel(lib.window);pump(root);other.grab_set()
    try:
        d.close(restore_focus=False);pump(root)
        assert root.grab_current() is other
    finally:
        other.grab_release();other.destroy()


def test_large_source_summary_and_json_can_be_scrolled_to_last_byte(gui):
    root,lib,_=gui;tree,iid=choose(gui)
    row=lib.selected_artifact();manifest=copy.deepcopy(row.manifest)
    manifest['qa_large_source']=['Y'*250 for _ in range(100)]
    lib.by_iid[iid]=replace(row,manifest=manifest)
    d=lib.show_details();pump(root)
    d.notebook.select(1);pump(root)
    d.raw.yview_moveto(1);d.raw.xview_moveto(1);pump(root)
    assert d.raw.yview()[1]==1.0 and d.raw.xview()[1]==1.0
    assert json.loads(d.raw.get('1.0','end'))['manifest']==manifest


def test_sha_for_multi_model_package_has_role_labels_and_full_values(gui):
    root,lib,_=gui;tree,iid=choose(gui)
    row=lib.selected_artifact()
    metadata={'plate':{'checkpoint_sha256':'1'*64},'character':{'checkpoint_sha256':'2'*64}}
    lib.by_iid[iid]=replace(row,kind='PACKAGE',role='MT+MZ',manifest={},child_metadata=metadata)
    d=lib.show_details();pump(root)
    d.sha_button.invoke()
    assert lib.window.clipboard_get()=='MT: '+'1'*64+'\nMZ: '+'2'*64


def test_mutating_nested_source_after_open_does_not_change_json(gui):
    root,lib,_=gui;tree,iid=choose(gui)
    row=lib.selected_artifact();d=lib.show_details();pump(root);before=d.raw.get('1.0','end')
    row.manifest['new_after_open']='must not appear'
    lib.artifact_selected();pump(root)
    assert d.raw.get('1.0','end')==before
    assert 'new_after_open' not in json.loads(d.snapshot.raw_json)['manifest']
