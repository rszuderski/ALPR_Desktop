# -*- coding: utf-8 -*-
"""FIX02: presentation, scientific scope, live widgets, layout and immutability.

Only synthetic archives/records. No real model, inference, phone, training,
conversion, Workspace or original export history is changed by these tests.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import json
import time
import gc
from types import SimpleNamespace

import pytest

from auto_annotation_tool.gui.z4_mobile_library_view import (
    short_sha, size_mib, display_utc, file_name, artifact_state, artifact_sheet,
    event_sheet, event_tone,
)
from auto_annotation_tool.gui.z4_mobile_artifact_validation_history import (
    record_desktop_check, history_key, observe_last_checks, read_check_events,
)
from auto_annotation_tool.gui.z4_mobile_history_presentation import SOURCE_CHECK, SOURCE_FILE
from auto_annotation_tool.gui import z4_mobile_artifact_registry as registry
from test_z4lib_history_sources_20261020 import package, check


@pytest.mark.parametrize('value,expected', [(1048576,'1,00'),(2715812,'2,59'),(0,'0,00'),(-1,'—'),(None,'—')])
def test_mib_is_binary_and_unknown_is_not_zero(value, expected):
    assert size_mib(value) == expected


@pytest.mark.parametrize('value', [None,'','a'*63,'z'*64,False])
def test_invalid_hash_does_not_become_a_valid_abbreviation(value):
    assert short_sha(value) == '—'


def test_short_hash_explicit_and_complete_value_not_changed():
    value='abcd'*16
    assert short_sha(value) == value[:12]+'…'+value[-8:]
    assert len(value)==64


@pytest.mark.parametrize('value,expected', [
    ('2026-10-20T09:47:57+02:00','20.10.2026 07:47:57'),
    ('2026-10-20T07:47:57Z','20.10.2026 07:47:57'),
    ('','—'),('unavailable','unavailable'),
])
def test_utc_display_does_not_change_instant(value,expected):
    assert display_utc(value)==expected


def test_timezone_is_not_invented_for_naive_timestamp():
    assert 'strefa niepodana' in display_utc('2026-10-20T07:47:57')


def test_windows_basename_is_portable_and_not_a_new_path():
    assert file_name(r'C:\Users\test\Workspace\some.alprmodel') == 'some.alprmodel'


@pytest.mark.parametrize('state', ['REGISTERED_SHA_MISMATCH','REGISTERED_SHA_UNREADABLE','READ_ERROR','MANIFEST_ERROR','RESEARCH_ONLY_NOT_PACKAGED','RESEARCH_SHA_MISMATCH','UNKNOWN'])
def test_provenance_blocker_wins_over_even_a_fresh_pc_flag(state):
    row=SimpleNamespace(path='some.alprmodel',kind='MODEL',state=state,error='')
    result=artifact_state(row,validated=True)
    assert result.tone!='success' and 'OK' not in result.label


def test_single_model_pc_ok_not_presented_as_damaged_or_android_test(tmp_path):
    root=tmp_path/'models';path=package(root)
    entry=check(root,path)
    rows=registry.library_snapshot(root,tmp_path)[0]
    history=observe_last_checks(root,rows)[history_key(path)]
    summary=artifact_sheet(rows[0],history=history)
    assert summary.tone=='success' and 'pojedynczy model' in summary.title
    assert 'niepełny' not in summary.state and 'uszkodz' not in summary.state
    assert 'MT+MZ' in summary.next_step
    assert 'nie potwierdza importu' in summary.note
    assert short_sha(entry['sha256']) in dict(summary.fields).values()


def test_changed_bytes_never_reuse_historical_green(tmp_path):
    root=tmp_path/'models';path=package(root);check(root,path)
    path.write_bytes(path.read_bytes()+b'changed')
    rows=registry.library_snapshot(root,tmp_path)[0]
    history=observe_last_checks(root,rows)[history_key(path)]
    assert artifact_state(rows[0],history=history).tone=='danger'


def test_pc_event_is_not_export_or_new_measurement():
    row={'_history_origin':SOURCE_CHECK,'outcome':'PC_PASSED','artifact_path':'C:/x.alprmodel',
         'artifact_sha256':'a'*64,'observed_utc':'2026-10-20T07:00:00+00:00','_source_record':{'detail':'test'}}
    before=copy.deepcopy(row);sheet=event_sheet(row)
    assert row==before
    assert 'wówczas' in sheet.state and 'Historyczny PASS' in sheet.note
    assert 'nie konwersja' in sheet.next_step


def test_disk_file_not_an_export_timestamp_or_pc_pass():
    sheet=event_sheet({'_history_origin':SOURCE_FILE,'outcome':'FOUND_NO_EXPORT_EVENT',
                       'artifact_path':'C:/x.alprmodel','_file_modified_utc':'2026-10-20T07:00:00+00:00'})
    assert sheet.tone=='neutral' and 'Data pliku NIE jest datą konwersji' in sheet.note
    assert 'Data modyfikacji PLIKU (UTC)' in dict(sheet.fields)
    assert not any('SHA' in label for label,_ in sheet.fields)


@pytest.mark.parametrize('outcome,tone', [('PC_PASSED','success'),('PC_FAILED','danger'),('FAILED','danger'),('INCONSISTENT_HISTORY','danger'),('STARTED_NO_RESULT','warning'),('FOUND_NO_EXPORT_EVENT','neutral')])
def test_history_color_does_not_replace_state(outcome,tone):
    assert event_tone({'outcome':outcome})==tone


def pump(root, rounds=4):
    for _ in range(rounds):
        root.update()
        time.sleep(.015)


@pytest.fixture
def real_gui(tmp_path):
    gc.collect()
    import tkinter as tk
    from auto_annotation_tool.gui.z4_mobile_artifact_library import MobileArtifactLibrary
    try:root=tk.Tk()
    except tk.TclError:pytest.skip('Tk display unavailable')
    root.withdraw()
    original_scaling = float(root.tk.call('tk', 'scaling'))
    root.tk.call('tk', 'scaling', 1.333)
    store=tmp_path/'packages'
    first=package(store,'mt.alprmodel');second=package(store,'mz.alprmodel')
    check(store,first)
    original={str(p.relative_to(store)):p.read_bytes() for p in store.rglob('*') if p.is_file()}
    lib=MobileArtifactLibrary(root,packages_root=store,repo_root=tmp_path)
    deadline=time.monotonic()+8
    while lib.busy and time.monotonic()<deadline:
        root.update();time.sleep(.02)
    pump(root)
    assert not lib.busy
    yield root,lib,store,original
    lib.close();root.tk.call('tk', 'scaling', original_scaling);root.destroy();gc.collect()


def select_first(root,lib):
    lib.art_tree.selection_set(next(iter(lib.by_iid)))
    pump(root)


@pytest.mark.parametrize('geometry,scale', [('940x570',1.333),('940x570',2.0),('1280x720',1.667),('1649x874',2.0)])
@pytest.mark.parametrize('tab', ['packages','events'])
def test_footer_widgets_visible_inside_window_at_resize_and_dpi(real_gui,geometry,scale,tab):
    root,lib,store,_=real_gui
    root.tk.call('tk','scaling',scale)
    # Force font remeasure without changing application-global fonts.
    lib._font_body.configure(size=10);lib._font_bold.configure(size=10);lib._font_title.configure(size=12)
    lib._apply_palette()
    w,h=map(int,geometry.split('x'))
    w=min(w,root.winfo_screenwidth()-70);h=min(h,root.winfo_screenheight()-100)
    lib.window.geometry(f'{w}x{h}+0+0');lib.window.lift()
    lib.notebook.select(lib.events_tab if tab=='events' else lib.packages_tab)
    pump(root,8)
    for widget in [*lib.action_buttons,lib.close_button,lib.status_label]:
        assert widget.winfo_ismapped(),str(widget)
        x=widget.winfo_rootx()-lib.window.winfo_rootx();y=widget.winfo_rooty()-lib.window.winfo_rooty()
        assert x>=0 and y>=0 and x+widget.winfo_width()<=lib.window.winfo_width()
        assert y+widget.winfo_height()<=lib.window.winfo_height(), (str(widget),y,widget.winfo_height())
    # Covers regression where row frames masked the mapped action buttons.
    for button in lib.action_buttons:
        x=button.winfo_rootx()+button.winfo_width()//2;y=button.winfo_rooty()+button.winfo_height()//2
        assert lib.window.winfo_containing(x,y)==button
    tree=lib.event_tree if tab=='events' else lib.art_tree
    first=tree.get_children()[0]
    assert tree.bbox(first), 'History controls must not conceal the entire table'
    assert lib._details_dialog is None
    assert not hasattr(lib, 'details_frame')
    assert tree.winfo_height() >= 2 * lib._font_body.metrics('linespace')


def test_selection_filter_and_clipboard_keep_full_source_not_short_label(real_gui):
    root,lib,store,original=real_gui
    lib.notebook.select(lib.events_tab);lib.history_source_filter.set('Kontrole PC');lib.refresh_views();pump(root)
    iid=next(iter(lib.events_by_iid));lib.event_tree.selection_set(iid);lib.event_selected();pump(root)
    event=lib.events_by_iid[iid];complete=event['artifact_sha256']
    assert short_sha(complete) in lib.event_tree.item(iid,'values')
    lib.copy_evidence_sha();assert lib.window.clipboard_get()==complete
    dialog=lib.show_details();pump(root)
    assert complete in dialog.raw.get('1.0','end')
    assert lib.validate_button.instate(['disabled'])
    assert dialog.sha_button.cget('text')=='Kopiuj SHA pliku z wpisu'
    dialog.sha_button.invoke();assert lib.window.clipboard_get()==complete
    dialog.close();pump(root)
    assert lib.audit_checkbox.instate(['disabled'])
    lib.filter_var.set('definitely-no-match');pump(root)
    assert lib._detail_json.strip()==''
    assert lib.details_button.instate(['disabled']) and lib.path_button.instate(['disabled'])
    assert 'Wiersze po filtrze: 0 / 1' in lib.event_view_status.get()
    assert lib.history_counts['pc_checks']==1
    after={str(p.relative_to(store)):p.read_bytes() for p in store.rglob('*') if p.is_file()}
    assert after==original and not registry.history_file(store).exists()


def test_visible_labels_can_be_searched_and_global_cards_do_not_follow_filter(real_gui):
    root,lib,_,_=real_gui
    lib.notebook.select(lib.events_tab);lib.filter_var.set('PASS (wówczas)');pump(root)
    assert len(lib.events_by_iid)==1
    assert lib.history_counts=={'export_operations':0,'export_events':0,'pc_checks':1,'unlogged_packages':2}
    assert 'BRAK DZIENNIKA' not in lib.event_view_status.get()
    lib.filter_var.set('');lib.notebook.select(lib.packages_tab);lib.filter_var.set('PC: OK');pump(root)
    assert len(lib.by_iid)==1


def test_empty_export_reader_warning_retained_in_compact_mode(real_gui):
    root,lib,_,_=real_gui
    lib._history_read_note='Pominięto 3 niepoprawne wpisy — sprawdź rejestr.'
    lib.notebook.select(lib.events_tab);lib.window.geometry('940x570+0+0');lib.refresh_views();pump(root,8)
    assert 'Pominięto 3' in lib.history_note_var.get()
    assert lib.history_note_label.winfo_ismapped()
    assert 'BRAK DZIENNIKA' not in lib.event_view_status.get()


def test_color_scheme_is_local_not_a_global_theme_change(real_gui):
    root,lib,_,_=real_gui
    from tkinter import ttk
    style=ttk.Style(root);old=style.theme_use()
    before=style.lookup('TLabel','foreground')
    lib._theme_changed();pump(root)
    assert style.theme_use()==old and style.lookup('TLabel','foreground')==before
    assert lib.art_tree.cget('style').startswith('Z4Library')
    select_first(root,lib);dialog=lib.show_details();pump(root)
    assert dialog.summary.tag_cget('success','foreground')==lib.colors['success']
    dialog.close()


def test_json_tab_tracks_visible_table_not_previous_selection(real_gui):
    root,lib,_,_=real_gui
    select_first(root,lib)
    assert lib._detail_json.strip()
    lib.notebook.select(lib.events_tab);pump(root)
    assert lib._detail_json.strip()==''
    assert lib.details_button.instate(['disabled'])
    lib.event_tree.selection_set(next(iter(lib.events_by_iid)));lib.event_selected()
    dialog=lib.show_details();pump(root)
    assert dialog.notebook.tab(1,'text')=='Wpis źródłowy (JSON)'
    dialog.close()


def test_status_sort_uses_presented_pc_state(real_gui):
    root,lib,_,_=real_gui
    lib.sort_column('artifacts','state')
    labels=[lib.art_tree.set(iid,'state') for iid in lib.art_tree.get_children()]
    assert labels==sorted(labels,key=str.casefold)
    lib.sort_column('artifacts','state')
    labels=[lib.art_tree.set(iid,'state') for iid in lib.art_tree.get_children()]
    assert labels==sorted(labels,key=str.casefold,reverse=True)


def test_close_cancels_pending_callbacks(real_gui):
    root,lib,_,_=real_gui
    lib._queue_layout();lib.close()
    assert lib._closed and lib._poll_after_id is None and lib._layout_after_id is None
    pump(root)


def test_choose_mode_footer_wraps_without_triggering_export(real_gui):
    root,original,store,_=real_gui
    from auto_annotation_tool.gui.z4_mobile_artifact_library import MobileArtifactLibrary
    original.close()
    lib=MobileArtifactLibrary(root,packages_root=store,repo_root=store.parent,
        on_choose=lambda _path: pytest.fail('Browsing must not start an export'), choose_role='MT')
    try:
        deadline=time.monotonic()+8
        while lib.busy and time.monotonic()<deadline:root.update();time.sleep(.02)
        lib.window.geometry('940x570+0+0');lib.window.lift();pump(root,8)
        assert lib.choose_button.winfo_ismapped()
        for b in [*lib.action_buttons,lib.close_button]:
            x=b.winfo_rootx()-lib.window.winfo_rootx();y=b.winfo_rooty()-lib.window.winfo_rooty()
            assert x>=0 and x+b.winfo_width()<=lib.window.winfo_width()
            assert y>=0 and y+b.winfo_height()<=lib.window.winfo_height()
            assert lib.window.winfo_containing(b.winfo_rootx()+b.winfo_width()//2,
                                             b.winfo_rooty()+b.winfo_height()//2)==b
    finally:lib.close()


def test_dark_palette_follows_parent_theme_without_forcing_a_theme(real_gui):
    root,lib,_,_=real_gui
    from tkinter import ttk
    style=ttk.Style(root);previous=style.theme_use()
    dark='qa_dark_'+str(id(lib))
    style.theme_create(dark,parent='clam',settings={'TFrame':{'configure':{'background':'#222831'}}})
    try:
        style.theme_use(dark);pump(root)
        assert style.theme_use()==dark
        assert lib.colors['surface']=='#29343E'
        select_first(root,lib);dialog=lib.show_details();pump(root)
        assert dialog.summary.cget('background')==lib.colors['surface']
        assert dialog.raw.cget('foreground')==lib.colors['ink']
        dialog.close()
        assert lib.event_tree.tag_configure('danger')['background']==lib.colors['danger_bg']
    finally:
        style.theme_use(previous);pump(root)
