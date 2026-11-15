# -*- coding: utf-8 -*-
"""FIX03/FIX04: row visibility and preservation of list state with modal details.

Synthetic local archives only. No model inference, Android or measurement data.
Unlike FIX02 these regressions count complete data rows, not just a nonempty bbox.
"""
from __future__ import annotations
import copy
import gc
from pathlib import Path
import time
from types import SimpleNamespace
import pytest
from test_z4lib_history_sources_20261020 import package, check
from auto_annotation_tool.gui import z4_mobile_artifact_registry as registry
from auto_annotation_tool.gui.z4_mobile_artifact_library import MobileArtifactLibrary


def pump(root, rounds=6):
    for _ in range(rounds):
        root.update()
        time.sleep(.01)


def whole_rows(tree):
    return [iid for iid in tree.get_children() if (box := tree.bbox(iid))
            and box[1] >= 0 and box[1] + box[3] <= tree.winfo_height() - 1]


@pytest.fixture
def gui(tmp_path):
    import tkinter as tk
    gc.collect()
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('Tk display unavailable')
    root.withdraw()
    initial_scaling = float(root.tk.call('tk', 'scaling'))
    root.tk.call('tk', 'scaling', 1.5)
    store = tmp_path / 'mobile_packages'
    for n in range(12):
        path = package(store, f'SYNTHETIC_{n:02d}.alprmodel')
        if n < 2:
            check(store, path)
    for n in range(2):
        (store / f'UNKNOWN_{n}.alprmodel').write_bytes(b'synthetic unsupported data')
    research = tmp_path / registry.RESEARCH_REL
    research.parent.mkdir(parents=True, exist_ok=True)
    research.write_bytes(b'synthetic research entry, not a real model')
    lib = MobileArtifactLibrary(root, packages_root=store, repo_root=tmp_path)
    deadline = time.monotonic() + 6
    while lib.busy and time.monotonic() < deadline:
        pump(root, 1)
    assert not lib.busy
    lib.window.geometry('1649x870+0+0')
    lib.notebook.select(lib.events_tab)
    pump(root)
    yield root, lib, tmp_path
    lib.close()
    root.tk.call('tk', 'scaling', initial_scaling)
    root.destroy()
    # Release this test's Tcl interpreter on the main thread before a later
    # library scan starts another worker. No test assertion is removed.
    del lib
    del root
    gc.collect()


def test_history_shows_at_least_eight_complete_rows_before_any_selection(gui):
    root, lib, _ = gui
    if lib.window.winfo_screenwidth() < 1700 or lib.window.winfo_screenheight() < 970:
        pytest.skip('Large-screen layout regression needs at least 1700x970')
    assert len(lib.events_by_iid) == 14
    assert len(whole_rows(lib.event_tree)) >= 8


def test_selected_history_keeps_four_rows_and_readable_details(gui):
    root, lib, _ = gui
    if lib.window.winfo_screenwidth() < 1700 or lib.window.winfo_screenheight() < 970:
        pytest.skip('Large-screen layout regression needs at least 1700x970')
    lib.event_tree.selection_set(lib.event_tree.get_children()[0])
    lib.event_selected(); pump(root)
    assert len(whole_rows(lib.event_tree)) >= 4
    assert lib._details_dialog is None
    dialog=lib.show_details();pump(root)
    assert dialog.summary.winfo_ismapped()
    assert dialog.summary.winfo_height() >= 4 * lib._font_body.metrics('linespace')
    dialog.close()


def test_empty_selection_does_not_reserve_an_inspector(gui):
    root, lib, _ = gui
    assert not hasattr(lib, 'details_frame') and not hasattr(lib, 'split')
    assert lib._details_dialog is None
    assert lib.details_button.instate(['disabled'])
    assert lib.event_tree.winfo_height() > 200


def test_filter_removing_selection_clears_pending_details(gui):
    root, lib, _ = gui
    lib.event_tree.selection_set(lib.event_tree.get_children()[0]); lib.event_selected(); pump(root)
    assert lib._detail_json.strip() and lib._details_dialog is None
    lib.filter_var.set('no matching source'); pump(root)
    assert lib._details_dialog is None and not lib._detail_json.strip()
    assert lib.details_button.instate(['disabled'])
    assert 'Wiersze po filtrze: 0 / 14' in lib.event_view_status.get()


@pytest.mark.parametrize('size,scale,selected', [
    ('940x570', 1.333, False), ('940x570', 1.333, True),
    ('1280x720', 1.667, False), ('1280x720', 1.667, True),
    ('1649x870', 2.0, False), ('1649x870', 2.0, True),
])
def test_resize_and_scale_keep_complete_rows_and_footer(gui, size, scale, selected):
    from tkinter import ttk
    root, lib, _ = gui
    root.tk.call('tk','scaling',scale)
    lib._font_body.configure(size=10); lib._font_bold.configure(size=10); lib._font_title.configure(size=12)
    lib._apply_palette()
    lib.window.geometry(size + '+0+0')
    if selected:
        lib.event_tree.selection_set(lib.event_tree.get_children()[0]); lib.event_selected()
    pump(root, 10)
    assert len(whole_rows(lib.event_tree)) >= 1
    assert lib._details_dialog is None
    assert lib.event_tree.winfo_height() >= lib._font_body.metrics('linespace') + 8
    for widget in [*lib.action_buttons, lib.close_button, lib.status_label]:
        assert widget.winfo_ismapped()
        top = widget.winfo_rooty()-lib.window.winfo_rooty()
        assert top >= 0 and top + widget.winfo_height() <= lib.window.winfo_height()
    for button in lib.action_buttons:
        assert root.winfo_containing(button.winfo_rootx()+button.winfo_width()//2,
                                    button.winfo_rooty()+button.winfo_height()//2) == button


def test_resize_from_small_to_large_does_not_leave_small_table(gui):
    root, lib, _ = gui
    if lib.window.winfo_screenwidth() < 1700 or lib.window.winfo_screenheight() < 970:
        pytest.skip('Large-screen layout regression needs at least 1700x970')
    lib.window.geometry('940x570+0+0'); pump(root)
    lib.window.geometry('1649x870+0+0'); pump(root)
    assert len(whole_rows(lib.event_tree)) >= 8


def test_selected_table_height_does_not_change_on_filter_or_resize(gui):
    root, lib, _ = gui
    wanted=lib.event_tree.winfo_height()
    lib.event_tree.selection_set(lib.event_tree.get_children()[0]);lib.event_selected();pump(root)
    assert lib.event_tree.winfo_height()==wanted
    lib.filter_var.set('SYNTHETIC');pump(root)
    assert lib.event_tree.winfo_height()==wanted
    lib.window.geometry('940x570+0+0');pump(root)
    lib.window.geometry('1649x870+0+0');pump(root)
    assert lib.event_tree.winfo_height()==wanted


def test_tabs_preserve_separate_choices_without_automatic_selection(gui):
    root, lib, _ = gui
    lib.notebook.select(lib.packages_tab);pump(root)
    assert not lib.art_tree.selection() and lib._details_dialog is None
    iid=lib.art_tree.get_children()[1];lib.art_tree.selection_set(iid);pump(root)
    lib.notebook.select(lib.events_tab);pump(root)
    assert not lib.event_tree.selection() and lib._details_dialog is None
    lib.notebook.select(lib.packages_tab);pump(root)
    assert lib.art_tree.selection()==(iid,)


def test_hide_restore_preserves_table_size_and_scroll(gui):
    root, lib, _ = gui
    wanted=lib.event_tree.winfo_height();lib.event_tree.yview_moveto(1);pump(root)
    scroll=lib.event_tree.yview()
    lib.window.withdraw();pump(root)
    lib.window.deiconify();pump(root)
    assert abs(lib.event_tree.winfo_height()-wanted)<=2
    assert lib.event_tree.yview()==scroll


def test_scroll_to_last_record_is_not_reset_by_relayout(gui):
    root, lib, _ = gui
    lib.event_tree.yview_moveto(1); pump(root)
    wanted = lib.event_tree.yview()
    lib._queue_layout(); pump(root)
    assert lib.event_tree.yview() == wanted
    assert lib.event_tree.get_children()[-1] in whole_rows(lib.event_tree)


def test_resize_filter_selection_do_not_change_data_or_history(gui):
    root, lib, repo = gui
    def snapshot():
        return {str(p.relative_to(repo)):p.read_bytes() for p in repo.rglob('*') if p.is_file()}
    before = snapshot(); records = copy.deepcopy(lib.history); checks = copy.deepcopy(lib._desktop_check_events)
    for choice in ['Kontrole PC','Pakiety z dysku','Eksporty','Wszystkie źródła']:
        lib.history_source_filter.set(choice); lib.refresh_views(); pump(root)
    lib.event_tree.selection_set(lib.event_tree.get_children()[0]); lib.event_selected(); pump(root)
    lib.window.geometry('1100x750+0+0'); pump(root)
    assert snapshot() == before
    assert lib.history == records and lib._desktop_check_events == checks
    assert not registry.history_file(lib.packages_root).exists()


def test_counts_explain_three_different_populations(gui):
    root, lib, _ = gui
    assert lib.history_counts == dict(export_operations=0, export_events=0, pc_checks=2, unlogged_packages=12)
    assert len(lib.rows)==15 and len(lib.events_by_iid)==14
    assert '15 plików' in lib.status.get()
    assert 'modele/pakiety: 12' in lib.status.get() and 'pozostałe: 3' in lib.status.get()
    assert 'Wpisy historii, nie liczba plików' in lib.event_view_status.get()


def test_disk_row_says_no_export_entry_not_missing_whole_ledger(gui):
    from auto_annotation_tool.gui.z4_mobile_history_presentation import SOURCE_FILE
    root, lib, _ = gui
    entries = [lib.event_tree.item(i, 'values') for i,e in lib.events_by_iid.items()
               if e['_history_origin']==SOURCE_FILE]
    assert len(entries)==12
    assert all(row[3]=='Brak wpisu eksportu' for row in entries)
    lib.filter_var.set('Brak wpisu eksportu'); pump(root)
    assert len(lib.events_by_iid)==12


@pytest.mark.parametrize('selected,preference', [(False,None),(True,None),(False,.1),(True,.99),(True,.7)])
def test_pure_layout_is_bounded_without_io(selected, preference):
    from auto_annotation_tool.gui.z4_mobile_library_view import library_split_position
    for h in [300,450,641,850]:
        value=library_split_position(h,120,30,90,selected=selected,preference=preference)
        assert isinstance(value,int) and 0 <= value < h
        assert h-8-value >= 90
        assert value>=150


@pytest.mark.parametrize('kinds,expected', [
    ([], 'Biblioteka: 0 plików · rozpoznane modele/pakiety: 0 · pozostałe: 0'),
    (['MODEL','PACKAGE'], 'Biblioteka: 2 plików · rozpoznane modele/pakiety: 2 · pozostałe: 0'),
    (['MODEL','PACKAGE','RESEARCH_TFLITE','INVALID'], 'Biblioteka: 4 plików · rozpoznane modele/pakiety: 2 · pozostałe: 2'),
])
def test_inventory_counts_are_not_history_counts(kinds, expected):
    from auto_annotation_tool.gui.z4_mobile_library_view import inventory_caption
    rows=[SimpleNamespace(kind=k) for k in kinds]
    assert inventory_caption(rows).startswith(expected)
