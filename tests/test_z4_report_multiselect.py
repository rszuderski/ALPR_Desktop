"""Native Tk report list gestures; no catalog, review or source writes."""
from types import SimpleNamespace as NS
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import pytest
from auto_annotation_tool.gui import z4_mobile_report_browser as module
from auto_annotation_tool.gui.mobile_report_selection import ReportListSelection
from auto_annotation_tool.gui.mobile_compare_review_gate import Verification


@pytest.fixture
def gui(tmp_path,monkeypatch):
    root=tk.Tk();root.geometry('850x550+20+20');errors=[]
    root.report_callback_exception=lambda *a:errors.append(a)
    b=object.__new__(module.MobileReportBrowser)
    b._comparison_checked=set();b._comparison_busy=False;b._suppress_report_select=False;b.current_report=None
    b._report_sort_column='Data';b._report_sort_descending=False
    b.report_table=module._TreeTable(root,('Wybor','Data','Wariant','Scena','Powt.','Weryfikacja','p95'),
        ('✓','Data','Wariant','Materiał','Powt.','Weryfikacja','P95'),(30,150,100,120,50,140,65),palette={},height=8,selectmode='extended')
    b.report_table.shell.pack(fill='both',expand=True)
    b.search_var=tk.StringVar();b.report_count_var=tk.StringVar();b.btn_compare_open=ttk.Button(root);b.btn_compare_open.pack()
    b.messages=[];b.opened=[];b._matching_cached_bundle=lambda r:None
    b._set_status=lambda *a:b.messages.append(a)
    def show(r,bundle):b.current_report=r;b.opened.append(r.report_id)
    b._show_report=show;b._show_report_hint=lambda *a:None;b._show_empty=lambda:None
    reports=[]
    for i in range(7):
        p=tmp_path/f'source_{i}.alprsession';p.write_bytes(b'Not read by list selection')
        reports.append(NS(report_id=f'report{i}',source_archive_sha256=f'{i+1:064x}',source_path=str(p),
            variant_id='DAY' if i%2==0 else 'NIGHT',package_id='TEST',measured_at=f'2026-11-01T12:0{i}:00+01:00',
            experiment_index={'session_id':f'session{i}','scenario_id':f'SCENE{i}','measured_at':f'2026-11-01T12:0{i}:00+01:00'},raw={},device={}))
    b._ordered_reports=reports;b.report_by_iid={}
    b._comparison_verification={b._comparison_key(r):Verification('READY','Zatwierdzona','fixture') for r in reports}
    monkeypatch.setattr(module,'_pipeline_p95',lambda r:10)
    monkeypatch.setattr(module,'SESSION',None)
    b._apply_report_filter(auto_select=True)
    tree=b.report_table.tree
    tree.bind('<<TreeviewSelect>>',b._on_report_selected)
    b._report_list_selection=ReportListSelection(b)
    root.update();tree.focus_force();root.update()
    try:yield root,b,tree
    finally:
        root.destroy()
        assert not errors,errors


def click(gui,index,mod=0,column='Data'):
    root,b,t=gui;iid=t.get_children()[index];t.see(iid);root.update();box=t.bbox(iid,column)
    x,y=box[0]+max(3,box[2]//2),box[1]+box[3]//2
    t.event_generate('<ButtonPress-1>',x=x,y=y,state=mod)
    t.event_generate('<ButtonRelease-1>',x=x,y=y,state=mod);root.update()
    return iid


def key(gui,name,mod=0):
    root,b,t=gui;t.focus_force();root.update();t.event_generate('<KeyPress>',keysym=name,state=mod)
    t.event_generate('<KeyRelease>',keysym=name,state=mod);root.update()


def selected(gui):
    root,b,t=gui;return {b.report_by_iid[i].report_id for i in t.selection()}


def test_ctrl_click_toggles_rows_without_checking_basket_and_previews_focus(gui):
    click(gui,1);click(gui,4,4)
    assert selected(gui)=={'report1','report4'} and gui[1].current_report.report_id=='report4'
    click(gui,1,4);assert selected(gui)=={'report4'} and not gui[1]._comparison_checked


def test_shift_range_anchor_stays_fixed_and_ctrl_shift_adds(gui):
    click(gui,1);click(gui,4,1);assert selected(gui)=={'report1','report2','report3','report4'}
    click(gui,2,1);assert selected(gui)=={'report1','report2'}
    click(gui,6,4);click(gui,5,5);assert selected(gui)=={'report1','report2','report5','report6'}


def test_space_sets_mixed_group_then_clears_it_without_changing_preview(gui):
    root,b,t=gui;click(gui,1,column='Wybor');click(gui,1);click(gui,3,1);opened=list(b.opened)
    key(gui,'space');assert len(b._comparison_checked)==3 and all(t.set(i,'Wybor')=='☑' for i in t.selection())
    key(gui,'space');assert not b._comparison_checked and b.opened==opened
    assert selected(gui)=={'report1','report2','report3'}


def test_modifier_click_in_checkbox_column_selects_range_without_toggling(gui):
    click(gui,1);click(gui,3,1,'Wybor');assert selected(gui)=={'report1','report2','report3'}
    assert not gui[1]._comparison_checked


def test_sort_keeps_group_and_anchor_but_ranges_follow_visible_order(gui):
    root,b,t=gui;click(gui,1);click(gui,3,4);before=selected(gui)
    b._sort_reports('Data');root.update();assert selected(gui)==before
    assert b.current_report.report_id=='report3'
    # Descending order: row0 is report6; anchor is last Ctrl-click report3.
    click(gui,0,1);assert selected(gui)=={'report3','report4','report5','report6'}
    key(gui,'space');assert len(b._comparison_checked)==4


def test_filter_keeps_visible_group_and_hidden_basket_checks(gui):
    root,b,t=gui;key(gui,'a',4);key(gui,'space');assert len(b._comparison_checked)==7
    b.search_var.set('DAY');b._apply_report_filter();root.update()
    assert selected(gui)=={'report0','report2','report4','report6'}
    key(gui,'space');assert {k[0] for k in b._comparison_checked}=={'report1','report3','report5'}
    b.search_var.set('');b._apply_report_filter();root.update()
    assert selected(gui)=={'report0','report2','report4','report6'}
    assert len(b._comparison_checked)==3


def test_keyboard_range_and_ctrl_a_space(gui):
    click(gui,2);key(gui,'Down',1);key(gui,'Down',1);assert selected(gui)=={'report2','report3','report4'}
    key(gui,'Up',1);assert selected(gui)=={'report2','report3'}
    key(gui,'a',4);assert len(selected(gui))==7
    key(gui,'space');assert len(gui[1]._comparison_checked)==7
    click(gui,0);assert selected(gui)=={'report0'} and len(gui[1]._comparison_checked)==7


def test_space_with_no_selection_is_readonly_and_does_not_check_focus(gui):
    click(gui,0);click(gui,0,4);assert not selected(gui)
    key(gui,'space');assert not gui[1]._comparison_checked
    gui[1]._sort_reports('Data');gui[0].update();assert not selected(gui)


def test_invalid_identity_and_duplicate_sources_cannot_enter_group_basket(gui):
    root,b,t=gui;r=b._ordered_reports
    r[2].source_archive_sha256=r[1].source_archive_sha256
    r[3].experiment_index['session_id']=r[1].experiment_index['session_id']
    r[4].source_archive_sha256='invalid'
    key(gui,'a',4);key(gui,'space')
    assert {k[0] for k in b._comparison_checked}=={'report0','report1','report5','report6'}
    assert 'Pominięte wiersze: 3' in b.messages[-1][0]
    key(gui,'space');assert not b._comparison_checked


def test_plain_checkbox_remains_single_row_and_does_not_change_preview(gui):
    root,b,t=gui;click(gui,0);opened=list(b.opened);click(gui,3,column='Wybor')
    assert selected(gui)=={'report0'} and b.opened==opened and {k[0] for k in b._comparison_checked}=={'report3'}


def test_range_after_anchor_filtered_out_cannot_select_invisible_rows(gui):
    root,b,t=gui;click(gui,1);b.search_var.set('DAY');b._apply_report_filter();root.update()
    click(gui,2,1);key(gui,'space')
    assert {k[0] for k in b._comparison_checked}=={'report4'}


def test_refresh_and_explicit_import_selection_have_separate_intent(gui):
    root,b,t=gui;click(gui,1);click(gui,4,1);b._apply_report_filter(refresh_preview=True);root.update()
    assert selected(gui)=={'report1','report2','report3','report4'}
    b._apply_report_filter(select_report_id='report6');root.update();assert selected(gui)=={'report6'}
