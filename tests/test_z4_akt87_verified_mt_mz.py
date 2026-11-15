# -*- coding: utf-8 -*-
"""AKT87 — operator approval and same-plate MT→MZ metrics, read-only."""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk

import pytest

from auto_annotation_tool.gui.mobile_compare_review_gate import (
    imported_verification, inspect_canvas, paired_geometry_text, Verification,
)
from auto_annotation_tool.gui.mobile_comparison_core import load_session_export, figures
from auto_annotation_tool.gui.mobile_import_comparison import export_imported_charts

SHA = "a"*64
APK = "b"*64
SID = "exp-akt87-test"
POLY = [[10.,10.],[110.,10.],[110.,30.],[10.,30.]]


def canvas_example():
    return dict(schema="alpr.mobile.canvas_review.v1",provenance="ASSISTED_CANVAS_SOURCE_ROI",
        independent_gt=False,source_archive_sha256=SHA,review_revision=7,
        scene_count=2,checked_scenes=2,excluded_scenes=0,pending_scenes=0,
        incomplete_checked_scenes=0,operator_errors=0,
        expected_plates=2,linked_plates=1,missed_plates=1,false_positive_mt=0,
        geometry_n=1,evaluated_reads=1,exact_reads=1,gt_characters=7,text_errors=0,
        scenes=[{"status":"CHECKED"},{"status":"CHECKED"}],
        rows=[
            dict(included=True,post_nms_suppressed_raw=False,outcome="LINKED_MT",
                image_entry="samples/image_01.jpg",object_id="GT:01",gt_text="ABC1234",
                mz_prediction="ABC1234",mz_status="READ",text_exclusion_reason="",
                polygon_iou_pct=95.0,nmce_pct=1.5,cer=0.0,
                original_mt_quad=POLY,operator_quad=POLY),
            dict(included=True,post_nms_suppressed_raw=False,outcome="MISSED_MT",
                image_entry="samples/image_02.jpg",object_id="MANUAL:02",gt_text="DEF4321",
                mz_prediction="",mz_status="NOT_RUN",text_exclusion_reason="",
                polygon_iou_pct=None,nmce_pct=None,cer=None,
                original_mt_quad=None,operator_quad=POLY),
        ])


def make_export(root: Path, *, canvas=None, rev=7, sid=SID, sha=SHA):
    canvas=canvas_example() if canvas is None else canvas
    canvas["source_archive_sha256"]=sha
    canvas["review_revision"]=rev
    folder=root/sid/f"review_r{rev:06d}_export_001"
    folder.mkdir(parents=True)
    review={"session_id":sid,"source_archive_sha256":sha,"review_revision":rev,
            "review_status":"COMPLETED","source_archive_path":f"C:\\tmp\\r01\\sample.alprsession",
            "provenance":{"model_provenance":{},"collection_session":{
                "app_build":{"apk_identity":{"base_apk_sha256":APK}}}}}
    sci={"session_id":sid,"source_archive_sha256":sha,"review_revision":rev,
         "review_status":"COMPLETED","scientific_state":"NOT_EVALUABLE",
         "conditions":{"scenario":"PAIR_DAY_01","device":"A12","android":"12","models":{
             "MT":{"precision":"fp32"},"MZ":{"precision":"fp32"}}},"performance":{}}
    data={"review.json":review,"canvas_review_summary.json":canvas,
          "ocena_naukowa_demo.json":sci}
    files=[]
    for name,payload in data.items():
        value=(json.dumps(payload,ensure_ascii=False,sort_keys=True)+"\n").encode('utf-8')
        (folder/name).write_bytes(value)
        files.append({"path":name,"bytes":len(value),"sha256":hashlib.sha256(value).hexdigest()})
    (folder/"export_manifest.json").write_text(json.dumps({
        "schema":"alpr_mobile_review_export_manifest_v1",
        "session_id":sid,"source_archive_sha256":sha,"review_revision":rev,
        "review_status":"COMPLETED","review_mode":"assisted", "files":files},indent=2),encoding='utf-8')
    return folder


def test_complete_mt_and_mz_required():
    c=canvas_example()
    ready,reason=inspect_canvas(c,'COMPLETED')
    assert ready and '18' not in reason
    assert len(list(paired_geometry_text(c)))==1
    assert not inspect_canvas(c,'IN_PROGRESS')[0]
    c['rows'][0]['operator_quad']=None
    assert not inspect_canvas(c,'COMPLETED')[0]
    c=canvas_example(); c['rows'][0]['cer']=None
    assert not inspect_canvas(c,'COMPLETED')[0]
    c=canvas_example(); c['pending_scenes']=1
    assert not inspect_canvas(c,'COMPLETED')[0]
    c=canvas_example(); c['rows'][1]['mz_status']='READ'
    assert not inspect_canvas(c,'COMPLETED')[0]


def test_mz_attempt_with_empty_prediction_is_scored_not_ignored():
    c=canvas_example()
    c['rows'][0].update(mz_status='NO_CHARACTERS', mz_prediction='', cer=1.0)
    c.update(exact_reads=0, text_errors=7, cer=1.0)
    ok, reason=inspect_canvas(c,'COMPLETED')
    assert ok, reason
    rows=list(paired_geometry_text(c))
    assert len(rows)==1 and rows[0]['cer_pct']==100.0 and rows[0]['exact'] is False
    # The Canvas review may also explicitly exclude illegible GT text while
    # retaining evaluated geometry. No artificial CER shall be invented.
    c['rows'][0].update(gt_text='', text_exclusion_reason='nieczytelny numer',cer=None)
    c.update(evaluated_reads=0,gt_characters=0,text_errors=0,cer=None)
    ok,reason=inspect_canvas(c,'COMPLETED')
    assert ok,reason
    assert list(paired_geometry_text(c))==[]
    c['rows'][0].update(gt_text='ABC1234',text_exclusion_reason='')
    assert not inspect_canvas(c,'COMPLETED')[0]


def test_source_pinned_manifest_and_revisions(tmp_path):
    root=tmp_path/'03_wyniki';root.mkdir()
    path=make_export(root)
    status=imported_verification(SID,SHA,root)
    assert status.ready and status.revision==7
    assert not imported_verification(SID,'d'*64,root).ready
    assert load_session_export(path).canvas_ready
    sidecar=tmp_path/'review_db'/'human_reviews';sidecar.mkdir(parents=True)
    (sidecar/(SHA+'.review.json')).write_text(json.dumps(dict(
        session_id=SID,source_archive_sha256=SHA,review_revision=8,review_status='IN_PROGRESS')),
        encoding='utf-8')
    stale=imported_verification(SID,SHA,root,tmp_path/'review_db')
    assert stale.state=='STALE' and not stale.ready
    (sidecar/(SHA+'.review.json')).write_text(json.dumps(dict(
        session_id=SID,source_archive_sha256=SHA,review_revision=7,review_status='COMPLETED')),
        encoding='utf-8')
    assert imported_verification(SID,SHA,root,tmp_path/'review_db').ready
    # Tampering is blocked, not treated as a newly measured low result.
    (path/'canvas_review_summary.json').write_text('{}',encoding='utf-8')
    assert imported_verification(SID,SHA,root,tmp_path/'review_db').state=='INVALID'


def test_approved_without_export_is_not_selectable(tmp_path):
    root=tmp_path/'03_wyniki';root.mkdir()
    sidecar=tmp_path/'review_db'/'human_reviews';sidecar.mkdir(parents=True)
    (sidecar/(SHA+'.review.json')).write_text(json.dumps(dict(
        session_id=SID,source_archive_sha256=SHA,review_revision=7,review_status='COMPLETED')),
        encoding='utf-8')
    assert imported_verification(SID,SHA,root,tmp_path/'review_db').state=='NEEDS_EXPORT'
    assert imported_verification(SID,SHA,root).state=='NONE'


def test_incomplete_geometry_rejected_even_with_fresh_manifest(tmp_path):
    root=tmp_path/'03_wyniki';root.mkdir()
    c=canvas_example();c['rows'][0]['original_mt_quad']=None
    make_export(root,canvas=c)
    status=imported_verification(SID,SHA,root)
    assert status.state=='INCOMPLETE' and not status.ready


def test_paired_geometry_figures_and_csv(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=tmp_path/'03_wyniki';root.mkdir();(root/'porownanie').mkdir()
    s=load_session_export(make_export(root))
    f=list(figures([s]))
    assert 'PAIR_DAY_01_assisted_iou_vs_cer_pairs' in [name for name,_ in f]
    assert 'PAIR_DAY_01_assisted_nmce_distribution' in [name for name,_ in f]
    for _,fig in f:plt.close(fig)
    dest=export_imported_charts([s],[[]],root)
    csv_path=dest/'geometria_MT_a_odczyt_MZ.csv'
    assert csv_path.exists()
    lines=csv_path.read_text(encoding='utf-8-sig').splitlines()
    assert len(lines)==2  # header and single same-plate MT/MZ pair; manual FN excluded
    assert 'GT:01' in lines[1] and '95.0' in lines[1]
    manifest=json.loads((dest/'wykresy_manifest.json').read_text(encoding='utf-8'))
    assert len(manifest['files']) >= 1
    assert all(hashlib.sha256((dest/r['name']).read_bytes()).hexdigest()==r['sha256'] for r in manifest['files'])
    s2=replace(s,session_id='exp-akt87-second',source_sha256='c'*64,precision='int8')
    assert not imported_verification(s2.session_id,s2.source_sha256,root).ready
    combo=export_imported_charts([s,s2],[[],[]],root)
    assert combo.parent==root/'porownanie'
    with pytest.raises(ValueError,match='scenariusze'):
        export_imported_charts([s,replace(s2,scenario='PAIR_NIGHT_01')],[[],[]],root)
    with pytest.raises(ValueError,match='zatwierdzone'):
        export_imported_charts([s,replace(s2,review_status='IN_PROGRESS')],[[],[]],root)


def test_native_identity_checkbox_click_keeps_single_report_preview():
    try:
        root=tk.Tk()
    except tk.TclError:
        pytest.skip('Brak ekranu Tk')
    try:
        from tkinter import ttk
        from auto_annotation_tool.gui.mobile_compare_review_gate import Verification
        browser_path=Path(__file__).resolve().parents[1]/'auto_annotation_tool/gui/z4_mobile_report_browser.py'
        parsed=ast.parse(browser_path.read_text(encoding='utf-8'))
        browser=next(n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name=='MobileReportBrowser')
        selected={'_on_report_selected','_comparison_key','_toggle_comparison_checkbox',
                  '_update_comparison_basket_status','_clear_comparison_checks'}
        short=ast.ClassDef(name='Short',bases=[],keywords=[],body=[x for x in browser.body
            if isinstance(x,ast.FunctionDef) and x.name in selected],decorator_list=[])
        env={'Verification':Verification,'__name__':'auto_annotation_tool.gui._checkbox_test','__package__':'auto_annotation_tool.gui'}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[short],type_ignores=[])),'<short>','exec'),env)
        widget=ttk.Treeview(root,columns=('Wybor','Data'),show='headings',selectmode='browse',height=3)
        widget.heading('Wybor',text='✓');widget.heading('Data',text='Dane')
        widget.column('Wybor',width=60);widget.column('Data',width=150)
        widget.insert('',tk.END,iid='row_0',values=('☐','foo'))
        widget.insert('',tk.END,iid='row_1',values=('☐','bar'))
        widget.pack();root.update()
        a=SimpleNamespace(report_id='one',source_archive_sha256=SHA,experiment_index={'session_id':'one'})
        b=SimpleNamespace(report_id='two',source_archive_sha256='c'*64,experiment_index={'session_id':'two'})
        obj=env['Short']();obj._comparison_checked=set();obj._comparison_busy=False
        obj.report_table=SimpleNamespace(tree=widget)
        obj.report_by_iid={'row_0':a,'row_1':b}
        obj._comparison_verification={(a.report_id,SHA):Verification('READY','Zatwierdzona','kompletne'),
               (b.report_id,b.source_archive_sha256):Verification('IN_PROGRESS','W toku','nieukończone')}
        obj.btn_compare_open=ttk.Button(root)
        msgs=[];opened=[]
        obj._set_status=lambda *args:msgs.append(args)
        obj._matching_cached_bundle=lambda r:None
        obj._show_report=lambda r,b:opened.append(r.report_id)
        obj._suppress_report_select=False
        widget.selection_set('row_0');obj._on_report_selected()
        def click_row(iid):
            bbox=widget.bbox(iid,'Wybor')
            return obj._toggle_comparison_checkbox(SimpleNamespace(x=bbox[0]+5,y=bbox[1]+5))
        assert click_row('row_0')=='break'
        assert widget.set('row_0','Wybor')=='☑' and len(obj._comparison_checked)==1
        assert click_row('row_1')=='break'
        # ALPR-ANALYSIS-1.0: IN_PROGRESS is selectable for analysis, not approved quality.
        assert widget.set('row_1','Wybor')=='☑' and len(obj._comparison_checked)==2
        assert opened==['one']  # checkbox click did not open a different report
        obj._clear_comparison_checks()
        assert widget.set('row_0','Wybor')=='☐'
        assert widget.set('row_1','Wybor')=='☐'
    finally:
        root.destroy()
