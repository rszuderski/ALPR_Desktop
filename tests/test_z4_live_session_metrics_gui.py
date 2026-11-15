"""Real Windows Tk widgets/bindings; screenshots are separate acceptance evidence."""
from pathlib import Path
import copy,hashlib,json,shutil
import tkinter as tk
from types import SimpleNamespace
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_mt_detection_evaluation_gui import detection_gui
from test_z4_frame_stacks import ready
from test_mobile_p2b2_canvas_review import MANUAL
from auto_annotation_tool.ranking import mobile_session_metrics as live
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.gui.mobile_canvas_stacks import view,write_reference

G='mt-one'
OUT=Path(__file__).resolve().parents[1]/'output/Z4_LIVE_SESSION_METRICS_20261026_001'

def refreshed(root,h,ed):
    native.pump(root,lambda:live.current(h.stats.get('live_session_metrics_v1'),live.result_key(h.session))
        and not ed.in_refresh and not ed.refresh_after and not h._saving,seconds=20)
    ready(root,h,ed)
    expected=live.presentation(h.stats['live_session_metrics_v1'],live.result_key(h.session))
    assert ed.contexts.live_metrics.data==expected
    for k,v in expected['values'].items():assert ed.contexts.live_metrics.cards[k].value.cget('text')==v
    return h.stats['live_session_metrics_v1']

def test_L21_space_ppm_undo_redo_automatically_refresh_current_session(detection_gui):
    root,h,ed,s=detection_gui;refreshed(root,h,ed)
    ed._choose_plate('GT:a0');native.key(h,'space')
    assert set(ed.contexts.live_metrics.data['values'].values())=={'—'}
    assert refreshed(root,h,ed)['mt']['tp'] is None
    native.key(h,'space');assert refreshed(root,h,ed)['mt']['tp']==2
    native.click(h,ed,(445,175),3)
    snapshot=refreshed(root,h,ed)
    assert (snapshot['mt']['tp'],snapshot['mt']['fp'],snapshot['mt']['fn'])==(1,1,0)
    assert ed.contexts.live_metrics.cards['precision'].value.cget('text')=='50,0%'
    ed.undo();assert refreshed(root,h,ed)['mt']['tp']==2
    ed.undo(redo=True);assert refreshed(root,h,ed)['mt']['fp']==1
    assert ed.report_host.master.index('current')==0

def test_L12_L21_D_updates_FN_without_an_ocr_read(detection_gui):
    root,h,ed,s=detection_gui;before=refreshed(root,h,ed)
    native.key(h,'d')
    for p in MANUAL:native.click(h,ed,p)
    after=refreshed(root,h,ed)
    assert (after['mt']['tp'],after['mt']['fn'])==(2,1)
    assert after['ocr']['n_reads']==before['ocr']['n_reads']==0
    assert ed.contexts.live_metrics.cards['recall'].value.cget('text')=='66,7%'
    ed.undo();assert refreshed(root,h,ed)['mt']['fn']==0

def test_L21_saved_gt_text_and_geometry_refresh_independently(detection_gui):
    root,h,ed,s=detection_gui;before=refreshed(root,h,ed)['mt']
    for oid,text in (('GT:a0','A'),('GT:a1','WA11234')):
        ed.start_entry(oid);ed.entry_var.set(text);ed.entry_enter();refreshed(root,h,ed)
    snapshot=h.stats['live_session_metrics_v1']
    assert snapshot['ocr']['n_reads']==2 and snapshot['geometry']['n_geometry']==2
    assert snapshot['mt']==before
    assert ed.contexts.live_metrics.exact.cget('text')=='Exact MZ: 1/2'
    quad=copy.deepcopy(view(ed)['plates'][view(ed)['candidates'][0]['plate_id']]['gt_quad'])
    quad[0][0]+=15
    ed._commit(lambda:write_reference(ed,'GT:a0',quad=quad),'Test saved correction')
    updated=refreshed(root,h,ed)
    assert updated['geometry']['polygon_iou_mean_pct']<snapshot['geometry']['polygon_iou_mean_pct']
    assert updated['ocr']==snapshot['ocr']

def test_L22_drag_keeps_object_preview_and_aggregates_only_after_save(detection_gui,monkeypatch):
    root,h,ed,s=detection_gui;refreshed(root,h,ed)
    ed._choose_plate('GT:a0');ed.mode='edit';ready(root,h,ed)
    calls=[];original=s.statistics
    def counted():calls.append(s.review.review_revision);return original()
    monkeypatch.setattr(s,'statistics',counted)
    frozen=s.sidecar_path.read_bytes();revision=s.review.review_revision
    quad=copy.deepcopy(core.scene(s,G)['objects'][0]['quad'])
    x,y=ed.to_canvas(quad[0]);h.canvas.event_generate('<ButtonPress-1>',x=round(x),y=round(y));native.pump(root)
    assert ed.drag is not None
    for dx in (4,8,12):
        h.canvas.event_generate('<B1-Motion>',x=round(x+dx),y=round(y+3));native.pump(root)
    assert not calls and s.sidecar_path.read_bytes()==frozen and s.review.review_revision==revision
    assert h.canvas.find_withtag('corner_handle')
    h.canvas.event_generate('<ButtonRelease-1>',x=round(x+12),y=round(y+3))
    snapshot=refreshed(root,h,ed)
    assert calls and s.review.review_revision==revision+1
    assert snapshot['review_revision']==s.review.review_revision

def test_L23_late_worker_cannot_overwrite_newer_values(detection_gui):
    root,h,ed,s=detection_gui;refreshed(root,h,ed)
    old=copy.deepcopy(h.stats)
    native.click(h,ed,(445,175),3);refreshed(root,h,ed)
    new=h.stats
    h._changed((old,'',None))
    assert h.stats is new
    assert ed.contexts.live_metrics.cards['precision'].value.cget('text')=='50,0%'
    refreshed(root,h,ed)

def test_L21_image_exclusion_and_restore_update_without_export(detection_gui):
    from auto_annotation_tool.gui import mobile_canvas_clean_view as clean
    root,h,ed,s=detection_gui;assert refreshed(root,h,ed)['mt']['n_qualified']==1
    clean.show_exclusion(ed);ed.clean_reason.set('Synthetic acceptance exclusion')
    assert clean.save_exclusion(ed)
    assert refreshed(root,h,ed)['mt']['n_qualified']==0
    assert ed.contexts.live_metrics.cards['precision'].value.cget('text')=='—'
    ed.clean_restore_button.invoke()
    assert refreshed(root,h,ed)['mt']['n_qualified']==1

def test_L24_view_selection_tabs_zoom_and_scroll_are_read_only(detection_gui,monkeypatch):
    root,h,ed,s=detection_gui;refreshed(root,h,ed)
    frozen=s.sidecar_path.read_bytes();revision=s.review.review_revision
    monkeypatch.setattr(s,'statistics',lambda:pytest.fail('view recomputed session statistics'))
    for key in ('Tab','ISO_Left_Tab'):
        ed._choose_plate('GT:a0');native.key(h,key)
    h._view_zoom=1.3;h._paint_image();native.pump(root)
    h.review_sidebar_scroll.yview_moveto(1);native.pump(root)
    for tab in (1,0):ed.report_host.master.select(tab);native.pump(root);ed.update_report()
    ed.sync();ed.overlay();native.pump(root)
    scroll=h.review_sidebar_scroll;scroll.yview_moveto(0);native.pump(root)
    x,y=scroll.winfo_rootx()+10,scroll.winfo_rooty()+10
    monkeypatch.setattr(h.window,'winfo_pointerxy',lambda:(x,y))
    assert ed.route_scroll(SimpleNamespace(delta=-120,num=0))=='break'
    assert scroll.yview()[0]>0 and h._view_zoom==1.3
    y=scroll.winfo_rooty()+scroll.winfo_height()+20
    position=scroll.yview()
    assert ed.route_scroll(SimpleNamespace(delta=-120,num=0)) is None
    assert scroll.yview()==position
    assert s.sidecar_path.read_bytes()==frozen and s.review.review_revision==revision

@pytest.mark.parametrize('detection_gui',['light','dark'],indirect=True)
def test_L25_L26_existing_scroll_two_rows_and_unchanged_outer_columns(detection_gui):
    root,h,ed,s=detection_gui;refreshed(root,h,ed)
    theme='light' if ed.p['bg']=='#F2F5F8' else 'dark'
    baseline=json.loads((OUT/f'native_before_{theme}.json').read_text(encoding='utf-8'))
    measurements=[];h.window.maxsize(3000,2400)
    for expected in baseline:
        width,height=expected['size'];h.window.geometry(f'{width}x{height}+10+10')
        native.pump(root,lambda:(h.window.winfo_width(),h.window.winfo_height())==(width,height))
        ed._choose_plate('GT:a0');ready(root,h,ed)
        origin=(h.window.winfo_rootx(),h.window.winfo_rooty())
        def bounds(w):return [w.winfo_rootx()-origin[0],w.winfo_rooty()-origin[1],w.winfo_width(),w.winfo_height()]
        assert bounds(h.canvas)==expected['canvas']
        assert bounds(h.subject_tree)==expected['tree']
        h.review_sidebar_scroll.yview_moveto(0);native.pump(root)
        for group,keys in (('image_cards',('gt','mt','fp','fn')),
                           ('progress',('inspected','remaining','decisions'))):
            assert {k:bounds(ed.contexts.counters[k]) for k in keys}==expected[group]
        assert [ed.report_host.master.tab(t,'text') for t in ed.report_host.master.tabs()]==expected['tabs']
        assert abs(h.window.winfo_fpixels('1i')-120)<1
        assert h.review_sidebar_scroll.winfo_ismapped()
        assert [ed.contexts.counter_labels[k].cget('text') for k in ('inspected','remaining','decisions')]==[
            'Sprawdzone','Pozostało','Pozostałe decyzje']
        metrics=ed.contexts.live_metrics
        if width==1649:
            # At the larger size all six values fit without scrolling. At the
            # smaller size the same existing sidebar scroll gives access.
            h.review_sidebar_scroll.yview_moveto(0);native.pump(root)
            edge=h.review_sidebar_scroll.winfo_rooty()+h.review_sidebar_scroll.winfo_height()
            assert all(c.value.winfo_rooty()+c.value.winfo_height()<=edge for c in metrics.cards.values()),dict(
                edge=edge,metrics=bounds(metrics),status=bounds(metrics.status),
                cards={k:dict(card=bounds(c),title=bounds(c.title),value=bounds(c.value)) for k,c in metrics.cards.items()})
        h.review_sidebar_scroll.yview_moveto(1);native.pump(root)
        top=h.review_sidebar_scroll.winfo_rooty();bottom=top+h.review_sidebar_scroll.winfo_height()
        for key,card in metrics.cards.items():
            assert card.winfo_rooty()>=top and card.winfo_rooty()+card.winfo_height()<=bottom
            assert int(card.grid_info()['row'])==(0 if key in ('precision','recall','f1') else 1)
            assert card.value.winfo_reqwidth()<=card.value.winfo_width()
            assert str(card.cget('takefocus'))=='0'
        assert metrics.exact.winfo_rooty()+metrics.exact.winfo_height()<=bottom
        assert float(h.review_sidebar_scroll.xview()[0])==0
        assert metrics.winfo_width()<=ed.contexts.winfo_width()
        measurements.append(dict(size=[width,height],dpi=h.window.winfo_fpixels('1i'),
            canvas=bounds(h.canvas),tree=bounds(h.subject_tree),metric_block=bounds(metrics),
            viewport=bounds(h.review_sidebar_scroll),scroll=list(h.review_sidebar_scroll.yview()),
            row_heights=[metrics.cards[k].winfo_height() for k in ('precision','iou')]))
        # The earlier M16 test writes into its historical report directory.
        # Preserve that report and repeat its native assertions here instead.
        panel=ed.report_host;panel.master.select(panel);native.pump(root)
        rate_cards=[panel.cards[k] for k in ('precision','recall','f1')]
        assert all(c.winfo_ismapped() and c.value.cget('text')=='100,0%' for c in rate_cards)
        assert {int(c.grid_info()['row']) for c in rate_cards}=={0}
        assert [int(c.grid_info()['column']) for c in rate_cards]==[0,1,2]
        assert panel.mt_rates_frame.winfo_height()<140
        assert all(str(c.cget('takefocus'))=='0' for c in rate_cards)
        panel.master.select(0);native.pump(root)
    (OUT/f'native_after_{theme}.json').write_text(json.dumps(measurements,indent=2),encoding='utf-8')

def test_L27_panel_current_results_and_existing_gui_export_agree(detection_gui,tmp_path,monkeypatch):
    from auto_annotation_tool.gui import z4_mobile_sample_review as module
    root,h,ed,s=detection_gui;refreshed(root,h,ed)
    base=tmp_path/'03_wyniki';base.mkdir()
    monkeypatch.setattr(module.filedialog,'askdirectory',lambda **kw:str(base))
    monkeypatch.setattr(module.messagebox,'askyesno',lambda *a,**kw:True)
    errors=[];monkeypatch.setattr(module.messagebox,'showerror',lambda *a,**kw:errors.append(a))
    frozen=s.sidecar_path.read_bytes();revision=s.review.review_revision
    h.export();native.pump(root,lambda:not h._saving,seconds=45);refreshed(root,h,ed)
    assert not errors
    path=next(base.glob('*/*/review.json'))
    data=json.loads(path.read_text(encoding='utf-8'))['derived_metrics']
    assert data['live_session_metrics_v1']==h.stats['live_session_metrics_v1']
    for key in ('precision','recall','f1'):
        assert ed.contexts.live_metrics.cards[key].value.cget('text')==ed.report_host.cards[key].value.cget('text')
    manifest=json.loads((path.parent/'export_manifest.json').read_text(encoding='utf-8'))
    assert 'review.json' in {f['path'] for f in manifest['files']}
    assert s.sidecar_path.read_bytes()==frozen and s.review.review_revision==revision

def test_L21_L25_L29_current_f4_copy_uses_current_saved_gt(tmp_path,monkeypatch):
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    baseline=json.loads((OUT/'baseline_f4.json').read_text(encoding='utf-8'))
    row=next(s for s in baseline['sessions'] if s['session_id']=='exp-1792702658896-47b4e60b')
    sidecar=tmp_path/'private.review.json';shutil.copy2(row['sidecar'],sidecar)
    s=MobileReviewSession.open(Path(row['path']),sidecar_path=sidecar)
    monkeypatch.setattr(native,'source_session',lambda _:s)
    original=tk.Tk;scales=[]
    def local_root(*a,**kw):
        root=original(*a,**kw);scales.append(root.tk.call('tk','scaling'))
        root.tk.call('tk','scaling',120/72);return root
    monkeypatch.setattr(native.tk,'Tk',local_root)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch)
    root,h,ed,_=next(gen)
    try:
        frozen=sidecar.read_bytes();h.window.maxsize(3000,2400)
        candidate=next(c for c in view(ed)['candidates'] if c['role']=='FINAL' and c['plate_id'])
        ed._choose_plate(candidate['object_id']);refreshed(root,h,ed)
        metrics=ed.contexts.live_metrics;snapshot=h.stats['live_session_metrics_v1']
        assert snapshot['review_revision']==row['review_revision']==89
        assert snapshot['mt']['n_qualified']==4
        assert snapshot['geometry']['n_geometry']==snapshot['ocr']['n_reads']==3
        measurements=[]
        for width,height in ((1649,896),(1180,760)):
            h.window.geometry(f'{width}x{height}+10+10')
            native.pump(root,lambda:(h.window.winfo_width(),h.window.winfo_height())==(width,height))
            ready(root,h,ed);h.review_sidebar_scroll.yview_moveto(0 if width==1649 else 1);native.pump(root)
            edge=h.review_sidebar_scroll.winfo_rooty()+h.review_sidebar_scroll.winfo_height()
            assert all(c.value.winfo_rooty()+c.value.winfo_height()<=edge for c in metrics.cards.values())
            measurements.append(dict(size=[width,height],dpi=h.window.winfo_fpixels('1i'),
                values={k:c.value.cget('text') for k,c in metrics.cards.items()},exact=metrics.exact.cget('text'),
                denominators=metrics.denominators.cget('text'),scroll=list(h.review_sidebar_scroll.yview())))
        assert sidecar.read_bytes()==frozen
        # Deliberate decisions are exclusively on the private sidecar.
        native.key(h,'space');after=refreshed(root,h,ed)
        assert after['review_revision']==90 and after['mt']['n_qualified']==3
        native.key(h,'space');after=refreshed(root,h,ed)
        assert after['review_revision']==91 and after['mt']['n_qualified']==4
        assert hashlib.sha256(Path(row['sidecar']).read_bytes()).hexdigest()==row['sidecar_file']['sha256']
        (OUT/'native_current_f4.json').write_text(json.dumps(dict(session_id=s.session_id,
            source_archive_sha256=s.review.source_archive_sha256,original_revision=89,
            private_copy_revision=91,measurements=measurements),ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        root.tk.call('tk','scaling',scales[0])
        try:next(gen)
        except StopIteration:pass
