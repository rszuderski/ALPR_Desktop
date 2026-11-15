"""Native Tk: existing image cards share P0; one read-only P1 row is added later."""
from pathlib import Path
import json
import tkinter as tk
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_mt_detection_evaluation import approved_session,counts
from auto_annotation_tool.ranking import mobile_mt_detection_evaluation as mt
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks
from auto_annotation_tool.gui import mobile_canvas_editor as editor

G='mt-one'

@pytest.fixture
def detection_gui(tmp_path,monkeypatch,request):
    s=approved_session(tmp_path/'data',n=2)
    monkeypatch.setattr(native,'source_session',lambda p:s)
    original=tk.Tk
    scaling=[]
    def test_root(*args,**kwargs):
        root=original(*args,**kwargs)
        scaling.append(root.tk.call('tk','scaling'))
        root.tk.call('tk','scaling',120/72)  # interpreter-local test scale, no OS setting
        return root
    monkeypatch.setattr(native.tk,'Tk',test_root)
    theme=getattr(request,'param','dark')
    original_palette=editor.palette
    monkeypatch.setattr(editor,'palette',lambda widget,background:original_palette(widget,'#FFFFFF' if theme=='light' else '#17232D'))
    gen=native.gui.__wrapped__(tmp_path,monkeypatch)
    value=next(gen)
    try:yield value
    finally:
        value[0].tk.call('tk','scaling',scaling[0])
        try:next(gen)
        except StopIteration:pass


def test_E23_P0_existing_four_cards_get_gt_tp_fp_fn_without_rearrangement(detection_gui):
    root,h,ed,s=detection_gui;ready(root,h,ed)
    assert [ed.contexts.counters[k].cget('text') for k in ('gt','mt','fp','fn')]==['2','2','0','0']
    assert [ed.contexts.counter_labels[k].cget('text') for k in ('gt','mt','fp','fn')]==[
        'Tablice GT','Trafne MT (TP)','Błędne MT (FP)','Pominięte (FN)']
    assert ed.contexts.counter_labels['gt'].master.grid_info()['row']==0
    assert ed.contexts.counter_labels['fn'].master.grid_info()['row']==1


def test_E17_existing_ppm_undo_redo_updates_same_canonical_values(detection_gui):
    root,h,ed,s=detection_gui;ready(root,h,ed)
    parents=core.scene(s,G)['parents']
    native.click(h,ed,(445,175),3);ready(root,h,ed)
    assert counts(mt.evaluate_unit(s,G))==(1,1,1,0)
    assert [ed.contexts.counters[k].cget('text') for k in ('gt','mt','fp','fn')]==['1','1','1','0']
    ed.undo();ready(root,h,ed);assert counts(mt.evaluate_unit(s,G))==(2,2,0,0)
    ed.undo(redo=True);ready(root,h,ed);assert counts(mt.evaluate_unit(s,G))==(1,1,1,0)
    assert core.scene(s,G)['parents']==parents


def test_E16_unresolved_presence_does_not_show_formal_zero(detection_gui):
    root,h,ed,s=detection_gui
    native.click(h,ed,(225,175));native.key(h,'space');ready(root,h,ed)
    assert mt.evaluate_unit(s,G)['tp'] is None
    assert all(ed.contexts.counters[k].cget('text')=='—' for k in ('gt','mt','fp','fn'))
    assert 'Rozstrzygnij ramki' in ed.contexts.fields['image'].cget('text')


def test_E20_selection_zoom_and_zx_are_read_only_for_shared_gt(detection_gui,monkeypatch):
    root,h,ed,s=detection_gui
    # Deliberate source-identity association; subsequent gestures only select.
    from auto_annotation_tool.ranking.mobile_canvas_resolution import link_manual_to_mt
    # An explicit duplicate decision is the established action for two FINAL.
    from auto_annotation_tool.ranking.mobile_canvas_physical_gt import mark_duplicate
    mark_duplicate(s,G,'GT:a1','GT:a0');ready(root,h,ed)
    before=s.sidecar_path.read_bytes();ed._choose_plate('GT:a0');ready(root,h,ed)
    native.key(h,'z');ready(root,h,ed);assert ed.selected=='GT:a1'
    native.key(h,'x');ready(root,h,ed);assert ed.selected=='GT:a0'
    h._view_zoom=1.4;h._paint_image();native.pump(root)
    h._view_zoom=1;h._paint_image();native.pump(root)
    assert s.sidecar_path.read_bytes()==before
    assert len(h.canvas.find_withtag('stack_badge_text'))==1


def test_P0_panel_and_report_share_exact_counts_and_view_does_not_write(detection_gui):
    root,h,ed,s=detection_gui;ready(root,h,ed)
    before=s.sidecar_path.read_bytes();h.stats=s.statistics();ed.update_report()
    d=h.stats['mt_detection_evaluation_v1'];assert d['tp']==2 and d['fn']==d['fp']==0
    assert [ed.report_host.cards[k].value.cget('text') for k in ('expected','linked','false','missed')]==['2','2','0','0']
    ed.report_host.master.select(ed.report_host);native.pump(root)
    ed.update_report();assert s.sidecar_path.read_bytes()==before


@pytest.mark.parametrize('detection_gui',['light','dark'],indirect=True)
def test_M16_one_compact_row_at_both_native_sizes_and_tk_dpi120(detection_gui):
    root,h,ed,s=detection_gui;ready(root,h,ed)
    h.stats=s.statistics();ed.update_report();panel=ed.report_host
    before=s.sidecar_path.read_bytes();tabs=[panel.master.tab(t,'text') for t in panel.master.tabs()]
    # Permit the requested client size even on the test host's smaller screen.
    # This is a test-window limit, not a change to application or OS settings.
    h.window.maxsize(3000,2400)
    measurements=[]
    for width,height in ((1649,896),(1180,760)):
        h.window.geometry(f'{width}x{height}+10+10')
        native.pump(root,lambda:(h.window.winfo_width(),h.window.winfo_height())==(width,height))
        panel.master.select(0);ready(root,h,ed)
        for w in (h.canvas,h.subject_tree,ed.contexts):assert w.winfo_ismapped()
        assert ed.contexts.winfo_rooty()+ed.contexts.winfo_height()<=h.window.winfo_rooty()+height
        assert abs(h.window.winfo_fpixels('1i')-120)<1
        panel.master.select(panel);native.pump(root)
        cards=[panel.cards[k] for k in ('precision','recall','f1')]
        assert all(c.winfo_ismapped() for c in cards)
        assert {int(c.grid_info()['row']) for c in cards}=={0}
        assert [int(c.grid_info()['column']) for c in cards]==[0,1,2]
        assert len({c.winfo_y() for c in cards})==1
        assert panel.mt_rates_frame.winfo_height()<140
        for c in cards:
            assert c.value.cget('text')=='100,0%'
            assert str(c.cget('takefocus'))=='0'
            assert c.winfo_x()+c.winfo_width()<=panel.mt_rates_frame.winfo_width()+1
            assert c.value.winfo_reqwidth()<=c.value.winfo_width()
            assert c.cget('bg')==ed.p['surface']
        assert panel.mt_rates_frame.winfo_y()>panel.mt_counts.winfo_y()
        assert [panel.master.tab(t,'text') for t in panel.master.tabs()]==tabs
        measurements.append(dict(size=[width,height],dpi=h.window.winfo_fpixels('1i'),
            rate_row_height=panel.mt_rates_frame.winfo_height(),theme=ed.p['bg']))
    # Report measurements are widget assertions, not a screenshot claim.
    out=Path(__file__).resolve().parents[1]/'output/Z4_P0_P1_APPROVED_20261026_001'
    (out/('native_layout_'+('light' if ed.p['bg']=='#F2F5F8' else 'dark')+'.json')).write_text(json.dumps(measurements,indent=2),encoding='utf-8')
    assert s.sidecar_path.read_bytes()==before


def test_M14_M17_rates_refresh_and_stale_revision_are_not_displayed(detection_gui):
    root,h,ed,s=detection_gui;ready(root,h,ed)
    h.stats=s.statistics();ed.update_report();panel=ed.report_host
    assert panel.cards['precision'].value.cget('text')=='100,0%'
    assert 'Ocenione 1/1' in panel.mt_counts.cget('text')
    panel.render(h.stats['canvas_review'] | {'mt_detection_evaluation_v1':h.stats['mt_detection_evaluation_v1']},ed.inventory(),refreshing=True)
    assert all(panel.cards[k].value.cget('text')=='—' for k in ('precision','recall','f1'))
    old=h.stats['mt_detection_evaluation_v1']
    stale=dict(old,review_revision=old['review_revision']-1)
    panel.render(h.stats['canvas_review'] | {'mt_detection_evaluation_v1':stale},ed.inventory())
    assert all(panel.cards[k].value.cget('text')=='—' for k in ('precision','recall','f1'))
    ed.update_report()
    assert panel.cards['f1'].value.cget('text')=='100,0%'

def test_E25_M19_existing_gui_export_includes_bound_detection_sidecar(detection_gui,tmp_path,monkeypatch):
    from auto_annotation_tool.gui import z4_mobile_sample_review as module
    root,h,ed,s=detection_gui;ready(root,h,ed)
    base=tmp_path/'03_wyniki';base.mkdir()
    monkeypatch.setattr(module.filedialog,'askdirectory',lambda **kw:str(base))
    monkeypatch.setattr(module.messagebox,'askyesno',lambda *a,**kw:True)
    errors=[];monkeypatch.setattr(module.messagebox,'showerror',lambda *a,**kw:errors.append(a))
    before=s.sidecar_path.read_bytes();revision=s.review.review_revision
    h.export();native.pump(root,lambda:not h._saving,seconds=45)
    assert not errors,errors
    manifests=list(base.glob('*/*/export_manifest.json'))
    assert len(manifests)==1,(h.status_var.get(),list(base.rglob('*.json')))
    manifest=json.loads(manifests[0].read_text(encoding='utf-8'))
    path=next(p for p in manifests[0].parent.rglob('mt_detection_evaluation.json'))
    exported=json.loads(path.read_text(encoding='utf-8'))
    assert exported==mt.evaluate_session(s)
    assert path.relative_to(manifests[0].parent).as_posix() in {f['path'] for f in manifest['files']}
    assert s.sidecar_path.read_bytes()==before and s.review.review_revision==revision

def test_E18_D_adds_a_valid_gt_and_immediate_formal_fn_without_space(detection_gui):
    from test_mobile_p2b2_canvas_review import MANUAL
    root,h,ed,s=detection_gui;ready(root,h,ed)
    native.key(h,'d')
    for point in MANUAL:native.click(h,ed,point)
    ready(root,h,ed)
    assert counts(mt.evaluate_unit(s,G))==(3,2,0,1)
    assert ed.contexts.counters['fn'].cget('text')=='1' and ed.checked_var.get()
    native.pump(root,lambda:ed.report_host.cards['recall'].value.cget('text')=='66,7%')
    assert ed.report_host.cards['f1'].value.cget('text')=='80,0%'
