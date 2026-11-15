"""Native Windows Tk acceptance on private real data, no physical-input claim."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy
import json
import os
import shutil
import time
import tkinter as tk
import pytest
from PIL import ImageGrab
from windows_dpi import enable_windows_dpi_awareness
DPI_MODE=enable_windows_dpi_awareness()
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from test_z4_ppm_seed_active_click import settled
from test_f4_eval2_integration import baseline,actual,SID,G,GOOD,BAD,stamp
from auto_annotation_tool.ranking import mobile_mt_object_evaluation as mt
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_canvas_presentation import subject_groups
from auto_annotation_tool.ranking.mobile_session_metrics import current,result_key,presentation
from auto_annotation_tool.gui import mobile_canvas_clean_view as clean,mobile_canvas_navigator as nav,mobile_canvas_geometry_layers as layers
from auto_annotation_tool.gui import mobile_canvas_editor as editor,mobile_canvas_review_feedback as feedback


def move(root,h,ed,s,gid):
    iid=next(i for i,k in h._subject_ids.items() if gid in subject_groups(s,k))
    h.subject_tree.selection_set(iid);h.subject_tree.event_generate('<<TreeviewSelect>>')
    native.pump(root,lambda:h.current_invocation==gid and h._image_entry==core.scene(s,gid)['image_entry'])
    settled(root,h,ed)


def latest(root,h,ed):
    native.pump(root,lambda:current(h.stats.get('live_session_metrics_v2'),result_key(h.session))
        and not ed.in_refresh and not ed.refresh_after and not h._saving,seconds=30)
    ready(root,h,ed)


@pytest.fixture
def gui(tmp_path,monkeypatch,actual,request):
    theme,width,height=getattr(request,'param',('light',1649,896))
    old=tk.Tk
    def make_root(*args,**kwargs):
        root=old(*args,**kwargs);root.tk.call('tk','scaling',120/72);return root
    monkeypatch.setattr(native.tk,'Tk',make_root)
    original=editor.palette
    monkeypatch.setattr(editor,'palette',lambda w,b:original(w,'#FFFFFF' if theme=='light' else '#17232D'))
    def namespace(**kwargs):
        if 'palette' in kwargs:
            kwargs['palette']=dict(panel='#F5F5F2',fg='#213547',muted='#5E7283',accent='#14628B') if theme=='light' else dict(panel='#17232D',fg='#EBF2F6',muted='#B5C4CE',accent='#77CBF0')
        return NS(**kwargs)
    monkeypatch.setattr(native,'SimpleNamespace',namespace)
    monkeypatch.setattr(native,'source_session',lambda _:actual)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    h.window.state('normal');h.window.geometry(f'{width}x{height}+10+10');h.window.deiconify();h.window.lift();root.update()
    move(root,h,ed,s,G);latest(root,h,ed)
    try:yield root,h,ed,s
    finally:
        try:next(gen)
        except StopIteration:pass


@pytest.mark.parametrize('kind',['GTR','POSE','BBOX'])
def test_T29_Space_on_each_layer_confirms_same_pair_and_Undo(gui,kind):
    root,h,ed,s=gui;before=copy.deepcopy(stacks.snapshot(s,G)['plates']);parents=copy.deepcopy(core.scene(s,G)['parents'])
    ed._commit(lambda:partial.toggle_stack(s,G,oid=GOOD),'Private test: pending pair');latest(root,h,ed)
    assert mt.evaluate_unit(s,G)['tp'] is None
    assert [ed.contexts.counters[k].cget('text') for k in ('tp','fp','fn')]==['—','≥1','—']
    assert nav.select_layer(ed,kind,GOOD);ready(root,h,ed);revision=s.review.review_revision
    native.key(h,'space',xy=(-1,-1));latest(root,h,ed)
    u=mt.evaluate_unit(s,G)
    assert (u['tp'],u['fp'],u['fn'])==(1,1,0) and s.review.review_revision==revision+1
    assert stacks.snapshot(s,G)['plates']==before and core.scene(s,G)['parents']==parents
    assert core._legacy_scene(s,G)['events'][-1]['action']=='operator_confirm_associated_frames'
    ed.undo();latest(root,h,ed);assert mt.evaluate_unit(s,G)['tp'] is None
    ed.undo(redo=True);latest(root,h,ed);assert mt.evaluate_unit(s,G)['tp']==1


def test_T31_FP_FN_inspection_keeps_exact_source_and_physical_ID_at_overlap(gui):
    root,h,ed,s=gui
    fence=next(p['quad'] for p in core.scene(s,G)['parents'] if p['attempt_id']==BAD[3:])
    created={}
    def add_private_control():
        oid=core.add_plate(s,G,fence,partial_review=True)
        pid=next(pid for pid,p in stacks.snapshot(s,G)['plates'].items() if p['origin_object_id']==oid)
        partial.confirm_stack(s,G,plate_id=pid);created.update(oid=oid,pid=pid)
    ed._commit(add_private_control,'Private test: overlapping FN reference');latest(root,h,ed)
    before=stamp(s.sidecar_path)
    assert mt.evaluate_unit(s,G)['fn']==1
    assert feedback.inspect(ed,'fp') and ed.selected==BAD
    assert layers.model(ed)['active'] in ('BBOX','POSE')
    assert feedback.inspect(ed,'fn')
    row=clean.selected(ed)
    assert row['pid']==created['pid']
    value=layers.model(ed)
    assert value is None or value['active']=='GTR'
    assert stamp(s.sidecar_path)==before


@pytest.mark.parametrize('gui',[('light',1649,896),('dark',1649,896),('light',1180,760),('dark',1180,760)],indirect=True)
def test_native_four_configurations_real_image01_and07(gui,request):
    root,h,ed,s=gui;theme,width,height=request.node.callspec.params['gui']
    out=Path(os.environ.get('F4_EVAL2_NATIVE_OUTPUT',str(request.node.path.parent.parent/'output/F4_EVAL2_NATIVE')))
    out.mkdir(parents=True,exist_ok=True)
    before=stamp(s.sidecar_path);receipt=[]
    for number,gid in ((1,SID+'-a00000001'),(7,G)):
        move(root,h,ed,s,gid);latest(root,h,ed)
        oid='GT:'+gid;assert nav.select_layer(ed,'POSE',oid)
        nav.show_geometry(ed,nav.quad(layers.model(ed),'POSE'));ready(root,h,ed)
        assert ed.selected==oid
        u=mt.evaluate_unit(s,gid)
        assert [ed.contexts.counters[k].cget('text') for k in ('gt','tp','fp','fn')]==(['1','1','0','0'] if number==1 else ['1','1','1','0'])
        shown=presentation(h.stats['live_session_metrics_v2'],result_key(s))
        assert ed.contexts.live_metrics.data==shown
        widgets=[c.value for c in ed.contexts.live_metrics.cards.values()]
        assert all(w.winfo_ismapped() and w.winfo_width()>25 for w in widgets)
        for w in widgets:
            assert w.winfo_rootx()+w.winfo_width()<=h.window.winfo_rootx()+h.window.winfo_width()+1
        # Retain the existing right-panel scroll and layout. Reachability means
        # inside its viewport, not merely Tcl's ismapped flag below the fold.
        h.review_sidebar_scroll.yview_moveto(1);root.update()
        for w in widgets+[ed.contexts.live_metrics.denominators]:
            assert w.winfo_rooty()>=h.review_sidebar_scroll.winfo_rooty()-1
            assert w.winfo_rooty()+w.winfo_height()<=h.review_sidebar_scroll.winfo_rooty()+h.review_sidebar_scroll.winfo_height()+1
        h.window.lift();root.update();time.sleep(.15);root.update()
        box=(h.window.winfo_rootx(),h.window.winfo_rooty(),h.window.winfo_rootx()+h.window.winfo_width(),h.window.winfo_rooty()+h.window.winfo_height())
        image=ImageGrab.grab(bbox=box,all_screens=True)
        assert image.size==(width,height)
        filename=f'{theme}_{width}x{height}_image{number:02d}.png';image.save(out/filename)
        receipt.append(dict(theme=theme,size=[width,height],image_number=number,source_gid=gid,
            review_revision=s.review.review_revision,policy_sha256=mt.POLICY_SHA256,
            counts={k:u[k] for k in ('tp','fp','fn')},cards=shown,
            tk_scaling=float(root.tk.call('tk','scaling')),dpi_mode=DPI_MODE,png=filename,
            input_method='native_Tk_event_generate_and_own_widget_API_not_physical_operator_input'))
    assert stamp(s.sidecar_path)==before
    (out/f'{theme}_{width}x{height}.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
