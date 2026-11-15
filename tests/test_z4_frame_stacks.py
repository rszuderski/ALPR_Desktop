"""T01–T28: additive GT contract, immutable evidence and real Tk events."""
import copy, importlib, json, hashlib, os
from pathlib import Path
import pytest
from test_z4_p0_physical_gt import session_fixture, prepare
from test_mobile_p2b2_canvas_review import source_session, MANUAL
from test_z4_akt89lm_resolution import invalid_source
from test_mobile_human_review import load
from auto_annotation_tool.ranking import mobile_canvas_review as core
import test_mobile_p2b2_canvas_gui as native

G='mt-one'
OUT=Path(__file__).resolve().parents[1]/'output/P0_Z4_FRAME_STACKS_20261025_001'

@pytest.fixture
def stacks():return importlib.import_module('auto_annotation_tool.ranking.mobile_candidate_stacks')

def seeded(path,n=3,suppressed=None):
    s=session_fixture(path,n=n,suppressed=suppressed)
    core.set_text(s,G,'GT:a0','WA01234')
    core.correct_plate(s,G,'GT:a0',core.scene(s,G)['objects'][0]['quad'])
    return s

def linked(path,stacks,n=3,suppressed=None):
    s=seeded(path,n,suppressed)
    for i in range(1,n):stacks.associate(s,G,f'GT:a{i}','GT:a0')
    core.check_scene(s,G)
    return s

@pytest.fixture
def stack_gui(tmp_path,monkeypatch,stacks):
    # Three original predictions actually 1 and 2 source pixels apart.
    import csv,io,zipfile
    from test_z4_p0_physical_gt import csv_bytes
    source=session_fixture(tmp_path/'data',n=3,suppressed=None)
    with zipfile.ZipFile(source.path) as z:payload={name:z.read(name) for name in z.namelist()}
    rows=list(csv.DictReader(io.StringIO(payload['samples/attempts.csv'].decode('utf-8'))))
    points=[tuple(map(float,p.split(':'))) for p in rows[0]['mt_pose_corners_norm'].split(';')]
    for i,row in enumerate(rows):
        row['mt_pose_corners_norm']=';'.join(f'{x+i/1000}:{y}' for x,y in points)
        row.update(plate_left=float(rows[0]['plate_left'])+i,plate_right=float(rows[0]['plate_right'])+i)
    payload['samples/attempts.csv']=csv_bytes(rows);path=tmp_path/'close.alprsession'
    with zipfile.ZipFile(path,'w') as z:
        for name,data in payload.items():z.writestr(name,data)
    s=load(path,tmp_path);s.set_review_mode('assisted')
    core.set_text(s,G,'GT:a0','WA01234');core.correct_plate(s,G,'GT:a0',core.scene(s,G)['objects'][0]['quad'])
    for i in (1,2):stacks.associate(s,G,f'GT:a{i}','GT:a0')
    core.check_scene(s,G)
    monkeypatch.setattr(native,'source_session',lambda p:s)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)

def ready(root,h,ed):
    native.pump(root,lambda:not h._saving and not ed.in_refresh and not ed.busy)
    ed.cache_key=None;ed.sync();ed.overlay();native.pump(root)

def stack_key(h,key):
    if key!='ISO_Left_Tab':return native.key(h,key)
    native.focus_canvas(h)
    h.canvas.event_generate('<KeyPress-Tab>',state=1);h.window.update()
    h.canvas.event_generate('<KeyRelease-Tab>',state=1);h.window.update()

def test_T01_single_reference_and_legacy_read_only(tmp_path,stacks):
    s=seeded(tmp_path,n=1);core.check_scene(s,G)
    before=s.sidecar_path.read_bytes();summary=core.summary(s);v=stacks.snapshot(s,G)
    assert len(v['plates'])==len(v['candidates'])==1
    assert next(iter(v['plates'].values()))['gt_text']=='WA01234'
    assert s.sidecar_path.read_bytes()==before and core.summary(s)==summary
    fresh=session_fixture(tmp_path/'fresh',n=1,suppressed=None)
    assert not stacks.snapshot(fresh,G)['plates']  # an unreviewed prediction is not GT
    stacks.accept_reference(fresh,G,'GT:a0')
    assert len(stacks.snapshot(fresh,G)['plates'])==1
    assert next(iter(stacks.snapshot(fresh,G)['plates'].values()))['gt_text']==''  # never OCR -> GT

def test_T02_three_candidates_one_active_and_one_reference(stack_gui,stacks):
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a0');ready(root,h,ed)
    assert len(h.canvas.find_withtag('mt_original'))==1
    assert len(h.canvas.find_withtag('operator_quad'))==1
    assert any('MT 1/3' in h.canvas.itemcget(i,'text') for i in h.canvas.find_withtag('clean_label_text'))
    # A compatible legacy duplicate group also adapts without a write. Its
    # first deliberate shared edit adds canonical state, retaining old fields.
    legacy=copy.deepcopy(core.scene(s,G));legacy.pop(stacks.KEY);legacy.pop(stacks.HISTORY);legacy.pop('_stack_legacy')
    changed=copy.deepcopy(s.review);changed.invocation_annotations[G][core.KEY]=legacy;s._save_change(changed)
    ready(root,h,ed);frozen=s.sidecar_path.read_bytes();old_objects=copy.deepcopy(legacy['objects'])
    ed._choose_plate('GT:a1');ready(root,h,ed)
    assert len(h.canvas.find_withtag('mt_original'))==len(h.canvas.find_withtag('operator_quad'))==1
    assert s.sidecar_path.read_bytes()==frozen and stacks.KEY not in s.invocation_annotation(G)[core.KEY]
    ed.start_entry('GT:a1');ed.entry_var.set('COMMON');ed.entry_enter();ready(root,h,ed)
    assert s.invocation_annotation(G)[core.KEY]['objects']==old_objects
    assert all(r['gt_text']=='COMMON' for r in stacks.diagnostics(s)['rows'])

def test_T03_zx_both_directions_wrap_without_writes(stack_gui,stacks):
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a0');ready(root,h,ed)
    frozen=s.sidecar_path.read_bytes();rev=s.review.review_revision
    for key,expected in [('z','GT:a1'),('z','GT:a2'),('Z','GT:a0'),('x','GT:a2'),('X','GT:a1')]:
        stack_key(h,key);ready(root,h,ed);assert ed.selected==expected
    undo_before=copy.deepcopy(ed.undo_stack)
    ed._commit(lambda:stacks.associate(s,G,'GT:a2','GT:a0'),'Powtórzone przypisanie')
    assert ed.undo_stack==undo_before
    assert s.sidecar_path.read_bytes()==frozen and s.review.review_revision==rev

def test_T04_tab_in_text_field_does_not_cycle(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a0');ed.start_entry('GT:a0');native.pump(root)
    ed.entry.focus_force();native.pump(root);selected=ed.selected
    entry=ed.entry;target=entry.tk_focusNext();ed.entry_var.set('WA01234X');revision=s.review.review_revision
    entry.event_generate('<KeyPress-Tab>');native.pump(root)
    assert ed.selected==selected
    assert h.window.focus_get() is not entry and ed.entry is None
    assert h.window.focus_get() is target
    assert core.scene(s,G)['objects'][0]['gt_text']=='WA01234X' and s.review.review_revision==revision+1

def test_T05_badge_without_selection_survives_zoom_and_restart(stack_gui,stacks):
    root,h,ed,s=stack_gui;ed._choose_plate('');ready(root,h,ed)
    ids=list(stacks.snapshot(s,G)['plates']);before=s.sidecar_path.read_bytes()
    for zoom in (1.0,1.6,1.0):
        h._view_zoom=zoom;h._paint_image();native.pump(root)
        assert h.canvas.find_withtag('clean_label_text')
    assert list(stacks.snapshot(load(s.path,s.sidecar_path.parent),G)['plates'])==ids
    assert s.sidecar_path.read_bytes()==before

def test_T06_hit_only_active_stack_and_four_handles(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a1');ready(root,h,ed)
    point=core.scene(s,G)['objects'][1]['quad'][0];x,y=ed.to_canvas(point)
    assert ed.hit(x,y)[0]=='GT:a1'
    assert len(h.canvas.find_withtag('corner_handle'))==4
    source=copy.deepcopy(core.scene(s,G)['parents'])
    assert abs(source[1]['quad'][0][0]-source[0]['quad'][0][0]-1)<1e-6
    assert abs(source[2]['quad'][0][0]-source[0]['quad'][0][0]-2)<1e-6
    h.canvas.event_generate('<ButtonPress-1>',x=round(x),y=round(y))
    h.canvas.event_generate('<B1-Motion>',x=round(x+2),y=round(y+1))
    h.canvas.event_generate('<ButtonRelease-1>',x=round(x+2),y=round(y+1));ready(root,h,ed)
    assert core.scene(s,G)['parents']==source

def test_T07_ppm_final_undo_redo_preserves_mt(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a0');ready(root,h,ed)
    parents=copy.deepcopy(core.scene(s,G)['parents']);before=s.review.review_revision
    from test_z4_bad_only_mt_final import source_badge,ppm
    ppm(h,*source_badge(h,ed));ready(root,h,ed)
    assert core.scene(s,G)['objects'][0]['error'] and s.review.review_revision==before+1
    ed.undo();ready(root,h,ed);assert not core.scene(s,G)['objects'][0]['error']
    ed.undo(redo=True);ready(root,h,ed);assert core.scene(s,G)['parents']==parents

def test_T08_direct_ppm_targets_the_visible_stack_candidate(stack_gui):
    root,h,ed,s=stack_gui;ed._choose_plate('');ready(root,h,ed);before=copy.deepcopy(core.scene(s,G)['parents']);rev=s.review.review_revision
    # The source badge explicitly identifies MT despite the overlapping GTR.
    from test_z4_bad_only_mt_final import source_badge,ppm
    ppm(h,*source_badge(h,ed));ready(root,h,ed)
    assert ed.selected=='GT:a0' and core._object(core.scene(s,G),'GT:a0')['error']
    assert s.review.review_revision==rev+1 and core.scene(s,G)['parents']==before

def test_T09_suppressed_unassociation_is_diagnostic_only(tmp_path,stacks):
    s=linked(tmp_path,stacks,n=2,suppressed=1);before=core.summary(s)
    stacks.toggle_unassociated(s,G,'GT:a1')
    assert core.scene(s,G)['checked']  # RAW diagnostics do not demand re-review of GT
    assert core.counts(core.scene(s,G))['false_positive']==before['false_positive_mt']==0
    row=next(r for r in stacks.diagnostics(s)['rows'] if r['attempt_id']=='a1')
    assert row['association_status']=='UNASSOCIATED_NO_PLATE' and row['role']=='SUPPRESSED'
    for key in ('expected_plates','linked_plates','false_positive_mt','missed_plates','geometry_n','evaluated_reads','cer'):
        assert core.summary(s)[key]==before[key]

def test_T10_low_iou_is_measurable_without_automatic_fp(tmp_path,stacks):
    s=linked(tmp_path,stacks,n=2)
    q=copy.deepcopy(core.scene(s,G)['objects'][0]['quad']);q=[[x+130,y+15] for x,y in q]
    core.correct_plate(s,G,'GT:a0',q)
    rows=stacks.diagnostics(s)['rows'];assert all(r['polygon_iou_pct'] is not None for r in rows)
    assert min(r['polygon_iou_pct'] for r in rows)<50
    assert not core.scene(s,G)['objects'][0]['error']

def test_T11_final_duplicates_two_measurements_one_tp(tmp_path,stacks):
    s=linked(tmp_path,stacks,n=2);summary=core.summary(s);rows=stacks.diagnostics(s)['rows']
    assert (summary['expected_plates'],summary['linked_plates'],summary['false_positive_mt'])==(1,1,1)
    assert len(rows)==2 and all(r['polygon_iou_pct'] is not None for r in rows)

def test_T12_two_real_plates_keep_two_ids(tmp_path,stacks):
    source=Path(__file__).resolve().parents[1]/'output/p0_obraz05_20261025'
    s=session_fixture(tmp_path,n=2,suppressed=None,real=source);prepare(s)
    before=core.summary(s);v=stacks.snapshot(s,G)
    assert len(v['plates'])==2 and len(set(v['plates']))==2 and core.summary(s)==before
    stacks.edit_gt(s,G,'GT:a0',text='SI62636')
    canonical=stacks.snapshot(s,G);ids=list(canonical['plates']);other=copy.deepcopy(canonical['plates'][ids[1]])
    q=copy.deepcopy(canonical['plates'][ids[0]]['gt_quad']);q[0][0]+=1
    core.correct_plate(s,G,'GT:a0',q)
    assert stacks.snapshot(s,G)['plates'][ids[1]]==other

def test_T13_overlap_does_not_merge_distinct_plates(tmp_path,stacks,monkeypatch):
    s=seeded(tmp_path,n=2);core.correct_plate(s,G,'GT:a1',core.scene(s,G)['objects'][0]['quad'])
    core.set_text(s,G,'GT:a1','SECOND');core.check_scene(s,G)
    v=stacks.snapshot(s,G);assert len(v['plates'])==2
    assert len({c['plate_id'] for c in v['candidates']})==2
    monkeypatch.setattr(native,'source_session',lambda p:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,_=next(gen)
    try:
        parents=copy.deepcopy(core.scene(s,G)['parents']);ids=list(v['plates']);rev=s.review.review_revision
        native.click(h,ed,(225,175),button=3);ready(root,h,ed)
        assert core.scene(s,G)['objects'][0]['error'] and not core.scene(s,G)['objects'][1]['error']
        assert s.review.review_revision==rev+1 and list(stacks.snapshot(s,G)['plates'])==ids
        assert core.scene(s,G)['parents']==parents and core._object(core.scene(s,G),'GT:a1')['gt_text']=='SECOND'
        ed._choose_plate('');native.click(h,ed,(225,175),button=3);ready(root,h,ed)
        assert not core._object(core.scene(s,G),'GT:a0')['error'] and list(stacks.snapshot(s,G)['plates'])==ids
    finally:
        try:next(gen)
        except StopIteration:pass

def test_T14_same_scene_two_calls_keeps_invocation_denominators(tmp_path,stacks):
    s=session_fixture(tmp_path,n=2,suppressed=None,two_calls=True)
    for gid in s.mt_invocations:prepare(s,gid)
    rows=stacks.diagnostics(s)['rows']
    assert len({r['mt_invocation_id'] for r in rows})==2
    assert stacks.diagnostics(s)['n_invocations']==2

def test_T15_real_night_three_raw_two_final_copy(tmp_path,stacks):
    import shutil
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    source=OUT/'copies/exp-1792702658896-47b4e60b';archive=next(source.glob('*.alprsession'))
    shutil.copy2(archive,tmp_path/archive.name);shutil.copy2(source/'review.json',tmp_path/'review.json')
    s=MobileReviewSession.open(tmp_path/archive.name,sidecar_path=tmp_path/'review.json');gid=list(s.mt_invocations)[4]
    scene=core.scene(s,gid);core.check_scene(s,gid)
    stacks.associate(s,gid,scene['objects'][1]['object_id'],scene['objects'][0]['object_id']);core.check_scene(s,gid)
    v=stacks.snapshot(s,gid);assert len(v['plates'])==1
    assert [c['role'] for c in v['candidates']]==['FINAL','SUPPRESSED','FINAL']
    assert v['candidates'][0]['plate_id']==v['candidates'][1]['plate_id'] and v['candidates'][2]['plate_id'] is None
    assert core.counts(core.scene(s,gid))['false_positive']==1

def test_T16_historical_day_raw_link_keeps_final_population(tmp_path,stacks):
    s=linked(tmp_path,stacks,n=2,suppressed=1)
    rows=stacks.diagnostics(s)['rows'];assert len(rows)==2 and {r['role'] for r in rows}=={'FINAL','SUPPRESSED'}
    d=core.summary(s);assert d['expected_plates']==d['linked_plates']==1 and d['false_positive_mt']==0

def test_T17_bbox_identity_never_becomes_quad(tmp_path,stacks):
    from auto_annotation_tool.ranking.mobile_canvas_resolution import link_manual_to_mt
    s=invalid_source(tmp_path);manual=core.add_plate(s,G,MANUAL);link_manual_to_mt(s,G,manual,'GT:a0')
    rows=stacks.diagnostics(s)['rows'];assert rows[0]['mt_quad'] is None
    assert rows[0]['polygon_iou_pct'] is rows[0]['nmce_pct'] is None and rows[0]['not_measurable_reason']
    # Real legacy DAY lacks the v2 marker before its first explicit link.
    import shutil
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    source=OUT/'copies/exp-1792699303016-c8b05efb';dest=tmp_path/'real_day';dest.mkdir()
    archive=next(source.glob('*.alprsession'));shutil.copy2(archive,dest/archive.name);shutil.copy2(source/'review.json',dest/'review.json')
    day=MobileReviewSession.open(dest/archive.name,sidecar_path=dest/'review.json');gid=list(day.mt_invocations)[7]
    prior=core.scene(day,gid);mt=next(o['object_id'] for o in prior['objects'] if o.get('parent_id'));manual=next(o['object_id'] for o in prior['objects'] if not o.get('parent_id'))
    revision=day.review.review_revision;stacks.associate(day,gid,mt,manual)
    current=core.scene(day,gid)
    assert day.review.review_revision==revision+1 and len(stacks.snapshot(day,gid)['plates'])==1
    assert current['parents']==prior['parents'] and current['parents'][0]['quad'] is None
    assert current['objects'][0]['identity_decision']['pose_status']=='INVALID'
    assert day.invocation_annotation(gid)[core.KEY]['objects']==prior['objects']

def test_T18_manual_gt_is_fn_not_candidate(tmp_path,stacks):
    s=source_session(tmp_path,n=0,mz=False);core.add_plate(s,G,MANUAL);core.check_scene(s,G)
    v=stacks.snapshot(s,G);assert len(v['plates'])==1 and not v['candidates']
    assert core.summary(s)['missed_plates']==1
    other=linked(tmp_path/'shared',stacks,n=2)
    oid=core.add_plate(other,G,MANUAL);core.set_text(other,G,oid,'MANUAL');core.check_scene(other,G)
    assert core.summary(other)['missed_plates']==1 and len(stacks.snapshot(other,G)['plates'])==2
    core.remove_operator_geometry(other,G,oid)
    assert len(stacks.snapshot(load(other.path,other.sidecar_path.parent),G)['plates'])==1

def test_T19_explicit_negative_only(tmp_path,stacks):
    from auto_annotation_tool.ranking.mobile_canvas_resolution import confirm_no_plates
    s=source_session(tmp_path,n=0,mz=False);assert core.summary(s).get('true_negative_frames',0)==0
    confirm_no_plates(s,G);assert not stacks.snapshot(s,G)['plates']
    assert core.summary(s)['true_negative_frames']==1
    other=linked(tmp_path/'shared',stacks,n=2);legacy=copy.deepcopy(other.invocation_annotation(G)[core.KEY]['objects'])
    confirm_no_plates(other,G)
    reopened=load(other.path,other.sidecar_path.parent)
    assert core.summary(reopened)['false_positive_mt']==2 and core.summary(reopened)['expected_plates']==0
    assert reopened.invocation_annotation(G)[core.KEY]['objects']==legacy

def test_T20_shared_quad_edit_once_preserves_legacy_and_mt(tmp_path,stacks):
    s=linked(tmp_path,stacks);saved=copy.deepcopy(s.invocation_annotation(G)[core.KEY]['objects']);mt=core.scene(s,G)['parents']
    before=stacks.diagnostics(s)['rows'];q=copy.deepcopy(core.scene(s,G)['objects'][0]['quad']);q[0][0]+=8
    core.correct_plate(s,G,'GT:a0',q);after=stacks.diagnostics(s)['rows']
    assert all(r['gt_quad']==q for r in after) and all(a['nmce_pct']!=b['nmce_pct'] for a,b in zip(after,before))
    assert s.invocation_annotation(G)[core.KEY]['objects']==saved and core.scene(s,G)['parents']==mt

def test_T21_one_text_no_mz_copy(tmp_path,stacks):
    s=linked(tmp_path,stacks);before=copy.deepcopy(core.scene(s,G)['parents'])
    core.set_text(s,G,'GT:a1','NEW123')
    v=stacks.snapshot(s,G);assert len(v['plates'])==1 and next(iter(v['plates'].values()))['gt_text']=='NEW123'
    assert all(r['gt_text']=='NEW123' for r in stacks.diagnostics(s)['rows'])
    assert core.scene(s,G)['parents']==before

def test_T22_shared_text_exclusion_preserves_each_mz(tmp_path,stacks):
    s=linked(tmp_path,stacks);core.set_text(s,G,'GT:a1','','nieczytelny')
    rows=stacks.diagnostics(s)['rows'];assert all(r['text_exclusion_reason']=='nieczytelny' and r['cer'] is None for r in rows)
    assert len({r['mz_prediction'] for r in rows})==3

def test_T23_shared_undo_redo_restart(stack_gui,stacks):
    root,h,ed,s=stack_gui;before=stacks.snapshot(s,G);ed._choose_plate('GT:a0')
    ed.start_entry('GT:a0');ed.entry_var.set('CHANGED');ed.entry_enter();ready(root,h,ed)
    assert next(iter(stacks.snapshot(s,G)['plates'].values()))['gt_text']=='CHANGED'
    ed.undo();ready(root,h,ed);assert list(stacks.snapshot(s,G)['plates'])==list(before['plates'])
    ed.undo(redo=True);ready(root,h,ed)
    reopened=load(s.path,s.sidecar_path.parent)
    assert stacks.snapshot(reopened,G)==stacks.snapshot(s,G)
    history=copy.deepcopy(reopened.invocation_annotation(G)[core.KEY][stacks.HISTORY]['entries'])
    assert stacks.restore_history(reopened,G)
    assert next(iter(stacks.snapshot(reopened,G)['plates'].values()))['gt_text']=='WA01234'
    assert stacks.restore_history(reopened,G,redo=True)
    assert next(iter(stacks.snapshot(reopened,G)['plates'].values()))['gt_text']=='CHANGED'
    assert reopened.invocation_annotation(G)[core.KEY][stacks.HISTORY]['entries']==history

def test_T24_all_twelve_originals_read_only(stacks,tmp_path):
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    base=json.loads((OUT/'baseline_f4.json').read_text(encoding='utf-8'))
    assert len(base)==12
    for row in base:
        path=Path(row['sidecar']);before=path.read_bytes() if path.exists() else None
        s=MobileReviewSession.open(row['path'],sidecar_path=path)
        for item in row['scenes']:
            if not item.get('error'):stacks.snapshot(s,item['gid'])
        assert core.summary(s)==row['summary']
        if row['sidecar_exists']:assert s.review.to_dict()==row['review']
        else:assert s.review.review_revision==0 and not s.review.invocation_annotations
        assert (path.read_bytes() if path.exists() else None)==before
    original=session_fixture(tmp_path/'template',n=1,suppressed=None)
    fresh=MobileReviewSession.open(original.path,sidecar_path=tmp_path/'new.json')
    assert not fresh.sidecar_path.exists() and fresh.review.review_revision==0
    core.scene(fresh,G);assert not fresh.sidecar_path.exists()
    core.set_text(fresh,G,'GT:a0','FIRST')
    assert fresh.review.review_revision==1 and fresh.review.review_mode=='assisted'
    locked=MobileReviewSession.open(original.path,sidecar_path=tmp_path/'locked.json')
    locked.set_subject(next(iter(locked.subjects)),ground_truth='OLD',evaluable=True)
    frozen=locked.sidecar_path.read_bytes()
    with pytest.raises(ValueError,match='zaślepion'):core.set_text(locked,G,'GT:a0','OTHER')
    assert locked.sidecar_path.read_bytes()==frozen and locked.review.review_mode=='blinded_gt_v1'

def test_T25_f4_pairs_remain_18_18_24(stacks):
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    rows=json.loads((OUT/'baseline_f4.json').read_text(encoding='utf-8'))[:3]
    assert [r['summary']['geometry_n'] for r in rows]==[18,18,24]
    for row in rows:
        s=MobileReviewSession.open(row['path'],sidecar_path=Path(row['sidecar']))
        stacks.diagnostics(s);assert core.summary(s)==row['summary']

def test_T26_conflicting_legacy_refs_block_conversion_not_old_review(tmp_path,stacks):
    from auto_annotation_tool.ranking.mobile_canvas_physical_gt import mark_duplicate
    s=session_fixture(tmp_path,n=2,suppressed=None);prepare(s)
    # Construct the actual pre-stack representation. New writes now always use
    # canonical GT, so calling current prepare/duplicate cannot seed legacy.
    legacy=copy.deepcopy(core.scene(s,G));legacy.pop(stacks.KEY);legacy.pop(stacks.HISTORY);legacy.pop('_stack_legacy')
    reg=legacy['physical_gt_v1'];reg.pop('decision_source',None)
    target=reg['associations']['GT:a0'];reg['plates']=[p for p in reg['plates'] if p['gt_id']==target]
    reg['associations']['GT:a1']=target;legacy['objects'][1]['duplicate_of']='GT:a0'
    changed=copy.deepcopy(s.review);changed.invocation_annotations[G][core.KEY]=legacy
    # Historical input is seeded directly in this isolated fixture, since new
    # writes/imports correctly reject introducing a conflicting reference.
    with pytest.raises(ValueError,match='[Kk]onflikt'):s._save_change(changed)
    s.sidecar_path.write_text(json.dumps(changed.to_dict(),ensure_ascii=False),encoding='utf-8')
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    s=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    before=s.sidecar_path.read_bytes();summary=core.summary(s)
    assert stacks.snapshot(s,G)['conflicts']
    assert all(r['cer'] is None and r['gt_quad'] is None and r['not_measurable_reason']=='LEGACY_CONFLICT' for r in stacks.diagnostics(s)['rows'])
    with pytest.raises(ValueError,match='[Kk]onflikt'):stacks.associate(s,G,'GT:a1','GT:a0')
    assert s.sidecar_path.read_bytes()==before and core.summary(s)==summary

def test_T27_export_new_version_and_manifest_no_overwrite(tmp_path,stacks):
    s=linked(tmp_path/'data',stacks,n=2,suppressed=1);old=tmp_path/'exports';core.export(s,old)
    before={p.name:p.read_bytes() for p in old.iterdir()}
    a=stacks.export(s,old);b=stacks.export(s,old);assert a!=b
    data=json.loads((a/'mt_candidates.json').read_text(encoding='utf-8'))
    assert data['n_candidates']==2 and data['n_unique_plates']==1 and data['provenance'].startswith('ASSISTED')
    manifest=json.loads((a/'manifest.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((a/n).read_bytes()).hexdigest()==v for n,v in manifest['files'].items())
    assert all((old/n).read_bytes()==v for n,v in before.items())

def test_T28_native_sizes_focus_and_reset_shared_gt(stack_gui,stacks):
    from z4_sidebar_assertions import assert_sidebar_reaches
    root,h,ed,s=stack_gui;ed._choose_plate('GT:a0');ready(root,h,ed)
    before=s.sidecar_path.read_bytes()
    for size in ('1649x896','1180x760'):
        h.window.geometry(size+'+10+10');native.pump(root)
        stack_key(h,'z');stack_key(h,'x');native.key(h,'r');ready(root,h,ed)
        assert s.sidecar_path.read_bytes()==before  # R cannot destroy shared GT
        assert_sidebar_reaches(root,h,[*ed.contexts.fields.values(),*ed.contexts.counters.values()])
        assert h.canvas.winfo_width()>300 and len(h.canvas.find_withtag('corner_handle'))==4
    # Current Windows S/A/R guard intentionally uses the real pointer. Warp
    # only inside this disposable Tk window, then deliver real Tk bindings.
    def gesture(name):
        x,y=ed.to_canvas((225,175))
        h.canvas.event_generate('<Motion>',x=round(x),y=round(y),warp=True);native.pump(root)
        if name.startswith('Control-'):
            native.focus_canvas(h);symbol=name.split('-')[-1]
            h.canvas.event_generate('<KeyPress-'+symbol+'>',state=4,x=round(x),y=round(y));h.window.update()
            h.canvas.event_generate('<KeyRelease-'+symbol+'>',state=4,x=round(x),y=round(y));h.window.update()
        else:native.key(h,name,xy=(x,y))
        ready(root,h,ed)
    gesture('s');assert ed.selected==''
    gesture('s');assert ed.selected=='GT:a0'
    gesture('a');gesture('a');gesture('f')
    visible=ed.show_roi_overlay;gesture('t');assert ed.show_roi_overlay!=visible
    gesture('t');gesture('n');assert ed.selected=='GT:a0'
    h.subject_tree.focus_force();native.pump(root)
    selected=ed.selected;h.subject_tree.event_generate('<KeyPress-Tab>');native.pump(root)
    assert ed.selected==selected  # Tab outside Canvas is native focus traversal.
    gesture('Escape');assert not ed.selected
    assert s.sidecar_path.read_bytes()==before
    gesture('d')
    for point in MANUAL:native.click(h,ed,point)
    ready(root,h,ed);assert core.counts(core.scene(s,G))['expected']==2
    gesture('r');assert core.counts(core.scene(s,G))['expected']==1
    gesture('Control-z');assert core.counts(core.scene(s,G))['expected']==2
    gesture('Control-y');assert core.counts(core.scene(s,G))['expected']==1
