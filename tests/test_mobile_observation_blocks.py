"""Navigation-only observation intervals. Source scopes and metrics stay separate."""
import copy
import hashlib
import json
import os
import shutil
from types import SimpleNamespace as NS

import pytest

from auto_annotation_tool.gui import mobile_canvas_observation_blocks as blocks


def synthetic(count=10):
    sid='synthetic-observation';rows=[]
    for i in range(count):
        stamp=dict(scene_generation=i+1,visual_epoch=i+1,camera_transform_generation=0,
            source_sequence=100+i*100,source_timestamp_nanos=1000000+i*100000,source_timestamp_domain='CAMERAX_SENSOR')
        rows.append(dict(scene_event_id=f'{sid}-s{i+1:08d}',session_id=sid,
            event_type='SESSION_START' if i==0 else 'SCENE_BOUNDARY',event_elapsed_nanos=1000+i*30000000000,
            event_stamp=stamp,source_stamp=dict(stamp,source_sequence=stamp['source_sequence']+1),
            mt_invocation_ids=[],write_status='WRITTEN'))
    bundle=NS(validation=NS(ok=True),report_payload={'analysis_mode':'static'},
        collection_session=dict(execution={'analysis_mode':'static'},scene_evidence_complete=True,
            scene_evidence_pending_loss_unknown=False,scene_evidence_expected=count,
            started_elapsed_nanos=1000,finished_elapsed_nanos=1000+count*30000000000),
        scene_evidence=dict(available=True,scenes=rows))
    return NS(session_id=sid,bundle=bundle,scene_frames={r['scene_event_id']:r for r in rows},
              mt_invocations={},mt_detection_roles={},attempts={},subjects={},samples={})


def add_call(s,index=0,*,detections=1):
    scene=list(s.scene_frames.values())[index];gid='call-'+str(len(s.mt_invocations)+1)
    stamp=dict(scene['event_stamp']);stamp['source_sequence']+=10;stamp['source_timestamp_nanos']+=100
    records=[]
    for i in range(max(1,detections)):
        aid=gid+'-det-'+str(i)
        row=dict(stamp,id=aid,attempt_id=aid,mt_invocation_id=gid,mt_detection_index=i if detections else None,
                 f5a_scene_event_id=scene['scene_event_id'],mz_executed=bool(detections))
        records.append(row);s.attempts[aid]=row
    s.mt_invocations[gid]=NS(records=records,executed=True,cancelled=False,execution_failed=False)
    s.mt_detection_roles[gid]=dict(mt_invocation_id=gid,scene_event_id=scene['scene_event_id'],source_stamp=stamp,
        raw_attempt_ids=[r['id'] for r in records] if detections else [],role_proof_status='COMPLETE',
        postprocess_status='COMPLETED',final_attempt_ids=[r['id'] for r in records] if detections else [])
    scene['mt_invocation_ids'].append(gid)
    return gid


def test_nine_boundaries_give_ten_observation_intervals_without_stability_claim():
    s=synthetic();m=blocks.build_blocks(s)
    assert m['enabled'] and len(m['blocks'])==10
    assert m['blocks'][0]['start_ns']==1000 and m['blocks'][-1]['end_ns']==300000001000
    assert all(b['stability'] is None for b in m['blocks'])
    assert all(not b['calls'] for b in m['blocks'])


@pytest.mark.parametrize('count',[1,3,12])
def test_block_population_comes_from_evidence_not_ten(count):
    assert len(blocks.build_blocks(synthetic(count))['blocks'])==count


def test_multiple_invocations_and_subjects_never_duplicate_a_call():
    s=synthetic();one=add_call(s,detections=2);two=add_call(s)
    s.subjects={'first':{'attempts':[r['id'] for r in s.mt_invocations[one].records]},
                'second':{'attempts':[r['id'] for r in s.mt_invocations[one].records]}}
    before=copy.deepcopy(s.attempts);m=blocks.build_blocks(s)
    assert m['blocks'][0]['calls']==[one,two] and s.attempts==before
    assert m['blocks'][3]['calls']==[]


def test_executed_mt_with_no_detections_is_still_a_call():
    s=synthetic();gid=add_call(s,detections=0)
    assert blocks.build_blocks(s)['blocks'][0]['calls']==[gid]


def test_cancelled_late_result_stays_with_original_source_scene():
    s=synthetic();gid=add_call(s);g=s.mt_invocations[gid]
    g.cancelled=True;s.mt_detection_roles[gid].update(role_proof_status='INCOMPLETE',postprocess_status='NOT_REACHED',
        stale_or_cancelled=True,cancel_reason='scene_superseded',result_available_nanos=999999999999)
    m=blocks.build_blocks(s)
    assert m['blocks'][0]['calls']==[gid] and not m['blocks'][1]['calls']
    assert blocks._call_state(s,gid,{'groups':{}})['incomplete']


def test_different_bitmap_stamps_are_not_required_to_match_representative():
    s=synthetic();gid=add_call(s)
    assert s.mt_detection_roles[gid]['source_stamp']!=next(iter(s.scene_frames.values()))['source_stamp']
    assert blocks.build_blocks(s)['blocks'][0]['calls']==[gid]


@pytest.mark.parametrize('damage',['missing_role','wrong_scene','wrong_stamp','duplicate_owner','missing_attempt'])
def test_broken_membership_is_quarantined_and_retained(damage):
    s=synthetic();gid=add_call(s);role=s.mt_detection_roles[gid]
    if damage=='missing_role':s.mt_detection_roles.clear()
    elif damage=='wrong_scene':role['scene_event_id']=list(s.scene_frames)[1]
    elif damage=='wrong_stamp':role['source_stamp']['source_sequence']+=1
    elif damage=='duplicate_owner':list(s.scene_frames.values())[1]['mt_invocation_ids'].append(gid)
    elif damage=='missing_attempt':s.attempts.clear()
    m=blocks.build_blocks(s)
    assert m['enabled'] and gid in m['unassigned']
    assert not any(gid in b['calls'] for b in m['blocks']) and gid in s.mt_invocations


@pytest.mark.parametrize('damage',['missing_end','equal_time','missing_scene','duplicate_scene','wrong_start','incomplete','missing_mode','dynamic','mode_conflict','same_generation'])
def test_ambiguous_or_dynamic_sessions_keep_legacy_navigation(damage):
    s=synthetic();data=s.bundle.collection_session;rows=s.bundle.scene_evidence['scenes']
    if damage=='missing_end':data.pop('finished_elapsed_nanos')
    elif damage=='equal_time':rows[1]['event_elapsed_nanos']=rows[0]['event_elapsed_nanos']
    elif damage=='missing_scene':rows.pop(2)
    elif damage=='duplicate_scene':rows[1]=rows[0]
    elif damage=='wrong_start':rows[0]['event_type']='SCENE_BOUNDARY'
    elif damage=='incomplete':data['scene_evidence_pending_loss_unknown']=True
    elif damage=='missing_mode':data['execution']={};s.bundle.report_payload={}
    elif damage=='dynamic':data['execution']['analysis_mode']='dynamic';s.bundle.report_payload['analysis_mode']='dynamic'
    elif damage=='mode_conflict':s.bundle.report_payload['analysis_mode']='dynamic'
    elif damage=='same_generation':rows[1]['event_stamp']=dict(rows[0]['event_stamp'])
    assert not blocks.build_blocks(s)['enabled']


def test_child_without_scene_id_uses_explicit_parent_and_matching_stamp():
    s=synthetic();gid=add_call(s,detections=2);s.attempts[s.mt_invocations[gid].records[1]['id']].pop('f5a_scene_event_id')
    assert blocks.build_blocks(s)['blocks'][0]['calls']==[gid]


def test_incomplete_final_role_is_na_not_a_final_detection_or_a_human_decision():
    s=synthetic();gid=add_call(s,detections=2)
    s.mt_detection_roles[gid].update(role_proof_status='INCOMPLETE',final_attempt_ids=[],unresolved_attempt_ids=list(s.attempts))
    state=blocks._call_state(s,gid,{'groups':{}})
    assert state['incomplete'] and not state['complete'] and state['pending']==state['saved']==0
    assert blocks.build_blocks(s)['blocks'][0]['calls']==[gid]


@pytest.mark.parametrize('persisted,empty,complete',[(False,False,False),(True,False,False),(True,True,True)])
def test_technical_na_is_not_an_operator_decision(persisted,empty,complete):
    s=synthetic();gid=add_call(s,detections=0)
    inventory={'groups':{gid:dict(persisted=persisted,review_tasks={'needs_review':False},
        frame_review=dict(state='NO_REFERENCE_EVIDENCE',complete=True,empty_confirmed=empty,resolved=2))}}
    state=blocks._call_state(s,gid,inventory)
    assert state['complete']==complete
    assert state['saved']==(2 if persisted else 0)


@pytest.fixture
def k02(tmp_path):
    source=os.environ.get('ALPR_OBSERVATION_K02')
    if not source:pytest.skip('Private K02 acceptance fixture not supplied')
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    path=tmp_path/'source.alprsession';shutil.copy2(source,path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='140c20115cf34c67de7e591752b540d152ec07fea2bf7488a8546e97d8899a10'
    s=MobileReviewSession.open(path,sidecar_path=tmp_path/'review.json')
    s.set_review_mode('assisted')
    return s


def test_private_k02_complete_identity_and_metric_invariance(k02):
    s=k02;before=copy.deepcopy(s.review.to_dict());stats=s.statistics();source_sha=hashlib.sha256(s.path.read_bytes()).hexdigest()
    m=blocks.build_blocks(s)
    assert m['enabled'] and len(m['blocks'])==10 and not m['unassigned']
    assert sum(len(b['calls']) for b in m['blocks'])==len(s.mt_invocations)==44
    assert sum(blocks._true(a.get('mz_executed')) for a in s.attempts.values())==51
    assert len(s.subjects)==10 and m['blocks'][3]['calls']==m['blocks'][4]['calls']==[]
    assert sum(len(r['final_attempt_ids']) for r in s.mt_detection_roles.values())==53
    assert sum(len(r['unresolved_attempt_ids']) for r in s.mt_detection_roles.values())==4
    assert s.review.to_dict()==before and s.statistics()==stats
    assert hashlib.sha256(s.path.read_bytes()).hexdigest()==source_sha


def test_native_k02_navigation_separate_bindings_save_resume(k02,tmp_path,monkeypatch):
    from windows_dpi import enable_windows_dpi_awareness
    enable_windows_dpi_awareness()
    import test_mobile_p2b2_canvas_gui as native
    from auto_annotation_tool.ranking import mobile_scene_frame_review as scenes
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    s=k02;monkeypatch.setattr(native,'source_session',lambda _:s)
    generator=native.gui.__wrapped__(tmp_path,monkeypatch)
    root,h,ed,s=next(generator)
    try:
        tree=h.subject_tree;m=ed.observation_model
        assert m and len(tree.get_children(''))==10 and len(ed.observation_mt_ids)==44
        assert all(tree.item(i,'text').startswith('Scena ') for i in tree.get_children(''))
        assert len([i for i in ed.observation_targets if i.startswith('obs:mz:')])==51
        first=m['blocks'][0];sid=first['scene_id']
        gid=next(g for b in m['blocks'] for g in b['calls'] if len(s.mt_detection_roles[g]['final_attempt_ids'])>1)
        assert h.current_record==('scene',sid)
        original=copy.deepcopy(s.review.to_dict());stats=s.statistics();sidecar=s.sidecar_path.read_bytes()
        scene_binding=ed._read()['source_binding'];scene_image=h._image_entry
        rootid='obs:scene:'+sid;tree.item(rootid,open=True)
        # Same selection and refresh preserve Canvas viewport and expansion.
        h._view_zoom=2.;h._view_center=(410.,520.);ed.refresh_tree();root.update()
        assert tree.item(rootid,'open') and h._view_zoom==2. and h._view_center==(410.,520.)
        ed.open_result_image(gid)
        native.pump(root,lambda:h.current_invocation==gid and h._image_entry==s.mt_detection_roles[gid]['source_frame_entry'])
        assert ed._read()['source_binding']!=scene_binding and h._image_entry!=scene_image
        assert not ed._read().get('scope')=='SCENE_FRAME'
        mz=next(i for i,t in reversed(list(ed.observation_targets.items())) if t and t[0]=='mz' and t[1]==gid)
        ed.select_tree_row(mz);root.update()
        assert h.current_invocation==gid and tree.selection()==(mz,)
        from auto_annotation_tool.gui.mobile_canvas_navigator import finals
        choice=next(c for c in finals(ed) if c['attempt_id']==ed.observation_targets[mz][2])
        assert ed.selected==choice['object_id']
        viewport=(h._view_zoom,h._view_center)
        ed.refresh_tree();root.update();assert tree.selection()==(mz,)
        assert (h._view_zoom,h._view_center)==viewport and ed.selected==choice['object_id']
        cancelled=next(g for g,r in s.mt_detection_roles.items() if r['stale_or_cancelled'])
        ed.open_result_image(cancelled);native.pump(root,lambda:h.current_invocation==cancelled)
        assert s.review.to_dict()==original and s.statistics()==stats and s.sidecar_path.read_bytes()==sidecar
        # Zero-MT scene remains navigable and needs its own explicit decision.
        zero=m['blocks'][3]['scene_id'];ed.open_result_image(zero)
        native.pump(root,lambda:h.current_record==('scene',zero) and ed.skey==zero)
        assert scenes.scene(s,zero)['presence']=='UNRESOLVED'
        ed.confirm_empty_image();root.update()
        assert scenes.scene(s,zero)['presence']=='NO_PLATES'
        assert len(s.review.invocation_annotations)==len(original['invocation_annotations'])
        # A positive scene decision does not approve its independent MT units.
        ed.open_result_image(sid);native.pump(root,lambda:h.current_record==('scene',sid) and ed.skey==sid)
        ed.confirm_empty_image();root.update()
        assert 'MT do oceny' in tree.item(rootid,'text')
        ed.undo();assert scenes.scene(s,sid)['presence']=='UNRESOLVED'
        ed.undo(redo=True);assert scenes.scene(s,sid)['presence']=='NO_PLATES'
        reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
        assert reopened.review.to_dict()==json.loads(json.dumps(s.review.to_dict())) and blocks.build_blocks(reopened)['enabled']
        # Queue clicks through the existing saving lifecycle, never lose intent.
        target='obs:scene:'+m['blocks'][4]['scene_id'];h._saving=True
        ed.select_tree_row(target);assert h.current_invocation==sid
        h._saving=False;blocks.resume(ed)
        native.pump(root,lambda:h.current_invocation==m['blocks'][4]['scene_id'] and ed.skey==h.current_invocation
            and h._image_entry==s.scene_frames[h.current_invocation]['source_frame_entry'])
        # Existing D/inline/Space/undo continue on the selected full source.
        native.key(h,'d')
        for point in ((100,200),(300,200),(300,350),(100,350)):native.click(h,ed,point)
        drawn=scenes.scene(s,h.current_invocation)
        assert len(drawn['objects'])==1
        ed.start_entry(drawn['objects'][0]['object_id'])
        assert ed.entry is not None
        ed.entry.delete(0,'end');ed.entry.insert(0,'SYNTHETIC');ed.entry_typed()
        native.pump(root,lambda:scenes.scene(s,h.current_invocation)['objects'][0]['gt_text']=='SYNTHETIC')
        ed.entry_enter()
        confirmed=scenes.scene(s,h.current_invocation)['objects'][0]['scene_confirmed']
        native.key(h,'space')
        assert scenes.scene(s,h.current_invocation)['objects'][0]['scene_confirmed']!=confirmed
        assert scenes.scene(s,h.current_invocation)['objects'][0]['gt_text']=='SYNTHETIC'
        ed.navigate(1);native.pump(root,lambda:h.current_invocation==m['blocks'][5]['scene_id'])
        saved=json.loads(json.dumps(s.review.to_dict()))
    finally:
        h._saving=False
        try:next(generator)
        except StopIteration:pass
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    monkeypatch.setattr(native,'source_session',lambda _:reopened)
    generator=native.gui.__wrapped__(tmp_path,monkeypatch)
    root,h,ed,s=next(generator)
    try:
        assert s.review.to_dict()==saved
        assert ed.observation_model and h.current_record==('scene',sid)
        assert scenes.scene(s,zero)['presence']=='NO_PLATES'
        assert 'MT do oceny' in h.subject_tree.item(rootid,'text')
    finally:
        try:next(generator)
        except StopIteration:pass


def test_native_diagnostics_and_dynamic_fallback(k02,tmp_path,monkeypatch):
    import test_mobile_p2b2_canvas_gui as native
    s=k02;monkeypatch.setattr(native,'source_session',lambda _:s)
    generator=native.gui.__wrapped__(tmp_path,monkeypatch)
    root,h,ed,s=next(generator)
    try:
        before=copy.deepcopy(s.review.to_dict());sidecar=s.sidecar_path.read_bytes()
        gid=next(iter(s.mt_invocations))
        role=s.mt_detection_roles.pop(gid)
        ed.refresh_tree();root.update()
        assert h.subject_tree.exists('obs:unassigned')
        assert h.subject_tree.parent('obs:mt:'+gid)=='obs:unassigned'
        assert len(ed.observation_mt_ids)==44
        assert not any(gid in b['calls'] for b in ed.observation_model['blocks'])
        orphan=next(i for i in ed.observation_targets if i.startswith('obs:orphan-sample:'))
        h.subject_tree.selection_set(orphan);ed.select_tree_row(orphan);root.update()
        assert h.current_record==('sample',ed.observation_targets[orphan][1])
        assert h.subject_tree.selection()==(orphan,)
        ed.refresh_tree();root.update();assert h.subject_tree.selection()==(orphan,)
        s.mt_detection_roles[gid]=role
        ed.refresh_tree();root.update()
        assert not h.subject_tree.exists('obs:unassigned')
        # A missing declared call remains diagnostic and cannot turn a block
        # with a confirmed full frame into a wholly completed observation.
        sid=ed.observation_model['blocks'][3]['scene_id']
        original_scene=blocks.scene_review.scene
        checked=dict(original_scene(s,sid),checked=True,presence='NO_PLATES')
        registry_row=next(r for r in s.bundle.scene_evidence['scenes'] if r['scene_event_id']==sid)
        rows={id(r):r for r in (registry_row,s.scene_frames[sid])}.values()
        for row in rows:row['mt_invocation_ids'].append('missing-call')
        with monkeypatch.context() as patch:
            patch.setattr(blocks.scene_review,'scene',lambda session,key,**kw:checked if key==sid else original_scene(session,key,**kw))
            ed.refresh_tree()
            assert h.subject_tree.exists('obs:orphan-call:missing-call')
            assert 'Niepełne' in h.subject_tree.item('obs:scene:'+sid,'text')
        for row in rows:row['mt_invocation_ids'].remove('missing-call')
        s.bundle.collection_session['execution']['analysis_mode']='dynamic'
        ed.refresh_tree();root.update()
        assert not ed.observation_model
        assert not any(i.startswith('obs:') for i in h.subject_tree.get_children())
        assert len(ed.scene_tree_ids)==10 and h.subject_tree.heading('#0','text')=='Obraz  ·  Co zrobić'
        assert s.review.to_dict()==before and s.sidecar_path.read_bytes()==sidecar
    finally:
        try:next(generator)
        except StopIteration:pass
