"""Fast current-state reads preserve history; report readiness is explicit."""
from pathlib import Path
import copy,json,shutil
import pytest
from test_z4_gtr_reapproval import verified,G,OID,pid
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
OUT=Path(__file__).resolve().parents[1]/'output/Z4_REVIEW_LATENCY_20261030_001'

def observed(tmp_path):
    tmp_path.mkdir(parents=True,exist_ok=True);side=tmp_path/'review.json';shutil.copyfile(OUT/'private/approval_observed.review.json',side)
    return MobileReviewSession.open(OUT/'private/day.alprsession',sidecar_path=side)

def test_current_state_view_matches_full_scene_without_copying_history(tmp_path):
    s=verified(tmp_path);full=core.scene(s,G);before=s.sidecar_path.read_bytes()
    thin=core.scene(s,G,include_history=False)
    assert stacks.HISTORY not in thin and stacks.HISTORY in full
    assert thin=={k:v for k,v in full.items() if k!=stacks.HISTORY}
    thin['objects'][0]['gt_text']='MUTATED';thin['parents'][0]['quad'][0][0]+=1
    assert core.scene(s,G)==full and s.sidecar_path.read_bytes()==before

def test_UI_queries_do_not_walk_historical_snapshots(tmp_path):
    from auto_annotation_tool.ranking.mobile_canvas_presentation import build_inventory
    class Historical(dict):
        def __deepcopy__(self,memo):raise AssertionError('Read-only UI copied audit history')
    s=verified(tmp_path);value=s.review.invocation_annotations[G][core.KEY]
    value[stacks.HISTORY]=Historical(value[stacks.HISTORY])
    core.group_issues(s,G);build_inventory(s);stacks.diagnostics(s)

def test_lightweight_actions_keep_append_only_history_and_full_disk_snapshot(tmp_path):
    s=verified(tmp_path);before=copy.deepcopy(s.review.to_dict());entries=copy.deepcopy(core.scene(s,G)[stacks.HISTORY]['entries'])
    for _ in range(4):partial.toggle_stack(s,G,oid=OID)
    assert core.scene(s,G)[stacks.HISTORY]['entries'][:len(entries)]==entries
    assert len(core.scene(s,G)[stacks.HISTORY]['entries'])==len(entries)+4
    assert json.loads(s.sidecar_path.read_text(encoding='utf-8'))==json.loads(json.dumps(s.review.to_dict()))
    assert before['review_revision']==s.review.review_revision-4
    assert stacks.restore_history(s,G) and not partial.progress(core.scene(s,G))['complete']
    assert stacks.restore_history(s,G,redo=True) and partial.progress(core.scene(s,G))['complete']

def test_actual_all_verified_session_can_be_approved_without_fabricating_Pose(tmp_path):
    from auto_annotation_tool.ranking.mobile_verification_policy import verification_issues
    from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import evaluate_unit
    s=observed(tmp_path);g=s.session_id+'-a00000008';before=copy.deepcopy(core.scene(s,g));metric=evaluate_unit(s,g)
    assert not verification_issues(s)
    assert core.group_issues(s,g)  # quantitative geometry still unavailable
    s.complete();assert s.review.review_status=='COMPLETED'
    assert core.scene(s,g)==before
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    from auto_annotation_tool.ranking.mobile_verification_snapshot import has_valid_ready_snapshot
    assert has_valid_ready_snapshot(reopened)
    now=evaluate_unit(s,g)
    assert {k:now[k] for k in ('tp','fp','fn','n_gt','n_final_mt','unavailable_reasons')}=={k:metric[k] for k in ('tp','fp','fn','n_gt','n_final_mt','unavailable_reasons')}

def test_missing_GTZ_and_removed_operator_geometry_still_block_report(tmp_path):
    from auto_annotation_tool.ranking.mobile_verification_policy import verification_issues
    s=observed(tmp_path);g=s.session_id+'-a00000009';oid='GT:'+g
    stacks.edit_gt(s,g,oid,text='');assert verification_issues(s)
    with pytest.raises(ValueError):s.complete()
    stacks.restore_history(s,g);core.remove_operator_geometry(s,g,oid)
    assert verification_issues(s)
    with pytest.raises(ValueError):s.complete()

def test_list_marks_real_missing_data_instead_of_verified(tmp_path):
    from auto_annotation_tool.gui.mobile_canvas_editor import CanvasReviewEditor
    from auto_annotation_tool.ranking.mobile_canvas_presentation import build_inventory
    s=observed(tmp_path);g=s.session_id+'-a00000009';stacks.edit_gt(s,g,'GT:'+g,text='')
    inv=build_inventory(s);row=inv['groups'][g]
    assert row['approval_issues'] and CanvasReviewEditor._image_action([row])==('Uzupełnij dane','canvas_pending')
    assert inv['approval_remaining']==1 and inv['approval_ready']==20
    assert inv['approval_decisions']==1


def test_copy_on_write_never_mutates_prior_review_versions(tmp_path):
    s=observed(tmp_path);versions=[]
    for number in (9,15,9,15):
        versions.append((s.review,copy.deepcopy(s.review.to_dict())))
        g=s.session_id+f'-a{number:08d}';partial.toggle_stack(s,g,oid='GT:'+g)
    assert all(review.to_dict()==frozen for review,frozen in versions)
    assert json.loads(s.sidecar_path.read_text(encoding='utf-8'))==json.loads(json.dumps(s.review.to_dict()))


def test_failed_atomic_write_preserves_memory_disk_and_history(tmp_path,monkeypatch):
    from auto_annotation_tool.ranking import mobile_human_review as human
    s=verified(tmp_path);before=s.sidecar_path.read_bytes();saved=copy.deepcopy(s.review.to_dict())
    def fail(*args):raise OSError('simulated storage failure')
    monkeypatch.setattr(human,'_atomic_write',fail)
    with pytest.raises(OSError):partial.toggle_stack(s,G,oid=OID)
    assert s.sidecar_path.read_bytes()==before and s.review.to_dict()==saved


def test_concurrent_revision_still_refuses_stale_write(tmp_path):
    s=verified(tmp_path);other=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    partial.toggle_stack(other,G,oid=OID);before=s.sidecar_path.read_bytes();saved=copy.deepcopy(s.review.to_dict())
    with pytest.raises(ValueError,match='innym oknie'):partial.toggle_stack(s,G,oid=OID)
    assert s.sidecar_path.read_bytes()==before and s.review.to_dict()==saved


def test_light_read_still_checks_source_binding(tmp_path):
    s=verified(tmp_path);s.review.invocation_annotations[G][core.KEY]['source_binding']='wrong-source'
    with pytest.raises(ValueError):core.scene(s,G,include_history=False)


@pytest.fixture
def gui(tmp_path,monkeypatch):
    import test_mobile_p2b2_canvas_gui as native
    s=observed(tmp_path/'data');monkeypatch.setattr(native,'source_session',lambda _:s);cls=native.module.MobileSampleReviewWindow
    def shown(*a,**kw):
        h=cls(*a,**kw);h.window.state('normal');h.window.deiconify();return h
    monkeypatch.setattr(native.module,'MobileSampleReviewWindow',shown)
    yield from native.gui.__wrapped__(tmp_path,monkeypatch)


def test_native_report_button_accepts_actual_all_verified_session(gui,monkeypatch):
    import tkinter.messagebox as boxes
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    from auto_annotation_tool.ranking.mobile_verification_snapshot import has_valid_ready_snapshot
    root,h,ed,s=gui;ready(root,h,ed)
    assert all('Zweryfikowany' in h.subject_tree.item(i,'text') for i in h.subject_tree.get_children(''))
    prompts=[];monkeypatch.setattr(boxes,'askyesno',lambda *a,**k:(prompts.append(a) or True))
    h.complete();native.pump(root,lambda:not h._saving and s.review.review_status=='COMPLETED',seconds=20);ready(root,h,ed)
    assert len(prompts)==1 and has_valid_ready_snapshot(s)


def test_native_missing_GTZ_is_visible_and_next_pending_finds_it(gui,monkeypatch):
    import tkinter.messagebox as boxes
    import test_mobile_p2b2_canvas_gui as native
    from test_z4_frame_stacks import ready
    root,h,ed,s=gui;g=s.session_id+'-a00000009';stacks.edit_gt(s,g,'GT:'+g,text='');ed.refresh_tree();ready(root,h,ed)
    assert any('Uzupełnij dane' in h.subject_tree.item(i,'text') for i in h.subject_tree.get_children(''))
    monkeypatch.setattr(boxes,'askyesno',lambda *a,**k:pytest.fail('Incomplete review cannot be approved'))
    h.complete();assert 'niekompletna' in h.status_var.get().lower()
    ed.next_pending();native.pump(root,lambda:h.current_invocation==g and not h._saving);ready(root,h,ed)
    assert ed.inventory()['approval_remaining']==1


def test_header_reuses_current_revision_only_and_approval_revalidates(gui,monkeypatch):
    from auto_annotation_tool.ranking import mobile_verification_policy as policy
    root,h,ed,s=gui;original=policy.verification_issues;calls=[]
    def counted(session):calls.append(session.review.review_revision);return original(session)
    monkeypatch.setattr(policy,'verification_issues',counted);ed._header_validation_key=None
    ed.update_header();ed.update_header();assert len(calls)==1
    g=s.session_id+'-a00000009';stacks.edit_gt(s,g,'GT:'+g,text='')
    ed.update_header();assert len(calls)==2 and not ed._header_complete
    with pytest.raises(ValueError):s.complete()
    assert len(calls)==3
