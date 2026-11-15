"""Creation guards must neither duplicate a manual GTR nor change existing data."""
from pathlib import Path
import copy,hashlib,json
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_mobile_p2b2_canvas_review import source_session,MANUAL
from test_z4_frame_stacks import ready
from test_z4_gtr_gtz_guides import gui,private_case,reference_quad,choose,G,OID
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/Z4_MANUAL_FRAME_GUARD_20261027_001'

def rect(l,t,r,b):return [[l,t],[r,t],[r,b],[l,b]]

def seeded(path):
    s=source_session(path);oid=core.add_plate(s,'mt-one',MANUAL,partial_review=True)
    return s,oid

@pytest.mark.parametrize('quad',[
    MANUAL,rect(675,360,775,410),rect(640,340,810,430),
    rect(750,400,840,470),rect(800,350,850,420),
],ids=['same','inside','enclosing','partial','edge_touch'])
def test_direct_creation_overlap_is_rejected_without_write(tmp_path,quad):
    s,_=seeded(tmp_path);frozen=s.sidecar_path.read_bytes();before=copy.deepcopy(s.review.to_dict())
    with pytest.raises(ValueError,match='GTR 1'):
        core.add_plate(s,'mt-one',quad,partial_review=True)
    assert s.sidecar_path.read_bytes()==frozen and s.review.to_dict()==before

@pytest.mark.parametrize('quad',[rect(625,375,825,390),rect(650,500,800,570)],ids=['crossing_no_vertex_inside','separate'])
def test_crossing_or_separate_reference_remains_allowed(tmp_path,quad):
    s,oid=seeded(tmp_path);rev=s.review.review_revision
    new=core.add_plate(s,'mt-one',quad,partial_review=True)
    v=stacks.snapshot(s,'mt-one')
    assert new!=oid and len(v['plates'])==2 and s.review.review_revision==rev+1

def test_bbox_overlap_without_polygon_overlap_is_allowed(tmp_path):
    s=source_session(tmp_path)
    core.add_plate(s,'mt-one',[[650,400],[700,350],[750,400],[700,450]],partial_review=True)
    core.add_plate(s,'mt-one',rect(650,350,675,360),partial_review=True)
    assert len(stacks.snapshot(s,'mt-one')['plates'])==2

def test_model_outline_is_not_a_manual_obstacle_or_an_identity_link(tmp_path):
    s=source_session(tmp_path);before=stacks.snapshot(s,'mt-one')['links']
    quad=core.scene(s,'mt-one')['parents'][0]['quad']
    core.add_plate(s,'mt-one',quad,partial_review=True)
    after=stacks.snapshot(s,'mt-one')
    assert after['links']['GT:a0']==before['GT:a0'] and len(after['plates'])==1

def test_linked_manual_reference_remains_protected(tmp_path):
    s=source_session(tmp_path)
    quad=core.scene(s,'mt-one')['parents'][0]['quad']
    oid=core.add_plate(s,'mt-one',quad,partial_review=True)
    stacks.associate(s,'mt-one','GT:a0',oid);frozen=s.sidecar_path.read_bytes()
    with pytest.raises(ValueError,match='GTR 1'):core.add_plate(s,'mt-one',quad,partial_review=True)
    assert s.sidecar_path.read_bytes()==frozen

def test_removing_a_manual_frame_releases_its_area(tmp_path):
    s,oid=seeded(tmp_path);core.remove_operator_geometry(s,'mt-one',oid)
    new=core.add_plate(s,'mt-one',MANUAL,partial_review=True)
    assert new!=oid and len(stacks.snapshot(s,'mt-one')['plates'])==1

def test_editing_existing_corners_is_not_blocked_as_new_creation(tmp_path):
    s,oid=seeded(tmp_path);second=core.add_plate(s,'mt-one',rect(650,500,800,570),partial_review=True)
    before=stacks.snapshot(s,'mt-one');rev=s.review.review_revision
    core.correct_plate(s,'mt-one',oid,rect(650,490,800,560))
    after=stacks.snapshot(s,'mt-one')
    assert set(after['plates'])==set(before['plates']) and s.review.review_revision==rev+1
    assert second in {p['origin_object_id'] for p in after['plates'].values()}

def test_bbox_completion_cannot_create_a_second_manual_reference(tmp_path):
    s=private_case(tmp_path,keep_manual=True);frozen=s.sidecar_path.read_bytes()
    pid=stacks.snapshot(s,G)['links'][OID]['plate_id']
    with pytest.raises(ValueError,match='GTR 1'):
        stacks.complete_bbox_reference(s,G,OID,reference_quad(),expected_plate_id=pid)
    assert s.sidecar_path.read_bytes()==frozen

@pytest.mark.parametrize('gui',[True],indirect=True)
def test_click_inside_existing_manual_GTR_does_not_start_another(gui):
    root,h,ed,s=gui;frozen=s.sidecar_path.read_bytes();native.key(h,'d')
    point=tuple(sum(p[i] for p in reference_quad())/4 for i in (0,1))
    native.click(h,ed,point)
    assert ed.points==[] and 'GTR 1' in h.status_var.get()
    assert s.sidecar_path.read_bytes()==frozen

@pytest.mark.parametrize('gui',[True],indirect=True)
def test_enclosing_quad_rejected_before_save_retains_three_points(gui):
    root,h,ed,s=gui;frozen=s.sidecar_path.read_bytes();native.key(h,'d')
    for point in rect(540,600,645,640):native.click(h,ed,point)
    ready(root,h,ed)
    assert len(ed.points)==3 and 'GTR 1' in h.status_var.get()
    assert s.sidecar_path.read_bytes()==frozen

@pytest.mark.parametrize('gui',[True],indirect=True)
def test_actual_corner_drag_still_edits_the_same_reference(gui):
    root,h,ed,s=gui
    manual=next(o for o in ed._read()['objects'] if not o['parent_id'])
    ed._choose_plate(manual['object_id']);ready(root,h,ed)
    before=stacks.snapshot(s,G);revision=s.review.review_revision
    x,y=map(round,ed.to_canvas(manual['quad'][0]))
    h.canvas.event_generate('<ButtonPress-1>',x=x,y=y)
    h.canvas.event_generate('<B1-Motion>',x=x-2,y=y-1)
    h.canvas.event_generate('<ButtonRelease-1>',x=x-2,y=y-1);ready(root,h,ed)
    assert set(stacks.snapshot(s,G)['plates'])==set(before['plates']) and s.review.review_revision==revision+1

def test_current_twelve_sessions_and_reports_are_preserved():
    b=json.loads((OUT/'baseline.json').read_text(encoding='utf-8'))
    for n,v in b['protected'].items():
        p=Path(n);assert p.exists()==v['exists']
        if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
    assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in b['prior_reports'].items())
