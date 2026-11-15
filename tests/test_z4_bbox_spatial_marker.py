"""V/S/O/M contracts: source zero-area markers and independent operator evidence."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,json,shutil
import pytest
from test_z4_space_gtr_hud_shift import fixture_archive
from test_z4_auto_spatial_gtr import editor
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial,mobile_mt_detection_evaluation as mt,mobile_mt_geometry_layers as geom
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.gui import mobile_canvas_bbox_marker as marker,mobile_bbox_spatial_review as spatial,mobile_canvas_geometry_layers as layers

OUT=Path(__file__).resolve().parents[1]/'output/Z4_ZERO_BBOX_OPERATOR_20261031_001'
SID='exp-1792699864138-add3e282';G='mt-one';OID='GT:a0'

def source(path,bbox=(150,170,300,170),n=1,mz=True):
    def change(rows,trace,payload):
        for i,r in enumerate(rows):
            r.update(zip(('plate_left','plate_top','plate_right','plate_bottom'),[v+(220*i if k%2==0 else 0) for k,v in enumerate(bbox)]))
            r['mt_confidence']=.9
    s=fixture_archive(path,n=n,mz=mz,mutate=change);s.set_review_mode('assisted');return s

def desc(s,g=G):
    sc=core.scene(s,g);return marker.describe(s,g,sc,stacks.snapshot(s,g)['candidates'][0])

@pytest.fixture
def actual(tmp_path):
    src=OUT/'private'/SID
    assert src.exists(),'Required private F4 source fixture missing'
    shutil.copyfile(src/'source.alprsession',tmp_path/'source.alprsession');shutil.copyfile(src/'review.json',tmp_path/'review.json')
    return MobileReviewSession.open(tmp_path/'source.alprsession',sidecar_path=tmp_path/'review.json')

def freeze(s):return copy.deepcopy(s.review.to_dict()),s.sidecar_path.read_bytes(),mt.evaluate_session(s),stacks.diagnostics(s)

def test_V01_S01_S05_M02_actual_exact_source_layers_and_metrics(actual):
    s=actual;gid=SID+'-a00000001';before=freeze(s);d=desc(s,gid)
    assert d['bbox']==[524.73663,392.71658,585.15454,392.71658] and d['validity']=='INVALID_ZERO_HEIGHT'
    value=layers.model(editor(s,gid));assert [l['kind'] for l in value['layers']]==['BBOX','POSE','GTR']
    assert len(stacks.snapshot(s,gid)['candidates'])==1 and value['gtr_count']==1
    assert geom.source_bbox(s,core.scene(s,gid),stacks.snapshot(s,gid)['candidates'][0]) is None
    unit=mt.evaluate_unit(s,gid);assert [unit[k] for k in ('tp','fp','fn')]==[None]*3
    diagnostic=stacks.diagnostics(s,[gid])['rows'][0]
    assert diagnostic['polygon_iou_pct']==pytest.approx(50.435609375923974)
    assert diagnostic['nmce_pct']==pytest.approx(12.998639100900974)
    assert freeze(s)==before and not spatial.path(s).exists()

@pytest.mark.parametrize('bbox,status',[( (150,170,300,170),'ZERO_HEIGHT'),((150,140,150,210),'ZERO_WIDTH'),((150,170,150,170),'POINT'),((150,140,300,210),'VALID'),((-1,170,300,170),'OUT_OF_BOUNDS'),((300,170,150,170),'REVERSED')])
def test_V02_V03_V04_V08_source_geometry(tmp_path,bbox,status):
    s=source(tmp_path,bbox);d=desc(s);assert d['validity']==('INVALID_'+status if status in ('ZERO_HEIGHT','ZERO_WIDTH','POINT') else status) and d['bbox']==list(bbox)
    assert d['marker']==(status in ('ZERO_HEIGHT','ZERO_WIDTH','POINT'))

@pytest.mark.parametrize('value',[None,'','nan','inf','-inf','not-a-number'])
def test_V05_V06_no_invented_point_or_verdict(tmp_path,value):
    s=source(tmp_path);s.attempts['a0']['plate_top']=value;before=s.sidecar_path.read_bytes()
    d=desc(s);assert not d['marker'] and d['validity'] in ('MISSING','INVALID_NONFINITE')
    with pytest.raises(ValueError):spatial.set_verdict(s,G,OID,'OFF_PLATES_BAD')
    assert s.sidecar_path.read_bytes()==before and not spatial.path(s).exists()

@pytest.mark.parametrize('field,value',[('source_frame_coordinate_space','roi_xy'),('source_frame_origin','unknown'),('roi_right',9000),('source_frame_width_px',9000)])
def test_V07_untrusted_mapping_blocked(tmp_path,field,value):
    s=source(tmp_path);scene=core.scene(s,G);candidate=stacks.snapshot(s,G)['candidates'][0]
    s.attempts['a0'][field]=value
    assert marker.describe(s,G,scene,candidate)['validity']=='UNTRUSTED_COORDINATE_SPACE'

@pytest.mark.parametrize('bbox,status',[( (150,170,300,170),'ZERO_HEIGHT'),((150,140,150,210),'ZERO_WIDTH'),((150,170,150,170),'POINT')])
@pytest.mark.parametrize('dpi',[96,144,192])
def test_V09_V10_V11_render_exact_endpoints_no_rectangle(bbox,status,dpi):
    class Canvas:
        def __init__(self):self.items=[]
        def winfo_fpixels(self,_):return dpi
        def create_line(self,*args,**kw):self.items.append(('line',args,kw))
        def create_oval(self,*args,**kw):self.items.append(('oval',args,kw))
    for zoom,pan in ((.3,(100,20)),(4.7,(-30,40))):
        c=Canvas();ed=NS(c=c,to_canvas=lambda p:(p[0]*zoom+pan[0],p[1]*zoom+pan[1]))
        marker.paint(ed,dict(bbox=bbox),color='red',tags=('test',))
        if status!='POINT':
            segment=next(v for v in c.items if 'bbox_marker_segment' in v[2]['tags'])
            assert segment[1]==(*ed.to_canvas(bbox[:2]),*ed.to_canvas(bbox[2:]))
        else:assert any('bbox_marker_cross' in v[2]['tags'] for v in c.items)
        assert all(i[0] in ('line','oval') for i in c.items)

@pytest.mark.parametrize('verdict',['ON_PLATE','OFF_PLATES_BAD','UNDECIDABLE','UNREVIEWED'])
def test_O01_O02_O03_O05_O06_O08_M01_independent_reopen(actual,verdict):
    s=actual;gid=SID+'-a00000001';oid='GT:'+gid;before=freeze(s)
    result=spatial.set_verdict(s,gid,oid,verdict)
    assert result['spatial_verdict']==verdict and result['reason'] and result['actor']
    assert result['source_identity']['geometry_kind']=='BBOX' and result['source_identity']['attempt_id']==gid
    assert result['validity']=='INVALID_ZERO_HEIGHT' and result['diagnostic_revision']==1
    assert freeze(s)==before
    reopened=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert spatial.load(reopened)==spatial.load(s)

def test_O04_mixed_history_branching_and_revisions(tmp_path):
    s=source(tmp_path);partial.confirm_stack(s,G,oid=OID);base=s.review.review_revision
    spatial.set_verdict(s,G,OID,'ON_PLATE');assert spatial.history_choice(s,G)=='diagnostic'
    partial.toggle_plate(s,G,OID);assert spatial.history_choice(s,G)=='canonical'
    stacks.restore_history(s,G);assert spatial.history_choice(s,G)=='diagnostic'
    spatial.restore(s,G);assert not spatial.load(s)['records']
    assert spatial.history_choice(s,G,redo=True)=='diagnostic'
    spatial.restore(s,G,redo=True);assert spatial.history_choice(s,G,redo=True)=='canonical'
    stacks.restore_history(s,G,redo=True);assert spatial.history_choice(s,G)=='canonical'
    stacks.restore_history(s,G);spatial.set_verdict(s,G,OID,'OFF_PLATES_BAD')
    assert spatial.history_choice(s,G,redo=True) is None
    spatial.restore(s,G);partial.toggle_plate(s,G,OID)
    assert spatial.history_choice(s,G,redo=True) is None
    assert s.review.review_revision>base and spatial.load(s)['diagnostic_revision']==5

def test_O04_atomic_failure_and_stale_writes(tmp_path,monkeypatch):
    from auto_annotation_tool.ranking import mobile_human_review as review
    s=source(tmp_path);spatial.set_verdict(s,G,OID,'ON_PLATE');before=spatial.path(s).read_bytes()
    with pytest.raises(ValueError):spatial.set_verdict(s,G,OID,'OFF_PLATES_BAD',expected_revision=0)
    def fail(*a):raise OSError('disk full')
    monkeypatch.setattr(review,'_atomic_write',fail)
    with pytest.raises(OSError):spatial.set_verdict(s,G,OID,'UNDECIDABLE')
    assert spatial.path(s).read_bytes()==before and not spatial.path(s).with_suffix('.json.lock').exists()

def test_M04_export_aggregate_and_no_historical_overwrite(tmp_path):
    s=source(tmp_path/'source',n=2);before=freeze(s)
    spatial.set_verdict(s,G,OID,'OFF_PLATES_BAD');spatial.set_verdict(s,G,'GT:a1','UNDECIDABLE')
    paths=spatial.export_addendum(s,tmp_path/'new_export');payload=json.loads(paths[0].read_text(encoding='utf-8'))
    assert payload['aggregate']['counts']['OFF_PLATES_BAD']==1 and payload['aggregate']['counts']['UNDECIDABLE']==1
    prior=[p.read_bytes() for p in paths]
    with pytest.raises(ValueError):spatial.export_addendum(s,tmp_path/'new_export')
    assert [p.read_bytes() for p in paths]==prior and freeze(s)==before

def test_S08_two_physical_references_stay_distinct(tmp_path):
    s=source(tmp_path,n=2)
    for oid in (OID,'GT:a1'):partial.confirm_stack(s,G,oid=oid)
    state=stacks.snapshot(s,G);before=freeze(s)
    assert len(state['plates'])==2 and len({c['plate_id'] for c in state['candidates']})==2
    spatial.set_verdict(s,G,OID,'ON_PLATE');spatial.set_verdict(s,G,'GT:a1','OFF_PLATES_BAD')
    assert freeze(s)==before

def test_S06_RAW_stays_diagnostic_and_cannot_receive_verdict(actual):
    from auto_annotation_tool.gui.mobile_canvas_shift_preview import ShiftPreview
    s=actual;gid=SID+'-a00000005';ed=editor(s,gid);preview=ShiftPreview.__new__(ShiftPreview);preview.ed=ed
    raw=next(c for c in stacks.snapshot(s,gid)['candidates'] if c['role']=='SUPPRESSED')
    before=freeze(s)
    assert any(f['role']=='SUPPRESSED' for f in preview.frames())
    with pytest.raises(ValueError):spatial.set_verdict(s,gid,raw['object_id'],'OFF_PLATES_BAD')
    assert len(mt._input(s,gid)['finals'])==1 and freeze(s)==before

def test_S07_D_redraw_preserves_zero_source_and_physical_identity(actual):
    from auto_annotation_tool.gui.mobile_canvas_draw_target import resolve
    s=actual;gid=SID+'-a00000001';ed=editor(s,gid);ed.entry=None;ed.drag=None;ed.busy=False
    state=stacks.snapshot(s,gid);source_before=copy.deepcopy(s.attempts);parents=copy.deepcopy(core.scene(s,gid)['parents'])
    target=resolve(ed);assert target['kind']=='focus'
    pid=target['reference']['plate_id'];q=copy.deepcopy(state['plates'][pid]['gt_quad']);q[0][0]+=1
    stacks.edit_gt(s,gid,ed.selected,quad=q)
    assert set(stacks.snapshot(s,gid)['plates'])==set(state['plates'])
    assert s.attempts==source_before and core.scene(s,gid)['parents']==parents

def test_M04_aggregate_counts_unreviewed_without_writing(actual):
    s=actual;before=freeze(s);summary=spatial.diagnostic_summary(s)
    assert summary['invalid_bbox_trusted_position']>0
    assert summary['counts']['OFF_PLATES_BAD']==0
    assert summary['counts']['UNREVIEWED']==summary['invalid_bbox_trusted_position']
    assert summary['raw_suppressed_excluded']>0 and freeze(s)==before
    assert not spatial.path(s).exists()

@pytest.mark.parametrize('width,tp',[(30,0),(105,1)])
def test_M03_valid_bbox_canonical_iou_020_070(tmp_path,width,tp):
    s=source(tmp_path,(150,140,150+width,210),mz=False);partial.confirm_stack(s,G,oid=OID)
    stacks.edit_gt(s,G,OID,quad=[[150,140],[300,140],[300,210],[150,210]])
    partial.confirm_gtr(s,G,stacks.snapshot(s,G)['links'][OID]['plate_id'])

    score=mt.evaluate_unit(s,G);assert [score[k] for k in ('tp','fp','fn')]==[tp,1-tp,1-tp], score
    with pytest.raises(ValueError):spatial.set_verdict(s,G,OID,'ON_PLATE')
