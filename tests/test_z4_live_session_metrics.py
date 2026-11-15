"""L02-L20/L27-L29: projection of canonical results, never a second scorer."""
import copy
import json
from types import SimpleNamespace
import pytest
from auto_annotation_tool.ranking import mobile_session_metrics as live
from auto_annotation_tool.ranking import mobile_mt_detection_evaluation as mt
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking import mobile_candidate_stacks as stacks
from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial
from test_z4_mt_detection_evaluation import approved_session,unit,with_source_confidence
from test_z4_p0_physical_gt import session_fixture
from test_mobile_p2b2_canvas_review import text_all

G='mt-one'

def empty_canvas(session):
    return dict(session_id=session.session_id,source_archive_sha256=session.review.source_archive_sha256,
        review_revision=session.review.review_revision,geometry_n=0,polygon_iou_mean_pct=None,nmce_mean_pct=None,
        evaluated_reads=0,gt_characters=0,text_errors=0,exact_reads=0,pending_scenes=1)

def synthetic_snapshot(tp,fp,fn):
    s=SimpleNamespace(session_id='session-one',review=SimpleNamespace(source_archive_sha256='a'*64,
        review_revision=3,review_status='IN_PROGRESS'))
    detection=dict(session_id=s.session_id,source_archive_sha256=s.review.source_archive_sha256,
        review_revision=3,matching_policy_id=mt.POLICY_ID,matching_policy_sha256=mt.POLICY_SHA256,
        status='PARTIAL',n_qualified=2,n_total=3,tp=tp,fp=fp,fn=fn,n_gt=tp+fn,n_final_mt=tp+fp,
        **mt.rates(tp,fp,fn))
    return s,live.from_results(s,empty_canvas(s),detection)

@pytest.mark.parametrize('counts,shown',[
    ((8,2,0),('80,0%','100,0%','88,9%')),((8,0,2),('100,0%','80,0%','88,9%')),
    ((0,0,0),('—','—','—')),((0,0,1),('—','0,0%','0,0%')),
    ((0,1,0),('0,0%','—','0,0%')),((1,1,1),('50,0%','50,0%','50,0%'))])
def test_L03_L06_existing_rates_preserved(counts,shown):
    s,snapshot=synthetic_snapshot(*counts)
    view=live.presentation(snapshot,live.result_key(s))
    assert tuple(view['values'][k] for k in ('precision','recall','f1'))==shown
    assert view['status']=='W toku · tylko ocenione dane'

def test_L07_canonical_sums_not_average_of_image_percentages():
    a=mt.score_unit(unit(gt=1,final=1));b=mt.score_unit(unit(gt=1,final=3))
    b.update(unit_id=b['unit_id']+':another',mt_invocation_id='another')
    total=mt.aggregate_units([a,b])
    assert (total['tp'],total['fp'],total['fn'])==(2,2,0)
    assert total['precision']==.5
    assert total['precision']!=(a['precision']+b['precision'])/2

def test_L13_geometry_is_exactly_the_existing_qualified_canvas_stream(tmp_path):
    s=approved_session(tmp_path,n=2);text_all(s)
    quad=copy.deepcopy(core.scene(s,G)['objects'][0]['quad']);quad[0][0]+=12
    core.correct_plate(s,G,'GT:a0',quad)
    old=core.summary(s);frozen=copy.deepcopy(old)
    stats=s.statistics();snapshot=stats['live_session_metrics_v1']
    assert old==frozen==stats['canvas_review']
    assert snapshot['geometry']==dict(n_geometry=2,polygon_iou_mean_pct=old['polygon_iou_mean_pct'],
        nmce_mean_pct=old['nmce_mean_pct'],unavailable_reasons=[])
    assert old['polygon_iou_mean_pct']!=100

def test_L14_shared_gt_updates_all_canonical_pairs_without_extra_plate(tmp_path):
    source=session_fixture(tmp_path/'source',n=2,suppressed=None)
    s=with_source_confidence(source,tmp_path)
    core.set_text(s,G,'GT:a0','WA01234')
    core.correct_plate(s,G,'GT:a0',core.scene(s,G)['objects'][0]['quad'])
    stacks.associate(s,G,'GT:a1','GT:a0')
    partial.toggle_plate(s,G,'GT:a0')
    before=s.statistics()['live_session_metrics_v1']
    ref=next(iter(stacks.snapshot(s,G)['plates'].values()))
    quad=copy.deepcopy(ref['gt_quad']);quad[1][1]+=9
    stacks.edit_gt(s,G,ref['origin_object_id'],quad=quad)
    after=s.statistics();snapshot=after['live_session_metrics_v1']
    assert snapshot['mt']['n_gt']==1
    assert snapshot['geometry']['n_geometry']==after['canvas_review']['geometry_n']
    assert snapshot['geometry']['polygon_iou_mean_pct']!=before['geometry']['polygon_iou_mean_pct']
    assert len(stacks.snapshot(s,G)['plates'])==1

def test_L17_L18_micro_cer_uses_existing_error_and_character_sums(tmp_path):
    s=approved_session(tmp_path,n=2)
    core.set_text(s,G,'GT:a0','A');core.set_text(s,G,'GT:a1','WA11234')
    stats=s.statistics();canvas=stats['canvas_review'];snapshot=stats['live_session_metrics_v1']
    ocr=snapshot['ocr']
    assert ocr['n_reads']==2 and ocr['n_gt_characters']==8 and ocr['n_exact']==1
    assert ocr['micro_cer']==canvas['text_errors']/canvas['gt_characters']
    values=[r['cer'] for r in canvas['rows'] if r['cer'] is not None]
    assert ocr['micro_cer']!=sum(values)/len(values)
    assert live.presentation(snapshot,live.result_key(s))['exact']=='Exact MZ: 1/2'

def test_L19_saved_missing_text_has_no_ocr_measurement_or_fictional_zero(tmp_path):
    s=approved_session(tmp_path,n=1)
    stats=s.statistics();snapshot=stats['live_session_metrics_v1']
    assert snapshot['mt']['n_qualified']==1
    assert snapshot['ocr']['n_reads']==0 and snapshot['ocr']['micro_cer'] is None
    assert snapshot['geometry']['n_geometry']==0  # existing complete-scene gate
    view=live.presentation(snapshot,live.result_key(s))
    assert view['values']['precision']=='100,0%' and view['values']['cer']=='—'
    assert view['exact']=='Exact MZ: —/—'

def test_L20_cer_can_exceed_one_hundred_percent(tmp_path):
    s=approved_session(tmp_path,n=1);core.set_text(s,G,'GT:a0','X')
    snapshot=s.statistics()['live_session_metrics_v1']
    assert snapshot['ocr']['micro_cer']>1
    shown=live.presentation(snapshot,live.result_key(s))['values']['cer']
    assert shown==live.percent(100*snapshot['ocr']['micro_cer']) and shown!='100,00%'

@pytest.mark.parametrize('field,value',[
    ('review_revision',999),('session_id','another-session'),('source_archive_sha256','b'*64),
    ('matching_policy_id','unknown'),('matching_policy_sha256','c'*64)])
def test_L23_wrong_worker_identity_cannot_be_displayed(field,value):
    s,snapshot=synthetic_snapshot(8,2,0)
    snapshot[field]=value
    assert not live.current(snapshot,live.result_key(s))
    assert set(live.presentation(snapshot,live.result_key(s))['values'].values())=={'—'}

def test_L23_mixed_revisions_cannot_create_one_snapshot(tmp_path):
    s=approved_session(tmp_path,n=1);text_all(s)
    canvas=core.summary(s);detection=mt.evaluate_session(s)
    detection['review_revision']-=1
    result=live.from_results(s,canvas,detection)
    assert result['status']=='STALE_REVIEW'
    assert not live.current(result,live.result_key(s))
    assert result['geometry']['polygon_iou_mean_pct'] is None
    assert result['ocr']['micro_cer'] is None and result['mt']['tp'] is None

def test_L24_L27_additive_export_matches_snapshot_and_never_writes_gt(tmp_path):
    s=approved_session(tmp_path/'data',n=2);text_all(s)
    before=s.sidecar_path.read_bytes();revision=s.review.review_revision
    canvas=core.summary(s);detection=mt.evaluate_session(s);stats=s.statistics()
    assert stats['canvas_review']==canvas and stats['mt_detection_evaluation_v1']==detection
    target=tmp_path/'new-export';target.mkdir();paths=s.export(target)
    exported=json.loads(next(p for p in paths if p.name=='review.json').read_text(encoding='utf-8'))
    assert exported['derived_metrics']['live_session_metrics_v1']==stats['live_session_metrics_v1']
    assert exported['derived_metrics']['canvas_review']==canvas
    assert s.sidecar_path.read_bytes()==before and s.review.review_revision==revision

def test_L29_only_the_current_session_and_partial_scope():
    s,snapshot=synthetic_snapshot(1,1,0)
    view=live.presentation(snapshot,live.result_key(s))
    assert snapshot['session_id']==s.session_id and snapshot['gt_mode']=='ASSISTED'
    assert 'MT: 2/3' in view['denominators'] and 'W toku' in view['status']
    assert not live.current(snapshot,('other',*live.result_key(s)[1:]))
