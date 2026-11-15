"""Read-only acceptance against the pinned twelve real F4 sources.

Set F4_EVAL2_BASELINE to the read-only baseline JSON prepared by the audit.
Every state-changing test copies its input into pytest's private directory.
"""
from dataclasses import replace
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import pytest

from auto_annotation_tool.ranking import mobile_mt_object_evaluation as mt
from auto_annotation_tool.ranking import mobile_mt_detection_evaluation as bbox_v1
from auto_annotation_tool.ranking import mobile_canvas_review as core, mobile_candidate_stacks as stacks
from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.gui.mobile_analysis import make_input, analyze
from auto_annotation_tool.ranking.mobile_session_metrics import from_results, current, result_key

SID='exp-1792699864138-add3e282'
G=SID+'-a00000008'
GOOD='GT:'+G
BAD='GT:'+SID+'-a00000009'


def stamp(path):
    path=Path(path)
    if not path.exists():return dict(exists=False)
    s=path.stat()
    return dict(exists=True,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size=s.st_size,mtime_ns=s.st_mtime_ns)


@pytest.fixture(scope='module')
def baseline():
    path=os.environ.get('F4_EVAL2_BASELINE')
    if not path or not Path(path).is_file():pytest.skip('Pinned real F4 baseline required (F4_EVAL2_BASELINE)')
    return json.loads(Path(path).read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def sessions(baseline):
    return {r['session_id']:MobileReviewSession.open(Path(r['private_source']),sidecar_path=Path(r['private_sidecar'])) for r in baseline['sessions']}


@pytest.fixture
def actual(tmp_path,baseline):
    r=next(r for r in baseline['sessions'] if r['session_id']==SID)
    source=tmp_path/'source.alprsession';side=tmp_path/'review.json'
    shutil.copyfile(r['private_source'],source);shutil.copyfile(r['private_sidecar'],side)
    return MobileReviewSession.open(source,sidecar_path=side)


@pytest.fixture(scope='module')
def cohort(tmp_path_factory,baseline,sessions):
    root=tmp_path_factory.mktemp('f4-analysis-v2');catalog=root/'catalog';(catalog/'human_reviews').mkdir(parents=True)
    results=root/'03_wyniki';(results/'porownanie').mkdir(parents=True)
    inputs=[]
    for r in baseline['sessions']:
        s=sessions[r['session_id']];side=Path(r['private_sidecar'])
        if side.exists():shutil.copyfile(side,catalog/'human_reviews'/(s.review.source_archive_sha256+'.review.json'))
        inputs.append(make_input(s.bundle.report,source_path=s.path,results_root=results,review_root=catalog,bundle=s.bundle))
    return analyze(inputs)


def test_T22_PPM_MT2_preserves_MT1_GT_and_both_raw_MZ(actual):
    before=copy.deepcopy(stacks.snapshot(actual,G));parents=copy.deepcopy(core.scene(actual,G)['parents'])
    raw=copy.deepcopy(actual.attempts)
    stacks.toggle_unassociated(actual,G,BAD,partial_review=True)
    stacks.toggle_unassociated(actual,G,BAD,partial_review=True)
    after=stacks.snapshot(actual,G)
    assert before['plates']==after['plates'] and before['links'][GOOD]==after['links'][GOOD]
    assert actual.attempts==raw and core.scene(actual,G)['parents']==parents
    d=mt.evaluate_session(actual);row=next(r for r in d['ocr_summary']['rows'] if r['attempt_id']==BAD[3:])
    assert row['quality_state']=='EXECUTED_EXCLUDED_MT_FP' and row['raw_prediction']=='W'


def test_T28_actual_image07_partial_FP_bound_then_one_TP_one_FP(actual):
    assert partial.progress(core.scene(actual,G))['object_decisions'][GOOD]=='PLATE'
    partial.toggle_stack(actual,G,oid=GOOD)
    pending=mt.evaluate_unit(actual,G)
    assert pending['tp'] is pending['fn'] is None and pending['partial_fp_lower_bound']==1
    partial.confirm_stack(actual,G,oid=GOOD)
    done=mt.evaluate_unit(actual,G)
    assert (done['tp'],done['fp'],done['fn'])==(1,1,0)
    assert done['matched_pairs'][0]['bbox_iou']<.5
    assert (bbox_v1.evaluate_unit(actual,G)['tp'],bbox_v1.evaluate_unit(actual,G)['fp'],bbox_v1.evaluate_unit(actual,G)['fn'])==(0,2,1)


def test_T32_image_session_snapshot_have_identical_v2_policy(sessions):
    s=sessions[SID];data=mt.evaluate_session(s);snapshot=from_results(s,core.summary(s),data)
    assert current(snapshot,result_key(s))
    assert snapshot['matching_policy_sha256']==mt.evaluate_unit(s,G)['matching_policy_sha256']==mt.POLICY_SHA256
    for k in ('tp','fp','fn','precision','recall','f1'):assert snapshot['mt'][k]==data[k]
    stale=copy.deepcopy(snapshot);stale['matching_policy_sha256']=bbox_v1.POLICY_SHA256
    assert not current(stale,result_key(s))


def test_T33_analysis_primary_uses_v2_with_explicit_secondary_v1(cohort,sessions):
    for s in cohort.sessions:
        expected=mt.evaluate_session(sessions[s.input.session_id])
        assert s.source_status=='VERIFIED'
        assert s.metadata['matching_policy_sha256']==mt.POLICY_SHA256
        for k in ('tp','fp','fn'):assert s.metrics['mt_'+k].value==expected[k]
        assert s.metrics['bbox_v1_tp'].value==bbox_v1.evaluate_session(sessions[s.input.session_id])['tp']
    target=next(s for s in cohort.sessions if s.input.session_id==SID)
    assert target.metrics['mt_tp'].value>target.metrics['bbox_v1_tp'].value


def test_T34_original_archives_and_sidecars_preserved(baseline):
    for r in baseline['sessions']:
        assert stamp(r['path'])==r['archive'] and stamp(r['sidecar'])==r['sidecar_state']


def test_T37_analysis_metric_denominators_are_independent(cohort):
    assert {s.metrics['mt_tp'].status for s in cohort.sessions} >= {'FULL','SUBSET','N/A'}
    for s in cohort.sessions:
        for key in ('polygon_iou_pct','exact_mz_pct','micro_cer_pct'):
            m=s.metrics[key]
            assert m.n_scored is None or m.n_total_eligible is None or m.n_scored<=m.n_total_eligible
            if m.value is None:assert m.status=='N/A'
    assert all(s.metrics['exact_e2e_pct'].value is None for s in cohort.sessions)


def test_T39_twelve_sessions_four_triplets_keep_session_percentiles(cohort,sessions):
    groups={}
    for s in cohort.sessions:groups.setdefault((s.metadata['scenario'],s.metadata['variant']),[]).append(s)
    assert len(cohort.sessions)==12 and len(groups)==4
    for group in groups.values():assert {s.metadata['repeat'] for s in group}=={'r01','r02','r03'}
    s=next(s for s in cohort.sessions if s.input.session_id=='exp-1792699303016-c8b05efb')
    assert s.metrics['mt_p95_ms'].n_scored==21 and s.metrics['mz_trace_p95_ms'].n_scored==20
    assert s.metrics['mz_calls'].value==23 and s.metrics['mz_call_p95_ms'].value is None
    assert s.metrics['mz_trace_p95_ms'].value==pytest.approx(1244.08064965)


def test_T42_all_twelve_every_TP_has_persisted_relation_and_conservation(sessions):
    assert len(sessions)==12
    for session in sessions.values():
        before=stamp(session.sidecar_path);review=copy.deepcopy(session.review.to_dict())
        result=mt.evaluate_session(session)
        for u in result['units']:
            if u['status']!='MEASURED':
                assert u['tp'] is u['fp'] is u['fn'] is None
                continue
            assert u['tp']+u['fp']==u['n_final_mt'] and u['tp']+u['fn']==u['n_gt']
            assert len({p['gt_id'] for p in u['matched_pairs']})==len({p['attempt_id'] for p in u['matched_pairs']})==u['tp']
            for p in u['matched_pairs']:
                evidence=p['operator_review_evidence']
                assert p['matched_by']=='OPERATOR_CONFIRMED_RELATION' and evidence['gt_id']==p['gt_id']
                assert evidence['source_binding']==u['source_binding'] and evidence['attempt_id']==p['attempt_id']
        assert stamp(session.sidecar_path)==before and session.review.to_dict()==review


def test_T43_analysis_reports_retrospective_date(cohort):
    payload=cohort.payload()
    assert payload['analysis_protocol_id']=='ALPR-ANALYSIS-1.1' and payload['protocol_date']=='2026-11-01'
    assert all(s['metadata']['quality_chronology']=='RETROSPECTIVE_NOT_PREREGISTERED' for s in payload['sessions'])
