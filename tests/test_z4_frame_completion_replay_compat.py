"""Eligibility provenance is additive; old math/exports remain replayable."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,json,shutil
from test_z4_auto_frame_completion import OUT,red,core,mt,export,frame,frozen,sha
from test_f4_eval2_object_scoring import unit

def test_300_legacy_pure_and_aggregate_cases_match_frozen_G0():
    b=json.loads((OUT/'private/golden_before.json').read_text(encoding='utf-8'))
    assert mt.POLICY_SHA256==b['policy_sha256']
    for value,result,aggregate in zip(b['cases'],b['results'],b['aggregates']):
        now=mt.score_unit_v2(value)
        assert json.loads(json.dumps(now))==result and json.loads(json.dumps(mt.aggregate_units([now])))==aggregate

def test_old_export_remains_replayable_and_unchanged():
    b=json.loads((OUT/'legacy_export_baseline.json').read_text(encoding='utf-8'));r=export.replay(b['namespace'])
    assert not r.get('completion_eligibility') and all(sha(p)==v for p,v in b['files'].items())

def test_mixed_old_new_eligibility_cannot_silently_aggregate():
    old=unit();new=copy.deepcopy(old);new['unit_id']='different-unit'
    new['completion_eligibility']=dict(frame.ELIGIBILITY,sha256=frame.ELIGIBILITY_SHA256)
    r=mt.aggregate_units([mt.score_unit_v2(old),mt.score_unit_v2(new)])
    assert r['tp'] is r['fp'] is r['fn'] is None and 'MIXED_SESSION_OR_POLICY_COHORT' in r['unavailable_reasons']

def test_invalid_eligibility_digest_is_rejected():
    value=unit();value['completion_eligibility']=dict(frame.ELIGIBILITY,sha256='0'*64)
    r=mt.score_unit_v2(value);assert r['status']=='NOT_EVALUABLE' and r['tp'] is None

def source_validation(s,tmp_path):
    from auto_annotation_tool.gui.mobile_analysis import AnalysisInput,SourceValidation
    catalog=tmp_path/'catalog';(catalog/'human_reviews').mkdir(parents=True)
    shutil.copyfile(s.sidecar_path,catalog/'human_reviews'/(s.review.source_archive_sha256+'.review.json'))
    inp=AnalysisInput(s.bundle.report,s.session_id,s.review.source_archive_sha256,str(s.bundle.report.report_id),str(s.path),None,catalog,True)
    return SourceValidation(inp,'VERIFIED','',quality_state='VALID')

def test_analysis_reports_historical_eligibility_without_invalidating_old_export(red,tmp_path):
    from auto_annotation_tool.gui.mobile_analysis_object_quality import quality
    before=frozen(red);b=json.loads((OUT/'legacy_export_baseline.json').read_text(encoding='utf-8'))
    source=source_validation(red,tmp_path);quality(source,NS(directory=Path(b['namespace']).parent))
    assert source.metadata['historical_export_eligibility']['previous']=='legacy_checked_v1'
    assert source.metadata['historical_export_eligibility']['current']==frame.ELIGIBILITY_SHA256
    assert source.metrics['mt_tp'].value is not None and 'OBJECT_COUNTS_NOT_VERIFIED' not in source.metrics['mt_tp'].missing_reasons
    assert frozen(red)==before and all(sha(p)==v for p,v in b['files'].items())

def test_analysis_still_rejects_old_export_with_other_review_revision(red,tmp_path,monkeypatch):
    from auto_annotation_tool.gui.mobile_analysis_object_quality import quality
    b=json.loads((OUT/'legacy_export_baseline.json').read_text(encoding='utf-8'));old=export.replay(b['namespace']);old['review_revision']+=1
    monkeypatch.setattr(export,'replay',lambda _:copy.deepcopy(old));source=source_validation(red,tmp_path)
    quality(source,NS(directory=Path(b['namespace']).parent))
    assert 'OBJECT_COUNTS_NOT_VERIFIED' in source.metrics['mt_tp'].missing_reasons

def test_analysis_same_eligibility_requires_exact_recomputation(red,tmp_path,monkeypatch):
    from auto_annotation_tool.gui.mobile_analysis_object_quality import quality
    paths=export.export(red,tmp_path/'export');data=export.replay(paths[0].parent);data['tp']+=1
    monkeypatch.setattr(export,'replay',lambda _:copy.deepcopy(data));source=source_validation(red,tmp_path)
    quality(source,NS(directory=paths[0].parent.parent))
    assert 'OBJECT_COUNTS_NOT_VERIFIED' in source.metrics['mt_tp'].missing_reasons

def test_cross_invocation_reference_uses_frame_approval_without_image_flag(tmp_path):
    from test_z4_p0_physical_gt import session_fixture
    from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial,mobile_candidate_stacks as stacks
    from auto_annotation_tool.ranking.mobile_canvas_physical_gt import share_across_invocations
    s=session_fixture(tmp_path,n=2,suppressed=None,two_calls=True)
    partial.confirm_stack(s,'g0',oid='GT:a0');partial.confirm_stack(s,'g1',oid='GT:a1')
    before=copy.deepcopy(stacks.snapshot(s,'g0')['plates'])
    assert not core.scene(s,'g0')['checked'] and not core.scene(s,'g1')['checked']
    share_across_invocations(s,'g1','GT:a1','g0','GT:a0')
    assert stacks.snapshot(s,'g0')['plates']==before and not core.scene(s,'g0')['checked']
    assert frame.for_session(s,'g1')['state']==frame.READY

def test_manifest_OCR_coverage_uses_TP_target_and_preserves_all_MZ_executions(red,tmp_path):
    before=frozen(red);paths=export.export(red,tmp_path/'export');root=paths[0].parent
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'));data=export.replay(root);ocr=data['ocr_summary']
    assert manifest['metric_coverage']['ocr']==[ocr['n_reads'],ocr['n_target']]
    assert manifest['metric_coverage']['mz_executed_all']==sum(r['executed'] for r in ocr['rows'])
    assert manifest['metric_coverage']['mz_executed_all']>ocr['n_target'] and frozen(red)==before
