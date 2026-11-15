"""Explain existing OCR eligibility per source readout, never re-score it."""
from pathlib import Path
import copy,csv,json

SCHEMA='alpr.mobile.mz_quality_context.v2'

def false_mt(c):
    return bool(c and c.get('role')=='FINAL' and c.get('operator_decision')=='NOT_PLATE')

def note(c):
    return ' · wykonano, poza CER/Exact: odrzucone MT' if false_mt(c) and c.get('mz_ran') else ''

def snapshot_v1(session):
    from ..ranking import mobile_candidate_stacks as stacks,mobile_canvas_review as core
    candidates={};canonical=core.summary(session)
    outcomes={r['attempt_id']:r for r in canonical['rows'] if r.get('attempt_id')}
    for gid in session.mt_invocations:
        try:state=stacks.snapshot(session,gid)
        except (ValueError,KeyError,TypeError):continue
        for c in state['candidates']:candidates.setdefault(c['attempt_id'],[]).append((gid,c))
    rows=[];source_rows=list(session.samples.items());represented={s.get('attempt_id') for _,s in source_rows}
    source_rows.extend((None,dict(attempt_id=aid)) for aid in session.attempts if aid not in represented)
    for sid,sample in source_rows:
        aid=sample.get('attempt_id');attempt=session.attempts.get(aid,{});found=candidates.get(aid,[])
        proven=len(found)==1 and str(attempt.get('mt_invocation_id') or '')==found[0][0]
        for key in ('scene_generation','session_id'):
            if sample.get(key) and str(sample[key])!=str(attempt.get(key,'')):proven=False
        crop=sample.get('image_entry') or sample.get('evidence_entry')
        if crop and attempt.get('evidence_entry') and crop!=attempt['evidence_entry']:proven=False
        gid,c=found[0] if proven else (None,None)
        executed={'true':True,'1':True,'false':False,'0':False}.get(str(attempt.get('mz_executed','')).lower()) if proven else None
        if not proven and (sample.get('fresh_mz_successful') in (True,'true') or sample.get('timing',{}).get('character_inference_ms') is not None):executed=True
        state='EXECUTION_UNKNOWN';reason='MZ_PARENT_UNPROVEN';eligible=False
        if proven:
            if not executed:state='NOT_RUN' if c['mz_status']=='NOT_RUN' else 'EXECUTION_UNKNOWN';reason=c['mz_status']
            elif c['role']=='SUPPRESSED':state='EXECUTED_EXCLUDED';reason='MZ_EXCLUDED_RAW_SUPPRESSED'
            elif false_mt(c):state='EXECUTED_EXCLUDED';reason='MZ_EXCLUDED_MT_FALSE_POSITIVE'
            elif c['mz_status']=='AMBIGUOUS_SAMPLES':state='EXECUTED_UNASSESSED';reason='AMBIGUOUS_SOURCE_READOUTS'
            elif outcomes.get(aid,{}).get('cer') is not None:state='EXECUTED_ELIGIBLE';reason='CANONICAL_EVALUATED';eligible=True
            else:state='EXECUTED_UNASSESSED';reason='CANONICAL_NOT_EVALUATED'
        elif executed:state='EXECUTED_UNASSESSED'
        rows.append(dict(session_id=session.session_id,source_archive_sha256=session.bundle.source_archive_sha256,
            review_revision=session.review.review_revision,source_sample_id=sid,attempt_id=aid,
            mt_invocation_id=gid,mt_detection_index=c['mt_detection_index'] if c else None,
            parent_proven=proven,source_mt_role=c['role'] if c else 'UNKNOWN',
            source_mz_status=attempt.get('mz_status') if proven else sample.get('mz_status'),
            source_execution_id=sample.get('mz_execution_id') or attempt.get('mz_execution_id') or None,
            source_mz_executed=executed,quality_state=state,quality_reason=reason,canonical_ocr_eligible=eligible,
            canonical_evaluation_key=aid if eligible else None,
            raw_prediction=sample.get('raw_prediction',sample.get('prediction',attempt.get('raw_prediction',attempt.get('prediction','')))),
            crop_entry=crop or attempt.get('evidence_entry'),timing=copy.deepcopy(sample.get('timing',{})),
            source_mz_ms=sample.get('mz_ms'),source_pipeline_ms=sample.get('pipeline_ms')))
    return dict(schema='alpr.mobile.mz_quality_context.v1',session_id=session.session_id,source_archive_sha256=session.bundle.source_archive_sha256,
        review_revision=session.review.review_revision,rows=rows,
        n_source_readouts=len(rows),n_readouts_excluded_mt_fp=sum(r['quality_reason']=='MZ_EXCLUDED_MT_FALSE_POSITIVE' for r in rows),
        n_readouts_unproven_parent=sum(not r['parent_proven'] for r in rows),
        canonical_evaluated_reads=canonical['evaluated_reads'],canonical_exact_reads=canonical['exact_reads'],
        canonical_gt_characters=canonical['gt_characters'],canonical_text_errors=canonical['text_errors'],
        counting_note='Rows are source readout evidence, not new inference-call counts or a replacement OCR denominator. Source timings and canonical scoring are unchanged.')

def snapshot(session):
    from ..ranking.mobile_mt_object_evaluation import evaluate_session
    return evaluate_session(session)['ocr_summary']


def export(session,directory):
    data=snapshot(session);root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    jp=root/'mz_quality_context_v2.json';cp=root/'mz_quality_context_v2.csv'
    if jp.exists() or cp.exists():raise ValueError('Eksport pochodzenia MZ już istnieje; wybierz nową wersję.')
    with jp.open('x',encoding='utf-8') as handle:json.dump(data,handle,ensure_ascii=False,indent=2)
    fields=list(data['rows'][0]) if data['rows'] else ['attempt_id','quality_reason']
    with cp.open('x',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in data['rows']:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    return [jp,cp]
