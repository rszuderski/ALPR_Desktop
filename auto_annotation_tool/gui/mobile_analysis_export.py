"""Figures and atomic, revalidated ALPR-ANALYSIS-1.1 / F4-EVAL-2.0 exports."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,shutil,tempfile
from .mobile_analysis import analyze,PROTOCOL_ID,IMPLEMENTATION_ID,METRICS,QUESTIONS
from .mobile_analysis_figure import make_figure
from .mobile_analysis_presentation import report_sections,DICTIONARY_PATH

DEFAULT_FIGURES=('mt_p50_ms','mt_p95_ms','mz_trace_p50_ms','mz_trace_p95_ms',
    'pipeline_p50_ms','pipeline_p95_ms','mt_f1_pct','polygon_iou_pct','exact_mz_pct','micro_cer_pct')

def figure_plan(cohort,metrics=None):
    # Empty charts remain selectable in the UI as diagnostics, but are not
    # exported as decorative evidence of an unperformed experiment.
    if cohort.question == 'F5A':
        from .mobile_analysis_f5a import DEFAULT_FIGURES as f5a_defaults, aggregate
        for scenario in sorted({s.metadata.get('scenario','NIEOKREŚLONY') for s in cohort.sessions}):
            for key in metrics or f5a_defaults:
                if any(r['value'] is not None for r in aggregate(cohort,key,scenario)):
                    yield scenario,key
        return
    for scenario in sorted({s.metadata.get('scenario','NIEOKREŚLONY') for s in cohort.sessions}):
        for key in metrics or DEFAULT_FIGURES:
            if any(s.metadata.get('scenario')==scenario and s.metrics[key].value is not None for s in cohort.sessions):
                yield scenario,key

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _fingerprint(cohort):
    result = dict(protocol_files=cohort.protocol_files,sessions={s.input.session_id:s.fingerprint for s in cohort.sessions})
    if cohort.question == 'F5A':result['performance']=cohort.performance
    return result

def export_analysis(cohort,results_root,*,metrics=None):
    """Quality may become N/A before rendering; changes during rendering abort."""
    fresh=analyze(cohort.inputs,cohort.question)
    if fresh.question == 'F5A' and cohort.duplicates:
        from .mobile_analysis_f5a import qualify
        fresh.duplicates=list(cohort.duplicates)
        fresh.performance=qualify(fresh.sessions,fresh.duplicates)
    old={s.input.session_id:s for s in cohort.sessions}
    for s in fresh.sessions:
        if old[s.input.session_id].source_status=='VERIFIED' and s.source_status!='VERIFIED':
            raise ValueError('Oryginał zmienił się lub jest niedostępny. Odśwież analizę: '+s.input.source_path)
    root=Path(results_root)
    if not root.is_dir() or root.name.lower()!='03_wyniki':raise ValueError('Wskaż istniejący katalog 03_wyniki.')
    parent=root/'porownanie'
    if not parent.is_dir():raise ValueError('Brak istniejącego katalogu porownanie.')
    stage=Path(tempfile.mkdtemp(prefix='.alpr_analysis_stage_',dir=parent));files=[]
    try:
        payload=fresh.payload()
        data=stage/'analysis.json';data.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding='utf-8');files.append(data)
        csv_path=stage/'readiness_matrix.csv'
        with csv_path.open('w',encoding='utf-8-sig',newline='') as stream:
            columns=['analysis_protocol_id','question','interpretation','session_id','report_id','source_archive_sha256','scenario','variant','repeat',
                'variant_provenance','declared_android_variant','variant_registry_sha256',
                'source_status','quality_state','metric_id','status','unit','observation_unit','value','n_scored','n_total_eligible','n_unscored',
                'coverage','numerator','denominator','gt_origin','review_revision','matching_policy_id','matching_policy_sha256',
                'method_adopted_date','missing_reasons','explanation','evidence_paths','implementation_id']
            if any(s.metadata.get('input_mode') == 'TV_BBOX_CROP' for s in fresh.sessions):
                columns += ['input_mode', 'tv_bbox_crop_digest_sha256', 'calibration_digest_sha256']
            if fresh.question == 'F5A':columns += ['block', 'performance_status', 'material_sha256']
            writer=csv.DictWriter(stream,fieldnames=columns,delimiter=';');writer.writeheader()
            for s in fresh.sessions:
                for m in s.metrics.values():
                    row=m.payload();row.update(question=fresh.question,interpretation=fresh.interpretation,
                        session_id=s.input.session_id,report_id=s.input.report_id,source_archive_sha256=s.input.source_sha256,
                        source_status=s.source_status,quality_state=s.quality_state,
                        **{k:s.metadata.get(k) for k in ('scenario','variant','repeat','variant_provenance','declared_android_variant')})
                    row['variant_registry_sha256']=(s.metadata.get('f4_variant') or {}).get('registry_sha256')
                    if fresh.question == 'F5A':
                        from .mobile_analysis_f5a import ALGORITHM_ID
                        diag=s.metadata.get('f5a_performance') or {}
                        row.update(block=s.metadata.get('block'),performance_status=diag.get('status'),
                            material_sha256=diag.get('material_sha256'),implementation_id=ALGORITHM_ID)
                    if 'input_mode' in columns:
                        row['input_mode'] = s.metadata.get('input_mode', 'CAMERA_FULL_FRAME')
                    if s.metadata.get('input_mode') == 'TV_BBOX_CROP':
                        ctx = s.metadata['input_context']
                        row.update(input_mode='TV_BBOX_CROP', tv_bbox_crop_digest_sha256=ctx['descriptor_sha256'],
                                   calibration_digest_sha256=ctx['calibration_sha256'])
                    row['missing_reasons']=' | '.join(row['missing_reasons']);row['evidence_paths']=' | '.join(row['evidence_paths'])
                    writer.writerow({k:row.get(k) for k in columns})
        files.append(csv_path)
        if fresh.question == 'F5A':
            aggregates=payload['variant_metrics']
            cp=stage/'variant_metrics.csv'
            with cp.open('w',encoding='utf-8-sig',newline='') as stream:
                from .mobile_analysis import metric_definition
                columns=['scenario','variant','metric_id','unit','value','status','aggregation','n_sessions','n_sessions_selected',
                    'n_source_observations','n_source_observations_total','missing_reasons','source_session_ids','source_sha256','session_values']
                writer=csv.DictWriter(stream,fieldnames=columns,delimiter=';');writer.writeheader()
                for row in aggregates:
                    row=dict(row,unit=metric_definition(row['metric_id'])[1])
                    writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
            files.append(cp)
        import matplotlib.pyplot as plt
        for scenario,key in figure_plan(fresh,metrics):
            import re
            name=re.sub(r'[^A-Za-z0-9_-]','_',scenario)+'_'+key
            fig=make_figure(fresh,key,scenario)
            try:
                description=json.dumps(fig._alpr_analysis,ensure_ascii=False)
                for ext in ('png','svg'):
                    p=stage/(name+'.'+ext);fig.savefig(p,dpi=300 if ext=='png' else None,metadata={'Description':description});files.append(p)
            finally:plt.close(fig)
        report=stage/'analysis_report.md'
        lines=report_sections(fresh)
        comparison=[]
        for s in fresh.sessions:
            for u in s.quality_units:
                old=u.get('bbox_v1',{})
                comparison.append(dict(session_id=s.input.session_id,source_archive_sha256=s.input.source_sha256,
                    review_revision=s.metadata.get('review_revision'),unit_id=u['unit_id'],image=u['label'],
                    v1_status=old.get('status'),v1_tp=old.get('tp'),v1_fp=old.get('fp'),v1_fn=old.get('fn'),
                    v2_status=u['status'],v2_tp=u['tp'],v2_fp=u['fp'],v2_fn=u['fn'],
                    v1_reasons=old.get('unavailable_reasons',[]),v2_reasons=u['missing_reasons'],
                    operator_evidence=[p['operator_review_evidence'] for p in u.get('matched_pairs',[])],
                    review_tasks=u.get('review_tasks',[])))
        cp=stage/'v1_v2_all_units.csv'
        with cp.open('w',encoding='utf-8-sig',newline='') as stream:
            columns=list(comparison[0]) if comparison else ['session_id','unit_id','v1_status','v2_status']
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
            for row in comparison:writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
        files.append(cp)
        report.write_text('\n'.join(lines),encoding='utf-8');files.append(report)
        final=analyze(fresh.inputs,fresh.question)
        if fresh.question == 'F5A' and fresh.duplicates:
            final.duplicates=list(fresh.duplicates)
            final.performance=qualify(final.sessions,final.duplicates)
        if _fingerprint(final)!=_fingerprint(fresh):raise ValueError('Dane zmieniły się podczas eksportu. Nie opublikowano częściowego wyniku.')
        from ..ranking.mobile_mt_object_evaluation import POLICY_ID,POLICY_SHA256
        manifest=dict(schema='alpr.analysis.export.v2',analysis_protocol_id=PROTOCOL_ID,implementation_id=IMPLEMENTATION_ID,
            primary_matching_policy_id=POLICY_ID,primary_matching_policy_sha256=POLICY_SHA256,
            quality_method='F4-EVAL-2.0',quality_method_adopted='2026-11-01',gt_mode='ASSISTED',
            protocol_files=fresh.protocol_files,
            chronology='RETROSPECTIVE_FOR_F4_NOT_PREREGISTERED',generated_utc=datetime.now(timezone.utc).isoformat(),
            question=fresh.question,interpretation='DESCRIPTIVE',fingerprints=_fingerprint(fresh),
            sources=[dict(session_id=s.input.session_id,report_id=s.input.report_id,source_archive_sha256=s.input.source_sha256,
                source_path=s.input.source_path,quality_state=s.quality_state,**s.metadata) for s in fresh.sessions],
            files=[dict(name=p.name,bytes=p.stat().st_size,sha256=_sha(p)) for p in files],
            algorithm_sha256={p.name:_sha(p) for p in (Path(__file__),Path(__file__).with_name('mobile_analysis.py'),
                Path(__file__).with_name('mobile_analysis_figure.py'),Path(__file__).with_name('mobile_analysis_presentation.py'),DICTIONARY_PATH)})
        if fresh.question == 'F5A':
            from .mobile_analysis_f5a import ALGORITHM_ID
            manifest.update(implementation_id=ALGORITHM_ID, quality_method=None, quality_method_adopted=None,
                primary_matching_policy_id=None, primary_matching_policy_sha256=None,
                gt_mode='NOT_APPLICABLE_TIMING', chronology='RETROSPECTIVE_DESCRIPTIVE_PERFORMANCE',
                performance=fresh.performance, aggregation=payload['variant_metrics'], limitations=payload['limitations'])
            f5a_path=Path(__file__).with_name('mobile_analysis_f5a.py')
            manifest['algorithm_sha256'][f5a_path.name]=_sha(f5a_path)
        (stage/'analysis_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        stem='analiza_'+fresh.question+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        for i in range(1,100000):
            target=parent/f'{stem}_{i:03d}'
            if not target.exists():break
        else:raise ValueError('Brak wolnego numeru eksportu.')
        stage.rename(target)
        return target,fresh
    finally:
        if stage.exists():
            assert stage.resolve().is_relative_to(parent.resolve())
            shutil.rmtree(stage)
