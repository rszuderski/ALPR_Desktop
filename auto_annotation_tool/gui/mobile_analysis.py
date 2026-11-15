"""ALPR-ANALYSIS-1.1 / F4-EVAL-2.0; read-only retrospective F4 analysis."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
import copy
import csv
import hashlib
import json
import math
import re

PROTOCOL_ID = 'ALPR-ANALYSIS-1.1'
IMPLEMENTATION_ID = 'alpr.desktop.analysis.v2'
PROTOCOL_DATE = '2026-11-01'
QUESTIONS = {'F4':'F4 · FP32 / INT8', 'F5A':'F5A · MP / ROI',
             'F5B':'F5B · AutoZoom', 'DIAGNOSTIC':'Diagnostyka źródeł'}
REGISTRY = Path(__file__).parents[1]/'ranking/protocols/alpr_analysis_1_0/f4_registry.csv'
def protocol_evidence():
    from ..ranking import mobile_mt_object_evaluation as object_policy
    sources = (REGISTRY, Path(__file__), Path(object_policy.__file__))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources} | {
        object_policy.POLICY_ID: object_policy.POLICY_SHA256,
    }
METRICS = {
    'mt_p50_ms':('Czasy · MT — mediana (p50)','ms','mt_inference_call'),
    'mt_p95_ms':('Czasy · MT — p95','ms','mt_inference_call'),
    'mz_trace_p50_ms':('Czasy · etap MZ / ślad — mediana (p50)','ms','mz_stage_per_trace'),
    'mz_trace_p95_ms':('Czasy · etap MZ / ślad — p95','ms','mz_stage_per_trace'),
    'pipeline_p50_ms':('Czasy · potok ALPR — mediana (p50)','ms','completed_pipeline_trace'),
    'pipeline_p95_ms':('Czasy · potok ALPR — p95','ms','completed_pipeline_trace'),
    'mt_calls':('Liczba wywołań MT','wywołania','recorded_mt_call'),
    'mz_calls':('Liczba wywołań MZ','wywołania','recorded_mz_call'),
    'mz_call_p95_ms':('MZ — p95 pojedynczego wywołania','ms','mz_inference_call'),
    'mt_tp':('MT v2 · TP','tablice','complete_mt_invocation_ROI'),
    'mt_fp':('MT v2 · FP','wskazania FINAL','complete_mt_invocation_ROI'),
    'mt_fn':('MT v2 · FN','tablice','complete_mt_invocation_ROI'),
    'mt_precision_pct':('MT v2 · Precision','%','complete_mt_invocation_ROI'),
    'mt_recall_pct':('MT v2 · Recall','%','complete_mt_invocation_ROI'),
    'mt_f1_pct':('MT v2 · F1','%','complete_mt_invocation_ROI'),
    'bbox_v1_tp':('Pomocnicze BBOX v1 · TP','tablice','complete_mt_invocation_ROI'),
    'bbox_v1_fp':('Pomocnicze BBOX v1 · FP','wskazania FINAL','complete_mt_invocation_ROI'),
    'bbox_v1_fn':('Pomocnicze BBOX v1 · FN','tablice','complete_mt_invocation_ROI'),
    'bbox_v1_precision_pct':('Pomocnicze BBOX v1 · Precision','%','complete_mt_invocation_ROI'),
    'bbox_v1_recall_pct':('Pomocnicze BBOX v1 · Recall','%','complete_mt_invocation_ROI'),
    'bbox_v1_f1_pct':('Pomocnicze BBOX v1 · F1','%','complete_mt_invocation_ROI'),
    'bbox_iou_pct':('Geometria · ciągłe BBOX IoU par v2','%','source_bbox_GT_pair'),
    'pose_availability_pct':('Pose · dostępność źródłowych narożników','%','source_final_linked_to_GT'),
    'polygon_iou_pct':('Geometria MT · Polygon IoU','%','source_pose_GT_pair'),
    'nmce_pct':('Geometria MT · NMCE','%','source_pose_GT_pair'),
    'exact_mz_pct':('Jakość MZ · pełne zgodne numery','%','executed_MZ_with_readable_GTZ'),
    'micro_cer_pct':('Jakość MZ · micro-CER','%','GT_character_in_executed_MZ'),
    'scene_coverage_pct':('Pokrycie zaplanowanych scen','%','planned_scene'),
    'exact_e2e_pct':('Pełny potok · Exact E2E','%','unique_plate_in_planned_window'),
    'no_read_e2e_pct':('Pełny potok · no-read','%','unique_plate_in_planned_window'),
    'pss_peak_mib':('PSS · maksimum dostępnych próbek','MiB','OS_PSS_measurement'),
}

def metric_definition(key):
    from .mobile_analysis_f5a import NEW_METRICS
    return NEW_METRICS[key] if key in NEW_METRICS else METRICS[key]


def metric_ids(question):
    from .mobile_analysis_f5a import METRIC_IDS
    return METRIC_IDS if question == 'F5A' else tuple(METRICS)

@dataclass(frozen=True)
class AnalysisInput:
    report: object
    session_id: str
    source_sha256: str
    report_id: str
    source_path: str
    results_root: Path | None
    review_root: Path
    review_seen: bool = False

def selection_identity(report, bundle=None):
    from .z4_mobile_report_browser_ux import report_identity
    info=report_identity(report,bundle)
    sid=info['session_id'];sha=str(getattr(report,'source_archive_sha256','')).lower()
    if not (str(getattr(report,'report_id','')).strip() and re.fullmatch(r'[A-Za-z0-9_.-]{1,160}',sid)
            and re.fullmatch(r'[a-f0-9]{64}',sha)):
        raise ValueError('Do analizy potrzebne są report_id, session_id i SHA-256 w indeksie importu.')
    return sid,sha

def make_input(report, *, source_path=None, results_root=None, review_root=None, bundle=None):
    from .mobile_report_source_recovery import stored_source
    from .mobile_report_canvas_view import results_root_for_source
    from ..config import CONFIG
    sid,sha=selection_identity(report,bundle)
    path=str(source_path or stored_source(report) or '')
    review_root=Path(review_root or CONFIG.DIR_7_RANKINGS_MOBILE_PACKAGES)
    return AnalysisInput(copy.deepcopy(report),sid,sha,str(report.report_id),path,
        Path(results_root) if results_root else results_root_for_source(path),
        review_root,(review_root/'human_reviews'/(sha+'.review.json')).is_file())

@dataclass(frozen=True)
class MetricEligibility:
    metric_id: str
    status: str = 'N/A'
    value: float | int | None = None
    n_scored: int | None = None
    n_total_eligible: int | None = None
    n_unscored: int | None = None
    missing_reasons: tuple[str,...] = ()
    numerator: float | int | None = None
    denominator: float | int | None = None
    gt_origin: str = 'NOT_APPLICABLE'
    review_revision: int | None = None
    inference_scope: str = 'UNAVAILABLE'
    evidence_paths: tuple[str,...] = ()
    explanation: str = ''
    population_basis: str = 'observed_source_units_not_planned_scenes'

    def payload(self):
        value=asdict(self)
        from ..ranking import mobile_mt_object_evaluation as objects,mobile_mt_detection_evaluation as bbox
        method=bbox if self.metric_id.startswith('bbox_v1_') else objects
        value.update(analysis_protocol_id=PROTOCOL_ID,implementation_id=IMPLEMENTATION_ID,
            matching_policy_id=method.POLICY_ID if self.gt_origin=='ASSISTED' else None,
            matching_policy_sha256=method.POLICY_SHA256 if self.gt_origin=='ASSISTED' else None,
            method_adopted_date=('2026-10-26' if method is bbox else '2026-11-01') if self.gt_origin=='ASSISTED' else None,
            unit=metric_definition(self.metric_id)[1],observation_unit=metric_definition(self.metric_id)[2],
            coverage=(self.n_scored/self.n_total_eligible if self.n_total_eligible and self.n_scored is not None else None),
            n_total_planned=None)
        return value

@dataclass
class SourceValidation:
    input: AnalysisInput
    source_status: str
    source_reason: str
    metadata: dict = field(default_factory=dict)
    quality_state: str = 'NONE'
    quality_reason: str = ''
    metrics: dict[str,MetricEligibility] = field(default_factory=dict)
    quality_units: list = field(default_factory=list)
    fingerprint: dict = field(default_factory=dict)
    conditions: dict = field(default_factory=dict)
    record: object = field(default=None,repr=False)
    bundle: object = field(default=None,repr=False)

    def payload(self):
        implementation=(self.metadata.get('f5a_performance') or {}).get('algorithm_id',IMPLEMENTATION_ID)
        return dict(session_id=self.input.session_id,report_id=self.input.report_id,
            source_archive_sha256=self.input.source_sha256,source_path=self.input.source_path,
            source_status=self.source_status,source_reason=self.source_reason,
            quality_state=self.quality_state,quality_reason=self.quality_reason,metadata=self.metadata,
            metrics={k:dict(m.payload(),implementation_id=implementation) for k,m in self.metrics.items()},quality_units=self.quality_units,
            fingerprint=self.fingerprint,conditions=self.conditions)

@dataclass
class AnalysisCohort:
    inputs: tuple[AnalysisInput,...]
    sessions: list[SourceValidation]
    question: str
    duplicates: list
    interpretation: str = 'DESCRIPTIVE'
    protocol_files: dict = field(default_factory=protocol_evidence)
    performance: dict = field(default_factory=dict)

    def payload(self):
        from ..ranking.mobile_mt_object_evaluation import POLICY_ID,POLICY_SHA256
        if self.question == 'F5A':
            from .mobile_analysis_f5a import ALGORITHM_ID, LIMITATIONS, aggregate
            return dict(analysis_protocol_id=PROTOCOL_ID, implementation_id=ALGORITHM_ID,
                chronology='RETROSPECTIVE_DESCRIPTIVE_PERFORMANCE', question=self.question,
                interpretation='DESCRIPTIVE', duplicates=self.duplicates, protocol_files=self.protocol_files,
                gt_mode='NOT_APPLICABLE_TIMING', limitations=LIMITATIONS, performance=self.performance,
                sessions=[s.payload() for s in self.sessions],
                variant_metrics=[dict(scenario=scenario, **row)
                    for scenario in sorted({s.metadata.get('scenario', 'NIEOKREŚLONY') for s in self.sessions})
                    for key in metric_ids('F5A') for row in aggregate(self, key, scenario)])
        return dict(analysis_protocol_id=PROTOCOL_ID,protocol_date=PROTOCOL_DATE,
            primary_matching_policy_id=POLICY_ID,primary_matching_policy_sha256=POLICY_SHA256,
            protocol_files=self.protocol_files,
            chronology='RETROSPECTIVE_FOR_F4_NOT_PREREGISTERED',implementation_id=IMPLEMENTATION_ID,
            question=self.question,interpretation=self.interpretation,duplicates=self.duplicates,
            limitations=['Brak dowodu parowania fotografii; brak kontrolowanego wniosku przyczynowego.',
                'F4-EVAL-2.0 przyjęto retrospektywnie 2026-11-01. Główne MT v2 to ocena operatora; BBOX v1 jest odrębną miarą pomocniczą.',
                'DAY i NIGHT są osobnymi materiałami TV, nie dwoma checkpointami dziennym i nocnym.',
                'GT ASSISTED nie jest niezależnym F6; różne podzbiory jakości nie tworzą rankingu wariantów.',
                'n śladów i n wywołań nie są liczbą niezależnych plansz; MZ/ślad może sumować wywołania.'],
            sessions=[s.payload() for s in self.sessions])

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()

def _missing(s, metric, reason, *, n=None, total=None, explanation=''):
    needs_gt=metric.startswith('bbox_v1_') or metric in ('bbox_iou_pct','mt_tp','mt_fp','mt_fn','mt_precision_pct','mt_recall_pct','mt_f1_pct',
        'pose_availability_pct','polygon_iou_pct','nmce_pct','exact_mz_pct','micro_cer_pct','exact_e2e_pct','no_read_e2e_pct')
    s.metrics[metric]=MetricEligibility(metric,n_scored=n,n_total_eligible=total,
        n_unscored=total-n if n is not None and total is not None else None,
        missing_reasons=(reason,),explanation=explanation or s.source_reason or s.quality_reason,
        evidence_paths=(s.input.source_path,),gt_origin=('ASSISTED' if s.metadata.get('review_revision') is not None else 'UNVERIFIED') if needs_gt else 'NOT_APPLICABLE',
        review_revision=s.metadata.get('review_revision') if needs_gt else None)

def _metric(s,key,value,n,total,*,reasons=(),quality=False,numerator=None,denominator=None,subset=False,explanation=''):
    status='N/A' if value is None else 'SUBSET' if subset or (total is not None and n<total) else 'FULL'
    s.metrics[key]=MetricEligibility(key,status,value,n,total,total-n if total is not None else None,
        tuple(dict.fromkeys(reasons)),numerator,denominator,'ASSISTED' if quality else 'NOT_APPLICABLE',
        s.metadata.get('review_revision') if quality else None,
        'UNAVAILABLE' if value is None else 'ASSISTED_SUBSET' if quality and status=='SUBSET' else 'ASSISTED' if quality else 'DESCRIPTIVE',
        tuple(s.metadata.get('quality_evidence',[])) if quality else (s.input.source_path,),explanation)


def _quality(s,review):
    from .mobile_analysis_object_quality import quality
    return quality(s,review)


def _source(inp, *, timing_only=False):
    from .mobile_report_source_recovery import restore_source
    from .mobile_import_comparison import record_from_import,_read_all_traces,_sha_file
    from .mobile_report_canvas_view import selected_canvas_export
    from .mobile_comparison_scope import trace_digest
    from .z4_mobile_report_browser_ux import report_identity
    info=report_identity(inp.report)
    s=SourceValidation(inp,'SOURCE_MISSING','',metadata=dict(scenario=info['scenario'] or 'NIEOKREŚLONY',
        variant=info['variant'],repeat='—',experiment='UNPROVEN'))
    restored=restore_source(inp.report,inp.source_path)
    if not restored.ok:
        s.source_status='SOURCE_MISMATCH' if restored.status in ('mismatch','no_fingerprint') else 'SOURCE_MISSING' if restored.status=='missing' else 'MANIFEST_INVALID'
        s.source_reason=restored.detail+' · '+inp.source_path
        for key in METRICS:_missing(s,key,s.source_status)
        s.fingerprint=dict(source_status=s.source_status,expected_sha=inp.source_sha256)
        return s
    s.bundle=restored.bundle
    try:
        record,rows,warnings=record_from_import(inp.report,s.bundle,include_review=False)
        if record.session_id!=inp.session_id or _sha_file(Path(inp.source_path))!=inp.source_sha256:
            raise ValueError('Sprzeczny session_id/SHA albo zmiana źródła podczas odczytu.')
        s.source_status='VERIFIED';s.source_reason='Oryginalne archiwum, manifest i tożsamość sprawdzone.';s.record=record
        _,trace_state=_read_all_traces(s.bundle)
        from ..ranking.mobile_experiment_identity import from_bundle as experiment_identity
        diagnostic = experiment_identity(s.bundle)
        if diagnostic.get('family') == 'F5A':
            f4_variant = None
        else:
            from ..ranking.mobile_f4_variant import from_bundle
            f4_variant=from_bundle(s.bundle,session_id=inp.session_id,registry_path=REGISTRY)
        mapping=f4_variant if f4_variant and f4_variant['status']=='VERIFIED' else None
        s.metadata.update(scenario=record.scenario,variant=record.precision.upper(),repeat=record.repeat_label,
            apk_sha256=record.apk_sha256,device=record.device,mt_checkpoint=record.mt_checkpoint,mz_checkpoint=record.mz_checkpoint,
            package_and_models=record.model_identity,trace_count=len(rows),trace_state=trace_state,registry_verified=bool(mapping))
        if f4_variant:
            s.metadata.update(f4_variant=f4_variant,variant_provenance=f4_variant['provenance'],
                declared_android_variant=f4_variant['declared_android_variant'])
        if diagnostic.get('family') == 'DIAGNOSTIC':
            s.metadata.update(experiment=diagnostic['experiment'], variant=diagnostic['variant_display'],
                registry_verified=False, measurement_purpose=diagnostic.get('measurement_purpose'),
                declared_android_variant=diagnostic.get('variant'), is_pilot=False,
                identity_status=diagnostic['status'], series_id=diagnostic.get('series_id'))
        elif mapping:
            s.metadata.update(experiment='F4',variant=mapping['analysis_variant'],repeat=mapping['repeat'],
                source_variant=mapping['source_variant'],source_replicate_index=mapping['source_replicate_index'],
                mapping_evidence=str(REGISTRY),mapping_sha256=_sha_file(REGISTRY))
        else:
            from ..ranking.mobile_experiment_identity import from_bundle
            identity = from_bundle(s.bundle)
            if identity.get('family') == 'F5A':
                s.metadata.update(experiment=identity['experiment'], experiment_type=identity.get('experiment_type'),
                    variant=identity['variant_display'], series_id=identity.get('series_id'),
                    is_pilot=identity['is_pilot'], identity_status=identity['status'],
                    identity_issues=identity['issues'], roi_budget_policy=identity.get('roi_budget_policy'),
                    variant_provenance=identity.get('provenance'), declared_android_variant=identity.get('variant'),
                    repeat=('PILOT · ' if identity['is_pilot'] else '') + record.repeat_label)
        from ..ranking.mobile_tv_bbox_evidence import input_context as tv_input_context
        tv_context = tv_input_context(s.bundle.collection_session, getattr(s.bundle, 'tv_bbox_evidence', {}) or {})
        if tv_context:
            s.metadata.update(input_mode=tv_context['input_mode'], input_context=tv_context)
        for stage,prefix in (('mt','mt'),('mz','mz_trace'),('pipeline','pipeline')):
            met=record.latency[stage];total=len(rows) if trace_state=='complete' else None
            for pct in ('p50','p95'):
                key=f'{prefix}_{pct}_ms'
                if trace_state!='complete':_missing(s,key,'INCOMPLETE_TRACE_SOURCE')
                else:_metric(s,key,getattr(met,pct+'_ms'),met.n,total,
                    reasons=('STAGE_NOT_RUN_OR_INVALID_TRACE',) if met.n<len(rows) else (),
                    explanation='Pełny strumień Androida; percentyl per sesja. MZ to suma etapu w śladzie, nie pojedyncze MZ.')
        valid=[r for r in rows if str(r.get('status','')).lower() not in ('scene_superseded','runtime_error','pipeline_error','cancelled','not_run')]
        for key,col in (('mt_calls','mt_runs_this_frame'),('mz_calls','mz_runs')):
            try:
                values=[int(r[col]) for r in valid];assert all(v>=0 for v in values) and trace_state=='complete'
                _metric(s,key,sum(values),len(valid),len(rows))
                if key=='mt_calls' and any(v!=1 for v in values):
                    for pct in ('p50','p95'):_missing(s,f'mt_{pct}_ms','MT_CALL_TIMING_UNPROVEN',explanation='Czas śladu może sumować wiele MT; brak pojedynczych czasów wywołań.')
            except (KeyError,ValueError,TypeError,AssertionError):_missing(s,key,'CALL_COUNT_UNAVAILABLE')
        if s.metadata.get('experiment') == 'F5A':
            # A stage sum is a call duration only if every timed trace proves
            # exactly one call. Missing MP-only counters cannot waive this guard.
            timed = [r for r in rows if r.get('plate_inference_ms') not in (None, '')
                     and str(r.get('status', '')).lower() not in
                     ('scene_superseded', 'runtime_error', 'pipeline_error', 'cancelled', 'not_run')]
            try:
                single = bool(timed) and all(int(r.get('mt_runs_this_frame', '')) == 1 for r in timed)
            except (ValueError, TypeError):
                single = False
            if not single:
                for pct in ('p50', 'p95'):
                    _missing(s, f'mt_{pct}_ms', 'MT_CALL_TIMING_UNPROVEN',
                        explanation='Zapisany czas MT jest sumą w śladzie; brak dowodu czasu pojedynczego wywołania. Surowa telemetria pozostaje dostępna.')
        _missing(s,'mz_call_p95_ms','PER_CALL_TIMING_UNAVAILABLE',explanation='Nie dzielimy percentyli sumy czasu przez liczbę MZ.')
        for key in ('scene_coverage_pct','exact_e2e_pct','no_read_e2e_pct'):
            _missing(s,key,'SCENE_UNMAPPED',explanation='Brak zweryfikowanej mapy plansza→GT i reguły wyboru odczytu z okna.'
                + ('' if timing_only else ' F4 nie jest niezależnym F6.'))
        pss=[]
        for r in rows:
            try:
                v=float(r.get('pss_end_kb',''));assert math.isfinite(v) and v>=0;pss.append(v/1024)
            except (ValueError,TypeError,AssertionError):pass
        if pss:_metric(s,'pss_peak_mib',max(pss),len(pss),len(rows))
        else:_missing(s,'pss_peak_mib','OS_PSS_NOT_MEASURED')
        if timing_only:
            if diagnostic.get('family') == 'F5A':
                from .mobile_analysis_f5a import project_timings
                project_timings(s, rows, trace_state)
            s.quality_state='NOT_APPLICABLE_TIMING'
            s.quality_reason='Automatyczne pomiary wydajności; ocena GT poza zakresem tego pytania.'
            for key in METRICS:
                if key not in s.metrics:_missing(s,key,'QUALITY_OUTSIDE_TIMING_SCOPE')
            s.fingerprint=dict(source_status=s.source_status,source_sha=inp.source_sha256,trace_digest=trace_digest(rows),
                metric_digest=_digest({k:m.payload() for k,m in s.metrics.items()}),metadata_digest=_digest(s.metadata))
            return s
        side=inp.review_root/'human_reviews'/(inp.source_sha256+'.review.json')
        side_hash=None
        try:
            side_hash=_sha_file(side) if side.is_file() else None
            gate,review=selected_canvas_export(inp.session_id,inp.source_sha256,inp.results_root,inp.review_root)
            s.quality_state=gate.state;s.quality_reason=gate.reason
            if inp.review_seen and not side.is_file():
                s.quality_state='STALE';s.quality_reason='Bieżąca ocena zniknęła po wyborze; jakość wymaga ponownej weryfikacji źródła eksportu.';review=None
            if review is not None and side.is_file():
                live=json.loads(side.read_text(encoding='utf-8-sig'));exported=json.loads((review.directory/'review.json').read_text(encoding='utf-8-sig'))
                for doc in (live,exported):doc.pop('derived_metrics',None)
                if live!=exported:
                    s.quality_state='STALE';s.quality_reason='Treść bieżącej oceny różni się od eksportu; wymagany świeży eksport.';review=None
            _quality(s,review)
            quality_manifest=review.manifest_sha256 if review else None
        except (OSError,ValueError,TypeError,KeyError) as exc:
            s.quality_state='UNREADABLE';s.quality_reason=str(exc);quality_manifest=None;_quality(s,None)
        s.fingerprint=dict(source_status=s.source_status,source_sha=inp.source_sha256,trace_digest=trace_digest(rows),
            quality_state=s.quality_state,review_manifest_sha=quality_manifest,sidecar_sha=side_hash,
            metric_digest=_digest({k:m.payload() for k,m in s.metrics.items()}),metadata_digest=_digest(s.metadata))
        return s
    except (OSError,ValueError,TypeError,KeyError) as exc:
        s.source_status='SOURCE_MISMATCH';s.source_reason=str(exc);s.record=None
        s.metrics={}
        for key in METRICS:_missing(s,key,s.source_status)
        s.fingerprint=dict(source_status=s.source_status,expected_sha=inp.source_sha256)
        return s

def analyze(inputs, question='F4'):
    if question not in QUESTIONS:raise ValueError('F6 wymaga osobnego niezależnego raportu; nie jest porównaniem wariantów F4.')
    unique=[];duplicates=[];ids=set();shas=set()
    for inp in inputs:
        if inp.session_id in ids or inp.source_sha256 in shas:
            duplicates.append(dict(session_id=inp.session_id,source_archive_sha256=inp.source_sha256,reason='DUPLICATE_SOURCE_NOT_INDEPENDENT'));continue
        unique.append(inp);ids.add(inp.session_id);shas.add(inp.source_sha256)
    sessions=[_source(inp,timing_only=question=='F5A') for inp in unique]
    if question == 'F5A':
        for s in sessions:
            for key in metric_ids('F5A'):
                if key not in s.metrics:_missing(s,key,s.source_status if s.source_status != 'VERIFIED' else 'QUESTION_SOURCE_NOT_PROVEN')
    from .mobile_report_scientific_comparability import assess_cohort
    reports=[s.bundle.report for s in sessions if s.bundle is not None and s.source_status=='VERIFIED']
    from ..ranking.mobile_mz_input_evidence import diagnostic_cohort_conflict, is_diagnostic_purpose
    mixed_purposes = diagnostic_cohort_conflict(s.metadata.get('measurement_purpose', '')
        for s in sessions if s.source_status == 'VERIFIED')
    from ..ranking.mobile_tv_bbox_evidence import cohort_conflict as tv_conflict
    input_conflict = tv_conflict(s.metadata.get('input_context') or {'input_mode': s.metadata.get('input_mode', 'CAMERA_FULL_FRAME')}
        for s in sessions if s.source_status == 'VERIFIED')
    for s in sessions:
        if question=='F4' and not s.metadata.get('registry_verified') and not is_diagnostic_purpose(s.metadata.get('measurement_purpose')):
            s.metadata.update(variant=(s.metadata.get('f4_variant') or {}).get('label','NIEPOTWIERDZONE'),repeat='—')
            if 'metadata_digest' in s.fingerprint:s.fingerprint['metadata_digest']=_digest(s.metadata)
        if s.source_status=='VERIFIED':s.conditions=assess_cohort(s.bundle.report,reports,bundle=s.bundle)
        if s.source_status=='VERIFIED' and (input_conflict or mixed_purposes or (question!='DIAGNOSTIC' and s.metadata.get('experiment')!=question)):
            reason = input_conflict or ('DIAGNOSTIC_PURPOSE_COHORT_MISMATCH' if mixed_purposes else 'QUESTION_SOURCE_NOT_PROVEN')
            allow_performance = (question == 'F5A' and not mixed_purposes and s.metadata.get('experiment') == 'F5A'
                and (s.metadata.get('f5a_performance') or {}).get('status') == 'PASS')
            s.metrics={k:m if allow_performance and k in metric_ids('F5A') else replace(m,status='NOT_COMPARABLE',value=None,missing_reasons=tuple((*m.missing_reasons,reason)),
                n_scored=None,n_total_eligible=None,n_unscored=None,numerator=None,denominator=None,review_revision=None,inference_scope='UNAVAILABLE',
                explanation='Brak udowodnionego materiału dla tego pytania. Źródło pozostaje w diagnostyce.') for k,m in s.metrics.items()}
    if question == 'F5A':
        f5a = [s for s in sessions if s.source_status == 'VERIFIED' and s.metadata.get('experiment') == 'F5A']
        cohorts = {(s.metadata.get('series_id'), s.metadata.get('is_pilot')) for s in f5a}
        if len(cohorts) > 1:
            for s in f5a:
                s.metrics = {k: replace(m, status='NOT_COMPARABLE', value=None,
                    missing_reasons=(*m.missing_reasons, 'F5A_SERIES_OR_PILOT_MISMATCH'),
                    numerator=None, denominator=None, inference_scope='UNAVAILABLE',
                    explanation='PILOT i właściwa seria F5A pozostają osobnymi kohortami.') for k, m in s.metrics.items()}
    result = AnalysisCohort(tuple(unique),sessions,question,duplicates)
    if question == 'F5A':
        from .mobile_analysis_f5a import qualify
        result.performance = qualify(sessions, duplicates)
        algorithm=Path(__file__).with_name('mobile_analysis_f5a.py')
        result.protocol_files={algorithm.name:hashlib.sha256(algorithm.read_bytes()).hexdigest()}
        for s in sessions:
            s.fingerprint.update(metric_digest=_digest({k:m.payload() for k,m in s.metrics.items()}), metadata_digest=_digest(s.metadata))
    return result
