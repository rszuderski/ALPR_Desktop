"""F5A/12B: existing metadata/UI contracts; no new Android wire format.

All archives below use the existing v2 fixture writer and synthetic pixels.
They are not producer fixtures for the blocked scene-evidence integration.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
import hashlib
import csv
import io
import json
import zipfile

import pytest

from auto_annotation_tool.ranking.mobile_experiment_identity import from_bundle


def metadata(variant='R2', series='F5A-SYNTHETIC-PILOT'):
    policy = {'R0': 'r0_full_frame', 'R1': 'r1_one_roi', 'R2': 'r2_two_roi'}[variant]
    config = dict(experiment_type='roi_budget', variant=variant, roi_budget_policy=policy,
                  analysis_mode='static', feature_flags=dict(lock=False, autozoom=False),
                  stages={k: dict(delegate='cpu', cpu_threads=2, precision='fp32') for k in ('mp', 'mt', 'mz')})
    common = dict(series_id=series, scenario_id='PAIRED_ROI_01', replicate_index=1)
    collection = dict(common, experiment_type='roi_budget', variant=variant,
                      execution=deepcopy(config), session_id='session-1')
    raw = dict(experiment=dict(common, type='roi_budget', variant=variant,
                             effective_execution_config=deepcopy(config)),
               research_execution_config=deepcopy(config),
               app_build=dict(build_type='debug', apk_sha256='a'*64))
    return NS(validation=NS(ok=True), report_payload=raw, collection_session=collection,
              experiment_session=dict(common, experiment_variant=variant),
              model_refs={}, report=NS(device=dict(name='Synthetic phone', android_version='12')))


def legacy_f5a_session(tmp_path, variant='R2', series='F5A-SYNTHETIC-PILOT', trace_rows=None):
    from test_mobile_p2b2_canvas_review import source_session
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    source = source_session(tmp_path/'template', n=1)
    with zipfile.ZipFile(source.path) as archive:
        entries = {n: archive.read(n) for n in archive.namelist()}
    meta = metadata(variant, series)
    report = json.loads(entries['report.json'])
    report.update(meta.report_payload)
    # The old Canvas-only helper used short checkpoint tokens; export reload
    # correctly requires SHA-256, so provide valid synthetic identities.
    for role, digest in [('plate', 'c'*64), ('character', 'd'*64)]:
        report.setdefault('model_provenance', {}).setdefault(role, {})['checkpoint_sha256'] = digest
    # The production importer pins measured_at. An absent timestamp would be
    # defaulted to "now" on each parse, making a synthetic source unstable.
    report['measured_at'] = '2026-01-01T00:00:00Z'
    collection = dict(meta.collection_session, session_id=source.session_id)
    report['research_collection'] = collection
    entries['report.json'] = json.dumps(report).encode()
    entries['session.json'] = json.dumps(collection).encode()
    if trace_rows is not None:
        stream = io.StringIO()
        columns = list(dict.fromkeys(k for row in trace_rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(trace_rows)
        entries['traces.csv'] = stream.getvalue().encode()
    unique_id = source.session_id + '-' + tmp_path.name
    def identity(value):
        if isinstance(value, dict):
            return {k: identity(v) for k, v in value.items()}
        if isinstance(value, list):
            return [identity(v) for v in value]
        if value == source.session_id:
            return unique_id
        if isinstance(value, str) and value.startswith(source.session_id+'/'):
            return unique_id + value[len(source.session_id):]
        return value
    for name, content in list(entries.items()):
        if name == 'manifest.json':
            continue
        if name.endswith('.json'):
            entries[name] = json.dumps(identity(json.loads(content))).encode()
        elif name.endswith('.jsonl'):
            entries[name] = ''.join(json.dumps(identity(json.loads(line)))+'\n' for line in content.decode().splitlines() if line).encode()
        elif name.endswith('.csv'):
            reader = csv.DictReader(io.StringIO(content.decode('utf-8')))
            out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
            writer.writeheader(); writer.writerows(identity(row) for row in reader)
            entries[name] = out.getvalue().encode()
    manifest = json.loads(entries.get('manifest.json', b'{"schema":"alpr.mobile_research_bundle.v1"}'))
    manifest['entry_sha256'] = {n: hashlib.sha256(b).hexdigest() for n, b in entries.items() if n != 'manifest.json'}
    entries['manifest.json'] = json.dumps(manifest).encode()
    path = tmp_path/'legacy_f5a.alprsession'
    with zipfile.ZipFile(path, 'w') as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    session = MobileReviewSession.open(path, sidecar_path=tmp_path/'review.json')
    session.set_review_mode('assisted')
    return session


@pytest.mark.parametrize('variant', ['R0', 'R1', 'R2'])
def test_explicit_roi_variant_and_pilot_are_not_precision_or_f4(variant):
    b = metadata(variant)
    before = deepcopy(b.__dict__)
    identity = from_bundle(b)
    assert identity['experiment'] == 'F5A' and identity['status'] == 'VERIFIED'
    assert identity['variant'] == variant and identity['is_pilot']
    assert identity['series_id'] == 'F5A-SYNTHETIC-PILOT'
    assert identity['variant_display'] == 'PILOT · '+variant
    assert b.__dict__ == before


@pytest.mark.parametrize('damage', ['variant', 'policy', 'series', 'type', 'source'])
def test_conflicting_or_unverified_identity_never_becomes_f5a(damage):
    b = metadata()
    if damage == 'variant': b.experiment_session['experiment_variant'] = 'R1'
    elif damage == 'policy': b.report_payload['research_execution_config']['roi_budget_policy'] = 'r1_one_roi'
    elif damage == 'series': b.collection_session['series_id'] = 'F5A-OFFICIAL'
    elif damage == 'type': b.report_payload['experiment']['type'] = 'quantization'
    else: b.validation.ok = False
    identity = from_bundle(b)
    assert identity['experiment'] == 'UNPROVEN' and identity['issues']


def test_models_alone_never_determine_experiment():
    b = NS(validation=NS(ok=True), report_payload=dict(variant_id='mp-ncnn-fp32-mt-tflite-fp32'),
           collection_session={}, experiment_session={})
    assert from_bundle(b)['status'] == 'NOT_APPLICABLE'


def test_pilot_and_official_series_remain_distinct():
    pilot, official = from_bundle(metadata()), from_bundle(metadata(series='F5A-SYNTHETIC'))
    assert pilot['is_pilot'] and not official['is_pilot']
    assert pilot['series_id'] != official['series_id']


def test_roi_metadata_aliases_reach_conditions_without_f4_label():
    from auto_annotation_tool.ranking.mobile_scientific_review import session_conditions, session_conditions_text
    session = NS(bundle=metadata(), session_id='session-1', review=NS(
        source_archive_sha256='b'*64, review_revision=15, review_status='COMPLETED', review_mode='assisted'))
    condition = session_conditions(session)
    text = session_conditions_text(condition)
    assert condition['variant'] == 'R2' and condition['series_id'].endswith('PILOT')
    assert set(condition['models']) == {'MP', 'MT', 'MZ'}
    assert condition['models']['MT']['delegate'] == 'cpu'
    assert 'F4' not in text and 'PILOT' in text and 'R2' in text


def test_operationally_resolved_na_is_not_forty_operator_reviews():
    from auto_annotation_tool.ranking.mobile_review_status import approval_caption
    inv = dict(total=40, inspected=40, assessed=2, operator_assessed=2, experiment_family='F5A',
               remaining=0, source_na=28, no_reference=10, decisions=0, approval_remaining=13)
    caption = approval_caption(inv, 'COMPLETED', 15)
    assert '2/40' in caption and 'N/A: 38' in caption
    assert '40/40 sprawdzone' not in caption and 'uzupełnij GTZ' not in caption


def test_legacy_f4_caption_remains_unchanged():
    from auto_annotation_tool.ranking.mobile_review_status import approval_caption
    assert approval_caption(dict(total=4, inspected=4, remaining=0), 'COMPLETED', 7) == '4/4 sprawdzone — zatwierdzona rewizja r7'


def test_new_scope_label_preserves_actual_rates_and_na():
    from auto_annotation_tool.ranking.mobile_session_metrics import presentation, snapshot_key
    from auto_annotation_tool.ranking.mobile_mt_object_evaluation import POLICY_ID, POLICY_SHA256
    snap = dict(session_id='s', source_archive_sha256='a'*64, review_revision=1,
                matching_policy_id=POLICY_ID, matching_policy_sha256=POLICY_SHA256,
                completion_eligibility_sha256='test', status='PARTIAL',
                mt=dict(n_qualified=2, n_total=40, precision=1., recall=1., f1=1.),
                geometry=dict(n_geometry=0, polygon_iou_mean_pct=None, nmce_mean_pct=None),
                ocr=dict(n_reads=0, n_gt_characters=0, n_exact=None, micro_cer=None))
    view = presentation(snap, snapshot_key(snap))
    assert view['assessment_scope'] == 'ASSISTED_SUBSET'
    assert view['values']['precision'] == '100,0%' and view['values']['cer'] == '—'
    assert '2/40' in view['denominators']


def test_native_legacy_import_keeps_mt_units_separate_from_tv(tmp_path):
    from auto_annotation_tool.ranking.mobile_canvas_presentation import build_inventory
    from auto_annotation_tool.gui.mobile_import_comparison import record_from_import
    session = legacy_f5a_session(tmp_path)
    inv = build_inventory(session)
    assert session.bundle.validation.ok and inv['total'] == 1
    assert inv['experiment_family'] == 'F5A' and inv['operator_assessed'] == 0
    rec, _, _ = record_from_import(session.bundle.report, session.bundle, include_review=False)
    assert rec.roi_variant == 'R2' and rec.is_pilot and rec.series_id.endswith('PILOT')
    assert 'PILOT' in rec.label and 'R2' in rec.label


def test_unified_f5a_routing_keeps_scene_e2e_unavailable(tmp_path):
    from auto_annotation_tool.gui.mobile_analysis import make_input, analyze
    session = legacy_f5a_session(tmp_path)
    inp = make_input(session.bundle.report, source_path=session.path, bundle=session.bundle,
                     results_root=tmp_path/'results', review_root=tmp_path/'reviews')
    result = analyze([inp], 'F5A').sessions[0]
    assert result.source_status == 'VERIFIED' and result.metadata['experiment'] == 'F5A'
    assert result.metadata['is_pilot'] and result.metadata['series_id'].endswith('PILOT')
    assert result.metrics['scene_coverage_pct'].value is None
    assert 'SCENE_UNMAPPED' in result.metrics['scene_coverage_pct'].missing_reasons
    assert 'QUESTION_SOURCE_NOT_PROVEN' not in result.metrics['pipeline_p50_ms'].missing_reasons
    f4 = analyze([inp], 'F4').sessions[0]
    assert f4.source_status == 'VERIFIED', f4.source_reason
    assert f4.metrics['mt_f1_pct'].status == 'NOT_COMPARABLE'


def test_pilot_cannot_join_official_r01_in_analysis(tmp_path):
    from auto_annotation_tool.gui.mobile_analysis import make_input, analyze
    inputs = []
    for name, series in [('pilot', 'F5A-SYNTHETIC-PILOT'), ('official', 'F5A-SYNTHETIC')]:
        session = legacy_f5a_session(tmp_path/name, series=series)
        # These are distinct synthetic archives; retain their native IDs.
        inp = make_input(session.bundle.report, source_path=session.path, bundle=session.bundle,
                         results_root=tmp_path/name/'results', review_root=tmp_path/name/'reviews')
        inputs.append(inp)
    cohort = analyze(inputs, 'F5A')
    assert len(cohort.sessions) == 2
    assert all(s.metrics['mt_f1_pct'].status == 'NOT_COMPARABLE' for s in cohort.sessions)
    assert all('F5A_SERIES_OR_PILOT_MISMATCH' in s.metrics['mt_f1_pct'].missing_reasons for s in cohort.sessions)


@pytest.mark.parametrize('calls', [2, 3, ''])
def test_stage_sums_or_missing_call_counts_are_not_per_call_latency(tmp_path, calls):
    from auto_annotation_tool.gui.mobile_analysis import make_input, analyze
    rows = [dict(frame_id=1, elapsed_ms=1, status='scan_queue_updated', total_ms=5,
                 plate_inference_ms='', mt_runs_this_frame='', character_inference_ms='', mz_runs=''),
            dict(frame_id=2, elapsed_ms=2, status='preliminary', total_ms=12,
                 plate_inference_ms=6, mt_runs_this_frame=calls, character_inference_ms=4, mz_runs=2)]
    session = legacy_f5a_session(tmp_path, trace_rows=rows)
    inp = make_input(session.bundle.report, source_path=session.path, bundle=session.bundle,
                     results_root=tmp_path/'results', review_root=tmp_path/'reviews')
    result = analyze([inp], 'F5A').sessions[0]
    assert result.source_status == 'VERIFIED', result.source_reason
    assert result.metrics['mt_p50_ms'].value is None
    assert 'MT_CALL_TIMING_UNPROVEN' in result.metrics['mt_p50_ms'].missing_reasons
    assert result.metrics['mz_trace_p50_ms'].value == 4
    assert result.metrics['pipeline_p50_ms'].value == 12


def test_f5a_dashboard_names_subset_pilot_and_trace_units():
    from test_z4mobrev031_science_dashboard import report
    from auto_annotation_tool.gui.mobile_science_viewmodel import make_science_dashboard
    r = report()
    r['conditions'] = dict(experiment_family='F5A', variant='R2', variant_display='PILOT · R2',
                           series_id='F5A-SYNTHETIC-PILOT', identity_status='VERIFIED')
    r['canvas_assessment'] = dict(n_qualified=2, n_total=40, scope='ASSISTED_SUBSET')
    model = make_science_dashboard(r, dict(status='READY_TO_APPROVE', total=15, verified=15))
    assert '2/40' in model['form_status'] and 'ASSISTED_SUBSET' in model['form_status']
    assert 'Formularz kompletny' not in model['form_status']
    assert 'Wariant F4' not in model['conditions'] and 'PILOT' in model['conditions']
    assert any(label == 'Seria' and value.endswith('PILOT') for label, value in model['source'])
    assert 'suma inferencji w śladzie' in model['latencies'][0]['label']
    assert model['progress'][1]['done'] is None


def test_existing_sidecar_export_reload_keeps_pilot_identity_and_decisions(tmp_path):
    from auto_annotation_tool.ranking import mobile_canvas_review as canvas
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    from auto_annotation_tool.ranking.mobile_scientific_review import make_scientific_summary, write_scientific_companion
    from auto_annotation_tool.gui.mobile_review_export_layout import finalize_export_manifest
    from auto_annotation_tool.gui.mobile_comparison_core import load_session_export
    session = legacy_f5a_session(tmp_path/'source')
    canvas.set_text(session, 'mt-one', 'GT:a0', 'SYN1234')
    before = json.loads(json.dumps(session.review.to_dict()))
    restored = MobileReviewSession.open(session.path, sidecar_path=session.sidecar_path)
    assert restored.review.to_dict() == before
    target = tmp_path/'export'; target.mkdir()
    paths = restored.export(target)
    # These existing mandatory companions carry no additional producer evidence.
    for name, content in [('mt_pose_geometry.csv', 'attempt_id,metric_status\n'),
                          ('mt_multi_plate_verified.json', '{}')]:
        path = target/name; path.write_text(content, encoding='utf-8'); paths.append(path)
    science = make_scientific_summary(restored, restored.statistics(), [])
    paths.extend(write_scientific_companion(science, target))
    finalize_export_manifest(restored, target, paths)
    exported = load_session_export(target)
    assert exported.experiment_type == 'roi_budget' and exported.identity_status == 'VERIFIED'
    assert exported.roi_variant == 'R2' and exported.is_pilot
    assert exported.series_id == 'F5A-SYNTHETIC-PILOT' and exported.scenario == 'PAIRED_ROI_01'
    assert exported.repeat_label == 'r01' and 'PILOT' in exported.label
    assert restored.review.to_dict() == before
    exported_review = json.loads((target/'review.json').read_text(encoding='utf-8'))
    assert exported_review['attempt_annotations'] == before['attempt_annotations']
