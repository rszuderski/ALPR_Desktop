"""Human progress never comes from source availability or automatic N/A.

All writes use generated archives and sidecars below pytest tmp_path.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
from PIL import Image

from auto_annotation_tool.config import CONFIG
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking.mobile_canvas_presentation import build_inventory, progress_caption, summary_caption
from auto_annotation_tool.ranking import mobile_scene_frame_review as scenes
from auto_annotation_tool.ranking.mobile_operator_progress import from_session, presentation, image_counts, evidence_caption
from auto_annotation_tool.ranking.mobile_review_status import approval_caption
from auto_annotation_tool.ranking.mobile_package_experiments import MobilePackageExperimentStore, read_mobile_report_bundle
from auto_annotation_tool.gui.mobile_report_canvas_view import canvas_overview, load_canvas_snapshot
from auto_annotation_tool.gui.mobile_compare_review_gate import imported_verification
from auto_annotation_tool.gui.mobile_report_science_dashboard import science_from_verified_bundle, build_overview


@pytest.fixture(autouse=True)
def private_review_root(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG, 'DIR_7_RANKINGS_MOBILE_PACKAGES', tmp_path / 'catalog')


def source(tmp_path, *, n=10, name='synthetic', pending_loss=False):
    """Native, manifest-validated scene registry with generated 160x100 pixels."""
    sid = 'synthetic-' + name
    session = dict(schema='alpr.mobile_research_session.v1', session_id=sid, experiment_type='roi_budget',
                   series_id='F5A-SYNTHETIC', scenario_id='SYNTHETIC', variant='R2', replicate_index=1,
                   scene_evidence_expected=n, scene_evidence_written=n, scene_evidence_missing=0,
                   scene_evidence_integrity_loss_count=0, scene_evidence_complete=not pending_loss,
                   scene_evidence_pending_loss_unknown=pending_loss, mt_detection_roles_complete=True,
                   execution=dict(experiment_type='roi_budget', variant='R2', roi_budget_policy='r2_two_roi'))
    report = dict(schema='alpr.mobile_benchmark_report.v1', report_id=sid, session_id=sid,
                  measured_at='2026-01-01T00:00:00Z', research_collection=deepcopy(session),
                  experiment=dict(type='roi_budget', series_id=session['series_id'], scenario_id='SYNTHETIC',
                                  variant='R2', replicate_index=1, session=dict(id=sid)),
                  latency=dict(pipeline=dict(count=10, median_ms=100, p95_ms=120)))
    descriptor = dict(schema='alpr.mobile_research_samples.v2', normalization_policy='uppercase_alphanumeric.v1',
                      capabilities=dict(scene_evidence_v1=True, mt_detection_roles_v1=True),
                      scene_evidence_file='samples/scene_evidence.jsonl', mt_detection_roles_file='samples/mt_detection_roles.jsonl')
    entries = {'session.json': json.dumps(session).encode(), 'report.json': json.dumps(report).encode(),
               'samples/schema.json': json.dumps(descriptor).encode(), 'samples/attempts.csv': b'attempt_id,mt_executed\n',
               'samples/index.csv': b'capture_id\n', 'samples/mt_detection_roles.jsonl': b''}
    rows = []
    for index in range(1, n + 1):
        key = f'{sid}-s{index:08d}'
        entry = 'samples/frames/scene-' + key + '.jpg'
        image = io.BytesIO(); Image.new('RGB', (160, 100), (index, 80, 90)).save(image, format='JPEG')
        entries[entry] = image.getvalue()
        stamp = dict(scene_generation=index, visual_epoch=index, camera_transform_generation=1,
                     source_sequence=index * 10, source_timestamp_nanos=index * 1000000, source_timestamp_domain='CAMERAX_SENSOR')
        rows.append(dict(schema='alpr.scene_evidence.v1', session_id=sid, scene_event_id=key,
                         event_type='SESSION_START' if index == 1 else 'SCENE_BOUNDARY',
                         event_reason='session_start' if index == 1 else 'synthetic_boundary', transition_revision=None,
                         event_stamp=stamp, source_stamp=stamp, observed_at_utc='2026-01-01T00:00:00Z',
                         event_elapsed_nanos=index, capture_elapsed_nanos=index, capture_duration_nanos=0,
                         source_frame_entry=entry, source_frame_sha256=hashlib.sha256(entries[entry]).hexdigest(),
                         source_frame_width_px=160, source_frame_height_px=100,
                         source_frame_coordinate_space='mt_source_bitmap_xy', source_frame_origin='pipeline_bitmap_before_roi',
                         write_status='WRITTEN', missing_reason=None, settling_status='NOT_ASSESSED', mp_observations=[],
                         mt_invocation_ids=[], mt_execution_status='NOT_RUN', mt_not_run_reason='synthetic_zero_roi', links_status='COMPLETE'))
    entries['samples/scene_evidence.jsonl'] = ''.join(json.dumps(r)+'\n' for r in rows).encode()
    manifest = dict(schema='alpr.mobile_research_bundle.v1', **{k:v for k,v in session.items()
        if k.startswith('scene_evidence_') or k == 'mt_detection_roles_complete'})
    manifest['entry_sha256'] = {k:hashlib.sha256(v).hexdigest() for k,v in entries.items()}
    entries['manifest.json'] = json.dumps(manifest).encode()
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / (name + '.alprsession')
    with zipfile.ZipFile(path, 'w') as archive:
        for k,v in entries.items(): archive.writestr(k, v)
    bundle = read_mobile_report_bundle(path)
    assert bundle.validation.ok, bundle.validation.errors
    return MobileReviewSession(bundle)


def progress(session):
    return build_inventory(session)['operator_review_progress']


def view(session):
    science = science_from_verified_bundle(session.bundle)
    science['_ui_canvas_snapshot'] = load_canvas_snapshot(session.session_id, session.bundle.source_archive_sha256,
                                                          None, CONFIG.DIR_7_RANKINGS_MOBILE_PACKAGES)
    return build_overview(session.bundle.report, session.bundle, science)


def test_T01_T13_new_import_never_writes_gt_or_invents_progress(tmp_path):
    s = source(tmp_path / 'source')
    before = s.path.read_bytes()
    store = MobilePackageExperimentStore(tmp_path / 'store')
    store.import_mobile_report_file(s.path)
    assert not s.sidecar_path.exists()
    data = view(s)
    assert data['canvas']['status'] == 'Nie rozpoczęto'
    assert data['canvas']['progress'] == 'Obrazy ocenione przez operatora: 0/10'
    assert data['canvas']['progress_ratio'] == 0
    assert imported_verification(s.session_id, s.bundle.source_archive_sha256, None, CONFIG.DIR_7_RANKINGS_MOBILE_PACKAGES).label == 'Nie rozpoczęto'
    assert not s.sidecar_path.exists() and s.path.read_bytes() == before
    assert not s.review.invocation_annotations and not s.review.scene_annotations


@pytest.mark.parametrize('detections', [0, 2])
def test_T02_predictions_and_auto_no_reference_never_count_as_human(tmp_path, detections):
    from test_mobile_p2b2_canvas_review import source_session
    s = source_session(tmp_path, n=detections)
    before = s.sidecar_path.read_bytes()
    inventory = build_inventory(s)
    p = inventory['operator_review_progress']
    assert p['operator_state'] == 'NOT_STARTED' and p['mt_assessed'] == 0
    assert p['assessed'] == 0 and p['total'] is None
    assert presentation(p)['progress_ratio'] is None
    if detections == 0:
        assert inventory['inspected'] == 1 and p['evidence']['no_reference'] == 1
        assert 'MT/ROI bez referencji: 1' in evidence_caption(p)
    assert before == s.sidecar_path.read_bytes()


def test_T03_two_written_frames_not_two_human_assessments(tmp_path):
    s = source(tmp_path, n=2)
    p = progress(s)
    assert p['evidence']['written_scenes'] == 2 and p['assessed'] == 0
    assert presentation(p)['progress'] == 'Obrazy ocenione przez operatora: 0/2'
    assert s.bundle.report.latency['pipeline']['count'] == 10  # timing is not the denominator


def test_T04_T05_real_partial_decisions_have_pinned_provenance(tmp_path):
    s = source(tmp_path)
    keys = list(s.scene_frames)
    s.set_reviewer('synthetic-reviewer')
    assert progress(s)['operator_state'] == 'NOT_STARTED'  # entering author is not a decision
    scenes.confirm_empty(s, keys[0])
    p = progress(s)
    assert (p['assessed'], p['total'], p['operator_state']) == (1, 10, 'IN_PROGRESS')
    assert p['units'][0]['id'] == keys[0] and p['units'][0]['source_binding']
    assert p['sidecar_sha256'] == hashlib.sha256(s.sidecar_path.read_bytes()).hexdigest()
    assert 'synthetic-reviewer' in p['provenance'] and str(s.sidecar_path) in p['provenance']
    scenes.add_plate(s, keys[1], [(10, 10), (40, 10), (40, 30), (10, 30)])
    assert progress(s)['assessed'] == 2 and s.review.review_status == 'IN_PROGRESS'
    assert 'nie oznacza ukończenia oceny GT/MT/MZ' in presentation(progress(s))['progress_subtitle']
    assert imported_verification(s.session_id, s.bundle.source_archive_sha256, None, CONFIG.DIR_7_RANKINGS_MOBILE_PACKAGES).label == 'W toku'


def test_T06_completed_review_is_distinct_from_missing_scientific_export(tmp_path):
    s = source(tmp_path)
    for key in s.scene_frames: scenes.confirm_empty(s, key)
    s.complete()
    p = progress(s)
    assert (p['assessed'], p['total'], p['operator_state']) == (10, 10, 'COMPLETED')
    data = view(s)['canvas']
    assert data['status'] == 'Zakończona' and data['progress_ratio'] == 1
    gate = imported_verification(s.session_id, s.bundle.source_archive_sha256, None, CONFIG.DIR_7_RANKINGS_MOBILE_PACKAGES)
    assert gate.state == 'NEEDS_EXPORT' and not gate.ready
    assert 'Zakończona' in gate.label and 'nie ma aktualnego eksportu' in gate.reason


def test_T07_T12_resume_and_fresh_process_preserve_real_decisions(tmp_path):
    s = source(tmp_path)
    scenes.confirm_empty(s, next(iter(s.scene_frames)))
    before = s.sidecar_path.read_bytes()
    reloaded = MobileReviewSession.open(s.path, sidecar_path=s.sidecar_path)
    assert progress(reloaded)['assessed'] == 1
    code = ('import sys,json; from pathlib import Path; '
            'from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession; '
            'from auto_annotation_tool.ranking.mobile_canvas_presentation import build_inventory; '
            's=MobileReviewSession.open(Path(sys.argv[1]),sidecar_path=Path(sys.argv[2])); '
            'print(json.dumps(build_inventory(s)["operator_review_progress"]))')
    output = subprocess.check_output([sys.executable, '-c', code, str(s.path), str(s.sidecar_path)], text=True)
    fresh = json.loads(output.splitlines()[-1])
    assert fresh['assessed'] == 1 and fresh['review_revision'] == s.review.review_revision
    assert before == s.sidecar_path.read_bytes()


@pytest.mark.parametrize('damage', ['sha', 'sid', 'invalid_json', 'binding'])
def test_T08_bad_sidecar_never_claims_zero_or_approval(tmp_path, damage):
    s = source(tmp_path)
    key = next(iter(s.scene_frames)); scenes.confirm_empty(s, key)
    data = json.loads(s.sidecar_path.read_text())
    if damage == 'sha': data['source_archive_sha256'] = '0' * 64
    if damage == 'sid': data['session_id'] = 'other'
    if damage == 'binding': data['scene_annotations'][key]['source_binding'] = '0' * 64
    s.sidecar_path.write_text('{' if damage == 'invalid_json' else json.dumps(data), encoding='utf-8')
    p = from_session(s, {'groups': {}, 'no_reference': 0, 'source_na': 0})
    assert p['operator_state'] == 'UNVERIFIED' and p['assessed'] is None
    shown = presentation(p)
    assert shown['progress_ratio'] is None and 'postęp operatora niepotwierdzony' in shown['progress']
    with pytest.raises(ValueError): MobileReviewSession.open(s.path, sidecar_path=s.sidecar_path)


def test_T09_unknown_population_does_not_use_timing_count(tmp_path):
    s = source(tmp_path, n=2, pending_loss=True)
    p = progress(s)
    assert p['total'] is None and p['evidence']['written_scenes'] == 2
    assert s.bundle.report.latency['pipeline']['count'] == 10
    assert presentation(p)['progress'] == 'Nie ustalono liczby obrazów do oceny'
    assert presentation(p)['progress_ratio'] is None


def test_T10_shared_counter_in_browser_review_status_and_canvas(tmp_path):
    s = source(tmp_path)
    scenes.confirm_empty(s, next(iter(s.scene_frames)))
    inv = build_inventory(s); p = inv['operator_review_progress']; shown = presentation(p)
    browser = view(s)
    assert browser['canvas']['progress'] == browser['progress'] == shown['progress']
    assert browser['canvas']['progress_ratio'] == shown['progress_ratio'] == .1
    assert shown['progress'] in approval_caption(inv, s.review.review_status, s.review.review_revision)
    assert shown['progress'] in progress_caption(inv)
    assert image_counts(p) == dict(assessed=1, total=10, remaining=9, fraction='1/10')


def test_T11_foreign_progress_cannot_leak_to_another_session(tmp_path):
    a = source(tmp_path / 'a', name='a'); b = source(tmp_path / 'b', name='b')
    scenes.confirm_empty(a, next(iter(a.scene_frames)))
    other = canvas_overview({}, session_id=b.session_id, source_sha=b.bundle.source_archive_sha256, progress=progress(a))
    assert '1/10' not in other['progress'] and other['progress_ratio'] is None
    assert view(b)['canvas']['progress'] == 'Obrazy ocenione przez operatora: 0/10'


def test_T14_legacy_metrics_and_auto_eligibility_are_not_rewritten(tmp_path):
    from test_mobile_p2b2_canvas_review import source_session
    from auto_annotation_tool.ranking import mobile_canvas_review as core, mobile_mt_object_evaluation as mt
    s = source_session(tmp_path, n=0)
    before = s.path.read_bytes(), s.sidecar_path.read_bytes(), core.summary(s), mt.evaluate_session(s)
    inv = build_inventory(s)
    assert inv['inspected'] == 1 and inv['operator_review_progress']['mt_assessed'] == 0
    assert (s.path.read_bytes(), s.sidecar_path.read_bytes(), core.summary(s), mt.evaluate_session(s)) == before


def test_T04_T05_T07_T14_legacy_explicit_mt_decisions_remain_separate(tmp_path):
    from test_z4_mt_geometry_layers import source_session
    from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial
    s = source_session(tmp_path, n=2, mz=False)
    partial.confirm_stack(s, 'mt-one', oid='GT:a0')
    p = progress(s)
    assert p['operator_state'] == 'IN_PROGRESS' and p['mt_assessed'] == 0
    partial.confirm_stack(s, 'mt-one', oid='GT:a1')
    before = s.sidecar_path.read_bytes()
    reloaded = MobileReviewSession.open(s.path, sidecar_path=s.sidecar_path)
    p = progress(reloaded)
    assert p['mt_assessed'] == 1 and p['total'] is None and p['assessed'] == 0
    assert 'MT/ROI: zapisane oceny 1/1' in presentation(p)['progress_subtitle']
    assert p['units'][0]['source_binding'] and before == s.sidecar_path.read_bytes()


def test_T08_invalid_canvas_binding_and_failed_load_do_not_claim_review(tmp_path):
    from test_z4_mt_geometry_layers import source_session
    from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial
    s = source_session(tmp_path, n=1, mz=False)
    partial.confirm_stack(s, 'mt-one', oid='GT:a0')
    s.review.invocation_annotations['mt-one']['canvas_review_v1']['source_binding'] = '0' * 64
    s.sidecar_path.write_text(json.dumps(s.review.to_dict()), encoding='utf-8')
    p = progress(s)
    assert p['sidecar_state'] == 'INVALID' and p['mt_assessed'] is None
    data = build_overview(s.bundle.report, s.bundle, error='Invalid source binding')['canvas']
    assert data['status'] == 'Nie można potwierdzić weryfikacji' and data['progress_ratio'] is None


def test_T01_status_without_any_decisions_does_not_invent_human_review(tmp_path):
    s = source(tmp_path)
    s.set_reviewer('synthetic-reviewer')
    data = json.loads(s.sidecar_path.read_bytes())
    data['review_status'] = 'IN_PROGRESS'
    s.sidecar_path.write_text(json.dumps(data), encoding='utf-8')
    s = MobileReviewSession.open(s.path, sidecar_path=s.sidecar_path)
    assert progress(s)['operator_state'] == 'NOT_STARTED'
    gate = imported_verification(s.session_id, s.bundle.source_archive_sha256, None, CONFIG.DIR_7_RANKINGS_MOBILE_PACKAGES)
    assert gate.label == 'Nie rozpoczęto'


def test_T10_native_sample_view_uses_same_persisted_counter(tmp_path, monkeypatch):
    import test_mobile_p2b2_canvas_gui as native
    from auto_annotation_tool.gui.mobile_canvas_clean_view import context
    s = source(tmp_path, n=2)
    monkeypatch.setattr(native, 'source_session', lambda _: s)
    generator = native.gui.__wrapped__(tmp_path, monkeypatch)
    root, host, editor, _ = next(generator)
    try:
        for expected in ('0/2', '1/2'):
            editor.update_header()
            assert 'Obrazy ocenione przez operatora: ' + expected in host.progress_label.cget('text')
            assert context(editor, {})['session_counts']['inspected'] == expected
            assert view(s)['canvas']['progress'] == 'Obrazy ocenione przez operatora: ' + expected
            if expected == '0/2':
                assert 'Nie rozpoczęto' in host.form_status_label.cget('text')
                scenes.confirm_empty(s, next(iter(s.scene_frames)))
        assert '1/2' in approval_caption(editor.inventory(), s.review.review_status, s.review.review_revision)
    finally:
        try: next(generator)
        except StopIteration: pass


def test_T14_verified_historical_export_not_zeroed_when_current_sidecar_absent(tmp_path):
    from test_z4_akt89f_results_identity import approved_export, SHA, SID
    from auto_annotation_tool.ranking.mobile_operator_progress import SCHEMA
    approved_export(tmp_path)
    snap = load_canvas_snapshot(SID, SHA, tmp_path, None)
    p = dict(schema=SCHEMA, source_verified=True, session_id=SID, source_sha256=SHA, sidecar_state='ABSENT',
             operator_state='NOT_STARTED', assessed=0, total=None)
    shown = canvas_overview(snap, session_id=SID, source_sha=SHA, progress=p)
    assert shown['ready'] and shown['status'] == 'ZATWIERDZONA'
    assert '19/19' in shown['progress'] and 'Pochodzenie zatwierdzonej oceny' in shown['progress_subtitle']
    # Historical LIMITED exports may contain automatic N/A in inspected.
    # Keep the approval/provenance, never turn that inventory into a human bar.
    limited = deepcopy(snap)
    limited['verification']['state'] = 'APPROVED_LIMITED'
    limited['summary']['session_id'] = SID
    limited['summary']['operator_progress'] = dict(inspected=19)
    shown = canvas_overview(limited, session_id=SID, source_sha=SHA, progress=p)
    assert shown['status'].startswith('ZATWIERDZONA')
    assert shown['progress'] == 'Nie ustalono liczby obrazów do oceny' and shown['progress_ratio'] is None
    assert 'Archiwalny stan operacyjny MT/ROI: 19/19' in shown['progress_subtitle']


def test_T10_native_progress_bar_hides_unknown_and_tracks_human_ratio(tmp_path):
    import tkinter as tk
    from auto_annotation_tool.gui.mobile_report_science_dashboard import ScientificDashboard
    s = source(tmp_path)
    root = tk.Tk(); root.withdraw()
    panel = ScientificDashboard(root, palette={}, on_review=lambda: None, on_refresh=lambda: None, on_details=lambda: None)
    panel.pack(fill='both', expand=True)
    try:
        data = view(s); panel.render(data); root.update()
        assert panel.status_var.get() == 'Nie rozpoczęto' and '0/10' in panel.progress_var.get()
        scenes.confirm_empty(s, next(iter(s.scene_frames)))
        panel.render(view(s)); root.update()
        assert float(panel.progress_bar.cget('value')) == 10
        unknown = deepcopy(data); unknown['canvas'].update(progress='Nie ustalono liczby obrazów do oceny', progress_ratio=None)
        panel.render(unknown); root.update()
        assert panel.progress_bar.winfo_manager() == ''
    finally:
        root.destroy()
