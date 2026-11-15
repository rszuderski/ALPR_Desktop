"""Z4MOBREP020: verified Desktop GT overlay and truthful X-only wheel zoom."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'auto_annotation_tool/gui'


def load_helper(filename, name):
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


zoom = load_helper('mobile_report_x_zoom.py', 'mob_view_zoom20')
review = load_helper('mobile_report_review_overlay.py', 'mob_review_overlay20')


def bundle_with_sample():
    source_sha = 'a' * 64
    row = {'capture_id': 's1', 'subject_key': 'exp/sg-1/track-1',
           'verification_status': 'not_reviewed', 'ground_truth': '',
           'prediction': 'MODEL'}
    bundle = SimpleNamespace(source_archive_sha256=source_sha, validation=SimpleNamespace(ok=True),
                             report=SimpleNamespace(report_id='r1'), sample_rows=[row])
    science = {'source_archive_sha256': source_sha, 'session_id': 'exp', 'review_revision': 4,
               'subjects': {'exp/sg-1/track-1': {'label': 'Ocenione MT i MZ'}}}
    return bundle, science


def make_payload(tmp_path, bundle, science, **updates):
    payload = {'schema': 'alpr.mobile_human_review.v1',
               'source_archive_sha256': bundle.source_archive_sha256,
               'session_id': science['session_id'], 'review_revision': 4,
               'subjects': {'exp/sg-1/track-1': {'ground_truth': 'TRUTH', 'evaluable': True,
                                                'gt_confirmed': True}},
               'sample_annotations': {}}
    payload.update(updates)
    folder = tmp_path / 'human_reviews'
    folder.mkdir(exist_ok=True)
    (folder / (bundle.source_archive_sha256 + '.review.json')).write_text(json.dumps(payload), encoding='utf-8')
    return payload


def decoder(payload, *, source_sha256, session_id):
    assert payload['source_archive_sha256'] == source_sha256
    assert payload['session_id'] == session_id
    return SimpleNamespace(review_revision=payload['review_revision'],
                           subjects=payload['subjects'], sample_annotations=payload.get('sample_annotations', {}))


def test_existing_operator_gt_overlays_immutable_phone_not_reviewed(tmp_path):
    bundle, science = bundle_with_sample()
    make_payload(tmp_path, bundle, science)
    result = review.project_desktop_samples(bundle, science, tmp_path, decode_review=decoder)
    assert result['state'] == 'current'
    assert result['samples']['s1']['status'] == 'Ocenione MT i MZ'
    assert result['samples']['s1']['gt'] == 'TRUTH'
    assert bundle.sample_rows[0]['verification_status'] == 'not_reviewed'
    assert bundle.sample_rows[0]['ground_truth'] == ''


def test_no_sidecar_is_not_started_without_fabricating_gt(tmp_path):
    bundle, science = bundle_with_sample()
    assert review.project_desktop_samples(bundle, science, tmp_path, decode_review=decoder) == {
        'state': 'not_started', 'revision': None, 'samples': {}}


def test_wrong_archive_or_wrong_session_cannot_link_gt(tmp_path):
    bundle, science = bundle_with_sample()
    make_payload(tmp_path, bundle, science, session_id='another-session')
    with pytest.raises(ValueError, match='session_id'):
        review.project_desktop_samples(bundle, science, tmp_path, decode_review=decoder)
    make_payload(tmp_path, bundle, science)
    science['source_archive_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='tożsamość'):
        review.project_desktop_samples(bundle, science, tmp_path, decode_review=decoder)


def test_changed_review_revision_is_not_mixed_with_cached_science(tmp_path):
    bundle, science = bundle_with_sample()
    make_payload(tmp_path, bundle, science, review_revision=5)
    with pytest.raises(ValueError, match='zmieniona'):
        review.project_desktop_samples(bundle, science, tmp_path, decode_review=decoder)


def test_draft_and_excluded_mz_do_not_claim_evaluated_gt(tmp_path):
    bundle, science = bundle_with_sample()
    make_payload(tmp_path, bundle, science, subjects={'exp/sg-1/track-1': {'draft_ground_truth': 'DRAFT', 'evaluable': None}},
                 sample_annotations={'s1': {'evaluable': False}})
    got = review.project_desktop_samples(bundle, science, tmp_path, decode_review=decoder)
    assert got['samples']['s1']['gt'] == ''
    assert got['samples']['s1']['status'] == 'Wyłączono odczyt MZ'


def test_ambiguous_identity_does_not_borrow_gt_from_other_subject(tmp_path):
    bundle, science = bundle_with_sample()
    make_payload(tmp_path, bundle, science)
    bundle.sample_rows.append({'capture_id': 's2', 'subject_key': 'another subject'})
    got = review.project_desktop_samples(bundle, science, tmp_path, decode_review=decoder)
    assert list(got['samples']) == ['s1']


def test_wheel_zooms_x_and_keeps_cursor_anchor():
    full = (1.0, 239.0)
    result = zoom.zoom_x_view(full, None, 0.20, zoom_in=True, integer_axis=True)
    assert result and result[1] - result[0] < 238
    assert result[0] >= full[0] and result[1] <= full[1]
    assert abs((result[0] + (result[1]-result[0])*.20) - (1 + 238*.20)) < 1e-8
    wider = zoom.zoom_x_view(full, result, 0.2, zoom_in=False, integer_axis=True)
    assert wider is None or wider[1] - wider[0] > result[1]-result[0]


def test_drag_pan_clamps_to_existing_frame_id_bounds():
    full = (1., 239.)
    viewport = (1., 21.)
    moved = zoom.pan_x_view(full, viewport, -100, 500)
    assert moved == pytest.approx((5.,25.))
    assert zoom.pan_x_view(full, viewport, 10000, 500) == pytest.approx(viewport)
    assert zoom.clamp_x_view(full, (-100., 500.)) is None


def test_zoom_draw_uses_only_visible_measurements_without_resampling():
    xs, ys = (1., 2., 3., 239.), (2., None, 4., 9.)
    visible_x, visible_y = zoom.visible_xy(xs, ys, 1., 5.)
    assert visible_x == (1.,2.,3.) and visible_y == (2.,None,4.)
    assert xs[-1] == 239. and ys[-1] == 9.


def test_browser_uses_real_desktop_overlay_and_canvas_event_bindings():
    source = (BASE / 'z4_mobile_report_browser.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MobileReportBrowser')
    crops = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_populate_crops')
    crop_src = ast.get_source_segment(source, crops)
    assert 'verification_status' not in crop_src and 'row.get("ground_truth")' not in crop_src
    assert 'decision["status"]' in crop_src and 'decision["gt"]' in crop_src
    assert 'self._zoom_latency_wheel' in source
    assert '"<MouseWheel>"' in source and '"<Double-Button-1>"' in source
    assert 'self._latency_x_view' in source
    assert 'visible_xy(xs, values, min_x, max_x)' in source
