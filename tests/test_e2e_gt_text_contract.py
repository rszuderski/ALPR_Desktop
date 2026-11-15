"""Frozen EVAL396 text contract: approved strings, unreadable nulls."""
from collections import Counter
from copy import deepcopy

import pytest

from auto_annotation_tool.ranking import e2e_eval_metrics as metrics
from auto_annotation_tool.ranking import e2e_eval_runner as runner
from test_e2e_geometry_and_metrics import full_reference


COUNTS = {
    'ALL': dict(scenes=396, bbox=479, pose=478, ocr=449, characters=3052),
    'DAY': dict(scenes=199, bbox=235, pose=234, ocr=230, characters=1522),
    'NIGHT': dict(scenes=197, bbox=244, pose=244, ocr=219, characters=1530),
}


@pytest.fixture(params=['metrics', 'runner'])
def validate(request, tmp_path, monkeypatch):
    if request.param == 'metrics':
        def check(reference):
            return reference, metrics.validate_reference(reference)
        return check

    # Only upstream file/cache loaders are replaced. Exercise the real runner's
    # decision/type handling and its call to validate_reference, without models.
    selection_dir = tmp_path / 'evaluation_selections' / 'SYNTHETIC'
    cache_dir = tmp_path / 'evaluation_runs' / 'EVAL396_MT_run_SYNTHETIC'
    cache_dir.mkdir(parents=True)
    (cache_dir / 'completion_manifest.json').write_text('SYNTHETIC ONLY')
    monkeypatch.setattr(runner, 'MT_COMPLETION_SHA', runner.mt._sha(cache_dir / 'completion_manifest.json'))

    def check(reference):
        plates = [dict(plate_id=p['plate_id'], source_file_sha256=s['sha256'],
                       decision=p['text_decision'], ground_truth_text=p['ground_truth_text'])
                  for s in reference for p in s['plates']]
        selection = dict(selection_dir=str(selection_dir), repo_root=str(tmp_path),
                         plates=plates, models=[], source_files={},
                         selection_sha=runner.SELECTION_SHA, gt_freeze_sha=runner.FREEZE_SHA)
        cache = dict(status='COMPLETE', recovery_required=False,
                     protocol=dict(protocol_sha256=runner.MT_PROTOCOL_SHA,
                                   metric_counts={'per_domain': {d: {k: v for k, v in c.items() if k != 'characters'}
                                                                 for d, c in COUNTS.items()}}),
                     preflight=dict(selection_dir=str(selection_dir), scenes=deepcopy(reference),
                                    source_scene_root=str(tmp_path / 'images'), freeze_dir=str(tmp_path / 'freeze')),
                     source_receipt={}, source_files={}, geometry_approval={'contract_sha256': 'TEST_ONLY'},
                     models=[dict(label=label, rows=[]) for label in runner.MT_LABELS])
        before = deepcopy((selection, cache))
        monkeypatch.setattr(runner, 'load_selection', lambda *a, **k: selection)
        monkeypatch.setattr(runner.mt, 'load_campaign', lambda *a, **k: cache)
        result = runner._source_context(selection_dir, cache_dir)
        assert (selection, cache) == before
        return result['reference'], result['counts']
    return check


def test_frozen_text_types_and_denominators_remain_unchanged(validate):
    reference = full_reference()
    before = deepcopy(reference)
    result, counts = validate(reference)
    assert counts == COUNTS and reference == before and result == before
    assert Counter((p['text_decision'], type(p['ground_truth_text']).__name__)
                   for s in result for p in s['plates']) == {
                       ('approved', 'str'): 449, ('excluded_unreadable', 'NoneType'): 30}


@pytest.mark.parametrize('decision,text,error', [
    ('approved', None, 'E2E_GT_TEXT_MISSING'),
    ('approved', 123, 'E2E_GT_TEXT_MISSING'),
    ('approved', '', 'E2E_APPROVED_GT_TEXT_EMPTY'),
    ('approved', ' - / ', 'E2E_APPROVED_GT_TEXT_EMPTY'),
    ('pending', None, 'E2E_GT_DECISION_INVALID'),
    ('excluded_unreadable', '', 'E2E_UNREADABLE_GT_TEXT_INVALID'),
    ('excluded_unreadable', 'ABC', 'E2E_UNREADABLE_GT_TEXT_INVALID'),
    ('excluded_unreadable', 123, 'E2E_UNREADABLE_GT_TEXT_INVALID'),
])
def test_invalid_text_or_decision_remains_blocking(validate, decision, text, error):
    reference = full_reference()
    plate = reference[0]['plates'][0]
    plate.update(text_decision=decision, ocr_eligible=decision == 'approved', ground_truth_text=text)
    with pytest.raises(ValueError, match=error):
        validate(reference)


def test_unreadable_nulls_never_enter_text_normalization_or_alignment(monkeypatch):
    normalizer, aligner = metrics.normalize_registration, metrics.align_plate_text
    normalized, aligned = [], []
    def normalize(text):
        assert isinstance(text, str)
        normalized.append(text)
        return normalizer(text)
    def align(text, prediction):
        assert isinstance(text, str) and text
        aligned.append(text)
        return aligner(text, prediction)
    monkeypatch.setattr(metrics, 'normalize_registration', normalize)
    monkeypatch.setattr(metrics, 'align_plate_text', align)
    reference = full_reference()
    rows = [dict(scene_sha256=s['sha256'], width=320, height=200, status='COMPLETE', detections=[])
            for s in reference]
    result = metrics.score_pair(rows, reference)
    assert len(normalized) == len(aligned) == 449
    assert sum(map(len, aligned)) == 3052
    unreadable = [r for r in result['gt_rows'] if not r['readable']]
    assert len(unreadable) == 30 and all(r['scoring'] is None for r in unreadable)
    for domain, counts in COUNTS.items():
        scored = result['domains'][domain]
        assert scored['box_fn'] == counts['bbox']
        assert scored['plates'] == scored['no_read'] == counts['ocr']
        assert scored['ground_truth_characters'] == counts['characters']
        assert scored['cer'] == 1 and scored['exact_match_rate'] == 0
