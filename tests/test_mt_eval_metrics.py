"""Pure metric fixtures; no model, decoder, GUI, inference or data writes."""
from copy import deepcopy
import json
from math import hypot

import pytest

from auto_annotation_tool.ranking.mt_eval_metrics import aggregate, box_iou, distribution, METRIC_PROTOCOL


def plate(identifier='p1', x=10, *, excluded=False):
    box = [x, 10, x+80, 40]
    return {'plate_id': identifier, 'bbox': box,
            'quad': [[x, 10], [x+80, 10], [x+80, 40], [x, 40]],
            'bbox_eligible': True, 'pose_eligible': not excluded,
            'pose_exclusion': {'reason_code': 'INDEX_IDENTITY_UNKNOWN'} if excluded else None,
            'pose_evaluation_status': 'EXCLUDED_BY_OPERATOR' if excluded else 'ELIGIBLE'}


def scene(plates=None, *, sha='a'*64, domain='DAY'):
    return {'sha256': sha, 'domain': domain, 'file': domain+'/fixture.jpg',
            'image': {'oriented_size': [500, 200]}, 'plates': [plate()] if plates is None else plates}


def prediction(gt=None, *, index=0, confidence=.9, keypoints='DEFAULT', **extra):
    gt = gt or plate()
    return {'index': index, 'confidence': confidence, 'class_id': 0, 'bbox': deepcopy(gt['bbox']),
            'keypoints': deepcopy(gt['quad'] if keypoints == 'DEFAULT' else keypoints), **extra}


def row(gt_scene, predictions):
    return {'scene_sha256': gt_scene['sha256'], 'domain': gt_scene['domain'],
            'file': gt_scene['file'], 'width': 500, 'height': 200, 'predictions': predictions}


def test_perfect_indexed_pose_has_real_zero_errors_and_unchanged_inputs():
    gt = scene()
    pred = prediction(keypoints=[[x, y, .3+i*.1] for i, (x, y) in enumerate(gt['plates'][0]['quad'])])
    rows, sources = [row(gt, [pred])], [gt]
    before = deepcopy((rows, sources))
    result = aggregate(rows, sources)
    metrics = result['domains']['ALL']
    assert result['status'] == 'COMPLETE'
    assert (metrics['box_tp'], metrics['box_fp'], metrics['box_fn']) == (1, 0, 0)
    assert metrics['box_precision'] == metrics['box_recall'] == metrics['box_f1'] == 1
    assert metrics['pose_coverage'] == 1 and metrics['pose_valid_matched'] == 1
    assert metrics['corner_error_px'] == {'n': 4, 'mean': 0, 'median': 0, 'p90': 0, 'p95': 0}
    assert result['scenes'][0]['matched'][0]['raw_keypoints'] == pred['keypoints']
    assert result['official_metrics'] == {'box_ap': 'NOT_COMPUTED', 'pose_ap': 'NOT_COMPUTED'}
    assert (rows, sources) == before


def test_box_all_gt_pose_exclusion_is_not_fp_and_duplicate_is_not_absorbed():
    first, second, excluded = plate('p1'), plate('p2', 110), plate('plate_000421', 210, excluded=True)
    gt = scene([first, second, excluded])
    bad_excluded = [excluded['quad'][i] for i in (0, 2, 1, 3)]
    output = aggregate([row(gt, [prediction(first), prediction(first, index=1, confidence=.8),
                                prediction(excluded, index=2, confidence=.7, keypoints=bad_excluded)])], [gt])
    m = output['domains']['ALL']
    assert (m['box_gt'], m['box_tp'], m['box_fp'], m['box_fn']) == (3, 2, 1, 1)
    assert m['box_precision'] == m['box_recall'] == m['box_f1'] == pytest.approx(2/3)
    assert (m['pose_gt'], m['pose_matched_gt'], m['pose_unmatched_gt'], m['pose_valid_matched']) == (2, 1, 1, 1)
    assert m['pose_coverage'] == .5 and m['pose_unmatched_predictions'] == 1
    assert m['pose_excluded_gt'] == m['pose_excluded_matched'] == 1
    assert m['pose_invalid_quad'] == 0
    excluded_match = output['scenes'][0]['matched'][1]
    assert excluded_match['pose_status'] == 'NOT_EVALUATED'
    assert excluded_match['corner_errors_px'] is None
    assert m['prediction_diagnostics']['raw_quad_status_counts']['SELF_INTERSECTING'] == 1


def test_no_detections_complete_scene_has_fn_and_undefined_precision_no_fake_errors():
    gt = scene([plate(), plate('plate_000421', 110, excluded=True)])
    output = aggregate([row(gt, [])], [gt])
    m = output['domains']['ALL']
    assert output['status'] == 'COMPLETE'
    assert (m['box_tp'], m['box_fp'], m['box_fn']) == (0, 0, 2)
    assert m['box_precision'] is None and m['box_recall'] == m['box_f1'] == 0
    assert m['pose_unmatched_gt'] == 1 and m['pose_excluded_gt'] == 1
    assert m['pose_coverage'] == 0 and m['corner_error_px']['mean'] is None
    assert m['no_detection_scenes'] == 1


def test_partial_results_do_not_turn_pending_scenes_into_fn():
    day, night = scene(), scene([plate('night1')], sha='b'*64, domain='NIGHT')
    result = aggregate([row(day, [prediction()])], [day, night])
    m = result['domains']['ALL']
    assert result['status'] == 'PARTIAL'
    assert (m['processed_scenes'], m['pending_scenes'], m['cohort_scenes']) == (1, 1, 2)
    assert (m['cohort_box_gt'], m['box_gt'], m['box_fn']) == (2, 1, 0)
    assert m['cohort_pose_gt'] == 2 and m['pose_coverage'] == 1
    assert result['domains']['NIGHT']['box_recall'] is None
    assert result['domains']['NIGHT']['pending_scenes'] == 1


def test_empty_partial_campaign_and_empty_gt_scene_are_distinct():
    gt = scene([])
    assert aggregate([], [gt])['status'] == 'PARTIAL'
    full = aggregate([row(gt, [prediction()])], [gt])
    m = full['domains']['ALL']
    assert full['status'] == 'COMPLETE' and m['box_fp'] == 1 and m['box_fn'] == 0
    assert m['box_precision'] == 0 and m['box_recall'] is None and m['box_f1'] == 0
    assert m['pose_coverage'] is None


def test_matching_is_confidence_first_and_ties_stable_under_input_reordering():
    a, b = plate('a'), plate('b')
    gt = scene([b, a])
    low, high = prediction(index=1, confidence=.8), prediction(index=9, confidence=.9)
    expected = aggregate([row(gt, [low, high])], [gt])
    assert [(m['prediction_index'], m['plate_id']) for m in expected['scenes'][0]['matched']] == [(9, 'a'), (1, 'b')]
    assert aggregate([row(scene([a, b]), [high, low])], [scene([a, b])]) == expected
    ties = aggregate([row(gt, [prediction(index=5), prediction(index=2)])], [gt])
    assert [(m['prediction_index'], m['plate_id']) for m in ties['scenes'][0]['matched']] == [(2, 'a'), (5, 'b')]


def test_confidence_boundary_matches_strict_ultralytics_filter_and_class_filter():
    gt = scene()
    predictions = [prediction(index=0, confidence=.25), prediction(index=1, confidence=.249),
                   prediction(index=2, confidence=.9, class_id=1), prediction(index=3, confidence=.250001)]
    m = aggregate([row(gt, predictions)], [gt])['domains']['ALL']
    assert (m['box_tp'], m['box_fp'], m['box_fn']) == (1, 0, 0)
    assert m['prediction_diagnostics']['filtered_confidence'] == 2
    assert m['prediction_diagnostics']['filtered_class'] == 1


def test_iou_threshold_is_inclusive_and_continuous_without_pixel_increment():
    gt = scene()
    pred = prediction()
    pred['bbox'] = [10, 10, 50, 40]
    assert box_iou(pred['bbox'], gt['plates'][0]['bbox']) == .5
    assert aggregate([row(gt, [pred])], [gt])['domains']['ALL']['box_tp'] == 1
    pred['bbox'][2] = 49.999
    m = aggregate([row(gt, [pred])], [gt])['domains']['ALL']
    assert (m['box_tp'], m['box_fp'], m['box_fn']) == (0, 1, 1)


def test_reverse_raw_order_measures_original_indices_instead_of_repairing():
    gt = scene()
    quad = [gt['plates'][0]['quad'][i] for i in (0, 3, 2, 1)]
    output = aggregate([row(gt, [prediction(keypoints=quad)])], [gt])
    m = output['domains']['ALL']
    diagonal = hypot(80, 30)
    assert m['pose_reversed_winding'] == m['pose_valid_matched'] == 1
    assert output['scenes'][0]['matched'][0]['corner_errors_px'] == pytest.approx([0, diagonal, 0, diagonal])
    assert output['scenes'][0]['matched'][0]['corner_errors_normalized'] == pytest.approx([0, 1, 0, 1])
    assert m['corner_error_normalized']['mean'] == .5


def test_excluded_pose_still_counts_in_unindexed_geometry_diagnostics():
    gt = scene([plate('plate_000421', excluded=True)])
    out = aggregate([row(gt, [prediction(gt['plates'][0])])], [gt])
    m = out['domains']['ALL']
    assert m['pose_gt'] == m['pose_valid_matched'] == 0
    assert m['pose_coverage'] is None
    assert m['unindexed_geometry_diagnostics'] == {'gt':1, 'matched':1, 'valid_matched':1, 'coverage':1}
    assert out['scenes'][0]['matched'][0]['corner_errors_px'] is None


@pytest.mark.parametrize('keypoints,extra,category,status', [
    (None, {}, 'pose_missing_keypoints', 'MISSING_KEYPOINTS'),
    ([], {}, 'pose_missing_keypoints', 'MISSING_KEYPOINTS'),
    ([[10,10], [90,10], [90,40]], {}, 'pose_invalid_quad', 'INVALID_QUAD'),
    ([[10,10], [90,40], [90,10], [10,40]], {}, 'pose_invalid_quad', 'INVALID_QUAD'),
    ([[float('nan'),10], [90,10], [90,40], [10,40]], {}, 'pose_invalid_quad', 'INVALID_QUAD'),
    ([[{'__nonfinite_float__':'nan'},10], [90,10], [90,40], [10,40]], {}, 'pose_invalid_quad', 'INVALID_QUAD'),
    ([[10,10], [90,10], [90,40], [10,40]], {'bbox_fallback':True}, 'pose_bbox_fallback', 'BBOX_FALLBACK'),
    ([[10,10], [90,10], [90,40], [10,40]], {'keypoint_source':'display_polygon'}, 'pose_bbox_fallback', 'BBOX_FALLBACK'),
    ([[-2,10], [90,10], [90,40], [-2,40]], {}, 'pose_invalid_quad', 'INVALID_QUAD'),
])
def test_box_tp_with_missing_invalid_or_fallback_pose_never_has_corner_error(keypoints, extra, category, status):
    gt = scene()
    output = aggregate([row(gt, [prediction(keypoints=keypoints, **extra)])], [gt])
    m = output['domains']['ALL']
    assert m['box_tp'] == 1 and m[category] == 1 and m['pose_coverage'] == 0
    assert output['scenes'][0]['matched'][0]['pose_status'] == status
    assert m['corner_error_px']['n'] == 0 and m['corner_error_px']['mean'] is None
    json.dumps(output, allow_nan=False)


def test_invalid_bbox_is_fp_and_does_not_repair_from_valid_keypoints():
    gt = scene()
    pred = prediction()
    pred['bbox'] = [float('inf'), 10, 90, 40]
    output = aggregate([row(gt, [pred])], [gt])
    m = output['domains']['ALL']
    assert (m['box_tp'], m['box_fp'], m['box_fn']) == (0, 1, 1)
    assert m['prediction_diagnostics']['invalid_bbox'] == 1
    assert output['scenes'][0]['predictions'][0]['bbox'][0] == {'__nonfinite_float__': 'inf'}
    json.dumps(output, allow_nan=False)


def test_linear_distribution_and_empty_values_are_explicit():
    assert distribution([0, 10, 20, 30]) == pytest.approx({'n':4, 'mean':15., 'median':15., 'p90':27., 'p95':28.5})
    assert distribution([]) == {'n':0, 'mean':None, 'median':None, 'p90':None, 'p95':None}


@pytest.mark.parametrize('edit,reason', [
    (lambda r: r.update(scene_sha256='unknown'), 'PREDICTION_SCENE_UNKNOWN_OR_DUPLICATE'),
    (lambda r: r.update(domain='NIGHT'), 'PREDICTION_DOMAIN_MISMATCH'),
    (lambda r: r.update(width=200,height=500), 'PREDICTION_INPUT_DIMENSIONS_MISMATCH'),
    (lambda r: r['predictions'][0].update(confidence=float('nan')), 'PREDICTION_CONFIDENCE_INVALID'),
    (lambda r: r['predictions'][0].update(index=-1), 'PREDICTION_INDEX_INVALID_OR_DUPLICATE'),
    (lambda r: r['predictions'].append(deepcopy(r['predictions'][0])), 'PREDICTION_INDEX_INVALID_OR_DUPLICATE'),
])
def test_corrupted_scene_or_prediction_identity_fails_closed(edit, reason):
    gt = scene(); record = row(gt, [prediction()]); edit(record)
    with pytest.raises(ValueError, match=reason):
        aggregate([record], [gt])


def test_unexplained_pose_exclusion_and_changed_protocol_are_rejected():
    gt = scene([plate(excluded=True)])
    gt['plates'][0]['pose_exclusion'] = None
    with pytest.raises(ValueError, match='GT_POSE_EXCLUSION_UNEXPLAINED'):
        aggregate([], [gt])
    protocol = deepcopy(METRIC_PROTOCOL); protocol['match_iou'] = .4
    with pytest.raises(ValueError, match='MT_METRIC_PROTOCOL_CHANGED'):
        aggregate([], [scene()], protocol)


def test_aggregate_domain_statistics_keep_scene_and_plate_denominators():
    day, night = scene(), scene([plate('night1'), plate('night2', 110)], sha='b'*64, domain='NIGHT')
    output = aggregate([row(night, [prediction(night['plates'][0])]), row(day, [])], [day, night])
    assert output['domains']['DAY']['box_fn'] == 1
    assert output['domains']['NIGHT']['box_recall'] == .5
    assert output['domains']['ALL']['box_recall'] == pytest.approx(1/3)
    assert output['domains']['ALL']['pose_coverage'] == pytest.approx(1/3)


def test_all_four_corners_have_their_own_pixels_and_diagonal_normalization():
    gt = scene()
    offsets = [(1, 0), (0, 2), (3, 0), (0, 4)]
    raw = [[point[0]+offset[0], point[1]+offset[1]] for point, offset in zip(gt['plates'][0]['quad'], offsets)]
    m = aggregate([row(gt, [prediction(keypoints=raw)])], [gt])['domains']['ALL']
    assert m['corner_error_px']['mean'] == 2.5
    for index, column in enumerate(m['per_corner']):
        assert column['index'] == index
        assert column['error_px']['n'] == 1
        assert column['error_px']['mean'] == index+1
        assert column['error_normalized']['mean'] == pytest.approx((index+1)/hypot(80,30))
