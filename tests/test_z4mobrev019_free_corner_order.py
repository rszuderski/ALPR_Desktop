"""V019: manual GT vertices may be clicked in any order, never MT keypoint reindexing."""
from itertools import permutations
from types import SimpleNamespace
import math
import pytest
from auto_annotation_tool.gui import z4_mobile_sample_review as mod
from test_z4mobrev018_missed_plate_gt import new_view, GT, ENTRY


def test_existing_polygon_validator_is_shared_not_duplicated():
    from auto_annotation_tool.rectification.polygon_validator import PolygonValidator
    assert PolygonValidator.order_points is not None
    assert mod.PolygonValidator is PolygonValidator


@pytest.mark.parametrize('order', list(permutations(range(4))))
def test_any_permutation_has_identical_canonical_gt(order):
    raw = [GT[i] for i in order]
    quad = mod._ordered_operator_gt_quad(raw, 512, 512)
    assert [[round(x,4), round(y,4)] for x,y in quad] == GT
    assert mod._ordered_operator_gt_quad(quad,512,512) == quad
    assert {tuple(p) for p in raw} == set(quad)


@pytest.mark.parametrize('order',[(0,2,1,3),(2,3,0,1),(3,1,0,2)])
def test_actual_mouse_path_reorders_and_saves_without_fake_detection(order):
    view,sess=new_view()
    view.canvas.winfo_width=lambda:512
    view.canvas.winfo_height=lambda:512
    view._pose_gt_cache_token=('attempt-5',ENTRY)
    view.pose_edit_var.set(True)
    for idx in order:
        x,y=GT[idx]
        assert view._pan_begin(SimpleNamespace(x=x,y=y))=='break'
        view._pan_end()
    decision=sess.review.attempt_annotations['attempt-5']
    assert decision['pose_gt_quad_mt_input_px']==GT
    assert view._pose_gt_points==list(map(tuple,GT))
    assert not view.pose_edit_var.get()
    assert sess.attempts['attempt-5']['mt_status']=='NO_DETECTION'
    assert 'is_plate' not in decision
    rows=mod._pose_geometry_rows(sess)
    assert rows[0]['metric_status']=='missed_detection_gt_saved'
    assert rows[0]['nmce_pct'] is None and rows[0]['polygon_iou'] is None


def test_detected_attempt_sorts_gt_but_never_reorders_prediction(monkeypatch):
    view,sess=new_view(status='VALID_QUAD',detection_count=1)
    sess.review.attempt_annotations['attempt-5']={'is_plate':True,'evaluable':True}
    source=[(x+1,y+1) for x,y in GT]
    sess.attempts['attempt-5']['mt_pose_corners_norm']=list(reversed(source))
    predicted=tuple(map(tuple,GT))
    monkeypatch.setattr(mod,'_linked_pose_quad_on_mt_input',lambda *_:predicted)
    view._pose_gt_points=[tuple(GT[i]) for i in (2,0,3,1)]
    view._pose_gt_drag_index=3
    view._finish_pose_gt_drag()
    assert sess.review.attempt_annotations['attempt-5']['pose_gt_quad_mt_input_px']==GT
    assert sess.attempts['attempt-5']['mt_pose_corners_norm']==list(reversed(source))
    metrics=mod.compare_mt_pose_to_gt(predicted, view._pose_gt_points,512,512)
    assert metrics['nmce_pct']==pytest.approx(0.0)
    assert metrics['polygon_iou_pct']==pytest.approx(100.0)


def test_existing_annotation_is_not_silently_migrated():
    view,sess=new_view()
    scrambled=[GT[i] for i in (0,2,1,3)]
    sess.review.attempt_annotations['attempt-5']={
        'pose_gt_quad_mt_input_px':scrambled,'pose_gt_schema':mod.POSE_GT_SCHEMA,
        'pose_gt_image_entry':ENTRY,'pose_gt_input_size_px':[512.0,512.0]}
    view._ensure_pose_gt_points('attempt-5')
    assert view._pose_gt_points==[]
    assert sess.review.attempt_annotations['attempt-5']['pose_gt_quad_mt_input_px']==scrambled


@pytest.mark.parametrize('raw',[
    [(150,160),(150,160),(265,183),(151,187)],
    [(150,160),(260,155),(265,183),(600,187)],
    [(150,160),(260,155),(265,183),(math.nan,187)],
    [(150,160),(260,155),(265,183),(math.inf,187)],
    [(10,10),(100,10),(45,15),(10,100)],
])
def test_invalid_user_quads_stay_unsaved_and_draft_is_rolled_back(raw):
    view,sess=new_view()
    view._pose_gt_points=list(raw)
    view._pose_gt_drag_index=3
    view._pose_gt_before_drag=list(raw[:3])
    view.pose_edit_var.set(True)
    view._finish_pose_gt_drag()
    assert 'attempt-5' not in sess.review.attempt_annotations
    assert view._pose_gt_points==list(raw[:3])
    assert view.pose_edit_var.get()
    assert 'Nie zapisano GT' in view.status_var.get()


def test_partial_draft_does_not_connect_arbitrary_clicks():
    view,_=new_view()
    view._pose_gt_cache_token=('attempt-5',ENTRY)
    view._pose_gt_points=[tuple(GT[i]) for i in (0,2,1)]
    view._draw_pose_gt_overlay('attempt-5',lambda x,y:(x,y))
    kinds=[v[0] for v in view.canvas.calls]
    assert kinds.count('oval')==3
    assert 'polygon' not in kinds and 'line' not in kinds


def test_four_clicks_preview_shows_canonical_polygon_and_labels():
    view,_=new_view()
    view._pose_gt_cache_token=('attempt-5',ENTRY)
    view._pose_gt_points=[tuple(GT[i]) for i in (0,2,1,3)]
    view.pose_edit_var.set(True)
    view._draw_pose_gt_overlay('attempt-5',lambda x,y:(x,y))
    polygon=[v for v in view.canvas.calls if v[0]=='polygon']
    assert len(polygon)==1
    assert polygon[0][1]==tuple(v for xy in GT for v in xy)
    labels=[v[2].get('text') for v in view.canvas.calls if v[0]=='text']
    assert labels==['TL','TR','BR','BL']


def test_strong_perspective_convex_quad_is_not_crossed():
    # Prior Z2 audit: legacy top/bottom sorting made a bow-tie here.
    points=[(0,0),(40,0),(50,20),(60,130)]
    order=[points[i] for i in (0,2,1,3)]
    ordered=mod._ordered_operator_gt_quad(order,512,512)
    assert set(ordered)==set(points)
    assert len(ordered)==4
    assert mod.validate_mt_quad(ordered,512,512)==ordered


def test_saved_legacy_gt_is_displayed_in_original_order_without_silent_reindexing():
    view,sess=new_view()
    view._pose_gt_cache_token=('attempt-5',ENTRY)
    backwards=list(reversed([tuple(p) for p in GT]))
    view._pose_gt_points=backwards
    view.pose_edit_var.set(False)
    view._draw_pose_gt_overlay('attempt-5',lambda x,y:(x,y))
    polygon=[v for v in view.canvas.calls if v[0]=='polygon']
    assert len(polygon)==1
    assert polygon[0][1] == tuple(v for xy in backwards for v in xy)
    assert view._pose_gt_points==backwards
    assert sess.review.attempt_annotations=={}


def test_drag_correction_updates_only_selected_gt_corner():
    view,sess=new_view()
    view._pose_gt_cache_token=('attempt-5',ENTRY)
    view._pose_gt_points=list(map(tuple,GT))
    view.canvas.winfo_width=lambda:512
    view.canvas.winfo_height=lambda:512
    view.pose_edit_var.set(True)
    assert view._pan_begin(SimpleNamespace(x=150,y=160))=='break'
    assert view._pan_move(SimpleNamespace(x=148,y=159))=='break'
    view._pan_end()
    corrected=[list(p) for p in GT]
    corrected[0]=[148.0,159.0]
    assert sess.review.attempt_annotations['attempt-5']['pose_gt_quad_mt_input_px']==corrected
    assert view._pose_gt_points==list(map(tuple,corrected))
    assert sess.attempts['attempt-5']['mt_status']=='NO_DETECTION'
