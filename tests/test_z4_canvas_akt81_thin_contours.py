"""AKT81: drawing one thin contour, no dark halo, no ROI dot grid.

Only the on-screen presentation changes, never original MT or GT data.
"""
from copy import deepcopy
from test_z4_canvas_akt70_selection_drag import setup, key_event


def _ops(canvas, tag, kind=None):
    return [op for op in canvas.ops
            if (kind is None or op[0] == kind)
            and tag in op[2].get('tags', ())]


def test_unselected_contours_are_single_and_thin():
    _mod, e, canvas, _h, _scene, _messages, _ = setup()
    e.overlay()
    mt = _ops(canvas, 'mt_original', 'polygon')
    gt = _ops(canvas, 'operator_quad', 'polygon')
    assert len(mt) == 1, mt
    assert len(gt) == 1, gt
    assert mt[0][2]['width'] == 1.5
    assert gt[0][2]['width'] == 1.5
    assert not any(op[2].get('stipple') for op in canvas.ops)
    roi = _ops(canvas, 'analyzed_roi', 'rect')
    assert len(roi) == 1
    assert roi[0][2]['fill'] == ''


def test_selected_gt_has_only_one_yellow_contour_and_handles():
    _mod, e, canvas, _h, _scene, _messages, _ = setup()
    e.select_mode(key_event(canvas))
    assert e.mode == 'edit' and e.selected
    assert not _ops(canvas, 'mt_original')
    gt = _ops(canvas, 'operator_quad', 'polygon')
    assert len(gt) == 1
    assert gt[0][2]['outline'] == '#FFD400'
    assert gt[0][2]['width'] == 2.0
    assert len(_ops(canvas, 'corner_handle', 'oval')) == 4


def test_nms_suppressed_raw_is_hidden_on_canvas_but_kept_in_data():
    _mod, e, canvas, _h, scene, _messages, _ = setup()
    p = deepcopy(scene['parents'][0]);p['parent_id'] = 'm2';p['detection_index'] = 1
    o = deepcopy(scene['objects'][0]);o['parent_id'] = 'm2';o['object_id'] = 'o2';o['post_nms_suppressed'] = True
    scene['parents'].append(p);scene['objects'].append(o)
    before = deepcopy(scene)
    e.overlay()
    assert len(_ops(canvas, 'mt_original', 'polygon')) == 1
    assert not _ops(canvas, 'raw_nms_suppressed')
    assert len(_ops(canvas, 'operator_quad', 'polygon')) == 1
    assert scene == before


def test_roi_toggle_does_not_change_scene():
    _mod, e, canvas, _h, scene, _messages, _ = setup()
    before = deepcopy(scene)
    e.show_roi_overlay = False
    e.overlay()
    assert not _ops(canvas, 'analyzed_roi')
    assert scene == before
