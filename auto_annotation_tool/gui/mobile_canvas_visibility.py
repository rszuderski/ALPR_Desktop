"""Window-owned REF/INF presentation; never modify review or source geometry."""
from __future__ import annotations
import copy


def enabled(editor, name):
    variable = getattr(editor, name + '_visible', None)
    return bool(variable.get()) if variable is not None else True


def gt_only(editor):
    """Transient manual-entry view; never relabel historical provenance."""
    return bool(getattr(editor,'manual_gt_first',False) and not enabled(editor,'inf'))


def contours(editor):
    """All saved references and one proven source contour per FINAL.

    Physical reference IDs deduplicate parent/child aliases. Source object IDs
    keep distinct FINAL detections, including those sharing a reference.
    Selection only overrides the selected FINAL's Pose with its source BBOX.
    """
    from . import mobile_canvas_clean_view as clean, mobile_canvas_geometry_layers as layers
    from ..ranking.mobile_canvas_operator_view import saved_reference_ids
    from ..ranking.mobile_mt_geometry_layers import source_bbox
    from ..ranking.mobile_canvas_review import validate_mt_quad
    scene = editor._read()
    state = clean.source(editor)
    if not scene or not state or editor.mode == 'draw':
        return []
    result = []
    rows = clean.records(editor)
    isolated=bool(getattr(editor,'gt_focus_id','') or editor.drag)
    selected=clean.selected(editor) if isolated else None
    isolated_pid=selected['pid'] if selected else None
    from .mobile_canvas_reservation import preview
    from .mobile_canvas_approval_visuals import approved_ids
    approved=approved_ids(scene,state)
    if enabled(editor, 'ref') or getattr(editor,'gt_focus_id',''):
        saved = saved_reference_ids(scene, state)
        for pid, plate in state['plates'].items():
            if pid not in saved:
                continue
            if isolated and pid!=isolated_pid:continue
            if not enabled(editor,'ref') and not preview(editor,pid):continue
            row = next((r for r in rows if r['pid'] == pid), None)
            oid = row['oid'] if row else plate['origin_object_id']
            quad = plate['gt_quad']
            if editor.drag and editor.drag['oid'] == oid:
                quad = editor.drag['quad']
            result.append(dict(oid=oid,kind='GTR',quad=copy.deepcopy(quad),
                               plate_id=pid,source='saved_reference',bad=False,
                               approved=pid in approved and not(editor.drag and editor.drag['oid']==oid)))
        for row in rows:
            if not row.get('seed') or (isolated and row['oid']!=editor.selected):continue
            q=editor.drag['quad'] if editor.drag and editor.drag['oid']==row['oid'] else row['gt']
            result.append(dict(oid=row['oid'],kind='GTR',quad=copy.deepcopy(q),plate_id=None,
                source='unsaved_operator_seed',seed=True,bad=False,approved=False))
    if enabled(editor, 'inf') and not isolated:
        selection = layers.model(editor)
        for candidate in state['candidates']:
            if candidate['role'] != 'FINAL':
                continue
            bad = candidate.get('operator_bad') or candidate['association_status'] == 'UNASSOCIATED_NO_PLATE'
            if bad and not clean.show_fp(editor):
                continue
            oid = candidate['object_id']
            quad = candidate['mt_quad']
            if quad:
                try:
                    validate_mt_quad(quad, *scene['image_size'])
                except (ValueError, TypeError):
                    quad = None
            inspect_bbox = bool(selection and selection['object_id'] == oid and selection['active'] == 'BBOX')
            kind = 'POSE'; marker=None
            if not quad or inspect_bbox:
                bbox = source_bbox(editor.h.session, scene, candidate)
                if bbox:
                    l,t,r,b = bbox
                    quad = [(l,t),(r,t),(r,b),(l,b)]
                    kind = 'BBOX'
                else:
                    from .mobile_canvas_bbox_marker import describe,quad as marker_quad
                    descriptor=describe(editor.h.session,editor.h.current_invocation,scene,candidate)
                    if descriptor['marker']:
                        marker=descriptor;quad=marker_quad(marker);kind='BBOX'
            if quad:
                result.append(dict(oid=oid,kind=kind,quad=copy.deepcopy(quad),
                                   plate_id=candidate['plate_id'],source='MT_FINAL',
                                   attempt_id=candidate['attempt_id'],bad=bad,marker=marker))
    return result
