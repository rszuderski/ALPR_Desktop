"""Explicit operator resolution of Canvas objects; never infer physical identity.

Original MT parents, GT text and canonical scoring rules remain unchanged.
An absent source Pose cannot be reconstructed from an operator's reference.
"""
from __future__ import annotations

import copy
import math

from . import mobile_canvas_review as core

CONTRACT = core.DECISION_CONTRACT


def identity_evidence(session, gid, scene, target):
    """Source-bound final detection identity, independent of Pose availability."""
    ctx = core.context(session, gid)
    parents = [p for p in scene["parents"] if p["parent_id"] == target.get("parent_id")]
    if (not ctx or len(parents) != 1 or scene.get("_post_nms_pending")
            or target.get("post_nms_suppressed") or target.get("attempt_id") != parents[0]["attempt_id"]):
        raise ValueError("Nie da się ustalić źródła końcowego wskazania MT.")
    parent = parents[0]
    row = session.attempts[parent["attempt_id"]]
    status = str(row.get("mt_status") or "").upper()
    group = session.mt_invocations[gid]
    if (status not in core.DETECTION_STATUSES or str(row.get("mt_executed")).lower() != "true"
            or str(row.get("mt_invocation_id") or "") != gid
            or group.detection_count is None or group.detection_count != len(scene["parents"])):
        raise ValueError("Nie potwierdzono finalnego wskazania wykonanego MT.")
    from .mobile_mt_postdedup import _trace_index, _source_integer
    stat = session.path.stat()
    traces = _trace_index(str(session.path), session.bundle.source_archive_sha256, stat.st_size, stat.st_mtime_ns)
    key = tuple(_source_integer(row.get(k)) for k in ('source_sequence','source_timestamp_nanos','scene_generation'))
    matches = traces.get(key, [])
    if len(matches) > 1:
        raise ValueError("Niejednoznaczny ślad końcowej detekcji MT.")
    if matches:
        final = _source_integer(matches[0].get('counters', {}).get('plate_detections_after_dedup'))
        effective = len(scene['parents']) - sum(bool(o.get('post_nms_suppressed')) for o in scene['objects'])
        if final is not None and final != effective:
            raise ValueError("Wykaz MT nie odpowiada końcowym detekcjom po NMS.")
    pose = ("VALID" if valid_quad(scene, parent.get("quad")) else "INVALID"
            if status == "DETECTION_INVALID_QUAD" or str(row.get("mt_pose_status") or "").upper() in
                {"INVALID_QUAD", "DETECTION_INVALID_QUAD"} else "EVIDENCE_UNAVAILABLE")
    return dict(decision="SAME_PLATE", source_archive_sha256=scene["source_archive_sha256"],
        source_binding=scene["source_binding"], mt_invocation_id=gid, image_entry=scene["image_entry"],
        parent_id=parent["parent_id"], attempt_id=parent["attempt_id"], mt_status=status, pose_status=pose,
        source_pose_status=str(row.get("mt_pose_status") or "").upper())


def verify_decisions(session, gid, scene):
    """Validate only explicit v2 additions; never migrate historical reviews."""
    for obj in scene["objects"]:
        if "identity_decision" in obj:
            if scene.get("decision_contract") != CONTRACT or obj["identity_decision"] != identity_evidence(session, gid, scene, obj):
                raise ValueError("Decyzja tożsamości nie odpowiada źródłowej detekcji MT.")
        if "identity_location" in obj:
            if obj["identity_location"] != model_location(session, gid, scene, obj):
                raise ValueError("Położenie wskazania nie odpowiada zapisanej decyzji.")
        if "identity_unresolved" in obj:
            expected = unresolved_evidence(scene, obj)
            if obj["identity_unresolved"] != expected or obj.get("identity_decision"):
                raise ValueError("Niespójny zapis nierozstrzygniętego wskazania.")
    if scene.get("roi_presence") == "NO_PLATES" and core.counts(scene)["expected"]:
        raise ValueError("Potwierdzono brak tablic, ale pozostaje aktywna ramka GT.")


def verify_frame(session, gid):
    from .mobile_source_frame_roi import read_mobile_source_frame
    ctx = core.context(session, gid)
    if not ctx:
        raise ValueError("Brak wiarygodnej pełnej klatki i wykonanego MT dla tego ROI.")
    if any(str(r.get('mt_executed')).lower() != 'true' or
           str(r.get('mt_status') or '').upper() not in core.DETECTION_STATUSES | {'NO_DETECTION'} for r in ctx['records']):
        raise ValueError("Brak jednoznacznego wyniku wykonanego MT.")
    session.verify_source()
    image = read_mobile_source_frame(session.path, ctx["entry"])
    try:
        if tuple(image.size) != tuple(ctx["size"]):
            raise ValueError("Rozmiar klatki nie odpowiada źródłowemu ROI.")
    finally:
        image.close()


def valid_quad(scene, quad):
    try:
        core.validate_mt_quad(quad, *scene["image_size"])
        return True
    except (ValueError, TypeError, KeyError, OverflowError):
        return False


def model_location(session, gid, scene, target):
    """Final source location for inspection only. A rectangle is never Pose.

    ResearchAttemptBatch records plate_* after DetectionCoordinateMapper.toSource.
    Require the explicit source-frame/ROI and encoder coordinate contracts; old
    or ambiguous records fail closed instead of guessing a transform.
    """
    identity_evidence(session, gid, scene, target)
    parent = next(p for p in scene['parents'] if p['parent_id'] == target['parent_id'])
    common = dict(object_id=target['object_id'], attempt_id=parent['attempt_id'],
                  mt_invocation_id=gid, image_entry=scene['image_entry'])
    if valid_quad(scene, parent.get('quad')):
        return dict(common, kind='POSE', quad=copy.deepcopy(parent['quad']))
    row = session.attempts[parent['attempt_id']]
    from .mobile_mt_postdedup import _trace_index, _source_integer
    stat = session.path.stat()
    traces = _trace_index(str(session.path), session.bundle.source_archive_sha256, stat.st_size, stat.st_mtime_ns)
    key = tuple(_source_integer(row.get(k)) for k in ('source_sequence','source_timestamp_nanos','scene_generation'))
    matches = traces.get(key, [])
    effective = len(scene['parents']) - sum(bool(o.get('post_nms_suppressed')) for o in scene['objects'])
    if (any(v is None for v in key) or len(matches) != 1
            or _source_integer(matches[0].get('counters', {}).get('plate_detections_after_dedup')) != effective
            or any(_source_integer(matches[0].get(k)) != _source_integer(row.get(k))
                   for k in ('visual_epoch','camera_transform_generation'))):
        raise ValueError('Brak jednoznacznego dowodu końcowego wskazania w tej klatce po NMS.')
    try:
        values = [float(row[k]) for k in ('plate_left','plate_top','plate_right','plate_bottom',
                  'mt_pose_source_width_px','mt_pose_source_height_px')]
        left, top, right, bottom, width, height = values
        rl, rt, rr, rb = scene['roi']
        if (not all(math.isfinite(v) for v in values)
                or [width,height] != list(scene['image_size'])
                or row.get('mt_pose_coordinate_space') != 'source_frame_normalized_v1'
                or row.get('source_frame_coordinate_space') != 'mt_source_bitmap_xy'
                or row.get('source_frame_origin') != 'pipeline_bitmap_before_roi'
                or not (rl <= left < right <= rr and rt <= top < bottom <= rb)):
            raise ValueError()
    except (ValueError, KeyError, TypeError, OverflowError):
        raise ValueError('Brak wiarygodnego położenia wskazania modelu w tej klatce i obszarze.') from None
    return dict(common, kind='BBOX', bbox=[left,top,right,bottom])


def unresolved_evidence(scene, target):
    return dict(decision='CANNOT_ESTABLISH', source_archive_sha256=scene['source_archive_sha256'],
                source_binding=scene['source_binding'], mt_invocation_id=scene['mt_invocation_id'],
                object_id=target['object_id'], attempt_id=target['attempt_id'])


def mark_identity_unresolved(session, gid, target_id):
    """Record uncertainty, never turn it into TP, FP, FN or an excluded image."""
    s = core.scene(session, gid)
    target = core._object(s, target_id)
    if target not in missing_mt_geometry(s):
        raise ValueError('Wybierz nierozstrzygnięte wskazanie modelu.')
    target['identity_unresolved'] = unresolved_evidence(s, target)
    s['decision_contract'] = CONTRACT
    core._store(session, gid, s, 'operator_cannot_establish_identity', target_id)


def missing_mt_geometry(scene):
    """Unresolved final MT children whose immutable source Pose is absent."""
    if not scene:
        return []
    parents = {p["parent_id"]: p for p in scene["parents"]}
    return [o for o in scene["objects"] if o.get("parent_id") in parents
            and not o.get("error") and not o.get("post_nms_suppressed")
            and o.get("attempt_id") != scene.get("_post_nms_pending")
            and not valid_quad(scene, parents[o["parent_id"]].get("quad"))
            and not (o.get("identity_decision") and valid_quad(scene, o.get("quad")))]


def resolution_detail(scene):
    missing = missing_mt_geometry(scene)
    if not missing:
        return ""
    return (f"MT: {len(missing)} wskazanie bez ocenialnego Pose. "
            "Zaznacz ramkę ręczną i określ, czy dotyczy tej samej tablicy.")


def link_manual_to_mt(session, gid, manual_id, target_id):
    """One explicit same-plate correction, preserving both prior objects in audit.

    Final detection identity may be confirmed even when original Pose is absent.
    The reference never replaces immutable source corners; geometry stays N/A.
    """
    s = core.scene(session, gid)
    if not s or s.get("_post_nms_pending"):
        raise ValueError("Najpierw rozstrzygnij stan wykonania i RAW/NMS.")
    manual, target = core._object(s, manual_id), core._object(s, target_id)
    if (manual.get("kind") != "manual" or manual.get("parent_id") or manual.get("attempt_id")
            or manual.get("error") or manual.get("post_nms_suppressed")):
        raise ValueError("Wybierz aktywną ręczną ramkę referencyjną.")
    parents = [p for p in s["parents"] if p["parent_id"] == target.get("parent_id")]
    if (len(parents) != 1 or target.get("attempt_id") != parents[0]["attempt_id"]
            or target.get("error") or target.get("post_nms_suppressed")):
        raise ValueError("Wybierz aktywną końcową detekcję MT.")
    verify_frame(session, gid)
    evidence = identity_evidence(session, gid, s, target)
    location = model_location(session, gid, s, target)
    quad = core.validate_mt_quad(manual.get("quad"), *s["image_size"])
    before = dict(manual=copy.deepcopy(manual), target=copy.deepcopy(target))
    if 'candidate_stacks_v1' in s:
        from .mobile_candidate_stacks import associate,roles
        if roles(session,gid,s).get(target['attempt_id'])!='FINAL':raise ValueError('Wybierz końcową detekcję MT; RAW pozostaje diagnostyczny.')
        return associate(session,gid,target_id,manual_id,keep_inspected=True,manual_replacement=before)
    target.update(quad=[list(p) for p in quad], geometry_state="operator_corrected")
    target["identity_decision"] = evidence
    target["identity_location"] = location
    target.pop('identity_unresolved', None)
    from .mobile_canvas_physical_gt import link_manual
    link_manual(s,manual_id,target_id)
    # Do not transfer historical manual GT to another identity. It remains in
    # object_before, and undo restores the complete manual object.
    s["objects"].remove(manual)
    # An already inspected image keeps that fact; this explicit identity
    # decision may fill its last gap. Other content issues still block READY.
    s.update(exclusion_reason="", decision_contract=CONTRACT)
    s.pop("roi_presence", None)
    core._store(session, gid, s, "operator_link_manual_to_mt", target_id, object_before=before)


def keep_separate(session, gid, manual_id, target_id):
    s = core.scene(session, gid)
    manual, target = core._object(s, manual_id), core._object(s, target_id)
    if manual.get("parent_id") or manual.get("kind") != "manual":
        raise ValueError("Wybierz ręcznie dodaną tablicę.")
    identity_evidence(session, gid, s, target)
    manual["separate_from_mt"] = sorted(set(manual.get("separate_from_mt", ())) | {target_id})
    s.update(checked=False, exclusion_reason="", decision_contract=CONTRACT)
    core._store(session, gid, s, "operator_keep_plates_separate", manual_id)


def confirm_no_plates(session, gid):
    """Explicit absence in this ROI: zero detections -> TN; final detections -> FP."""
    if 'candidate_stacks_v1' in session.invocation_annotation(gid).get(core.KEY, {}):
        from .mobile_candidate_stacks import confirm_negative
        return confirm_negative(session,gid)
    verify_frame(session, gid)
    s = core.scene(session, gid)
    if s.get("_post_nms_pending"):
        raise ValueError("Najpierw rozstrzygnij RAW/NMS.")
    if any(not o.get("parent_id") and not o.get("error") for o in s["objects"]):
        raise ValueError("Pozostaje ręczna tablica. Usuń błędną ramkę albo pozostaw obecność tablic.")
    before = copy.deepcopy(s["objects"])
    for obj in s["objects"]:
        if obj.get("parent_id") and not obj.get("post_nms_suppressed"):
            identity_evidence(session, gid, s, obj)
            obj["error"] = True
            obj.pop('identity_unresolved', None)
    s.update(checked=True, exclusion_reason="", decision_contract=CONTRACT, roi_presence="NO_PLATES")
    if 'physical_gt_v1' in s:
        s.pop('physical_gt_v1')
    core._store(session, gid, s, "operator_confirm_no_plates_in_roi", object_before=before)


def exclude_missing_source_pose(session, gid):
    raise ValueError("Sam brak Pose nie wyłącza obrazu. Rozstrzygnij tożsamość wskazania MT.")
