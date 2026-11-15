"""Canvas-first, source-ROI review with immutable MT parents (P2B2).

Coordinates are verified pre-ROI Bitmap pixels, never screen pixels. The
SHA-bound sidecar owns editable children and manual plates. Nothing writes
predictions, source images, or the legacy review. Metrics are ASSISTED, not
independent/blind ground truth. Completion follows the saved FINAL/GTR
decisions; FN is limited to the operator's registered physical references.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import uuid
from pathlib import Path

from .mobile_source_frame_roi import source_invocation_context, source_pose_quad
from .mobile_pose_metrics import validate_mt_quad, compare_mt_pose_to_gt
from ..rectification.polygon_validator import PolygonValidator

SCHEMA = "alpr.mobile.canvas_review.v1"
KEY = "canvas_review_v1"
PROVENANCE = "ASSISTED_CANVAS_SOURCE_ROI"
DECISION_CONTRACT = "mt_object_pose_v2"
MAX_OBJECTS = 128
DETECTION_STATUSES = frozenset({"VALID_QUAD", "DETECTION_INVALID_QUAD"})
GTR_BAD_MESSAGE = 'GTR to obrys referencyjny. Popraw lub usuń ramkę; BAD dotyczy tylko źródłowego MT FINAL. PPM na badge MT wybiera wskazanie modelu.'


def has_scene(session, gid):
    if gid in getattr(session, 'scene_frames', {}):
        return gid in session.review.scene_annotations
    return KEY in session.review.invocation_annotations.get(gid, {})


def context(session, gid, image_size=None):
    if gid in getattr(session, 'scene_frames', {}):
        from .mobile_scene_frame_review import context as scene_context
        return scene_context(session, gid, image_size)
    group = session.mt_invocations.get(gid)
    if not group or not group.executed or group.cancelled or group.execution_failed:
        return None
    return source_invocation_context(group.records, gid, session.entry_names, image_size)


def _prediction(session, aid):
    attempt = session.attempts.get(aid, {})
    samples = [session.samples[sid] for sid in session.sample_ids_by_attempt.get(aid, ())
               if sid in session.samples and not session.cancelled(session.samples[sid])]
    # An attempt is one inference event, not an OCR consensus or a subject.
    values = {str(s.get("raw_prediction", s.get("prediction", "")) or "") for s in samples}
    status = str(attempt.get("mz_status") or "").upper()
    if not status and samples:
        statuses = {str(s.get("mz_status") or "").upper() for s in samples}
        if len(statuses) == 1:
            status = next(iter(statuses))
    if status == "NOT_RUN":
        return "", "NOT_RUN", False
    if len(values) > 1:
        return str(attempt.get("raw_prediction", attempt.get("prediction", "")) or ""), "AMBIGUOUS_SAMPLES", True
    value = (next(iter(values)) if values else
             str(attempt.get("raw_prediction", attempt.get("prediction", "")) or ""))
    return value, status or "UNKNOWN", status not in ("", "UNKNOWN") or bool(value)


def _parents(session, ctx):
    result = []
    for row in ctx["records"]:
        if str(row.get("mt_status") or "").upper() not in DETECTION_STATUSES:
            continue
        aid = str(row.get("id") or row.get("attempt_id") or "")
        if not aid or aid not in session.attempts or any(p["attempt_id"] == aid for p in result):
            raise ValueError("Niespójny identyfikator wskazania MT")
        quad = source_pose_quad(row, ctx["size"])
        pred, status, ran = _prediction(session, aid)
        result.append(dict(parent_id="MT:" + aid, attempt_id=aid,
            detection_index=row.get("mt_detection_index"),
            quad=[list(p) for p in quad] if quad else None,
            mz_prediction=pred, mz_status=status, mz_ran=ran))
    indices = []
    for parent in result:
        try:
            index = int(parent["detection_index"])
        except (ValueError, TypeError, OverflowError):
            raise ValueError("Brak jednoznacznego indeksu detekcji MT") from None
        if index < 0 or index in indices:
            raise ValueError("Powielony lub nieprawidłowy indeks detekcji MT")
        indices.append(index)
    group = session.mt_invocations[ctx["invocation_id"]]
    declared = getattr(group, "detection_count", None)
    if declared is not None and int(declared) != len(result):
        raise ValueError("Niepełny wykaz wyników inferencji MT")
    return result


def _binding(session, ctx, parents):
    payload = dict(source_archive_sha256=session.review.source_archive_sha256,
        mt_invocation_id=ctx["invocation_id"], image_entry=ctx["entry"],
        image_size=list(ctx["size"]), roi=list(ctx["roi"]),
        source_sequence=ctx["source_sequence"], source_timestamp_nanos=ctx["source_timestamp_nanos"],
        parents=parents)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _post_nms_proof(session, gid, parents):
    """Accept attributable trace-backed suppression, never bbox overlap alone.

    This is not a physical-plate counter. An additional physical plate missed
    by the final MT is still entered manually by the operator.
    """
    from .mobile_mt_postdedup import canvas_post_nms_proof,final_ids,suppressed_ids
    proof = canvas_post_nms_proof(session, gid)
    aids = {p["attempt_id"] for p in parents}
    if not isinstance(proof,dict):return None
    final,raw=final_ids(proof),suppressed_ids(proof)
    if (proof.get('raw') != len(parents) or proof.get('after_dedup') != len(final)
            or proof.get('suppressed') != len(raw) or len(set(final+raw)) != len(final+raw)
            or set(final+raw)!=aids):
        return None
    return copy.deepcopy(proof)


def _pending_nms_review(s, proof):
    """Unmigrated reviews remain intact and must be reconciled explicitly."""
    if not proof:
        return
    from .mobile_mt_postdedup import suppressed_ids
    pending=[]
    for aid in suppressed_ids(proof):
        objects=[o for o in s['objects'] if o['attempt_id']==aid]
        if len(objects)==1 and not objects[0].get('post_nms_suppressed',False):
            obj=objects[0]
            linked=(s.get('candidate_stacks_v1') or {}).get('links',{}).get(obj['object_id'],{}).get('plate_id')
            untouched=(not linked and not obj['error'] and not obj['gt_text'] and not obj['text_confirmed']
                and not obj['text_exclusion_reason'] and obj['geometry_state']=='unconfirmed'
                and not obj.get('legacy_geometry_source') and not obj.get('legacy_text_source'))
            if proof.get('schema')=='alpr.postdedup.explicit_ids.v2' and untouched:
                # A transient source-role projection, never a sidecar migration.
                obj['post_nms_suppressed']=True;s['post_nms_evidence']=copy.deepcopy(proof)
            else:pending.append(aid)
    if pending:s['_post_nms_pending']=pending[0] if len(pending)==1 else pending


def _verify_nms_review(s, proof):
    flagged = [o for o in s["objects"] if o.get("post_nms_suppressed", False)]
    if not flagged:
        return
    from .mobile_mt_postdedup import suppressed_ids
    if (proof is None or {o['attempt_id'] for o in flagged} != set(suppressed_ids(proof))
            or s.get("post_nms_evidence") != proof):
        raise ValueError("Brak zgodnego dowodu NMS dla wyłączenia surowego MT")


def _new_object(oid, parent=None, quad=None):
    return dict(object_id=oid, parent_id=parent["parent_id"] if parent else None,
        attempt_id=parent["attempt_id"] if parent else "", kind="child" if parent else "manual",
        quad=copy.deepcopy(parent["quad"] if parent else quad),
        geometry_state="unconfirmed" if parent else "operator_drawn",
        error=False, gt_text="", text_exclusion_reason="", text_confirmed=False)


def _legacy_to_source(points, attempt, ctx):
    """Convert only a declared, proven MT-input transform. Never guess size."""
    try:
        scale, px, py = (float(attempt[k]) for k in ("input_scale", "input_pad_x", "input_pad_y"))
        roi = tuple(float(attempt[k]) for k in ("roi_left", "roi_top", "roi_right", "roi_bottom"))
        if roi != tuple(ctx["roi"]) or scale <= 0 or not all(math.isfinite(v) for v in (scale, px, py)):
            return None
        quad = [((float(x)-px)/scale + roi[0], (float(y)-py)/scale + roi[1]) for x,y in points]
        return [list(p) for p in validate_mt_quad(quad, *ctx["size"])]
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _seed(session, ctx, parents):
    objects = [_new_object("GT:"+p["attempt_id"], parent=p) for p in parents]
    old = session.invocation_annotation(ctx["invocation_id"]).get("multi_plate_v1", {})
    old_plates = old.get("plates", []) if isinstance(old, dict) else []
    warnings = []
    for obj, parent in zip(objects, parents):
        aid = parent["attempt_id"]
        annotation = session.review.attempt_annotations.get(aid, {})
        obj["error"] = annotation.get("is_plate") is False
        # Restore a real saved correction, not a visual "imprecise" label.
        correction = annotation.get("pose_mt_correction_v1")
        if correction:
            from .mobile_pose_correction import correction_for_attempt
            attempt = session.attempts[aid]
            try:
                scale,px,py = (float(attempt[k]) for k in ("input_scale","input_pad_x","input_pad_y"))
                l,t,_,_ = ctx["roi"]
                model = [((x-l)*scale+px,(y-t)*scale+py) for x,y in parent["quad"]]
                points = correction_for_attempt(annotation, attempt_id=aid,
                    source_sha=session.review.source_archive_sha256,
                    image_entry=attempt["mt_input_evidence_entry"],
                    size=(float(attempt["input_width"]),float(attempt["input_height"])), prediction=model)
                mapped = _legacy_to_source(points, attempt, ctx)
                if mapped is None:
                    raise ValueError("Brak zgodnej transformacji historycznej korekty")
                obj.update(quad=mapped,geometry_state="operator_corrected",
                    legacy_geometry_source="pose_mt_correction_v1")
            except (KeyError, TypeError, ValueError, OverflowError):
                warnings.append("Historyczna korekta nie ma zgodnej proweniencji: " + aid)
        matches = [p for p in old_plates if p.get("attempt_id") == aid]
        if len(matches) == 1:
            source = matches[0]
            obj.update(gt_text=str(source.get("gt_text") or ""),
                text_exclusion_reason=str(source.get("text_exclusion_reason") or ""))
            obj["text_confirmed"] = bool(obj["gt_text"] or obj["text_exclusion_reason"])
            obj["legacy_text_source"] = str(source.get("plate_id") or "")
        elif len(parents) == 1 and not old_plates:
            subject = session.review.subjects.get(session.attempts[aid].get("subject_key"), {})
            if subject.get("evaluable") is True and subject.get("ground_truth"):
                obj.update(gt_text=subject["ground_truth"], text_confirmed=True,
                    legacy_text_source="single_subject_confirmed")
    # Old unmatched rectangles can be displayed, but remain rectangles, NOT
    # four independently annotated corners. Their old IDs remain auditable.
    for plate in old_plates:
        if plate.get("attempt_id"):
            continue
        box = plate.get("bbox_mt_px")
        transformed = None
        if isinstance(box, (list, tuple)) and len(box) == 4:
            x1,y1,x2,y2 = box
            for record in ctx["records"]:
                if record.get("mt_input_evidence_entry") != plate.get("bbox_image_entry"):
                    continue
                transformed = _legacy_to_source([(x1,y1),(x2,y1),(x2,y2),(x1,y2)], record, ctx)
                if transformed:
                    break
        if transformed:
            obj = _new_object("LEGACY:"+str(plate["plate_id"]), quad=transformed)
            obj.update(geometry_state="legacy_bbox", gt_text=str(plate.get("gt_text") or ""),
                text_exclusion_reason=str(plate.get("text_exclusion_reason") or ""),
                text_confirmed=bool(plate.get("gt_text") or plate.get("text_exclusion_reason")),
                legacy_plate_id=plate["plate_id"])
            objects.append(obj)
        else:
            warnings.append("Historyczna tablica bez zgodnej geometrii: " + str(plate.get("plate_id")))
    result = dict(schema=SCHEMA, provenance=PROVENANCE,
        decision_contract=DECISION_CONTRACT,
        source_archive_sha256=session.review.source_archive_sha256,
        mt_invocation_id=ctx["invocation_id"], image_entry=ctx["entry"],
        image_size=list(ctx["size"]), roi=list(ctx["roi"]),
        source_binding=_binding(session, ctx, parents), parents=parents, objects=objects,
        checked=False, exclusion_reason="", migration_warnings=warnings, events=[])
    proof = _post_nms_proof(session, ctx["invocation_id"], parents)
    if proof:
        from .mobile_mt_postdedup import suppressed_ids
        raws=[o for o in objects if o['attempt_id'] in suppressed_ids(proof)]
        # A legacy GT, corrected geometry or error flag is operator evidence.
        # Never hide it automatically or change past decisions.
        untouched = all(not raw["error"] and not raw["gt_text"]
            and not raw["text_exclusion_reason"] and not raw["text_confirmed"]
            and raw["geometry_state"] == "unconfirmed"
            and not raw.get("legacy_geometry_source")
            and not raw.get("legacy_text_source") for raw in raws)
        if untouched:
            for raw in raws:raw['post_nms_suppressed']=True
            result["post_nms_evidence"] = proof
        else:
            _pending_nms_review(result, proof)
    else:
        from .mobile_mt_postdedup import unresolved_filter_reason
        reason=unresolved_filter_reason(session,ctx['invocation_id'])
        if reason:result['_post_nms_unknown']=reason
    return result


def _legacy_scene(session, gid, *, include_history=True):
    if gid in getattr(session, 'scene_frames', {}):
        from .mobile_scene_frame_review import scene as scene_frame
        return scene_frame(session, gid, include_history=include_history)
    ctx = context(session, gid)
    if ctx is None:
        raise ValueError("Brak zweryfikowanej pełnej klatki z ROI tego wywołania MT")
    parents = _parents(session, ctx)
    saved = session.invocation_annotation(gid).get(KEY)
    if saved is None:
        return _seed(session, ctx, parents)
    if not isinstance(saved, dict) or saved.get("schema") != SCHEMA:
        raise ValueError("Nieobsługiwany zapis weryfikacji canvas")
    if (saved.get("source_binding") != _binding(session, ctx, parents)
            or saved.get("parents") != parents or saved.get("mt_invocation_id") != gid
            or saved.get("source_archive_sha256") != session.review.source_archive_sha256
            or saved.get("image_entry") != ctx["entry"] or saved.get("image_size") != list(ctx["size"])
            or saved.get("roi") != list(ctx["roi"])):
        raise ValueError("Weryfikacja canvas nie odpowiada źródłowej inferencji/ROI")
    _validate_objects(saved)
    from .mobile_canvas_resolution import verify_decisions
    # Legacy child fields are retained evidence when a shared contract exists;
    # current ROI decisions are validated on its effective view below.
    decision_source = dict(saved) if 'candidate_stacks_v1' in saved else saved
    if decision_source is not saved:decision_source.pop('roi_presence', None)
    verify_decisions(session, gid, decision_source)
    proof = _post_nms_proof(session, gid, parents)
    _verify_nms_review(saved, proof)
    from .mobile_canvas_physical_gt import validate as validate_physical
    validate_physical(session, gid, saved)
    # Views and measurements need current state, not every undo snapshot.
    # Keep the default complete for editing/history callers and external tools.
    source=saved if include_history else {k:v for k,v in saved.items() if k!='candidate_stack_history_v1'}
    result = copy.deepcopy(source)
    if proof and proof.get('schema')=='alpr.postdedup.explicit_ids.v2':
        result.pop('_post_nms_unknown',None);result.pop('_post_nms_pending',None)
        from .mobile_canvas_partial_review import project_proven_source_roles
        project_proven_source_roles(session,gid,result)
    _pending_nms_review(result, proof)
    if not proof:
        from .mobile_mt_postdedup import unresolved_filter_reason
        reason=unresolved_filter_reason(session,gid)
        if reason:result['_post_nms_unknown']=reason
    return result


def scene(session, gid, *, include_history=True):
    if gid in getattr(session, 'scene_frames', {}):
        return _legacy_scene(session, gid, include_history=include_history)
    result = _legacy_scene(session, gid, include_history=include_history)
    from .mobile_canvas_partial_review import validate as validate_partial
    validate_partial(session, gid, result)
    if 'candidate_stacks_v1' in result:
        from .mobile_candidate_stacks import project
        effective = project(session, gid, result)
        _validate_objects(effective)
        from .mobile_canvas_resolution import verify_decisions
        from .mobile_canvas_physical_gt import validate
        verify_decisions(session, gid, effective)
        validate(session, gid, effective)
        return effective
    return result


def _validate_objects(s):
    objects = s.get("objects")
    if type(s.get("checked")) is not bool:
        raise ValueError("Nieprawidłowe potwierdzenie obrazu")
    reason = s.get("exclusion_reason", "")
    if not isinstance(reason, str) or len(reason) > 240 or (reason and not s["checked"]):
        raise ValueError("Niespójny stan sprawdzenia / wyłączenia obrazu")
    if not isinstance(objects, list) or len(objects) > MAX_OBJECTS:
        raise ValueError("Nieprawidłowy wykaz ramek canvas")
    ids = [o.get("object_id") for o in objects if isinstance(o, dict)]
    if len(ids) != len(objects) or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Powielone lub brakujące identyfikatory ramek")
    parents = {p["parent_id"]:p for p in s["parents"]}
    seen = set()
    for obj in objects:
        parent = obj.get("parent_id")
        if parent:
            if parent not in parents or parent in seen or obj.get("attempt_id") != parents[parent]["attempt_id"] or obj.get("kind") != "child":
                raise ValueError("Korekta musi mieć dokładnie jednego, niezmiennego rodzica MT")
            seen.add(parent)
        elif obj.get("kind") != "manual" or obj.get("attempt_id"):
            raise ValueError("Ręczna ramka nie może udawać wyniku MT")
        if type(obj.get("error")) is not bool or type(obj.get("text_confirmed")) is not bool:
            raise ValueError("Nieprawidłowa flaga ramki")
        suppressed = obj.get("post_nms_suppressed", False)
        if type(suppressed) is not bool or (suppressed and not parent):
            raise ValueError("Nieprawidłowy stan RAW odrzuconego przez NMS")
        if not all(isinstance(obj.get(k), str) for k in ("gt_text","text_exclusion_reason","geometry_state")):
            raise ValueError("Nieprawidłowe GT ramki")
        if len(obj["gt_text"]) > 256 or len(obj["text_exclusion_reason"]) > 240 or (obj["gt_text"] and obj["text_exclusion_reason"]):
            raise ValueError("Niespójne GT numeru i przyczyna nieczytelności")
        if obj.get("quad") is not None:
            quad = validate_mt_quad(obj["quad"], *s["image_size"])
            if not parent:
                l,t,r,b=s["roi"]
                if any(not (l <= x <= r and t <= y <= b) for x,y in quad):
                    raise ValueError("Ręczna tablica poza analizowanym ROI")
    if seen != set(parents):
        raise ValueError("Brak ramki potomnej dla wskazania MT")
    suppressed_objects = [o for o in objects if o.get("post_nms_suppressed", False)]
    if suppressed_objects:
        evidence = s.get("post_nms_evidence")
        from .mobile_mt_postdedup import suppressed_ids
        if (not isinstance(evidence,dict)
                or set(suppressed_ids(evidence)) != {o['attempt_id'] for o in suppressed_objects}
                or evidence.get('raw') != len(parents)
                or evidence.get('after_dedup') != len(parents)-len(suppressed_objects)
                or evidence.get('suppressed') != len(suppressed_objects)):
            raise ValueError("Wyłączenie RAW wymaga jednoznacznego dowodu końcowych wyników NMS")


def _object(s, oid):
    hits = [o for o in s["objects"] if o["object_id"] == oid]
    if len(hits) != 1:
        raise ValueError("Nieznana ramka canvas")
    return hits[0]


def require_source_final(session, gid, oid):
    """BAD targets the actual source child, never a related physical GTR."""
    from .mobile_candidate_stacks import roles
    raw = _legacy_scene(session, gid)
    obj = _object(raw, oid)
    parents = [p for p in raw['parents'] if p['parent_id'] == obj.get('parent_id')
               and p['attempt_id'] == obj.get('attempt_id')]
    if (obj.get('kind') != 'child' or oid != 'GT:'+str(obj.get('attempt_id', ''))
            or len(parents) != 1):
        raise ValueError(GTR_BAD_MESSAGE)
    if roles(session, gid, raw).get(obj['attempt_id']) != 'FINAL':
        raise ValueError('BAD dotyczy tylko źródłowego MT FINAL. RAW lub nieustalona rola pozostają informacją źródłową.')
    return obj


def manual_bad_objects(s):
    """Read-only legacy conflict audit, including originals behind projection."""
    objects = {o['object_id']: o for o in s['objects']}
    objects.update({o['object_id']: o for o in (s.get('_stack_legacy') or {}).get('objects', [])})
    return [o for o in objects.values() if o.get('error')
            and (o.get('kind') == 'manual' or not o.get('parent_id'))]


def ordered_quad(points, size, roi=None):
    if len(points) != 4:
        raise ValueError("Wskaż cztery narożniki")
    # Same polygon ordering utility as Z2; never apply it to MT parents.
    raw = [(float(x),float(y)) for x,y in points]
    ordered = PolygonValidator.order_points(raw)
    quad = validate_mt_quad(ordered, *size)
    if roi is not None:
        l,t,r,b = roi
        if any(not (l <= x <= r and t <= y <= b) for x,y in quad):
            raise ValueError("Brakującą tablicę zaznacz wyłącznie wewnątrz analizowanego ROI")
    return [list(p) for p in quad]


def counts(s):
    # RAW rejected by proven Android NMS is neither a physical GT nor a final FP.
    # It remains in `parents` and `objects` as immutable/auditable evidence.
    scored = [o for o in s["objects"] if not o.get("post_nms_suppressed", False)]
    active = [o for o in scored if not o["error"]]
    result = dict(expected=len(active), mt=len(s["parents"]),
        linked=sum(bool(o["parent_id"]) for o in active),
        missed=sum(not o["parent_id"] for o in active),
        false_positive=sum(bool(o["parent_id"]) and o["error"] for o in scored),
        operator_errors=sum(not o["parent_id"] and o["error"] for o in scored),
        raw_suppressed=len(s["objects"]) - len(scored))
    from .mobile_canvas_physical_gt import tally
    physical=tally(s)
    if physical:result.update(physical)
    elif not s['checked'] and not s.get('events'):
        # Legacy prediction copies remain evidence, never provisional physical GT.
        from .mobile_candidate_stacks import _legacy
        state=_legacy(s)
        matched={state['links'][o['object_id']]['plate_id'] for o in active if o.get('parent_id')
                 and state['links'].get(o['object_id'],{}).get('plate_id') and not o.get('duplicate_of')}
        result.update(expected=len(state['plates']),linked=len(matched),missed=len(state['plates'])-len(matched))
    return result


def _store(session, gid, s, action, oid="", *, object_before=None, event_details=None, expected_sidecar_sha256=None):
    if gid in getattr(session, 'scene_frames', {}):
        from .mobile_scene_frame_review import store
        return store(session, gid, s, action, oid, expected_sidecar_sha256=expected_sidecar_sha256)
    # Validate BAD transitions before touching flags, events, history or disk.
    old = _legacy_scene(session, gid, include_history=False)
    from .mobile_gt_link_contract import validate_transition
    merge=(gid,oid) if action=='operator_same_physical_plate_duplicate_final' else None
    if action=='operator_same_physical_plate_across_invocations':merge=(gid,oid,'cross')
    validate_transition(old,s,history=action in {'undo_stack_history','redo_stack_history'},merge_oid=oid if merge else None,
                        cross_identity=bool(merge and len(merge)>2))
    old_objects = {o['object_id']: o for o in old['objects']}
    proposed = {o['object_id']: o for o in s['objects']}
    proposed.update({o['object_id']: o for o in (s.get('_stack_legacy') or {}).get('objects', [])
                     if not o.get('parent_id')})
    for key, obj in proposed.items():
        if bool(obj.get('error')) != bool(old_objects.get(key, {}).get('error')):
            if obj.get('kind') == 'manual' or not obj.get('parent_id'):
                raise ValueError(GTR_BAD_MESSAGE)
            require_source_final(session, gid, key)
    if s.get('checked') and manual_bad_objects(s):
        raise ValueError('Historyczne BAD na GTR wymaga jawnej korekty lub usunięcia referencji; nie jest oceną MT.')
    diagnostic_only=bool(s.pop('_stack_diag_only',False))
    from .mobile_canvas_physical_gt import sync_references, validate
    from .mobile_candidate_stacks import adopt_canvas_write
    if action not in {'undo_stack_history','redo_stack_history'}:
        s=adopt_canvas_write(session,gid,s)
        from .mobile_canvas_partial_review import KEY as PARTIAL_KEY, sync_write, sync_gtr_review
        sync_gtr_review(session,gid,s,old,action,oid)
        if PARTIAL_KEY in s:
            from .mobile_candidate_stacks import project, pack
            sync_write(session, gid, s, action, oid, previous=old)
            s = project(session, gid, pack(s))
        from .mobile_frame_review_state import from_scene
        if from_scene(s)['metric_complete'] and not s.get('exclusion_reason'):
            from .mobile_candidate_stacks import record_scene_identity,project,pack
            record_scene_identity(session,gid,s)
            s=project(session,gid,pack(s))
            if 'frame_completion_v1' in s:s['frame_completion_v1']['state']=from_scene(s)['state']
    from .mobile_canvas_partial_review import validate as validate_partial
    validate_partial(session, gid, s)
    sync_references(s)
    validate(session,gid,s)
    _validate_objects(s)
    from .mobile_canvas_resolution import verify_decisions
    verify_decisions(session, gid, s)
    # Source/trace binding is checked on *every* write, also on undo/redo.
    _verify_nms_review(s, _post_nms_proof(session, gid, s["parents"]))
    s.pop("_post_nms_pending", None)
    # This action replaces exactly one already-detached Canvas scene. Other
    # annotations are immutable inputs; copying their audit histories on every
    # Space press is unnecessary. _save_change changes only header fields.
    changed = copy.copy(session.review)
    changed.invocation_annotations = dict(session.review.invocation_annotations)
    if changed.review_mode == 'blinded_gt_v1':
        if changed.review_mode_locked:
            raise ValueError('Zapis asystowanej zmiany zablokowany: zachowano historyczną, zaślepioną ocenę. Użyj odrębnej kopii z jawnym trybem asystowanym.')
        changed.review_mode = 'assisted'
        changed.review_mode_origin = changed.review_mode_origin or 'blinded_gt_v1'
    inv = dict(changed.invocation_annotations.get(gid, {}))
    from .mobile_human_review import _utc_now_iso
    s["events"].append(dict(action=action, object_id=oid, saved_at=_utc_now_iso(),
        object_after=copy.deepcopy(next((o for o in s["objects"] if o["object_id"] == oid), None)),
        review_revision_before=session.review.review_revision))
    if object_before is not None:
        s["events"][-1]["object_before"] = copy.deepcopy(object_before)
    if event_details is not None:
        s['events'][-1].update(copy.deepcopy(event_details))
    s["expected_plate_count"] = counts(s)["expected"]
    from .mobile_candidate_stacks import pack, record_history
    inv[KEY] = pack(s)
    record_history(session, gid, inv[KEY], action, diagnostic_only=diagnostic_only)
    # Old forms/GT are retained untouched for history. Canvas has its own
    # verification path; never feed this derived count into legacy denominators.
    changed.invocation_annotations[gid] = inv
    changed.review_status = session.review.review_status if diagnostic_only and s['checked'] else "IN_PROGRESS"
    save_options=dict(_gtr_history_restore=action in {'undo_stack_history','redo_stack_history'},_gtr_merge=merge)
    if expected_sidecar_sha256 is not None:save_options['_expected_sidecar_sha256']=expected_sidecar_sha256
    session._save_change(changed,**save_options)


def _edit(session, gid, oid, action, modify):
    s = scene(session, gid)
    if s.get('_post_nms_unknown') and _object(s,oid).get('parent_id'):
        raise ValueError('Nie ustalono roli RAW/FINAL; oznacz fizyczną tablicę przez D, bez zgadywania powiązania z MT.')
    if _object(s, oid).get("post_nms_suppressed", False):
        raise ValueError("RAW odrzucone przez NMS nie jest osobną tablicą GT; użyj D dla pominiętej tablicy")
    modify(_object(s, oid), s)
    s.update(checked=False, exclusion_reason="")
    s.pop("roi_presence", None)
    _store(session, gid, s, action, oid)


def add_plate(session, gid, points, *, partial_review=False):
    if gid in getattr(session, 'scene_frames', {}):
        from .mobile_scene_frame_review import add_plate as add_scene_plate
        return add_scene_plate(session, gid, points)
    s = scene(session, gid)
    if partial_review:
        from .mobile_candidate_stacks import adopt_canvas_write
        from .mobile_canvas_partial_review import enable
        s = adopt_canvas_write(session, gid, s)
        enable(session, gid, s)
    if len(s["objects"]) >= MAX_OBJECTS:
        raise ValueError("Osiągnięto limit ramek canvas")
    quad = ordered_quad(points, s["image_size"], s["roi"])
    from .mobile_manual_frame_guard import require_free_manual_area
    require_free_manual_area(s, quad, session=session, gid=gid)
    oid = "MANUAL:" + uuid.uuid4().hex
    s["objects"].append(_new_object(oid, quad=quad))
    s.update(checked=False, exclusion_reason="")
    s.pop("roi_presence", None)
    _store(session, gid, s, "add_manual_plate", oid)
    return oid


def correct_plate(session, gid, oid, points):
    if gid in getattr(session, 'scene_frames', {}):
        from .mobile_scene_frame_review import edit
        return edit(session, gid, oid, quad=points)
    if 'candidate_stacks_v1' in session.invocation_annotation(gid).get(KEY, {}):
        from .mobile_candidate_stacks import edit_gt
        return edit_gt(session, gid, oid, quad=points)
    def edit(obj, s):
        if obj.get('duplicate_of'):
            raise ValueError('Końcowy duplikat nie ma osobnej referencji; edytuj właściwą tablicę.')
        # Dragging an indexed corner keeps its correspondence with its MT
        # parent; sorting on every drag would silently swap corner identities.
        quad = validate_mt_quad(points, *s["image_size"])
        if not obj["parent_id"]:
            l,t,r,b=s["roi"]
            if any(not (l <= x <= r and t <= y <= b) for x,y in quad):
                raise ValueError("Ręczna tablica poza analizowanym ROI")
        obj["quad"] = [list(p) for p in quad]
        obj["geometry_state"] = "operator_corrected" if obj["parent_id"] else "operator_drawn"
        if obj["parent_id"] and not next(p for p in s["parents"] if p["parent_id"] == obj["parent_id"])["quad"]:
            from .mobile_canvas_resolution import identity_evidence, model_location, verify_frame
            verify_frame(session, gid)
            obj['identity_location'] = model_location(session, gid, s, obj)
            obj["identity_decision"] = identity_evidence(session, gid, s, obj)
            obj.pop('identity_unresolved', None)
            s["decision_contract"] = DECISION_CONTRACT
    _edit(session, gid, oid, "correct_child_geometry", edit)



def remove_operator_geometry(session, gid, oid):
    if gid in getattr(session, 'scene_frames', {}):
        from .mobile_scene_frame_review import remove
        return remove(session, gid, oid)
    """Remove operator geometry without adopting any MT geometry as GT.

    MT evidence is immutable. A linked identity and its GTZ remain without corners.
    Deleted manual objects live only in the audit events, not in metrics or view.
    """
    s = scene(session, gid)
    obj = _object(s, oid)
    if obj.get("post_nms_suppressed", False):
        raise ValueError("Odrzuconego RAW nie można edytować ani usuwać; pozostaje dowodem MT")
    if obj.get('parent_id'):
        from .mobile_candidate_stacks import remove_linked_geometry
        return remove_linked_geometry(session,gid,oid)
    before = copy.deepcopy(obj)
    s["objects"].remove(obj)
    from .mobile_canvas_physical_gt import remove_reference
    remove_reference(s, oid)
    action = "remove_manual_plate"
    s.update(checked=False, exclusion_reason="")
    _store(session, gid, s, action, oid, object_before=before)
    return action


def accept_post_nms_raw_only(session, gid):
    """Explicit reconciliation of an old Canvas scene with attributable RAW.

    History, GT, geometry, and error flags of the suppressed raw attempt remain
    stored. The old `checked` decision is invalidated for reinspection; no
    physical plate count is inferred from the Android trace alone.
    """
    s = scene(session, gid)
    proof = _post_nms_proof(session, gid, s["parents"])
    from .mobile_mt_postdedup import suppressed_ids
    raw_ids=suppressed_ids(proof);pending=s.get('_post_nms_pending')
    if not proof or set(pending if isinstance(pending,list) else [pending]) != set(raw_ids):
        raise ValueError("Brak nierozstrzygniętej ramki RAW z jednoznacznym dowodem NMS")
    raw_objects=[o for o in s['objects'] if o['attempt_id'] in raw_ids]
    before=copy.deepcopy(raw_objects[0] if len(raw_objects)==1 else raw_objects)
    for obj in raw_objects:
        obj['post_nms_suppressed']=True
        if s.get('physical_gt_v1') and obj['object_id'] in s['physical_gt_v1']['associations']:
            registry=s['physical_gt_v1'];gt=registry['associations'].pop(obj['object_id'])
            plate=next(p for p in registry['plates'] if p['gt_id']==gt)
            if plate['reference_object_id']==obj['object_id']:
                manual=_new_object('MANUAL:'+uuid.uuid4().hex,quad=plate['quad'])
                manual.update(gt_text=obj['gt_text'],text_exclusion_reason=obj['text_exclusion_reason'],text_confirmed=obj['text_confirmed'])
                s['objects'].append(manual);plate['reference_object_id']=manual['object_id']
                registry['associations'][manual['object_id']]=gt
    s["post_nms_evidence"] = proof
    s.update(checked=False, exclusion_reason="")
    _store(session, gid, s, "operator_accept_trace_proven_nms_raw_only", raw_objects[0]['object_id'], object_before=before)
    return raw_objects[0]['object_id'] if len(raw_objects)==1 else [o['object_id'] for o in raw_objects]


def set_scene_review_state(session, gid, *, checked, exclusion_reason="", expected_sidecar_sha256=None):
    """Inspected is distinct from complete/evaluable; validation remains mandatory.

    A user can inspect an image with incomplete GT. Such an image stays pending
    in group_issues and outside metric denominators until the data are complete.
    An exclusion must also be inspected. Unchecking it does not erase any GT.
    """
    if type(checked) is not bool:
        raise ValueError("Stan sprawdzenia musi być logiczny")
    reason = " ".join(str(exclusion_reason or "").split())
    if len(reason) > 240 or (reason and not checked):
        raise ValueError("Obraz nie do oceny musi być również sprawdzony; opis do 240 znaków")
    s = scene(session, gid)
    from .mobile_candidate_stacks import adopt_canvas_write
    s=adopt_canvas_write(session,gid,s)
    if checked and not reason:
        from .mobile_canvas_partial_review import progress
        if progress(s)['pending_gtr_ids']:raise ValueError('Najpierw zatwierdź nowe lub zmienione GTR.')
    s.update(checked=checked, exclusion_reason=reason)
    if checked and not reason:_inspect_established_references(s)
    _explicit_inspection_presence(s)
    _store(session, gid, s, "set_roi_review_checkboxes", expected_sidecar_sha256=expected_sidecar_sha256)


def merge_review_cells(rows):
    """Merge per-image/per-plate presence states, without turning gaps into success."""
    rows = list(rows)
    def cell(values):
        values = [v for v in values if v != "—"]
        if not values:
            return "—"
        if all(v in ("✓", "×") for v in values):
            return "✓"
        return "○" if all(v == "○" for v in values) else "◐"
    return tuple(cell(row[i] for row in rows) for i in range(5))


def object_review_cells(s, obj):
    """ROI inspected, frame decision, text GT, corners, completion."""
    from .mobile_frame_review_state import from_scene
    frame=from_scene(s)
    inspected = "✓" if frame['complete'] else "○"
    if obj.get("post_nms_suppressed", False):
        return (inspected, "—", "—", "—", inspected)
    if s.get("exclusion_reason") and s["checked"]:
        return ("✓", "—", "—", "—", "✓")
    if obj["error"]:
        return (inspected, "×", "—", "—", inspected)
    parent = next((p for p in s["parents"] if p["parent_id"] == obj["parent_id"]), None)
    corners = bool(obj.get("quad")) and obj["geometry_state"] != "unconfirmed"
    text = ("✓" if obj["text_confirmed"] and obj["gt_text"] else
            "—" if obj["text_confirmed"] and obj["text_exclusion_reason"] else
            "○" if parent and parent["mz_ran"] else "—")
    ready = frame['metric_complete'] and bool(obj.get("quad"))
    return (inspected, "✓" if corners else "○", text,
            "✓" if corners else "○", "✓" if ready else "○")


def scene_review_cells(s):
    if s.get("exclusion_reason") and s["checked"]:
        return ("✓", "—", "—", "—", "✓")
    rows = [object_review_cells(s, obj) for obj in s["objects"]]
    cells = list(merge_review_cells(rows))
    from .mobile_frame_review_state import from_scene,SOURCE_NA,NO_REFERENCE
    frame=from_scene(s)
    if frame['state'] in {SOURCE_NA,NO_REFERENCE}:return ('—',)*5
    cells[0] = "✓" if frame['complete'] else "○"
    if not rows:
        cells[1] = cells[0]  # Explicitly inspected zero, never an inferred zero.
    cells[-1] = "✓" if frame['complete'] else "○"
    return tuple(cells)


def toggle_error(session, gid, oid):
    require_source_final(session, gid, oid)
    from .mobile_candidate_stacks import toggle_unassociated
    return toggle_unassociated(session, gid, oid)


def set_text(session, gid, oid, text, reason=""):
    if gid in getattr(session, 'scene_frames', {}):
        from .mobile_scene_frame_review import edit
        return edit(session, gid, oid, text=text, reason=reason)
    if 'candidate_stacks_v1' in session.invocation_annotation(gid).get(KEY, {}):
        from .mobile_candidate_stacks import edit_gt
        return edit_gt(session, gid, oid, text=text, reason=reason)
    from .mobile_human_review import normalize_registration
    gt = normalize_registration(text)
    reason = " ".join(str(reason or "").split())
    if len(gt) > 256 or len(reason) > 240 or (gt and reason):
        raise ValueError("Wpisz numer GT albo przyczynę nieczytelności, nie oba naraz")
    def edit(obj, s):
        obj.update(gt_text=gt, text_exclusion_reason=reason, text_confirmed=bool(gt or reason))
        if (gt or reason) and obj.get('quad') and not obj['error']:
            obj['canvas_reference_decision']=True
    _edit(session, gid, oid, "edit_plate_gt", edit)


def _content_issues(s):
    if s.get('scope') == 'SCENE_FRAME':
        from .mobile_scene_frame_review import progress
        return [] if progress(s)['complete'] else ['Potwierdź obecność tablic albo świadomy brak tablic na pełnej scenie']
    issues = []
    issues.extend('Historyczne BAD na GTR '+o['object_id']+': wymagana jawna korekta lub usunięcie referencji; to nie FP modelu.'
                  for o in manual_bad_objects(s))
    if s.get("_post_nms_pending"):
        issues.append("Uzgodnij dodatkowy surowy MT odrzucony przez NMS przed oceną ROI")
    if s.get('_post_nms_unknown'):
        issues.append(s['_post_nms_unknown'])
    parents = {p["parent_id"]:p for p in s["parents"]}
    for obj in s["objects"]:
        if obj.get("post_nms_suppressed", False) or obj["error"] or obj.get('duplicate_of'):
            continue
        oid = obj["object_id"]
        if ('candidate_stacks_v1' in s and obj.get('parent_id') and
                not s['candidate_stacks_v1']['links'].get(oid,{}).get('plate_id')):
            issues.append('Rozstrzygnij wskazanie na Canvas: ' + oid)
            continue
        if not obj.get("quad"):
            issues.append(("Nie można ustalić tożsamości wskazania: " if obj.get('identity_unresolved')
                           else "Brak czterech rogów: ") + oid)
        parent = parents.get(obj["parent_id"])
        if parent and not parent["quad"] and obj.get("quad") and not obj.get("identity_decision"):
            issues.append("Potwierdź tożsamość wskazania MT bez Pose: " + oid)
        if parent and parent["mz_ran"] and not (obj["text_confirmed"] and (obj["gt_text"] or obj["text_exclusion_reason"])):
            issues.append(f"Brak GT numeru albo przyczyny nieczytelności: {oid}")
        if parent and parent["mz_status"] == "AMBIGUOUS_SAMPLES":
            issues.append(f"Sprzeczne odczyty MZ przypisane do jednej detekcji: {oid}")
    if s.get("migration_warnings"):
        issues.append("Historyczne dane wymagają sprawdzenia na obrazie")
    from .mobile_canvas_partial_review import KEY as PARTIAL_KEY, progress as partial_progress
    if PARTIAL_KEY in s:
        pending = partial_progress(s)
        issues.extend('Potwierdź obecność tablicy w ramce: ' + oid for oid in pending['pending_ids'])
        issues.extend('Zatwierdź nową lub zmienioną GTR: ' + pid for pid in pending['pending_gtr_ids'])
        if pending['source_unknown']:
            issues.append('Nieustalone pochodzenie końcowych ramek MT.')
    from .mobile_canvas_physical_gt import issues as physical_issues
    issues.extend(physical_issues(s))
    return issues


def _inspect_established_references(s):
    """Inspection validates only already established GT, never new candidates."""
    state=s.get('candidate_stacks_v1')
    if state is None:return
    for pid,p in state['plates'].items():
        if p['confirmed'] and p['gt_quad'] and p['geometry_state']=='unconfirmed':
            p['geometry_state']='operator_confirmed_unchanged'
            for obj in s['objects']:
                if state['links'].get(obj['object_id'],{}).get('plate_id')==pid:
                    obj['geometry_state']='operator_confirmed_unchanged'


def _explicit_inspection_presence(s):
    """Inspection can establish a negative only after every candidate is resolved."""
    if not s['checked'] or s.get('exclusion_reason'):return
    state=s.get('candidate_stacks_v1')
    from .mobile_canvas_operator_view import eligible_physical_plate_ids
    if (state is not None and not eligible_physical_plate_ids(s,state) and not s.get('_post_nms_unknown') and not s.get('_post_nms_pending')
            and all(o.get('post_nms_suppressed') or o['error'] for o in s['objects'])):
        s.update(roi_presence='NO_PLATES',decision_contract=DECISION_CONTRACT)


def check_scene(session, gid, *, expected_sidecar_sha256=None):
    s = scene(session, gid)
    from .mobile_candidate_stacks import adopt_canvas_write
    s=adopt_canvas_write(session,gid,s)
    s.update(checked=True, exclusion_reason="")
    _inspect_established_references(s)
    _explicit_inspection_presence(s)
    issues = _content_issues(s)
    if issues:
        raise ValueError(issues[0])
    _store(session, gid, s, "confirm_complete_roi_review", expected_sidecar_sha256=expected_sidecar_sha256)


def exclude_scene(session, gid, reason):
    reason = " ".join(str(reason or "").split())
    if not reason or len(reason) > 240:
        raise ValueError("Podaj przyczynę wyłączenia obrazu (maksymalnie 240 znaków)")
    s = scene(session, gid)
    s.update(checked=True, exclusion_reason=reason)
    _store(session, gid, s, "exclude_roi_review")


def group_issues(session, gid):
    try:
        s = scene(session, gid, include_history=False)
    except (ValueError, TypeError, KeyError) as e:
        return [f"Canvas {gid}: {e}"]
    if s.get("checked") is True and str(s.get("exclusion_reason") or "").strip():
        return []
    issues = _content_issues(s)
    from .mobile_frame_review_state import from_scene,PENDING
    if from_scene(s)['state']==PENDING and not issues:issues.append('Rozstrzygnij pozostałe ramki MT/GTR.')
    return [f"Canvas {gid}: {i}" for i in issues]


def approval_scene_issues(s):
    """Form approval permits explicit BBOX identity without optional Pose GT.

    Quantitative qualification stays in _content_issues/group_issues. Removed
    operator geometry, unresolved identity and missing OCR GT still block.
    """
    if s.get('checked') and s.get('exclusion_reason'):return []
    from .mobile_frame_review_state import from_scene,SOURCE_NA,NO_REFERENCE
    if from_scene(s)['state'] in {SOURCE_NA,NO_REFERENCE}:return []
    from .mobile_canvas_partial_review import progress,PLATE
    decisions=progress(s)['object_decisions'];parents={p['parent_id']:p for p in s['parents']}
    optional=set()
    for obj in s['objects']:
        parent=parents.get(obj.get('parent_id'))
        if (parent and not parent['quad'] and obj.get('identity_decision') and not obj.get('quad')
                and obj['geometry_state']!='operator_removed' and decisions.get(obj['object_id'])==PLATE):
            optional.add('Brak czterech rogów: '+obj['object_id'])
    return [i for i in _content_issues(s) if i not in optional]


def approval_group_issues(session,gid):
    try:s=scene(session,gid,include_history=False)
    except (ValueError,TypeError,KeyError):return []  # Technical N/A; not an image-confirmation task.
    issues=approval_scene_issues(s)
    return [f'Canvas {gid}: {i}' for i in issues]


def subject_issues(session, subject_key):
    ids = {session.invocation_by_attempt.get(aid) for aid in session.subjects.get(subject_key, {}).get("attempts", ())}
    return [i for gid in ids if gid and has_scene(session, gid) for i in group_issues(session, gid)]


def summary(session):
    from .mobile_human_review import align_plate_text
    result = dict(schema=SCHEMA, provenance=PROVENANCE,
        independent_gt=False, session_id=getattr(session, "session_id", None),
        source_archive_sha256=session.review.source_archive_sha256,
        review_revision=session.review.review_revision, scene_count=0, checked_scenes=0,
        inspected_scenes=0, incomplete_checked_scenes=0,
        pending_scenes=0, excluded_scenes=0, expected_plates=0, linked_plates=0,
        missed_plates=0, false_positive_mt=0, operator_errors=0, raw_suppressed_mt=0, gt_characters=0,
        text_errors=0, evaluated_reads=0, exact_reads=0, geometry_n=0,
        polygon_iou_mean_pct=None, nmce_mean_pct=None, cer=None,
        note="Ocena asystowana, powiązania operatora; nie niezależny test ślepy ani automatyczny matching po progu IoU.",
        scenes=[], rows=[])
    geom = []
    for gid in session.mt_invocations:
        if not has_scene(session, gid):
            continue
        result["scene_count"] += 1
        try:
            s = scene(session, gid, include_history=False)
        except (ValueError, TypeError, KeyError) as e:
            result["pending_scenes"] += 1
            result["scenes"].append(dict(mt_invocation_id=gid, status="INVALID", reason=str(e)))
            continue
        issues = group_issues(session, gid)
        excluded = bool(s.get("exclusion_reason")) and s.get("checked") is True
        from .mobile_frame_review_state import from_scene
        frame=from_scene(s)
        evaluable = frame['metric_complete'] and not issues and not excluded
        result["inspected_scenes"] += int(frame['complete'])
        result["incomplete_checked_scenes"] += int(s.get("checked") is True and bool(issues) and not excluded)
        result["excluded_scenes" if excluded else "checked_scenes" if evaluable else "pending_scenes"] += 1
        tally = counts(s)
        result["scenes"].append(dict(mt_invocation_id=gid, image_entry=s["image_entry"],
            status="EXCLUDED" if excluded else "CHECKED" if evaluable else "PENDING",
            roi=s["roi"], counts=tally, issues=issues, exclusion_reason=s.get("exclusion_reason", "")))
        if not s['checked'] and not s.get('physical_gt_v1'):
            result['scenes'][-1]['physical_gt_status']='UNCONFIRMED'
            result['scenes'][-1]['physical_gt_count']=None
        if s.get("decision_contract") == DECISION_CONTRACT:
            result["decision_contract"] = DECISION_CONTRACT
            for key in ("negative_frames", "true_negative_frames", "negative_frames_with_fp", "pose_invalid_matches", "pose_evidence_unavailable_matches"):
                result.setdefault(key, 0)
            result["scenes"][-1]["roi_presence"] = s.get("roi_presence", "UNSPECIFIED")
            if evaluable and s.get("roi_presence") == "NO_PLATES":
                result["negative_frames"] += 1
                key = "negative_frames_with_fp" if tally["false_positive"] else "true_negative_frames"
                result[key] += 1
                result["scenes"][-1]["frame_outcome"] = "NEGATIVE_WITH_FP" if tally["false_positive"] else "TN"
        if evaluable:
            result["raw_suppressed_mt"] += tally["raw_suppressed"]
            for dst,src in (("expected_plates","expected"),("linked_plates","linked"),
                    ("missed_plates","missed"),("false_positive_mt","false_positive"),("operator_errors","operator_errors")):
                result[dst] += tally[src]
        parents = {p["parent_id"]:p for p in s["parents"]}
        for obj in s["objects"]:
            parent = parents.get(obj["parent_id"])
            raw_only = obj.get("post_nms_suppressed", False)
            duplicate = bool(obj.get('duplicate_of'))
            row = dict(mt_invocation_id=gid, image_entry=s["image_entry"],
                object_id=obj["object_id"], parent_id=obj["parent_id"], attempt_id=obj["attempt_id"],
                object_kind=obj["kind"], error_flag=obj["error"] or duplicate, included=evaluable and not raw_only,
                post_nms_suppressed_raw=raw_only,
                outcome="RAW_SUPPRESSED_POST_NMS" if raw_only else
                    (("FALSE_POSITIVE_MT" if parent else "OPERATOR_ERROR") if obj["error"] or duplicate else ("LINKED_MT" if parent else "MISSED_MT")),
                original_mt_quad=parent["quad"] if parent else None, operator_quad=obj["quad"],
                geometry_state=obj["geometry_state"], gt_text=obj["gt_text"],
                text_exclusion_reason=obj["text_exclusion_reason"],
                mz_prediction=parent["mz_prediction"] if parent else "",
                mz_status=parent["mz_status"] if parent else "NOT_RUN",
                polygon_iou_pct=None, nmce_pct=None, cer=None, provenance=PROVENANCE)
            physical = s.get('physical_gt_v1')
            if physical:
                row['physical_gt_id'] = physical['associations'].get(obj['object_id'])
                row['physical_scene_id'] = physical['scope']
                if duplicate:row['false_positive_subtype']='duplicate_final'
            if ('candidate_stacks_v1' in s and parent and not raw_only and not obj['error'] and
                    not s['candidate_stacks_v1']['links'].get(obj['object_id'],{}).get('plate_id')):
                row['outcome']='UNRESOLVED_MT_IDENTITY'
            if obj.get("identity_decision"):
                row["identity_decision"] = copy.deepcopy(obj["identity_decision"])
                row["pose_status"] = obj["identity_decision"]["pose_status"]
                row["geometry_availability"] = "AVAILABLE" if parent and parent["quad"] else "NOT_AVAILABLE"
                if evaluable and not obj["error"] and not duplicate and row["pose_status"] != "VALID":
                    result["pose_invalid_matches" if row["pose_status"] == "INVALID" else "pose_evidence_unavailable_matches"] += 1
            if obj.get('identity_unresolved'):
                row['identity_unresolved'] = copy.deepcopy(obj['identity_unresolved'])
                row['outcome'] = 'UNRESOLVED_MT_IDENTITY'
            elif not parent and not obj['error'] and any(o.get('identity_unresolved') for o in s['objects']):
                row['outcome'] = 'UNRESOLVED_MT_IDENTITY'
            if obj.get('identity_location'):
                row['identity_location'] = copy.deepcopy(obj['identity_location'])
            if evaluable and not raw_only and not obj["error"] and not duplicate:
                if parent and parent["quad"] and obj["quad"] and obj["geometry_state"] != "unconfirmed":
                    m = compare_mt_pose_to_gt(parent["quad"], obj["quad"], *s["image_size"])
                    row.update(polygon_iou_pct=m["polygon_iou_pct"], nmce_pct=m["nmce_pct"])
                    geom.append(m)
                if parent and parent["mz_ran"] and obj["text_confirmed"] and obj["gt_text"]:
                    a = align_plate_text(obj["gt_text"], parent["mz_prediction"])
                    row["cer"] = a.cer
                    result["gt_characters"] += len(a.ground_truth)
                    result["text_errors"] += a.edit_distance
                    result["evaluated_reads"] += 1
                    result["exact_reads"] += int(a.exact_match)
            result["rows"].append(row)
        if s.get('physical_gt_v1'):
            physical=s['physical_gt_v1']
            result['scenes'][-1]['physical_gt']=copy.deepcopy(physical)
            current=[r for r in result['rows'] if r['mt_invocation_id']==gid]
            covered={r.get('physical_gt_id') for r in current if r['outcome'] in {'LINKED_MT','MISSED_MT','UNRESOLVED_MT_IDENTITY'} and not r.get('post_nms_suppressed_raw')}
            for plate in physical['plates']:
                if plate['gt_id'] in covered:continue
                result['rows'].append(dict(mt_invocation_id=gid,image_entry=s['image_entry'],
                    object_id=plate['gt_id']+':FN',parent_id=None,attempt_id='',object_kind='physical_reference',
                    error_flag=False,included=evaluable,post_nms_suppressed_raw=False,outcome='MISSED_MT',
                    original_mt_quad=None,operator_quad=copy.deepcopy(plate['quad']),geometry_state='operator_drawn',
                    gt_text='',text_exclusion_reason='',mz_prediction='',mz_status='NOT_RUN',
                    polygon_iou_pct=None,nmce_pct=None,cer=None,provenance=PROVENANCE,
                    physical_gt_id=plate['gt_id'],physical_scene_id=physical['scope']))
    if result["gt_characters"]:
        result["cer"] = result["text_errors"] / result["gt_characters"]
    if geom:
        result.update(geometry_n=len(geom),
            polygon_iou_mean_pct=sum(m["polygon_iou_pct"] for m in geom)/len(geom),
            nmce_mean_pct=sum(m["nmce_pct"] for m in geom)/len(geom))
    result["mt_precision_operator"] = (result["linked_plates"] / (result["linked_plates"] + result["false_positive_mt"])
        if result["linked_plates"] + result["false_positive_mt"] else None)
    result["mt_recall_operator"] = (result["linked_plates"] / result["expected_plates"] if result["expected_plates"] else None)
    if any(s.get('physical_gt') for s in result['scenes']):
        result['physical_gt_contract']='alpr.mobile.physical_gt.v1'
        result['physical_plates']=[dict(physical_scene_id=scope_id,gt_id=gt_id,invocations=sorted(gids))
            for (scope_id,gt_id),gids in _physical_inventory(result['scenes']).items()]
        unresolved_identity=any(p.get('scene_identity')=='UNRESOLVED_ACROSS_INVOCATIONS'
            for s in result['scenes'] for p in s.get('physical_gt',{}).get('plates',()))
        result['physical_gt_scene_n']=None if unresolved_identity else len(result['physical_plates'])
        result['physical_gt_known_scene_n']=len(result['physical_plates'])
        result['physical_gt_legacy_invocations']=sum(s['status']=='CHECKED' and not s.get('physical_gt') for s in result['scenes'])
        result['physical_gt_scope']='explicit_ids_only'
        result['expected_plates_unit']='physical_gt_per_invocation'
    if any(row.get('identity_location') or row.get('identity_unresolved') for row in result['rows']):
        from .mobile_canvas_presentation import build_inventory
        inventory = build_inventory(session)
        result['operator_progress'] = {k:inventory[k] for k in ('total','inspected','assessed','incomplete','decisions','evidence_gaps')}
    return result


def _physical_inventory(scenes):
    result={}
    for s in scenes:
        if s['status']!='CHECKED':continue
        reg=s.get('physical_gt')
        if not reg:continue
        for plate in reg['plates']:
            result.setdefault((reg['scope'],plate['gt_id']),set()).add(s['mt_invocation_id'])
    return result


def export(session, directory):
    from .mobile_human_review import _atomic_write
    from .mobile_gt_link_contract import validate_review
    validate_review(session,session.review)
    data = summary(session)
    directory = Path(directory)
    names = ["canvas_review_summary.json", "canvas_review_plates.csv", "canvas_review_report.md"]
    paths = [directory/n for n in names]
    _atomic_write(paths[0], json.dumps(data, ensure_ascii=False, indent=2))
    output = io.StringIO(newline="")
    rows = data["rows"]
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["mt_invocation_id", "object_id", "provenance"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    _atomic_write(paths[1], "\ufeff"+output.getvalue())
    def val(n):
        return "nie dotyczy" if n is None else f"{n:.4f}" if isinstance(n,float) else str(n)
    text = "# Weryfikacja ALPR na canvasie\n\n" + data["note"] + "\n\n"
    text += f"Źródło SHA-256: `{data['source_archive_sha256']}`. Rewizja: {data['review_revision']}.\n\n"
    if data.get('operator_progress'):
        from .mobile_canvas_presentation import progress_caption
        text += progress_caption(data['operator_progress']) + '\n\n'
    if data.get('physical_gt_contract'):
        text += (f"Fizyczne tablice z jawnymi identyfikatorami na scenach: {data['physical_gt_scene_n'] if data['physical_gt_scene_n'] is not None else 'nieustalone'}. "
                 f"Starsze wywołania bez nowego rejestru: {data['physical_gt_legacy_invocations']} (nie scalano ich automatycznie). "
                 f"Obserwacje GT w wywołaniach: {data['expected_plates']}. "
                 "Jedna GT może mieć osobny wynik w każdym wywołaniu; dopasowanie wewnątrz wywołania jest 1:1.\n\n")
    text += "| Miara | Wynik |\n|---|---:|\n"
    for label,key in (("Obrazy przejrzane (w tym nie do oceny)","inspected_scenes"),
        ("Sprawdzone i ocenialne","checked_scenes"),("Sprawdzone z brakami danych","incomplete_checked_scenes"),
        ("Obrazy oczekujące na komplet decyzji","pending_scenes"),
        ("Obrazy wyłączone","excluded_scenes"),("Tablice oczekiwane w ROI","expected_plates"),
        ("Powiązane wskazania MT","linked_plates"),("Pominięte tablice","missed_plates"),
        ("Fałszywe wskazania MT","false_positive_mt"),("RAW odrzucone przez NMS","raw_suppressed_mt"),
        ("Błędy ramek operatora","operator_errors"),
        ("CER znaków","cer"),("IoU polygonów, średnia [%]","polygon_iou_mean_pct"),
        ("NMCE narożników, średnia [%]","nmce_mean_pct")):
        text += f"| {label} | {val(data[key])} |\n"
    if data.get("decision_contract") == DECISION_CONTRACT:
        for label, key in (("Negatywne klatki/ROI (jawne GT=0)", "negative_frames"),
                          ("TN na jednostce klatka/ROI", "true_negative_frames"),
                          ("Negatywne klatki z fałszywym alarmem", "negative_frames_with_fp"),
                          ("Powiązane wskazania z błędem Pose", "pose_invalid_matches"),
                          ("Powiązane wskazania bez dowodu Pose", "pose_evidence_unavailable_matches")):
            text += f"| {label} | {val(data[key])} |\n"
        text += "\nTożsamość wskazania MT i jakość Pose są odrębne. NOT_AVAILABLE nie jest IoU=0 ani NMCE=100.\n"
    text += "\nOryginał MT jest porównywany z potwierdzoną ramką potomną. Korekta nie jest nową tablicą. "
    text += "RAW odrzucone na podstawie zweryfikowanego śladu NMS pozostaje dowodem, nie jest drugim GT ani FP. "
    text += "Ramka ręczna bez rodzica oznacza pominięcie; jej odrzucenie nie jest fałszywą detekcją MT. "
    text += "Brak odczytu MZ/NOT_RUN i wyłączony numer GT nie są zastępowane sztuczną predykcją. "
    text += "Niepotwierdzone obrazy nie wchodzą do mianowników. Szczegóły i obie geometrie: canvas_review_plates.csv.\n"
    _atomic_write(paths[2], text)
    if data.get('physical_gt_contract'):
        physical_csv=io.StringIO(newline='')
        writer=csv.DictWriter(physical_csv,fieldnames=['physical_scene_id','gt_id','invocations'])
        writer.writeheader()
        for plate in data['physical_plates']:writer.writerow(dict(plate,invocations=json.dumps(plate['invocations'])))
        _atomic_write(directory/'canvas_physical_plates.csv','\ufeff'+physical_csv.getvalue())
    if data.get("decision_contract") == DECISION_CONTRACT:
        frames = io.StringIO(newline="")
        columns = ["source_archive_sha256", "review_revision", "mt_invocation_id", "image_entry", "status",
                   "roi_presence", "frame_outcome", "expected", "linked", "missed", "false_positive", "exclusion_reason"]
        writer = csv.DictWriter(frames, fieldnames=columns); writer.writeheader()
        for scene_row in data["scenes"]:
            entry = {k:scene_row.get(k, "") for k in columns}
            entry.update(source_archive_sha256=data["source_archive_sha256"], review_revision=data["review_revision"])
            entry.update({k:scene_row.get("counts", {}).get(k, "") for k in ("expected", "linked", "missed", "false_positive")})
            writer.writerow(entry)
        path = directory / "canvas_review_frames.csv"
        _atomic_write(path, "\ufeff" + frames.getvalue()); paths.append(path)
    if data.get('physical_gt_contract'):
        paths.append(directory/'canvas_physical_plates.csv')
    return paths
