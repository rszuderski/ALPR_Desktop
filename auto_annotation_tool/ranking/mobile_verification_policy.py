# Z4MOBREV032_FIX02_CURRENT_SHA_CANVAS_20261023
"""Z4MOBREV022 — human MT verification policy, separate from Pose GT geometry.

Only operator decisions count as verification. The optional geometric measurement
requires independent GT and is deliberately not a form-completeness requirement.
This module never modifies inference data, model predictions, or source archives.
"""
from __future__ import annotations

from collections import Counter

VISUAL_CHOICES = (
    ("not_plate", "Nie tablica"),
    ("imprecise", "Niedokładny obrys"),
    ("accurate", "Dobry obrys"),
    ("uncertain", "Nie można ustalić"),
)
DETECTION_STATUSES = frozenset({"VALID_QUAD", "DETECTION_INVALID_QUAD"})


def selected_visual_state(annotation: dict | None) -> str:
    """Read saved decisions (including the earlier pose_quality vocabulary)."""
    a = annotation or {}
    if a.get("is_plate") is False:
        return "not_plate"
    if a.get("is_plate") is True:
        q = str(a.get("pose_quality") or "").lower()
        if q in {"accurate", "imprecise"}:
            return q
        if q == "incorrect":
            return "imprecise"  # legacy subjective judgment, NOT a measured IoU
        return ""
    if a.get("plate_visibility") == "uncertain" or a.get("pose_quality") == "uncertain":
        return "uncertain"
    return ""


def visual_decision(state: str) -> dict:
    """One atomic operator choice per predicted MT region."""
    if state == "not_plate":
        return {"plate_visibility": "invisible", "is_plate": False, "evaluable": False,
                "pose_quality": None}
    if state in ("accurate", "imprecise"):
        return {"plate_visibility": "visible", "is_plate": True, "evaluable": True,
                "pose_quality": state}
    if state == "uncertain":
        return {"plate_visibility": "uncertain", "is_plate": None,
                "evaluable": False, "pose_quality": "uncertain"}
    raise ValueError("Nieznany wybór weryfikacji wizualnej MT")


def _guided_correction_issue(session, record: dict) -> str | None:
    """A newly requested correction remains pending until a provenance-pinned quad exists.

    Applies equally to one-plate and multi-plate model invocations. Historical
    imprecise choices (before v026) are not retroactively made incomplete.
    """
    aid = str(record.get("id") or "")
    annotation = session.review.attempt_annotations.get(aid, {})
    if not (annotation.get("pose_correction_required_v1") is True and
            selected_visual_state(annotation) == "imprecise"):
        return None
    from .mobile_pose_correction import correction_for_attempt
    try:
        # The original model geometry, not its bbox or edited display polygon.
        from ..gui.z4_mobile_sample_review import _linked_pose_quad_on_mt_input
        original = _linked_pose_quad_on_mt_input(session, record)
        if original is None:
            raise ValueError("Brak źródłowych czterech rogów MT")
        correction = correction_for_attempt(annotation, attempt_id=aid,
            source_sha=session.review.source_archive_sha256,
            image_entry=record["mt_input_evidence_entry"],
            size=(float(record["input_width"]), float(record["input_height"])),
            prediction=original)
        if correction is None:
            raise ValueError("Brak zatwierdzonej korekty obrysu")
        return None
    except (ValueError, TypeError, KeyError):
        return f"Niezakończona korekta obrysu MT: {aid}"


def _attempt_status(attempt: dict) -> str:
    return str(attempt.get("mt_status") or "").upper()


def _active(session, group) -> bool:
    return bool(group and group.executed and not group.cancelled and not group.execution_failed)


def _subject_requires_text_gt(session, subject: dict) -> bool:
    """Require an explicit GT decision for non-cancelled MZ sample records.

    Legacy crop indexes do not always carry ``mz_status``. Missing/unknown
    status must not silently waive GT confirmation for those records; they
    can already contribute to the legacy read metrics. An empty prediction
    is likewise not evidence that MZ was NOT_RUN. This is a conservative
    verification gate, not a claim that an unknown inference did execute.
    Explicit NOT_RUN and cancelled records remain outside this requirement.
    """
    for sample_id in subject.get("samples", ()):
        sample = session.samples.get(sample_id)
        if sample is None or session.cancelled(sample):
            continue
        attempt = session.attempts.get(str(sample.get("attempt_id") or ""), {})
        from .mobile_canvas_review import has_scene,context
        gid=session.invocation_by_attempt.get(str(sample.get("attempt_id") or ""), "")
        if has_scene(session,gid) or context(session,gid) is not None:
            continue  # Canvas owns this plate-number decision, never subject-level GT.
        mz_status = str(attempt.get("mz_status") or sample.get("mz_status") or "").strip().upper()
        if mz_status != "NOT_RUN":
            return True
    return False


def subject_verification_issues(session, subject_key: str) -> list[str]:
    from .mobile_mt_postdedup import eligible_suppressed_id
    """Required actions for an observation; no mandatory manual Pose polygon."""
    subject = session.subjects.get(subject_key, {})
    attempt_ids = subject.get("attempts", ())
    group_ids = list(dict.fromkeys(session.invocation_by_attempt[aid] for aid in attempt_ids
                                   if aid in session.invocation_by_attempt))
    issues: list[str] = []
    for group_id in group_ids:
        group = session.mt_invocations.get(group_id)
        if not _active(session, group):
            continue
        from .mobile_canvas_review import has_scene,context,approval_group_issues as canvas_group_issues,approval_scene_issues,scene
        if has_scene(session, group_id) or context(session,group_id) is not None or any(r.get('source_frame_entry') for r in group.records):
            issues.extend(canvas_group_issues(session, group_id) if has_scene(session,group_id)
                          else _frame_issues(session,group_id))
            continue
        decision = session.invocation_annotation(group_id)
        count = decision.get("visible_plate_count")
        if count not in {"none", "one", "multiple", "uncertain"}:
            issues.append(f"Niezweryfikowana liczba tablic: {group_id}")
            continue
        if count == "multiple":
            from .mobile_multi_plate_verification import group_issues
            issues.extend(group_issues(session, group_id))
            continue  # multi: GT tekstowe jest PER TABLICA, nie per subject
        if count == "uncertain" or decision.get("evaluable") is False:
            continue  # explicit operator abstention, not a false model result
        # An empty reference list is also missing evidence, not a waiver.
        if not any(entry in session.entry_names for entry in group.evidence_entries):
            issues.append(f"Brak dowodu wejścia MT; oznacz wywołanie jako nie do oceny: {group_id}")
            continue
        discarded = eligible_suppressed_id(session, group_id)
        for record in group.records:
            aid = str(record.get("id") or "")
            if discarded and aid == discarded:
                # Telemetry confirms pre-NMS diagnostic-only candidate.
                continue
            if aid not in attempt_ids:
                continue
            if _attempt_status(record) not in DETECTION_STATUSES or session.cancelled(record):
                continue
            annotation = session.review.attempt_annotations.get(aid, {})
            if not selected_visual_state(annotation):
                issues.append(f"Brak weryfikacji wskazania MT: {aid}")
            guided_issue = _guided_correction_issue(session, record)
            if guided_issue:
                issues.append(guided_issue)
    # Legacy MZ samples still require GT or a reasoned exclusion even when
    # the source did not export mz_status. Do not auto-confirm a saved draft.
    mz_requires_gt = _subject_requires_text_gt(session, subject)
    from .mobile_multi_plate_verification import is_multi
    exclusively_multi = bool(group_ids) and all(is_multi(session, gid) for gid in group_ids)
    all_no_plate = bool(group_ids) and all(
        session.invocation_annotation(gid).get("visible_plate_count") == "none" for gid in group_ids)
    # False-positive OCR crop cannot have a genuine plate-number GT.
    if mz_requires_gt and not exclusively_multi and not all_no_plate:
        decision = session.review.subjects.get(subject_key, {})
        if decision.get("evaluable") is False:
            if not str(decision.get("note") or "").strip():
                issues.append(f"Brak uzasadnienia wyłączenia GT numeru: {subject_key}")
        else:
            gt = str(decision.get("ground_truth") or "").strip()
            if not gt or decision.get("evaluable") is not True:
                issues.append(f"Brak GT numeru albo wyłączenia odczytu: {subject_key}")
            elif not session.predictions_visible(subject_key):
                issues.append(f"GT numeru wymaga jawnego potwierdzenia: {subject_key}")
    # No MZ invokes: an MT miss can be verified without fabricating text GT.
    # Pose GT remains an explicitly optional *quantitative* follow-up.
    return issues


def _frame_issues(session,gid):
    from .mobile_scene_frame_review import mt_review_issues
    current=mt_review_issues(session,gid)
    if current is not None:return current
    from .mobile_frame_review_state import for_session,SOURCE_NA,NO_REFERENCE
    from .mobile_canvas_review import scene,approval_scene_issues
    state=for_session(session,gid)
    if state['state'] in {SOURCE_NA,NO_REFERENCE}:return []
    return approval_scene_issues(scene(session,gid,include_history=False))


def verification_issues(session) -> list[str]:
    if not session.subjects and not getattr(session, 'scene_frames', {}):
        return ["Archiwum nie zawiera obserwacji do weryfikacji"]
    from .mobile_scene_frame_review import issues
    result: list[str] = issues(session)
    for key in session.subjects:
        result.extend(subject_verification_issues(session, key))
    return list(dict.fromkeys(result))


def verification_progress(session) -> dict:
    if not session:
        return {"total": 0, "verified": 0, "excluded": 0, "issues": (), "status": "NOT_STARTED"}
    total = len(session.subjects)
    by_subject=[subject_verification_issues(session,key) for key in session.subjects]
    issues=list(dict.fromkeys(issue for items in by_subject for issue in items)) if by_subject else verification_issues(session)
    verified=sum(not items for items in by_subject)
    excluded = 0
    class_counts = Counter()
    for key, subject in session.subjects.items():
        for aid in subject.get("attempts", ()):
            state = selected_visual_state(session.review.attempt_annotations.get(aid, {}))
            if state:
                class_counts[state] += 1
        if (session.review.subjects.get(key, {}).get("evaluable") is False or
                (bool(subject.get("attempts", ())) and all(
                    not _active(session, session.mt_invocations.get(session.invocation_by_attempt.get(aid)))
                    for aid in subject.get("attempts", ())))):
            excluded += 1
    stored = str(session.review.review_status).upper()
    if not total:
        status = "NOT_STARTED"
    elif stored == "COMPLETED" and not issues:
        from .mobile_verification_snapshot import has_valid_ready_snapshot
        status = "READY_FOR_EVALUATION" if has_valid_ready_snapshot(session) else "APPROVED_PENDING_SNAPSHOT"
    elif not issues:
        status = "READY_TO_APPROVE"  # never certify automatically
    else:
        status = "IN_PROGRESS"
    return {"total": total, "verified": verified, "excluded": excluded,
            "issues": tuple(issues), "status": status, "visual_counts": dict(class_counts),
            "geometry_gt_optional": True,
            "review_revision": getattr(session.review, "review_revision", 0)}


def observation_verification_cells(session, subject_key: str, geometry_rows=None) -> tuple[str, ...]:
    """Short, binary *data presence* columns, not scores or per-sample accuracy.

    Columns: MT cardinality, indicated regions, human text GT, optional manual
    polygon GT, mandatory form completeness. ✓ means saved, not that MT was
    correct. ○ means missing; — means not applicable/excluded from that step.
    """
    subject = session.subjects.get(subject_key, {})
    from .mobile_mt_postdedup import suppressed_subject
    if suppressed_subject(session, subject):
        # Historic raw-only child; original GT and decisions remain stored.
        # This is *not* another physical plate awaiting review.
        return ("—", "—", "—", "—", "✓")
    attempts = [session.attempts[aid] for aid in subject.get("attempts", ())
                if aid in session.attempts]
    groups = {session.invocation_by_attempt[aid]:session.mt_invocations.get(session.invocation_by_attempt[aid])
              for aid in subject.get("attempts", ()) if aid in session.invocation_by_attempt}
    active = {gid:g for gid,g in groups.items() if _active(session,g)}
    from .mobile_canvas_review import has_scene, group_issues as canvas_group_issues
    if active and all(has_scene(session, gid) for gid in active):
        from .mobile_canvas_review import scene, scene_review_cells, merge_review_cells
        rows = []
        for gid in active:
            try:
                rows.append(scene_review_cells(scene(session, gid)))
            except (ValueError, TypeError, KeyError):
                rows.append(("○", "○", "○", "○", "○"))
        cells = list(merge_review_cells(rows))
        cells[-1] = "✓" if not subject_verification_issues(session, subject_key) else "○"
        return tuple(cells)
    cardinality = "—" if not active else "✓" if all(
        session.invocation_annotation(gid).get("visible_plate_count")
        in {"none", "one", "multiple", "uncertain"} for gid in active) else "○"
    multi_ids = [gid for gid in active if session.invocation_annotation(gid).get("visible_plate_count") == "multiple"]
    if multi_ids:
        from .mobile_multi_plate_verification import inventory,group_issues
        ready = all(not group_issues(session,gid) for gid in multi_ids)
        # A multi scene is complete only after matching every detected MT
        # region to an independently registered GT, or declaring it an FP.
        other = subject_verification_issues(session,subject_key)
        form = "✓" if not other else "○"
        has_gt = all(inventory(session,gid) is not None for gid in multi_ids)
        # The aggregate row is not a single plate; its children show which
        # GT records have linked detections and independent transcription.
        return ("✓" if has_gt else "○", "✓" if ready else "○",
                "✓" if ready else "○", "—", form)
    regions = [attempt for attempt in attempts
               if _attempt_status(attempt) in DETECTION_STATUSES
               and session.invocation_by_attempt.get(attempt["id"]) in active
               and not session.cancelled(attempt)]
    indicated = "—" if not regions else "✓" if all(
        selected_visual_state(session.review.attempt_annotations.get(row["id"], {}))
        for row in regions) else "○"
    decision = session.review.subjects.get(subject_key, {})
    gt = str(decision.get("ground_truth") or "").strip()
    has_mz = _subject_requires_text_gt(session, subject)
    no_real_plate = bool(active) and all(session.invocation_annotation(gid).get("visible_plate_count")=="none" for gid in active)
    gt_state = ("—" if no_real_plate else
                "✓" if gt and decision.get("evaluable") is True and session.predictions_visible(subject_key) else
                "—" if decision.get("evaluable") is False or not has_mz else "○")
    geom = geometry_rows or {}
    vals = [geom.get(row["id"], {}).get("metric_status", "") for row in attempts]
    has_manual_gt = any(status in {"measured", "missed_detection_gt_saved"} for status in vals)
    eligible_geometry = any(row["id"] in geom and geom[row["id"]].get("eligible") for row in attempts)
    # Optional GT is separate; an empty optional column cannot block Form.
    corners = "✓" if has_manual_gt else "○" if eligible_geometry else "—"
    form = "✓" if not subject_verification_issues(session, subject_key) else "○"
    return (cardinality, indicated, gt_state, corners, form)
