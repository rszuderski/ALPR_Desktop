"""Read-only UI projection of persisted human decisions, separate from eligibility.

Neither source N/A nor an empty detector output is a human decision. This
module never changes Canvas eligibility, scientific denominators or sidecars.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "alpr.desktop.operator_progress.v1"


def has_decisions(value):
    """Explicit saved review fields/events; never predictions or an empty seed."""
    if not isinstance(value, dict):
        return False
    if value.get("decision_source") != "legacy_cardinality_unknown":
        if any(type(value.get(k)) is bool for k in ("evaluable", "is_plate")):
            return True
        if value.get("visible_plate_count") in {"none", "one", "multiple", "uncertain"}:
            return True
    if any(value.get(k) is True for k in ("checked", "gt_confirmed", "text_confirmed", "scene_confirmed", "empty_confirmed")):
        return True
    if value.get("presence") in {"PRESENT", "NO_PLATES"} or value.get("roi_presence") == "NO_PLATES":
        return True
    decisions = value.get("decisions", {})
    if not isinstance(decisions, dict):
        raise ValueError("Nieprawidłowy wykaz decyzji operatora.")
    if any(v in {"PLATE", "NOT_PLATE"} for v in decisions.values() if isinstance(v, str)):
        return True
    if str(value.get("geometry_state", "")).startswith("operator_"):
        return True
    for event in value.get("events", ()):
        action = str(event.get("action", "")) if isinstance(event, dict) else ""
        if action.startswith("operator_") or action in {
            "add_scene_plate", "edit_scene_plate", "remove_scene_plate", "confirm_complete_roi_review",
            "exclude_roi_review", "undo_scene_frame", "redo_scene_frame", "undo_stack_history", "redo_stack_history",
        }:
            return True
    # Do not walk parents (source predictions), provenance, or old undo snapshots.
    for key in ("subjects", "attempt_annotations", "sample_annotations", "invocation_annotations", "scene_annotations"):
        mapping = value.get(key, {})
        if not isinstance(mapping, dict):
            raise ValueError("Nieprawidłowy wykaz zapisanych ocen.")
        if any(has_decisions(v) for v in mapping.values() if isinstance(v, dict)):
            return True
    for key in ("canvas_review_v1", "operator_frame_decisions_v1"):
        if has_decisions(value.get(key)):
            return True
    return any(has_decisions(v) for v in value.get("objects", ()) if isinstance(v, dict))


def _provenance(progress):
    if progress["sidecar_state"] == "ABSENT":
        return "Brak zapisanych decyzji operatora i sidecara dla tego session_id oraz SHA-256."
    if progress["sidecar_state"] != "BOUND":
        return "Nie można potwierdzić zapisu operatora: " + progress.get("error", "brak powiązania źródła")
    return (f"Zapis operatora: {progress['sidecar_path']} · rewizja {progress['review_revision']} · "
            f"autor: {progress.get('reviewer_id') or 'nie zapisano'} · data: {progress.get('updated_at') or 'nie zapisano'} · "
            f"SHA sidecara: {progress['sidecar_sha256']}")


def from_session(session, inventory):
    """Use a source-validated session and its existing inventory; no metric calls."""
    review, bundle = session.review, session.bundle
    result = dict(schema=SCHEMA, session_id=session.session_id,
        source_sha256=review.source_archive_sha256, review_revision=review.review_revision,
        review_status=review.review_status, source_verified=False, sidecar_state="UNVERIFIED",
        sidecar_path=str(getattr(session, "sidecar_path", "")), sidecar_sha256="",
        operator_state="UNVERIFIED", assessed=None, total=None, mt_assessed=0,
        mt_total=len(session.mt_invocations), touched=0, has_decisions=False, units=[],
        unit_kind="UNKNOWN", denominator_reason="Brak kompletnego rejestru obrazów źródłowych.")
    result["evidence"] = dict(
        registered_scenes=len(getattr(session, "scene_frames", {})),
        written_scenes=sum(r.get("write_status") == "WRITTEN" for r in getattr(session, "scene_frames", {}).values()),
        mt_invocations=len(session.mt_invocations), attempts=len(session.attempts), crops=len(session.samples),
        no_reference=inventory.get("no_reference", 0), source_na=inventory.get("source_na", 0))
    try:
        if (not bundle.validation.ok or bundle.source_archive_sha256 != review.source_archive_sha256
                or session.session_id != review.session_id):
            raise ValueError("Niezgodna tożsamość sesji lub integralność źródła.")
        session.verify_source()
        result["source_verified"] = True
        frames = getattr(session, "scene_frames", {})
        extension = getattr(bundle, "scene_evidence", {}) or {}
        if (frames and extension.get("available") and extension.get("scene_evidence_complete") is True
                and extension.get("pending_loss_unknown") is False
                and all(r.get("write_status") == "WRITTEN" and r.get("source_frame_entry") in session.entry_names
                        for r in frames.values())):
            result.update(total=len(frames), unit_kind="SCENE_FRAME", denominator_reason="Jawny, kompletny rejestr SCENE_FRAME; jednostka: scene_event_id.")
        elif session.mt_invocations and session.attempts_available:
            result["unit_kind"] = "MT_ROI"
        path = Path(session.sidecar_path)
        if not path.is_file():
            result.update(sidecar_state="ABSENT", assessed=0, operator_state="NOT_STARTED")
        else:
            from .mobile_human_review import MobileHumanReview
            if path.stat().st_size > 12 * 1024 * 1024:
                raise ValueError("Za duży sidecar weryfikacji.")
            raw = path.read_bytes()
            saved = MobileHumanReview.from_dict(json.loads(raw.decode("utf-8-sig")),
                source_sha256=result["source_sha256"], session_id=result["session_id"])
            # Match the persisted JSON representation: provenance contains tuples
            # in a freshly opened bundle, and the native loader normalizes locks.
            current = MobileHumanReview.from_dict(json.loads(json.dumps(review.to_dict())),
                source_sha256=result["source_sha256"], session_id=result["session_id"])
            if saved.to_dict() != current.to_dict():
                raise ValueError("Sidecar zmienił się względem otwartej rewizji; odśwież weryfikację.")
            session._validate_decisions(saved)
            result.update(sidecar_state="BOUND", sidecar_sha256=hashlib.sha256(raw).hexdigest(),
                          reviewer_id=saved.reviewer_id, updated_at=saved.updated_at,
                          has_decisions=has_decisions(saved.to_dict()), assessed=0)
            from . import mobile_scene_frame_review as scene_review
            for key, annotation in saved.scene_annotations.items():
                # Revalidate the exact source binding. A checked projection alone
                # is not sufficient without its persisted human annotation.
                projected = scene_review.scene(session, key, include_history=False)
                explicit = has_decisions(annotation)
                assessed = explicit and projected.get("checked") is True
                result["touched"] += int(explicit)
                result["assessed"] += int(assessed)
                result["units"].append(dict(kind="SCENE_FRAME", id=key, assessed=assessed,
                                            source_binding=annotation.get("source_binding")))
            for gid, item in inventory.get("groups", {}).items():
                annotation = saved.invocation_annotations.get(gid, {})
                canvas = annotation.get("canvas_review_v1")
                explicit = has_decisions(annotation)
                if isinstance(canvas, dict):
                    if item.get("scene") is None:
                        raise ValueError(f"Nieprawidłowe powiązanie zapisanej oceny Canvas: {gid}")
                    frame = item.get("frame_review", {})
                    assessed = explicit and (canvas.get("checked") is True or frame.get("state") == "FRAME_READY_ASSISTED")
                else:
                    assessed = explicit and not session.invocation_completion_issues(gid)
                result["mt_assessed"] += int(assessed)
                result["touched"] += int(explicit)
                if explicit:
                    result["units"].append(dict(kind="MT_ROI", id=gid, assessed=assessed,
                                                source_binding=(canvas or {}).get("source_binding")))
            result["operator_state"] = ("COMPLETED" if result["has_decisions"] and saved.review_status == "COMPLETED"
                                        else "IN_PROGRESS" if result["has_decisions"] else "NOT_STARTED")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result.update(sidecar_state="INVALID", operator_state="UNVERIFIED", assessed=None, mt_assessed=None, error=str(exc))
    result["provenance"] = _provenance(result)
    return result


def presentation(progress):
    """Same operator counter/status in the browser and sample review."""
    p = progress or {}
    valid = p.get("schema") == SCHEMA and p.get("source_verified") and p.get("sidecar_state") in {"BOUND", "ABSENT"}
    state = p.get("operator_state") if valid else "UNVERIFIED"
    status = {"NOT_STARTED": "Nie rozpoczęto", "IN_PROGRESS": "W toku", "COMPLETED": "Zakończona"}.get(state, "Nie można potwierdzić weryfikacji")
    total, assessed = p.get("total"), p.get("assessed")
    population_known = p.get('source_verified') and type(total) is int and total >= 0
    known = valid and population_known and type(assessed) is int and 0 <= assessed <= total
    counter = (f"Obrazy ocenione przez operatora: {assessed}/{total}" if known
               else f"Obrazy do oceny: {total} · postęp operatora niepotwierdzony" if population_known
               else "Nie ustalono liczby obrazów do oceny")
    ratio = assessed / total if known and total else None
    details = [p.get("provenance", "Brak potwierdzonego zapisu operatora.")]
    if valid and p.get("unit_kind") != "SCENE_FRAME":
        details.append(f"Oddzielny zakres MT/ROI: zapisane oceny {p.get('mt_assessed', 0)}/{p.get('mt_total', 0)}; nie jest to liczba ocenionych pełnych obrazów.")
    elif valid and p.get("mt_assessed"):
        details.append(f"Oddzielnie zapisane oceny MT/ROI: {p['mt_assessed']}/{p['mt_total']}; nie zastępują oceny pełnej sceny.")
    if known:
        details.append("Jednostka: zarejestrowany obraz pełnej sceny (scene_event_id), nie pomiar czasu ani crop.")
        details.append("Ocena obrazu nie oznacza ukończenia oceny GT/MT/MZ; jej status podano osobno.")
    else:
        details.append(p.get("denominator_reason", "Brak potwierdzonej populacji obrazów."))
    return dict(status=status, progress=counter, progress_ratio=ratio, progress_subtitle="\n".join(details))


def image_counts(progress):
    p = progress or {}
    total, assessed = p.get('total'), p.get('assessed')
    known = (p.get('source_verified') and p.get('sidecar_state') in {'BOUND', 'ABSENT'}
             and type(total) is int and type(assessed) is int and 0 <= assessed <= total)
    return dict(assessed=assessed if known else None, total=total if p.get('source_verified') else None,
                remaining=total-assessed if known else None,
                fraction=f'{assessed}/{total}' if known else 'Nie ustalono')


def evidence_caption(progress):
    e = (progress or {}).get("evidence", {})
    if not e:
        return ""
    return (f"Dowody źródłowe: zapisane pełne klatki SCENE_FRAME {e['written_scenes']}/{e['registered_scenes']}; "
            f"wywołania MT: {e['mt_invocations']}; rekordy attempts: {e['attempts']}; cropy MZ: {e['crops']}. "
            f"MT/ROI bez referencji: {e['no_reference']}; źródło nieocenialne: {e['source_na']}. "
            "Są to dowody i stany techniczne, nie decyzje operatora.")
