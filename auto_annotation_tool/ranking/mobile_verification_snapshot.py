"""Immutable, revision-bound human verification evidence for later evaluation.

READY_FOR_EVALUATION is evidence completeness, not a verified scientific result.
Original Android archives and existing human-review sidecars are never replaced.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

from .mobile_verification_policy import verification_issues, verification_progress


def _bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _write_once(path: Path, content: bytes) -> None:
    """Exclusive file creation; on collision verify immutable contents."""
    if path.exists():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == content:
            return
        raise ValueError(f"Istnieje inna wersja zatwierdzonego dokumentu: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f"Konflikt niezmiennego dokumentu: {path}")


def _manifest_path(session) -> Path:
    source_sha = str(session.review.source_archive_sha256).lower()
    sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(session.session_id))[:90]
    rev = int(session.review.review_revision)
    return (session.sidecar_path.parent / "ready_for_evaluation" /
            f"{source_sha[:16]}_{sid}_rev{rev:06d}" / "manifest.json")


def has_valid_ready_snapshot(session) -> bool:
    try:
        path = _manifest_path(session)
        if path.is_symlink() or not path.is_file():
            return False
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return False
        target = path.parent / str(manifest.get("review_snapshot_file") or "")
        if target.is_symlink() or not target.is_file() or target.name != "verified_review.json":
            return False
        data = target.read_bytes()
        return (manifest.get("schema") == "alpr.desktop.review_ready_manifest.v1" and
                manifest.get("status") == "READY_FOR_EVALUATION" and
                manifest.get("source_archive_sha256") == session.review.source_archive_sha256 and
                manifest.get("session_id") == session.session_id and
                manifest.get("review_id") == session.review.review_id and
                manifest.get("review_revision") == session.review.review_revision and
                manifest.get("review_snapshot_sha256") == _sha(data) and
                data == _bytes(session.review.to_dict()))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def create_ready_snapshot(session) -> Path:
    if str(session.review.review_status).upper() != "COMPLETED":
        raise ValueError("Weryfikacja nie została zatwierdzona")
    issues = verification_issues(session)
    if issues:
        raise ValueError("Niekompletny formularz: " + issues[0])
    session.verify_source(force=True)
    source_sha = str(session.review.source_archive_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ValueError("Brak zweryfikowanego SHA-256 źródła")
    sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(session.session_id))[:90]
    if not sid:
        raise ValueError("Brak identyfikatora sesji")
    rev = int(session.review.review_revision)
    directory = session.sidecar_path.parent / "ready_for_evaluation" / f"{source_sha[:16]}_{sid}_rev{rev:06d}"
    review_payload = session.review.to_dict()
    review_bytes = _bytes(review_payload)
    progress = verification_progress(session)
    manifest = {
        "schema": "alpr.desktop.review_ready_manifest.v1",
        "status": "READY_FOR_EVALUATION",
        "scientific_results_verified": False,
        "source_archive_sha256": source_sha,
        "session_id": session.session_id,
        "review_id": session.review.review_id,
        "review_revision": rev,
        "review_mode": session.review.review_mode,
        "review_snapshot_file": "verified_review.json",
        "review_snapshot_sha256": _sha(review_bytes),
        "verification_policy": "mobile_visual_mt_v1_polygon_gt_optional",
        "form_total_observations": progress["total"],
        "form_verified_observations": progress["verified"],
        "form_excluded_observations": progress["excluded"],
        "visual_mt_class_counts": progress["visual_counts"],
        "geometry_gt_required_for_readiness": False,
        "multi_plate_protocol": "alpr.mobile.mt_multi_plate.v1",
        "multi_plate_counted_in_single_mt_metrics": False,
        "closed_at_utc": session.review.updated_at,
    }
    _write_once(directory / "verified_review.json", review_bytes)
    _write_once(directory / "manifest.json", _bytes(manifest))
    return directory / "manifest.json"
