"""Read-only Desktop GT/status projection for source-verified Android crops.

The Android archive's verification_status is an immutable historical field.
Desktop decisions live in a SHA-pinned .review.json. Stage status comes from the
same canonical scientific computation used by the actual review window; never
infer completed MT or MZ simply because a GT field is non-empty.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

_SHA = re.compile(r'[0-9a-f]{64}\Z')


def project_desktop_samples(bundle: Any, science: dict, root: Path | str, *,
                            decode_review: Callable | None = None) -> dict:
    """Build a preview, not a scoring artifact. Fails closed on wrong provenance."""
    bundle_sha = str(getattr(bundle, 'source_archive_sha256', '') or '').lower()
    science_sha = str(science.get('source_archive_sha256') or '').lower()
    session_id = str(science.get('session_id') or '')
    review_subjects = science.get('subjects')
    if (not getattr(getattr(bundle, 'validation', None), 'ok', False)
            or not _SHA.fullmatch(bundle_sha) or science_sha != bundle_sha
            or not session_id or not isinstance(review_subjects, dict)
            or str(getattr(getattr(bundle, 'report', None), 'report_id', '') or '') !=
               str(science.get('report_id') or getattr(getattr(bundle, 'report', None), 'report_id', ''))):
        raise ValueError('Niezweryfikowana tożsamość raportu / weryfikacji Desktop')
    path = Path(root) / 'human_reviews' / f'{bundle_sha}.review.json'
    if not path.exists():
        return {'state': 'not_started', 'revision': None, 'samples': {}}
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('Nieprawidłowy lub zbyt duży sidecar weryfikacji')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(payload, dict) or str(payload.get('source_archive_sha256') or '').lower() != bundle_sha
            or str(payload.get('session_id') or '') != session_id):
        raise ValueError('Nie zgadza się SHA-256 lub session_id sidecara')
    if decode_review is None:
        from ..ranking.mobile_human_review import MobileHumanReview
        decode_review = MobileHumanReview.from_dict
    review = decode_review(payload, source_sha256=bundle_sha, session_id=session_id)
    if review.review_revision != science.get('review_revision'):
        raise ValueError('Weryfikacja została zmieniona podczas odczytu — odśwież widok')
    sample_view = {}
    for row in getattr(bundle, 'sample_rows', ()):
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get('capture_id') or row.get('sample_id') or row.get('id') or '')
        subject_key = str(row.get('subject_key') or '')
        if not sample_id or sample_id in sample_view or subject_key not in review_subjects:
            continue
        stage = review_subjects.get(subject_key)
        if not isinstance(stage, dict):
            continue
        decision = review.subjects.get(subject_key, {})
        sample_decision = review.sample_annotations.get(sample_id, {})
        gt = str(decision.get('ground_truth') or '').strip()
        gt_ok = bool(gt and decision.get('evaluable') is True and decision.get('gt_confirmed') is True)
        status = str(stage.get('label') or 'Weryfikacja nieustalona')
        if sample_decision.get('evaluable') is False:
            status = 'Wyłączono odczyt MZ'
        sample_view[sample_id] = {
            'status': status,
            'gt': gt if gt_ok else '',
            'subject_key': subject_key,
        }
    return {'state': 'current', 'revision': review.review_revision, 'samples': sample_view}
