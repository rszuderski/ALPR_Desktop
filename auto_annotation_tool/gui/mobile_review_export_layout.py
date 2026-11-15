"""Safe per-session, per-review-revision export layout for F4/F5/F6.

This module does not import Tk or write to source .alprsession files.  It
creates new directories below the already existing ``03_wyniki`` and never
replaces a previous export.  It intentionally does not interpret rNN folders:
some source sessions have known variant/replicate metadata discrepancies.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z", re.ASCII)
_SHA256 = re.compile(r"[a-fA-F0-9]{64}\Z", re.ASCII)
_MAX_EXPORTS_PER_REVISION = 999


def validate_export_destination(base, session_id, revision):
    """Read-only validation of the export root and session identifier."""
    base = Path(base)
    if base.name.casefold() != "03_wyniki" or not base.is_dir() or base.is_symlink():
        raise ValueError("Wybierz istniejący folder 03_wyniki kampanii (nie porownanie ani 02_sesje).")
    if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
        raise ValueError("Nieprawidłowy session_id. Eksport przerwany bez zapisu.")
    if session_id in (".", "..") or session_id.endswith(".") or session_id.endswith(" "):
        raise ValueError("Nieprawidłowa nazwa katalogu sesji.")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0 or revision > 999999:
        raise ValueError("Nieprawidłowa rewizja weryfikacji; nie zapisuję eksportu.")
    session_dir = base / session_id
    if session_dir.is_symlink() or (session_dir.exists() and not session_dir.is_dir()):
        raise ValueError("Ścieżka sesji koliduje z istniejącym plikiem/dowiązaniem.")
    return base, session_dir, revision


def _export_dirname(revision, attempt):
    return f"review_r{revision:06d}_export_{attempt:03d}"


def next_export_path(base, session_id, revision):
    """Preview the next free path; does not create folders."""
    _, session_dir, revision = validate_export_destination(base, session_id, revision)
    for attempt in range(1, _MAX_EXPORTS_PER_REVISION + 1):
        candidate = session_dir / _export_dirname(revision, attempt)
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise ValueError("Osiągnięto limit numeracji eksportów tej rewizji.")


def reserve_export_path(base, session_id, revision):
    """Create a NEW destination directory without overwriting old exports.

    Allocation uses mkdir(exist_ok=False), so parallel reviews cannot reserve
    the same export number.  Never deletes or moves pre-existing files.
    """
    base, session_dir, revision = validate_export_destination(base, session_id, revision)
    session_dir.mkdir(exist_ok=True)
    if session_dir.is_symlink() or not session_dir.is_dir():
        raise ValueError("Katalog sesji nie jest zwykłym folderem.")
    for attempt in range(1, _MAX_EXPORTS_PER_REVISION + 1):
        candidate = session_dir / _export_dirname(revision, attempt)
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise ValueError("Osiągnięto limit numeracji eksportów tej rewizji.")


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_manifest(session, export_dir, rows, variant=None):
    review = session.review
    source_sha = str(review.source_archive_sha256).lower()
    if not _SHA256.fullmatch(source_sha):
        raise ValueError("Nieprawidłowy SHA-256 źródłowej sesji; manifest nie powstanie.")
    export_dir = Path(export_dir)
    all_entries = sorted(export_dir.rglob("*"))
    if any(p.is_symlink() for p in all_entries):
        raise ValueError("Eksport zawiera dowiązanie; manifest nie powstanie.")
    all_files = [p for p in all_entries if p.is_file()]
    expected = {"review.json", "summary.csv", "subjects.csv",
                "character_confusion.csv", "mt_pose_geometry.csv",
                "mt_multi_plate_verified.json"}
    missing = expected - {p.name for p in all_files if p.parent == export_dir}
    if missing:
        raise ValueError("Brakuje obowiązkowych plików eksportu: " + ", ".join(sorted(missing)))
    files = [{"path": p.relative_to(export_dir).as_posix(),
              "bytes": p.stat().st_size, "sha256": _sha256_file(p)}
             for p in all_files]
    paths_returned = {Path(p).resolve() for p in rows}
    # The thesis exporter returns its `sources/` directory as well as files.
    if not paths_returned.issubset({p.resolve() for p in all_entries}):
        raise ValueError("Eksporter zwrócił ścieżkę spoza nowego katalogu lub brakujący wynik.")
    provenance = review.provenance if isinstance(review.provenance, dict) else {}
    primary=[]
    for path in export_dir.glob('mt_object_evaluation_v2_*/manifest.json'):
        item=json.loads(path.read_text(encoding='utf-8'))
        if (item['session_id']!=session.session_id or item['review_revision']!=review.review_revision
                or item['source_archive_sha256']!=source_sha):
            raise ValueError('Niespójna rewizja lub źródło eksportu F4-EVAL-2.0.')
        primary.append(dict(manifest=path.relative_to(export_dir).as_posix(),
            methodology_version=item['methodology_version'],matching_policy_id=item['matching_policy_id'],
            matching_policy_sha256=item['matching_policy_sha256']))
    if len(primary)>1:raise ValueError('Niejednoznaczny eksport PRIMARY F4-EVAL-2.0.')
    return {
        "schema": "alpr_mobile_review_export_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "export_layout": "03_wyniki / session_id / review_revision_export_N",
        "session_id": str(session.session_id),
        "review_id": str(review.review_id),
        "review_revision": review.review_revision,
        "review_status": review.review_status,
        "review_mode": review.review_mode,
        "source_archive_name": Path(session.path).name,
        "source_archive_sha256": source_sha,
        "variant_as_recorded": provenance.get("variant_id"),
        "experiment_session_as_recorded": provenance.get("experiment_session"),
        "f4_variant": variant,
        "primary_evaluation": primary[0] if primary else None,
        "note": "Etykieta F4 ma osobną proweniencję rejestru SHA. Surowych metadanych telefonu nie poprawiano.",
        "files": files,
    }


def finalize_export_manifest(session, export_dir, exported_paths, *, variant=None, before_publish=None):
    """Write an exclusive manifest after every exporter finished successfully."""
    export_dir = Path(export_dir)
    manifest_path = export_dir / "export_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Manifest eksportu już istnieje: " + str(manifest_path))
    manifest = _plain_manifest(session, export_dir, exported_paths, variant)
    # Reject side-effecting edits to source/archive after the exported files were written.
    session.verify_source(force=True)
    if before_publish is not None:before_publish()
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest_path


def mark_export_incomplete(export_dir, reason):
    """Keep partial outputs for diagnosis; never claim they are final."""
    target = Path(export_dir) / "EKSPORT_NIEKOMPLETNY.txt"
    try:
        with target.open("x", encoding="utf-8") as handle:
            handle.write("Eksport nie został ukończony; NIE wykorzystywać plików do porównań.\n")
            handle.write(str(reason)[:2000] + "\n")
    except OSError:
        pass  # Preserve the original export exception.
