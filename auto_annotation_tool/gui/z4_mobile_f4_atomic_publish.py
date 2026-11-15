#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F4-only zero-conversion publish: atomic creation without overwriting files.

The normal exporter still builds AND fully validates the complete ALPR package.
Only in F4 no-conversion mode we direct its output to a unique staging path in
its final directory, adjust the *root* destination provenance, and publish via
os.link(src, dst). Unlike os.replace, link creation fails atomically if another
process has created the final name during packaging.

No model conversions, graph approval, Android approval or source modifications.
An unsupported filesystem fails closed rather than silently overwriting.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
from typing import Callable
import zipfile

ROOT_MANIFEST_MAX = 2 * 1024 * 1024


class F4AtomicPublishError(RuntimeError):
    """F4 publication could not be completed without a possible overwrite."""


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _destination_is_occupied(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _restore_f4_original_sources(root: dict, archive: zipfile.ZipFile,
                                 source_proofs: dict | None) -> dict[Path, str]:
    """Replace temporary MT/MZ source paths with verified ORIGINAL artifact paths.

    The normal Z4 F4 route first snapshots the verified children for a safe,
    stable bundle. Without this correction export_reproducibility.models[*]
    would point to ephemeral snapshots deleted after export, not the real
    registered packages. No child model bytes or manifests are modified.

    This additional provenance correction is only active when the caller
    provides the two SHA-pinned source proofs. Historical/synthetic callers
    without source proofs retain the previous ZIP behavior.
    """
    proof_keys = {'plate': 'reused_verified_mt', 'character': 'reused_verified_mz'}
    proofs = source_proofs if isinstance(source_proofs, dict) else {}
    provided = {role: key in proofs for role, key in proof_keys.items()}
    if not any(provided.values()):
        return {}
    if not all(provided.values()):
        raise F4AtomicPublishError('Bezpieczny zapis: brak jednego z dwóch przypiętych źródeł MT/MZ.')

    declared_meta = root.get('metadata')
    if (not isinstance(declared_meta, dict) or
            declared_meta.get('f4_zero_conversion_only') is not True):
        raise F4AtomicPublishError('Bezpieczny zapis: manifest końcowy nie potwierdza trybu bez konwersji.')
    repro = root.get('export_reproducibility')
    model_sources = repro.get('models') if isinstance(repro, dict) else None
    model_entries = root.get('models')
    if not isinstance(model_sources, dict) or not isinstance(model_entries, dict):
        raise F4AtomicPublishError('Bezpieczny zapis: brak proweniencji modeli źródłowych w pakiecie.')

    verified: dict[Path, str] = {}
    for role, proof_key in proof_keys.items():
        declared = declared_meta.get(proof_key)
        proof = proofs.get(proof_key)
        if not isinstance(declared, dict) or not isinstance(proof, dict):
            raise F4AtomicPublishError(f'Bezpieczny zapis: brak dowodu pochodzenia {role}.')
        expected_sha = str(proof.get('package_sha256') or '').lower()
        declared_sha = str(declared.get('package_sha256') or '').lower()
        if not re.fullmatch(r'[0-9a-f]{64}', expected_sha) or declared_sha != expected_sha:
            raise F4AtomicPublishError(f'Bezpieczny zapis: SHA przypiętego źródła {role} różni się od metadanych pakietu.')
        text_path = str(proof.get('path') or '').strip()
        declared_path = str(declared.get('path') or '').strip()
        if not text_path or not declared_path:
            raise F4AtomicPublishError(f'Bezpieczny zapis: brak pierwotnej ścieżki źródła {role}.')
        original = Path(text_path)
        if (original.is_symlink() or not original.is_file() or
                original.suffix.lower() != '.alprmodel' or
                Path(declared_path).resolve() != original.resolve()):
            raise F4AtomicPublishError(f'Bezpieczny zapis: pierwotne źródło {role} nie istnieje lub zostało podmienione.')
        original = original.resolve()
        if _digest(original) != expected_sha:
            raise F4AtomicPublishError(f'Bezpieczny zapis: paczka {role} zmieniła się od weryfikacji źródła.')

        root_entry = model_entries.get(role)
        repro_entry = model_sources.get(role)
        if not isinstance(root_entry, dict) or not isinstance(repro_entry, dict):
            raise F4AtomicPublishError(f'Bezpieczny zapis: brak metadanych modelu {role} w pakiecie.')
        rel = f'models/{role}/model.alprmodel'
        if root_entry.get('package_file') != rel:
            raise F4AtomicPublishError(f'Bezpieczny zapis: niezgodna ścieżka zagnieżdżonej paczki {role}.')
        hashes = root_entry.get('sha256')
        if not isinstance(hashes, dict) or str(hashes.get(rel) or '').lower() != expected_sha:
            raise F4AtomicPublishError(f'Bezpieczny zapis: SHA zagnieżdżonej paczki {role} nie zgadza się z oryginałem.')
        try:
            embedded_sha = hashlib.sha256(archive.read(rel)).hexdigest()
        except (KeyError, RuntimeError, OSError, zipfile.BadZipFile) as exc:
            raise F4AtomicPublishError(f'Bezpieczny zapis: brak lub błąd odczytu modelu {role}.') from exc
        if embedded_sha != expected_sha:
            raise F4AtomicPublishError(f'Bezpieczny zapis: zawartość zagnieżdżonej paczki {role} różni się od oryginału.')

        old_source = repro_entry.get('source_package')
        if (repro_entry.get('source_kind') != 'existing_package' or
                not isinstance(old_source, dict) or
                not str(old_source.get('path') or '').strip() or
                str(old_source.get('sha256') or '').lower() != expected_sha):
            raise F4AtomicPublishError(f'Bezpieczny zapis: dowód migawki źródłowej {role} jest niespójny.')
        old_source.update({'path': str(original), 'name': original.name,
                           'exists': True, 'is_file': True, 'is_dir': False,
                           'sha256': expected_sha})
        verified[original] = expected_sha
    return verified


def _copy_zip_correcting_destination(source: Path, destination: Path, final: Path,
                                     *, source_proofs: dict | None = None) -> dict[Path, str]:
    """Change only the root provenance of a package already validated by exporter.

    Every other ZIP member must keep exactly the same uncompressed bytes (SHA).
    No change to a child .alprmodel, TFLite file, or model manifest is permitted.
    """
    before = {}
    with zipfile.ZipFile(source, 'r') as archive:
        entries = archive.infolist()
        roots = [info for info in entries if info.filename == 'manifest.json' and not info.is_dir()]
        if len(roots) != 1 or len(entries) != len({info.filename for info in entries}):
            raise F4AtomicPublishError('Bezpieczny zapis: archiwum wynikowe ma niejednoznaczny manifest lub powielone wpisy.')
        info = roots[0]
        if info.file_size < 2 or info.file_size > ROOT_MANIFEST_MAX:
            raise F4AtomicPublishError('Bezpieczny zapis: manifest wynikowy ma nieprawidłowy rozmiar.')
        try:
            root = json.loads(archive.read(info).decode('utf-8'))
        except (ValueError, UnicodeError) as exc:
            raise F4AtomicPublishError('Bezpieczny zapis: manifest wynikowy nie jest prawidłowym JSON.') from exc
        if not isinstance(root, dict) or root.get('schema') != 'alpr.package.v1':
            raise F4AtomicPublishError('Bezpieczny zapis: oczekiwano zweryfikowanego pakietu alpr.package.v1.')
        provenance = root.get('export_reproducibility')
        package = provenance.get('package') if isinstance(provenance, dict) else None
        stored = package.get('destination') if isinstance(package, dict) else None
        if not isinstance(stored, dict) or Path(str(stored.get('path') or '')).resolve() != source.resolve():
            raise F4AtomicPublishError('Bezpieczny zapis: nie można bezpiecznie skorygować ścieżki docelowej w proweniencji.')
        if _destination_is_occupied(final):
            raise F4AtomicPublishError('Bezpieczny zapis: docelowy plik już istnieje; nie zostanie nadpisany.')
        # Same format as _path_reproducibility_payload(..., include_hash=False):
        # the intended output did not exist when the F4 export began.
        stored.update({'path': str(final.resolve()), 'name': final.name,
                       'exists': False, 'is_file': False, 'is_dir': False})
        original_source_pins = _restore_f4_original_sources(root, archive, source_proofs)
        updated_root = json.dumps(root, ensure_ascii=False, indent=2).encode('utf-8')
        if len(updated_root) > ROOT_MANIFEST_MAX:
            raise F4AtomicPublishError('Bezpieczny zapis: skorygowany manifest przekracza limit Android 2 MiB.')
        with zipfile.ZipFile(destination, 'w') as target:
            target.comment = archive.comment
            for member in entries:
                if member.is_dir():
                    target.writestr(member, b'')
                    continue
                payload = updated_root if member.filename == 'manifest.json' else archive.read(member)
                if member.filename != 'manifest.json':
                    before[member.filename] = hashlib.sha256(payload).hexdigest()
                target.writestr(member, payload)
    # Validate ZIP CRC and the SHA-256 of every unchanged internal member.
    with zipfile.ZipFile(destination, 'r') as result:
        if result.testzip() is not None:
            raise F4AtomicPublishError('Bezpieczny zapis: ponownie zapisany ZIP nie przeszedł kontroli CRC.')
        if set(result.namelist()) != set(before) | {'manifest.json'} | {
            entry.filename for entry in entries if entry.is_dir()
        }:
            raise F4AtomicPublishError('Bezpieczny zapis: zmieniła się lista elementów ZIP.')
        for path, old_sha in before.items():
            if hashlib.sha256(result.read(path)).hexdigest() != old_sha:
                raise F4AtomicPublishError('Bezpieczny zapis: plik modelu wewnątrz ZIP zmienił się podczas zapisu.')
        recorded = json.loads(result.read('manifest.json').decode('utf-8'))
        actual_destination = recorded['export_reproducibility']['package']['destination']['path']
        if Path(actual_destination) != final.resolve():
            raise F4AtomicPublishError('Bezpieczny zapis: proweniencja nie wskazuje ostatecznego pliku.')
    return original_source_pins


def export_f4_atomic_no_overwrite(
    request, *, export_fn: Callable, validate_final: Callable[[Path], None]
) -> Path:
    """Execute *already-validated* F4-only bundling without output name races.

    `export_fn` is the existing exporter route incl. MT/MZ SHA snapshots and its
    mandatory complete-package/graph validation. Never call this for a fresh
    conversion request. Caller rechecks the F4 no-conversion request first.
    """
    if not callable(export_fn):
        raise F4AtomicPublishError('Bezpieczny zapis: brak właściwego eksportera kompletnej paczki.')
    if not callable(validate_final):
        raise F4AtomicPublishError(
            'Bezpieczny zapis: brak obowiązkowej walidacji gotowego archiwum po korekcie proweniencji.'
        )
    meta = getattr(request, 'metadata', None)
    if not isinstance(meta, dict) or meta.get('f4_zero_conversion_only') is not True:
        raise F4AtomicPublishError('Bezpieczny zapis: zapis bez nadpisywania jest dostępny tylko dla trybu bez konwersji.')
    if any(getattr(request, key, None) is not None for key in (
            'plate_request', 'character_request', 'vehicle_request', 'vehicle_package')):
        raise F4AtomicPublishError('Bezpieczny zapis: próba uruchomienia konwersji lub dołączenia MP zablokowana.')
    raw = str(getattr(request, 'destination', '') or '').strip()
    if not raw or Path(raw).suffix.lower() != '.alprmodel':
        raise F4AtomicPublishError('Bezpieczny zapis: docelowy plik musi mieć rozszerzenie .alprmodel.')
    final = Path(raw).absolute()
    final.parent.mkdir(parents=True, exist_ok=True)
    if _destination_is_occupied(final):
        raise F4AtomicPublishError('Bezpieczny zapis: docelowy plik już istnieje; nie zostanie nadpisany.')
    # Use the same directory/filesystem to permit atomic, fail-if-exists os.link.
    stage_fd, stage_name = tempfile.mkstemp(prefix=f'.{final.stem}.f4-',
                                           suffix='.alprmodel', dir=str(final.parent))
    os.close(stage_fd)
    stage = Path(stage_name)
    fixed_stage = stage.with_name(stage.name + '.fixed.zip')
    try:
        staged_request = replace(request, destination=stage)
        produced = Path(export_fn(staged_request))
        if produced.resolve() != stage.resolve() or not stage.is_file() or stage.is_symlink():
            raise F4AtomicPublishError('Bezpieczny zapis: eksporter nie zwrócił oczekiwanego pliku pośredniego.')
        if _destination_is_occupied(final):
            raise F4AtomicPublishError('Bezpieczny zapis: plik docelowy powstał podczas pakowania; nie nadpisuję.')
        original_source_pins = _copy_zip_correcting_destination(
            stage, fixed_stage, final, source_proofs=meta,
        )
        # This ZIP, not the pre-rewrite archive, is exactly what F4 publishes.
        # Perform authoritative complete ALPR + nested model/precision checks
        # after updating the provenance and before an atomic create-if-absent.
        fixed_sha = _digest(fixed_stage)
        try:
            validation_result = validate_final(fixed_stage)
        except Exception as exc:
            raise F4AtomicPublishError(
                f'Bezpieczny zapis: końcowe archiwum nie przeszło pełnej walidacji po zmianie proweniencji: {exc}'
            ) from exc
        if validation_result is False:
            raise F4AtomicPublishError('Bezpieczny zapis: walidator odrzucił końcowe archiwum.')
        if _digest(fixed_stage) != fixed_sha:
            raise F4AtomicPublishError(
                'Bezpieczny zapis: końcowe archiwum zmieniło się podczas walidacji; publikacja zablokowana.'
            )
        # The original on-disk sources must still match the pinned bytes,
        # even after the expensive complete-package graph validation.
        for original, expected_sha in original_source_pins.items():
            if original.is_symlink() or not original.is_file() or _digest(original) != expected_sha:
                raise F4AtomicPublishError(
                    'Bezpieczny zapis: oryginalna paczka źródłowa zmieniła się podczas walidacji; publikacja zablokowana.'
                )
        if _destination_is_occupied(final):
            raise F4AtomicPublishError('Bezpieczny zapis: docelowy plik powstał przed publikacją; nie nadpisuję.')
        try:
            # Atomic CREATE, never replace; explicit failure on unsupported FS.
            os.link(fixed_stage, final)
        except FileExistsError as exc:
            raise F4AtomicPublishError('Bezpieczny zapis: plik docelowy powstał równolegle — zapis zablokowany.') from exc
        except OSError as exc:
            raise F4AtomicPublishError(
                f'Bezpieczny zapis: system nie pozwolił na atomowy zapis bez nadpisywania ({exc}).'
            ) from exc
        if _digest(final) != fixed_sha:
            raise F4AtomicPublishError('Bezpieczny zapis: wynik zmienił się po atomowej publikacji — wymaga kontroli.')
        return final
    finally:
        # Cleanup only our uniquely named staging files, NEVER final destination.
        for file in (stage, fixed_stage):
            try:
                file.unlink(missing_ok=True)
            except OSError:
                pass
