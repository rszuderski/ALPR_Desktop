"""One SHA-256 boundary for every campaign image manifest writer.

Only selected files and prior manifest members are read. Call from an ingest or
iteration worker; resource dialogs merely read the persisted summary.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from collections import Counter

SCHEMA = "alpr.campaign.image-hash-index.v1"
MANIFEST_IDENTITY = "alpr.campaign.image-identity.v1"


class IdentityRejected(ValueError):
    def __init__(self, summary):
        self.summary = summary
        examples = ", ".join(f"{item['name']} → {item['existing'].get('name', '')}"
                             for item in summary.get("duplicates", [])[:3])
        super().__init__(f"Nie dodano nowych obrazów. Identyczne: {summary['duplicate_sha256']}; "
                         f"konflikty nazw: {summary['name_collisions']}; niedostępne pliki: {summary['missing_files']}."
                         + (f"\nPrzykłady: {examples}." if examples else ""))


def _key(path):
    return os.path.normcase(str(Path(path).resolve()))


def _read(path):
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Nieprawidłowy indeks lub manifest: {path}")
    return value


def _atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                          prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _write_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+b") as stream:
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        deadline = time.monotonic() + 30
        while True:
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Trwa inny zapis zasobów tego projektu.")
                time.sleep(.02)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ImageHashIndex:
    def __init__(self, path):
        self.path = Path(path)
        self.data = _read(self.path) or {"schema": SCHEMA, "algorithm": "sha256", "hashes": {}, "path_cache": {}, "manifests": {}}
        if self.data.get("schema") != SCHEMA or self.data.get("algorithm") != "sha256":
            raise ValueError("Nieobsługiwany indeks tożsamości obrazów.")
        for key in ("hashes", "path_cache", "manifests"):
            self.data.setdefault(key, {})

    def fingerprint(self, path):
        path = Path(path)
        key = _key(path)
        before = path.stat()
        stamp = (before.st_size, before.st_mtime_ns)
        cached = self.data["path_cache"].get(key) or {}
        if (cached.get("size"), cached.get("mtime_ns")) == stamp and re.fullmatch(r"[0-9a-f]{64}", str(cached.get("sha256") or "")):
            return cached["sha256"]
        digest = file_sha256(path)
        after = path.stat()
        if stamp != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Plik zmienił się podczas sprawdzania: {path.name}. Spróbuj ponownie.")
        self.data["path_cache"][key] = {"size": stamp[0], "mtime_ns": stamp[1], "sha256": digest}
        return digest

    def register(self, entries, manifest_path, iteration):
        owner = _key(manifest_path)
        for record in self.data["hashes"].values():
            record["occurrences"] = [item for item in record.get("occurrences", []) if item.get("manifest") != owner]
        for entry in entries:
            digest = entry["content_sha256"]
            occurrence = {"manifest": owner, "iteration": iteration,
                          "name": entry.get("name", ""), "source_path": entry.get("source_path", "")}
            record = self.data["hashes"].setdefault(digest, {
                "first_seen_iteration": iteration, "first_seen_manifest": owner,
                "name": occurrence["name"], "source_path": occurrence["source_path"], "occurrences": []})
            if occurrence not in record["occurrences"]:
                record["occurrences"].append(occurrence)

    def refresh_prior_manifests(self, directory, progress=None):
        missing = []
        for manifest_path in sorted(Path(directory).glob("iter_*_manifest.json")):
            stat = manifest_path.stat()
            stamp = [stat.st_size, stat.st_mtime_ns]
            if self.data["manifests"].get(_key(manifest_path)) == stamp:
                continue
            manifest = _read(manifest_path)
            entries = []
            for item in manifest.get("selected_images") or []:
                if not isinstance(item, dict):
                    continue
                digest = str(item.get("content_sha256") or "")
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    source = _entry_path(item, manifest)
                    if source is None:
                        missing.append({"manifest": str(manifest_path), "name": item.get("name", "")})
                        continue
                    digest = self.fingerprint(source)
                entries.append({**item, "content_sha256": digest})
                if progress:
                    progress(5, "Sprawdzam wcześniejsze zasoby projektu.", detail=str(item.get("name", "")))
            self.register(entries, manifest_path, int(manifest.get("iteration") or 0))
            # Missing legacy files can become available again; retry on next write.
            if not any(item["manifest"] == str(manifest_path) for item in missing):
                self.data["manifests"][_key(manifest_path)] = stamp
        return missing


def _entry_path(entry, manifest):
    # The source has priority: a same-named old target may hold different bytes.
    for key in ("source_path", "target_path", "iteration_target_path"):
        raw = str(entry.get(key) or "").strip()
        if raw:
            path = Path(raw)
            if path.is_file():
                return path
    raw = str(manifest.get("source_dir") or "").strip()
    if raw and entry.get("name"):
        path = Path(raw) / entry["name"]
        if path.is_file():
            return path
    return None


def save_manifest_with_identity(manifest, manifest_path, *, progress=None, approved_entries=()):
    """Atomically write a validated manifest and its recoverable project index."""
    from .campaign_ingest_planner import CampaignIngestPlanner
    manifest_path = Path(manifest_path)
    state_dir = manifest_path.parent.parent if manifest_path.parent.name == "ingest" else manifest_path.parent
    index_path = state_dir / "image_hash_index.json"
    planner = CampaignIngestPlanner()
    with _write_lock(index_path):
        index = ImageHashIndex(index_path)
        unresolved = index.refresh_prior_manifests(manifest_path.parent, progress)
        legacy = []
        for item in approved_entries:
            source = Path(str(item.get("source_image_path") or ""))
            digest = str(item.get("source_file_sha256") or "")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                if not source.is_file():
                    continue
                digest = index.fingerprint(source)
            legacy.append({"name": item.get("image_name") or source.name,
                           "source_path": str(source), "content_sha256": digest})
        if legacy:
            index.register(legacy, state_dir / "approved-geometry-source", 0)
        owner = _key(manifest_path)
        iteration = int(manifest.get("iteration") or 0)
        reference_mode = manifest.get("identity_mode") == "reference"
        source_iteration = int(manifest.get("reused_from_iteration") or 0)
        accepted, duplicates, collisions, missing = [], [], [], []
        batch_hashes, batch_names = {}, {}
        project_names = {}
        for digest, record in index.data["hashes"].items():
            project_names.setdefault(str(record.get("name") or "").lower(), set()).add(digest)
        inputs = list(manifest.get("selected_images") or [])
        reused_count = 0
        for number, item in enumerate(inputs, 1):
            if not isinstance(item, dict):
                raise ValueError("Nieprawidłowa pozycja manifestu obrazów.")
            path = _entry_path(item, manifest)
            if path is None:
                missing.append({"name": item.get("name", ""), "reason": "missing_file"})
                continue
            digest = index.fingerprint(path)
            previous = index.data["hashes"].get(digest)
            same_manifest = bool(previous and (
                previous.get("first_seen_manifest") == owner
                or any(occ.get("manifest") == owner for occ in previous.get("occurrences", []))))
            reference = bool(reference_mode and previous and (
                not source_iteration or previous.get("first_seen_iteration") == source_iteration
                or any(occ.get("iteration") == source_iteration for occ in previous.get("occurrences", []))))
            duplicate = batch_hashes.get(digest) or (previous if previous and not same_manifest and not reference else None)
            name = str(item.get("name") or path.name)
            if duplicate:
                duplicates.append({"name": name, "source_path": str(path), "content_sha256": digest,
                                   "existing": duplicate, "reason": "duplicate_sha256"})
            elif ((name.lower() in batch_names and batch_names[name.lower()] != digest)
                  or (name.lower() in project_names and digest not in project_names[name.lower()])):
                collisions.append({"name": name, "source_path": str(path), "content_sha256": digest,
                                   "existing_sha256": batch_names.get(name.lower()) or sorted(project_names[name.lower()])[0],
                                   "reason": "name_collision"})
            else:
                entry = {**item, **planner.normalize_text_metadata(name, item), "name": name,
                         "source_path": str(path.resolve()), "target_path": str(path.resolve()),
                         "content_sha256": digest, "reused_existing": reference}
                accepted.append(entry)
                batch_hashes[digest] = {"name": name, "source_path": entry["source_path"], "iteration": iteration}
                batch_names[name.lower()] = digest
                reused_count += int(reference)
            if progress:
                progress(10 + 75 * number / max(1, len(inputs)), "Sprawdzam tożsamość obrazów.",
                         detail=f"Sprawdzono {number}/{len(inputs)}. Identyczne: {len(duplicates)}.")
        summary = {"selected": len(inputs), "accepted": len(accepted), "reused": reused_count,
                   "duplicate_sha256": len(duplicates), "name_collisions": len(collisions),
                   "missing_files": len(missing), "duplicates": duplicates, "collisions": collisions,
                   "missing": missing, "unresolved_legacy": unresolved}
        _atomic_write(state_dir / "last_image_identity_report.json", summary)
        if inputs and not accepted:
            _atomic_write(index_path, index.data)
            raise IdentityRejected(summary)
        prepared = {**manifest, "image_identity_schema": MANIFEST_IDENTITY,
                    "selected_images": accepted, "selected_count": len(accepted),
                    "identity_summary": summary, **planner.gt_statistics(accepted)}
        names = sorted({item["name"].strip().lower() for item in accepted})
        prepared["image_set_token"] = (f"iset_{len(names):05d}_" + hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()[:20]) if names else ""
        histogram = Counter()
        for item in accepted:
            histogram.update(item.get("char_histogram") or {})
        prepared["char_histogram"] = dict(histogram)
        prepared["proposal_summary"] = {**(manifest.get("proposal_summary") or {}),
                                         "current_iteration_package_count": len(accepted),
                                         "new_to_project_count": sum(item["content_sha256"] not in index.data["hashes"] for item in accepted),
                                         "duplicate_sha256": len(duplicates), "name_collisions": len(collisions)}
        # This writer is the final boundary, not a UI-only filter.
        hashes = [item["content_sha256"] for item in accepted]
        if len(set(hashes)) != len(hashes):
            raise ValueError("Manifest zawiera powtórzoną zawartość obrazu.")
        _atomic_write(manifest_path, prepared)
        index.register(accepted, manifest_path, iteration)
        stat = manifest_path.stat()
        index.data["manifests"][owner] = [stat.st_size, stat.st_mtime_ns]
        _atomic_write(index_path, index.data)
        return prepared
