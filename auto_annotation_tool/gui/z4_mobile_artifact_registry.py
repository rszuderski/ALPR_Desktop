
"""Z4: read-only discovery of mobile artifacts and append-only export events.

Never rewrites or approves an artifact. Discovery from disk is NOT a proof of
conversion execution, full graph precision, or Android compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
import zipfile

INDEX_SCHEMA = "alpr.desktop.mobile_artifact_catalog.v1"
EVENT_SCHEMA = "alpr.desktop.mobile_export_event.v1"
MANIFEST_CAP = 2 * 1024 * 1024
MAX_PACKAGE_BYTES = 1024 * 1024 * 1024
MODEL_SHA_MZ_CLEAN = "a2f715399bdcb574da899e202388b5324d66f574b8519626cb611bf0e263e792"
RESEARCH_REL = Path("output/mz_day_int8_raw_probe/MZ_DAY_RAW_INT8_CLEAN_EXPLORATORY.tflite")


@dataclass(frozen=True)
class ArtifactRow:
    path: str
    name: str
    schema: str
    kind: str
    role: str
    model_id: str
    variants: str
    size_bytes: int
    modified_utc: str
    state: str
    manifest: dict
    error: str = ""
    # Compact metadata from integrity-checked child sidecars. This is NOT graph validation.
    child_metadata: dict = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


def _roles(models: dict) -> str:
    names = []
    for role, token in (("vehicle", "MP"), ("plate", "MT"), ("character", "MZ")):
        if role in models:
            names.append(token)
    return "+".join(names) or "?"


def _variants(variants) -> str:
    if not isinstance(variants, list):
        return "-"
    items = []
    for row in variants:
        if not isinstance(row, dict):
            continue
        runtime = str(row.get("runtime") or "?").upper()
        precision = str(row.get("precision") or "?").upper()
        value = f"{runtime}/{precision}"
        if value not in items:
            items.append(value)
    return ", ".join(items) if items else "-"


def _child_precision(archive: zipfile.ZipFile, *, role: str, entry: dict) -> tuple[str, dict, str]:
    """Inspect one *sidecar* manifest without unpacking an embedded model.

    The root package references models/<role>/manifest.json and its SHA-256.
    That root SHA is checked before trusting even a UI precision label. The
    nested model graph/package is NOT validated here (separate full check).
    """
    label = {"plate": "MT", "character": "MZ", "vehicle": "MP"}[role]
    entry_variants = _variants(entry.get("variants"))
    sidecar = f"models/{role}/manifest.json"
    declared = str(entry.get("manifest_file") or "")
    if not declared:
        # Legacy packages may carry per-model variants in the root manifest.
        return entry_variants, {"origin": "ROOT_ONLY_UNVERIFIED", "variants": entry_variants}, (
            f"{label}: brak odnośnika do manifestu potomnego" if entry_variants == "-" else ""
        )
    if declared != sidecar:
        return "-", {}, f"{label}: nieobsługiwana ścieżka manifestu potomnego"
    expected_hashes = entry.get("sha256")
    if not isinstance(expected_hashes, dict):
        return "-", {}, f"{label}: brak SHA-256 manifestu potomnego"
    expected_hash = str(expected_hashes.get(sidecar) or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        return "-", {}, f"{label}: nieprawidłowy SHA-256 manifestu potomnego"
    matches = [member for member in archive.infolist() if member.filename == sidecar]
    if len(matches) != 1 or matches[0].is_dir():
        return "-", {}, f"{label}: nie znaleziono dokładnie jednego manifestu potomnego"
    member = matches[0]
    if member.file_size < 2 or member.file_size > MANIFEST_CAP:
        return "-", {}, f"{label}: manifest potomny pusty lub przekracza limit 2 MiB"
    try:
        raw = archive.read(member)
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            return "-", {}, f"{label}: SHA-256 manifestu potomnego nie zgadza się z root"
        manifest = json.loads(raw.decode("utf-8-sig"))
    except (RuntimeError, OSError, ValueError, EOFError, UnicodeError, zipfile.BadZipFile) as exc:
        return "-", {}, f"{label}: błędny manifest potomny ({type(exc).__name__})"
    if not isinstance(manifest, dict) or manifest.get("schema") != "alpr.model.v1":
        return "-", {}, f"{label}: nieprawidłowy schemat manifestu potomnego"
    if manifest.get("role") != role:
        return "-", {}, f"{label}: rola manifestu potomnego niezgodna"
    expected_task = "pose" if role == "plate" else "detect"
    if manifest.get("task") != expected_task:
        return "-", {}, f"{label}: typ zadania manifestu potomnego niezgodny"
    declared_model = str(entry.get("model_id") or "").strip()
    if declared_model and declared_model != str(manifest.get("model_id") or ""):
        return "-", {}, f"{label}: identyfikator modelu potomnego niezgodny"
    variants = _variants(manifest.get("variants"))
    if variants == "-":
        return "-", {}, f"{label}: brak wariantów w zweryfikowanym manifeście potomnym"
    input_meta = manifest.get("input") or {}
    source_meta = manifest.get("source") or {}
    return variants, {
        "origin": "CHILD_SIDECAR_ROOT_SHA_MATCH_NOT_GRAPH_VERIFIED",
        "manifest_file": sidecar,
        "manifest_sha256": expected_hash,
        "model_id": str(manifest.get("model_id") or ""),
        "role": role,
        "task": expected_task,
        "variants": variants,
        "input": input_meta if isinstance(input_meta, dict) else {},
        "checkpoint_sha256": (str(source_meta.get("checkpoint_sha256") or "")
                              if isinstance(source_meta, dict) else ""),
    }, ""


def _compound_precisions(path: Path, models: dict) -> tuple[str, dict, str]:
    """Read only bounded manifest sidecars for roles present in a complete set."""
    parts: list[str] = []
    children: dict = {}
    warnings: list[str] = []
    with zipfile.ZipFile(path, "r") as archive:
        for role in ("vehicle", "plate", "character"):
            entry = models.get(role)
            if not isinstance(entry, dict):
                continue
            label = {"vehicle": "MP", "plate": "MT", "character": "MZ"}[role]
            variants, child, problem = _child_precision(archive, role=role, entry=entry)
            parts.append(f"{label}: {variants}")
            if child:
                children[role] = child
            if problem:
                warnings.append(problem)
    return " | ".join(parts), children, "; ".join(warnings)[:500]


def read_archive(path: Path) -> ArtifactRow:
    """Read *only* root manifest.json; never extract or mutate archive contents."""
    path = Path(path).resolve()
    size, stamp = path.stat().st_size, _mtime(path)
    def broken(message: str) -> ArtifactRow:
        return ArtifactRow(str(path), path.name, "-", "INVALID", "-", "-", "-",
                           size, stamp, "MANIFEST_ERROR", {}, message[:500])
    if size == 0 or size > MAX_PACKAGE_BYTES:
        return broken("Niedozwolony rozmiar pakietu")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            matches = [info for info in archive.infolist() if info.filename == "manifest.json"]
            if len(matches) != 1:
                return broken("Wymagany dokładnie jeden manifest.json w katalogu głównym")
            info = matches[0]
            if info.file_size > MANIFEST_CAP or info.file_size < 2:
                return broken("Manifest przekracza limit lub jest pusty")
            obj = json.loads(archive.read(info).decode("utf-8-sig"))
        if not isinstance(obj, dict):
            return broken("Manifest nie jest obiektem JSON")
        schema = str(obj.get("schema") or "")
        if schema == "alpr.model.v1":
            role = str(obj.get("role") or "")
            if role not in ("plate", "character", "vehicle"):
                return broken("Niepoprawna rola pojedynczego modelu")
            role_label = {"plate": "MT", "character": "MZ", "vehicle": "MP"}[role]
            return ArtifactRow(str(path), path.name, schema, "MODEL", role_label,
                               str(obj.get("model_id") or "-"), _variants(obj.get("variants")),
                               size, stamp, "DISCOVERED_NOT_VERIFIED", obj)
        if schema == "alpr.package.v1":
            models = obj.get("models")
            if not isinstance(models, dict) or "plate" not in models or "character" not in models:
                return broken("Brak kompletu MT+MZ w manifeście pakietu")
            variant_text, children, warning = _compound_precisions(path, models)
            # Explicitly flag untrusted/missing child precision without implying
            # the entire archive has passed Android or graph validation.
            state = ("CHILD_METADATA_INCOMPLETE_NOT_VERIFIED" if warning
                     else "DISCOVERED_NOT_VERIFIED")
            return ArtifactRow(str(path), path.name, schema, "PACKAGE", _roles(models),
                               str(obj.get("package_id") or "-"), variant_text,
                               size, stamp, state, obj, warning, children)
        return broken("Nieznany schemat manifestu")
    except (ValueError, OSError, EOFError, RuntimeError, UnicodeError, zipfile.BadZipFile) as exc:
        return broken(f"Nie można odczytać manifestu: {type(exc).__name__}: {exc}")


def list_managed_packages(root: Path, *, max_files: int = 1200, max_dirs: int = 160) -> tuple[list[ArtifactRow], bool]:
    """Bounded scan; no symlinks or arbitrary Workspace tree traversal."""
    root = Path(root).resolve()
    if not root.is_dir():
        return [], False
    rows = []
    seen = set()
    directories = 0
    truncated = False
    for base, dirs, files in os.walk(root, followlinks=False):
        directories += 1
        if directories > max_dirs:
            truncated = True
            break
        folder = Path(base)
        depth = len(folder.relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if depth < 3 and d not in {"_registry", ".git", "__pycache__"}
                         and not (folder / d).is_symlink())
        for filename in sorted(files):
            if not filename.lower().endswith(".alprmodel"):
                continue
            file = folder / filename
            if file.is_symlink() or not file.is_file():
                continue
            key = str(file.resolve()).casefold()
            if key in seen:
                continue
            seen.add(key)
            if len(rows) >= max_files:
                truncated = True
                break
            try:
                rows.append(read_archive(file))
            except (OSError, ValueError) as exc:
                rows.append(ArtifactRow(str(file), filename, "-", "INVALID", "-", "-", "-", 0,
                                        "-", "READ_ERROR", {}, str(exc)[:500]))
        if truncated:
            break
    return sorted(rows, key=lambda r: (r.modified_utc, r.name.lower()), reverse=True), truncated


def _redirecting_path(path: Path) -> bool:
    """Block symlink/junction redirection in provenance and journal locations.

    Windows directory junctions need an additional check: is_symlink() alone
    need not recognize a junction. If the check itself fails, fail closed.
    """
    try:
        if path.is_symlink():
            return True
        junction = getattr(path, "is_junction", None)
        return bool(junction()) if callable(junction) else False
    except (OSError, ValueError):
        return True


def research_candidate(repo: Path) -> ArtifactRow | None:
    """One known research TFLite; never follow redirects or approve Android."""
    base = Path(repo).resolve()
    path = base
    for part in RESEARCH_REL.parts:
        path /= part
        if _redirecting_path(path):
            return None
    if not path.is_file():
        return None
    try:
        # A non-symlink source must also remain inside the selected repository.
        path.resolve(strict=True).relative_to(base)
    except (OSError, ValueError):
        return None
    try:
        size = path.stat().st_size
        ok = size == 10255456 and sha256_file(path) == MODEL_SHA_MZ_CLEAN
        return ArtifactRow(str(path), path.name, "research-only", "RESEARCH_TFLITE", "MZ",
                           "MZ-DAY CLEAN", "TFLITE/INT8 [statyczny audyt]", size, _mtime(path),
                           "RESEARCH_ONLY_NOT_PACKAGED" if ok else "RESEARCH_SHA_MISMATCH", {},
                           "Brak zatwierdzenia eksportu .alprmodel" if ok else "SHA lub rozmiar niezgodny")
    except OSError:
        return None


def history_file(root: Path) -> Path:
    return Path(root) / "_registry" / "mobile_export_events.v1.jsonl"


def append_event(root: Path, *, operation_id: str, operation: str, outcome: str,
                 artifact_path: str = "", detail: str = "",
                 conversion: dict | None = None) -> dict:
    """Append an observed event. Never infer historic conversion events from file mtimes."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]{8,90}", operation_id):
        raise ValueError("Nieprawidłowy identyfikator operacji")
    if operation not in {"SINGLE_MODEL_EXPORT", "COMPLETE_PACKAGE_EXPORT", "MODEL_CONVERSION"}:
        raise ValueError("Nieznany rodzaj operacji")
    allowed = {"STARTED", "SUCCEEDED", "UNCONFIRMED"} if operation == "MODEL_CONVERSION" else {"STARTED", "SUCCEEDED", "FAILED"}
    if outcome not in allowed:
        raise ValueError("Nieznany wynik operacji")
    if operation == "MODEL_CONVERSION":
        if not isinstance(conversion, dict) or conversion.get("role") not in {"MT", "MZ", "MP"}:
            raise ValueError("Konwersja musi wskazywać rolę MT/MZ/MP")
        if not re.fullmatch(r"[0-9a-f]{64}", str(conversion.get("checkpoint_sha256") or "")):
            raise ValueError("Konwersja musi mieć SHA-256 checkpointu")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{8,90}", str(conversion.get("parent_export_operation_id") or "")):
            raise ValueError("Brak identyfikatora nadrzędnego eksportu")
        if len(json.dumps(conversion, ensure_ascii=False, default=str)) > 6000:
            raise ValueError("Metadane konwersji przekraczają bezpieczny limit")
    elif conversion is not None:
        raise ValueError("Metadane konwersji nie mogą być przypisane do zwykłego eksportu")
    artifact = Path(artifact_path).resolve() if artifact_path else None
    digest = None
    if outcome == "SUCCEEDED":
        if not artifact or not artifact.is_file() or artifact.suffix.lower() != ".alprmodel":
            raise ValueError("Zdarzenie SUCCEEDED wymaga istniejącej paczki .alprmodel")
        digest = sha256_file(artifact)
    event = {
        "schema": EVENT_SCHEMA,
        "event_id": str(uuid.uuid4()),
        "operation_id": operation_id,
        "observed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "operation": operation,
        "outcome": outcome,
        "artifact_path": str(artifact) if artifact else "",
        "artifact_sha256": digest,
        "detail": str(detail or "")[:800],
    }
    if operation == "MODEL_CONVERSION":
        event["conversion"] = conversion
    target = history_file(root)
    if _redirecting_path(target.parent):
        raise ValueError("Katalog _registry jest dowiązaniem lub junction; zapis historii zablokowany")
    target.parent.mkdir(parents=True, exist_ok=True)
    if _redirecting_path(target.parent) or _redirecting_path(target):
        raise ValueError("Rejestr lub katalog _registry przekierowuje zapis; operacja zablokowana")
    # JSONL append-only; truncated final line is ignored by reader, never approved.
    line = (json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with target.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    return event


def load_history(root: Path, *, max_bytes: int = 12 * 1024 * 1024) -> tuple[list[dict], str]:
    file = history_file(root)
    if _redirecting_path(file.parent):
        return [], "Katalog _registry przekierowuje odczyt; historia nie została wczytana"
    if not file.exists():
        return [], "Rejestr nie zawiera jeszcze zdarzeń. Starsze pliki są tylko odnalezione na dysku."
    if _redirecting_path(file) or file.stat().st_size > max_bytes:
        return [], "Rejestr nieczytelny lub zbyt duży — brak wiarygodnej kompletnej historii"
    result = []
    invalid = 0
    with file.open("rb") as stream:
        for line in stream:
            try:
                event = json.loads(line.decode("utf-8"))
                if isinstance(event, dict) and event.get("schema") == EVENT_SCHEMA:
                    result.append(event)
                else:
                    invalid += 1
            except (ValueError, UnicodeError):
                invalid += 1
    note = f"Pominięto {invalid} niepoprawnych wpisów — sprawdź rejestr." if invalid else ""
    return result, note


def library_snapshot(root: Path, repo: Path) -> tuple[list[ArtifactRow], list[dict], str]:
    artifacts, truncated = list_managed_packages(root)
    research = research_candidate(repo)
    if research is not None:
        artifacts.append(research)
    events, note = load_history(root)
    # Paths deliberately chosen outside managed packages are discoverable via event record.
    recorded_paths = {Path(row.path).resolve() for row in artifacts}
    for event in events:
        p = str(event.get("artifact_path") or "")
        if event.get("outcome") != "SUCCEEDED" or not p:
            continue
        path = Path(p)
        if path.suffix.lower() != ".alprmodel" or not path.is_file() or path.is_symlink():
            continue
        resolved = path.resolve()
        if resolved not in recorded_paths:
            try:
                artifacts.append(read_archive(resolved))
                recorded_paths.add(resolved)
            except (OSError, ValueError):
                pass
    # A historical ZIP is never an export event. For NEW logged exports, compare
    # the observed archive SHA with the present bytes, without claiming graph proof.
    expected_by_path = {}
    for event in events:
        if event.get("outcome") == "SUCCEEDED" and event.get("artifact_path") and event.get("artifact_sha256"):
            expected_by_path[str(Path(event["artifact_path"]).resolve()).casefold()] = event["artifact_sha256"]
    checked = []
    for row in artifacts:
        pinned = expected_by_path.get(str(Path(row.path).resolve()).casefold())
        if pinned and row.kind in ("MODEL", "PACKAGE") and row.state in (
                "DISCOVERED_NOT_VERIFIED", "CHILD_METADATA_INCOMPLETE_NOT_VERIFIED"):
            try:
                actual = sha256_file(Path(row.path))
                row = replace(row, state=("REGISTERED_SHA_MATCH_NOT_VERIFIED" if actual == pinned
                                          else "REGISTERED_SHA_MISMATCH"))
            except OSError:
                row = replace(row, state="REGISTERED_SHA_UNREADABLE")
        checked.append(row)
    if truncated:
        note = "Skan pakietów przekroczył limit — spis jest niekompletny. " + note
    # JSONL append order is authoritative when several events share the same
    # second. Do not reverse it here: operation summaries must see STARTED before
    # SUCCEEDED even if their observed_utc values are identical. The GUI decides
    # the display order separately, without changing this audit stream.
    return checked, events, note
