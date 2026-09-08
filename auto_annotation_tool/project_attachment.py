"""Inspect and attach a copied project without recreating its workspace."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET

from .config import CONFIG, logger

PROJECT_SCHEMA = "alpr.project.v1"
DESCRIPTOR = Path("_campaign_state/project.json")
_MARKERS = (DESCRIPTOR, Path("_campaign_state/artifact_registry.json"), Path("5_training_runs/training_history.json"))
_SKIP_DIRS = {"images", "labels", "weights", "_ipc", "__pycache__", ".git", "attachments", "1_raw_images"}


class ProjectAttachmentError(ValueError):
    pass


@dataclass
class ProjectAttachmentPlan:
    root: Path
    name: str
    project_data: dict
    state_source: str
    source_roots: list[str]
    unresolved_paths: list[str]
    training_runs: int
    preview_count: int
    changes: dict[Path, bytes] = field(repr=False)
    originals: dict[Path, bytes | None] = field(repr=False)
    source_copies: dict[Path, Path] = field(default_factory=dict, repr=False)
    source_signatures: dict[Path, tuple[int, int]] = field(default_factory=dict, repr=False)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_bytes(data) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _read_json(path: Path, default=None):
    if not path.is_file():
        return copy.deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ProjectAttachmentError(f"Nie można odczytać {path.name}: {exc}") from exc


def _checked_root(folder: Path, projects_dir: Path) -> Path:
    try:
        root = Path(folder).resolve(strict=True)
    except OSError as exc:
        raise ProjectAttachmentError(f"Nie można otworzyć katalogu projektu: {folder}") from exc
    if not root.is_dir() or root.parent != projects_dir.resolve():
        raise ProjectAttachmentError("Wybierz katalog projektu znajdujący się bezpośrednio w katalogu projektów programu.")
    if not (root / "_campaign_state").resolve().is_relative_to(root):
        raise ProjectAttachmentError("Katalog stanu projektu wskazuje poza projekt.")
    if not any((root / marker).is_file() for marker in _MARKERS):
        raise ProjectAttachmentError("Katalog nie zawiera opisu projektu, rejestru artefaktów ani historii treningów.")
    if any((root / marker).is_file() and not (root / marker).resolve().is_relative_to(root) for marker in _MARKERS):
        raise ProjectAttachmentError("Opis albo historia projektu wskazują poza jego katalog.")
    return root


def _metadata_files(root: Path):
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in _SKIP_DIRS and not (Path(parent) / name).is_symlink())
        for name in sorted(files):
            path = Path(parent) / name
            if path.suffix.lower() not in {".json", ".yaml", ".yml", ".xml"}:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ProjectAttachmentError(f"Metadane projektu wskazują poza jego katalog: {path}")
            yield path
        # Checkpoint sidecars are metadata too; never read or modify .pt files.
        weights = Path(parent) / "weights"
        if weights.is_dir() and not weights.is_symlink():
            for path in sorted(weights.glob("*.json")):
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ProjectAttachmentError(f"Metadane checkpointu wskazują poza projekt: {path}")
                yield path


def _looks_absolute(value: str) -> bool:
    return value.startswith(("/", "\\\\")) or (len(value) >= 3 and value[0].isalpha() and value[1] == ":" and value[2] in "/\\")


def _walk_strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and _looks_absolute(key):
                yield key
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str) and _looks_absolute(value):
        yield value


def _path_text(value: str) -> str:
    return value.replace("\\", "/").rstrip("/")


def _discover_roots(values, root: Path, descriptor: dict) -> list[str]:
    names = {root.name}
    if descriptor.get("project", {}).get("folder_name"):
        names.add(str(descriptor["project"]["folder_name"]))
    roots = set()
    if descriptor.get("root"):
        roots.add(_path_text(str(descriptor["root"])))
    patterns = [re.compile(r"/" + re.escape(name) + r"(?=/|$|\|)", flags=re.IGNORECASE) for name in names]
    for value in set(values):
        normalized = _path_text(value)
        if not _looks_absolute(normalized):
            continue
        for pattern in patterns:
            match = pattern.search(normalized)
            if match:
                roots.add(normalized[:match.end()])
    return sorted(roots, key=len, reverse=True)


class _Relocator:
    def __init__(self, root: Path, source_roots: list[str], workspace: Path, values=()):
        self.root, self.workspace = root, workspace
        self.roots = [(old, old.casefold()) for old in source_roots]
        self.workspaces = []
        for old in source_roots:
            marker = re.search(r"/9_projects/", old, flags=re.IGNORECASE)
            if marker:
                self.workspaces.append(old[:marker.start()])
        # A project may have moved more than once; unresolved global resources
        # can still refer to an instance older than the descriptor's root.
        for value in values:
            normalized = _path_text(value)
            marker = normalized.casefold().find("/workspace/")
            if marker >= 0:
                self.workspaces.append(normalized[:marker + len("/workspace")])
        self.workspaces = sorted(set(self.workspaces), key=len, reverse=True)
        self.memo = {}
        self.existence = {}
        self.unresolved = set()
        self.aliases = {}

    def exists(self, path):
        if path not in self.existence:
            self.existence[path] = path.exists()
        return self.existence[path]

    def __call__(self, value):
        if isinstance(value, dict):
            return {self(key): self(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self(item) for item in value]
        if not isinstance(value, str) or not _looks_absolute(value):
            return value
        normalized = _path_text(value)
        if normalized.casefold() in self.aliases:
            return self.aliases[normalized.casefold()]
        if normalized not in self.memo:
            self.memo[normalized] = self.path(normalized)
        return self.memo[normalized] or value

    def path(self, normalized):
        folded = normalized.casefold()
        for old, old_folded in self.roots:
            if folded == old_folded or folded.startswith(old_folded + "/"):
                suffix, separator, token = normalized[len(old):].lstrip("/").partition("|")
                if ".." in PurePosixPath(suffix).parts:
                    raise ProjectAttachmentError(f"Ścieżka w metadanych wychodzi poza projekt: {normalized}")
                return str(self.root / suffix) + (separator + token if separator else "")
        # Check parent directories once. A missing image pool should not trigger
        # thousands of repeated stat/resolve calls or thousands of UI warnings.
        for old in self.workspaces:
            if not folded.startswith(old.casefold() + "/"):
                continue
            relative, separator, token = normalized[len(old) + 1:].partition("|")
            parts = PurePosixPath(relative).parts
            if ".." in parts:
                self.unresolved.add(normalized)
                return None
            candidate = self.workspace
            for index, part in enumerate(parts):
                candidate /= part
                if not self.exists(candidate):
                    self.unresolved.add(old + "/" + "/".join(parts[:index + 1]))
                    return None
            if candidate.resolve().is_relative_to(self.workspace.resolve()):
                return str(candidate) + (separator + token if separator else "")
            self.unresolved.add(normalized)
        return None


def _regular_files(directory: Path):
    try:
        with os.scandir(directory) as entries:
            return {entry.name: (Path(entry.path), entry.stat(follow_symlinks=False))
                    for entry in entries if entry.is_file(follow_symlinks=False)}
    except FileNotFoundError:
        return {}


def _recover_approved_sources(root, payloads, originals, rebase):
    """Retain images whose cache explicitly references this ApprovedSet.

    The generated char source can be cleared on its next rebuild. Registering
    those disposable paths as original sources would lose the only image copy.
    """
    approved_path = root / "_campaign_state/plate_approved_set.json"
    cache_root = root / "_campaign_state/char_effective_source"
    state = payloads.get(cache_root / "source_state.json") or {}
    approved = payloads.get(approved_path) or {}
    token = str(state.get("approved_manifest_token") or "").split("|")
    if len(token) != 3 or not token[2].isdigit() or int(token[2]) != len(originals.get(approved_path, b"")):
        return {}, {}
    if Path(rebase(token[0])) != approved_path or not (cache_root / "annotations.xml").is_file():
        return {}, {}
    try:
        nodes = {node.get("name"): node for node in ET.parse(cache_root / "annotations.xml").getroot().findall("image")}
    except (OSError, ET.ParseError):
        return {}, {}
    image_dir = cache_root / "images"
    if not image_dir.resolve().is_relative_to(root):
        return {}, {}
    cached_images = _regular_files(image_dir)
    archive = root / "1_raw_images/attached_approved_sources"
    archived_images = _regular_files(archive)
    local_sources = {archive: archived_images}
    copies, signatures = {}, {}
    entries = approved.get("entries") or {}
    for entry in entries.values() if isinstance(entries, dict) else entries:
        image_name = str(entry.get("image_name") or "")
        original = str(entry.get("source_image_path") or "")
        if not original or not image_name or Path(image_name).name != image_name:
            continue
        mapped = rebase(original)
        mapped_path = Path(mapped)
        if mapped_path.is_relative_to(root) and mapped_path.parent != image_dir:
            if mapped_path.parent not in local_sources:
                local_sources[mapped_path.parent] = _regular_files(mapped_path.parent)
            if mapped_path.name in local_sources[mapped_path.parent]:
                continue
        node = nodes.get(image_name)
        image_record = cached_images.get(image_name)
        if node is None or image_record is None:
            continue
        source, stat = image_record
        if str(node.get("width")) != str(entry.get("width")) or str(node.get("height")) != str(entry.get("height")):
            continue
        # Verify the cache still describes the same plate geometry, not just a
        # matching filename from another collection.
        polygons = [plate.get("polygon") or [] for plate in entry.get("plates", [])]
        cached = [polygon.get("points", "") for polygon in node.findall("polygon") if polygon.get("label") == "plate"]
        try:
            cached_points = [[[float(c) for c in point.split(",")] for point in text.split(";")] for text in cached]
            compatible = len(polygons) == len(cached_points) and bool(polygons) and all(
                len(a) == len(b) and all(abs(float(x) - float(y)) <= 0.05 for p, q in zip(a, b) for x, y in zip(p, q))
                for a, b in zip(polygons, cached_points))
        except (TypeError, ValueError):
            compatible = False
        if not compatible:
            continue
        destination = archive / image_name
        if image_name in archived_images:
            if not os.path.samefile(source, destination):
                raise ProjectAttachmentError(f"Katalog zabezpieczonych źródeł zawiera już inny plik: {destination.name}")
        else:
            copies[destination] = source
        signatures[source] = (stat.st_size, stat.st_mtime_ns)
        rebase.aliases[_path_text(original).casefold()] = str(destination)
    return copies, signatures


def _legacy_project_data(manager, root: Path, artifact: dict, history: list[dict], training: dict) -> tuple[str, dict]:
    name = str(artifact.get("project") or re.sub(r"_[0-9a-fA-F]{6}$", "", root.name)).strip()
    data = manager._get_default_project_template(name)
    iterations = [int(key) for key in artifact.get("iteration_state", {}) if str(key).isdigit()]
    iterations += [int(event.get("iteration", 0) or 0) for event in history]
    data["current_iteration"] = max(iterations or [1])
    current = [event for event in history if int(event.get("iteration", 0) or 0) == data["current_iteration"]
               and event.get("status") in {"ok", "success"}]
    if history:
        data["created_at"] = str(history[0].get("created_at") or data["created_at"])
    for event in current:
        step = int(event.get("step", 0) or 0)
        if step in {1, 2, 3, 4}:
            data["current_step"] = step
        path = (event.get("details") or {}).get("path")
        if path in {"plate_training", "char_from_images", "char_from_ready_plates"}:
            data["iteration_path"] = path
            data["iteration_target"] = "plate" if path == "plate_training" else "char"
    state = artifact.get("iteration_state", {}).get(str(data["current_iteration"]), {})
    work = state.get("t06_work_session") or {}
    if work.get("work_area") == "z3":
        data["current_step"] = 3
        data["step3_substep"] = min(3, max(1, int(work.get("substep", 1) or 1)))
        data["iteration_target"] = "char"
        data["step3_preview_dir"] = str(work.get("preview_dir") or "")
    if data["current_step"] >= 2:
        data["step1_status"] = "approved"
    if data["current_step"] >= 3 and data["iteration_path"] != "char_from_ready_plates":
        data["step2_status"] = "approved"
    contracts = state.get("t06_contracts") or {}
    data["step3_stage2_done"] = bool((contracts.get("pz2_char_boxes") or {}).get("fulfilled"))
    data["project_start_mode"] = "assets"
    return name, data


def inspect_existing_project(manager, folder: Path, *, projects_dir: Path | None = None) -> ProjectAttachmentPlan:
    projects_dir = Path(projects_dir or CONFIG.DIR_9_PROJECTS).resolve()
    root = _checked_root(folder, projects_dir)
    descriptor = _read_json(root / DESCRIPTOR, {})
    if not isinstance(descriptor, dict) or (descriptor and (descriptor.get("schema") != PROJECT_SCHEMA or not isinstance(descriptor.get("project"), dict))):
        raise ProjectAttachmentError("Nieobsługiwany albo uszkodzony opis projektu project.json.")
    artifact = _read_json(root / "_campaign_state/artifact_registry.json", {})
    training = _read_json(root / "5_training_runs/training_history.json", {})
    if not isinstance(artifact, dict) or not isinstance(training, dict):
        raise ProjectAttachmentError("Rejestr projektu i historia treningów muszą być obiektami JSON.")
    history_file = root / "_campaign_state/project_history.jsonl"
    try:
        history = [json.loads(line) for line in history_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()] if history_file.exists() else []
    except (OSError, ValueError) as exc:
        raise ProjectAttachmentError(f"Nie można odczytać historii projektu: {exc}") from exc
    if any(not isinstance(event, dict) for event in history):
        raise ProjectAttachmentError("Historia projektu zawiera nieprawidłowe wpisy.")
    if not descriptor and not artifact.get("project") and not training.get("runs") and not history:
        raise ProjectAttachmentError("W katalogu brakuje danych pozwalających rozpoznać istniejący projekt.")
    if descriptor:
        name, data = str(descriptor.get("name") or root.name), copy.deepcopy(descriptor["project"])
    else:
        name, data = _legacy_project_data(manager, root, artifact, history, training)
    data["folder_name"] = root.name
    data = manager._ensure_project_defaults(data)
    payloads, originals = {}, {}
    for file in _metadata_files(root):
        raw = file.read_bytes()
        originals[file] = raw
        if file.suffix == ".json":
            payloads[file] = _read_json(file)
    values = list(_walk_strings(data)) + list(_walk_strings(history))
    for payload in payloads.values():
        values.extend(_walk_strings(payload))
    source_roots = _discover_roots(values, root, descriptor)
    rebase = _Relocator(root, source_roots, projects_dir.parent, values)
    source_copies, source_signatures = _recover_approved_sources(root, payloads, originals, rebase)
    data = rebase(data)
    changes = {}
    for file, payload in payloads.items():
        if file == root / DESCRIPTOR:
            continue
        relocated = rebase(payload)
        if relocated != payload:
            changes[file] = _json_bytes(relocated)
    if history_file.exists():
        originals[history_file] = history_file.read_bytes()
        moved_history = rebase(history)
        if moved_history != history:
            changes[history_file] = ("\n".join(json.dumps(event, ensure_ascii=False) for event in moved_history) + "\n").encode("utf-8")
    # YAML/XML keep their formatting. Only recognized project-root prefixes move.
    for file, raw in originals.items():
        if file.suffix in {".json", ".jsonl"}:
            continue
        text = raw.decode("utf-8-sig")
        moved = text
        for old in source_roots:
            for prefix in (old, old.replace("/", "\\")):
                moved = re.sub(re.escape(prefix) + r"(?=[/\\]|[\"'\s<]|$)", lambda _: root.as_posix(), moved, flags=re.IGNORECASE)
        if moved != text:
            changes[file] = moved.encode("utf-8")
    # View caches contain signatures for the source machine; rebuild them lazily.
    cache = root / "_campaign_state/wizard_view_cache.json"
    if cache in originals:
        changes[cache] = b"{}\n"
    preview_count = 0
    preview = Path(data.get("step3_preview_dir") or root / "__no_preview__")
    if preview.is_relative_to(root) and (preview / "metadata.json").is_file():
        preview_count = len(_read_json(preview / "metadata.json", {}))
        extraction = rebase(_read_json(preview / "extract_manifest.json", {}))
        source = extraction.get("source") or {}
        data["step3_stage1_done"] = bool(preview_count and extraction.get("status") == "completed")
        for source_key, project_key in (("annotation_run_dir", "step3_extract_annotation_run_dir"),
                                        ("xml_path", "step3_extract_xml_path"), ("images_dir", "step3_extract_images_dir")):
            data[project_key] = str(source.get(source_key) or "")
    unresolved = sorted({path.casefold(): path for path in rebase.unresolved}.values())
    return ProjectAttachmentPlan(root, name, data, "descriptor" if descriptor else "recovered",
                                 source_roots, unresolved, len(training.get("runs", {})), preview_count,
                                 changes, originals, source_copies, source_signatures)


def save_project_descriptor(manager, name: str) -> None:
    data = manager.state.get("projects", {}).get(name)
    if not isinstance(data, dict):
        return
    root = Path(CONFIG.DIR_9_PROJECTS) / str(data.get("folder_name") or "")
    if not root.is_dir() or root.resolve().parent != Path(CONFIG.DIR_9_PROJECTS).resolve():
        return
    payload = {"schema": PROJECT_SCHEMA, "name": name, "root": str(root.resolve()), "project": data}
    encoded = _json_bytes(payload)
    path = root / DESCRIPTOR
    if not path.resolve().is_relative_to(root.resolve()):
        raise ProjectAttachmentError("Opis projektu wskazuje poza katalog projektu.")
    if not path.exists() or path.read_bytes() != encoded:
        _atomic_bytes(path, encoded)


def list_attachable_projects(manager, *, projects_dir: Path | None = None) -> list[Path]:
    parent = Path(projects_dir or CONFIG.DIR_9_PROJECTS)
    known = {str(p.get("folder_name", "")).casefold() for p in manager.state.get("projects", {}).values()}
    if not parent.is_dir():
        return []
    return sorted((p for p in parent.iterdir() if p.is_dir() and not p.is_symlink()
                   and p.name.casefold() not in known and any((p / marker).is_file() for marker in _MARKERS)), key=lambda p: p.name.casefold())


def attach_existing_project(manager, plan: ProjectAttachmentPlan, *, name: str | None = None,
                            projects_dir: Path | None = None) -> dict:
    parent = Path(projects_dir or CONFIG.DIR_9_PROJECTS).resolve()
    root = _checked_root(plan.root, parent)
    name = str(name or plan.name).strip()
    if not name or len(name) > 160 or any(ord(char) < 32 for char in name):
        raise ProjectAttachmentError("Podaj nazwę projektu (1–160 znaków).")
    registry_path = Path(manager.state_file)
    registry_before = registry_path.read_bytes() if registry_path.exists() else None
    try:
        disk_state = json.loads(registry_before.decode("utf-8-sig")) if registry_before is not None else copy.deepcopy(manager.state)
    except ValueError as exc:
        raise ProjectAttachmentError("Nie można odczytać rejestru projektów.") from exc
    projects = disk_state.get("projects")
    if not isinstance(projects, dict):
        raise ProjectAttachmentError("Rejestr projektów jest uszkodzony; podłączenie przerwane.")
    for known_name, known in projects.items():
        if known_name.casefold() == name.casefold():
            raise ProjectAttachmentError(f"Nazwa projektu jest już zajęta: {known_name}.")
        if str(known.get("folder_name", "")).casefold() == root.name.casefold():
            raise ProjectAttachmentError(f"Ten katalog jest już podłączony jako: {known_name}.")
    for file, original in plan.originals.items():
        if not file.resolve().is_relative_to(root) or not file.is_file() or file.read_bytes() != original:
            raise ProjectAttachmentError("Pliki projektu zmieniły się od sprawdzenia. Sprawdź katalog ponownie.")
    source_directories = {source.parent for source in plan.source_signatures}
    if any(not parent.resolve().is_relative_to(root) for parent in source_directories):
        raise ProjectAttachmentError("Katalog obrazów źródłowych wskazuje poza projekt.")
    source_records = {parent: _regular_files(parent) for parent in source_directories}
    for source, signature in plan.source_signatures.items():
        image_record = source_records[source.parent].get(source.name)
        if image_record is None or (image_record[1].st_size, image_record[1].st_mtime_ns) != signature:
            raise ProjectAttachmentError("Obrazy źródłowe zmieniły się od sprawdzenia. Sprawdź projekt ponownie.")
    destinations = {path.parent for path in plan.source_copies}
    if any(not parent.resolve().is_relative_to(root) for parent in destinations):
        raise ProjectAttachmentError("Katalog zabezpieczonych obrazów wskazuje poza projekt.")
    existing_names = {parent: {p.name for p in parent.iterdir()} if parent.is_dir() else set() for parent in destinations}
    for destination, source in plan.source_copies.items():
        if source not in plan.source_signatures or destination.name in existing_names[destination.parent]:
            raise ProjectAttachmentError("Nie można bezpiecznie zabezpieczyć obrazów źródłowych projektu.")
    data = copy.deepcopy(plan.project_data)
    data["folder_name"] = root.name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    backup = root / "_campaign_state/attachments" / stamp
    backup.mkdir(parents=True)
    report = {"schema": "alpr.project.attachment.v1", "name": name, "source_roots": plan.source_roots,
              "state_source": plan.state_source, "unresolved_paths": plan.unresolved_paths, "files": [], "status": "prepared",
              "retained_images": [str(path.relative_to(root)) for path in plan.source_copies]}
    if registry_before is not None:
        (backup / "campaigns_registry.before.json").write_bytes(registry_before)
    changes = dict(plan.changes)
    descriptor = root / DESCRIPTOR
    changes[descriptor] = _json_bytes({"schema": PROJECT_SCHEMA, "name": name, "root": str(root), "project": data})
    originals = {p: p.read_bytes() if p.exists() else None for p in changes}
    for file, original in originals.items():
        if not file.resolve().is_relative_to(root):
            raise ProjectAttachmentError("Plik do zmiany znajduje się poza projektem.")
        relative = file.relative_to(root)
        if original is not None:
            target = backup / "files" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original)
        report["files"].append({"path": relative.as_posix(),
                                "before_sha256": hashlib.sha256(original).hexdigest() if original is not None else None,
                                "after_sha256": hashlib.sha256(changes[file]).hexdigest()})
    report_path = backup / "attachment.json"
    _atomic_bytes(report_path, _json_bytes(report))
    written, created_images = [], []
    try:
        for destination, source in plan.source_copies.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
                created_images.append(destination)
            except FileExistsError:
                raise
            except OSError:
                with destination.open("xb") as output:
                    created_images.append(destination)
                    with source.open("rb") as input_stream:
                        shutil.copyfileobj(input_stream, output)
                shutil.copystat(source, destination)
        for file, content in changes.items():
            _atomic_bytes(file, content)
            written.append(file)
        # Register last; a failed relocation cannot create an unusable list entry.
        registry_now = registry_path.read_bytes() if registry_path.exists() else None
        if registry_now != registry_before:
            raise ProjectAttachmentError("Rejestr projektów zmienił się w trakcie podłączania. Spróbuj ponownie.")
        projects[name] = data
        _atomic_bytes(registry_path, _json_bytes(disk_state))
    except Exception:
        for file in reversed(written):
            original = originals[file]
            if original is None:
                file.unlink(missing_ok=True)
            else:
                _atomic_bytes(file, original)
        for image_path in created_images:
            image_path.unlink(missing_ok=True)
        report["status"] = "rolled_back"
        _atomic_bytes(report_path, _json_bytes(report))
        raise
    manager.state["projects"] = projects
    manager._registered_project_names = set(projects)
    from .project_cache import PROJECT_CACHE
    for file in changes:
        PROJECT_CACHE.invalidate_json(file)
    for key, cache in list(vars(manager).items()):
        if key.endswith("_cache") and isinstance(cache, dict):
            cache.clear()
    report["status"] = "attached"
    try:
        _atomic_bytes(report_path, _json_bytes(report))
    except OSError as exc:
        logger.warning(f"Projekt podłączono, ale nie udało się uzupełnić raportu {report_path}: {exc}")
    return {"name": name, "root": root, "backup_dir": backup, "changed_files": len(changes),
            "unresolved_paths": plan.unresolved_paths, "state_source": plan.state_source,
            "retained_images": len(created_images)}
