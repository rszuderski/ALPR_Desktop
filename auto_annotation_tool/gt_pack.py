"""Portable, merge-safe ALPR ground-truth pack v1.

Canonical identity is independent from project/run paths and filenames.
Ground-truth and geometry changes are immutable revisions arranged in a DAG.
Mutable plate records only point at the revision set/heads. Merge is therefore
lossless: concurrent edits remain as explicit heads instead of last-write-wins.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps

from .registration_text import NORMALIZATION_POLICY, normalize_registration
from .plate_ground_truth import normalize_plate_layout_gt


PACK_SCHEMA = "alpr.gt.pack.v1"
IMAGE_SCHEMA = "alpr.gt.image.v1"
PLATE_SCHEMA = "alpr.gt.plate.v1"
GEOMETRY_SCHEMA = "alpr.gt.geometry.v1"
REVISION_SCHEMA = "alpr.gt.revision.v1"
LAYOUT_REVISION_SCHEMA = "alpr.gt.layout-revision.v1"
CONFLICT_SCHEMA = "alpr.gt.conflict.v1"

IMAGE_ID_PREFIX = "img-sha256-"
PLATE_ID_PREFIX = "plate-ann-"
GEOMETRY_ID_PREFIX = "geom-sha256-"
REVISION_ID_PREFIX = "rev-sha256-"
LAYOUT_REVISION_ID_PREFIX = "layout-rev-sha256-"
CONFLICT_ID_PREFIX = "conflict-sha256-"

DEFAULT_PRODUCER = "auto_annotation_tool"
LOCK_FILENAME = ".pack.lock"
LOCK_STALE_SECONDS = 3600.0


class GTPackError(RuntimeError):
    """Base error for ALPR GT Pack operations."""


class GTPackFormatError(GTPackError):
    """Raised when a pack or record violates the v1 contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _read_json(path: Path, default=None):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default
    except Exception as exc:
        raise GTPackFormatError(f"Nie można odczytać JSON: {path}: {exc}") from exc


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except Exception:
                pass
        os.replace(temp_path, path)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


def _unique_sorted(values: Iterable[Any]) -> list[str]:
    return sorted(
        {
            str(value or "").strip()
            for value in (values or [])
            if str(value or "").strip()
        }
    )


def _safe_record_filename(record_id: str) -> str:
    value = str(record_id or "").strip()
    if not value:
        raise GTPackFormatError("Pusty identyfikator rekordu.")
    safe = "".join(
        char if (char.isalnum() or char in "._-") else "_"
        for char in value
    )
    if not safe:
        raise GTPackFormatError(
            f"Nieprawidłowy identyfikator rekordu: {record_id!r}"
        )
    return f"{safe}.json"


def _normalized_record(record: dict) -> dict:
    return json.loads(_canonical_json_bytes(record).decode("utf-8"))


def _deterministic_choice(left: dict, right: dict) -> dict:
    return (
        left
        if _canonical_json_bytes(left) <= _canonical_json_bytes(right)
        else right
    )


def new_plate_id() -> str:
    return f"{PLATE_ID_PREFIX}{uuid.uuid4()}"


def normalize_plate_id(value: Any) -> str:
    prepared = str(value or "").strip()
    return prepared or new_plate_id()


def normalize_polygon(
    points,
    *,
    image_width: int,
    image_height: int,
) -> list[list[float]]:
    try:
        width = float(image_width)
        height = float(image_height)
    except Exception as exc:
        raise GTPackFormatError("Nieprawidłowe wymiary obrazu.") from exc

    if width <= 0 or height <= 0:
        raise GTPackFormatError("Wymiary obrazu muszą być dodatnie.")

    prepared = list(points or [])
    if len(prepared) < 4:
        raise GTPackFormatError(
            "Polygon tablicy musi zawierać co najmniej 4 punkty."
        )

    normalized: list[list[float]] = []
    for point in prepared[:4]:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise GTPackFormatError("Nieprawidłowy punkt polygonu.")
        x = max(0.0, min(1.0, float(point[0]) / width))
        y = max(0.0, min(1.0, float(point[1]) / height))
        normalized.append([round(x, 8), round(y, 8)])
    return normalized


def denormalize_polygon(
    points,
    *,
    image_width: int,
    image_height: int,
) -> list[list[float]]:
    width = float(image_width)
    height = float(image_height)
    if width <= 0 or height <= 0:
        raise GTPackFormatError("Wymiary obrazu muszą być dodatnie.")

    result = []
    for point in list(points or [])[:4]:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise GTPackFormatError("Nieprawidłowy punkt polygonu.")
        result.append([
            float(point[0]) * width,
            float(point[1]) * height,
        ])
    if len(result) < 4:
        raise GTPackFormatError("Polygon tablicy musi zawierać 4 punkty.")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(image: Image.Image) -> str:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    width, height = rgb.size
    digest = hashlib.sha256()
    digest.update(b"alpr.gt.rgb.v1\0")
    digest.update(int(width).to_bytes(4, "big", signed=False))
    digest.update(int(height).to_bytes(4, "big", signed=False))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _flattened_pixels(image: Image.Image):
    getter = getattr(image, "get_flattened_data", None)
    if callable(getter):
        return list(getter())
    # Compatibility with older Pillow versions.
    return list(image.getdata())


def _dhash64(image: Image.Image) -> str:
    prepared = ImageOps.exif_transpose(image).convert("L")
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    reduced = prepared.resize((9, 8), resample=resample)
    pixels = _flattened_pixels(reduced)
    value = 0
    bit = 0
    for row in range(8):
        base = row * 9
        for col in range(8):
            if pixels[base + col] > pixels[base + col + 1]:
                value |= (1 << bit)
            bit += 1
    return f"{value:016x}"


def fingerprint_image_identity(path: Path | str) -> dict:
    """Exact file identity and oriented dimensions, without decoding pixel hashes.

    PZ1 already decodes the image for cropping. Its provenance only needs the
    exact identity; recovery fingerprints remain part of fingerprint_image().
    """
    image_path = Path(path)
    source_file_sha256 = _file_sha256(image_path)
    try:
        with Image.open(image_path) as source:
            width, height = source.size
            if source.getexif().get(274) in {5, 6, 7, 8}:
                width, height = height, width
    except Exception as exc:
        raise GTPackError(f"Nie można odczytać obrazu {image_path}: {exc}") from exc
    return {
        "image_id": f"{IMAGE_ID_PREFIX}{source_file_sha256}",
        "source_file_sha256": source_file_sha256,
        "width": int(width),
        "height": int(height),
    }


def fingerprint_image(path: Path | str) -> dict:
    """Return cross-program exact identity plus recovery fingerprints.

    `image_id` is based on exact source-file bytes. This is intentionally easy
    to reproduce in Python/Kotlin/Java/C++ and survives rename/move/copy 1:1.
    Re-encoding creates a new exact image_id; pixel/perceptual hashes may then
    be used only to propose reconciliation, never to silently merge GT.
    """
    image_path = Path(path)
    if not image_path.exists() or not image_path.is_file():
        raise GTPackError(f"Brak obrazu: {image_path}")

    source_file_sha256 = _file_sha256(image_path)
    try:
        with Image.open(image_path) as source:
            transposed = ImageOps.exif_transpose(source)
            width, height = transposed.size
            pixel_sha256 = _pixel_sha256(source)
            dhash64 = _dhash64(source)
    except Exception as exc:
        raise GTPackError(
            f"Nie można odczytać obrazu {image_path}: {exc}"
        ) from exc

    return {
        "image_id": f"{IMAGE_ID_PREFIX}{source_file_sha256}",
        "source_file_sha256": source_file_sha256,
        "pixel_sha256": pixel_sha256,
        "perceptual_dhash64": dhash64,
        "width": int(width),
        "height": int(height),
    }


def _geometry_core(
    *,
    plate_id: str,
    points: list[list[float]],
    parents: Iterable[str] = (),
) -> dict:
    return {
        "plate_id": str(plate_id),
        "type": "polygon4-normalized",
        "points": [
            [round(float(point[0]), 8), round(float(point[1]), 8)]
            for point in list(points or [])[:4]
        ],
        "parents": _unique_sorted(parents),
    }


def make_geometry_record(
    *,
    plate_id: str,
    points: list[list[float]],
    parents: Iterable[str] = (),
) -> dict:
    core = _geometry_core(
        plate_id=plate_id,
        points=points,
        parents=parents,
    )
    geometry_id = f"{GEOMETRY_ID_PREFIX}{_content_sha256(core)}"
    return {
        "schema": GEOMETRY_SCHEMA,
        "geometry_id": geometry_id,
        **core,
    }


def _revision_core(
    *,
    plate_id: str,
    operation: str,
    text: str | None,
    parents: Iterable[str],
    source: str,
) -> dict:
    operation_key = str(operation or "").strip().lower()
    if operation_key not in {"set", "clear"}:
        raise GTPackFormatError(
            f"Nieobsługiwana operacja GT: {operation!r}"
        )

    normalized_value = (
        normalize_registration(text)
        if operation_key == "set"
        else None
    )
    if operation_key == "set" and not normalized_value:
        raise GTPackFormatError(
            "Operacja SET wymaga niepustego GT."
        )

    return {
        "plate_id": str(plate_id),
        "operation": operation_key,
        "value": normalized_value,
        "normalization_policy": NORMALIZATION_POLICY,
        "parents": _unique_sorted(parents),
        "source": str(source or "manual").strip() or "manual",
    }


def make_revision_record(
    *,
    plate_id: str,
    text: str | None = None,
    operation: str = "set",
    parents: Iterable[str] = (),
    source: str = "manual",
    producer: str = DEFAULT_PRODUCER,
) -> dict:
    core = _revision_core(
        plate_id=plate_id,
        operation=operation,
        text=text,
        parents=parents,
        source=source,
    )
    revision_id = f"{REVISION_ID_PREFIX}{_content_sha256(core)}"
    return {
        "schema": REVISION_SCHEMA,
        "revision_id": revision_id,
        **core,
        "observed_by": _unique_sorted([producer]),
    }


def _layout_revision_core(
    *,
    plate_id: str,
    operation: str,
    layout: str | None,
    parents: Iterable[str],
    source: str,
) -> dict:
    operation_key = str(operation or "").strip().lower()
    if operation_key not in {"set", "clear"}:
        raise GTPackFormatError(f"Nieobsługiwana operacja layout GT: {operation!r}")
    normalized = (
        normalize_plate_layout_gt(layout, default="")
        if operation_key == "set"
        else None
    )
    if operation_key == "set" and not normalized:
        raise GTPackFormatError("Operacja SET layout wymaga single_row albo two_row.")
    return {
        "plate_id": str(plate_id),
        "operation": operation_key,
        "value": normalized,
        "parents": _unique_sorted(parents),
        "source": str(source or "manual").strip() or "manual",
    }


def make_layout_revision_record(
    *,
    plate_id: str,
    layout: str | None = None,
    operation: str = "set",
    parents: Iterable[str] = (),
    source: str = "manual",
    producer: str = DEFAULT_PRODUCER,
) -> dict:
    core = _layout_revision_core(
        plate_id=plate_id,
        operation=operation,
        layout=layout,
        parents=parents,
        source=source,
    )
    revision_id = f"{LAYOUT_REVISION_ID_PREFIX}{_content_sha256(core)}"
    return {
        "schema": LAYOUT_REVISION_SCHEMA,
        "layout_revision_id": revision_id,
        **core,
        "observed_by": _unique_sorted([producer]),
    }


def _layout_revision_id_from_record(record: dict) -> str:
    core = _layout_revision_core(
        plate_id=str(record.get("plate_id") or ""),
        operation=str(record.get("operation") or ""),
        layout=record.get("value"),
        parents=record.get("parents", []) or [],
        source=str(record.get("source") or "manual"),
    )
    return f"{LAYOUT_REVISION_ID_PREFIX}{_content_sha256(core)}"


def _semantic_layout_revision_state(record: dict) -> tuple[str, str | None]:
    operation = str(record.get("operation") or "").strip().lower()
    if operation == "clear":
        return ("clear", None)
    if operation == "set":
        return ("set", normalize_plate_layout_gt(record.get("value"), default=""))
    return ("", None)


def _geometry_id_from_record(record: dict) -> str:
    core = _geometry_core(
        plate_id=str(record.get("plate_id") or ""),
        points=list(record.get("points") or []),
        parents=list(record.get("parents") or []),
    )
    return f"{GEOMETRY_ID_PREFIX}{_content_sha256(core)}"


def _revision_id_from_record(record: dict) -> str:
    core = _revision_core(
        plate_id=str(record.get("plate_id") or ""),
        operation=str(record.get("operation") or "set"),
        text=record.get("value"),
        parents=list(record.get("parents") or []),
        source=str(record.get("source") or "manual"),
    )
    return f"{REVISION_ID_PREFIX}{_content_sha256(core)}"


def _semantic_revision_state(record: dict) -> tuple[str, str | None]:
    operation = str(record.get("operation") or "set").strip().lower()
    if operation == "clear":
        return ("clear", None)
    return ("set", normalize_registration(record.get("value")))


def _graph_heads(
    record_ids: Iterable[str],
    records: dict[str, dict],
) -> list[str]:
    ids = set(_unique_sorted(record_ids))
    referenced_parents: set[str] = set()
    for record_id in ids:
        record = records.get(record_id)
        if not isinstance(record, dict):
            continue
        for parent_id in _unique_sorted(record.get("parents", []) or []):
            if parent_id in ids:
                referenced_parents.add(parent_id)
    return sorted(ids - referenced_parents)


def _graph_cycle_nodes(
    record_ids: Iterable[str],
    records: dict[str, dict],
) -> set[str]:
    ids = set(_unique_sorted(record_ids))
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle_nodes: set[str] = set()

    def visit(record_id: str, stack: list[str]) -> None:
        if record_id in visited:
            return
        if record_id in visiting:
            try:
                idx = stack.index(record_id)
                cycle_nodes.update(stack[idx:])
            except ValueError:
                cycle_nodes.add(record_id)
            return

        visiting.add(record_id)
        stack.append(record_id)
        record = records.get(record_id)
        if isinstance(record, dict):
            for parent_id in _unique_sorted(
                record.get("parents", []) or []
            ):
                if parent_id in ids:
                    visit(parent_id, stack)
        stack.pop()
        visiting.discard(record_id)
        visited.add(record_id)

    for record_id in sorted(ids):
        visit(record_id, [])
    return cycle_nodes


class ALPRGTPack:
    """Directory-backed ALPR GT Pack v1."""

    def __init__(
        self,
        root: Path | str,
        *,
        create: bool = False,
        producer: str = DEFAULT_PRODUCER,
    ):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.images_dir = self.root / "images"
        self.plates_dir = self.root / "plates"
        self.geometries_dir = self.root / "geometries"
        self.revisions_dir = self.root / "revisions"
        self.layout_revisions_dir = self.root / "layout_revisions"
        self.conflicts_dir = self.root / "conflicts"
        self.blobs_dir = self.root / "blobs" / "images"
        self.lock_path = self.root / LOCK_FILENAME
        self._lock_depth = 0
        self._create_mode = bool(create)

        if create:
            self._ensure_structure()
            if not self.manifest_path.exists():
                _atomic_write_json(
                    self.manifest_path,
                    self._new_manifest(producer),
                )
        self._load_manifest()

    @staticmethod
    def _new_manifest(producer: str) -> dict:
        return {
            "schema": PACK_SCHEMA,
            "version": 1,
            "image_identity": "source_file_sha256.v1",
            "normalization_policy": NORMALIZATION_POLICY,
            "producers": _unique_sorted([producer]),
            "record_counts": {
                "images": 0,
                "plates": 0,
                "geometries": 0,
                "revisions": 0,
                "layout_revisions": 0,
                "conflicts": 0,
                "blobs": 0,
            },
        }

    @classmethod
    def create(
        cls,
        root: Path | str,
        *,
        producer: str = DEFAULT_PRODUCER,
    ) -> "ALPRGTPack":
        return cls(root, create=True, producer=producer)

    @classmethod
    def open(cls, root: Path | str) -> "ALPRGTPack":
        return cls(root, create=False)

    def _ensure_structure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.images_dir,
            self.plates_dir,
            self.geometries_dir,
            self.revisions_dir,
            self.layout_revisions_dir,
            self.conflicts_dir,
            self.blobs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> dict:
        manifest = _read_json(self.manifest_path, default=None)
        if not isinstance(manifest, dict):
            raise GTPackFormatError(
                f"Brak manifestu ALPR GT Pack: {self.manifest_path}"
            )
        if str(manifest.get("schema") or "") != PACK_SCHEMA:
            raise GTPackFormatError(
                f"Nieobsługiwany schemat GT Pack: "
                f"{manifest.get('schema')!r}"
            )
        if (
            str(manifest.get("normalization_policy") or "")
            != NORMALIZATION_POLICY
        ):
            raise GTPackFormatError(
                "GT Pack używa innej polityki normalizacji rejestracji."
            )
        self.manifest = manifest
        # Opening a source pack is read-only. Missing directories are tolerated
        # and created lazily only by write operations.
        if self._create_mode:
            self._ensure_structure()
        return manifest

    @contextmanager
    def write_lock(
        self,
        *,
        timeout_seconds: float = 5.0,
    ):
        """Best-effort cross-process single-writer lock.

        The lock is re-entrant inside one ALPRGTPack instance. A very old lock
        is considered stale only after one hour; normal writes should finish
        much earlier.
        """
        if self._lock_depth > 0:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return

        deadline = time.monotonic() + max(
            0.1,
            float(timeout_seconds),
        )

        while True:
            try:
                fd = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                try:
                    payload = (
                        f"pid={os.getpid()}\n"
                        f"created={time.time():.6f}\n"
                    ).encode("utf-8")
                    os.write(fd, payload)
                finally:
                    os.close(fd)
                break
            except FileExistsError:
                try:
                    age = time.time() - self.lock_path.stat().st_mtime
                except Exception:
                    age = 0.0
                if age > LOCK_STALE_SECONDS:
                    try:
                        self.lock_path.unlink()
                        continue
                    except Exception:
                        pass
                if time.monotonic() >= deadline:
                    raise GTPackError(
                        f"GT Pack jest aktualnie zapisywany: {self.root}"
                    )
                time.sleep(0.05)

        self._lock_depth = 1
        try:
            yield
        finally:
            self._lock_depth = 0
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _record_path(
        self,
        directory: Path,
        record_id: str,
    ) -> Path:
        return directory / _safe_record_filename(record_id)

    def _read_record(
        self,
        directory: Path,
        record_id: str,
    ) -> dict | None:
        payload = _read_json(
            self._record_path(directory, record_id),
            default=None,
        )
        return payload if isinstance(payload, dict) else None

    def _write_record(
        self,
        directory: Path,
        record_id: str,
        payload: dict,
    ) -> None:
        _atomic_write_json(
            self._record_path(directory, record_id),
            _normalized_record(payload),
        )

    def get_image(self, image_id: str) -> dict | None:
        return self._read_record(
            self.images_dir,
            image_id,
        )

    def get_plate(self, plate_id: str) -> dict | None:
        return self._read_record(
            self.plates_dir,
            plate_id,
        )

    def get_geometry(self, geometry_id: str) -> dict | None:
        return self._read_record(
            self.geometries_dir,
            geometry_id,
        )

    def get_revision(self, revision_id: str) -> dict | None:
        return self._read_record(
            self.revisions_dir,
            revision_id,
        )

    def get_layout_revision(self, revision_id: str) -> dict | None:
        return self._read_record(
            self.layout_revisions_dir,
            revision_id,
        )

    def _list_records(self, directory: Path) -> list[dict]:
        result = []
        for path in sorted(directory.glob("*.json")):
            payload = _read_json(path, default=None)
            if isinstance(payload, dict):
                result.append(payload)
        return result

    def list_images(self) -> list[dict]:
        return self._list_records(self.images_dir)

    def list_plates(self) -> list[dict]:
        return self._list_records(self.plates_dir)

    def list_revisions(self) -> list[dict]:
        return self._list_records(self.revisions_dir)

    def list_layout_revisions(self) -> list[dict]:
        return self._list_records(self.layout_revisions_dir)

    def list_geometries(self) -> list[dict]:
        return self._list_records(self.geometries_dir)

    def list_conflicts(self) -> list[dict]:
        return self._list_records(self.conflicts_dir)

    def _revision_record_map(
        self,
        plate: dict | None = None,
    ) -> dict[str, dict]:
        restrict = plate is not None
        allowed = set(
            _unique_sorted(
                (plate or {}).get("revision_ids", []) or []
            )
        )
        result: dict[str, dict] = {}
        records = (
            (self.get_revision(record_id) for record_id in sorted(allowed))
            if restrict else self.list_revisions()
        )
        for record in records:
            if not isinstance(record, dict):
                continue
            record_id = str(record.get("revision_id") or "")
            if not record_id:
                continue
            if restrict and record_id not in allowed:
                continue
            result[record_id] = record
        return result

    def _layout_revision_record_map(
        self,
        plate: dict | None = None,
    ) -> dict[str, dict]:
        restrict = plate is not None
        allowed = set(
            _unique_sorted(
                (plate or {}).get("layout_revision_ids", []) or []
            )
        )
        result: dict[str, dict] = {}
        records = (
            (self.get_layout_revision(record_id) for record_id in sorted(allowed))
            if restrict else self.list_layout_revisions()
        )
        for record in records:
            if not isinstance(record, dict):
                continue
            record_id = str(record.get("layout_revision_id") or "")
            if not record_id:
                continue
            if restrict and record_id not in allowed:
                continue
            result[record_id] = record
        return result

    def _geometry_record_map(
        self,
        plate: dict | None = None,
    ) -> dict[str, dict]:
        restrict = plate is not None
        allowed = set(
            _unique_sorted(
                (plate or {}).get(
                    "geometry_revision_ids",
                    [],
                )
                or []
            )
        )
        result: dict[str, dict] = {}
        records = (
            (self.get_geometry(record_id) for record_id in sorted(allowed))
            if restrict else self.list_geometries()
        )
        for record in records:
            if not isinstance(record, dict):
                continue
            record_id = str(record.get("geometry_id") or "")
            if not record_id:
                continue
            if restrict and record_id not in allowed:
                continue
            result[record_id] = record
        return result

    def _compute_revision_heads(
        self,
        plate: dict,
    ) -> list[str]:
        records = self._revision_record_map(plate)
        return _graph_heads(
            plate.get("revision_ids", []) or [],
            records,
        )

    def _compute_layout_heads(
        self,
        plate: dict,
    ) -> list[str]:
        records = self._layout_revision_record_map(plate)
        return _graph_heads(
            plate.get("layout_revision_ids", []) or [],
            records,
        )

    def _compute_geometry_heads(
        self,
        plate: dict,
    ) -> list[str]:
        records = self._geometry_record_map(plate)
        return _graph_heads(
            plate.get("geometry_revision_ids", []) or [],
            records,
        )

    def _normalize_plate_heads(
        self,
        plate: dict,
    ) -> dict:
        normalized = dict(plate)
        normalized["revision_ids"] = _unique_sorted(
            normalized.get("revision_ids", []) or []
        )
        normalized["geometry_revision_ids"] = _unique_sorted(
            normalized.get("geometry_revision_ids", []) or []
        )
        normalized["layout_revision_ids"] = _unique_sorted(
            normalized.get("layout_revision_ids", []) or []
        )
        normalized["revision_heads"] = self._compute_revision_heads(
            normalized
        )
        normalized["layout_heads"] = self._compute_layout_heads(
            normalized
        )
        normalized["geometry_heads"] = self._compute_geometry_heads(
            normalized
        )
        return normalized

    def _normalize_all_plate_heads(self) -> None:
        for plate in self.list_plates():
            plate_id = str(plate.get("plate_id") or "").strip()
            if not plate_id:
                continue
            normalized = self._normalize_plate_heads(plate)
            if normalized != plate:
                self._write_record(
                    self.plates_dir,
                    plate_id,
                    normalized,
                )

    def add_image(
        self,
        image_path: Path | str,
        *,
        alias: str | None = None,
        copy_blob: bool = False,
        producer: str = DEFAULT_PRODUCER,
    ) -> dict:
        image_path = Path(image_path)
        fingerprint = fingerprint_image(image_path)
        image_id = fingerprint["image_id"]

        with self.write_lock():
            existing = self.get_image(image_id) or {}
            if existing:
                for field in (
                    "source_file_sha256",
                    "width",
                    "height",
                ):
                    existing_value = existing.get(field)
                    incoming_value = fingerprint.get(field)
                    if (
                        existing_value not in (None, "")
                        and incoming_value not in (None, "")
                        and existing_value != incoming_value
                    ):
                        raise GTPackFormatError(
                            f"Kolizja tożsamości obrazu {image_id}: {field}"
                        )

            aliases = _unique_sorted(
                list(existing.get("aliases", []) or [])
                + [str(alias or image_path.name)]
            )
            producers = _unique_sorted(
                list(existing.get("producers", []) or [])
                + [producer]
            )
            blob_refs = list(
                existing.get("blob_refs", []) or []
            )

            if copy_blob:
                blob_name = fingerprint["source_file_sha256"]
                self.blobs_dir.mkdir(parents=True, exist_ok=True)
                blob_path = self.blobs_dir / blob_name
                if not blob_path.exists():
                    shutil.copy2(image_path, blob_path)
                blob_refs = _unique_sorted(
                    blob_refs
                    + [f"blobs/images/{blob_name}"]
                )

            record = {
                "schema": IMAGE_SCHEMA,
                "image_id": image_id,
                "source_file_sha256": fingerprint[
                    "source_file_sha256"
                ],
                "pixel_sha256s": _unique_sorted(
                    list(existing.get("pixel_sha256s", []) or [])
                    + [fingerprint["pixel_sha256"]]
                ),
                "perceptual_dhash64s": _unique_sorted(
                    list(
                        existing.get(
                            "perceptual_dhash64s",
                            [],
                        )
                        or []
                    )
                    + [fingerprint["perceptual_dhash64"]]
                ),
                "width": fingerprint["width"],
                "height": fingerprint["height"],
                "aliases": aliases,
                "plate_ids": _unique_sorted(
                    existing.get("plate_ids", []) or []
                ),
                "blob_refs": _unique_sorted(blob_refs),
                "producers": producers,
            }
            self._write_record(
                self.images_dir,
                image_id,
                record,
            )
            self.refresh_manifest()
            return record

    def ensure_plate(
        self,
        *,
        image_id: str,
        polygon,
        plate_id: str | None = None,
    ) -> dict:
        with self.write_lock():
            image_record = self.get_image(image_id)
            if not image_record:
                raise GTPackError(
                    f"Nieznany image_id: {image_id}"
                )

            plate_id = normalize_plate_id(plate_id)
            existing = self.get_plate(plate_id)

            if (
                existing
                and str(existing.get("image_id") or "")
                != image_id
            ):
                raise GTPackFormatError(
                    f"plate_id {plate_id} jest już "
                    "przypisany do innego obrazu."
                )

            normalized_points = normalize_polygon(
                polygon,
                image_width=int(image_record["width"]),
                image_height=int(image_record["height"]),
            )

            existing = dict(existing or {})
            existing.setdefault("revision_ids", [])
            existing.setdefault("geometry_revision_ids", [])
            existing.setdefault("layout_revision_ids", [])
            existing = self._normalize_plate_heads(existing)

            current_geometry = (
                self.resolve_plate_geometry(plate_id)
                if self.get_plate(plate_id)
                else None
            )

            if (
                isinstance(current_geometry, dict)
                and current_geometry.get("resolved")
                and current_geometry.get("points")
                == normalized_points
            ):
                geometry_ids = _unique_sorted(
                    existing.get(
                        "geometry_revision_ids",
                        [],
                    )
                    or []
                )
            else:
                geometry_record = make_geometry_record(
                    plate_id=plate_id,
                    points=normalized_points,
                    parents=existing.get(
                        "geometry_heads",
                        [],
                    ),
                )
                geometry_id = geometry_record["geometry_id"]
                if not self.get_geometry(geometry_id):
                    self._write_record(
                        self.geometries_dir,
                        geometry_id,
                        geometry_record,
                    )
                geometry_ids = _unique_sorted(
                    list(
                        existing.get(
                            "geometry_revision_ids",
                            [],
                        )
                        or []
                    )
                    + [geometry_id]
                )

            plate_record = {
                "schema": PLATE_SCHEMA,
                "plate_id": plate_id,
                "image_id": image_id,
                "geometry_revision_ids": geometry_ids,
                "geometry_heads": [],
                "revision_ids": _unique_sorted(
                    existing.get("revision_ids", []) or []
                ),
                "revision_heads": [],
                "layout_revision_ids": _unique_sorted(
                    existing.get("layout_revision_ids", []) or []
                ),
                "layout_heads": [],
                "legacy_ids": _unique_sorted(
                    existing.get("legacy_ids", []) or []
                ),
            }
            plate_record = self._normalize_plate_heads(
                plate_record
            )
            self._write_record(
                self.plates_dir,
                plate_id,
                plate_record,
            )

            image_record["plate_ids"] = _unique_sorted(
                list(image_record.get("plate_ids", []) or [])
                + [plate_id]
            )
            self._write_record(
                self.images_dir,
                image_id,
                image_record,
            )
            self.refresh_manifest()
            return plate_record

    def _append_gt_revision(
        self,
        plate_id: str,
        *,
        operation: str,
        text: str | None,
        source: str,
        producer: str,
    ) -> dict:
        plate = self.get_plate(plate_id)
        if not plate:
            raise GTPackError(
                f"Nieznany plate_id: {plate_id}"
            )
        plate = self._normalize_plate_heads(plate)

        current = self.resolve_ground_truth(plate_id)
        desired_state = (
            ("clear", None)
            if str(operation).lower() == "clear"
            else ("set", normalize_registration(text))
        )
        if (
            current.get("resolved")
            and _semantic_resolution_state(current)
            == desired_state
            and len(current.get("revision_ids", []) or []) == 1
        ):
            existing_revision = self.get_revision(
                current["revision_ids"][0]
            )
            if existing_revision:
                return existing_revision

        revision = make_revision_record(
            plate_id=plate_id,
            text=text,
            operation=operation,
            parents=plate.get("revision_heads", []),
            source=source,
            producer=producer,
        )
        revision_id = revision["revision_id"]

        existing_revision = self.get_revision(revision_id)
        if existing_revision:
            revision["observed_by"] = _unique_sorted(
                list(
                    existing_revision.get(
                        "observed_by",
                        [],
                    )
                    or []
                )
                + list(revision.get("observed_by", []) or [])
            )

        self._write_record(
            self.revisions_dir,
            revision_id,
            revision,
        )

        plate["revision_ids"] = _unique_sorted(
            list(plate.get("revision_ids", []) or [])
            + [revision_id]
        )
        plate = self._normalize_plate_heads(plate)
        self._write_record(
            self.plates_dir,
            plate_id,
            plate,
        )
        self.refresh_manifest()
        return revision

    def set_ground_truth(
        self,
        plate_id: str,
        text: str,
        *,
        source: str = "manual_z2",
        producer: str = DEFAULT_PRODUCER,
    ) -> dict:
        normalized = normalize_registration(text)
        if not normalized:
            raise GTPackFormatError(
                "Nie można zapisać pustego GT przez SET. "
                "Użyj clear_ground_truth()."
            )
        with self.write_lock():
            return self._append_gt_revision(
                plate_id,
                operation="set",
                text=normalized,
                source=source,
                producer=producer,
            )

    def clear_ground_truth(
        self,
        plate_id: str,
        *,
        source: str = "manual_z2",
        producer: str = DEFAULT_PRODUCER,
    ) -> dict:
        with self.write_lock():
            return self._append_gt_revision(
                plate_id,
                operation="clear",
                text=None,
                source=source,
                producer=producer,
            )

    def resolve_ground_truth(
        self,
        plate_id: str,
    ) -> dict:
        plate = self.get_plate(plate_id)
        if not plate:
            return {
                "resolved": False,
                "conflict": False,
                "has_ground_truth": False,
                "operation": "",
                "text": "",
                "revision_ids": [],
                "reason": "missing_plate",
            }

        records = self._revision_record_map(plate)
        head_ids = _graph_heads(
            plate.get("revision_ids", []) or [],
            records,
        )
        revisions = [
            records[revision_id]
            for revision_id in head_ids
            if revision_id in records
        ]

        if not revisions:
            return {
                "resolved": False,
                "conflict": False,
                "has_ground_truth": False,
                "operation": "",
                "text": "",
                "revision_ids": [],
                "reason": "missing_ground_truth",
            }

        states = {
            _semantic_revision_state(revision)
            for revision in revisions
        }

        if len(states) == 1:
            operation, value = next(iter(states))
            is_set = operation == "set"
            return {
                "resolved": True,
                "conflict": False,
                "has_ground_truth": bool(is_set and value),
                "operation": operation,
                "text": str(value or "") if is_set else "",
                "revision_ids": head_ids,
                "reason": (
                    "resolved"
                    if is_set
                    else "cleared"
                ),
            }

        values = sorted(
            {
                str(value or "")
                if operation == "set"
                else "<CLEAR>"
                for operation, value in states
            }
        )
        return {
            "resolved": False,
            "conflict": True,
            "has_ground_truth": False,
            "operation": "",
            "text": "",
            "values": values,
            "revision_ids": head_ids,
            "reason": "concurrent_ground_truth",
        }

    def _append_layout_revision(
        self,
        plate_id: str,
        *,
        operation: str,
        layout: str | None,
        source: str,
        producer: str,
    ) -> dict:
        plate = self.get_plate(plate_id)
        if not plate:
            raise GTPackError(f"Nieznany plate_id: {plate_id}")
        plate = self._normalize_plate_heads(plate)

        current = self.resolve_plate_layout_gt(plate_id)
        desired_state = (
            ("clear", None)
            if str(operation).lower() == "clear"
            else ("set", normalize_plate_layout_gt(layout, default=""))
        )
        if (
            current.get("resolved")
            and (
                ("clear", None)
                if str(current.get("operation") or "").lower() == "clear"
                else ("set", normalize_plate_layout_gt(current.get("layout"), default=""))
            ) == desired_state
            and len(current.get("layout_revision_ids", []) or []) == 1
        ):
            existing_revision = self.get_layout_revision(current["layout_revision_ids"][0])
            if existing_revision:
                return existing_revision

        revision = make_layout_revision_record(
            plate_id=plate_id,
            layout=layout,
            operation=operation,
            parents=plate.get("layout_heads", []),
            source=source,
            producer=producer,
        )
        revision_id = revision["layout_revision_id"]
        existing_revision = self.get_layout_revision(revision_id)
        if existing_revision:
            revision["observed_by"] = _unique_sorted(
                list(existing_revision.get("observed_by", []) or [])
                + list(revision.get("observed_by", []) or [])
            )
        self._write_record(self.layout_revisions_dir, revision_id, revision)
        plate["layout_revision_ids"] = _unique_sorted(
            list(plate.get("layout_revision_ids", []) or []) + [revision_id]
        )
        plate = self._normalize_plate_heads(plate)
        self._write_record(self.plates_dir, plate_id, plate)
        self.refresh_manifest()
        return revision

    def set_plate_layout_gt(
        self,
        plate_id: str,
        layout: str,
        *,
        source: str = "manual_z2",
        producer: str = DEFAULT_PRODUCER,
    ) -> dict:
        normalized = normalize_plate_layout_gt(layout, default="")
        if not normalized:
            raise GTPackFormatError("Nieprawidłowy layout GT. Użyj single_row albo two_row.")
        with self.write_lock():
            return self._append_layout_revision(
                plate_id,
                operation="set",
                layout=normalized,
                source=source,
                producer=producer,
            )

    def clear_plate_layout_gt(
        self,
        plate_id: str,
        *,
        source: str = "manual_z2",
        producer: str = DEFAULT_PRODUCER,
    ) -> dict:
        with self.write_lock():
            return self._append_layout_revision(
                plate_id,
                operation="clear",
                layout=None,
                source=source,
                producer=producer,
            )

    def resolve_plate_layout_gt(self, plate_id: str) -> dict:
        plate = self.get_plate(plate_id)
        if not plate:
            return {
                "resolved": False,
                "conflict": False,
                "has_layout": False,
                "operation": "",
                "layout": "",
                "layout_revision_ids": [],
                "reason": "missing_plate",
            }
        records = self._layout_revision_record_map(plate)
        head_ids = _graph_heads(plate.get("layout_revision_ids", []) or [], records)
        revisions = [records[revision_id] for revision_id in head_ids if revision_id in records]
        if not revisions:
            return {
                "resolved": False,
                "conflict": False,
                "has_layout": False,
                "operation": "",
                "layout": "",
                "layout_revision_ids": [],
                "reason": "missing_layout",
            }
        states = {_semantic_layout_revision_state(revision) for revision in revisions}
        if len(states) == 1:
            operation, value = next(iter(states))
            is_set = operation == "set"
            return {
                "resolved": True,
                "conflict": False,
                "has_layout": bool(is_set and value),
                "operation": operation,
                "layout": str(value or "") if is_set else "",
                "layout_revision_ids": head_ids,
                "reason": "resolved" if is_set else "cleared",
            }
        values = sorted(
            {str(value or "") if operation == "set" else "<CLEAR>" for operation, value in states}
        )
        return {
            "resolved": False,
            "conflict": True,
            "has_layout": False,
            "operation": "",
            "layout": "",
            "values": values,
            "layout_revision_ids": head_ids,
            "reason": "concurrent_layout",
        }

    def resolve_plate_geometry(
        self,
        plate_id: str,
    ) -> dict:
        plate = self.get_plate(plate_id)
        if not plate:
            return {
                "resolved": False,
                "conflict": False,
                "points": [],
                "geometry_ids": [],
                "reason": "missing_plate",
            }

        records = self._geometry_record_map(plate)
        head_ids = _graph_heads(
            plate.get("geometry_revision_ids", []) or [],
            records,
        )
        geometries = [
            records[geometry_id]
            for geometry_id in head_ids
            if geometry_id in records
        ]

        if not geometries:
            return {
                "resolved": False,
                "conflict": False,
                "points": [],
                "geometry_ids": [],
                "reason": "missing_geometry",
            }

        points_by_key = {}
        for geometry in geometries:
            points = geometry.get("points", [])
            points_by_key[
                _canonical_json_bytes(points)
            ] = geometry

        if len(points_by_key) == 1:
            geometry = next(iter(points_by_key.values()))
            return {
                "resolved": True,
                "conflict": False,
                "points": geometry.get("points", []),
                "geometry_id": geometry.get(
                    "geometry_id",
                    "",
                ),
                "geometry_ids": head_ids,
                "reason": "resolved",
            }

        return {
            "resolved": False,
            "conflict": True,
            "points": [],
            "geometry_ids": head_ids,
            "reason": "concurrent_geometry",
        }

    def upsert_z2_plate(
        self,
        *,
        image_path: Path | str,
        polygon,
        plate_annotation_id: str | None,
        ground_truth_text: str | None = None,
        ground_truth_source: str = "manual_z2",
        plate_layout_gt: str | None = None,
        alias: str | None = None,
        copy_blob: bool = False,
        producer: str = DEFAULT_PRODUCER,
    ) -> dict:
        with self.write_lock():
            image = self.add_image(
                image_path,
                alias=alias,
                copy_blob=copy_blob,
                producer=producer,
            )
            plate = self.ensure_plate(
                image_id=image["image_id"],
                polygon=polygon,
                plate_id=plate_annotation_id,
            )
            revision = None
            normalized_gt = normalize_registration(
                ground_truth_text
            )
            if normalized_gt:
                revision = self.set_ground_truth(
                    plate["plate_id"],
                    normalized_gt,
                    source=ground_truth_source,
                    producer=producer,
                )
            layout_revision = None
            normalized_layout = normalize_plate_layout_gt(plate_layout_gt, default="")
            if normalized_layout:
                layout_revision = self.set_plate_layout_gt(
                    plate["plate_id"],
                    normalized_layout,
                    source=ground_truth_source,
                    producer=producer,
                )
            return {
                "image": image,
                "plate": self.get_plate(
                    plate["plate_id"]
                ),
                "geometry": self.resolve_plate_geometry(
                    plate["plate_id"]
                ),
                "ground_truth": self.resolve_ground_truth(
                    plate["plate_id"]
                ),
                "layout_ground_truth": self.resolve_plate_layout_gt(
                    plate["plate_id"]
                ),
                "revision": revision,
                "layout_revision": layout_revision,
            }

    def _merge_image_record(
        self,
        incoming: dict,
    ) -> None:
        image_id = str(
            incoming.get("image_id") or ""
        ).strip()
        if not image_id:
            raise GTPackFormatError(
                "Rekord obrazu bez image_id."
            )
        existing = self.get_image(image_id)
        if not existing:
            self._write_record(
                self.images_dir,
                image_id,
                incoming,
            )
            return

        immutable_fields = (
            "source_file_sha256",
            "width",
            "height",
        )
        if any(
            existing.get(key) != incoming.get(key)
            for key in immutable_fields
        ):
            self._record_conflict(
                kind="image_identity",
                record_id=image_id,
                candidates=[existing, incoming],
            )
            chosen = dict(
                _deterministic_choice(existing, incoming)
            )
        else:
            chosen = dict(existing)

        for field in (
            "aliases",
            "pixel_sha256s",
            "perceptual_dhash64s",
            "plate_ids",
            "blob_refs",
            "producers",
        ):
            chosen[field] = _unique_sorted(
                list(existing.get(field, []) or [])
                + list(incoming.get(field, []) or [])
            )

        self._write_record(
            self.images_dir,
            image_id,
            chosen,
        )

    def _merge_plate_record(
        self,
        incoming: dict,
    ) -> None:
        plate_id = str(
            incoming.get("plate_id") or ""
        ).strip()
        if not plate_id:
            raise GTPackFormatError(
                "Rekord tablicy bez plate_id."
            )
        existing = self.get_plate(plate_id)
        if not existing:
            self._write_record(
                self.plates_dir,
                plate_id,
                incoming,
            )
            return

        if (
            str(existing.get("image_id") or "")
            != str(incoming.get("image_id") or "")
        ):
            self._record_conflict(
                kind="plate_image_identity",
                record_id=plate_id,
                candidates=[existing, incoming],
            )
            chosen = dict(
                _deterministic_choice(
                    existing,
                    incoming,
                )
            )
        else:
            chosen = dict(existing)

        for field in (
            "geometry_revision_ids",
            "revision_ids",
            "layout_revision_ids",
            "legacy_ids",
        ):
            chosen[field] = _unique_sorted(
                list(existing.get(field, []) or [])
                + list(incoming.get(field, []) or [])
            )

        # Heads are derived data. Do not union them blindly; they will be
        # recalculated after all immutable revisions have been imported.
        chosen["geometry_heads"] = []
        chosen["revision_heads"] = []
        chosen["layout_heads"] = []
        self._write_record(
            self.plates_dir,
            plate_id,
            chosen,
        )

    def _merge_revision_record(
        self,
        incoming: dict,
    ) -> None:
        revision_id = str(
            incoming.get("revision_id") or ""
        ).strip()
        if not revision_id:
            raise GTPackFormatError(
                "Rekord GT bez revision_id."
            )
        existing = self.get_revision(revision_id)
        if not existing:
            self._write_record(
                self.revisions_dir,
                revision_id,
                incoming,
            )
            return

        left_core = {
            key: value
            for key, value in existing.items()
            if key != "observed_by"
        }
        right_core = {
            key: value
            for key, value in incoming.items()
            if key != "observed_by"
        }

        if left_core != right_core:
            self._record_conflict(
                kind="revision_identity",
                record_id=revision_id,
                candidates=[existing, incoming],
            )
            chosen = dict(
                _deterministic_choice(
                    existing,
                    incoming,
                )
            )
        else:
            chosen = dict(existing)

        chosen["observed_by"] = _unique_sorted(
            list(existing.get("observed_by", []) or [])
            + list(incoming.get("observed_by", []) or [])
        )
        self._write_record(
            self.revisions_dir,
            revision_id,
            chosen,
        )

    def _merge_layout_revision_record(
        self,
        incoming: dict,
    ) -> None:
        revision_id = str(incoming.get("layout_revision_id") or "").strip()
        if not revision_id:
            raise GTPackFormatError("Rekord layout GT bez layout_revision_id.")
        existing = self.get_layout_revision(revision_id)
        if not existing:
            self._write_record(self.layout_revisions_dir, revision_id, incoming)
            return
        left_core = {key: value for key, value in existing.items() if key != "observed_by"}
        right_core = {key: value for key, value in incoming.items() if key != "observed_by"}
        if left_core != right_core:
            self._record_conflict(
                kind="layout_revision_identity",
                record_id=revision_id,
                candidates=[existing, incoming],
            )
            chosen = dict(_deterministic_choice(existing, incoming))
        else:
            chosen = dict(existing)
        chosen["observed_by"] = _unique_sorted(
            list(existing.get("observed_by", []) or [])
            + list(incoming.get("observed_by", []) or [])
        )
        self._write_record(self.layout_revisions_dir, revision_id, chosen)

    def _merge_geometry_record(
        self,
        incoming: dict,
    ) -> None:
        geometry_id = str(
            incoming.get("geometry_id") or ""
        ).strip()
        if not geometry_id:
            raise GTPackFormatError(
                "Rekord geometrii bez geometry_id."
            )
        existing = self.get_geometry(geometry_id)
        if not existing:
            self._write_record(
                self.geometries_dir,
                geometry_id,
                incoming,
            )
            return

        if existing != incoming:
            self._record_conflict(
                kind="geometry_identity",
                record_id=geometry_id,
                candidates=[existing, incoming],
            )
            self._write_record(
                self.geometries_dir,
                geometry_id,
                _deterministic_choice(
                    existing,
                    incoming,
                ),
            )

    def _record_conflict(
        self,
        *,
        kind: str,
        record_id: str,
        candidates: Iterable[dict],
    ) -> dict:
        normalized_candidates = []
        seen = set()
        for candidate in candidates:
            normalized = _normalized_record(candidate)
            key = _canonical_json_bytes(normalized)
            if key in seen:
                continue
            seen.add(key)
            normalized_candidates.append(normalized)
        normalized_candidates.sort(
            key=_canonical_json_bytes
        )

        identity = {
            "kind": str(kind),
            "record_id": str(record_id),
            "candidates": normalized_candidates,
        }
        conflict_id = (
            f"{CONFLICT_ID_PREFIX}"
            f"{_content_sha256(identity)}"
        )
        payload = {
            "schema": CONFLICT_SCHEMA,
            "conflict_id": conflict_id,
            **identity,
            "resolution": "unresolved",
        }
        self._write_record(
            self.conflicts_dir,
            conflict_id,
            payload,
        )
        return payload

    def merge_from(
        self,
        other: "ALPRGTPack",
    ) -> dict:
        if not isinstance(other, ALPRGTPack):
            other = ALPRGTPack.open(other)

        with self.write_lock():
            for geometry in other.list_geometries():
                self._merge_geometry_record(geometry)
            for revision in other.list_revisions():
                self._merge_revision_record(revision)
            for layout_revision in other.list_layout_revisions():
                self._merge_layout_revision_record(layout_revision)
            for plate in other.list_plates():
                self._merge_plate_record(plate)
            for image in other.list_images():
                self._merge_image_record(image)
            for conflict in other.list_conflicts():
                conflict_id = str(
                    conflict.get("conflict_id") or ""
                ).strip()
                if (
                    conflict_id
                    and not self._read_record(
                        self.conflicts_dir,
                        conflict_id,
                    )
                ):
                    self._write_record(
                        self.conflicts_dir,
                        conflict_id,
                        conflict,
                    )

            for blob_path in sorted(
                other.blobs_dir.glob("*")
            ):
                if not blob_path.is_file():
                    continue
                target = self.blobs_dir / blob_path.name
                if not target.exists():
                    shutil.copy2(
                        blob_path,
                        target,
                    )
                elif _file_sha256(target) != _file_sha256(
                    blob_path
                ):
                    self._record_conflict(
                        kind="blob_identity",
                        record_id=blob_path.name,
                        candidates=[
                            {
                                "blob": target.name,
                                "sha256": _file_sha256(
                                    target
                                ),
                            },
                            {
                                "blob": blob_path.name,
                                "sha256": _file_sha256(
                                    blob_path
                                ),
                            },
                        ],
                    )

            self._normalize_all_plate_heads()

            self.manifest["producers"] = _unique_sorted(
                list(
                    self.manifest.get(
                        "producers",
                        [],
                    )
                    or []
                )
                + list(
                    other.manifest.get(
                        "producers",
                        [],
                    )
                    or []
                )
            )
            self.refresh_manifest()
            return self.summary()

    def refresh_manifest(self) -> dict:
        counts = {
            "images": len(
                list(self.images_dir.glob("*.json"))
            ),
            "plates": len(
                list(self.plates_dir.glob("*.json"))
            ),
            "geometries": len(
                list(self.geometries_dir.glob("*.json"))
            ),
            "revisions": len(
                list(self.revisions_dir.glob("*.json"))
            ),
            "layout_revisions": len(
                list(self.layout_revisions_dir.glob("*.json"))
            ),
            "conflicts": len(
                list(self.conflicts_dir.glob("*.json"))
            ),
            "blobs": len(
                [
                    path
                    for path in self.blobs_dir.glob("*")
                    if path.is_file()
                ]
            ),
        }
        self.manifest["schema"] = PACK_SCHEMA
        self.manifest["version"] = 1
        self.manifest[
            "image_identity"
        ] = "source_file_sha256.v1"
        self.manifest[
            "normalization_policy"
        ] = NORMALIZATION_POLICY
        self.manifest["producers"] = _unique_sorted(
            self.manifest.get("producers", []) or []
        )
        self.manifest["record_counts"] = counts
        _atomic_write_json(
            self.manifest_path,
            self.manifest,
        )
        return self.manifest

    def rebuild_manifest(self) -> dict:
        with self.write_lock():
            return self.refresh_manifest()

    def summary(self, *, refresh_manifest: bool = True) -> dict:
        if refresh_manifest:
            self.refresh_manifest()
        unresolved_gt = 0
        unresolved_geometry = 0
        unresolved_layout = 0
        cleared_gt = 0
        cleared_layout = 0

        for plate in self.list_plates():
            plate_id = str(plate.get("plate_id") or "")
            gt_state = self.resolve_ground_truth(plate_id)
            if gt_state.get("conflict"):
                unresolved_gt += 1
            if gt_state.get("resolved") and gt_state.get("operation") == "clear":
                cleared_gt += 1
            layout_state = self.resolve_plate_layout_gt(plate_id)
            if layout_state.get("conflict"):
                unresolved_layout += 1
            if layout_state.get("resolved") and layout_state.get("operation") == "clear":
                cleared_layout += 1
            if self.resolve_plate_geometry(plate_id).get("conflict"):
                unresolved_geometry += 1

        result = {
            "images": len(list(self.images_dir.glob("*.json"))),
            "plates": len(list(self.plates_dir.glob("*.json"))),
            "geometries": len(list(self.geometries_dir.glob("*.json"))),
            "revisions": len(list(self.revisions_dir.glob("*.json"))),
            "layout_revisions": len(list(self.layout_revisions_dir.glob("*.json"))),
            "conflicts": len(list(self.conflicts_dir.glob("*.json"))),
            "blobs": len([path for path in self.blobs_dir.glob("*") if path.is_file()]),
        }
        result.update(
            {
                "unresolved_gt_conflicts": int(unresolved_gt),
                "unresolved_geometry_conflicts": int(unresolved_geometry),
                "unresolved_layout_conflicts": int(unresolved_layout),
                "cleared_ground_truth": int(cleared_gt),
                "cleared_layout_ground_truth": int(cleared_layout),
            }
        )
        return result

    def validate(
        self,
        *,
        deep: bool = True,
    ) -> dict:
        issues: list[str] = []

        images = self.list_images()
        plates = self.list_plates()
        geometries = self.list_geometries()
        revisions = self.list_revisions()
        layout_revisions = self.list_layout_revisions()

        image_records = {
            str(record.get("image_id") or ""): record
            for record in images
            if str(record.get("image_id") or "")
        }
        plate_records = {
            str(record.get("plate_id") or ""): record
            for record in plates
            if str(record.get("plate_id") or "")
        }
        geometry_records = {
            str(record.get("geometry_id") or ""): record
            for record in geometries
            if str(record.get("geometry_id") or "")
        }
        revision_records = {
            str(record.get("revision_id") or ""): record
            for record in revisions
            if str(record.get("revision_id") or "")
        }
        layout_revision_records = {
            str(record.get("layout_revision_id") or ""): record
            for record in layout_revisions
            if str(record.get("layout_revision_id") or "")
        }

        for image_id, image in image_records.items():
            file_hash = str(
                image.get("source_file_sha256") or ""
            ).strip()
            expected_image_id = (
                f"{IMAGE_ID_PREFIX}{file_hash}"
                if file_hash
                else ""
            )
            if not file_hash:
                issues.append(
                    f"{image_id}: brak source_file_sha256"
                )
            elif image_id != expected_image_id:
                issues.append(
                    f"{image_id}: image_id nie zgadza się "
                    "z source_file_sha256"
                )

        for geometry_id, geometry in geometry_records.items():
            try:
                expected = _geometry_id_from_record(
                    geometry
                )
                if expected != geometry_id:
                    issues.append(
                        f"{geometry_id}: błędny content hash geometrii"
                    )
            except Exception as exc:
                issues.append(
                    f"{geometry_id}: nieprawidłowa geometria: {exc}"
                )

        for revision_id, revision in revision_records.items():
            try:
                expected = _revision_id_from_record(
                    revision
                )
                if expected != revision_id:
                    issues.append(
                        f"{revision_id}: błędny content hash rewizji GT"
                    )
            except Exception as exc:
                issues.append(
                    f"{revision_id}: nieprawidłowa rewizja GT: {exc}"
                )

        for layout_revision_id, revision in layout_revision_records.items():
            try:
                expected = _layout_revision_id_from_record(revision)
                if expected != layout_revision_id:
                    issues.append(
                        f"{layout_revision_id}: błędny content hash rewizji layout GT"
                    )
            except Exception as exc:
                issues.append(
                    f"{layout_revision_id}: nieprawidłowa rewizja layout GT: {exc}"
                )

        for plate_id, plate in plate_records.items():
            image_id = str(
                plate.get("image_id") or ""
            )
            if image_id not in image_records:
                issues.append(
                    f"{plate_id}: brak obrazu {image_id}"
                )

            revision_ids = _unique_sorted(
                plate.get("revision_ids", []) or []
            )
            geometry_ids = _unique_sorted(
                plate.get(
                    "geometry_revision_ids",
                    [],
                )
                or []
            )
            layout_revision_ids = _unique_sorted(
                plate.get("layout_revision_ids", []) or []
            )

            local_revisions = {
                record_id: revision_records[record_id]
                for record_id in revision_ids
                if record_id in revision_records
            }
            local_geometries = {
                record_id: geometry_records[record_id]
                for record_id in geometry_ids
                if record_id in geometry_records
            }
            local_layout_revisions = {
                record_id: layout_revision_records[record_id]
                for record_id in layout_revision_ids
                if record_id in layout_revision_records
            }

            for revision_id in revision_ids:
                revision = revision_records.get(
                    revision_id
                )
                if not revision:
                    issues.append(
                        f"{plate_id}: brak rewizji GT "
                        f"{revision_id}"
                    )
                    continue
                if (
                    str(revision.get("plate_id") or "")
                    != plate_id
                ):
                    issues.append(
                        f"{plate_id}: rewizja "
                        f"{revision_id} należy do innej tablicy"
                    )
                for parent_id in _unique_sorted(
                    revision.get("parents", []) or []
                ):
                    parent = revision_records.get(
                        parent_id
                    )
                    if not parent:
                        issues.append(
                            f"{revision_id}: brak parent GT "
                            f"{parent_id}"
                        )
                    elif (
                        str(parent.get("plate_id") or "")
                        != plate_id
                    ):
                        issues.append(
                            f"{revision_id}: parent GT "
                            f"{parent_id} należy do innej tablicy"
                        )

            revision_cycles = _graph_cycle_nodes(
                revision_ids,
                local_revisions,
            )
            if revision_cycles:
                issues.append(
                    f"{plate_id}: cykl grafu GT: "
                    + ", ".join(
                        sorted(revision_cycles)
                    )
                )

            expected_revision_heads = _graph_heads(
                revision_ids,
                local_revisions,
            )
            stored_revision_heads = _unique_sorted(
                plate.get("revision_heads", []) or []
            )
            if (
                stored_revision_heads
                != expected_revision_heads
            ):
                issues.append(
                    f"{plate_id}: nieaktualne revision_heads"
                )

            for layout_revision_id in layout_revision_ids:
                revision = layout_revision_records.get(layout_revision_id)
                if not revision:
                    issues.append(
                        f"{plate_id}: brak rewizji layout GT {layout_revision_id}"
                    )
                    continue
                if str(revision.get("plate_id") or "") != plate_id:
                    issues.append(
                        f"{plate_id}: rewizja layout {layout_revision_id} należy do innej tablicy"
                    )
                for parent_id in _unique_sorted(revision.get("parents", []) or []):
                    parent = layout_revision_records.get(parent_id)
                    if not parent:
                        issues.append(
                            f"{layout_revision_id}: brak parent layout GT {parent_id}"
                        )
                    elif str(parent.get("plate_id") or "") != plate_id:
                        issues.append(
                            f"{layout_revision_id}: parent layout GT {parent_id} należy do innej tablicy"
                        )

            layout_cycles = _graph_cycle_nodes(layout_revision_ids, local_layout_revisions)
            if layout_cycles:
                issues.append(
                    f"{plate_id}: cykl grafu layout GT: " + ", ".join(sorted(layout_cycles))
                )
            expected_layout_heads = _graph_heads(layout_revision_ids, local_layout_revisions)
            stored_layout_heads = _unique_sorted(plate.get("layout_heads", []) or [])
            if stored_layout_heads != expected_layout_heads:
                issues.append(f"{plate_id}: nieaktualne layout_heads")

            for geometry_id in geometry_ids:
                geometry = geometry_records.get(
                    geometry_id
                )
                if not geometry:
                    issues.append(
                        f"{plate_id}: brak geometrii "
                        f"{geometry_id}"
                    )
                    continue
                if (
                    str(geometry.get("plate_id") or "")
                    != plate_id
                ):
                    issues.append(
                        f"{plate_id}: geometria "
                        f"{geometry_id} należy do innej tablicy"
                    )
                for parent_id in _unique_sorted(
                    geometry.get("parents", []) or []
                ):
                    parent = geometry_records.get(
                        parent_id
                    )
                    if not parent:
                        issues.append(
                            f"{geometry_id}: brak parent geometrii "
                            f"{parent_id}"
                        )
                    elif (
                        str(parent.get("plate_id") or "")
                        != plate_id
                    ):
                        issues.append(
                            f"{geometry_id}: parent geometrii "
                            f"{parent_id} należy do innej tablicy"
                        )

            geometry_cycles = _graph_cycle_nodes(
                geometry_ids,
                local_geometries,
            )
            if geometry_cycles:
                issues.append(
                    f"{plate_id}: cykl grafu geometrii: "
                    + ", ".join(
                        sorted(geometry_cycles)
                    )
                )

            expected_geometry_heads = _graph_heads(
                geometry_ids,
                local_geometries,
            )
            stored_geometry_heads = _unique_sorted(
                plate.get("geometry_heads", []) or []
            )
            if (
                stored_geometry_heads
                != expected_geometry_heads
            ):
                issues.append(
                    f"{plate_id}: nieaktualne geometry_heads"
                )

        for image_id, image in image_records.items():
            for plate_id in _unique_sorted(
                image.get("plate_ids", []) or []
            ):
                plate = plate_records.get(plate_id)
                if not plate:
                    issues.append(
                        f"{image_id}: brak tablicy {plate_id}"
                    )
                elif (
                    str(plate.get("image_id") or "")
                    != image_id
                ):
                    issues.append(
                        f"{image_id}: tablica {plate_id} "
                        "wskazuje inny obraz"
                    )

        if deep:
            for image_id, image in image_records.items():
                expected_hash = str(
                    image.get(
                        "source_file_sha256",
                        "",
                    )
                    or ""
                ).strip()
                for blob_ref in _unique_sorted(
                    image.get("blob_refs", []) or []
                ):
                    blob_path = self.root / Path(
                        blob_ref
                    )
                    if not blob_path.exists():
                        issues.append(
                            f"{image_id}: brak blob {blob_ref}"
                        )
                        continue
                    try:
                        if (
                            expected_hash
                            and _file_sha256(blob_path)
                            != expected_hash
                        ):
                            issues.append(
                                f"{image_id}: blob {blob_ref} "
                                "ma inny SHA-256"
                            )
                    except Exception as exc:
                        issues.append(
                            f"{image_id}: błąd blob "
                            f"{blob_ref}: {exc}"
                        )

        actual_counts = {
            "images": len(images),
            "plates": len(plates),
            "geometries": len(geometries),
            "revisions": len(revisions),
            "layout_revisions": len(layout_revisions),
            "conflicts": len(self.list_conflicts()),
            "blobs": len(
                [
                    path
                    for path in self.blobs_dir.glob("*")
                    if path.is_file()
                ]
            ),
        }
        manifest_counts = dict(
            self.manifest.get(
                "record_counts",
                {},
            )
            or {}
        )
        for key, value in actual_counts.items():
            default_manifest_value = 0 if key == "layout_revisions" else -1
            if int(manifest_counts.get(key, default_manifest_value)) != int(
                value
            ):
                issues.append(
                    f"manifest: nieaktualny licznik {key}"
                )

        return {
            "ok": not issues,
            "issues": issues,
            "summary": self.summary(refresh_manifest=False),
        }


def _semantic_resolution_state(
    resolution: dict,
) -> tuple[str, str | None]:
    operation = str(
        resolution.get("operation") or ""
    ).strip().lower()
    if operation == "clear":
        return ("clear", None)
    if operation == "set":
        return (
            "set",
            normalize_registration(
                resolution.get("text")
            ),
        )
    return ("", None)


def merge_gt_packs(
    sources: Iterable[Path | str],
    output: Path | str,
    *,
    overwrite: bool = False,
    producer: str = DEFAULT_PRODUCER,
) -> ALPRGTPack:
    source_paths = [
        Path(source)
        for source in sources
    ]
    if not source_paths:
        raise GTPackError(
            "Brak zbiorów GT do scalenia."
        )

    output_path = Path(output)
    if output_path.exists():
        if not overwrite:
            if any(output_path.iterdir()):
                raise GTPackError(
                    f"Katalog docelowy nie jest pusty: "
                    f"{output_path}"
                )
        else:
            shutil.rmtree(output_path)

    result = ALPRGTPack.create(
        output_path,
        producer=producer,
    )

    for source_path in sorted(
        source_paths,
        key=lambda path: path.as_posix().lower(),
    ):
        result.merge_from(
            ALPRGTPack.open(source_path)
        )

    result.rebuild_manifest()
    return result
