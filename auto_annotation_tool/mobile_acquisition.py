"""Normal Android acquisition archives and their separate desktop annotation review."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
import zipfile

from PIL import Image
from .registration_text import NORMALIZATION_POLICY, normalize_registration


CROP_SCHEMA = "alpr_crop_session_v1"
REVIEW_SCHEMA = "alpr.mobile_crop_review.v1"

LEGACY_GROUPING_POLICY = "uppercase.v1"
GROUPING_POLICY = NORMALIZATION_POLICY

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


def registration_key(text: str) -> str:
    """Canonical Android/Desktop registration key."""
    return normalize_registration(text)


def _grouping_policy(manifest: dict) -> str:
    """
    Nowe paczki jawnie deklarują normalization_policy.
    Brak pola oznacza starą paczkę uppercase.v1.
    """
    declared = str(manifest.get("normalization_policy") or "").strip()

    if not declared:
        return LEGACY_GROUPING_POLICY

    if declared != GROUPING_POLICY:
        raise ValueError(
            f"Unsupported acquisition normalization_policy: {declared!r}"
        )

    return declared


def _registration_key_for_policy(text: str, policy: str) -> str:
    if policy == LEGACY_GROUPING_POLICY:
        return str(text or "").upper()

    if policy == GROUPING_POLICY:
        return registration_key(text)

    raise ValueError(
        f"Unsupported acquisition grouping policy: {policy!r}"
    )


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def acquisition_provenance(data: dict) -> dict:
    return {key: copy.deepcopy(data[key]) for key in ("dataset_group", "mobile_acquisition") if key in data}


def _json_object(data: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")

    result = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs,
                        parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def _safe_entry(name: str) -> bool:
    parts = name.rstrip("/").split("/")
    return bool(name and not name.startswith("/") and "\\" not in name
                and all(part not in {"", ".", ".."}
                        and not re.search(r'[<>:"|?*\x00-\x1f]', part)
                        and not part.endswith((".", " "))
                        for part in parts))


def _read_entry(archive, info, limit):
    if info.file_size > limit or info.flag_bits & 1 or info.is_dir():
        raise ValueError(f"Unsupported or oversized archive entry: {info.filename}")
    with archive.open(info) as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Archive entry exceeds limit: {info.filename}")
    return data


def _image(data: bytes, expected_size: tuple[int, int]) -> Image.Image:
    with Image.open(io.BytesIO(data)) as candidate:
        if candidate.format != "JPEG" or candidate.size != expected_size:
            raise ValueError("Crop must be a JPEG matching its declared dimensions")
        if candidate.width * candidate.height > MAX_IMAGE_PIXELS:
            raise ValueError("Crop exceeds the pixel limit")
        candidate.load()
        return candidate.convert("RGB")


@dataclass(frozen=True)
class CropGroup:
    plate_id: str
    key: str
    crops: tuple[dict, ...]

    @property
    def base(self) -> dict:
        return self.crops[0]

    @property
    def observation_count(self) -> int:
        return sum(len(crop["observations"]) for crop in self.crops)


@dataclass(frozen=True)
class CropSession:
    path: Path
    archive_sha256: str
    manifest: dict
    groups: tuple[CropGroup, ...]

    @property
    def session_id(self) -> str:
        return self.manifest["session_id"]

    @property
    def grouping_policy(self) -> str:
        return _grouping_policy(self.manifest)

    def read_image(self, group: CropGroup, crop_index: int = 0) -> Image.Image:
        crop = group.crops[crop_index]
        with zipfile.ZipFile(self.path) as archive:
            data = _read_entry(archive, archive.getinfo(crop["image"]), MAX_IMAGE_BYTES)
        return _image(data, (crop["image_width"], crop["image_height"]))


def read_crop_session(path: Path, *, max_entries=50_000,
                      max_total_bytes=2 * 1024**3, max_manifest_bytes=32 * 1024**2) -> CropSession:
    path = Path(path).resolve()
    source_hash = file_sha256(path)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > max_entries or sum(info.file_size for info in infos) > max_total_bytes:
            raise ValueError("Acquisition archive exceeds the size or entry limit")
        seen = set()
        for info in infos:
            key = info.filename.rstrip("/").casefold()
            if not _safe_entry(info.orig_filename) or key in seen:
                raise ValueError(f"Unsafe or duplicate archive entry: {info.filename}")
            seen.add(key)
        if "session.json" not in archive.namelist():
            raise ValueError("Missing session.json")
        manifest = _json_object(_read_entry(archive, archive.getinfo("session.json"), max_manifest_bytes))
        if manifest.get("schema") != CROP_SCHEMA:
            raise ValueError(f"Expected {CROP_SCHEMA}, got {manifest.get('schema')!r}")
        if not isinstance(manifest.get("session_id"), str) or not manifest["session_id"].strip():
            raise ValueError("Missing session_id")

        grouping_policy = _grouping_policy(manifest)

        rows = manifest.get("crops")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Acquisition session contains no crops")
        grouped = {}
        images = set()
        for crop in rows:
            if not isinstance(crop, dict):
                raise ValueError("Invalid crop record")
            text = crop.get("text")
            name = crop.get("image")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Normal acquisition crop requires a nonempty prediction"
                )

            raw_prediction = crop.get("raw_prediction", text)

            if not isinstance(raw_prediction, str) or not raw_prediction.strip():
                raise ValueError(
                    "Normal acquisition crop requires a nonempty raw_prediction"
                )

            if grouping_policy == GROUPING_POLICY and raw_prediction != text:
                raise ValueError(
                    f"Crop text/raw_prediction mismatch: {name!r}"
                )

            expected_key = _registration_key_for_policy(
                raw_prediction,
                grouping_policy,
            )

            declared_key = crop.get("registration_key")

            if grouping_policy == GROUPING_POLICY:
                if (
                    not isinstance(declared_key, str)
                    or declared_key != expected_key
                ):
                    raise ValueError(
                        f"Invalid registration_key for crop: {name!r}"
                    )

                group_key = declared_key
            else:
                group_key = expected_key

            if not isinstance(name, str) or not re.fullmatch(r"crop-[0-9]+\.jpg", name) or name in images:
                raise ValueError(f"Invalid or repeated crop image: {name!r}")
            images.add(name)
            size = (crop.get("image_width"), crop.get("image_height"))
            if any(type(value) is not int or value <= 0 for value in size):
                raise ValueError(f"Invalid dimensions: {name}")
            if size[0] * size[1] > MAX_IMAGE_PIXELS:
                raise ValueError(f"Crop exceeds the pixel limit: {name}")
            observations = crop.get("observations")
            if not isinstance(observations, list) or not observations:
                raise ValueError(f"Missing observations: {name}")
            observation_ids = set()
            for observation in observations:
                if not isinstance(observation, dict):
                    raise ValueError(f"Invalid observation: {name}")
                identity = observation.get("observation_id")
                if not isinstance(identity, str) or not identity or identity in observation_ids:
                    raise ValueError(f"Missing or duplicate observation_id: {name}")
                observation_ids.add(identity)
                prediction = observation.get("text")

                if not isinstance(prediction, str):
                    raise ValueError(
                        f"Observation prediction does not match crop: {name}"
                    )

                raw_observation = observation.get(
                    "raw_prediction",
                    prediction,
                )

                if not isinstance(raw_observation, str):
                    raise ValueError(
                        f"Invalid raw_prediction in observation: {name}"
                    )

                if (
                    grouping_policy == GROUPING_POLICY
                    and raw_observation != prediction
                ):
                    raise ValueError(
                        f"Observation text/raw_prediction mismatch: {name}"
                    )

                observation_key = _registration_key_for_policy(
                    raw_observation,
                    grouping_policy,
                )

                if grouping_policy == GROUPING_POLICY:
                    declared_observation_key = observation.get(
                        "registration_key"
                    )

                    if (
                        not isinstance(declared_observation_key, str)
                        or declared_observation_key != observation_key
                    ):
                        raise ValueError(
                            f"Invalid registration_key in observation: {name}"
                        )

                if observation_key != group_key:
                    raise ValueError(
                        f"Observation prediction does not match crop: {name}"
                    )
            characters = crop.get("characters")
            if not isinstance(characters, list):
                raise ValueError(f"Invalid character list: {name}")
            for character in characters:
                if not isinstance(character, dict) or not isinstance(character.get("label"), str):
                    raise ValueError(f"Invalid character: {name}")
                coordinates = [character.get(field) for field in ("left", "top", "right", "bottom")]
                if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                       for value in coordinates):
                    raise ValueError(f"Invalid normalized character coordinates: {name}")
                if coordinates[2] < coordinates[0] or coordinates[3] < coordinates[1]:
                    raise ValueError(f"Inverted character coordinates: {name}")
            try:
                _image(_read_entry(archive, archive.getinfo(name), MAX_IMAGE_BYTES), size).close()
            except KeyError as exc:
                raise ValueError(f"Missing crop image: {name}") from exc
            grouped.setdefault(group_key, []).append(crop)
    if file_sha256(path) != source_hash:
        raise ValueError("Acquisition archive changed while reading")
    groups = tuple(CropGroup(f"plate_{index:06d}", key, tuple(crops))
                   for index, (key, crops) in enumerate(grouped.items(), 1))
    return CropSession(path, source_hash, manifest, groups)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CropReview:
    def __init__(self, session: CropSession, path: Path):
        self.session = session
        self.path = Path(path)
        self.revision = 0
        self.decisions = {}
        if self.path.exists():
            payload = _json_object(self.path.read_bytes())
            if (payload.get("schema") != REVIEW_SCHEMA
                    or payload.get("archive_sha256") != session.archive_sha256
                    or payload.get("session_id") != session.session_id
                    or payload.get("grouping_policy") != session.grouping_policy):
                raise ValueError("Review does not belong to this acquisition archive")
            self.revision = int(payload["revision"])
            self.decisions = payload["decisions"]
            if not isinstance(self.decisions, dict):
                raise ValueError("Invalid acquisition review decisions")
            for key, value in self.decisions.items():
                self._validate(key, value)

    def _validate(self, plate_id, decision):
        if plate_id not in {group.plate_id for group in self.session.groups}:
            raise ValueError("Unknown acquisition group")
        if (not isinstance(decision, dict) or decision.get("status") not in {"draft", "confirmed", "rejected"}
                or not isinstance(decision.get("gt"), str)):
            raise ValueError("Invalid acquisition decision")
        if decision["status"] == "confirmed" and not decision["gt"].strip():
            raise ValueError("Confirmed acquisition review requires ground truth")

    def set_decision(self, plate_id: str, gt: str, status: str) -> None:
        decision = {"gt": gt.upper(), "status": status}
        self._validate(plate_id, decision)
        next_decisions = {**self.decisions, plate_id: decision}
        if next_decisions == self.decisions:
            return
        if self.path.exists():
            existing = _json_object(self.path.read_bytes())
            if existing.get("archive_sha256") != self.session.archive_sha256 or existing.get("revision") != self.revision:
                raise ValueError("Acquisition review was modified in another window; reopen it")
        elif self.revision:
            raise ValueError("Acquisition review was removed; reopen it")
        payload = self.to_dict()
        payload.update(decisions=next_decisions, revision=self.revision + 1)
        _atomic_json(self.path, payload)
        self.decisions = next_decisions
        self.revision += 1

    def to_dict(self):
        return {
            "schema": REVIEW_SCHEMA,
            "archive_sha256": self.session.archive_sha256,
            "session_id": self.session.session_id,
            "grouping_policy": self.session.grouping_policy,
            "revision": self.revision,
            "decisions": copy.deepcopy(self.decisions),
        }


def export_annotation_preview(review: CropReview, destination: Path) -> Path:
    """Produce a new Z3 preview, keeping the source ZIP and review separate."""
    session = review.session
    selected = [group for group in session.groups
                if review.decisions.get(group.plate_id, {}).get("status") == "confirmed"]
    if not selected:
        raise ValueError("No confirmed ground truth to send to annotation")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Annotation destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".acquisition-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "preview"
        staging.mkdir()
        (staging / "images").mkdir()
        source_zip = staging / "source.zip"
        shutil.copyfile(session.path, source_zip)
        if file_sha256(source_zip) != session.archive_sha256:
            raise ValueError("Source acquisition archive changed; reopen it")
        metadata = {}
        with zipfile.ZipFile(source_zip) as archive:
            for group in selected:
                crop = group.base
                width, height = crop["image_width"], crop["image_height"]
                data = _read_entry(archive, archive.getinfo(crop["image"]), MAX_IMAGE_BYTES)
                (staging / "images" / f"{group.plate_id}.jpg").write_bytes(data)
                characters = [{"character": char["label"].upper(),
                               "bbox": [char["left"] * width, char["top"] * height,
                                        char["right"] * width, char["bottom"] * height],
                               "confidence": char.get("confidence") or 0.0,
                               "method": "yolo", "source_tag": "yolo"}
                              for char in crop["characters"]]
                metadata[group.plate_id] = {
                    "source_image": str(destination / "source.zip") + "!/" + crop["image"],
                    "source_image_name": crop["image"],
                    "ocr_text": crop["text"],
                    "characters": characters,
                    "status": "needs_fix",
                    "source_expected_text": review.decisions[group.plate_id]["gt"],
                    "source_expected_text_source": "mobile_crop_human_review",
                    "dataset_group": "mobile-session:" + session.session_id,
                    "mobile_acquisition": {
                        "schema": CROP_SCHEMA, "archive_sha256": session.archive_sha256,
                        "session_id": session.session_id,"grouping_policy": session.grouping_policy,
                        "group_key": group.key, "base_image": crop["image"],
                        "crops": copy.deepcopy(list(group.crops)),
                        "review_revision": review.revision,
                    },
                }
        _atomic_json(staging / "metadata.json", metadata)
        _atomic_json(staging / "acquisition_review.json", review.to_dict())
        _atomic_json(staging / "acquisition_import.json", {
            "schema": "alpr.desktop_crop_import.v1", "source_schema": CROP_SCHEMA,
            "session_id": session.session_id, "archive_sha256": session.archive_sha256,
            "source_archive": str(session.path), "grouping_policy": session.grouping_policy,
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "plate_count": len(selected), "selection_policy": "nonempty_fresh_mz",
            "pipeline_quality_available": False,
        })
        staging.rename(destination)
    return destination
