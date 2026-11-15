#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contracts for ranking complete Android ALPR model packages.

The regular mobile exporter still builds one logical model package from one
checkpoint.  This module describes the research-level package candidate:
which optional vehicle model, which plate model, which character model, which
runtime variants and which mobile benchmark report belong to one comparable
experiment.
"""

from __future__ import annotations

import datetime as _dt
import csv
import hashlib
import io
import json
import os
import re
import zipfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from ..config import CONFIG, logger


MOBILE_PACKAGE_EXPERIMENT_SCHEMA = "alpr.mobile_package_experiment.v1"
MOBILE_BENCHMARK_REPORT_SCHEMA = "alpr.mobile_benchmark_report.v1"
MOBILE_ALPR_PACKAGE_SCHEMA = "alpr.package.v1"
MOBILE_RESEARCH_BUNDLE_SCHEMA = "alpr.mobile_research_bundle.v1"
MOBILE_THESIS_BUNDLE_SCHEMA = "alpr.mobile_thesis_bundle.v1"

MOBILE_REPORT_MAX_TEXT_BYTES = 32 * 1024 * 1024
MOBILE_REPORT_MAX_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MOBILE_REPORT_MAX_ENTRIES = 50000
MOBILE_REPORT_TRACE_PREVIEW_ROWS = 5000
MOBILE_PACKAGE_EXPERIMENT_STORE_FILE_NAME = "mobile_package_experiments.json"

DEFAULT_SCORE_WEIGHTS = {
    "quality": 0.50,
    "latency": 0.25,
    "memory": 0.15,
    "reliability": 0.10,
}

DEFAULT_SCORE_TARGETS = {
    "pipeline_p95_ms": 250.0,
    "ram_peak_mb": 512.0,
    "package_size_mb": 120.0,
}


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _safe_id(value: str, fallback: str = "pkg") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip(".-_")
    if not text:
        text = fallback
    if not re.match(r"^[A-Za-z0-9]", text):
        text = f"p-{text}"
    return text[:96]


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        text = str(value).strip().replace(",", ".")
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    parsed = _safe_float(value, None)
    if parsed is None:
        return default
    try:
        return int(parsed)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "tak", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "nie", "no", "n", "off"}:
        return False
    return default


def _clamp01(value: Any, default: float = 0.0) -> float:
    parsed = _safe_float(value, None)
    if parsed is None:
        return float(default)
    return max(0.0, min(1.0, float(parsed)))


def _metric01(value: Any, default: float = 0.0) -> float:
    parsed = _safe_float(value, None)
    if parsed is None:
        return float(default)
    if 1.0 < parsed <= 100.0:
        parsed /= 100.0
    return _clamp01(parsed, default)


def _nested_value(data: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = data
        ok = True
        for part in str(path).split("."):
            if isinstance(current, dict) and part in current:
                current = current.get(part)
            else:
                ok = False
                break
        if ok and current is not None and current != "":
            return current
    return None


def _optional_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _role_marker(role: str) -> str:
    raw = str(role or "").strip().lower()
    if raw in {"plate", "plates", "pose", "tablica", "tablice", "mt"}:
        return "MT"
    if raw in {"character", "characters", "char", "chars", "znak", "znaki", "mz"}:
        return "MZ"
    if raw in {"vehicle", "vehicles", "pojazd", "pojazdy", "mp"}:
        return "MP"
    return raw.upper()[:4] or "M?"


_MODEL_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "mp": ("mp", "MP", "vehicle", "vehicles", "Vehicle", "VEHICLE"),
    "mt": ("mt", "MT", "plate", "plates", "Plate", "PLATE"),
    "mz": ("mz", "MZ", "character", "characters", "char", "chars", "Character", "CHARACTER"),
}


def _is_present_model_fingerprint(value: Any) -> bool:
    return value not in (None, "", {}, [])


def _merge_mobile_model_fingerprints(target: dict[str, Any], value: Any) -> None:
    if not isinstance(value, dict):
        return
    for nested_key in ("model_fingerprints", "models"):
        nested = value.get(nested_key)
        if isinstance(nested, dict):
            _merge_mobile_model_fingerprints(target, nested)
    for canonical, aliases in _MODEL_ROLE_ALIASES.items():
        for alias in aliases:
            if alias not in value:
                continue
            role_value = value.get(alias)
            if _is_present_model_fingerprint(role_value) and canonical not in target:
                target[canonical] = role_value
                break


def _normalize_mobile_model_fingerprints(*values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        _merge_mobile_model_fingerprints(result, value)
    return result


def read_alprmodel_manifest(package_path: Path) -> dict[str, Any]:
    """Read the manifest from a single-model ``.alprmodel`` package."""
    safe_path = Path(package_path)
    if not safe_path.exists() or not safe_path.is_file():
        raise FileNotFoundError(f"Missing .alprmodel package: {safe_path}")
    with zipfile.ZipFile(safe_path, "r") as archive:
        if "manifest.json" not in set(archive.namelist()):
            raise ValueError(f"Package has no manifest.json: {safe_path}")
        return json.loads(archive.read("manifest.json").decode("utf-8"))


def read_alpr_package_manifest(package_path: Path) -> dict[str, Any]:
    """Read the manifest from a complete ``MT+MZ`` or ``MP+MT+MZ`` ALPR package."""
    manifest = read_alprmodel_manifest(package_path)
    if str(manifest.get("schema") or "") != MOBILE_ALPR_PACKAGE_SCHEMA:
        raise ValueError(f"Package is not a complete ALPR package: {package_path}")
    models = dict(manifest.get("models") or {})
    if not models.get("plate") or not models.get("character"):
        raise ValueError(f"Complete ALPR package has no required MT+MZ models: {package_path}")
    return manifest


@dataclass(frozen=True)
class ReportBundleEntry:
    """One safe, indexed entry inside a mobile report archive."""

    name: str
    file_size: int = 0
    compress_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReportBundleValidation:
    """Validation result shown before report metrics."""

    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    checked_hashes: int = 0
    skipped_hashes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _declared_capability(
    capabilities: dict[str, Any],
    name: str,
) -> bool | None:
    value = capabilities.get(name)

    if isinstance(value, bool):
        return value

    return None


def _derive_mobile_report_capabilities(
    sample_schema: dict[str, Any],
    collection_session: dict[str, Any],
    model_refs: dict[str, Any],
    manifest: dict[str, Any],
    checked_hashes: int,
    skipped_hashes: int,
) -> dict[str, Any]:
    declared = sample_schema.get("capabilities", {})

    if not isinstance(declared, dict):
        declared = {}

    collection_complete = collection_session.get(
        "collection_complete"
    )

    if not isinstance(collection_complete, bool):
        collection_complete = None

    raw_hashes = (
        manifest.get("entry_sha256")
        or manifest.get("sha256")
        or {}
    )

    if not isinstance(raw_hashes, dict):
        raw_hashes = {}

    declared_hashes = sum(
        1
        for name, digest in raw_hashes.items()
        if name != "manifest.json"
        and str(digest or "").strip()
    )

    return {
        "attempt_registry": _declared_capability(
            declared,
            "attempt_registry",
        ),
        "mt_invocation_identity": _declared_capability(
            declared,
            "mt_invocation_identity",
        ),
        "mt_input_evidence_references": _declared_capability(
            declared,
            "mt_input_evidence_references",
        ),
        "raw_prediction": _declared_capability(
            declared,
            "raw_prediction",
        ),
        "registration_key": _declared_capability(
            declared,
            "registration_key",
        ),
        "source_timestamp_domain": _declared_capability(
            declared,
            "source_timestamp_domain",
        ),
        "actual_backend_fields": _declared_capability(
            declared,
            "actual_backend_fields",
        ),

        # Nie zgadujemy brakujących możliwości.
        "fresh_mz_result": _declared_capability(
            declared,
            "fresh_mz_result",
        ),

        "model_refs": True if model_refs else None,
        "collection_complete": collection_complete,

        "hash_coverage": {
            "manifest_present": bool(manifest),
            "declared_entries": declared_hashes,
            "checked_entries": checked_hashes,
            "skipped_entries": skipped_hashes,
            "complete": (
                declared_hashes > 0
                and checked_hashes == declared_hashes
                and skipped_hashes == 0
            ),
        },
    }

@dataclass(frozen=True)
class MobileReportBundle:
    """A safely parsed report bundle exported by the Android ALPR client."""

    path: str
    bundle_kind: str
    bundle_schema: str
    report: "MobileBenchmarkReport"
    source_archive_sha256: str = ""
    experiment_session: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    report_payload: dict[str, Any] = field(default_factory=dict)
    pipeline_manifests: dict[str, Any] = field(default_factory=dict)
    model_refs: dict[str, Any] = field(default_factory=dict)
    trace_columns: tuple[str, ...] = field(default_factory=tuple)
    trace_rows: tuple[dict[str, str], ...] = field(default_factory=tuple)
    trace_total: int = 0
    thermal_columns: tuple[str, ...] = field(default_factory=tuple)
    thermal_rows: tuple[dict[str, str], ...] = field(default_factory=tuple)
    thermal_total: int = 0
    frame_flow_columns: tuple[str, ...] = field(default_factory=tuple)
    frame_flow_rows: tuple[dict[str, str], ...] = field(default_factory=tuple)
    frame_flow_total: int = 0
    event_columns: tuple[str, ...] = field(default_factory=tuple)
    event_rows: tuple[dict[str, str], ...] = field(default_factory=tuple)
    event_total: int = 0
    sample_rows: tuple[dict[str, str], ...] = field(default_factory=tuple)
    sample_total: int = 0
    crop_count: int = 0
    annotation_count: int = 0
    sample_schema: dict[str, Any] = field(default_factory=dict)
    scene_evidence: dict[str, Any] = field(default_factory=dict)
    mz_input_evidence: dict[str, Any] = field(default_factory=dict)
    tv_bbox_evidence: dict[str, Any] = field(default_factory=dict)
    f6_evidence: dict[str, Any] = field(default_factory=dict)
    collection_session: dict[str, Any] = field(default_factory=dict)
    attempt_rows: tuple[dict[str, str], ...] = field(default_factory=tuple)
    attempt_total: int = 0
    attempts_available: bool = False
    log_preview: str = ""
    entries: tuple[ReportBundleEntry, ...] = field(default_factory=tuple)

    capabilities: dict[str, Any] = field(
        default_factory=dict
    )

    validation: ReportBundleValidation = field(
        default_factory=lambda: ReportBundleValidation(ok=True)
    )
    imported_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source_archive_sha256": self.source_archive_sha256,
            "bundle_kind": self.bundle_kind,
            "bundle_schema": self.bundle_schema,
            "report": self.report.to_dict(),
            "experiment_session": dict(self.experiment_session or {}),
            "manifest": dict(self.manifest or {}),
            "metadata": dict(self.metadata or {}),
            "pipeline_manifests": dict(self.pipeline_manifests or {}),
            "model_refs": dict(self.model_refs or {}),
            "trace_columns": list(self.trace_columns),
            "trace_rows": [dict(row) for row in self.trace_rows],
            "trace_total": self.trace_total,
            "thermal_columns": list(self.thermal_columns),
            "thermal_rows": [dict(row) for row in self.thermal_rows],
            "thermal_total": self.thermal_total,
            "frame_flow_columns": list(self.frame_flow_columns),
            "frame_flow_rows": [dict(row) for row in self.frame_flow_rows],
            "frame_flow_total": self.frame_flow_total,
            "event_columns": list(self.event_columns),
            "event_rows": [dict(row) for row in self.event_rows],
            "event_total": self.event_total,
            "sample_rows": [dict(row) for row in self.sample_rows],
            "sample_total": self.sample_total,
            "crop_count": self.crop_count,
            "annotation_count": self.annotation_count,
            "sample_schema": dict(self.sample_schema),
            "mz_input_evidence": dict(self.mz_input_evidence),
            "tv_bbox_evidence": dict(self.tv_bbox_evidence),
            "f6_evidence": dict(self.f6_evidence),
            "collection_session": dict(self.collection_session),
            "attempt_rows": list(self.attempt_rows),
            "attempt_total": self.attempt_total,
            "attempts_available": self.attempts_available,
            "log_preview": self.log_preview,
            "entries": [entry.to_dict() for entry in self.entries],
            "capabilities": dict(self.capabilities),
            "validation": self.validation.to_dict(),
            "imported_at": self.imported_at,
        }


def _archive_name_safe(raw_name: str) -> tuple[bool, str]:
    normalized = str(raw_name or "").replace("\\", "/").strip()
    if not normalized:
        return False, normalized
    if normalized.startswith("/") or normalized.startswith("//"):
        return False, normalized
    if re.match(r"^[A-Za-z]:", normalized):
        return False, normalized
    parts = [part for part in normalized.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        return False, normalized
    return True, "/".join(parts)


def _compact_json_text(value: Any, *, limit: int = 900) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _model_provenance_from_pipeline_manifests(
    pipeline_manifests: dict[str, Any],
    model_refs: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    role_keys = (("vehicle", "vehicle"), ("plate", "plate"), ("character", "character"))
    for role, manifest_key in role_keys:
        manifest = pipeline_manifests.get(manifest_key)
        manifest = dict(manifest) if isinstance(manifest, dict) else {}
        ref = model_refs.get(role)
        ref = dict(ref) if isinstance(ref, dict) else {}
        if not manifest and not ref:
            continue
        result[role] = _model_provenance_entry_from_manifest(role, manifest, ref)
    return result


def _model_provenance_entry_from_manifest(
    role: str,
    manifest: dict[str, Any],
    ref: dict[str, Any],
) -> dict[str, Any]:
    source = dict(manifest.get("source") or {})
    model = dict(manifest.get("model") or {})
    training = dict(manifest.get("training") or {})
    ref_training = dict(ref.get("training") or {}) if isinstance(ref.get("training"), dict) else {}
    metrics = dict(manifest.get("metrics") or {})
    dataset = training.get("dataset") if isinstance(training.get("dataset"), dict) else {}
    variants = [dict(variant) for variant in list(manifest.get("variants") or []) if isinstance(variant, dict)]
    primary_variant = variants[0] if variants else {}
    variant_hashes = ref.get("variant_artifact_sha256")
    if not isinstance(variant_hashes, list):
        variant_hashes = []
        sha_map = primary_variant.get("sha256") if isinstance(primary_variant.get("sha256"), dict) else {}
        for digest in dict(sha_map or {}).values():
            text = str(digest or "").strip()
            if text:
                variant_hashes.append(text)
    return {
        "model_id": str(ref.get("model_id") or manifest.get("model_id") or ""),
        "role": role,
        "architecture": str(
            model.get("architecture_label")
            or model.get("family")
            or source.get("architecture_label")
            or source.get("source_model_name")
            or ""
        ),
        "parameter_count": _safe_int(source.get("parameter_count") or model.get("parameter_count")),
        "checkpoint_sha256": str(ref.get("checkpoint_sha256") or source.get("checkpoint_sha256") or ""),
        "package_sha256": str(ref.get("package_sha256") or ""),
        "installed_model_fingerprint": str(ref.get("installed_model_fingerprint") or ""),
        "variant_id": str(ref.get("variant_id") or primary_variant.get("id") or ""),
        "variant_artifact_sha256": [str(item) for item in variant_hashes if str(item or "").strip()],
        "runtime": str(ref.get("runtime") or primary_variant.get("runtime") or ""),
        "precision": str(ref.get("precision") or primary_variant.get("precision") or ""),
        "task": str(manifest.get("task") or ref.get("task") or ""),
        "training": {
            "run_id": str(training.get("run_id") or ref_training.get("run_id") or ""),
            "run_epochs_completed": training.get("run_epochs_completed", ref_training.get("run_epochs_completed")),
            "total_epochs": training.get("total_epochs", ref_training.get("total_epochs")),
            "total_epochs_known": training.get("total_epochs_known", ref_training.get("total_epochs_known")),
            "known_epochs_minimum": training.get("known_epochs_minimum", ref_training.get("known_epochs_minimum")),
            "total_epochs_scope": str(training.get("total_epochs_scope") or ref_training.get("total_epochs_scope") or ""),
            "lineage_total_epochs": training.get("lineage_total_epochs", ref_training.get("lineage_total_epochs")),
            "lineage_total_epochs_known": training.get("lineage_total_epochs_known", ref_training.get("lineage_total_epochs_known")),
            "lineage_stage_count": training.get("lineage_stage_count", ref_training.get("lineage_stage_count")),
            "lineage_stage_count_known": training.get(
                "lineage_stage_count_known",
                ref_training.get("lineage_stage_count_known"),
            ),
            "known_stage_count_minimum": training.get(
                "known_stage_count_minimum",
                ref_training.get("known_stage_count_minimum"),
            ),
            "run_train_images": training.get("run_train_images", ref_training.get("run_train_images")),
            "run_nominal_sample_presentations": training.get(
                "run_nominal_sample_presentations",
                ref_training.get("run_nominal_sample_presentations"),
            ),
            "lineage_nominal_sample_presentations": training.get(
                "lineage_nominal_sample_presentations",
                ref_training.get("lineage_nominal_sample_presentations"),
            ),
            "sample_presentations_known": training.get("sample_presentations_known", ref_training.get("sample_presentations_known")),
            "known_sample_presentations_minimum": training.get(
                "known_sample_presentations_minimum",
                ref_training.get("known_sample_presentations_minimum"),
            ),
            "provenance_capture": str(training.get("provenance_capture") or ref_training.get("provenance_capture") or ""),
            "best_epoch_source": str(training.get("best_epoch_source") or ref_training.get("best_epoch_source") or ""),
            "dataset": {
                "dataset_id": str(dataset.get("dataset_id") or training.get("dataset_id") or ""),
                "manifest_sha256": str(dataset.get("manifest_sha256") or ""),
                "split_sha256": str(dataset.get("split_sha256") or ""),
                "data_yaml_sha256": str(dataset.get("data_yaml_sha256") or ""),
                "train_images": dataset.get("train_images"),
                "val_images": dataset.get("val_images"),
                "test_images": dataset.get("test_images"),
            },
        },
        "metrics": metrics,
        "provenance_status": str(training.get("provenance_status") or ref_training.get("provenance_status") or ref.get("provenance_status") or ""),
    }


def _report_payload_from_thesis_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = dict(metadata or {})
    payload.setdefault("schema", MOBILE_BENCHMARK_REPORT_SCHEMA)
    payload.setdefault(
        "report_id",
        payload.get("report_id")
        or payload.get("bundle_id")
        or payload.get("session_id")
        or payload.get("package_id")
        or "thesis-report",
    )
    payload.setdefault("package_id", payload.get("package_id") or payload.get("model_package_id") or "thesis-package")
    payload.setdefault("variant_id", payload.get("variant_id") or payload.get("runtime") or "thesis")
    if "quality" not in payload and isinstance(payload.get("metrics"), dict):
        payload["quality"] = dict(payload.get("metrics") or {})
    payload.setdefault("raw_metadata_schema", metadata.get("schema", ""))
    return payload


class ReportBundleReader:
    """Open Android report bundles without unsafe extraction or full-image loading."""

    def __init__(
        self,
        *,
        max_text_bytes: int = MOBILE_REPORT_MAX_TEXT_BYTES,
        max_total_uncompressed_bytes: int = MOBILE_REPORT_MAX_TOTAL_UNCOMPRESSED_BYTES,
        max_entries: int = MOBILE_REPORT_MAX_ENTRIES,
        max_trace_rows: int = MOBILE_REPORT_TRACE_PREVIEW_ROWS,
    ):
        self.max_text_bytes = int(max_text_bytes)
        self.max_total_uncompressed_bytes = int(max_total_uncompressed_bytes)
        self.max_entries = int(max_entries)
        self.max_trace_rows = int(max_trace_rows)

    def read(self, path: Path) -> MobileReportBundle:
        safe_path = Path(path)
        if not safe_path.exists() or not safe_path.is_file():
            raise FileNotFoundError(f"Nie znaleziono raportu mobilnego: {safe_path}")
        if self._is_zip_bundle(safe_path):
            return self._read_zip_bundle(safe_path)
        return self._read_json_report(safe_path)

    def read_many(self, path: Path) -> tuple[MobileReportBundle, ...]:
        safe_path = Path(path)
        if not safe_path.exists() or not safe_path.is_file():
            raise FileNotFoundError(f"Nie znaleziono raportu mobilnego: {safe_path}")
        if self._is_zip_bundle(safe_path):
            return (self._read_zip_bundle(safe_path),)
        return self._read_json_bundles(safe_path)

    @staticmethod
    def _is_zip_bundle(path: Path) -> bool:
        if zipfile.is_zipfile(path):
            return True
        # A truncated ZIP lacks its central directory. Do not mistake the
        # entire archive size for a JSON size, or read binary data as text.
        # Older exports are JSON objects/arrays even with .alprsession suffix.
        # Recognize only a text opening; the normal JSON reader retains its size
        # cap and syntax checks. A PK signature can never fall through to JSON.
        with path.open("rb") as stream:
            signature = stream.read(4)
            stream.seek(0)
            opening = stream.read(3)
            if opening != b"\xef\xbb\xbf":
                stream.seek(0)
            first = b""
            remaining = 32 * 1024 * 1024 + 1
            while remaining > 0:
                chunk = stream.read(min(65536, remaining))
                remaining -= len(chunk)
                opening = chunk.lstrip(b" \r\n\t")
                if opening or not chunk:
                    first = opening[:1]
                    break
        looks_json = first in (b"{", b"[")
        if (path.suffix.lower() in (".alprsession", ".zip") and not looks_json) or signature.startswith(b"PK"):
            raise ValueError(
                "Import integralności: archiwum ZIP jest niekompletne lub uszkodzone. "
                "Nie można odczytać katalogu ZIP. "
                "Wskaż kompletną kopię lub ponowny eksport tej samej zapisanej sesji. "
                f"Plik: {path.name} ({path.stat().st_size} B)."
            )
        return False

    def _read_json_report(self, path: Path) -> MobileReportBundle:
        bundles = self._read_json_bundles(path)
        if not bundles:
            raise ValueError("Plik JSON nie zawiera raportów.")
        return bundles[0]

    def _read_json_bundles(self, path: Path) -> tuple[MobileReportBundle, ...]:
        size = path.stat().st_size
        if size > self.max_text_bytes:
            raise ValueError(f"Plik raportu JSON jest za duży do bezpiecznego podglądu: {size} B")
        source_hash = _file_sha256(path)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict) and str(payload.get("schema") or "") == MOBILE_PACKAGE_EXPERIMENT_SCHEMA:
            raise ValueError(
                "Wybrany plik jest wewnetrznym magazynem eksperymentow desktopa, "
                "a nie raportem z aplikacji mobilnej."
            )
        payloads = self._json_report_payloads(payload)
        return tuple(self._json_payload_to_bundle(path, item, source_hash) for item in payloads)

    def _json_report_payloads(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            if not payload:
                raise ValueError("Plik JSON nie zawiera raportów.")
            reports = payload
        elif isinstance(payload, dict) and isinstance(payload.get("reports"), list):
            reports = list(payload.get("reports") or [])
            if not reports:
                raise ValueError("Plik JSON nie zawiera raportów.")
        elif isinstance(payload, dict):
            reports = [payload]
        else:
            raise ValueError("Raport JSON musi być obiektem albo listą obiektów.")
        result = [dict(item or {}) for item in reports if isinstance(item, dict)]
        if not result:
            raise ValueError("Plik JSON nie zawiera poprawnych obiektów raportów.")
        return result

    def _json_payload_to_bundle(self, path: Path, payload: dict[str, Any], source_hash: str) -> MobileReportBundle:
        if isinstance(payload, dict) and isinstance(payload.get("reports"), list):
            reports = list(payload.get("reports") or [])
            if not reports:
                raise ValueError("Plik JSON nie zawiera raportów.")
            payload = dict(reports[0] or {})
        if not isinstance(payload, dict):
            raise ValueError("Raport JSON musi być obiektem albo listą obiektów.")
        report = MobileBenchmarkReport.from_dict(payload)
        traces, columns, trace_total = self._trace_rows_from_json(payload.get("traces"))
        thermal_rows, thermal_columns, thermal_total = self._rows_from_json_records(
            payload.get("thermal") or payload.get("thermal_samples") or payload.get("thermal_trace")
        )
        frame_flow_rows, frame_flow_columns, frame_flow_total = self._rows_from_json_records(
            payload.get("frame_flow") or payload.get("frame_flow_buckets") or payload.get("flow")
        )
        event_rows, event_columns, event_total = self._rows_from_json_records(
            payload.get("events") or payload.get("event_stream")
        )
        sample_rows, _sample_columns, sample_total = self._rows_from_json_records(
            payload.get("samples") or _nested_value(payload, "crop_session.records"),
            max_rows=1000,
        )
        crop_count = _safe_int(_nested_value(payload, "crop_session.collected_count", "samples.crop_count"))
        annotation_count = _safe_int(_nested_value(payload, "crop_session.annotation_count", "samples.annotation_count"))
        bundle_schema = str(payload.get("schema") or MOBILE_BENCHMARK_REPORT_SCHEMA)
        validation = ReportBundleValidation(
            ok=bundle_schema == MOBILE_BENCHMARK_REPORT_SCHEMA,
            warnings=()
            if bundle_schema == MOBILE_BENCHMARK_REPORT_SCHEMA
            else (f"Nieoczekiwany schemat raportu: {payload.get('schema')}",),
        )
        artifact_flags = {
            "has_traces": bool(trace_total),
            "has_thermal": bool(thermal_total),
            "has_frame_flow": bool(frame_flow_total),
            "has_events": bool(event_total),
            "has_samples": bool(sample_total or payload.get("samples") or payload.get("crop_session")),
            "has_log": bool(payload.get("application_log") or payload.get("log")),
        }
        artifact_counts = {
            "traces": trace_total,
            "thermal": thermal_total,
            "frame_flow": frame_flow_total,
            "events": event_total,
            "samples": sample_total,
            "crops": crop_count,
            "annotations": annotation_count,
        }
        report, experiment_session = _attach_report_ingest_metadata(
            report,
            path=path,
            source_archive_sha256=source_hash,
            bundle_kind="json",
            bundle_schema=bundle_schema,
            trace_total=trace_total,
            trace_rows_preview=len(traces),
            validation=validation,
            artifact_flags=artifact_flags,
            artifact_counts=artifact_counts,
        )
        return MobileReportBundle(
            path=str(path),
            source_archive_sha256=source_hash,
            bundle_kind="json",
            bundle_schema=bundle_schema,
            report=report,
            experiment_session=experiment_session,
            report_payload=payload,
            trace_columns=tuple(columns),
            trace_rows=tuple(traces),
            trace_total=trace_total,
            thermal_columns=tuple(thermal_columns),
            thermal_rows=tuple(thermal_rows),
            thermal_total=thermal_total,
            frame_flow_columns=tuple(frame_flow_columns),
            frame_flow_rows=tuple(frame_flow_rows),
            frame_flow_total=frame_flow_total,
            event_columns=tuple(event_columns),
            event_rows=tuple(event_rows),
            event_total=event_total,
            sample_rows=tuple(sample_rows),
            sample_total=sample_total,
            crop_count=crop_count,
            annotation_count=annotation_count,
            validation=validation,
        )

    def _read_zip_bundle(self, path: Path) -> MobileReportBundle:
        errors: list[str] = []
        warnings: list[str] = []
        checked_hashes = 0
        skipped_hashes = 0
        source_hash = _file_sha256(path)
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if len(infos) > self.max_entries:
                errors.append(f"Archiwum ma zbyt dużo wpisów: {len(infos)}.")
            normalized_names: dict[str, zipfile.ZipInfo] = {}
            entries: list[ReportBundleEntry] = []
            total_uncompressed = 0
            for info in infos:
                ok, normalized = _archive_name_safe(info.filename)
                if not ok:
                    errors.append(f"Niebezpieczna ścieżka w archiwum: {info.filename}")
                    continue
                if normalized in normalized_names:
                    errors.append(f"Zduplikowany wpis w archiwum: {normalized}")
                    continue
                normalized_names[normalized] = info
                total_uncompressed += int(info.file_size or 0)
                entries.append(
                    ReportBundleEntry(
                        name=normalized,
                        file_size=int(info.file_size or 0),
                        compress_size=int(info.compress_size or 0),
                    )
                )
            if total_uncompressed > self.max_total_uncompressed_bytes:
                errors.append(
                    "Archiwum deklaruje zbyt duży rozmiar po rozpakowaniu: "
                    f"{total_uncompressed / (1024 * 1024):.1f} MB."
                )
            embedded_model_entries = [
                name
                for name in normalized_names
                if name.lower().endswith((".pt", ".tflite", ".onnx", ".param", ".bin", ".alprmodel"))
            ]

            def read_text(name: str, *, optional: bool = False, limit: int | None = None) -> str:
                info = normalized_names.get(name)
                if info is None:
                    if not optional:
                        errors.append(f"Brakuje wpisu {name}.")
                    return ""
                max_bytes = int(limit or self.max_text_bytes)
                if int(info.file_size or 0) > max_bytes:
                    warnings.append(f"Pominięto zbyt duży wpis tekstowy {name}.")
                    return ""
                with archive.open(info, "r") as handle:
                    return handle.read(max_bytes + 1).decode("utf-8-sig", errors="replace")

            def read_json(name: str, *, optional: bool = False) -> dict[str, Any]:
                text = read_text(name, optional=optional)
                if not text:
                    return {}
                try:
                    value = json.loads(text)
                    if isinstance(value, dict):
                        return value
                    warnings.append(f"Wpis {name} nie jest obiektem JSON.")
                except Exception as exc:
                    errors.append(f"Nie udało się odczytać JSON {name}: {exc}")
                return {}

            manifest = read_json("manifest.json", optional=True)
            bundle_schema = str(manifest.get("schema") or "")

            supported_bundle_schemas = {
                MOBILE_RESEARCH_BUNDLE_SCHEMA,
                MOBILE_THESIS_BUNDLE_SCHEMA,
                MOBILE_BENCHMARK_REPORT_SCHEMA,
            }

            if manifest and bundle_schema not in supported_bundle_schemas:
                errors.append(
                    f"Nieobsługiwany schemat manifestu raportu: "
                    f"{bundle_schema or 'brak'}."
                )

            metadata = read_json("metadata.json", optional=True)
            collection_session = read_json("session.json", optional=True)
            if collection_session.get("schema") == "alpr_crop_session_v1":
                errors.append(
                    "To paczka Akwizycji Androida, a nie sesja badawcza. "
                    "Otwórz ją przez Z3 > Akwizycja Androida. "
                    "Selekcja niepustych MZ nie pozwala ocenić jakości całego pipeline'u."
                )
            report_payload = read_json("report.json", optional=True)
            if not report_payload and metadata:
                report_payload = _report_payload_from_thesis_metadata(metadata)

            pipeline_manifests: dict[str, Any] = {}
            for key, entry_name in (
                ("package", "pipeline/package_manifest.json"),
                ("vehicle", "pipeline/vehicle_manifest.json"),
                ("plate", "pipeline/plate_manifest.json"),
                ("character", "pipeline/character_manifest.json"),
            ):
                payload = read_json(entry_name, optional=True)
                if payload:
                    pipeline_manifests[key] = payload
            model_refs = read_json("pipeline/model_refs.json", optional=True)
            if not model_refs:
                package_manifest = pipeline_manifests.get("package")
                if isinstance(package_manifest, dict) and isinstance(package_manifest.get("model_refs"), dict):
                    model_refs = dict(package_manifest.get("model_refs") or {})

            if not report_payload:
                errors.append("Archiwum nie zawiera czytelnego report.json ani metadata.json.")
            elif not isinstance(report_payload.get("model_provenance"), dict):
                rebuilt_provenance = _model_provenance_from_pipeline_manifests(
                    pipeline_manifests,
                    model_refs,
                )
                if rebuilt_provenance:
                    report_payload = dict(report_payload)
                    report_payload["model_provenance"] = rebuilt_provenance
                    warnings.append("Uzupełniono model_provenance na podstawie manifestów pipeline.")
            if report_payload and model_refs and not isinstance(report_payload.get("pipeline_model_refs"), dict):
                report_payload = dict(report_payload)
                report_payload["pipeline_model_refs"] = dict(model_refs)

            report_schema = str(report_payload.get("schema") or "")
            if report_payload and report_schema != MOBILE_BENCHMARK_REPORT_SCHEMA:
                warnings.append(f"Nieoczekiwany schemat report.json: {report_schema or 'brak'}.")

            if manifest:
                hash_errors, hash_warnings, checked_hashes, skipped_hashes = self._verify_manifest_hashes(
                    archive,
                    normalized_names,
                    manifest,
                )
                errors.extend(hash_errors)
                warnings.extend(hash_warnings)
            elif report_payload:
                warnings.append("Brak manifest.json, więc sprawdzono tylko strukturę raportu.")

            traces, trace_columns, trace_total = self._read_csv_from_zip(
                archive,
                normalized_names,
                "traces.csv" if "traces.csv" in normalized_names else "tables/trace_data.csv",
                optional=True,
            )
            if not traces and report_payload:
                traces, trace_columns, trace_total = self._trace_rows_from_json(report_payload.get("traces"))

            thermal_rows, thermal_columns, thermal_total, _thermal_source = self._read_first_csv_from_zip(
                archive,
                normalized_names,
                ("thermal.csv", "tables/thermal.csv", "tables/thermal_data.csv"),
                max_rows=self.max_trace_rows,
            )
            if not thermal_rows and report_payload:
                thermal_rows, thermal_columns, thermal_total = self._rows_from_json_records(
                    report_payload.get("thermal") or report_payload.get("thermal_samples") or report_payload.get("thermal_trace")
                )

            frame_flow_rows, frame_flow_columns, frame_flow_total, _frame_flow_source = self._read_first_csv_from_zip(
                archive,
                normalized_names,
                ("frame_flow.csv", "tables/frame_flow.csv", "tables/frame_flow_data.csv"),
                max_rows=self.max_trace_rows,
            )
            if not frame_flow_rows and report_payload:
                frame_flow_rows, frame_flow_columns, frame_flow_total = self._rows_from_json_records(
                    report_payload.get("frame_flow") or report_payload.get("frame_flow_buckets") or report_payload.get("flow")
                )

            event_rows, event_columns, event_total, _event_source = self._read_first_csv_from_zip(
                archive,
                normalized_names,
                ("events.csv", "tables/events.csv", "tables/event_data.csv"),
                max_rows=self.max_trace_rows,
            )
            if not event_rows:
                event_rows, event_columns, event_total, _event_source = self._read_first_jsonl_from_zip(
                    archive,
                    normalized_names,
                    ("events.jsonl", "tables/events.jsonl", "event_stream.jsonl"),
                    max_rows=self.max_trace_rows,
                )
            if not event_rows and report_payload:
                event_rows, event_columns, event_total = self._rows_from_json_records(
                    report_payload.get("events") or report_payload.get("event_stream")
                )

            sample_rows, _sample_columns, sample_total = self._read_csv_from_zip(
                archive,
                normalized_names,
                "samples/index.csv",
                optional=True,
                max_rows=1000,
            )
            sample_schema = read_json("samples/schema.json", optional=True)
            attempt_rows, _attempt_columns, attempt_total = self._read_csv_from_zip(
                archive, normalized_names, "samples/attempts.csv", optional=True, max_rows=1000,
            )
            attempts_available = "samples/attempts.csv" in normalized_names
            crop_count = sum(
                1
                for name in normalized_names
                if name.startswith("samples/crops/") and name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            )
            annotation_count = self._count_text_lines(
                archive,
                normalized_names.get("samples/annotations.jsonl"),
            )
            log_preview = read_text("application.log", optional=True, limit=512 * 1024)

            if not bundle_schema:
                if path.name.lower().endswith(".alprsession"):
                    bundle_schema = MOBILE_RESEARCH_BUNDLE_SCHEMA
                elif "tables/trace_data.csv" in normalized_names:
                    bundle_schema = MOBILE_THESIS_BUNDLE_SCHEMA
                else:
                    bundle_schema = MOBILE_BENCHMARK_REPORT_SCHEMA

            if bundle_schema == MOBILE_RESEARCH_BUNDLE_SCHEMA:
                bundle_kind = "alprsession"
            elif bundle_schema == MOBILE_THESIS_BUNDLE_SCHEMA:
                bundle_kind = "thesis"
            elif bundle_schema == MOBILE_BENCHMARK_REPORT_SCHEMA:
                bundle_kind = "legacy_zip"
            else:
                bundle_kind = "unknown"

            from .mobile_f6_execution_evidence import read_extension as read_f6_extension
            f6_evidence = dict(available=False, status='ABSENT')
            try:
                f6_evidence = read_f6_extension(archive, sample_schema, collection_session, manifest, report_payload)
                if errors and f6_evidence.get('available'):
                    raise ValueError('ARCHIVE_INTEGRITY_FAILURE')
                if f6_evidence.get('status') == 'PARTIAL':warnings.append('F6: niekompletny dziennik wykonania; brak dowodu pozostaje N/A.')
            except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
                errors.append('F6 execution evidence contract: ' + str(exc))
                f6_evidence = dict(available=False, status='INVALID', reason=str(exc))
            scene_evidence = dict(available=False, status='ABSENT', scenes=(), roles=())
            if errors and any((sample_schema.get('capabilities') or {}).get(k) for k in ('scene_evidence_v1','mt_detection_roles_v1')):
                scene_evidence = dict(available=False, status='INVALID', scenes=(), roles=(), reason='ARCHIVE_INTEGRITY_FAILURE')
            if not errors:
                from .mobile_scene_evidence import read_extension
                try:
                    scene_evidence = read_extension(archive, sample_schema, collection_session, manifest, report_payload, f6=f6_evidence)
                    if scene_evidence['status'] == 'PARTIAL':
                        warnings.append('F5A: niekompletne dowody scen/ról; brakujące jednostki pozostają N/A.')
                except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
                    errors.append('F5A evidence contract: ' + str(exc))
                    scene_evidence = dict(available=False, status='INVALID', scenes=(), roles=(), reason=str(exc))
            from .mobile_mz_input_evidence import read_extension as read_mz_extension, absent as mz_absent
            mz_input_evidence = mz_absent(collection_session.get('measurement_purpose'))
            try:
                mz_input_evidence = read_mz_extension(archive, sample_schema, collection_session, manifest, report_payload)
                if errors and mz_input_evidence.get('available'):
                    raise ValueError('ARCHIVE_INTEGRITY_FAILURE')
                if mz_input_evidence['status'] == 'PARTIAL':
                    warnings.append('MZ: niekompletny dowód wejścia; brak dowodu pozostaje N/A.')
            except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
                errors.append('MZ input evidence contract: ' + str(exc))
                mz_input_evidence = dict(available=False, status='INVALID', rows=[], dependencies={}, reason=str(exc))
            from .mobile_tv_bbox_evidence import read_extension as read_tv_extension, absent as tv_absent
            tv_bbox_evidence = tv_absent()
            try:
                tv_bbox_evidence = read_tv_extension(archive, sample_schema, collection_session, manifest, report_payload, f6=f6_evidence)
                if errors and tv_bbox_evidence.get('available'):
                    raise ValueError('ARCHIVE_INTEGRITY_FAILURE')
                if tv_bbox_evidence['status'] == 'PARTIAL':
                    warnings.append('TV_BBOX_CROP: niekompletne dowody wejścia; brak pozostaje N/A.')
            except (ValueError, KeyError, TypeError, AttributeError, IndexError, ArithmeticError, OSError, zipfile.BadZipFile) as exc:
                errors.append('TV BBOX evidence contract: ' + str(exc))
                tv_bbox_evidence = dict(available=False, status='INVALID', rows=[], dependencies={}, reason=str(exc))
            if f6_evidence.get('available') and not errors:
                from .mobile_f6_execution_evidence import validate_bindings
                try:
                    validate_bindings(archive, manifest, collection_session, f6_evidence, tv_bbox_evidence, scene_evidence)
                except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
                    errors.append('F6 binding contract: ' + str(exc))
                    f6_evidence = dict(available=False, status='INVALID', reason=str(exc))
            tensor_entries = set(mz_input_evidence.get('tensor_payloads', ()))
            if any(name not in tensor_entries for name in embedded_model_entries):
                warnings.append(
                    "Raport zawiera binarne artefakty modeli. Nowy lekki standard .alprsession "
                    "powinien przechowywać tylko manifesty pipeline i referencje SHA-256."
                )
            validation = ReportBundleValidation(
                ok=not errors,
                errors=tuple(errors),
                warnings=tuple(warnings),
                checked_hashes=checked_hashes,
                skipped_hashes=skipped_hashes,
            )

            capabilities = _derive_mobile_report_capabilities(
                sample_schema=sample_schema,
                collection_session=collection_session,
                model_refs=model_refs,
                manifest=manifest,
                checked_hashes=checked_hashes,
                skipped_hashes=skipped_hashes,
            )
            if scene_evidence.get('available'):
                capabilities.update(scene_evidence_v1=True, mt_detection_roles_v1=True,
                                    scene_evidence_status=scene_evidence['status'])
            capabilities.update(mz_input_evidence_v1=mz_input_evidence.get('available', False),
                                mz_input_evidence_status=mz_input_evidence['status'])
            if tv_bbox_evidence.get('available'):
                capabilities.update(tv_bbox_crop_v1=True, tv_bbox_status=tv_bbox_evidence['status'])
            if f6_evidence.get('available'):
                capabilities.update(f6_final180_v1=True, f6_status=f6_evidence['status'])
            artifact_flags = {
                "has_traces": bool(trace_total),
                "has_thermal": bool(thermal_total),
                "has_frame_flow": bool(frame_flow_total),
                "has_events": bool(event_total),
                "has_samples": bool(sample_total or crop_count or annotation_count or attempt_total),
                "has_mt_attempts": attempts_available,
                "has_log": bool(log_preview),
            }
            artifact_counts = {
                "traces": trace_total,
                "thermal": thermal_total,
                "frame_flow": frame_flow_total,
                "events": event_total,
                "samples": sample_total,
                "crops": crop_count,
                "annotations": annotation_count,
                "attempts": attempt_total,
            }
            report = MobileBenchmarkReport.from_dict(report_payload or {})
            if not report_payload.get('measured_at'):
                # Some archives do not declare this report date.
                # Keep it unavailable instead of generating a different identity
                # on each native source re-open.
                report = replace(report, measured_at='')
            report, experiment_session = _attach_report_ingest_metadata(
                report,
                path=path,
                source_archive_sha256=source_hash,
                bundle_kind=bundle_kind,
                bundle_schema=bundle_schema,
                trace_total=trace_total,
                trace_rows_preview=len(traces),
                validation=validation,
                artifact_flags=artifact_flags,
                artifact_counts=artifact_counts,
            )
            return MobileReportBundle(
                path=str(path),
                source_archive_sha256=source_hash,
                bundle_kind=bundle_kind,
                bundle_schema=bundle_schema,
                report=report,
                experiment_session=experiment_session,
                manifest=manifest,
                metadata=metadata,
                report_payload=report_payload,
                pipeline_manifests=pipeline_manifests,
                model_refs=model_refs,
                trace_columns=tuple(trace_columns),
                trace_rows=tuple(traces),
                trace_total=trace_total,
                thermal_columns=tuple(thermal_columns),
                thermal_rows=tuple(thermal_rows),
                thermal_total=thermal_total,
                frame_flow_columns=tuple(frame_flow_columns),
                frame_flow_rows=tuple(frame_flow_rows),
                frame_flow_total=frame_flow_total,
                event_columns=tuple(event_columns),
                event_rows=tuple(event_rows),
                event_total=event_total,
                sample_rows=tuple(sample_rows),
                sample_total=sample_total,
                crop_count=crop_count,
                annotation_count=annotation_count,
                sample_schema=sample_schema,
                collection_session=collection_session,
                attempt_rows=tuple(attempt_rows),
                scene_evidence=scene_evidence,
                mz_input_evidence=mz_input_evidence,
                tv_bbox_evidence=tv_bbox_evidence,
                f6_evidence=f6_evidence,
                attempt_total=attempt_total,
                attempts_available=attempts_available,
                log_preview=log_preview,
                entries=tuple(entries),
                capabilities=capabilities,
                validation=validation
            )

    def _verify_manifest_hashes(
        self,
        archive: zipfile.ZipFile,
        normalized_names: dict[str, zipfile.ZipInfo],
        manifest: dict[str, Any],
    ) -> tuple[list[str], list[str], int, int]:
        errors: list[str] = []
        warnings: list[str] = []
        checked = 0
        skipped = 0
        raw_hashes = manifest.get("entry_sha256") or manifest.get("sha256") or {}
        if not isinstance(raw_hashes, dict):
            warnings.append("Manifest nie zawiera słownika entry_sha256.")
            return errors, warnings, checked, skipped
        for raw_name, expected in raw_hashes.items():
            ok, normalized = _archive_name_safe(str(raw_name or ""))
            if not ok or normalized == "manifest.json":
                continue
            info = normalized_names.get(normalized)
            if info is None:
                errors.append(f"Manifest wymienia brakujący wpis: {normalized}")
                continue
            expected_hash = str(expected or "").strip().lower()
            if not expected_hash:
                skipped += 1
                continue
            digest = hashlib.sha256()
            try:
                with archive.open(info, "r") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        if not chunk:
                            break
                        digest.update(chunk)
                checked += 1
                actual = digest.hexdigest().lower()
                if actual != expected_hash:
                    errors.append(f"SHA-256 nie zgadza się dla {normalized}.")
            except Exception as exc:
                errors.append(f"Nie udało się policzyć SHA-256 dla {normalized}: {exc}")
        return errors, warnings, checked, skipped

    def _read_csv_from_zip(
        self,
        archive: zipfile.ZipFile,
        normalized_names: dict[str, zipfile.ZipInfo],
        name: str,
        *,
        optional: bool = False,
        max_rows: int | None = None,
    ) -> tuple[list[dict[str, str]], list[str], int]:
        info = normalized_names.get(name)
        if info is None:
            if not optional:
                raise FileNotFoundError(name)
            return [], [], 0
        rows: list[dict[str, str]] = []
        columns: list[str] = []
        total = 0
        limit = self.max_trace_rows if max_rows is None else int(max_rows)
        with archive.open(info, "r") as binary:
            wrapper = io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="")
            reader = csv.DictReader(wrapper)
            columns = [str(item or "") for item in (reader.fieldnames or [])]
            for row in reader:
                total += 1
                if len(rows) < limit:
                    rows.append({str(key or ""): str(value or "") for key, value in dict(row or {}).items()})
        return rows, columns, total

    def _read_first_csv_from_zip(
        self,
        archive: zipfile.ZipFile,
        normalized_names: dict[str, zipfile.ZipInfo],
        names: tuple[str, ...],
        *,
        max_rows: int | None = None,
    ) -> tuple[list[dict[str, str]], list[str], int, str]:
        for name in names:
            if name not in normalized_names:
                continue
            rows, columns, total = self._read_csv_from_zip(
                archive,
                normalized_names,
                name,
                optional=True,
                max_rows=max_rows,
            )
            return rows, columns, total, name
        return [], [], 0, ""

    def _read_jsonl_from_zip(
        self,
        archive: zipfile.ZipFile,
        normalized_names: dict[str, zipfile.ZipInfo],
        name: str,
        *,
        max_rows: int | None = None,
    ) -> tuple[list[dict[str, str]], list[str], int]:
        info = normalized_names.get(name)
        if info is None:
            return [], [], 0
        rows: list[dict[str, str]] = []
        columns: list[str] = []
        seen: set[str] = set()
        total = 0
        limit = self.max_trace_rows if max_rows is None else int(max_rows)
        with archive.open(info, "r") as binary:
            wrapper = io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="")
            for line in wrapper:
                text = str(line or "").strip()
                if not text:
                    continue
                total += 1
                try:
                    item = json.loads(text)
                except Exception:
                    continue
                if not isinstance(item, dict):
                    continue
                row = self._json_record_to_row(item)
                for key in row:
                    if key not in seen:
                        seen.add(key)
                        columns.append(key)
                if len(rows) < limit:
                    rows.append(row)
        return rows, columns, total

    def _read_first_jsonl_from_zip(
        self,
        archive: zipfile.ZipFile,
        normalized_names: dict[str, zipfile.ZipInfo],
        names: tuple[str, ...],
        *,
        max_rows: int | None = None,
    ) -> tuple[list[dict[str, str]], list[str], int, str]:
        for name in names:
            if name not in normalized_names:
                continue
            rows, columns, total = self._read_jsonl_from_zip(
                archive,
                normalized_names,
                name,
                max_rows=max_rows,
            )
            return rows, columns, total, name
        return [], [], 0, ""

    def _count_text_lines(self, archive: zipfile.ZipFile, info: zipfile.ZipInfo | None) -> int:
        if info is None:
            return 0
        total = 0
        with archive.open(info, "r") as handle:
            for _line in handle:
                total += 1
        return total

    def _json_record_to_row(self, item: dict[str, Any]) -> dict[str, str]:
        row: dict[str, str] = {}
        for key, value in dict(item or {}).items():
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    row[f"{key}.{child_key}"] = _compact_json_text(child_value, limit=180)
            elif isinstance(value, list):
                row[str(key)] = _compact_json_text(value, limit=220)
            else:
                row[str(key)] = "" if value is None else str(value)
        return row

    def _rows_from_json_records(
        self,
        records_value: Any,
        *,
        max_rows: int | None = None,
    ) -> tuple[list[dict[str, str]], list[str], int]:
        if isinstance(records_value, dict):
            for key in ("records", "rows", "samples", "events", "traces", "data"):
                nested = records_value.get(key)
                if isinstance(nested, list):
                    records_value = nested
                    break
        if not isinstance(records_value, list):
            return [], [], 0
        rows: list[dict[str, str]] = []
        columns: list[str] = []
        seen: set[str] = set()
        limit = self.max_trace_rows if max_rows is None else int(max_rows)
        total = 0
        for item in records_value:
            if not isinstance(item, dict):
                continue
            total += 1
            row = self._json_record_to_row(item)
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
            if len(rows) < limit:
                rows.append(row)
        return rows, columns, total

    def _trace_rows_from_json(self, traces_value: Any) -> tuple[list[dict[str, str]], list[str], int]:
        if isinstance(traces_value, dict):
            for key in ("records", "rows", "traces", "data"):
                nested = traces_value.get(key)
                if isinstance(nested, list):
                    traces_value = nested
                    break
        if not isinstance(traces_value, list):
            return [], [], 0
        rows: list[dict[str, str]] = []
        columns: list[str] = []
        seen: set[str] = set()
        for item in traces_value:
            if not isinstance(item, dict):
                continue
            row: dict[str, str] = {}
            for key in ("frame_id", "timestamp_ms", "status", "text"):
                row[key] = str(item.get(key, ""))
            for nested_name, suffix in (("stage_ms", "_ms"), ("confidence", ""), ("counters", ""), ("memory", "")):
                nested = item.get(nested_name)
                if not isinstance(nested, dict):
                    continue
                for key, value in nested.items():
                    column = str(key)
                    if suffix and not column.endswith(suffix):
                        column = f"{column}{suffix}"
                    row[column] = str(value)
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
            if len(rows) < self.max_trace_rows:
                rows.append(row)
        return rows, columns, len([item for item in traces_value if isinstance(item, dict)])


def read_mobile_report_bundle(path: Path, *, max_trace_rows: int = MOBILE_REPORT_TRACE_PREVIEW_ROWS) -> MobileReportBundle:
    """Read an Android report bundle according to the mobile-report handoff."""
    return ReportBundleReader(max_trace_rows=max_trace_rows).read(path)


def read_mobile_report_bundles(
    path: Path,
    *,
    max_trace_rows: int = MOBILE_REPORT_TRACE_PREVIEW_ROWS,
) -> tuple[MobileReportBundle, ...]:
    """Read one or many Android report bundles from a single selected file."""
    return ReportBundleReader(max_trace_rows=max_trace_rows).read_many(path)


def iter_full_trace_rows(bundle_or_path) -> Any:
    """Stream all trace rows from a report source, not only the UI preview."""
    yield from _iter_full_report_rows(bundle_or_path, "trace")


def iter_full_thermal_rows(bundle_or_path) -> Any:
    """Stream all thermal rows from a report source, not only the UI preview."""
    yield from _iter_full_report_rows(bundle_or_path, "thermal")


def iter_full_frame_flow_rows(bundle_or_path) -> Any:
    """Stream all frame-flow rows from a report source, not only the UI preview."""
    yield from _iter_full_report_rows(bundle_or_path, "frame_flow")


def iter_full_event_rows(bundle_or_path) -> Any:
    """Stream all event rows from a report source, not only the UI preview."""
    yield from _iter_full_report_rows(bundle_or_path, "events")


def iter_full_sample_rows(bundle_or_path) -> Any:
    """Stream all sample-index rows from a report source, not only the UI preview."""
    yield from _iter_full_report_rows(bundle_or_path, "samples")


def iter_full_attempt_rows(bundle_or_path) -> Any:
    """Stream the full mobile MT attempt register, including attempts without crops."""
    yield from _iter_full_report_rows(bundle_or_path, "attempts")


def iter_full_sample_annotations(bundle_or_path) -> Any:
    """Read original annotations without flattening nested evidence metadata."""
    path = _full_report_source_path(bundle_or_path)
    if path is None or not zipfile.is_zipfile(path):
        return
    with zipfile.ZipFile(path, "r") as archive:
        entries = _normalized_zip_entries_for_full_read(archive)
        info = entries.get("samples/annotations.jsonl")
        if info is None:
            return
        with archive.open(info, "r") as binary:
            while True:
                line = binary.readline(MOBILE_REPORT_MAX_TEXT_BYTES + 1)
                if not line:
                    break
                if len(line) > MOBILE_REPORT_MAX_TEXT_BYTES:
                    raise ValueError("Zbyt duży rekord adnotacji próbki.")
                if line.strip():
                    record = json.loads(line.decode("utf-8-sig"))
                    if not isinstance(record, dict):
                        raise ValueError("Adnotacja próbki nie jest obiektem JSON.")
                    yield record


def read_mobile_sample_image(bundle_or_path, entry_name: str, *, max_entry_bytes: int = 20 * 1024 * 1024,
                             max_dimension: int = 12000, max_pixels: int = 40_000_000):
    """Decode one bounded crop/evidence image through the shared archive validator."""
    from PIL import Image
    safe, name = _archive_name_safe(str(entry_name))
    if not safe or not name.startswith(("samples/crops/", "samples/evidence/")):
        raise ValueError("Obraz musi należeć do cropów lub dowodów sesji.")
    path = _full_report_source_path(bundle_or_path)
    with zipfile.ZipFile(path, "r") as archive:
        entries = _normalized_zip_entries_for_full_read(archive)
        info = entries.get(name)
        if info is None:
            raise FileNotFoundError(f"Brak obrazu w sesji: {name}")
        if info.file_size > max_entry_bytes:
            raise ValueError("Obraz przekracza limit rozmiaru wpisu.")
        with archive.open(info, "r") as handle:
            data = handle.read(max_entry_bytes + 1)
        if len(data) > max_entry_bytes:
            raise ValueError("Obraz przekracza limit rozmiaru wpisu.")
    with Image.open(io.BytesIO(data)) as source:
        width, height = source.size
        if max(width, height) > max_dimension or width * height > max_pixels:
            raise ValueError("Obraz przekracza limit wymiarów po dekodowaniu.")
        source.load()
        return source.convert("RGB")


_FULL_ROW_SOURCES = {
    "trace": {
        "csv": ("traces.csv", "tables/trace_data.csv"),
        "json": ("traces",),
    },
    "thermal": {
        "csv": ("thermal.csv", "tables/thermal.csv", "tables/thermal_data.csv"),
        "json": ("thermal", "thermal_samples", "thermal_trace"),
    },
    "frame_flow": {
        "csv": ("frame_flow.csv", "tables/frame_flow.csv", "tables/frame_flow_data.csv"),
        "json": ("frame_flow", "frame_flow_buckets", "flow"),
    },
    "events": {
        "csv": ("events.csv", "tables/events.csv", "tables/event_data.csv"),
        "jsonl": ("events.jsonl", "tables/events.jsonl", "event_stream.jsonl"),
        "json": ("events", "event_stream"),
    },
    "samples": {
        "csv": ("samples/index.csv",),
        "json": ("samples", "crop_session.records"),
    },
    "attempts": {"csv": ("samples/attempts.csv",), "json": ("attempts",)},
}


def _iter_full_report_rows(bundle_or_path, kind: str) -> Any:
    source_path = _full_report_source_path(bundle_or_path)
    if source_path is not None and source_path.exists() and zipfile.is_zipfile(source_path):
        yielded = False
        for row in _iter_full_zip_rows(source_path, kind):
            yielded = True
            yield row
        if yielded:
            return

    payload = _full_report_payload(bundle_or_path)
    if payload is not None:
        yield from _iter_full_json_rows(payload, kind)
        return

    if source_path is not None and source_path.exists() and source_path.is_file():
        for payload_item in _read_full_json_payloads(source_path):
            yield from _iter_full_json_rows(payload_item, kind)


def _full_report_source_path(bundle_or_path) -> Path | None:
    if isinstance(bundle_or_path, (str, Path)):
        return Path(bundle_or_path)
    raw_path = getattr(bundle_or_path, "path", "")
    if raw_path:
        try:
            return Path(raw_path)
        except Exception:
            return None
    raw_report_path = getattr(bundle_or_path, "source_path", "")
    if raw_report_path:
        try:
            return Path(raw_report_path)
        except Exception:
            return None
    return None


def _full_report_payload(bundle_or_path) -> dict[str, Any] | None:
    payload = getattr(bundle_or_path, "report_payload", None)
    if isinstance(payload, dict):
        return payload
    raw = getattr(bundle_or_path, "raw", None)
    if isinstance(raw, dict):
        return raw
    report = getattr(bundle_or_path, "report", None)
    raw = getattr(report, "raw", None)
    if isinstance(raw, dict):
        return raw
    return None


def _read_full_json_payloads(path: Path) -> list[dict[str, Any]]:
    if path.stat().st_size > MOBILE_REPORT_MAX_TEXT_BYTES:
        raise ValueError(f"Plik raportu JSON jest za duży do bezpiecznego odczytu: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return [dict(item or {}) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("reports"), list):
        return [dict(item or {}) for item in payload.get("reports", []) if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _iter_full_zip_rows(path: Path, kind: str) -> Any:
    spec = _FULL_ROW_SOURCES.get(kind, {})
    with zipfile.ZipFile(path, "r") as archive:
        normalized_names = _normalized_zip_entries_for_full_read(archive)
        for name in tuple(spec.get("csv", ())):
            if name in normalized_names:
                yield from _iter_csv_zip_entry(archive, normalized_names[name])
                return
        for name in tuple(spec.get("jsonl", ())):
            if name in normalized_names:
                yield from _iter_jsonl_zip_entry(archive, normalized_names[name])
                return
        payload = _read_report_payload_from_zip_for_full_read(archive, normalized_names)
        if payload:
            yield from _iter_full_json_rows(payload, kind)


def _normalized_zip_entries_for_full_read(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > MOBILE_REPORT_MAX_ENTRIES:
        raise ValueError(f"Archiwum raportu ma zbyt dużo wpisów: {len(infos)}.")
    total_uncompressed = sum(int(info.file_size or 0) for info in infos)
    if total_uncompressed > MOBILE_REPORT_MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ValueError(f"Archiwum raportu jest zbyt duże: {total_uncompressed} B.")
    normalized_names: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        ok, normalized = _archive_name_safe(info.filename)
        if not ok:
            raise ValueError(f"Niebezpieczna ścieżka w archiwum: {info.filename}")
        if normalized in normalized_names:
            raise ValueError(f"Zduplikowany wpis w archiwum: {normalized}")
        normalized_names[normalized] = info
    return normalized_names


def _iter_csv_zip_entry(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> Any:
    with archive.open(info, "r") as binary:
        wrapper = io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="")
        reader = csv.DictReader(wrapper)
        for row in reader:
            yield {str(key or ""): str(value or "") for key, value in dict(row or {}).items()}


def _iter_jsonl_zip_entry(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> Any:
    parser = ReportBundleReader(max_trace_rows=0)
    with archive.open(info, "r") as binary:
        wrapper = io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="")
        for line in wrapper:
            text = str(line or "").strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except Exception:
                continue
            if isinstance(item, dict):
                yield parser._json_record_to_row(item)


def _read_report_payload_from_zip_for_full_read(
    archive: zipfile.ZipFile,
    normalized_names: dict[str, zipfile.ZipInfo],
) -> dict[str, Any]:
    for name in ("report.json", "metadata.json"):
        info = normalized_names.get(name)
        if info is None:
            continue
        if int(info.file_size or 0) > MOBILE_REPORT_MAX_TEXT_BYTES:
            continue
        try:
            with archive.open(info, "r") as handle:
                payload = json.loads(handle.read().decode("utf-8-sig", errors="replace"))
        except Exception:
            continue
        if isinstance(payload, dict):
            if name == "metadata.json" and str(payload.get("schema") or "") != MOBILE_BENCHMARK_REPORT_SCHEMA:
                return _report_payload_from_thesis_metadata(payload)
            return payload
    return {}


def _iter_full_json_rows(payload: dict[str, Any], kind: str) -> Any:
    spec = _FULL_ROW_SOURCES.get(kind, {})
    parser = ReportBundleReader(max_trace_rows=0)
    value = None
    for path in tuple(spec.get("json", ())):
        value = _nested_value(payload, path)
        if value not in (None, ""):
            break
    if isinstance(value, dict):
        for key in ("records", "rows", "samples", "events", "traces", "data"):
            nested = value.get(key)
            if isinstance(nested, list):
                value = nested
                break
    if not isinstance(value, list):
        return
    for item in value:
        if isinstance(item, dict):
            if kind == "trace":
                row: dict[str, str] = {}
                for key in ("frame_id", "timestamp_ms", "status", "text"):
                    row[key] = str(item.get(key, ""))
                for nested_name, suffix in (("stage_ms", "_ms"), ("confidence", ""), ("counters", ""), ("memory", "")):
                    nested = item.get(nested_name)
                    if not isinstance(nested, dict):
                        continue
                    for key, child_value in nested.items():
                        column = str(key)
                        if suffix and not column.endswith(suffix):
                            column = f"{column}{suffix}"
                        row[column] = str(child_value)
                yield row
            else:
                yield parser._json_record_to_row(item)


@dataclass(frozen=True)
class ExperimentModelRef:
    """A stable model reference used by package-level experiments."""

    model_id: str
    role: str
    checkpoint: str = ""
    package_path: str = ""
    run_id: str = ""
    run_label: str = ""
    dataset_id: str = ""
    dataset_path: str = ""
    yolo_family: str = ""
    yolo_version: str = ""
    parameter_count: int = 0
    file_size_mb: float = 0.0
    best_epoch: int = 0
    total_epochs: int = 0
    total_epochs_known: bool = False
    known_epochs_minimum: int = 0
    provenance_status: str = ""
    dataset_manifest_sha256: str = ""
    dataset_split_sha256: str = ""
    checkpoint_sha256: str = ""
    package_sha256: str = ""
    installed_model_fingerprint: str = ""
    variant_id: str = ""
    variant_artifact_sha256: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def marker(self) -> str:
        return _role_marker(self.role)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentModelRef":
        fields = set(cls.__dataclass_fields__)
        payload = {key: value for key, value in dict(data or {}).items() if key in fields}
        return cls(
            model_id=_safe_id(str(payload.get("model_id") or payload.get("checkpoint") or "model")),
            role=str(payload.get("role") or ""),
            checkpoint=str(payload.get("checkpoint") or ""),
            package_path=str(payload.get("package_path") or ""),
            run_id=str(payload.get("run_id") or ""),
            run_label=str(payload.get("run_label") or ""),
            dataset_id=str(payload.get("dataset_id") or ""),
            dataset_path=str(payload.get("dataset_path") or ""),
            yolo_family=str(payload.get("yolo_family") or ""),
            yolo_version=str(payload.get("yolo_version") or ""),
            parameter_count=_safe_int(payload.get("parameter_count")),
            file_size_mb=float(_safe_float(payload.get("file_size_mb"), 0.0) or 0.0),
            best_epoch=_safe_int(payload.get("best_epoch")),
            total_epochs=_safe_int(payload.get("total_epochs")),
            total_epochs_known=_safe_bool(payload.get("total_epochs_known")),
            known_epochs_minimum=_safe_int(payload.get("known_epochs_minimum")),
            provenance_status=str(payload.get("provenance_status") or ""),
            dataset_manifest_sha256=str(payload.get("dataset_manifest_sha256") or ""),
            dataset_split_sha256=str(payload.get("dataset_split_sha256") or ""),
            checkpoint_sha256=str(payload.get("checkpoint_sha256") or ""),
            package_sha256=str(payload.get("package_sha256") or ""),
            installed_model_fingerprint=str(payload.get("installed_model_fingerprint") or ""),
            variant_id=str(payload.get("variant_id") or ""),
            variant_artifact_sha256=tuple(str(item) for item in payload.get("variant_artifact_sha256") or ()),
            created_at=str(payload.get("created_at") or ""),
            metrics=dict(payload.get("metrics") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_mobile_export_candidate(cls, candidate: dict[str, Any]) -> "ExperimentModelRef":
        """Build a model reference from the candidate dict used by the export UI."""
        run = candidate.get("run")
        metadata = candidate.get("model_metadata") if isinstance(candidate.get("model_metadata"), dict) else {}
        info = candidate.get("model_info") if isinstance(candidate.get("model_info"), dict) else {}
        provenance = candidate.get("training_provenance") if isinstance(candidate.get("training_provenance"), dict) else {}
        dataset = provenance.get("dataset") if isinstance(provenance.get("dataset"), dict) else {}
        best_weights = candidate.get("best_weights")
        checkpoint = str(best_weights or candidate.get("checkpoint") or "").strip()
        model_id = str(candidate.get("model_label") or Path(checkpoint).stem or "model")
        metrics = {
            "map50": candidate.get("best_map50"),
            "map50_95": candidate.get("best_map50_95"),
        }
        return cls(
            model_id=_safe_id(model_id, fallback="model"),
            role=str(candidate.get("role") or candidate.get("target") or ""),
            checkpoint=checkpoint,
            run_id=str(getattr(run, "id", "") or candidate.get("run_id") or ""),
            run_label=str(candidate.get("run_label") or getattr(run, "name", "") or ""),
            dataset_id=str(dataset.get("dataset_id") or candidate.get("dataset_label") or ""),
            dataset_path=str(candidate.get("dataset_path") or getattr(run, "dataset_path", "") or ""),
            yolo_family=str(info.get("architecture_label") or info.get("yolo_variant") or ""),
            yolo_version=str(candidate.get("model_version") or info.get("version") or ""),
            parameter_count=_safe_int(info.get("parameter_count") or candidate.get("parameter_count")),
            file_size_mb=float(_safe_float(candidate.get("file_size_mb"), 0.0) or 0.0),
            best_epoch=_safe_int(candidate.get("best_epoch")),
            total_epochs=_safe_int(candidate.get("total_epochs") or getattr(run, "current_epoch", 0)),
            total_epochs_known=_safe_bool(candidate.get("total_epochs_known"), _safe_bool(provenance.get("total_epochs_known"))),
            known_epochs_minimum=_safe_int(candidate.get("known_epochs_minimum") or provenance.get("known_epochs_minimum")),
            provenance_status=str(candidate.get("provenance_status") or provenance.get("provenance_status") or ""),
            dataset_manifest_sha256=str(dataset.get("manifest_sha256") or ""),
            dataset_split_sha256=str(dataset.get("split_sha256") or ""),
            created_at=str(candidate.get("created_at") or getattr(run, "created_at", "") or ""),
            metrics=metrics,
            metadata={"source": "mobile_export_candidate", "training_provenance": provenance, **metadata},
        )

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any], *, package_path: str = "") -> "ExperimentModelRef":
        training = dict(manifest.get("training") or {})
        source = dict(manifest.get("source") or {})
        metrics = dict(manifest.get("metrics") or {})
        model = dict(manifest.get("model") or {})
        dataset = training.get("dataset") if isinstance(training.get("dataset"), dict) else {}
        variants = [dict(variant) for variant in list(manifest.get("variants") or []) if isinstance(variant, dict)]
        primary_variant = variants[0] if variants else {}
        variant_hashes = primary_variant.get("sha256") if isinstance(primary_variant.get("sha256"), dict) else {}
        checkpoint = str(source.get("checkpoint") or "").strip()
        return cls(
            model_id=_safe_id(str(manifest.get("model_id") or Path(checkpoint).stem or "model")),
            role=str(manifest.get("role") or ""),
            checkpoint=checkpoint,
            package_path=str(package_path or ""),
            run_id=str(training.get("run_id") or training.get("id") or ""),
            run_label=str(training.get("run_label") or training.get("name") or ""),
            dataset_id=str(dataset.get("dataset_id") or training.get("dataset_id") or training.get("dataset_label") or ""),
            dataset_path=str(training.get("dataset_path") or ""),
            yolo_family=str(model.get("family") or model.get("architecture_label") or source.get("architecture_label") or ""),
            yolo_version=str(model.get("version") or source.get("model_version") or ""),
            parameter_count=_safe_int(source.get("parameter_count") or model.get("parameter_count")),
            file_size_mb=float(_safe_float(source.get("file_size_mb"), 0.0) or 0.0),
            best_epoch=_safe_int(metrics.get("best_epoch")),
            total_epochs=_safe_int(training.get("total_epochs") or training.get("known_epochs_minimum") or training.get("current_epoch") or training.get("epochs")),
            total_epochs_known=_safe_bool(training.get("total_epochs_known")),
            known_epochs_minimum=_safe_int(training.get("known_epochs_minimum")),
            provenance_status=str(training.get("provenance_status") or ""),
            dataset_manifest_sha256=str(dataset.get("manifest_sha256") or ""),
            dataset_split_sha256=str(dataset.get("split_sha256") or ""),
            checkpoint_sha256=str(source.get("checkpoint_sha256") or ""),
            variant_id=str(primary_variant.get("id") or ""),
            variant_artifact_sha256=tuple(str(item) for item in dict(variant_hashes or {}).values() if str(item or "").strip()),
            created_at=str(source.get("exported_at") or training.get("finished_at") or training.get("created_at") or ""),
            metrics=metrics,
            metadata={"source": "manifest", "manifest_schema": manifest.get("schema", ""), "training_provenance": training},
        )


@dataclass(frozen=True)
class RuntimeVariantSpec:
    id: str
    runtime: str
    precision: str = "fp32"
    image_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    calibration_dataset_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeVariantSpec":
        return cls(
            id=_safe_id(str(data.get("id") or f"{data.get('runtime', 'runtime')}-{data.get('precision', 'fp32')}")),
            runtime=str(data.get("runtime") or ""),
            precision=str(data.get("precision") or "fp32").lower(),
            image_size=_safe_int(data.get("image_size") or data.get("imgsz"), 640),
            confidence_threshold=float(_safe_float(data.get("confidence_threshold"), 0.25) or 0.25),
            iou_threshold=float(_safe_float(data.get("iou_threshold"), 0.45) or 0.45),
            calibration_dataset_id=str(data.get("calibration_dataset_id") or ""),
        )


@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str = ""
    path: str = ""
    split_name: str = ""
    image_count: int = 0
    plate_count: int = 0
    char_count: int = 0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DatasetRef":
        payload = dict(data or {})
        return cls(
            dataset_id=str(payload.get("dataset_id") or payload.get("id") or ""),
            path=str(payload.get("path") or ""),
            split_name=str(payload.get("split_name") or payload.get("split") or ""),
            image_count=_safe_int(payload.get("image_count") or payload.get("images")),
            plate_count=_safe_int(payload.get("plate_count") or payload.get("plates")),
            char_count=_safe_int(payload.get("char_count") or payload.get("chars")),
            created_at=str(payload.get("created_at") or ""),
        )


@dataclass(frozen=True)
class MobilePackageCandidate:
    """A complete ALPR candidate composed of MT+MZ and optional MP."""

    package_id: str
    plate_model: ExperimentModelRef
    character_model: ExperimentModelRef
    vehicle_model: ExperimentModelRef | None = None
    variants: tuple[RuntimeVariantSpec, ...] = field(default_factory=tuple)
    ranking_dataset: DatasetRef = field(default_factory=DatasetRef)
    calibration_dataset: DatasetRef = field(default_factory=DatasetRef)
    created_at: str = field(default_factory=_utc_now_iso)
    status: str = "candidate"
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def model_ids(self) -> tuple[str, ...]:
        if self.vehicle_model is not None:
            return (self.vehicle_model.model_id, self.plate_model.model_id, self.character_model.model_id)
        return (self.plate_model.model_id, self.character_model.model_id)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.plate_model.marker != "MT":
            problems.append("plate_model must have MT/plate role")
        if self.character_model.marker != "MZ":
            problems.append("character_model must have MZ/character role")
        if self.vehicle_model is not None and self.vehicle_model.marker != "MP":
            problems.append("vehicle_model must have MP/vehicle role")
        if not self.variants:
            problems.append("at least one runtime variant is required")
        if not str(self.package_id or "").strip():
            problems.append("package_id is required")
        return problems

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "package_id": self.package_id,
            "plate_model": self.plate_model.to_dict(),
            "character_model": self.character_model.to_dict(),
            "variants": [variant.to_dict() for variant in self.variants],
            "ranking_dataset": self.ranking_dataset.to_dict(),
            "calibration_dataset": self.calibration_dataset.to_dict(),
            "created_at": self.created_at,
            "status": self.status,
            "notes": self.notes,
            "metadata": dict(self.metadata or {}),
        }
        if self.vehicle_model is not None:
            payload["vehicle_model"] = self.vehicle_model.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MobilePackageCandidate":
        vehicle_payload = data.get("vehicle_model")
        return cls(
            package_id=_safe_id(str(data.get("package_id") or "package")),
            plate_model=ExperimentModelRef.from_dict(dict(data.get("plate_model") or {})),
            character_model=ExperimentModelRef.from_dict(dict(data.get("character_model") or {})),
            vehicle_model=ExperimentModelRef.from_dict(dict(vehicle_payload or {})) if vehicle_payload else None,
            variants=tuple(RuntimeVariantSpec.from_dict(item) for item in list(data.get("variants") or [])),
            ranking_dataset=DatasetRef.from_dict(data.get("ranking_dataset")),
            calibration_dataset=DatasetRef.from_dict(data.get("calibration_dataset")),
            created_at=str(data.get("created_at") or _utc_now_iso()),
            status=str(data.get("status") or "candidate"),
            notes=str(data.get("notes") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class MobileBenchmarkReport:
    """A report produced by the Android application for one package variant."""

    report_id: str
    package_id: str
    variant_id: str
    measured_at: str = field(default_factory=_utc_now_iso)
    source_archive_sha256: str = ""
    source_path: str = ""
    experiment_index: dict[str, Any] = field(default_factory=dict)
    device: dict[str, Any] = field(default_factory=dict)
    runtime: str = ""
    delegate: str = ""
    latency: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> tuple[str, ...]:
        return _mobile_report_dedupe_key(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MOBILE_BENCHMARK_REPORT_SCHEMA,
            "report_id": self.report_id,
            "package_id": self.package_id,
            "variant_id": self.variant_id,
            "measured_at": self.measured_at,
            "source_archive_sha256": self.source_archive_sha256,
            "source_path": self.source_path,
            "experiment_index": dict(self.experiment_index or {}),
            "device": dict(self.device or {}),
            "runtime": self.runtime,
            "delegate": self.delegate,
            "latency": dict(self.latency or {}),
            "memory": dict(self.memory or {}),
            "quality": dict(self.quality or {}),
            "errors": dict(self.errors or {}),
            "raw": dict(self.raw or {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MobileBenchmarkReport":
        payload = dict(data or {})
        stored_raw = payload.get("raw")
        raw_payload = dict(stored_raw) if isinstance(stored_raw, dict) else payload
        device = dict(payload.get("device") or {})
        if not device:
            device = {
                "name": payload.get("device_name", ""),
                "android_version": payload.get("android_version", ""),
            }
        latency = dict(payload.get("latency") or payload.get("latencies") or {})
        memory = dict(payload.get("memory") or {})
        quality = dict(payload.get("quality") or payload.get("metrics") or {})
        errors = dict(payload.get("errors") or payload.get("error_counts") or {})
        package_id = _safe_id(str(payload.get("package_id") or payload.get("model_package_id") or "package"))
        variant_id = _safe_id(str(payload.get("variant_id") or payload.get("variant") or payload.get("runtime") or "variant"))
        report_seed = "|".join(
            [
                package_id,
                variant_id,
                str(device.get("name") or ""),
                str(payload.get("runtime") or ""),
                str(payload.get("delegate") or ""),
                str(payload.get("measured_at") or ""),
            ]
        )
        return cls(
            report_id=_safe_id(str(payload.get("report_id") or report_seed), fallback="report"),
            package_id=package_id,
            variant_id=variant_id,
            measured_at=str(payload.get("measured_at") or "") if "measured_at" in payload else _utc_now_iso(),
            source_archive_sha256=str(
                payload.get("source_archive_sha256")
                or _nested_value(raw_payload, "desktop_ingest.source_archive_sha256")
                or ""
            ),
            source_path=str(payload.get("source_path") or _nested_value(raw_payload, "desktop_ingest.source_path") or ""),
            experiment_index=dict(
                payload.get("experiment_index")
                or payload.get("desktop_experiment_index")
                or _nested_value(raw_payload, "desktop_experiment_index")
                or {}
            ),
            device=device,
            runtime=str(payload.get("runtime") or ""),
            delegate=str(payload.get("delegate") or ""),
            latency=latency,
            memory=memory,
            quality=quality,
            errors=errors,
            raw=raw_payload,
        )


def _mobile_report_device_name(report: MobileBenchmarkReport) -> str:
    try:
        device = dict(report.device or {})
    except Exception:
        device = {}
    return str(device.get("name") or device.get("device_name") or "").strip()


def _mobile_report_dedupe_key(report: MobileBenchmarkReport) -> tuple[str, ...]:
    # A session archive contains one report. Its bytes remain the same even
    # when a legacy report omits measured_at (which otherwise defaults to now).
    kind = _nested_value(report.raw, "desktop_ingest.bundle_kind")
    if (report.source_archive_sha256 and (kind == "alprsession" or
            (Path(report.source_path).suffix.lower() == ".alprsession" and kind != "json"))):
        return ("archive", report.source_archive_sha256)
    return (
        "report",
        str(report.report_id or "").strip(),
        str(report.package_id or "").strip(),
        str(report.variant_id or "").strip(),
        _mobile_report_device_name(report),
        str(report.runtime or "").strip(),
        str(report.delegate or "").strip(),
        str(report.measured_at or "").strip(),
    )


def _mobile_report_source_path(report: MobileBenchmarkReport) -> str:
    raw = dict(report.raw or {}) if isinstance(report.raw, dict) else {}
    ingest = dict(raw.get("desktop_ingest") or {}) if isinstance(raw.get("desktop_ingest"), dict) else {}
    return str(report.source_path or ingest.get("source_path") or "").strip()


def _mobile_report_source_is_internal_store(report: MobileBenchmarkReport) -> bool:
    source_path = _mobile_report_source_path(report)
    if not source_path:
        return False
    try:
        return Path(source_path).name.lower() == MOBILE_PACKAGE_EXPERIMENT_STORE_FILE_NAME.lower()
    except Exception:
        return source_path.lower().endswith(MOBILE_PACKAGE_EXPERIMENT_STORE_FILE_NAME.lower())


def _mobile_report_keep_rank(report: MobileBenchmarkReport) -> tuple[int, int, int, int]:
    source_path = _mobile_report_source_path(report)
    source_hash = str(report.source_archive_sha256 or "").strip()
    raw = dict(report.raw or {}) if isinstance(report.raw, dict) else {}
    try:
        raw_size = len(json.dumps(raw, ensure_ascii=False, sort_keys=True))
    except Exception:
        raw_size = len(str(raw))
    return (
        0 if _mobile_report_source_is_internal_store(report) else 1,
        1 if source_path else 0,
        1 if source_hash else 0,
        raw_size,
    )


def deduplicate_mobile_reports(
    reports: list[MobileBenchmarkReport] | tuple[MobileBenchmarkReport, ...],
) -> list[MobileBenchmarkReport]:
    selected: dict[tuple[str, ...], MobileBenchmarkReport] = {}
    order: list[tuple[str, ...]] = []
    for report in list(reports or []):
        key = _mobile_report_dedupe_key(report)
        if key not in selected:
            selected[key] = report
            order.append(key)
            continue
        if _mobile_report_keep_rank(report) > _mobile_report_keep_rank(selected[key]):
            selected[key] = report
    return [selected[key] for key in order if key in selected]


@dataclass(frozen=True)
class ExperimentSessionRecord:
    """Normalized session-level index used by desktop-side experiment analysis."""

    report_id: str
    experiment_session_id: str = ""
    series_id: str = ""
    scenario_id: str = ""
    experiment_variant: str = ""
    replicate_index: int = 0
    package_id: str = ""
    variant_id: str = ""
    measured_at: str = ""
    source_archive_sha256: str = ""
    source_path: str = ""
    bundle_kind: str = ""
    bundle_schema: str = ""
    app_git_sha: str = ""
    app_version: str = ""
    build_type: str = ""
    device: dict[str, Any] = field(default_factory=dict)
    runtime: str = ""
    delegate: str = ""
    model_fingerprints: dict[str, Any] = field(default_factory=dict)
    resolution: dict[str, Any] = field(default_factory=dict)
    recognition_profile: dict[str, Any] = field(default_factory=dict)
    autozoom_config: dict[str, Any] = field(default_factory=dict)
    trace_total_source: int = 0
    trace_rows_preview: int = 0
    trace_preview_truncated: bool = False
    artifact_flags: dict[str, bool] = field(default_factory=dict)
    artifact_counts: dict[str, int] = field(default_factory=dict)
    validation_ok: bool = True
    validation_errors: tuple[str, ...] = field(default_factory=tuple)
    validation_warnings: tuple[str, ...] = field(default_factory=tuple)
    raw_extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_report(
        cls,
        report: MobileBenchmarkReport,
        *,
        source_archive_sha256: str = "",
        source_path: str = "",
        bundle_kind: str = "",
        bundle_schema: str = "",
        trace_total_source: int = 0,
        trace_rows_preview: int = 0,
        validation: ReportBundleValidation | None = None,
        artifact_flags: dict[str, bool] | None = None,
        artifact_counts: dict[str, int] | None = None,
    ) -> "ExperimentSessionRecord":
        raw = dict(report.raw or {})
        experiment = _optional_dict(raw.get("experiment"))
        app_build = _optional_dict(raw.get("app_build") or raw.get("build") or raw.get("application"))
        capture = _optional_dict(raw.get("capture") or raw.get("camera") or raw.get("input"))
        recognition_profile = _optional_dict(raw.get("recognition_profile") or raw.get("profile"))
        autozoom_config = _optional_dict(
            raw.get("autozoom")
            or raw.get("autozoom_config")
            or recognition_profile.get("autozoom")
            or recognition_profile.get("autozoom_config")
        )
        resolution: dict[str, Any] = {}
        for key, paths in {
            "width_px": ("capture.width_px", "capture.width", "camera.width_px", "input.width_px", "resolution.width_px"),
            "height_px": ("capture.height_px", "capture.height", "camera.height_px", "input.height_px", "resolution.height_px"),
            "fps": ("capture.fps", "camera.fps", "input.fps"),
        }.items():
            value = _nested_value(raw, *paths)
            if value not in (None, ""):
                resolution[key] = value

        model_fingerprints = _normalize_mobile_model_fingerprints(
            _nested_value(raw, "model_fingerprints"),
            _nested_value(raw, "models"),
            _nested_value(raw, "execution"),
            _nested_value(raw, "execution.models"),
            _nested_value(raw, "execution.model_fingerprints"),
            _nested_value(raw, "runtime_composition"),
            _nested_value(raw, "runtime_composition.models"),
        )

        validation_ok = True if validation is None else bool(validation.ok)
        validation_errors = tuple() if validation is None else tuple(validation.errors)
        validation_warnings = tuple() if validation is None else tuple(validation.warnings)
        preview_count = max(0, int(trace_rows_preview or 0))
        trace_total = max(0, int(trace_total_source or 0))
        return cls(
            report_id=report.report_id,
            experiment_session_id=str(
                _nested_value(
                    raw,
                    "experiment.session.id",
                    "experiment.session_id",
                    "experiment_session_id",
                    "session_id",
                )
                or experiment.get("session_id")
                or experiment.get("id")
                or report.report_id
            ),
            series_id=str(_nested_value(raw, "experiment.series_id", "series_id") or experiment.get("series_id") or ""),
            scenario_id=str(_nested_value(raw, "experiment.scenario_id", "scenario_id") or experiment.get("scenario_id") or ""),
            experiment_variant=str(
                _nested_value(raw, "experiment.variant", "experiment.variant_id", "variant")
                or experiment.get("variant")
                or report.variant_id
            ),
            replicate_index=_safe_int(
                _nested_value(raw, "experiment.replicate_index", "replicate_index")
                or experiment.get("replicate_index")
            ),
            package_id=report.package_id,
            variant_id=report.variant_id,
            measured_at=report.measured_at,
            source_archive_sha256=source_archive_sha256,
            source_path=source_path,
            bundle_kind=bundle_kind,
            bundle_schema=bundle_schema,
            app_git_sha=str(
                _nested_value(raw, "app_build.git_commit", "app_build.git_sha", "build.git_commit", "app_git_sha")
                or app_build.get("git_commit")
                or app_build.get("git_sha")
                or ""
            ),
            app_version=str(
                _nested_value(raw, "app_build.version", "app_version", "application.version")
                or app_build.get("version")
                or ""
            ),
            build_type=str(
                _nested_value(raw, "app_build.build_type", "build.type", "build_type")
                or app_build.get("build_type")
                or app_build.get("type")
                or ""
            ),
            device=dict(report.device or {}),
            runtime=str(report.runtime or _nested_value(raw, "runtime", "execution.runtime") or ""),
            delegate=str(report.delegate or _nested_value(raw, "delegate", "execution.delegate") or ""),
            model_fingerprints=model_fingerprints,
            resolution=resolution or capture,
            recognition_profile=recognition_profile,
            autozoom_config=autozoom_config,
            trace_total_source=trace_total,
            trace_rows_preview=preview_count,
            trace_preview_truncated=bool(trace_total > preview_count > 0),
            artifact_flags={str(k): bool(v) for k, v in dict(artifact_flags or {}).items()},
            artifact_counts={str(k): int(v or 0) for k, v in dict(artifact_counts or {}).items()},
            validation_ok=validation_ok,
            validation_errors=validation_errors,
            validation_warnings=validation_warnings,
            raw_extra={
                "schema": raw.get("schema", ""),
                "quality_available": bool(_nested_value(raw, "quality.available")),
                "measured_runs": _nested_value(raw, "measured_runs", "runs", "samples"),
            },
        )


def _attach_report_ingest_metadata(
    report: MobileBenchmarkReport,
    *,
    path: Path,
    source_archive_sha256: str,
    bundle_kind: str,
    bundle_schema: str,
    trace_total: int,
    trace_rows_preview: int,
    validation: ReportBundleValidation,
    artifact_flags: dict[str, bool] | None = None,
    artifact_counts: dict[str, int] | None = None,
) -> tuple[MobileBenchmarkReport, dict[str, Any]]:
    session = ExperimentSessionRecord.from_report(
        report,
        source_archive_sha256=source_archive_sha256,
        source_path=str(path),
        bundle_kind=bundle_kind,
        bundle_schema=bundle_schema,
        trace_total_source=trace_total,
        trace_rows_preview=trace_rows_preview,
        validation=validation,
        artifact_flags=artifact_flags or {},
        artifact_counts=artifact_counts or {},
    )
    ingest = {
        "source_path": str(path),
        "source_archive_sha256": source_archive_sha256,
        "bundle_kind": bundle_kind,
        "bundle_schema": bundle_schema,
        "imported_at": _utc_now_iso(),
    }
    raw = dict(report.raw or {})
    raw["desktop_ingest"] = ingest
    raw["desktop_experiment_index"] = session.to_dict()
    updated = replace(
        report,
        source_archive_sha256=source_archive_sha256,
        source_path=str(path),
        experiment_index=session.to_dict(),
        raw=raw,
    )
    return updated, session.to_dict()


@dataclass(frozen=True)
class MobilePackageScore:
    package_id: str
    variant_id: str
    total: float
    quality: float
    latency: float
    memory: float
    reliability: float
    rejected: bool = False
    reasons: tuple[str, ...] = field(default_factory=tuple)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_package_id(
    plate_model: ExperimentModelRef,
    character_model: ExperimentModelRef,
    *,
    vehicle_model: ExperimentModelRef | None = None,
    variant_suffix: str = "",
    prefix: str = "PKG",
) -> str:
    parts = [prefix]
    if vehicle_model is not None:
        parts.append(vehicle_model.model_id)
    parts.extend([plate_model.model_id, character_model.model_id, variant_suffix])
    return _safe_id("-".join(part for part in parts if str(part or "").strip()), fallback="PKG")


def default_runtime_variants(image_size: int = 640) -> tuple[RuntimeVariantSpec, ...]:
    return (
        RuntimeVariantSpec(id="tflite-fp32", runtime="tflite", precision="fp32", image_size=image_size),
        RuntimeVariantSpec(id="onnx-fp32", runtime="onnx", precision="fp32", image_size=image_size),
    )


def build_package_candidate(
    plate_model: ExperimentModelRef,
    character_model: ExperimentModelRef,
    *,
    vehicle_model: ExperimentModelRef | None = None,
    variants: tuple[RuntimeVariantSpec, ...] | None = None,
    ranking_dataset: DatasetRef | None = None,
    calibration_dataset: DatasetRef | None = None,
    package_id: str = "",
    notes: str = "",
    metadata: dict[str, Any] | None = None,
) -> MobilePackageCandidate:
    safe_variants = variants or default_runtime_variants()
    candidate = MobilePackageCandidate(
        package_id=_safe_id(package_id or build_package_id(plate_model, character_model, vehicle_model=vehicle_model)),
        plate_model=plate_model,
        character_model=character_model,
        vehicle_model=vehicle_model,
        variants=tuple(safe_variants),
        ranking_dataset=ranking_dataset or DatasetRef(),
        calibration_dataset=calibration_dataset or DatasetRef(),
        notes=notes,
        metadata=dict(metadata or {}),
    )
    problems = candidate.validate()
    if problems:
        raise ValueError("; ".join(problems))
    return candidate


def score_mobile_report(
    report: MobileBenchmarkReport,
    *,
    weights: dict[str, float] | None = None,
    targets: dict[str, float] | None = None,
) -> MobilePackageScore:
    safe_weights = dict(DEFAULT_SCORE_WEIGHTS)
    safe_weights.update(dict(weights or {}))
    safe_targets = dict(DEFAULT_SCORE_TARGETS)
    safe_targets.update(dict(targets or {}))

    quality_data = dict(report.quality or {})
    latency_data = dict(report.latency or {})
    memory_data = dict(report.memory or {})
    errors_data = dict(report.errors or {})

    exact_match = _metric01(
        _nested_value(
            quality_data,
            "plate_exact_match",
            "exact_match_rate",
            "plate_accuracy",
            "success_rate",
            "accuracy_plate",
            "end_to_end_accuracy",
        ),
        default=0.0,
    )
    cer = max(0.0, _safe_float(_nested_value(quality_data, "cer", "character_error_rate"), 0.0) or 0.0)
    char_f1 = _metric01(_nested_value(quality_data, "char_f1", "f1", "f1_score"), default=0.0)
    quality = max(exact_match, max(0.0, 1.0 - cer) * 0.65 + char_f1 * 0.35 if cer > 0 or char_f1 > 0 else 0.0)
    if quality_data.get("quality_source") == "human_review":
        # Verified CER is an unbounded ratio, never a legacy percentage.
        cer = max(0.0, _safe_float(quality_data.get("cer"), 0.0) or 0.0)
        exact_match = _clamp01(quality_data.get("exact_read_rate"), 0.0)
        quality = max(exact_match, max(0.0, 1.0 - cer) * 0.65)
        if quality_data.get("review_status") != "COMPLETED" or not quality_data.get("available"):
            quality = 0.0

    p95 = _safe_float(
        _nested_value(
            latency_data,
            "pipeline_ms_p95",
            "latency_pipeline_ms_p95",
            "pipeline.p95_ms",
            "p95_ms",
        ),
        None,
    )
    latency = 0.0
    if p95 and p95 > 0:
        latency = _clamp01(float(safe_targets["pipeline_p95_ms"]) / float(p95))

    ram_peak = _safe_float(
        _nested_value(memory_data, "ram_peak_mb", "peak_ram_mb", "process_peak_mb"),
        None,
    )
    package_size = _safe_float(
        _nested_value(memory_data, "package_size_mb", "model_size_mb", "variant_size_mb"),
        None,
    )
    memory_parts: list[float] = []
    if ram_peak and ram_peak > 0:
        memory_parts.append(_clamp01(float(safe_targets["ram_peak_mb"]) / float(ram_peak)))
    if package_size and package_size > 0:
        memory_parts.append(_clamp01(float(safe_targets["package_size_mb"]) / float(package_size)))
    memory = sum(memory_parts) / len(memory_parts) if memory_parts else 0.0

    crash_count = _safe_int(_nested_value(errors_data, "crash_count", "crashes"))
    measured_runs = max(1, _safe_int(_nested_value(report.raw, "measured_runs", "runs", "samples"), 1))
    crash_rate = _clamp01(crash_count / measured_runs)
    reliability = 1.0 - crash_rate

    reasons: list[str] = []
    if not p95:
        reasons.append("missing pipeline p95 latency")
    if quality <= 0:
        reasons.append("missing end-to-end quality")
    if crash_count > 0:
        reasons.append(f"runtime crashes: {crash_count}")

    total_weight = sum(max(0.0, float(value)) for value in safe_weights.values()) or 1.0
    total = (
        quality * max(0.0, float(safe_weights.get("quality", 0.0)))
        + latency * max(0.0, float(safe_weights.get("latency", 0.0)))
        + memory * max(0.0, float(safe_weights.get("memory", 0.0)))
        + reliability * max(0.0, float(safe_weights.get("reliability", 0.0)))
    ) / total_weight

    rejected = bool(crash_count > 0 or quality <= 0 or not p95)
    return MobilePackageScore(
        package_id=report.package_id,
        variant_id=report.variant_id,
        total=round(total, 4),
        quality=round(quality, 4),
        latency=round(latency, 4),
        memory=round(memory, 4),
        reliability=round(reliability, 4),
        rejected=rejected,
        reasons=tuple(reasons),
        metrics={
            "plate_exact_match": exact_match,
            "cer": cer,
            "char_f1": char_f1,
            "pipeline_p95_ms": p95,
            "ram_peak_mb": ram_peak,
            "package_size_mb": package_size,
        },
    )


F5A_IMPORT_SERIES = "F5A-ROI-A12-01"
F5A_IMPORT_SCENARIO = "PAIRED_ROI_01"
F5A_IMPORT_GROUP = (F5A_IMPORT_SERIES, F5A_IMPORT_SCENARIO)
F5A_IMPORT_COMBINATIONS = tuple(
    (variant, f"r{repeat:02d}", block)
    for variant in ("R0", "R1", "R2")
    for repeat in (1, 2, 3)
    for block in ("A", "B", "C", "D")
)


@dataclass
class MobileSessionImportItem:
    path: Path
    status: str = "NEW"
    session_id: str = ""
    sha256: str = ""
    combination: tuple[str, str, str] | None = None
    report: MobileBenchmarkReport | None = None
    reason: str = ""
    roles_incomplete: bool = False
    archive_ok: bool | None = None
    profile_ok: bool | None = None
    observations: dict[str, tuple[str, ...]] = field(default_factory=dict)
    codes: tuple[str, ...] = ()
    details: tuple[str, ...] = ()
    mt_interruptions: dict[str, int] = field(default_factory=dict)
    group: tuple[str, str] | None = None

    @property
    def is_f5a(self) -> bool:
        return self.group == F5A_IMPORT_GROUP

    def observed(self, key: str) -> str:
        return " / ".join(self.observations.get(key, ())) or "brak w odczytanych metadanych"


@dataclass
class MobileSessionImportPlan:
    root: Path
    f5a: bool = False
    items: list[MobileSessionImportItem] = field(default_factory=list)
    scan_errors: list[str] = field(default_factory=list)
    cancelled: bool = False
    finished: bool = False
    selected_group: tuple[str, str] | None = None
    found_count: int = 0

    @property
    def groups(self) -> dict[tuple[str, str], list[MobileSessionImportItem]]:
        result = {}
        for item in self.items:
            if item.group is not None:
                result.setdefault(item.group, []).append(item)
        return result

    def select_group(self, group: tuple[str, str]) -> None:
        if group not in self.groups:
            raise ValueError("Nie znaleziono wybranej grupy w tym skanie.")
        self.selected_group = group

    @property
    def eligible_items(self) -> list[MobileSessionImportItem]:
        if self.cancelled or self.selected_group is None:
            return []
        return [i for i in self.items if i.group == self.selected_group and i.status == "NEW"
                and (not i.is_f5a or i.profile_ok is True)]

    @property
    def import_caption(self) -> str:
        n = len(self.eligible_items)
        ending = "zgodną sesję" if n == 1 else "zgodne sesje" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else "zgodnych sesji"
        return f"Importuj {n} {ending}"

    @staticmethod
    def group_caption(group) -> str:
        return " / ".join(v or "brak deklaracji" for v in group) if group is not None else "wybierz grupę w podsumowaniu"

    @staticmethod
    def mt_counts(items) -> tuple[int, dict[str, int]]:
        items = list(items)
        reasons, seen = {}, set()
        for item in items:
            if not item.archive_ok or item.sha256 in seen:
                continue
            seen.add(item.sha256)
            for reason, number in item.mt_interruptions.items():
                reasons[reason] = reasons.get(reason, 0) + number
        return sum(i.roles_incomplete for i in items), reasons

    def headline(self) -> list[str]:
        primary = [i for i in self.items if i.is_f5a]
        others = [i for i in self.items if not i.is_f5a and i.group is not None]
        return [f"Wszystkie znalezione archiwa: {self.found_count or len(self.items)}",
                f"Sprawdzone archiwa: {len(self.items)}",
                f"Integralne: {sum(i.archive_ok is True for i in self.items)}",
                f"Seria główna / wybrana: {self.group_caption(self.selected_group)}",
                f"Właściwa seria F5A: {len(primary)}",
                f"Zgodne z protokołem F5A: {sum(i.profile_ok is True for i in primary)}",
                f"Niezgodne z protokołem F5A: {sum(i.status == 'CONFLICT' for i in primary)}",
                f"Pilotaże / inne serie: {len(others)}" if primary else "Inne eksperymenty: kwalifikacja ogólna, bez przypisania do F5A",
                f"Już istniejące: {self.count('EXISTING')}",
                f"Uszkodzone lub nieczytelne: {self.count('INVALID')}",
                f"Konflikty tożsamości: {self.count('IDENTITY_CONFLICT')}",
                f"Duplikaty: {self.count('DUPLICATE')}",
                f"Gotowe do importu: {len(self.eligible_items)}"]

    def overview(self) -> list[str]:
        """The compact GUI and copied diagnostics share these exact counters."""
        selected = self.groups.get(self.selected_group, [])
        others = [i for i in self.items if i.group is not None and i.group != self.selected_group]
        return [f"Wszystkie archiwa: {self.found_count or len(self.items)}",
                f"Integralne: {sum(i.archive_ok is True for i in self.items)}",
                f"Poprawne w wybranej serii: {sum(i.status in {'NEW', 'EXISTING', 'IMPORTED'} for i in selected)}",
                f"Niezgodne: {sum(i.status == 'CONFLICT' for i in selected)}",
                f"Pilotaże / inne serie poza wyborem: {len(others)}",
                f"Już istniejące: {self.count('EXISTING')}",
                f"Uszkodzone: {self.count('INVALID')}",
                f"Gotowe do importu: {len(self.eligible_items)}"]

    def item_summary(self, item: MobileSessionImportItem) -> str:
        labels = {"NEW": "nowa", "EXISTING": "już zaimportowana", "IMPORTED": "zaimportowano",
                  "DUPLICATE": "duplikat", "INVALID": "uszkodzona lub nieczytelna",
                  "CONFLICT": "niezgodna z protokołem", "IDENTITY_CONFLICT": "konflikt tożsamości", "ERROR": "błąd zapisu lub zmiana źródła"}
        lines = [f"[{labels.get(item.status, item.status)}] {item.path}",
                 f"Seria / scenariusz: {self.group_caption(item.group)}",
                 f"session_id: {item.observed('session_id')}",
                 f"Wariant zapisany: {item.observed('variant')}",
                 f"Powtórzenie: {item.observed('replicate_index')} | Blok: {item.observed('block')}",
                 f"Integralność: {'potwierdzona' if item.archive_ok else 'niepotwierdzona'}",
                 f"Protokół: {'zgodny F5A' if item.profile_ok else 'niezgodny F5A' if item.profile_ok is False and item.is_f5a else 'brak znanego protokołu — ogólna kontrola archiwum'}",
                 f"SHA-256: {item.sha256 or 'nie udało się odczytać'}"]
        if item.reason:
            lines.append(item.reason)
        lines.extend(item.details)
        if item.codes:
            lines.append("Kod: " + ", ".join(item.codes))
        if item.roles_incomplete:
            lines.extend(["Nie wszystkie wywołania MT zakończyły postprocessing — szczegóły w raporcie.",
                          "mt_detection_roles_complete=False (ostrzeżenie, nie błąd ZIP/SHA)"])
            lines.extend(f"{reason}: {number}" for reason, number in sorted(item.mt_interruptions.items()))
        return "\n".join(lines)

    def count(self, status: str) -> int:
        return sum(item.status == status for item in self.items)

    @property
    def missing(self) -> tuple[tuple[str, str, str], ...]:
        accepted = {"EXISTING", "IMPORTED"} if self.finished else {"NEW", "EXISTING", "IMPORTED"}
        present = {item.combination for item in self.items if item.is_f5a and item.status in accepted}
        return tuple(c for c in F5A_IMPORT_COMBINATIONS if c not in present) if self.f5a else ()

    @property
    def missing_files(self) -> tuple[tuple[str, str, str], ...]:
        # Presence is independent of qualification: a policy conflict still
        # proves that this recorded session exists, without proving valid R1.
        present = {i.combination for i in self.items if i.archive_ok
                   and i.observations.get("series_id") == (F5A_IMPORT_SERIES,)
                   and i.observations.get("scenario_id") == (F5A_IMPORT_SCENARIO,)}
        return tuple(c for c in F5A_IMPORT_COMBINATIONS if c not in present) if self.f5a else ()

    def summary(self, *, completed: bool = False) -> str:
        lines = ["Podsumowanie kwalifikacji", f"Folder: {self.root}", *self.headline(), "", "Liczniki widoczne w oknie:", *self.overview()]
        if self.cancelled:
            lines.append("Operacja anulowana. Skan lub zapis mogą być częściowe; zapisane sesje pozostają w magazynie.")
        if not self.items and not self.scan_errors:
            lines.append("Nie znaleziono plików .alprsession w tym folderze ani jego podfolderach.")
        if self.selected_group is None and self.groups:
            lines.append("Wybierz świadomie jedną z grup. Żadnej nie wybrano arbitralnie.")
        if completed or self.finished:
            lines.extend(["", "Wynik zapisu dla całego skanu:",
                          f"Zaimportowano: {self.count('IMPORTED')}",
                          f"Już istniały: {self.count('EXISTING')}",
                          f"Nie zaimportowano: {len(self.items) - self.count('IMPORTED') - self.count('EXISTING')}",
                          f"Błędy: {self.count('ERROR') + self.count('INVALID') + self.count('IDENTITY_CONFLICT')}"])
        lines.extend(["", "Grupy rozpoznane z metadanych:"])
        for group, items in self.groups.items():
            lines.append(f"{self.group_caption(group)}: {len(items)} archiwów; "
                         f"nowe: {sum(i.status == 'NEW' for i in items)}; "
                         f"już zaimportowane: {sum(i.status == 'EXISTING' for i in items)}; "
                         f"niezgodne: {sum(i.status == 'CONFLICT' for i in items)}")
        excluded = [i for i in self.items if i.group is not None and i.group != self.selected_group]
        if excluded and self.selected_group is not None:
            pilots = all(re.search(r"(?:^|[-_])PILOT(?:$|[-_])", i.group[0], re.I) for i in excluded)
            lines.append(f"Pominięto {len(excluded)} sesje pilotażowe należące do innej serii." if pilots
                         else f"Pominięto {len(excluded)} archiwów innych serii. Można jawnie wybrać ich grupę.")
        if self.f5a:
            unknown = self.cancelled or self.scan_errors or any(not i.archive_ok or (i.is_f5a and not i.combination) for i in self.items)
            lines.append("Brakujących plików oryginalnych F5A: nie można ustalić — niepełny odczyt" if unknown
                         else f"Brakujących plików oryginalnych F5A: {len(self.missing_files)}")
            lines.append(f"Brakujących poprawnych pomiarów F5A: {len(self.missing)}")
            lines.append("Do uzupełnienia: " + (", ".join("/".join(c) for c in self.missing) or "brak"))
        for label, items in (("Cały skan", self.items),
                             ("Właściwa seria F5A", [i for i in self.items if i.is_f5a]),
                             ("Inne serie", [i for i in self.items if not i.is_f5a])):
            incomplete, reasons = self.mt_counts(items)
            lines.extend(["", f"Dowody MT — {label}:", f"Archiwa z niekompletnymi dowodami: {incomplete}",
                          f"Wywołania z niedokończonym postprocessingiem: {sum(reasons.values())}"])
            lines.extend(f"{reason}: {number}" for reason, number in sorted(reasons.items()))
            if incomplete:
                lines.append("Nie wszystkie wywołania MT zakończyły postprocessing — szczegóły w raporcie.")
        lines.extend(["", "Szczegóły archiwów:"])
        for item in sorted(self.items, key=lambda i: i.status in {"NEW", "EXISTING", "IMPORTED", "DUPLICATE"}):
            lines.extend(["", self.item_summary(item)])
        lines.extend(self.scan_errors)
        if not completed:
            lines.extend(["", "Samo sprawdzanie i podsumowanie nie zapisują raportów.", self.import_caption])
        return "\n".join(lines)


class _SessionImportIssue(ValueError):
    def __init__(self, reason, *, status="CONFLICT", codes=(), details=()):
        super().__init__(reason)
        self.status, self.codes, self.details = status, tuple(codes), tuple(details)


def mobile_import_path_error(error: Exception, path: Path | str) -> str:
    """Operator-facing Windows/filesystem errors, retaining raw details separately."""
    winerror = getattr(error, "winerror", None)
    if isinstance(error, PermissionError) or winerror == 5:
        reason = "Brak dostępu. Sprawdź uprawnienia do folderu lub pliku"
    elif isinstance(error, FileNotFoundError) or winerror in {3, 15, 21, 53}:
        reason = "Ścieżka nie istnieje lub dysk jest niedostępny"
    elif winerror == 206 or getattr(error, "errno", None) == 36:
        reason = "Ścieżka przekracza limit obsługiwany przez system Windows"
    elif isinstance(error, NotADirectoryError):
        reason = "Wskazana ścieżka nie jest folderem"
    else:
        reason = "Nie można odczytać wskazanej lokalizacji"
    return f"{reason}: {path}"


def _observe_session_import(bundle: MobileReportBundle, item: MobileSessionImportItem) -> None:
    """Retain declared facts even when subsequent qualification fails."""
    raw, session = bundle.report.raw, bundle.collection_session
    experiment = _optional_dict(raw.get("experiment"))
    recorded = _optional_dict(experiment.get("session"))
    collection = _optional_dict(raw.get("research_collection"))
    sources = [session, experiment, recorded, collection, bundle.manifest, raw, bundle.report_payload]
    def values(key):
        return tuple(dict.fromkeys((str(s[key]) if key in {"series_id", "scenario_id"} else str(s[key]).strip())
                                   for s in sources if s.get(key) not in (None, "")))
    item.observations = {k: values(k) for k in ("series_id", "scenario_id", "variant", "replicate_index", "completion_reason", "state")}
    series, scenario = item.observations["series_id"], item.observations["scenario_id"]
    if len(series) <= 1 and len(scenario) <= 1:
        item.group = (series[0] if series else "", scenario[0] if scenario else "")
    item.observations["session_id"] = tuple(dict.fromkeys(str(v).strip() for v in (
        session.get("session_id"), recorded.get("id"), recorded.get("session_id"),
        raw.get("session_id"), experiment.get("session_id"), collection.get("session_id"),
        bundle.manifest.get("session_id")) if v not in (None, "")))
    item.session_id = item.observations["session_id"][0] if item.observations["session_id"] else ""
    blocks = []
    for source in sources:
        blocks.extend(str(source[k]).removeprefix("BLOK_") for k in ("block", "block_id") if source.get(k))
        blocks.extend(re.findall(r"\bBLOK_([A-D])\b", str(source.get("notes") or "")))
    item.observations["block"] = tuple(dict.fromkeys(blocks))
    configs = [session.get("execution"), raw.get("research_execution_config"),
               experiment.get("effective_execution_config"), collection.get("execution")]
    item.observations["roi_budget_policy"] = tuple(dict.fromkeys(
        str(s[k]) for s in [*sources, *configs] if isinstance(s, dict)
        for k in ("roi_budget_policy", "effective_roi_budget_policy") if s.get(k)))
    v, r, b = (item.observations[k] for k in ("variant", "replicate_index", "block"))
    if len(v) == len(r) == len(b) == 1 and r[0] in {"1", "2", "3"}:
        candidate = (v[0], f"r{int(r[0]):02d}", b[0])
        if candidate in F5A_IMPORT_COMBINATIONS:
            item.combination = candidate
    item.roles_incomplete = any(s.get("mt_detection_roles_complete") is False for s in sources)
    for role in bundle.scene_evidence.get("roles", ()):
        if role.get("postprocess_status") == "NOT_REACHED":
            reason = role.get("cancel_reason") or role.get("processing_error") or role.get("execution_error") or "brak zapisanej przyczyny"
            item.mt_interruptions[reason] = item.mt_interruptions.get(reason, 0) + 1


def _session_import_metadata(bundle: MobileReportBundle, item: MobileSessionImportItem, *, f5a: bool) -> tuple[str, tuple | None, dict]:
    """Inspect the native reader's validated objects; never infer from a path."""
    session, manifest, raw = bundle.collection_session, bundle.manifest, bundle.report_payload
    if not bundle.validation.ok:
        raise ValueError("; ".join(bundle.validation.errors))
    if session.get("schema") != "alpr.mobile_research_session.v1" or not session.get("session_id"):
        raise ValueError("Brak poprawnego session.json lub session_id.")
    hashes = manifest.get("entry_sha256") or manifest.get("sha256") or {}
    if not isinstance(hashes, dict) or any(not re.fullmatch(r"[a-fA-F0-9]{64}", str(hashes.get(n, "")))
                                           for n in ("session.json", "report.json")):
        raise ValueError("Manifest musi zawierać SHA-256 session.json i report.json.")
    item.archive_ok = True
    if item.group is None:
        raise _SessionImportIssue("Sprzeczne deklaracje serii lub scenariusza w archiwum.",
                                  status="IDENTITY_CONFLICT", codes=("EXPERIMENT_IDENTITY_CONFLICT",))
    experiment = _optional_dict(raw.get("experiment"))
    recorded = _optional_dict(experiment.get("session"))
    collection = _optional_dict(raw.get("research_collection"))
    sid = str(session["session_id"]).strip()
    if not sid or any(str(v).strip() != sid for v in (
        raw.get("session_id"), experiment.get("session_id"), recorded.get("id"),
        recorded.get("session_id"), collection.get("session_id"), manifest.get("session_id")
    ) if v is not None):
        raise _SessionImportIssue("Niezgodny identyfikator sesji w metadanych archiwum.", status="IDENTITY_CONFLICT", codes=("SESSION_ID_CONFLICT",))
    if bundle.experiment_session.get("experiment_session_id") != sid:
        raise _SessionImportIssue("Identyfikator sesji raportu nie odpowiada session.json.", status="IDENTITY_CONFLICT", codes=("SESSION_ID_CONFLICT",))
    if manifest.get("report_id") and manifest["report_id"] != raw.get("report_id"):
        raise _SessionImportIssue("Niezgodny report_id w manifeście i report.json.", status="IDENTITY_CONFLICT", codes=("REPORT_ID_CONFLICT",))
    sources = [session, experiment, recorded, collection, manifest, raw]
    roles = {name: value["mt_detection_roles_complete"] for name, value in
             (("session.json", session), ("manifest.json", manifest), ("research_collection", collection))
             if "mt_detection_roles_complete" in value}
    combination = None
    if f5a:
        # All repeated declarations must agree. Runtime variant_id (fp32 etc.)
        # is deliberately not an alias for the experimental ROI variant.
        def agreed(key):
            values = {str(s[key]).strip() for s in sources if key in s and s[key] is not None}
            if len(values) != 1 or not next(iter(values), ""):
                raise _SessionImportIssue(f"Brak lub konflikt pola {key} w metadanych.", codes=(key.upper() + "_CONFLICT",))
            return values.pop()
        series, scenario = agreed("series_id"), agreed("scenario_id")
        if series != F5A_IMPORT_SERIES or scenario != F5A_IMPORT_SCENARIO:
            raise _SessionImportIssue(f"Sesja należy do innej serii lub scenariusza: {series} / {scenario}.", status="FOREIGN", codes=("PROFILE_SCOPE_MISMATCH",))
        from .mobile_experiment_identity import from_bundle
        identity = from_bundle(bundle)
        if identity.get("status") != "VERIFIED" or identity.get("is_pilot"):
            codes = identity.get("issues", [])
            if "VARIANT_POLICY_CONFLICT" in codes:
                captions = {"r0_full_frame": "R0 — pełna klatka", "full_frame": "R0 — pełna klatka",
                            "r1_one_roi": "R1 — maksymalnie 1 ROI", "r2_two_roi": "R2 — maksymalnie 2 ROI"}
                expected = {"R0": "r0_full_frame", "R1": "r1_one_roi", "R2": "r2_two_roi"}.get(identity.get("variant"), "")
                policy = identity.get("roi_budget_policy") or item.observed("roi_budget_policy")
                raise _SessionImportIssue("Niezgodna strategia ROI", codes=codes, details=(
                    f"Wariant zapisany: {item.observed('variant')}",
                    f"Strategia wykonania: {captions.get(policy, policy)} ({policy})",
                    f"Oczekiwano: {captions.get(expected, 'brak jednoznacznego wariantu')} ({expected})",
                    "Archiwum pozostaje integralne, ale nie jest poprawnym pomiarem zaplanowanego wariantu."))
            raise _SessionImportIssue("Nie można potwierdzić konfiguracji wymaganej przez profil F5A.", codes=codes,
                                      details=(f"Zapisana strategia ROI: {item.observed('roi_budget_policy')}",))
        variant, repeat = agreed("variant"), agreed("replicate_index")
        blocks = set()
        for source in sources:
            for key in ("block", "block_id"):
                if source.get(key):
                    blocks.add(str(source[key]).removeprefix("BLOK_"))
            blocks.update(re.findall(r"\bBLOK_([A-D])\b", str(source.get("notes") or "")))
        if len(blocks) != 1:
            raise _SessionImportIssue("Brak jednoznacznego bloku A–D w metadanych (np. notatka BLOK_A).", codes=("BLOCK_UNPROVEN",))
        if repeat not in {"1", "2", "3"}:
            raise _SessionImportIssue("Powtórzenie poza zakresem r01–r03.", codes=("REPEAT_OUT_OF_RANGE",))
        combination = (variant, f"r{int(repeat):02d}", blocks.pop())
        if combination not in F5A_IMPORT_COMBINATIONS:
            raise _SessionImportIssue("Wariant lub blok poza zakresem F5A.", codes=("COMBINATION_OUT_OF_RANGE",))
        item.profile_ok = True
    return sid, combination, roles


def _stored_session_id(report: MobileBenchmarkReport) -> str:
    return str(_nested_value(report.raw, "desktop_ingest.session_id", "research_collection.session_id",
                             "experiment.session.id", "experiment.session_id", "session_id") or "")


def validate_known_mobile_experiment(bundle: MobileReportBundle) -> None:
    """File import cannot bypass the protocol of an exactly identified F5A source."""
    item = MobileSessionImportItem(Path(bundle.path))
    _observe_session_import(bundle, item)
    if item.is_f5a:
        _session_import_metadata(bundle, item, f5a=True)
    elif item.group is None:
        raise _SessionImportIssue("Sprzeczne deklaracje serii lub scenariusza w archiwum.",
                                  status="IDENTITY_CONFLICT", codes=("EXPERIMENT_IDENTITY_CONFLICT",))


def _classify_session_import(plan: MobileSessionImportPlan, existing_reports) -> None:
    """Fail closed on ambiguous session bytes or multiple sessions in one slot."""
    valid = [i for i in plan.items if i.status in {"NEW", "EXISTING", "DUPLICATE"}]
    for attribute in ("session_id", "combination"):
        groups = {}
        for item in valid:
            key = getattr(item, attribute)
            if key:
                groups.setdefault(key, []).append(item)
        for group in groups.values():
            if len({i.sha256 for i in group}) > 1:
                for item in group:
                    item.status, item.reason = "IDENTITY_CONFLICT", "Różne archiwa deklarują tę samą sesję lub kombinację. Wymaga wyjaśnienia."
                    item.codes = (attribute.upper() + "_CONFLICT",)
    existing_hashes = {r.source_archive_sha256 for r in existing_reports if r.source_archive_sha256}
    existing_ids, existing_combinations = {}, {}
    for report in existing_reports:
        if _stored_session_id(report):
            existing_ids.setdefault(_stored_session_id(report), set()).add(report.source_archive_sha256)
        ingest = _optional_dict(report.raw.get("desktop_ingest"))
        if (ingest.get("series_id") == F5A_IMPORT_SERIES
                and ingest.get("scenario_id") == F5A_IMPORT_SCENARIO
                and isinstance(ingest.get("f5a_combination"), list)):
            key = tuple(ingest["f5a_combination"])
            existing_combinations.setdefault(key, set()).add(report.source_archive_sha256)
    seen = set()
    for item in valid:
        if item.status == "IDENTITY_CONFLICT":
            continue
        if item.session_id in existing_ids and existing_ids[item.session_id] != {item.sha256}:
            item.status, item.reason = "IDENTITY_CONFLICT", "Magazyn zawiera ten session_id z innym lub nieznanym SHA-256."
            item.codes = ("STORED_SESSION_CONFLICT",)
        elif item.combination in existing_combinations and existing_combinations[item.combination] != {item.sha256}:
            item.status, item.reason = "IDENTITY_CONFLICT", "Magazyn zawiera inną sesję dla tej kombinacji F5A."
            item.codes = ("STORED_COMBINATION_CONFLICT",)
        elif item.sha256 in seen:
            item.status, item.reason = "DUPLICATE", "Identyczne archiwum występuje wcześniej w tym folderze."
        else:
            item.status = "EXISTING" if item.sha256 in existing_hashes else "NEW"
            seen.add(item.sha256)
    for item in valid:
        if item.status == "IDENTITY_CONFLICT":
            logger.warning("Import sesji %s: %s", item.path, item.reason)


def plan_mobile_session_import(root: Path, existing_reports=(), *, f5a: bool | None = None,
                               progress: Callable | None = None, cancelled: Callable | None = None,
                               prepare_report: Callable | None = None) -> MobileSessionImportPlan:
    """Read-only preflight. The legacy f5a argument cannot override source identity."""
    try:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(str(root))
        with os.scandir(root):
            pass
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(mobile_import_path_error(exc, root)) from exc
    plan = MobileSessionImportPlan(root)
    paths = []
    for directory, dirs, files in os.walk(root, onerror=lambda e: plan.scan_errors.append(
            mobile_import_path_error(e, e.filename or root) + f"\nSzczegóły: {e}"), followlinks=False):
        if cancelled and cancelled():
            plan.cancelled = True
            return plan
        # Avoid cycles and escaping through Windows junctions as well as symlinks.
        dirs[:] = sorted(d for d in dirs if not Path(directory, d).is_symlink()
                         and not getattr(os.path, "isjunction", lambda p: False)(Path(directory, d)))
        paths.extend(Path(directory, name) for name in files if Path(name).suffix.lower() == ".alprsession")
        plan.found_count = len(paths)
        if progress:
            progress(0, 0, Path(directory))
    plan.found_count = len(paths)
    for index, path in enumerate(sorted(paths, key=lambda p: (str(p).casefold(), str(p))), 1):
        if cancelled and cancelled():
            plan.cancelled = True
            break
        if progress:
            progress(index - 1, len(paths), path)
        item = MobileSessionImportItem(path)
        plan.items.append(item)
        try:
            if path.is_symlink():
                raise ValueError("Dowiązanie do pliku nie jest archiwum źródłowym.")
            bundles = read_mobile_report_bundles(path)
            if len(bundles) != 1:
                raise ValueError("Archiwum musi zawierać jedną sesję.")
            bundle = bundles[0]
            item.sha256 = bundle.source_archive_sha256
            _observe_session_import(bundle, item)
            if _file_sha256(path) != item.sha256:
                raise _SessionImportIssue("SHA-256 źródła zmieniło się podczas odczytu.", status="ERROR", codes=("SOURCE_CHANGED",))
            item.session_id, item.combination, roles = _session_import_metadata(bundle, item, f5a=item.is_f5a)
            raw = dict(bundle.report.raw)
            raw["desktop_ingest"] = dict(raw.get("desktop_ingest") or {}, session_id=item.session_id,
                                          mt_detection_roles_complete_sources=roles)
            if item.is_f5a:
                raw["desktop_ingest"].update(series_id=F5A_IMPORT_SERIES,
                                             scenario_id=F5A_IMPORT_SCENARIO,
                                             f5a_combination=list(item.combination))
            report = replace(bundle.report, raw=raw)
            item.report = prepare_report(report, bundle=bundle) if prepare_report else report
        except _SessionImportIssue as exc:
            item.status, item.reason, item.codes, item.details = exc.status, str(exc), exc.codes, exc.details
            if item.is_f5a:
                item.profile_ok = False
        except Exception as exc:
            item.status, item.archive_ok = "INVALID", False
            item.reason = mobile_import_path_error(exc, path) if isinstance(exc, OSError) else "Archiwum nie przeszło kontroli formatu lub integralności."
            item.codes, item.details = ("ARCHIVE_INVALID",), (str(exc),)
        if item.status not in {"NEW", "EXISTING", "DUPLICATE"}:
            logger.warning("Import sesji %s: %s", path, item.reason)
        if progress:
            progress(index, len(paths), path)
    plan.items.sort(key=lambda i: (i.combination or ("~", "~", "~"), str(i.path).casefold(), str(i.path)))
    _classify_session_import(plan, existing_reports)
    plan.f5a = F5A_IMPORT_GROUP in plan.groups
    plan.selected_group = F5A_IMPORT_GROUP if plan.f5a else next(iter(plan.groups)) if len(plan.groups) == 1 else None
    if cancelled and cancelled():
        plan.cancelled = True
    return plan


class MobilePackageExperimentStore:
    """Persistent storage for package candidates and Android reports."""

    FILE_NAME = MOBILE_PACKAGE_EXPERIMENT_STORE_FILE_NAME

    def __init__(self, root_dir: Path | None = None):
        default_root = getattr(CONFIG, "DIR_7_RANKINGS_MOBILE_PACKAGES", CONFIG.DIR_7_RANKINGS / "mobile_packages")
        self.root_dir = Path(root_dir) if root_dir else Path(default_root)
        self.file_path = self.root_dir / self.FILE_NAME
        self.candidates: dict[str, MobilePackageCandidate] = {}
        self.reports: list[MobileBenchmarkReport] = []
        self._load()

    def _load(self) -> None:
        self.candidates = {}
        self.reports = []
        if not self.file_path.exists():
            return
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8-sig"))
            self.candidates = {
                candidate.package_id: candidate
                for candidate in (
                    MobilePackageCandidate.from_dict(item)
                    for item in list(data.get("candidates") or [])
                )
            }
            loaded_reports = [
                MobileBenchmarkReport.from_dict(item)
                for item in list(data.get("reports") or [])
            ]
            self.reports = deduplicate_mobile_reports(loaded_reports)
            removed_count = len(loaded_reports) - len(self.reports)
            if removed_count > 0:
                logger.info(f"Usunieto logiczne duplikaty raportow mobilnych z widoku: {removed_count}")
        except Exception as exc:
            logger.error(f"Could not load mobile package experiments: {exc}")
            self.candidates = {}
            self.reports = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MOBILE_PACKAGE_EXPERIMENT_SCHEMA,
            "updated_at": _utc_now_iso(),
            "candidates": [candidate.to_dict() for candidate in self.candidates.values()],
            "reports": [report.to_dict() for report in self.reports],
            "scores": [score.to_dict() for score in self.score_reports()],
        }

    def save(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.reports = deduplicate_mobile_reports(self.reports)
        tmp_path = self.file_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.file_path)

    def add_candidate(self, candidate: MobilePackageCandidate, *, save: bool = True) -> MobilePackageCandidate:
        problems = candidate.validate()
        if problems:
            raise ValueError("; ".join(problems))
        self.candidates[candidate.package_id] = candidate
        if save:
            self.save()
        return candidate

    def add_report(self, report: MobileBenchmarkReport, *, save: bool = True) -> MobileBenchmarkReport:
        identity = _mobile_report_dedupe_key(report)
        self.reports = [existing for existing in self.reports if _mobile_report_dedupe_key(existing) != identity]
        self.reports.append(report)
        if save:
            self.save()
        return report

    def import_mobile_report_file(self, report_path: Path, *, save: bool = True) -> list[MobileBenchmarkReport]:
        safe_path = Path(report_path)
        imported, changed = [], False
        for bundle in read_mobile_report_bundles(safe_path):
            if not bundle.validation.ok:
                continue
            validate_known_mobile_experiment(bundle)
            existing = self.existing_import(bundle.report)
            imported.append(existing or self.add_report(bundle.report, save=False))
            changed |= existing is None
        if save and changed:
            self.save()
        return imported

    def existing_import(self, report: MobileBenchmarkReport) -> MobileBenchmarkReport | None:
        """Do not replace operator reviews or locators when importing the same bytes."""
        return next((r for r in self.reports if report.source_archive_sha256
                     and r.source_archive_sha256 == report.source_archive_sha256
                     and r.identity == report.identity), None)

    def import_mobile_session_plan(self, plan: MobileSessionImportPlan, *,
                                   progress: Callable | None = None,
                                   cancelled: Callable | None = None) -> MobileSessionImportPlan:
        """Commit a reviewed plan through add_report/save; retain existing reviews."""
        if plan.cancelled:
            return plan
        if plan.selected_group is None and plan.groups:
            raise ValueError("Wybierz jawnie grupę sesji przed importem.")
        _classify_session_import(plan, self.reports)
        pending = list(plan.eligible_items)
        for index, item in enumerate(pending, 1):
            if cancelled and cancelled():
                plan.cancelled = True
                break
            previous = list(self.reports)
            try:
                if _file_sha256(item.path) != item.sha256:
                    raise ValueError("SHA-256 źródła zmieniło się od podsumowania. Wykonaj ponowny skan.")
                if cancelled and cancelled():
                    plan.cancelled = True
                    break
                self.add_report(item.report, save=True)
                item.status = "IMPORTED"
            except Exception as exc:
                self.reports = previous
                item.status, item.reason = "ERROR", "Nie udało się zapisać sesji lub jej źródło uległo zmianie."
                item.codes, item.details = ("IMPORT_WRITE_OR_SOURCE_ERROR",), (str(exc),)
                logger.exception("Nie udało się zaimportować sesji: %s", item.path)
            if progress:
                progress(index, len(pending), item.path)
        plan.finished = True
        return plan

    def score_reports(
        self,
        *,
        weights: dict[str, float] | None = None,
        targets: dict[str, float] | None = None,
    ) -> list[MobilePackageScore]:
        scores = [score_mobile_report(self._current_review_quality(report), weights=weights, targets=targets) for report in self.reports]
        return sorted(scores, key=lambda score: (score.rejected, -score.total, score.package_id, score.variant_id))

    @staticmethod
    def _current_review_quality(report: MobileBenchmarkReport) -> MobileBenchmarkReport:
        reference = report.raw.get("human_review", {})
        if report.quality.get("quality_source") != "human_review" or not isinstance(reference, dict):
            return report
        try:
            review = json.loads(Path(reference["sidecar_path"]).read_text(encoding="utf-8"))
            current = (review.get("review_status") == "COMPLETED"
                       and review.get("source_archive_sha256") == report.source_archive_sha256
                       and review.get("review_revision") == reference.get("review_revision"))
        except (OSError, ValueError, KeyError):
            current = False
        return report if current else replace(report, quality=dict(reference.get("original_quality", {})))

    def apply_human_review(self, session, *, save: bool = True) -> bool:
        """Publish completed quality; editing again restores the imported baseline."""
        session.verify_source(force=session.review.review_status == "COMPLETED")
        review = session.review
        completed = review.review_status == "COMPLETED"
        if completed and session.completion_issues():
            raise ValueError("Weryfikacja nie ma wszystkich wymaganych decyzji.")
        stats = session.statistics()["summary"] if completed else {}
        changed = False
        for index, report in enumerate(self.reports):
            if report.source_archive_sha256 != review.source_archive_sha256:
                continue
            previous = report.raw.get("human_review", {})
            if not completed and not previous:
                continue
            original = dict(previous.get("original_quality", report.quality))
            reference = {"review_id": review.review_id, "review_revision": review.review_revision,
                         "review_mode": review.review_mode,
                         "review_status": review.review_status, "source_archive_sha256": review.source_archive_sha256,
                         "sidecar_path": str(session.sidecar_path.resolve()), "original_quality": original}
            quality = dict(stats, quality_source="human_review", review_status="COMPLETED",
                           source_archive_sha256=review.source_archive_sha256, units="ratio",
                           available=bool(stats.get("evaluable_reads")), exact_match_rate=stats.get("exact_read_rate"),
                           ground_truth_samples=stats.get("evaluable_subjects"), unit="crop") if completed else original
            self.reports[index] = replace(report, quality=quality, raw=dict(report.raw, human_review=reference))
            changed = True
        if changed and save:
            self.save()
        return changed
