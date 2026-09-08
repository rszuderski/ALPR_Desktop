#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Character-class balance diagnostics for YOLO MZ datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
from datetime import datetime
import hashlib
import json
from math import ceil
from pathlib import Path
import re
from statistics import median
from typing import Any, Mapping

from ..utils import safe_load_yaml


CHARACTER_BALANCE_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARACTER_CLASS_DISTRIBUTION_SCHEMA = "alpr.character_class_distribution.v1"
CHARACTER_BALANCE_PLAN_SCHEMA = "alpr.character_balance_plan.v1"
CHARACTER_TRAINING_VARIANT_SCHEMA = "alpr.mz_training_variant.v1"
CHARACTER_REPRESENTATION_THRESHOLD_POLICY = "auto_v1"
CHARACTER_REPRESENTATION_DEFAULT_MAX_VARIANTS_PER_SOURCE = 24
CHARACTER_BALANCE_SPLITS = ("train", "val", "test")
CHARACTER_BALANCE_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_ANALYSIS_CACHE: dict[tuple[str, str, str], "CharacterClassDistribution"] = {}
_EXPLICIT_AUG_SUFFIX_RE = re.compile(
    r"(?i)(?:__aug[_-]?\d+|[_-]aug(?:mented)?[_-]?\d*)$"
)


@dataclass(frozen=True)
class CharacterClassMapValidation:
    ok: bool
    status: str
    expected: tuple[str, ...]
    detected: tuple[str, ...] = field(default_factory=tuple)
    first_mismatch_index: int | None = None
    message: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "status": self.status,
            "expected": list(self.expected),
            "detected": list(self.detected),
            "first_mismatch_index": self.first_mismatch_index,
            "message": self.message,
            "warnings": list(self.warnings),
        }


class CharacterClassMapValidationError(ValueError):
    """Raised when data.yaml cannot be safely interpreted as the MZ alphabet."""

    def __init__(self, validation: CharacterClassMapValidation):
        self.validation = validation
        super().__init__(validation.message or "Mapa klas datasetu nie jest zgodna ze standardem MZ.")


@dataclass(frozen=True)
class CharacterClassDistributionRow:
    class_id: int
    symbol: str
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    total_count: int = 0
    train_share: float = 0.0
    unique_train_plate_count: int = 0
    unique_val_plate_count: int = 0
    unique_test_plate_count: int = 0
    unique_total_plate_count: int = 0
    count_status: str = "OK"
    diversity_status: str = "OK"
    status: str = "OK"
    target_count: int = 0
    deficit_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": int(self.class_id),
            "symbol": self.symbol,
            "train_count": int(self.train_count),
            "val_count": int(self.val_count),
            "test_count": int(self.test_count),
            "total_count": int(self.total_count),
            "train_share": float(self.train_share),
            "unique_train_plate_count": int(self.unique_train_plate_count),
            "unique_val_plate_count": int(self.unique_val_plate_count),
            "unique_test_plate_count": int(self.unique_test_plate_count),
            "unique_total_plate_count": int(self.unique_total_plate_count),
            "count_status": self.count_status,
            "diversity_status": self.diversity_status,
            "status": self.status,
            "target_count": int(self.target_count),
            "deficit_count": int(self.deficit_count),
        }


@dataclass(frozen=True)
class CharacterClassDistribution:
    schema: str
    dataset_root: str
    dataset_yaml: str
    generated_at: str
    alphabet: str
    layout: str
    diagnostic_split: str
    summary: dict[str, Any] = field(default_factory=dict)
    classes: list[CharacterClassDistributionRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "dataset_root": self.dataset_root,
            "dataset_yaml": self.dataset_yaml,
            "generated_at": self.generated_at,
            "alphabet": self.alphabet,
            "layout": self.layout,
            "diagnostic_split": self.diagnostic_split,
            "summary": dict(self.summary),
            "classes": [row.to_dict() for row in self.classes],
        }

    def csv_rows(self) -> list[dict[str, Any]]:
        return [row.to_dict() for row in self.classes]


@dataclass(frozen=True)
class CharacterBalanceAugmentationCandidate:
    source_key: str
    label_path: str
    image_path: str = ""
    symbols: tuple[str, ...] = field(default_factory=tuple)
    symbol_counts: dict[str, int] = field(default_factory=dict)
    priority: float = 0.0
    max_augmented_variants: int = 0
    planned_variants: int = 0
    existing_augmented_variants: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "label_path": self.label_path,
            "image_path": self.image_path,
            "symbols": list(self.symbols),
            "symbol_counts": {str(key): int(value) for key, value in dict(self.symbol_counts or {}).items()},
            "priority": float(self.priority),
            "max_augmented_variants": int(self.max_augmented_variants),
            "planned_variants": int(self.planned_variants),
            "existing_augmented_variants": int(self.existing_augmented_variants),
        }


@dataclass(frozen=True)
class CharacterBalancePlan:
    schema: str
    created_at: str
    base_dataset: str
    alphabet: str
    target_ratio: float
    target_count: int
    max_augmented_variants_per_source: int
    selection_policy: str = "deficit_progressive_reuse_v1"
    threshold_policy: str = CHARACTER_REPRESENTATION_THRESHOLD_POLICY
    planned_images: int = 0
    target_count_by_symbol: dict[str, int] = field(default_factory=dict)
    requested_extra_by_symbol: dict[str, int] = field(default_factory=dict)
    synthetic_count_by_symbol: dict[str, int] = field(default_factory=dict)
    predicted_deficit_after: dict[str, int] = field(default_factory=dict)
    feasible: bool = True
    completion_status: str = "REPRESENTATION_OK"
    unique_real_sources_used: int = 0
    reuse_rounds_used: int = 0
    mean_augmented_variants_per_used_source: float = 0.0
    max_augmented_variants_from_single_source: int = 0
    source_diversity_warnings: tuple[str, ...] = field(default_factory=tuple)
    train_sources_by_symbol: dict[str, int] = field(default_factory=dict)
    base_dataset_fingerprint_sha256: str = ""
    deficit_by_symbol: dict[str, int] = field(default_factory=dict)
    candidates: tuple[CharacterBalanceAugmentationCandidate, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "created_at": self.created_at,
            "base_dataset": self.base_dataset,
            "alphabet": self.alphabet,
            "target_ratio": float(self.target_ratio),
            "target_count": int(self.target_count),
            "selection_policy": self.selection_policy,
            "threshold_policy": self.threshold_policy,
            "planned_images": int(self.planned_images),
            "target_count_by_symbol": dict(self.target_count_by_symbol),
            "requested_extra_by_symbol": dict(self.requested_extra_by_symbol),
            "synthetic_count_by_symbol": dict(self.synthetic_count_by_symbol),
            "predicted_deficit_after": dict(self.predicted_deficit_after),
            "feasible": bool(self.feasible),
            "completion_status": self.completion_status,
            "unique_real_sources_used": int(self.unique_real_sources_used),
            "reuse_rounds_used": int(self.reuse_rounds_used),
            "mean_augmented_variants_per_used_source": float(self.mean_augmented_variants_per_used_source),
            "max_augmented_variants_from_single_source": int(self.max_augmented_variants_from_single_source),
            "source_diversity_warnings": list(self.source_diversity_warnings),
            "train_sources_by_symbol": dict(self.train_sources_by_symbol),
            "base_dataset_fingerprint_sha256": self.base_dataset_fingerprint_sha256,
            "max_augmented_variants_per_source": int(self.max_augmented_variants_per_source),
            "deficit_by_symbol": dict(self.deficit_by_symbol),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CharacterRealSourceCandidate:
    source_key: str
    source_path: str
    expected_text: str
    metadata_field: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "source_path": self.source_path,
            "expected_text": self.expected_text,
            "metadata_field": self.metadata_field,
        }


@dataclass(frozen=True)
class CharacterRealSourceSearchRow:
    symbol: str
    current_train_count: int = 0
    current_unique_train: int = 0
    available_unused_real_sources: int = 0
    candidates: tuple[CharacterRealSourceCandidate, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "current_train_count": int(self.current_train_count),
            "current_unique_train": int(self.current_unique_train),
            "available_unused_real_sources": int(self.available_unused_real_sources),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def analyze_character_class_distribution(
    dataset_root: Path | str,
    *,
    target_ratio: float = 0.50,
) -> CharacterClassDistribution:
    """Analyze YOLO label files for the fixed MZ alphabet."""

    root, yaml_path = _normalize_dataset_root(dataset_root)
    target_ratio = _normalize_balance_target_ratio(target_ratio)
    root_key = _path_key(root)
    split_dirs, layout = _discover_label_dirs(root)
    class_map_validation = _validate_character_class_map(yaml_path, layout)
    signature = _build_label_signature(split_dirs, yaml_path)
    cache_key = (root_key, signature, f"target={target_ratio:.6f}")
    cached = _ANALYSIS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    alphabet = CHARACTER_BALANCE_ALPHABET
    class_count = len(alphabet)
    counts = {
        split: [0 for _ in range(class_count)]
        for split in (*CHARACTER_BALANCE_SPLITS, "unsplit")
    }
    unique_sources = {
        split: [set() for _ in range(class_count)]
        for split in (*CHARACTER_BALANCE_SPLITS, "unsplit", "total")
    }
    label_files_by_split = {split: 0 for split in (*CHARACTER_BALANCE_SPLITS, "unsplit")}
    files_read = 0
    empty_label_files = 0
    invalid_label_lines = 0
    invalid_class_ids = 0
    source_map = _load_augmentation_source_map(root)

    for split_name, label_dirs in split_dirs.items():
        for label_dir in label_dirs:
            try:
                label_paths = sorted(
                    path for path in label_dir.iterdir()
                    if path.is_file() and path.suffix.lower() == ".txt"
                )
            except Exception:
                label_paths = []
            label_files_by_split[split_name] = label_files_by_split.get(split_name, 0) + len(label_paths)
            for label_path in label_paths:
                files_read += 1
                source_key = _normalize_source_plate_key(label_path.stem, source_map)
                classes_in_source: set[int] = set()
                file_had_payload = False
                try:
                    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                except Exception:
                    invalid_label_lines += 1
                    continue
                for line in lines:
                    parsed, status = _parse_label_class_id(line, class_count)
                    if status == "skip":
                        continue
                    file_had_payload = True
                    if status == "invalid_line":
                        invalid_label_lines += 1
                        continue
                    if status == "invalid_class":
                        invalid_class_ids += 1
                        continue
                    if parsed is None:
                        continue
                    counts[split_name][parsed] += 1
                    classes_in_source.add(parsed)
                if not file_had_payload:
                    empty_label_files += 1
                for class_id in classes_in_source:
                    unique_sources[split_name][class_id].add(source_key)
                    unique_sources["total"][class_id].add(source_key)

    train_counts = counts["train"]
    val_counts = counts["val"]
    test_counts = counts["test"]
    total_counts = [
        train_counts[index] + val_counts[index] + test_counts[index] + counts["unsplit"][index]
        for index in range(class_count)
    ]
    total_train_labels = sum(train_counts)
    has_train_split = any(split_dirs.get(split) for split in CHARACTER_BALANCE_SPLITS)
    diagnostic_split = "train" if has_train_split and any(split_dirs.get("train")) else "total"
    diagnostic_counts = train_counts if diagnostic_split == "train" else total_counts
    diagnostic_unique_counts = [
        len(unique_sources["train"][index]) if diagnostic_split == "train" else len(unique_sources["total"][index])
        for index in range(class_count)
    ]

    median_nonzero_count = _median_nonzero(diagnostic_counts)
    median_nonzero_unique = _median_nonzero(diagnostic_unique_counts)
    low_count_limit = 0.25 * median_nonzero_count if median_nonzero_count > 0 else 0.0
    low_diversity_limit = 0.25 * median_nonzero_unique if median_nonzero_unique > 0 else 0.0
    target_count = (
        int(ceil(float(target_ratio) * float(median_nonzero_count)))
        if median_nonzero_count > 0 and target_ratio > 0
        else 0
    )

    rows: list[CharacterClassDistributionRow] = []
    zero_classes: list[str] = []
    low_count_classes: list[str] = []
    low_diversity_classes: list[str] = []
    deficit_classes: list[str] = []
    total_deficit_count = 0

    for class_id, symbol in enumerate(alphabet):
        diagnostic_count = int(diagnostic_counts[class_id])
        diagnostic_unique = int(diagnostic_unique_counts[class_id])
        deficit_count = max(0, int(target_count) - diagnostic_count)
        if deficit_count > 0:
            deficit_classes.append(symbol)
            total_deficit_count += deficit_count
        is_zero = diagnostic_count <= 0
        is_low_count = (
            not is_zero
            and median_nonzero_count > 0
            and diagnostic_count < low_count_limit
        )
        is_low_diversity = (
            not is_zero
            and median_nonzero_unique > 0
            and diagnostic_unique < low_diversity_limit
        )
        if is_zero:
            count_status = "CRITICAL"
            zero_classes.append(symbol)
        elif is_low_count:
            count_status = "LOW"
            low_count_classes.append(symbol)
        else:
            count_status = "OK"

        if is_zero or is_low_diversity:
            diversity_status = "LOW_DIVERSITY"
            if not is_zero:
                low_diversity_classes.append(symbol)
        else:
            diversity_status = "OK"

        if count_status == "CRITICAL":
            status = "CRITICAL"
        elif count_status == "LOW" and diversity_status == "LOW_DIVERSITY":
            status = "LOW+LOW_DIVERSITY"
        elif count_status == "LOW":
            status = "LOW"
        elif diversity_status == "LOW_DIVERSITY":
            status = "LOW_DIVERSITY"
        else:
            status = "OK"

        rows.append(
            CharacterClassDistributionRow(
                class_id=class_id,
                symbol=symbol,
                train_count=int(train_counts[class_id]),
                val_count=int(val_counts[class_id]),
                test_count=int(test_counts[class_id]),
                total_count=int(total_counts[class_id]),
                train_share=(float(train_counts[class_id]) / float(total_train_labels) if total_train_labels > 0 else 0.0),
                unique_train_plate_count=len(unique_sources["train"][class_id]),
                unique_val_plate_count=len(unique_sources["val"][class_id]),
                unique_test_plate_count=len(unique_sources["test"][class_id]),
                unique_total_plate_count=len(unique_sources["total"][class_id]),
                count_status=count_status,
                diversity_status=diversity_status,
                status=status,
                target_count=target_count,
                deficit_count=deficit_count,
            )
        )

    minimum_count = min(diagnostic_counts) if diagnostic_counts else 0
    maximum_count = max(diagnostic_counts) if diagnostic_counts else 0
    minimum_unique = min(diagnostic_unique_counts) if diagnostic_unique_counts else 0
    nonzero_min = min((value for value in diagnostic_counts if value > 0), default=0)
    max_min_ratio = (float(maximum_count) / float(nonzero_min)) if nonzero_min > 0 else None

    summary = {
        "total_labels": int(total_train_labels if diagnostic_split == "train" else sum(total_counts)),
        "total_labels_all_splits": int(sum(total_counts)),
        "minimum_class_count": int(minimum_count),
        "maximum_class_count": int(maximum_count),
        "median_class_count": float(median(diagnostic_counts)) if diagnostic_counts else 0.0,
        "median_nonzero_class_count": float(median_nonzero_count),
        "balance_target_ratio": float(target_ratio),
        "target_class_count": int(target_count),
        "total_deficit_count": int(total_deficit_count),
        "deficit_classes": deficit_classes,
        "deficit_policy": "max(0, target_count - diagnostic_count)",
        "max_min_ratio": max_min_ratio,
        "zero_classes": zero_classes,
        "low_count_classes": low_count_classes,
        "minimum_unique_plate_count": int(minimum_unique),
        "median_unique_plate_count": float(median(diagnostic_unique_counts)) if diagnostic_unique_counts else 0.0,
        "median_nonzero_unique_plate_count": float(median_nonzero_unique),
        "low_diversity_classes": low_diversity_classes,
        "low_count_threshold": float(low_count_limit),
        "low_diversity_threshold": float(low_diversity_limit),
        "files_read": int(files_read),
        "empty_label_files": int(empty_label_files),
        "invalid_label_lines": int(invalid_label_lines),
        "invalid_class_ids": int(invalid_class_ids),
        "class_map": class_map_validation.to_dict(),
        "warnings": list(class_map_validation.warnings),
        "label_files": {
            split: int(label_files_by_split.get(split, 0) or 0)
            for split in (*CHARACTER_BALANCE_SPLITS, "unsplit")
        },
    }

    distribution = CharacterClassDistribution(
        schema=CHARACTER_CLASS_DISTRIBUTION_SCHEMA,
        dataset_root=str(root),
        dataset_yaml=str(yaml_path or ""),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        alphabet=alphabet,
        layout=layout,
        diagnostic_split=diagnostic_split,
        summary=summary,
        classes=rows,
    )
    if len(_ANALYSIS_CACHE) > 12:
        _ANALYSIS_CACHE.clear()
    _ANALYSIS_CACHE[cache_key] = distribution
    return distribution


def plan_character_train_augmentation(
    dataset_root: Path | str,
    *,
    target_ratio: float = 0.50,
    target_count: int | None = None,
    max_augmented_variants_per_source: int = CHARACTER_REPRESENTATION_DEFAULT_MAX_VARIANTS_PER_SOURCE,
    target_count_by_symbol: Mapping[str, int] | None = None,
    extra_count_by_symbol: Mapping[str, int] | None = None,
    respect_existing_augmented_variants: bool = True,
) -> CharacterBalancePlan:
    """Build a train-only MZ representation plan.

    The function is deliberately non-destructive: it does not create images and
    never touches val/test. The allocation prefers many real sources first and
    only then reuses the same source again when a deficit still exists.
    """

    root, _yaml_path = _normalize_dataset_root(dataset_root)
    distribution = analyze_character_class_distribution(root, target_ratio=target_ratio)
    target_ratio = float(distribution.summary.get("balance_target_ratio", target_ratio) or 0.50)
    resolved_target_count = int(distribution.summary.get("target_class_count", 0) or 0)
    if target_count is not None:
        try:
            resolved_target_count = max(0, int(float(str(target_count).strip().replace(",", "."))))
        except Exception:
            resolved_target_count = int(distribution.summary.get("target_class_count", 0) or 0)
    max_per_source = max(0, int(max_augmented_variants_per_source or 0))
    base_fingerprint = build_character_dataset_file_fingerprint(root)
    manual_targets = _normalize_target_count_by_symbol(target_count_by_symbol)
    manual_extras = _normalize_target_count_by_symbol(extra_count_by_symbol)
    diagnostic_counts = {
        row.symbol: int(row.train_count if distribution.diagnostic_split == "train" else row.total_count)
        for row in distribution.classes
    }
    desired_counts = {}
    for row in distribution.classes:
        current_count = int(diagnostic_counts.get(row.symbol, 0) or 0)
        desired = int(manual_targets.get(row.symbol, resolved_target_count))
        if row.symbol in manual_extras:
            desired = max(desired, current_count + int(manual_extras.get(row.symbol, 0) or 0))
        desired_counts[row.symbol] = max(0, int(desired))
    deficit_by_symbol = {
        row.symbol: max(
            0,
            int(desired_counts.get(row.symbol, resolved_target_count)) - int(diagnostic_counts.get(row.symbol, 0) or 0),
        )
        for row in distribution.classes
        if (int(desired_counts.get(row.symbol, resolved_target_count)) - int(diagnostic_counts.get(row.symbol, 0) or 0)) > 0
    }
    distribution_rows = {row.symbol: row for row in distribution.classes}

    split_dirs, layout = _discover_label_dirs(root)
    planning_label_dirs, planning_scope = _character_augmentation_planning_label_dirs(split_dirs, layout)
    warnings: list[str] = []
    if not planning_label_dirs:
        warnings.append("Brak jawnego splitu train; plan augmentacji train-only nie wybiera źródeł.")
    elif planning_scope == "unsplit":
        warnings.append(
            "Brak jawnego splitu train; używam płaskiego katalogu labels jako prognozy przed utworzeniem splitu."
        )

    grouped_candidates: dict[str, list[CharacterBalanceAugmentationCandidate]] = {}
    source_map = _load_augmentation_source_map(root)
    existing_augmented_by_source = (
        _count_existing_augmented_variants_by_source(root, source_map)
        if bool(respect_existing_augmented_variants)
        else {}
    )
    for label_dir in planning_label_dirs:
        for label_path in _iter_label_files(label_dir):
            source_key = _normalize_source_plate_key(label_path.stem, source_map)
            symbol_counts = _symbol_counts_from_label_file(label_path)
            symbols = tuple(symbol for symbol in CHARACTER_BALANCE_ALPHABET if int(symbol_counts.get(symbol, 0) or 0) > 0)
            priority = sum(
                float(min(int(deficit_by_symbol.get(symbol, 0) or 0), int(count or 0)))
                for symbol, count in symbol_counts.items()
            )
            if priority <= 0:
                continue
            image_path = _resolve_image_for_label(root, label_path)
            candidate = CharacterBalanceAugmentationCandidate(
                source_key=source_key,
                label_path=str(label_path),
                image_path=str(image_path or ""),
                symbols=symbols,
                symbol_counts=symbol_counts,
                priority=round(priority, 4),
                max_augmented_variants=max_per_source,
                existing_augmented_variants=int(existing_augmented_by_source.get(source_key, 0) or 0),
            )
            grouped_candidates.setdefault(source_key, []).append(candidate)

    source_candidates: list[CharacterBalanceAugmentationCandidate] = []
    for source_key, group in grouped_candidates.items():
        group.sort(
            key=lambda item: (
                0 if _is_original_source_label(Path(item.label_path), item.source_key, source_map) else 1,
                str(Path(item.label_path).name).lower(),
            )
        )
        selected = group[0]
        if len(group) > 1:
            has_original = any(
                _is_original_source_label(Path(item.label_path), item.source_key, source_map)
                for item in group
            )
            if not has_original:
                warnings.append(
                    "Źródło "
                    f"{source_key} ma tylko kopie augmentowane; wybrano deterministycznie "
                    f"{Path(selected.label_path).name}."
                )
        source_candidates.append(selected)

    planned_by_source, predicted_deficit_after = _allocate_character_representation_plan(
        deficit_by_symbol,
        source_candidates,
        existing_augmented_by_source=existing_augmented_by_source,
        max_augmented_variants_per_source=max_per_source,
    )
    planned_candidates: list[CharacterBalanceAugmentationCandidate] = []
    for candidate in source_candidates:
        planned_variants = int(planned_by_source.get(candidate.source_key, 0) or 0)
        if planned_variants <= 0:
            continue
        planned_candidates.append(
            CharacterBalanceAugmentationCandidate(
                source_key=candidate.source_key,
                label_path=candidate.label_path,
                image_path=candidate.image_path,
                symbols=candidate.symbols,
                symbol_counts=dict(candidate.symbol_counts or {}),
                priority=float(candidate.priority),
                max_augmented_variants=int(candidate.max_augmented_variants),
                planned_variants=planned_variants,
                existing_augmented_variants=int(candidate.existing_augmented_variants),
            )
        )
    planned_candidates.sort(
        key=lambda item: (
            -int(item.planned_variants),
            -float(item.priority),
            str(item.source_key),
        )
    )
    synthetic_count_by_symbol = _synthetic_counts_from_planned_candidates(planned_candidates)

    missing_sources = [
        symbol
        for symbol in deficit_by_symbol
        if all(int(candidate.symbol_counts.get(symbol, 0) or 0) <= 0 for candidate in source_candidates)
    ]
    if missing_sources:
        warnings.append(
            "Brak źródeł train zawierających: "
            + ", ".join(missing_sources)
            + ". Tego deficytu nie da się uzupełnić samą augmentacją istniejących tablic."
        )
    source_diversity_warnings = tuple(
        symbol
        for symbol, row in distribution_rows.items()
        if symbol in deficit_by_symbol and str(row.diversity_status or "").upper() == "LOW_DIVERSITY"
    )
    if source_diversity_warnings:
        warnings.append(
            "Niska różnorodność realnych źródeł dla: "
            + ", ".join(source_diversity_warnings)
            + ". Augmentacja zwiększy liczebność, ale nie zastąpi nowych realnych tablic."
        )

    planned_images = sum(int(value or 0) for value in planned_by_source.values())
    train_sources_by_symbol = {
        symbol: sum(1 for candidate in source_candidates if int(candidate.symbol_counts.get(symbol, 0) or 0) > 0)
        for symbol in deficit_by_symbol
    }
    remaining_deficits = {
        symbol: int(value)
        for symbol, value in predicted_deficit_after.items()
        if int(value or 0) > 0
    }
    feasible = not remaining_deficits
    if not deficit_by_symbol:
        completion_status = "REPRESENTATION_OK"
    elif feasible:
        completion_status = (
            "REPRESENTATION_OK_WITH_DIVERSITY_WARNING"
            if source_diversity_warnings
            else "REPRESENTATION_OK"
        )
    else:
        completion_status = "PLAN_NOT_FEASIBLE"
    used_counts = [int(value or 0) for value in planned_by_source.values() if int(value or 0) > 0]
    mean_reuse = (sum(used_counts) / len(used_counts)) if used_counts else 0.0

    return CharacterBalancePlan(
        schema=CHARACTER_BALANCE_PLAN_SCHEMA,
        created_at=datetime.now().isoformat(timespec="seconds"),
        base_dataset=str(root),
        alphabet=CHARACTER_BALANCE_ALPHABET,
        target_ratio=target_ratio,
        target_count=resolved_target_count,
        target_count_by_symbol={symbol: count for symbol, count in sorted(desired_counts.items()) if count > resolved_target_count},
        requested_extra_by_symbol=manual_extras,
        synthetic_count_by_symbol=dict(sorted(synthetic_count_by_symbol.items())),
        max_augmented_variants_per_source=max_per_source,
        selection_policy="deficit_progressive_reuse_v1",
        threshold_policy=CHARACTER_REPRESENTATION_THRESHOLD_POLICY,
        planned_images=planned_images,
        predicted_deficit_after=remaining_deficits,
        feasible=feasible,
        completion_status=completion_status,
        unique_real_sources_used=len(used_counts),
        reuse_rounds_used=max(used_counts) if used_counts else 0,
        mean_augmented_variants_per_used_source=round(mean_reuse, 4),
        max_augmented_variants_from_single_source=max(used_counts) if used_counts else 0,
        source_diversity_warnings=source_diversity_warnings,
        train_sources_by_symbol=train_sources_by_symbol,
        base_dataset_fingerprint_sha256=str(base_fingerprint.get("sha256") or ""),
        deficit_by_symbol=deficit_by_symbol,
        candidates=tuple(planned_candidates),
        warnings=tuple(warnings),
    )


def collect_character_train_augmentation_source_pool(
    dataset_root: Path | str,
    *,
    max_augmented_variants_per_source: int = CHARACTER_REPRESENTATION_DEFAULT_MAX_VARIANTS_PER_SOURCE,
    respect_existing_augmented_variants: bool = True,
) -> tuple[tuple[CharacterBalanceAugmentationCandidate, ...], dict[str, int]]:
    """Collect train source candidates once for responsive histogram previews."""

    root, _yaml_path = _normalize_dataset_root(dataset_root)
    max_per_source = max(0, int(max_augmented_variants_per_source or 0))
    split_dirs, layout = _discover_label_dirs(root)
    planning_label_dirs, _planning_scope = _character_augmentation_planning_label_dirs(split_dirs, layout)
    if not planning_label_dirs:
        return (), {}

    source_map = _load_augmentation_source_map(root)
    existing_augmented_by_source = (
        _count_existing_augmented_variants_by_source(root, source_map)
        if bool(respect_existing_augmented_variants)
        else {}
    )
    grouped_candidates: dict[str, list[CharacterBalanceAugmentationCandidate]] = {}
    for label_dir in planning_label_dirs:
        for label_path in _iter_label_files(label_dir):
            source_key = _normalize_source_plate_key(label_path.stem, source_map)
            symbol_counts = _symbol_counts_from_label_file(label_path)
            symbols = tuple(symbol for symbol in CHARACTER_BALANCE_ALPHABET if int(symbol_counts.get(symbol, 0) or 0) > 0)
            if not symbols:
                continue
            image_path = _resolve_image_for_label(root, label_path)
            candidate = CharacterBalanceAugmentationCandidate(
                source_key=source_key,
                label_path=str(label_path),
                image_path=str(image_path or ""),
                symbols=symbols,
                symbol_counts=symbol_counts,
                priority=float(len(symbols)),
                max_augmented_variants=max_per_source,
                existing_augmented_variants=int(existing_augmented_by_source.get(source_key, 0) or 0),
            )
            grouped_candidates.setdefault(source_key, []).append(candidate)

    source_candidates: list[CharacterBalanceAugmentationCandidate] = []
    for source_key, group in grouped_candidates.items():
        group.sort(
            key=lambda item: (
                0 if _is_original_source_label(Path(item.label_path), item.source_key, source_map) else 1,
                str(Path(item.label_path).name).lower(),
            )
        )
        source_candidates.append(group[0])
    return tuple(source_candidates), dict(existing_augmented_by_source)


def preview_character_train_synthetic_counts_from_pool(
    extra_count_by_symbol: Mapping[str, int] | None,
    source_candidates: tuple[CharacterBalanceAugmentationCandidate, ...] | list[CharacterBalanceAugmentationCandidate],
    existing_augmented_by_source: Mapping[str, int] | None,
    *,
    max_augmented_variants_per_source: int = CHARACTER_REPRESENTATION_DEFAULT_MAX_VARIANTS_PER_SOURCE,
    respect_existing_augmented_variants: bool = True,
) -> dict[str, int]:
    """Calculate visible per-symbol gains from a cached train source pool."""

    manual_extras = _normalize_target_count_by_symbol(extra_count_by_symbol)
    if not manual_extras:
        return {}
    max_per_source = max(0, int(max_augmented_variants_per_source or 0))
    planned_by_source, _remaining = _allocate_character_representation_plan(
        manual_extras,
        list(source_candidates or ()),
        existing_augmented_by_source=(
            dict(existing_augmented_by_source or {})
            if bool(respect_existing_augmented_variants)
            else {}
        ),
        max_augmented_variants_per_source=max_per_source,
    )
    by_key = {
        str(candidate.source_key): candidate
        for candidate in source_candidates or ()
        if str(getattr(candidate, "source_key", "") or "").strip()
    }
    planned_candidates: list[CharacterBalanceAugmentationCandidate] = []
    for source_key, variants in planned_by_source.items():
        candidate = by_key.get(str(source_key))
        planned_variants = max(0, int(variants or 0))
        if candidate is None or planned_variants <= 0:
            continue
        planned_candidates.append(
            CharacterBalanceAugmentationCandidate(
                source_key=candidate.source_key,
                label_path=candidate.label_path,
                image_path=candidate.image_path,
                symbols=candidate.symbols,
                symbol_counts=dict(candidate.symbol_counts or {}),
                priority=float(candidate.priority),
                max_augmented_variants=int(candidate.max_augmented_variants),
                planned_variants=planned_variants,
                existing_augmented_variants=int(candidate.existing_augmented_variants),
            )
        )
    return _synthetic_counts_from_planned_candidates(planned_candidates)


def find_character_real_source_candidates(
    dataset_root: Path | str,
    search_roots: list[Path | str] | tuple[Path | str, ...],
    *,
    target_ratio: float = 0.50,
    candidate_limit_per_symbol: int = 200,
) -> dict[str, dict[str, Any]]:
    """Find unused real metadata sources that can reduce MZ class deficits.

    The function scans existing JSON/YAML metadata fields only. It does not
    create a new database and it does not mutate the dataset.
    """

    root, _yaml_path = _normalize_dataset_root(dataset_root)
    distribution = analyze_character_class_distribution(root, target_ratio=target_ratio)
    source_rows = {
        row.symbol: row
        for row in distribution.classes
        if (
            int(row.deficit_count) > 0
            or str(row.count_status or "").upper() == "CRITICAL"
            or str(row.diversity_status or "").upper() == "LOW_DIVERSITY"
            or str(row.status or "").upper() in {"CRITICAL", "LOW_DIVERSITY", "LOW+LOW_DIVERSITY"}
        )
    }
    if not source_rows:
        return {}

    used_source_keys = _collect_train_source_keys(root)
    result: dict[str, dict[str, Any]] = {
        symbol: CharacterRealSourceSearchRow(
            symbol=symbol,
            current_train_count=int(row.train_count),
            current_unique_train=int(row.unique_train_plate_count),
        ).to_dict()
        for symbol, row in source_rows.items()
    }
    by_symbol: dict[str, dict[str, CharacterRealSourceCandidate]] = {
        symbol: {} for symbol in source_rows
    }
    limit = max(0, int(candidate_limit_per_symbol or 0))

    for metadata_path in _iter_metadata_files(search_roots):
        payload = _load_metadata_payload(metadata_path)
        if not isinstance(payload, (dict, list)):
            continue
        for source_key, expected_text, field_name in _iter_metadata_expected_texts(payload, metadata_path):
            if not source_key or source_key in used_source_keys:
                continue
            symbols = set(str(expected_text or ""))
            if not symbols:
                continue
            for symbol in source_rows:
                if symbol not in symbols:
                    continue
                bucket = by_symbol.setdefault(symbol, {})
                if source_key in bucket:
                    continue
                if limit and len(bucket) >= limit:
                    continue
                bucket[source_key] = CharacterRealSourceCandidate(
                    source_key=source_key,
                    source_path=str(metadata_path),
                    expected_text=expected_text,
                    metadata_field=field_name,
                )

    for symbol, candidates_by_key in by_symbol.items():
        row = dict(result.get(symbol) or {})
        candidates = tuple(sorted(candidates_by_key.values(), key=lambda item: item.source_key))
        row["available_unused_real_sources"] = len(candidates)
        row["candidates"] = [candidate.to_dict() for candidate in candidates]
        result[symbol] = row
    return result


def build_character_dataset_file_fingerprint(
    dataset_root: Path | str,
    *,
    splits: tuple[str, ...] = CHARACTER_BALANCE_SPLITS,
    include_config: bool | None = None,
) -> dict[str, Any]:
    """Create a content fingerprint for YOLO dataset files.

    The freeze guard for val/test deliberately excludes data.yaml, while the
    whole-dataset fingerprint may include it as configuration metadata.
    """

    root, yaml_path = _normalize_dataset_root(dataset_root)
    requested_splits = tuple(str(split or "").strip() for split in (splits or CHARACTER_BALANCE_SPLITS) if str(split or "").strip())
    if not requested_splits:
        requested_splits = CHARACTER_BALANCE_SPLITS
    if include_config is None:
        include_config = tuple(requested_splits) == tuple(CHARACTER_BALANCE_SPLITS)

    entries: list[dict[str, Any]] = []
    if include_config and yaml_path is not None and yaml_path.exists():
        entries.append(_fingerprint_entry(root, yaml_path, "config"))
    for split_name in requested_splits:
        for folder_name in ("images", "labels"):
            suffixes = {".txt"} if folder_name == "labels" else set(CHARACTER_BALANCE_IMAGE_EXTENSIONS)
            for folder in _dataset_split_file_dirs(root, split_name, folder_name):
                try:
                    paths = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in suffixes)
                except Exception:
                    paths = []
                for path in paths:
                    entries.append(_fingerprint_entry(root, path, split_name))
    entries.sort(key=lambda item: str(item.get("relative_path", "")))
    digest = hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "dataset_root": str(root),
        "splits": list(requested_splits),
        "mode": "content_sha256",
        "includes_config": bool(include_config),
        "file_count": len(entries),
        "sha256": digest,
        "entries": entries,
    }


def compare_character_val_test_unchanged(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two val/test content fingerprints."""

    before_digest = str((before or {}).get("sha256") or "")
    after_digest = str((after or {}).get("sha256") or "")
    return {
        "unchanged": bool(before_digest and before_digest == after_digest),
        "before_sha256": before_digest,
        "after_sha256": after_digest,
        "before_file_count": int((before or {}).get("file_count", 0) or 0),
        "after_file_count": int((after or {}).get("file_count", 0) or 0),
    }


def save_character_distribution_artifacts(
    distribution: CharacterClassDistribution,
    output_dir: Path | str,
    *,
    prefix: str,
) -> dict[str, dict[str, str]]:
    """Persist distribution as CSV/JSON and return path+SHA references."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safe_prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", str(prefix or "").strip()).strip("._-")
    if not safe_prefix:
        safe_prefix = "character_class_distribution"
    csv_path = save_character_class_distribution_csv(distribution, output / f"{safe_prefix}.csv")
    json_path = save_character_class_distribution_json(distribution, output / f"{safe_prefix}.json")
    return {
        "csv": _file_artifact_ref(csv_path),
        "json": _file_artifact_ref(json_path),
    }


def build_character_training_variant_manifest(
    *,
    base_dataset: Path | str,
    before_distribution: CharacterClassDistribution | Path | str,
    after_distribution: CharacterClassDistribution | Path | str | None = None,
    plan: CharacterBalancePlan | None = None,
    sources: Mapping[str, int] | None = None,
    base_dataset_sha_or_fingerprint: Mapping[str, Any] | str | None = None,
    val_test_unchanged: bool | Mapping[str, Any] | None = None,
    augmentation_mode: str | None = None,
    requested_images: int | None = None,
    planned_images: int | None = None,
    generated_images: int | None = None,
    completion_status: str | None = None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    """Create the research manifest skeleton for an MZ training variant."""

    max_per_source = int(getattr(plan, "max_augmented_variants_per_source", 0) or 0)
    source_counts = {
        "real": 0,
        "added_real": 0,
        "augmented_real": 0,
        "synthetic": 0,
        "augmented_synthetic": 0,
    }
    if sources:
        for key in source_counts:
            source_counts[key] = max(0, int(sources.get(key, 0) or 0))
    if isinstance(base_dataset_sha_or_fingerprint, Mapping):
        base_fingerprint_ref: dict[str, Any] | str = _compact_fingerprint_ref(base_dataset_sha_or_fingerprint)
    elif base_dataset_sha_or_fingerprint:
        base_fingerprint_ref = str(base_dataset_sha_or_fingerprint)
    else:
        base_fingerprint_ref = build_character_dataset_file_fingerprint(base_dataset).get("sha256", "")
    planned = int(planned_images if planned_images is not None else getattr(plan, "planned_images", 0) or 0)
    generated = int(generated_images if generated_images is not None else source_counts.get("augmented_real", 0) or 0)
    requested = int(requested_images if requested_images is not None else planned or generated or 0)
    status = str(completion_status or getattr(plan, "completion_status", "") or "").strip()
    return {
        "schema": CHARACTER_TRAINING_VARIANT_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_dataset": str(base_dataset),
        "base_dataset_sha_or_fingerprint": base_fingerprint_ref,
        "base_dataset_content_fingerprint": base_fingerprint_ref,
        "alphabet": CHARACTER_BALANCE_ALPHABET,
        "augmentation_mode": str(augmentation_mode or "manual_train_augmentation"),
        "target_count": int(getattr(plan, "target_count", 0) or 0),
        "target_count_by_symbol": dict(getattr(plan, "target_count_by_symbol", {}) or {}),
        "requested_extra_by_symbol": dict(getattr(plan, "requested_extra_by_symbol", {}) or {}),
        "synthetic_count_by_symbol": dict(getattr(plan, "synthetic_count_by_symbol", {}) or {}),
        "target_ratio": float(getattr(plan, "target_ratio", 0.50) or 0.50),
        "selection_policy": str(getattr(plan, "selection_policy", "deficit_progressive_reuse_v1") or "deficit_progressive_reuse_v1"),
        "threshold_policy": str(getattr(plan, "threshold_policy", CHARACTER_REPRESENTATION_THRESHOLD_POLICY) or CHARACTER_REPRESENTATION_THRESHOLD_POLICY),
        "planned_images": max(0, planned),
        "requested_images": max(0, requested),
        "generated_images": max(0, generated),
        "completion_status": status,
        "stop_reason": str(stop_reason or ""),
        "predicted_deficit_after": dict(getattr(plan, "predicted_deficit_after", {}) or {}),
        "deficit_before": dict(getattr(plan, "deficit_by_symbol", {}) or {}),
        "unique_real_sources_used": int(getattr(plan, "unique_real_sources_used", 0) or 0),
        "reuse_rounds_used": int(getattr(plan, "reuse_rounds_used", 0) or 0),
        "max_augmented_variants_from_single_source": int(getattr(plan, "max_augmented_variants_from_single_source", 0) or 0),
        "mean_augmented_variants_per_used_source": float(getattr(plan, "mean_augmented_variants_per_used_source", 0.0) or 0.0),
        "max_augmented_variants_per_source": max_per_source,
        "source_reuse_safety_guard": {
            "enabled": bool(max_per_source > 0),
            "max_augmented_variants_per_source": max_per_source,
            "scope": "single_original_source",
            "purpose": "prevent one plate crop from dominating MZ class balancing",
        },
        "train_sources_by_symbol": dict(getattr(plan, "train_sources_by_symbol", {}) or {}),
        "sources": source_counts,
        "before_distribution": _distribution_ref(before_distribution),
        "after_distribution": _distribution_ref(after_distribution) if after_distribution is not None else "",
        "augmentation": {
            "max_variants_per_source": max_per_source,
            "mode": str(augmentation_mode or "manual_train_augmentation"),
            "planned_images": max(0, planned),
            "requested_images": max(0, requested),
            "generated_images": max(0, generated),
        },
        "balance_plan": plan.to_dict() if plan is not None else {},
        "val_test_unchanged": (
            bool(val_test_unchanged.get("unchanged"))
            if isinstance(val_test_unchanged, Mapping)
            else (True if val_test_unchanged is None else bool(val_test_unchanged))
        ),
        "val_test_guard": dict(val_test_unchanged) if isinstance(val_test_unchanged, Mapping) else {},
    }


def save_character_class_distribution_csv(
    distribution: CharacterClassDistribution,
    output_path: Path | str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_id",
        "symbol",
        "train_count",
        "val_count",
        "test_count",
        "total_count",
        "train_share",
        "unique_train_plate_count",
        "unique_val_plate_count",
        "unique_test_plate_count",
        "unique_total_plate_count",
        "count_status",
        "diversity_status",
        "status",
        "target_count",
        "deficit_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in distribution.csv_rows():
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def save_character_class_distribution_json(
    distribution: CharacterClassDistribution,
    output_path: Path | str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(distribution.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _normalize_dataset_root(dataset_root: Path | str) -> tuple[Path, Path | None]:
    root = Path(dataset_root)
    if root.is_file() and root.name.lower() == "data.yaml":
        return root.parent, root
    yaml_path = root / "data.yaml"
    return root, yaml_path if yaml_path.exists() else None


def _dataset_split_file_dirs(root: Path, split_name: str, folder_name: str) -> list[Path]:
    """Return YOLO split directories for both supported layouts."""

    candidates = (
        Path(root) / folder_name / split_name,
        Path(root) / split_name / folder_name,
    )
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        try:
            key = str(candidate.resolve()).lower()
        except Exception:
            key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _normalize_balance_target_ratio(value: Any) -> float:
    try:
        ratio = float(str(value).strip().replace(",", "."))
    except Exception:
        ratio = 0.50
    if ratio > 1.0:
        ratio /= 100.0
    return max(0.0, min(2.0, ratio))


def _normalize_target_count_by_symbol(value: Mapping[str, int] | None) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for raw_symbol, raw_count in value.items():
        symbol = str(raw_symbol or "").strip().upper()
        if symbol not in CHARACTER_BALANCE_ALPHABET:
            continue
        try:
            count = int(float(str(raw_count).strip().replace(",", ".")))
        except Exception:
            continue
        if count <= 0:
            continue
        result[symbol] = max(0, count)
    return dict(sorted(result.items()))


def _iter_label_files(label_dir: Path) -> list[Path]:
    try:
        return sorted(
            path for path in Path(label_dir).iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        )
    except Exception:
        return []


def _symbols_from_label_file(label_path: Path) -> tuple[str, ...]:
    counts = _symbol_counts_from_label_file(label_path)
    return tuple(symbol for symbol in CHARACTER_BALANCE_ALPHABET if int(counts.get(symbol, 0) or 0) > 0)


def _character_augmentation_planning_label_dirs(
    split_dirs: Mapping[str, list[Path]],
    layout: str,
) -> tuple[list[Path], str]:
    train_dirs = list(split_dirs.get("train") or [])
    if train_dirs:
        return train_dirs, "train"
    unsplit_dirs = list(split_dirs.get("unsplit") or [])
    if str(layout or "").strip().lower() == "flat" and unsplit_dirs:
        return unsplit_dirs, "unsplit"
    return [], "empty"


def _symbol_counts_from_label_file(label_path: Path) -> dict[str, int]:
    class_count = len(CHARACTER_BALANCE_ALPHABET)
    counts: dict[str, int] = {}
    try:
        lines = Path(label_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        lines = []
    for line in lines:
        class_id, status = _parse_label_class_id(line, class_count)
        if status != "ok" or class_id is None:
            continue
        symbol = CHARACTER_BALANCE_ALPHABET[int(class_id)]
        counts[symbol] = int(counts.get(symbol, 0) or 0) + 1
    return counts


def _count_existing_augmented_variants_by_source(root: Path, source_map: dict[str, str]) -> dict[str, int]:
    split_dirs, layout = _discover_label_dirs(root)
    planning_label_dirs, _planning_scope = _character_augmentation_planning_label_dirs(split_dirs, layout)
    result: dict[str, int] = {}
    for label_dir in planning_label_dirs:
        for label_path in _iter_label_files(label_dir):
            source_key = _normalize_source_plate_key(label_path.stem, source_map)
            if _is_original_source_label(label_path, source_key, source_map):
                continue
            result[source_key] = int(result.get(source_key, 0) or 0) + 1
    return result


def _allocate_character_representation_plan(
    deficit_by_symbol: Mapping[str, int],
    candidates: list[CharacterBalanceAugmentationCandidate],
    *,
    existing_augmented_by_source: Mapping[str, int],
    max_augmented_variants_per_source: int,
) -> tuple[dict[str, int], dict[str, int]]:
    remaining = {
        str(symbol): max(0, int(value or 0))
        for symbol, value in dict(deficit_by_symbol or {}).items()
        if int(value or 0) > 0
    }
    if not remaining:
        return {}, {}

    max_per_source = max(0, int(max_augmented_variants_per_source or 0))
    if max_per_source <= 0:
        return {}, dict(remaining)

    by_key = {
        str(candidate.source_key): candidate
        for candidate in candidates
        if str(candidate.source_key or "").strip()
    }
    planned: dict[str, int] = {source_key: 0 for source_key in by_key}
    candidate_rows: list[tuple[str, int, dict[str, int]]] = []
    for source_key, candidate in by_key.items():
        raw_symbol_counts = dict(candidate.symbol_counts or {})
        symbol_counts = {
            symbol: max(0, int(raw_symbol_counts.get(symbol, 0) or 0))
            for symbol in remaining
        }
        symbol_counts = {symbol: count for symbol, count in symbol_counts.items() if count > 0}
        if not symbol_counts:
            continue
        existing = max(0, int(existing_augmented_by_source.get(source_key, 0) or 0))
        candidate_rows.append((source_key, existing, symbol_counts))

    while any(int(value or 0) > 0 for value in remaining.values()):
        best_key = ""
        best_score: tuple[int, int, str] | None = None
        best_gain = 0
        best_counts: dict[str, int] | None = None
        for source_key, existing, symbol_counts in candidate_rows:
            already_planned = max(0, int(planned.get(source_key, 0) or 0))
            if existing + already_planned >= max_per_source:
                continue
            gain = sum(
                min(
                    max(0, int(remaining.get(symbol, 0) or 0)),
                    int(count),
                )
                for symbol, count in symbol_counts.items()
            )
            if gain <= 0:
                continue
            # Progressive source reuse: first spread across real sources, then
            # only reuse the same source in later rounds when deficits remain.
            score = (existing + already_planned, -int(gain), source_key)
            if best_score is None or score < best_score:
                best_score = score
                best_key = source_key
                best_gain = int(gain)
                best_counts = symbol_counts
        if not best_key or best_gain <= 0:
            break
        planned[best_key] = int(planned.get(best_key, 0) or 0) + 1
        for symbol, count in dict(best_counts or {}).items():
            if symbol in remaining:
                remaining[symbol] = max(0, int(remaining.get(symbol, 0) or 0) - int(count or 0))

    planned = {key: int(value) for key, value in planned.items() if int(value or 0) > 0}
    remaining = {key: int(value) for key, value in remaining.items() if int(value or 0) > 0}
    return planned, remaining


def _synthetic_counts_from_planned_candidates(
    candidates: list[CharacterBalanceAugmentationCandidate] | tuple[CharacterBalanceAugmentationCandidate, ...],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for candidate in candidates or ():
        variants = max(0, int(getattr(candidate, "planned_variants", 0) or 0))
        if variants <= 0:
            continue
        for symbol, count in dict(getattr(candidate, "symbol_counts", {}) or {}).items():
            amount = max(0, int(count or 0)) * variants
            if amount <= 0:
                continue
            symbol_key = str(symbol or "").strip().upper()
            if not symbol_key:
                continue
            result[symbol_key] = int(result.get(symbol_key, 0) or 0) + amount
    return dict(sorted(result.items()))


def _resolve_image_for_label(root: Path, label_path: Path) -> Path | None:
    label_path = Path(label_path)
    root = Path(root)
    relative: Path | None = None
    try:
        relative = label_path.relative_to(root)
    except Exception:
        relative = None

    candidates: list[Path] = []
    if relative is not None:
        parts = list(relative.parts)
        if len(parts) >= 3 and parts[0] == "labels":
            candidates.append(root / "images" / parts[1] / label_path.name)
        if len(parts) >= 3 and parts[1] == "labels":
            candidates.append(root / parts[0] / "images" / label_path.name)
    parent_parts = list(label_path.parent.parts)
    if "labels" in parent_parts:
        index = parent_parts.index("labels")
        image_parent = Path(*parent_parts[:index], "images", *parent_parts[index + 1 :])
        candidates.append(image_parent / label_path.name)

    for candidate in candidates:
        for suffix in CHARACTER_BALANCE_IMAGE_EXTENSIONS:
            image_path = candidate.with_suffix(suffix)
            if image_path.exists():
                return image_path
    return None


def _is_original_source_label(label_path: Path, source_key: str, source_map: dict[str, str]) -> bool:
    stem = Path(label_path).stem
    if stem in source_map:
        return False
    return stem == str(source_key or stem)


def _collect_train_source_keys(root: Path) -> set[str]:
    source_map = _load_augmentation_source_map(root)
    split_dirs, _layout = _discover_label_dirs(root)
    keys: set[str] = set()
    for label_dir in split_dirs.get("train", []):
        for label_path in _iter_label_files(label_dir):
            source_key = _normalize_source_plate_key(label_path.stem, source_map)
            if source_key:
                keys.add(source_key)
    return keys


def _iter_metadata_files(search_roots: list[Path | str] | tuple[Path | str, ...]):
    seen: set[str] = set()
    for raw_root in search_roots or []:
        root = Path(raw_root)
        candidates: list[Path]
        if root.is_file():
            candidates = [root]
        elif root.exists() and root.is_dir():
            candidates = sorted(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}
            )
        else:
            candidates = []
        for path in candidates:
            key = _path_key(path)
            if key in seen:
                continue
            seen.add(key)
            yield path


def _load_metadata_payload(path: Path) -> Any:
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            return safe_load_yaml(Path(path))
        except Exception:
            return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig", errors="ignore"))
    except Exception:
        return {}


_REAL_SOURCE_TEXT_FIELDS = {
    "source_expected_text",
    "expected_text",
    "plate_text",
    "ground_truth",
    "ground_truth_text",
    "source_expected_texts",
    "expected_texts",
    "ground_truth_texts",
}
_REAL_SOURCE_KEY_FIELDS = {
    "source_key",
    "source_id",
    "plate_id",
    "crop_id",
    "image_id",
    "source_image",
    "image_path",
    "path",
    "filename",
    "file_name",
    "name",
}


def _iter_metadata_expected_texts(payload: Any, path: Path):
    stack: list[tuple[Any, str]] = [(payload, "")]
    while stack:
        value, field_name = stack.pop()
        if isinstance(value, dict):
            texts: list[tuple[str, str]] = []
            source_key = _metadata_source_key(value, path)
            for key, raw_value in value.items():
                key_text = str(key or "")
                if key_text in _REAL_SOURCE_TEXT_FIELDS:
                    texts.extend((token, key_text) for token in _expected_text_tokens(raw_value))
                elif isinstance(raw_value, (dict, list, tuple)):
                    stack.append((raw_value, key_text))
            for text, text_field in texts:
                if text:
                    yield source_key, text, text_field
            continue
        if isinstance(value, (list, tuple)):
            for item in reversed(value):
                stack.append((item, field_name))


def _metadata_source_key(payload: Mapping[str, Any], path: Path) -> str:
    for field_name in _REAL_SOURCE_KEY_FIELDS:
        raw_value = payload.get(field_name)
        if raw_value in (None, "", [], {}):
            continue
        if isinstance(raw_value, (list, tuple)):
            raw_value = raw_value[0] if raw_value else ""
        text = str(raw_value or "").strip()
        if text:
            return _normalize_source_plate_key(Path(text).stem, {})
    return _normalize_source_plate_key(Path(path).stem, {})


def _expected_text_tokens(value: Any) -> list[str]:
    values: list[Any]
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    tokens: list[str] = []
    for item in values:
        text = str(item or "").strip().upper()
        if not text:
            continue
        pieces = re.split(r"[_;,|/\\]+", text)
        if len(pieces) <= 1:
            pieces = [text]
        for piece in pieces:
            token = re.sub(r"[^0-9A-Z]+", "", piece.upper())
            if token and token not in tokens:
                tokens.append(token)
    return tokens


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="replace")).hexdigest()


def _file_artifact_ref(path: Path | str) -> dict[str, str]:
    safe_path = Path(path)
    ref = {"path": str(safe_path), "sha256": ""}
    try:
        if safe_path.exists() and safe_path.is_file():
            ref["sha256"] = _file_sha256(safe_path)
    except Exception:
        ref["sha256"] = ""
    return ref


def _compact_fingerprint_ref(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_root": str(value.get("dataset_root") or ""),
        "splits": list(value.get("splits") or []),
        "mode": str(value.get("mode") or "content_sha256"),
        "includes_config": bool(value.get("includes_config", False)),
        "file_count": int(value.get("file_count", 0) or 0),
        "sha256": str(value.get("sha256") or ""),
    }


def _fingerprint_entry(root: Path, path: Path, split_name: str) -> dict[str, Any]:
    try:
        relative = str(Path(path).relative_to(root)).replace("\\", "/")
    except Exception:
        relative = str(path)
    try:
        size = int(Path(path).stat().st_size or 0)
    except Exception:
        size = 0
    try:
        content_sha256 = _file_sha256(Path(path))
    except Exception:
        content_sha256 = ""
    return {
        "split": split_name,
        "relative_path": relative,
        "size": size,
        "sha256": content_sha256,
    }


def _distribution_ref(value: CharacterClassDistribution | Path | str) -> dict[str, str]:
    if isinstance(value, CharacterClassDistribution):
        payload = value.to_dict()
        return {
            "path": str(value.dataset_yaml or value.dataset_root or ""),
            "sha256": _sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
        }
    return _file_artifact_ref(Path(str(value or "")))


def _validate_character_class_map(yaml_path: Path | None, layout: str) -> CharacterClassMapValidation:
    expected = tuple(CHARACTER_BALANCE_ALPHABET)
    if yaml_path is None or not yaml_path.exists():
        if layout == "flat":
            return CharacterClassMapValidation(
                ok=True,
                status="WARNING",
                expected=expected,
                message="Brak mapy klas; interpretacja oparta na standardowym alfabecie MZ.",
                warnings=("Brak mapy klas; interpretacja oparta na standardowym alfabecie MZ.",),
            )
        validation = CharacterClassMapValidation(
            ok=False,
            status="ERROR",
            expected=expected,
            message=(
                "Brak data.yaml. Splitowany dataset MZ musi mieć jawną mapę klas, "
                "żeby symbolika 0-9/A-Z była jednoznaczna."
            ),
        )
        raise CharacterClassMapValidationError(validation)

    payload = safe_load_yaml(yaml_path)
    detected = _extract_yaml_class_names(payload)
    if not detected:
        detected = _extract_yaml_class_names_from_text(yaml_path)
    first_mismatch = _first_class_map_mismatch(expected, detected)
    if first_mismatch is None:
        return CharacterClassMapValidation(
            ok=True,
            status="OK",
            expected=expected,
            detected=detected,
            message="Mapa klas data.yaml jest zgodna ze standardem MZ.",
        )

    validation = CharacterClassMapValidation(
        ok=False,
        status="ERROR",
        expected=expected,
        detected=detected,
        first_mismatch_index=first_mismatch,
        message=_format_class_map_error(expected, detected, first_mismatch),
    )
    raise CharacterClassMapValidationError(validation)


def _extract_yaml_class_names(payload: dict[str, Any]) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return tuple()
    names = payload.get("names")
    if isinstance(names, dict):
        items: list[tuple[int, str]] = []
        for raw_key, raw_value in names.items():
            try:
                key = int(str(raw_key).strip())
            except Exception:
                return tuple(str(value).strip() for value in names.values())
            items.append((key, str(raw_value).strip()))
        return tuple(value for _key, value in sorted(items, key=lambda item: item[0]))
    if isinstance(names, (list, tuple)):
        return tuple(str(value).strip() for value in names)
    return tuple()


def _extract_yaml_class_names_from_text(yaml_path: Path) -> tuple[str, ...]:
    try:
        lines = yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return tuple()
    names_started = False
    list_values: list[str] = []
    dict_values: dict[int, str] = {}
    base_indent: int | None = None
    for raw_line in lines:
        line_without_comment = str(raw_line or "").split("#", 1)[0].rstrip()
        stripped = line_without_comment.strip()
        if not stripped:
            continue
        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        if not names_started:
            if re.match(r"^names\s*:\s*$", stripped):
                names_started = True
                base_indent = indent
                continue
            inline_match = re.match(r"^names\s*:\s*\[(.*)\]\s*$", stripped)
            if inline_match:
                return tuple(_clean_yaml_scalar(item) for item in inline_match.group(1).split(",") if item.strip())
            continue
        if base_indent is not None and indent <= base_indent:
            break
        if stripped.startswith("-"):
            value = stripped[1:].strip()
            list_values.append(_clean_yaml_scalar(value))
            continue
        mapping = re.match(r"^([0-9]+)\s*:\s*(.+?)\s*$", stripped)
        if mapping:
            dict_values[int(mapping.group(1))] = _clean_yaml_scalar(mapping.group(2))
    if dict_values:
        return tuple(value for _index, value in sorted(dict_values.items(), key=lambda item: item[0]))
    return tuple(list_values)


def _clean_yaml_scalar(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def _first_class_map_mismatch(expected: tuple[str, ...], detected: tuple[str, ...]) -> int | None:
    limit = max(len(expected), len(detected))
    for index in range(limit):
        expected_value = expected[index] if index < len(expected) else None
        detected_value = detected[index] if index < len(detected) else None
        if expected_value != detected_value:
            return index
    return None


def _format_class_map_error(
    expected: tuple[str, ...],
    detected: tuple[str, ...],
    first_mismatch_index: int,
) -> str:
    expected_value = expected[first_mismatch_index] if first_mismatch_index < len(expected) else "<brak>"
    detected_value = detected[first_mismatch_index] if first_mismatch_index < len(detected) else "<brak>"
    return (
        "Mapa klas datasetu nie jest zgodna ze standardem MZ.\n"
        f"Pierwsza niezgodność: indeks {first_mismatch_index}, "
        f"oczekiwano '{expected_value}', wykryto '{detected_value}'.\n"
        f"Oczekiwana mapa: {''.join(expected)}\n"
        f"Wykryta mapa: {' '.join(detected) if detected else '<brak names>'}"
    )


def _discover_label_dirs(root: Path) -> tuple[dict[str, list[Path]], str]:
    split_dirs: dict[str, list[Path]] = {split: [] for split in CHARACTER_BALANCE_SPLITS}
    split_dirs["unsplit"] = []

    for split_name in CHARACTER_BALANCE_SPLITS:
        for candidate in (root / "labels" / split_name, root / split_name / "labels"):
            if candidate.exists() and candidate.is_dir():
                _append_unique_path(split_dirs[split_name], candidate)

    if any(split_dirs[split] for split in CHARACTER_BALANCE_SPLITS):
        return split_dirs, "split"

    flat_labels = root / "labels"
    if flat_labels.exists() and flat_labels.is_dir():
        split_dirs["unsplit"].append(flat_labels)
        return split_dirs, "flat"

    return split_dirs, "empty"


def _append_unique_path(paths: list[Path], candidate: Path) -> None:
    candidate_key = _path_key(candidate)
    if all(_path_key(existing) != candidate_key for existing in paths):
        paths.append(candidate)


def _build_label_signature(split_dirs: dict[str, list[Path]], yaml_path: Path | None) -> str:
    parts: list[str] = []
    if yaml_path is not None:
        parts.append(_stat_token("yaml", yaml_path))
    for split_name in (*CHARACTER_BALANCE_SPLITS, "unsplit"):
        for label_dir in split_dirs.get(split_name, []):
            parts.append(_stat_token(split_name, label_dir))
            try:
                label_paths = [
                    path for path in label_dir.iterdir()
                    if path.is_file() and path.suffix.lower() == ".txt"
                ]
            except Exception:
                label_paths = []
            max_mtime = 0
            total_size = 0
            for path in label_paths:
                try:
                    stat = path.stat()
                except Exception:
                    continue
                max_mtime = max(max_mtime, int(stat.st_mtime_ns))
                total_size += int(stat.st_size or 0)
            parts.append(f"{_path_key(label_dir)}:{len(label_paths)}:{max_mtime}:{total_size}")
    return "|".join(parts)


def _stat_token(label: str, path: Path) -> str:
    try:
        stat = path.stat()
        return f"{label}:{_path_key(path)}:{int(stat.st_mtime_ns)}:{int(stat.st_size or 0)}"
    except Exception:
        return f"{label}:{_path_key(path)}:missing"


def _load_augmentation_source_map(root: Path) -> dict[str, str]:
    manifest_path = root / "augmentation_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    generated_files = payload.get("generated_files")
    if not isinstance(generated_files, list):
        return {}

    mapping: dict[str, str] = {}
    for item in generated_files:
        if not isinstance(item, dict):
            continue
        source = item.get("source_label") or item.get("source_image")
        generated = item.get("label") or item.get("image")
        if not source or not generated:
            continue
        source_stem = Path(str(source)).stem
        generated_stem = Path(str(generated)).stem
        if source_stem and generated_stem:
            mapping[generated_stem] = source_stem
    return mapping


def _normalize_source_plate_key(stem: str, source_map: dict[str, str]) -> str:
    current = str(stem or "").strip()
    seen: set[str] = set()
    while current in source_map and current not in seen:
        seen.add(current)
        current = str(source_map.get(current) or current).strip()
    while True:
        stripped = _EXPLICIT_AUG_SUFFIX_RE.sub("", current)
        if stripped == current:
            break
        current = stripped
    return current or str(stem or "")


def _parse_label_class_id(line: str, class_count: int) -> tuple[int | None, str]:
    text = str(line or "").strip()
    if not text or text.startswith("#"):
        return None, "skip"
    parts = text.split()
    if not parts:
        return None, "skip"
    try:
        raw_value = float(parts[0])
    except Exception:
        return None, "invalid_line"
    class_id = int(raw_value)
    if raw_value != float(class_id):
        return None, "invalid_line"
    if class_id < 0 or class_id >= class_count:
        return None, "invalid_class"
    return class_id, "ok"


def _median_nonzero(values: list[int]) -> float:
    nonzero = [int(value) for value in values if int(value or 0) > 0]
    return float(median(nonzero)) if nonzero else 0.0


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)
