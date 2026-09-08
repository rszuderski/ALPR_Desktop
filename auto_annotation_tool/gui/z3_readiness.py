#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Readiness and validation helpers for Z3/PZ3 flows."""

from pathlib import Path

from ..campaign_manager import CAMPAIGN
from ..config import CONFIG, logger
from ..validators import validate_yolo_dataset
from .z3_metadata_cache import readiness_key


def get_step3_yolo_export_readiness_snapshot(host, *, selected_strategies=None, selected_sources=None) -> dict:
    strategies = set(selected_strategies if selected_strategies is not None else host._get_selected_gold_export_strategy_buckets())
    sources = set(selected_sources if selected_sources is not None else host._get_selected_gold_export_source_buckets())
    try:
        key = readiness_key(host, strategies, sources, (
            bool(getattr(host, "_step3_linear_mode", False)), CAMPAIGN.get_active_project_name(),
            int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10)),
        ))
    except (OSError, ValueError, TypeError):
        key = None
    cached = getattr(host, "_preview_export_readiness_cache", None)
    if key is not None and isinstance(cached, tuple) and cached[0] == key:
        return dict(cached[1])
    result = _compute_step3_yolo_export_readiness(host, selected_strategies=strategies, selected_sources=sources)
    if key is not None:
        host._preview_export_readiness_cache = (key, dict(result))
    return result


def _compute_step3_yolo_export_readiness(host, *, selected_strategies=None, selected_sources=None) -> dict:
    self = host
    try:
        in_campaign = bool(getattr(self, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
    except Exception:
        in_campaign = False
    min_exportable_plate_count = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10) if in_campaign else 1
    selected_buckets = set(selected_strategies if selected_strategies is not None else self._get_selected_gold_export_strategy_buckets())
    selected_source_buckets = set(selected_sources if selected_sources is not None else self._get_selected_gold_export_source_buckets())
    result = {
        "ok": False,
        "reason": "",
        "message": "",
        "selected_plate_count": 0,
        "selected_char_count": 0,
        "candidate_count": 0,
        "min_exportable_plate_count": int(min_exportable_plate_count),
        "missing_exportable_plate_count": int(min_exportable_plate_count),
    }

    if not selected_buckets:
        result.update(
            reason="missing_strategy_filter",
            message="PZ3 nie ma wybranej strategii gold packa, więc eksport datasetu znaków nie jest możliwy.",
        )
        return result
    if not selected_source_buckets:
        result.update(
            reason="missing_source_filter",
            message="PZ3 nie ma wybranego źródła gold packa, więc eksport datasetu znaków nie jest możliwy.",
        )
        return result

    try:
        options = {"prepare_records": False} if isinstance(getattr(self, "_preview_metadata_revision", None), int) else {}
        plate_entries, _strategy_counts, _selected_strategy_counts, _source_counts, _selected_source_counts = (
            self._collect_gold_export_plate_candidates(selected_buckets, selected_source_buckets, **options)
        )
    except Exception as exc:
        result.update(
            reason="candidate_scan_failed",
            message=f"Nie udało się sprawdzić gotowości eksportu PZ3: {exc}",
        )
        return result

    exportable_plate_count = 0
    exportable_char_count = 0
    for plate_entry in list(plate_entries or []):
        data = plate_entry.get("data", {}) if isinstance(plate_entry, dict) else {}
        chars = list(data.get("characters", []) or []) if isinstance(data, dict) else []
        char_count = sum(1 for rec in chars if self._is_exportable_character_record(rec))
        if char_count <= 0:
            continue
        exportable_plate_count += 1
        exportable_char_count += int(char_count)

    missing_exportable = max(0, int(min_exportable_plate_count) - int(exportable_plate_count or 0))
    result.update(
        ok=bool(exportable_plate_count >= int(min_exportable_plate_count) and exportable_char_count > 0),
        selected_plate_count=int(exportable_plate_count),
        selected_char_count=int(exportable_char_count),
        candidate_count=int(len(plate_entries or [])),
        missing_exportable_plate_count=int(missing_exportable),
    )
    if not result["ok"]:
        if exportable_plate_count < int(min_exportable_plate_count):
            message = (
                (
                    f"Bramka E3 pozostaje zamknięta. Minimum kampanii to {min_exportable_plate_count} "
                    f"eksportowalnych tablic perfect, obecnie PZ3 widzi {exportable_plate_count}. "
                    f"Brakuje {missing_exportable}."
                )
                if in_campaign
                else (
                    f"PZ3 potrzebuje co najmniej {min_exportable_plate_count} eksportowalnej tablicy perfect, "
                    f"obecnie widzi {exportable_plate_count}."
                )
            )
        else:
            message = (
                "Bramka E3 pozostaje zamknięta, bo PZ3 nie ma jeszcze poprawnych boxów znaków i etykiet "
                "objętych aktualnym zakresem gold packa."
                if in_campaign
                else (
                    "PZ3 nie ma jeszcze poprawnych boxów znaków, etykiet i aktualnego zakresu gold packa."
                )
            )
        result.update(
            reason="no_exportable_yolo_candidates",
            message=message,
        )
    return result


def get_campaign_step3_annotation_readiness(host) -> dict:
    self = host
    result = {
        "ok": False,
        "reason": "missing_char_boxes",
        "message": (
            "Warunek eksportu PZ3 nie jest jeszcze spełniony. W kampanii potrzeba co najmniej 10 tablic "
            "ze statusem perfect, z poprawnymi boxami znaków i etykietami, objętych aktualnym zakresem gold packa."
        ),
        "exportable_plate_count": 0,
        "exportable_char_count": 0,
        "perfect_count": 0,
        "min_exportable_plate_count": 10,
        "missing_exportable_plate_count": 10,
    }

    try:
        selected_strategies = self._get_selected_gold_export_strategy_buckets()
        selected_sources = self._get_selected_gold_export_source_buckets()
        export_readiness = self._get_step3_yolo_export_readiness_snapshot(
            selected_strategies=selected_strategies,
            selected_sources=selected_sources,
        )
        exportable_chars = int(export_readiness.get("selected_char_count", 0) or 0)
        selected_plates = int(export_readiness.get("selected_plate_count", 0) or 0)
        min_exportable_plates = int(export_readiness.get("min_exportable_plate_count", 10) or 10)
        missing_exportable_plates = int(export_readiness.get("missing_exportable_plate_count", 0) or 0)
        export_message = str(export_readiness.get("message", "") or "")
    except Exception:
        exportable_chars = 0
        selected_plates = 0
        min_exportable_plates = 10
        missing_exportable_plates = 10
        export_message = ""

    try:
        perfect_count = int(self._count_preview_statuses().get("perfect", 0) or 0)
    except Exception:
        perfect_count = 0

    result.update(
        exportable_plate_count=int(selected_plates),
        exportable_char_count=int(exportable_chars),
        perfect_count=int(perfect_count),
        min_exportable_plate_count=int(min_exportable_plates),
        missing_exportable_plate_count=int(max(0, int(min_exportable_plates) - int(selected_plates or 0))),
    )
    if perfect_count < int(min_exportable_plates):
        missing_perfect = max(0, int(min_exportable_plates) - int(perfect_count or 0))
        result.update(
            reason="missing_perfect_plates",
            message=(
                f"Warunek wejścia dalej w E3 nie jest jeszcze spełniony. Minimum kampanii to {min_exportable_plates} "
                f"tablic perfect z poprawnymi boxami znaków i etykietami. Masz {perfect_count}, brakuje {missing_perfect}."
            ),
        )
        return result

    if selected_plates < int(min_exportable_plates) or exportable_chars <= 0:
        if perfect_count > 0 or selected_plates > 0:
            result["message"] = export_message or (
                "E3 ma tablice oznaczone jako perfect, ale PZ3 nie ma jeszcze materiału, "
                "który da się wyeksportować do datasetu YOLO. W PZ2 popraw lub dodaj ramki znaków, "
                "zapisz je i sprawdź filtry gold packa w PZ3."
            )
        return result

    result.update(
        ok=True,
        reason="",
        message="",
        exportable_plate_count=int(selected_plates),
        exportable_char_count=int(exportable_chars),
    )
    return result


def inspect_pz3_dataset_source_dir(host, dataset_dir: Path | None) -> dict:
    self = host
    result = {
        "ok": False,
        "path": "",
        "message": "",
        "total_pairs": 0,
        "flat_pairs": 0,
        "split_pairs": {"train": 0, "val": 0, "test": 0},
        "validation": None,
        "validation_message": "",
        "validation_stats": {},
    }

    if dataset_dir is None:
        result["message"] = "Wskaż gotowy dataset YOLO, aby wznowić split bez wracania do PZ2."
        return result

    try:
        dataset_dir = Path(dataset_dir)
    except Exception:
        result["message"] = "Ścieżka datasetu jest niepoprawna."
        return result

    result["path"] = str(dataset_dir)

    if not dataset_dir.exists():
        result["message"] = "Wskazany folder datasetu nie istnieje."
        return result

    if not dataset_dir.is_dir():
        result["message"] = "Wskazana ścieżka nie jest katalogiem."
        return result

    data_yaml = dataset_dir / "data.yaml"
    images_root = dataset_dir / "images"
    labels_root = dataset_dir / "labels"

    if not data_yaml.exists():
        result["message"] = "Brak pliku data.yaml w wskazanym folderze."
        return result

    if not images_root.exists() or not images_root.is_dir():
        result["message"] = "Brak katalogu images/ w wskazanym datasiecie."
        return result

    if not labels_root.exists() or not labels_root.is_dir():
        result["message"] = "Brak katalogu labels/ w wskazanym datasiecie."
        return result

    try:
        validation_ok, validation_msg, validation_stats = validate_yolo_dataset(dataset_dir)
    except Exception as exc:
        validation_ok, validation_msg, validation_stats = False, str(exc), {}

    result["validation"] = bool(validation_ok)
    result["validation_message"] = str(validation_msg or "")
    result["validation_stats"] = validation_stats if isinstance(validation_stats, dict) else {}

    total_pairs = 0
    split_pairs = {"train": 0, "val": 0, "test": 0}
    flat_pairs = 0

    for split_name in ("train", "val", "test", ""):
        img_dir = images_root / split_name if split_name else images_root
        lbl_dir = labels_root / split_name if split_name else labels_root
        if not img_dir.exists() or not img_dir.is_dir() or not lbl_dir.exists() or not lbl_dir.is_dir():
            continue

        pair_count = 0
        try:
            for img_path in img_dir.iterdir():
                if not img_path.is_file() or img_path.suffix.lower() not in CONFIG.IMAGE_EXTENSIONS:
                    continue
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if lbl_path.exists() and lbl_path.is_file():
                    pair_count += 1
        except Exception:
            continue

        if split_name:
            split_pairs[split_name] = pair_count
        else:
            flat_pairs = pair_count
        total_pairs += pair_count

    result["total_pairs"] = int(total_pairs)
    result["flat_pairs"] = int(flat_pairs)
    result["split_pairs"] = {key: int(value) for key, value in split_pairs.items()}

    if total_pairs <= 0:
        result["message"] = (
            "Nie znaleziono żadnej pary obraz + etykieta. "
            "Obsługiwane są układy images/labels oraz images/{train,val,test}."
        )
        return result

    flat_note = f"źródło bez splitu images/labels: {flat_pairs}" if flat_pairs > 0 else ""
    split_note = ""
    if any(int(split_pairs.get(name, 0) or 0) for name in ("train", "val", "test")):
        split_note = "gotowy split: " + ", ".join(
            f"{name}={split_pairs.get(name, 0)}" for name in ("train", "val", "test")
        )
    summary_parts = [part for part in (flat_note, split_note) if part]
    validation_note = "Walidacja YOLO pełna: OK." if validation_ok else f"Walidacja YOLO pełna: {validation_msg or 'pominięta'}."
    result["message"] = (
        f"Dataset gotowy do pracy: {total_pairs} par obraz + etykieta"
        + (f" ({'; '.join(summary_parts)})" if summary_parts else "")
        + f" {validation_note}"
    )
    result["ok"] = True
    return result


def get_campaign_step3_training_readiness(host) -> dict:
    self = host
    default_result = {
        "ok": False,
        "reason": "missing_char_dataset",
        "message": (
            "Brakuje eksportu datasetu znaków z PZ3. Najpierw w PZ2 doprowadź tablice do statusu perfect "
            "przez oznaczenie boxów znaków, a następnie w PZ3 wyeksportuj dataset znaków YOLO Detect."
        ),
        "ready_dataset": "",
        "dataset_hint": "",
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "exportable_plate_count": 0,
        "exportable_char_count": 0,
        "perfect_count": 0,
        "validation_message": "",
    }

    try:
        in_campaign = bool(getattr(self, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
        if not in_campaign:
            preferred = self._get_preferred_step3_training_dataset_dir()
            if preferred is not None:
                default_result.update(ok=True, reason="", ready_dataset=str(preferred))
            return default_result
    except Exception:
        return default_result

    annotation_readiness = self._get_campaign_step3_annotation_readiness()
    default_result.update(
        exportable_plate_count=int(annotation_readiness.get("exportable_plate_count", 0) or 0),
        exportable_char_count=int(annotation_readiness.get("exportable_char_count", 0) or 0),
        perfect_count=int(annotation_readiness.get("perfect_count", 0) or 0),
    )
    if not bool(annotation_readiness.get("ok")):
        default_result.update(
            reason=str(annotation_readiness.get("reason") or "missing_char_boxes"),
            message=str(annotation_readiness.get("message") or default_result["message"]),
        )

    try:
        summary = self._read_step3_export_summary()
        if not isinstance(summary, dict) or not summary:
            return default_result

        dataset_path_raw = str(summary.get("gold_dataset_path", "") or "").strip()
        dataset_path = Path(dataset_path_raw) if dataset_path_raw else None
        validation_message = str(summary.get("gold_dataset_validation_message", "") or "").strip()
        summary_char_count = int(summary.get("exportable_char_count", 0) or 0)
        summary_plate_count = int(summary.get("exportable_plate_count", 0) or 0)
        summary_perfect_count = int(summary.get("perfect_count", 0) or 0)
        label_object_count = 0
        if dataset_path is not None and dataset_path.exists():
            try:
                label_stats = self._inspect_yolo_dataset_label_objects(dataset_path)
                label_object_count = int(label_stats.get("objects", 0) or 0)
            except Exception:
                label_object_count = 0

        if dataset_path is not None and dataset_path.exists():
            default_result.update(
                exportable_plate_count=max(
                    int(default_result.get("exportable_plate_count", 0) or 0),
                    summary_plate_count,
                ),
                exportable_char_count=max(
                    int(default_result.get("exportable_char_count", 0) or 0),
                    summary_char_count,
                    label_object_count,
                ),
                perfect_count=max(
                    int(default_result.get("perfect_count", 0) or 0),
                    summary_perfect_count,
                ),
            )

        if bool(summary.get("gold_dataset_created")) and dataset_path is not None and dataset_path.exists():
            if not self._is_valid_step3_training_dataset_dir(dataset_path):
                merged = dict(default_result)
                merged.update(
                    ok=False,
                    reason="invalid_char_dataset",
                    ready_dataset=str(dataset_path),
                    dataset_hint=str(dataset_path),
                    validation_message=validation_message,
                    message=(
                        "Znaleziony dataset nie wygląda jak dataset znaków YOLO-Detect. "
                        "E3 nie może zostać zatwierdzone na podstawie datasetu tablic albo innego celu."
                    ),
                )
                return merged

            if max(int(default_result.get("perfect_count", 0) or 0), summary_perfect_count) <= 0:
                merged = dict(default_result)
                min_exportable_plates = int(default_result.get("min_exportable_plate_count", 10) or 10)
                merged.update(
                    ok=False,
                    reason="missing_perfect_plates",
                    ready_dataset=str(dataset_path),
                    dataset_hint=str(dataset_path),
                    validation_message=validation_message,
                    message=(
                        f"Dataset znaków istnieje, ale E3 nie ma wymaganego minimum {min_exportable_plates} tablic ze statusem perfect. "
                        "W PZ2 oznacz znaki na tablicach, doprowadź je do statusu perfect, "
                        "a następnie ponownie wykonaj eksport w PZ3."
                    ),
                )
                return merged

            if max(summary_char_count, label_object_count) <= 0:
                merged = dict(default_result)
                merged.update(
                    ok=False,
                    reason="missing_char_boxes",
                    ready_dataset=str(dataset_path),
                    dataset_hint=str(dataset_path),
                    validation_message=validation_message,
                    message=(
                        str(annotation_readiness.get("message") or "").strip()
                        or "Dataset znaków istnieje, ale nie zawiera żadnych etykiet z boxami znaków. Wróć do PZ2/PZ3 i wyeksportuj realne boxy znaków."
                    ),
                )
                return merged

            merged = dict(default_result)
            merged.update(
                ok=bool(summary.get("gold_dataset_valid", True)),
                reason=("" if bool(summary.get("gold_dataset_valid", True)) else "invalid_char_dataset"),
                ready_dataset=str(dataset_path),
                dataset_hint=str(dataset_path),
                exportable_plate_count=max(int(default_result.get("exportable_plate_count", 0) or 0), summary_plate_count),
                exportable_char_count=max(int(default_result.get("exportable_char_count", 0) or 0), summary_char_count, label_object_count),
                perfect_count=max(int(default_result.get("perfect_count", 0) or 0), summary_perfect_count),
                validation_message=validation_message,
                message=(
                    ""
                    if bool(summary.get("gold_dataset_valid", True))
                    else (validation_message or default_result["message"])
                ),
            )
            return merged

        if summary.get("gold_dataset_created") is False or summary.get("gold_dataset_valid") is False:
            merged = dict(default_result)
            merged.update(
                reason="invalid_char_dataset",
                validation_message=validation_message,
                message=(validation_message or default_result["message"]),
            )
            return merged
    except Exception as e:
        logger.debug(f"Nie udało się pobrac walidacji datasetu znaków dla kroku 3: {e}")

    return default_result
