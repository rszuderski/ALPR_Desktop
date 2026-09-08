#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runtime uruchamiania detekcji znaków Z3/PZ2."""

import copy
import json
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

import cv2

from ..campaign_manager import CAMPAIGN
from ..config import get_yolo_class, logger
from ..character_recognition import CharacterDetector, DetectionMethod
from ..ocr import PlateOCR
from ..utils import cleanup_gpu_memory

YOLO_DETECTION_METHODS = {
    DetectionMethod.YOLO,
    DetectionMethod.BOTH,
    DetectionMethod.YOLO_OCR,
    DetectionMethod.YOLO_BOX,
    DetectionMethod.YOLO_SYMBOL,
}
YOLO_GEOMETRY_METHODS = {
    DetectionMethod.YOLO,
    DetectionMethod.BOTH,
    DetectionMethod.YOLO_OCR,
    DetectionMethod.YOLO_BOX,
}
OCR_DETECTION_METHODS = {
    DetectionMethod.OCR,
    DetectionMethod.BOTH,
    DetectionMethod.YOLO_OCR,
}
YOLO_BOX_RECALL_CONFIDENCE = 0.00001


def _method_uses_yolo(method: DetectionMethod) -> bool:
    return method in YOLO_DETECTION_METHODS


def _method_uses_yolo_geometry(method: DetectionMethod) -> bool:
    return method in YOLO_GEOMETRY_METHODS


def get_detection_review_snapshot_path(host, preview_dir: str | Path | None = None) -> Path | None:
    try:
        raw_dir = str(preview_dir or host.preview_dir_var.get() or "").strip()
    except Exception:
        raw_dir = ""
    if not raw_dir:
        return None
    return Path(raw_dir) / ".last_detection_before_metadata.json"


def snapshot_detection_metadata_before_run(host, plate_ids, preview_dir: str | Path | None = None) -> bool:
    self = host
    path = get_detection_review_snapshot_path(self, preview_dir)
    if path is None:
        return False
    metadata = getattr(self, "preview_metadata", None)
    if not isinstance(metadata, dict):
        return False
    prepared_ids = [str(pid) for pid in list(plate_ids or []) if str(pid or "").strip()]
    snapshot_meta = {
        str(pid): copy.deepcopy(metadata.get(str(pid), {}))
        for pid in prepared_ids
        if isinstance(metadata.get(str(pid), {}), dict)
    }
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "preview_dir": str(Path(preview_dir or path.parent)),
        "plate_ids": prepared_ids,
        "metadata": snapshot_meta,
    }
    try:
        self._atomic_write_json(path, payload)
        return True
    except Exception as exc:
        logger.debug(f"Nie udalo sie zapisac snapshotu cofania detekcji PZ2: {exc}")
        return False


def load_detection_review_snapshot(host, preview_dir: str | Path | None = None) -> dict:
    path = get_detection_review_snapshot_path(host, preview_dir)
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.debug(f"Nie udalo sie odczytac snapshotu cofania detekcji PZ2: {exc}")
        return {}


def _remove_detection_review_snapshot(host, preview_dir: str | Path | None = None) -> None:
    path = get_detection_review_snapshot_path(host, preview_dir)
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
    except Exception as exc:
        logger.debug(f"Nie udalo sie usunac snapshotu cofania detekcji PZ2: {exc}")


def refresh_detection_review_controls(host) -> None:
    self = host
    snapshot_exists = bool(get_detection_review_snapshot_path(self) and get_detection_review_snapshot_path(self).exists())
    can_use = bool(snapshot_exists and not getattr(self, "fast_test_running", False) and not getattr(self, "is_processing", False))
    widget = getattr(self, "btn_undo_detection_result", None)
    if widget is not None:
        try:
            widget.config(state=(tk.NORMAL if can_use else tk.DISABLED))
        except Exception:
            pass


def undo_last_detection_result(host) -> None:
    self = host
    snapshot = load_detection_review_snapshot(self)
    snapshot_meta = snapshot.get("metadata") if isinstance(snapshot, dict) else None
    if not isinstance(snapshot_meta, dict) or not snapshot_meta:
        messagebox.showinfo("Brak wyniku do cofniecia", "Nie ma zapisanego snapshotu sprzed ostatniej detekcji.")
        refresh_detection_review_controls(self)
        return
    if not messagebox.askyesno(
        "Cofnac wynik detekcji?",
        "Przywroce stan ramek i znakow sprzed ostatniej detekcji dla aktualnego katalogu PZ2.",
    ):
        return
    current_meta = copy.deepcopy(getattr(self, "preview_metadata", {}) if isinstance(getattr(self, "preview_metadata", None), dict) else {})
    for pid, data in snapshot_meta.items():
        if isinstance(data, dict):
            current_meta[str(pid)] = copy.deepcopy(data)
    try:
        out_dir = Path(str(self.preview_dir_var.get() or "").strip())
        self._atomic_write_json(out_dir / "metadata.json", current_meta)
    except Exception as exc:
        logger.error(f"Nie udalo sie zapisac metadata po cofaniu detekcji PZ2: {exc}")
        messagebox.showerror("Nie zapisano cofniecia", str(exc))
        return
    try:
        self._apply_preview_metadata_update(current_meta, preserve_selection=True)
    except Exception:
        self.preview_metadata = current_meta
    try:
        self._set_test_status("Cofnieto wynik ostatniej detekcji", "warning")
    except Exception:
        pass
    _remove_detection_review_snapshot(self)
    refresh_detection_review_controls(self)


def confirm_last_detection_result(host) -> None:
    self = host
    snapshot = load_detection_review_snapshot(self)
    if not snapshot:
        messagebox.showinfo("Brak wyniku do zatwierdzenia", "Nie ma aktywnego wyniku detekcji oczekujacego na decyzje.")
        refresh_detection_review_controls(self)
        return
    _remove_detection_review_snapshot(self)
    try:
        self._set_test_status("Wynik detekcji zostal zatwierdzony", "success")
    except Exception:
        pass
    refresh_detection_review_controls(self)


def clear_detection_review_snapshot_after_manual_edit(host) -> None:
    self = host
    path = get_detection_review_snapshot_path(self)
    if path is None or not path.exists():
        return
    _remove_detection_review_snapshot(self)
    try:
        self._refresh_detection_review_controls()
    except Exception:
        pass


def _record_bbox(host, rec):
    return host._char_record_bbox(rec)


def _bbox_center(bbox):
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox[:4])
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    except Exception:
        return (0.0, 0.0)


def build_yolo_box_only_records(host, yolo_records) -> list[dict]:
    self = host
    prepared: list[dict] = []
    for rec in self._sort_character_records_by_x(list(yolo_records or [])):
        bbox = _record_bbox(self, rec)
        if not bbox:
            continue
        try:
            confidence = float(self._char_record_confidence(rec))
        except Exception:
            confidence = 0.0
        prepared.append(
            {
                "character": "",
                "bbox": [float(value) for value in bbox[:4]],
                "confidence": confidence,
                "method": "yolo_box",
                "source_tag": "yolo_box",
                "source_kind": "yolo_box",
                "box_source": "yolo_box",
                "sign_source": "",
                "geometry_source": "yolo",
                "geometry_method": "yolo_box",
                "box_backend": "yolo",
                "box_backend_source": "yolo_box_only",
                "box_backend_confidence": confidence,
            }
        )
    return self._sort_character_records_by_x(prepared)


def apply_yolo_symbols_to_existing_boxes(
    host,
    existing_chars,
    yolo_symbol_records,
    *,
    data=None,
    protect_manual: bool = True,
) -> tuple[list[dict], str, dict]:
    self = host
    boxes = self._sort_character_records_by_x(copy.deepcopy(list(existing_chars or [])), data=data)
    symbols = self._sort_character_records_by_x(list(yolo_symbol_records or []), data=data)
    if not boxes:
        return [], "yolo_symbol_no_boxes", {
            "source": "yolo_symbol_existing_boxes",
            "reason": "no_existing_boxes",
            "yolo_symbol_count": int(len(symbols)),
        }

    assigned: dict[int, object] = {}
    if len(boxes) == len(symbols):
        assigned = {idx: symbols[idx] for idx in range(len(boxes))}
    else:
        used_symbols: set[int] = set()
        for box_idx, box_rec in enumerate(boxes):
            box_bbox = _record_bbox(self, box_rec)
            if not box_bbox:
                continue
            box_center = _bbox_center(box_bbox)
            best_idx = None
            best_score = -1.0
            for symbol_idx, symbol_rec in enumerate(symbols):
                if symbol_idx in used_symbols:
                    continue
                symbol_bbox = _record_bbox(self, symbol_rec)
                if not symbol_bbox:
                    continue
                overlap = float(self._character_record_overlap_score(box_rec, symbol_rec))
                symbol_center = _bbox_center(symbol_bbox)
                dx = abs(box_center[0] - symbol_center[0])
                dy = abs(box_center[1] - symbol_center[1])
                bw = max(1.0, float(box_bbox[2]) - float(box_bbox[0]))
                bh = max(1.0, float(box_bbox[3]) - float(box_bbox[1]))
                distance_score = max(0.0, 1.0 - ((dx / bw) + (dy / bh * 0.5)))
                score = max(overlap * 3.0, distance_score)
                if score > best_score:
                    best_score = score
                    best_idx = symbol_idx
            if best_idx is not None and best_score >= 0.20:
                used_symbols.add(best_idx)
                assigned[box_idx] = symbols[best_idx]

    prepared: list[dict] = []
    updated_count = 0
    manual_preserved = 0
    missed_count = 0
    for idx, rec in enumerate(boxes):
        if not isinstance(rec, dict):
            continue
        updated = copy.deepcopy(rec)
        is_manual = bool(protect_manual and self._is_manual_character_record(updated, data=data))
        symbol_rec = assigned.get(idx)
        if is_manual:
            manual_preserved += 1
            prepared.append(updated)
            continue
        if symbol_rec is None:
            updated["character"] = ""
            updated["method"] = "yolo_symbol"
            updated.setdefault("box_source", self._get_character_box_source_tag(updated, data=data, fallback_index=idx))
            updated["sign_source"] = ""
            updated["source_tag"] = self._compose_character_source_tag(updated.get("box_source", ""), updated.get("sign_source", ""))
            updated["source_kind"] = str(updated.get("source_tag", "") or "ocr")
            updated["symbol_source"] = "yolo"
            updated["symbol_method"] = "yolo_symbol"
            missed_count += 1
            prepared.append(updated)
            continue
        symbol, _x = self._char_record_to_symbol_and_x(symbol_rec)
        symbol_value = self._sanitize_preview_char_symbol(symbol)
        updated["character"] = symbol_value
        try:
            updated["confidence"] = float(max(float(updated.get("confidence", 0.0) or 0.0), self._char_record_confidence(symbol_rec)))
        except Exception:
            pass
        updated["method"] = "yolo_symbol"
        updated.setdefault("box_source", self._get_character_box_source_tag(updated, data=data, fallback_index=idx))
        updated["sign_source"] = "yolo_symbol"
        updated["source_tag"] = self._compose_character_source_tag(updated.get("box_source", ""), updated.get("sign_source", ""))
        updated["source_kind"] = str(updated.get("source_tag", "") or "yolo_symbol")
        updated["symbol_source"] = "yolo"
        updated["symbol_method"] = "yolo_symbol"
        if symbol_value:
            updated_count += 1
        else:
            missed_count += 1
        prepared.append(updated)

    details = {
        "source": "yolo_symbol_existing_boxes",
        "existing_box_count": int(len(boxes)),
        "yolo_symbol_count": int(len(symbols)),
        "updated_symbol_count": int(updated_count),
        "missing_symbol_count": int(missed_count),
        "manual_preserved_count": int(manual_preserved),
    }
    return self._sort_character_records_by_x(prepared, data=data), "yolo_symbol_existing_boxes", details


def run_detection_stage(host):
    self = host
    method = self._get_detection_method_key()
    try:
        self.detection_method_var.set(method)
    except Exception:
        pass
    all_plate_ids = self._get_sorted_preview_plate_ids(list(getattr(self, "preview_metadata", {}).keys()))
    if not all_plate_ids:
        return messagebox.showinfo("Brak", "Wczytaj katalog wyodrębnionych tablic.")

    guard_options = self._prompt_pz2_detection_guard_options(method)
    if guard_options is None:
        return

    try:
        from .z3_campaign_flow import _mark_t06_z3_work_session

        _mark_t06_z3_work_session(
            self,
            state="active",
            substep=2,
            reason="run_detection",
        )
    except Exception:
        pass

    effective_device_choice = self._get_effective_detection_device_choice()
    effective_yolo_device = self._device_to_ultralytics(effective_device_choice)
    effective_ocr_device = self._device_to_ocr(effective_device_choice)

    if method in ("YOLO", "BOTH", "YOLO_OCR", "YOLO_BOX", "YOLO_SYMBOL"):
        try:
            yolo_runtime = self._get_yolo_runtime_settings()
        except Exception as e:
            messagebox.showwarning("Błędne parametry YOLO", str(e))
            return

        try:
            resolved_model = self._ensure_yolo_model_checkpoint()
            self.yolo_model_path_var.set(resolved_model)
            version, size = self._infer_yolo_arch_from_model_path(resolved_model)
            model_name = Path(resolved_model).name
            model_desc = f"{model_name} (YOLOv{version}{size})" if version and size else model_name

            self._log(
                self.test_log_text,
                f"[INFO] Model detekcji znaków: {model_desc}",
                "INFO"
            )
            self._log(
                self.test_log_text,
                f"[INFO] Model gotowy do użycia: {resolved_model}",
                "INFO"
            )
            self._log(
                self.test_log_text,
                "[INFO] Urzadzenie: "
                f"{effective_device_choice} | "
                f"YOLO={effective_yolo_device} | "
                f"OCR={effective_ocr_device}",
                "INFO"
            )
            self._log(
                self.test_log_text,
                "[INFO] Parametry YOLO: "
                f"infer_conf={yolo_runtime['conf']:.2f}, "
                f"yb_conf={yolo_runtime.get('box_conf', yolo_runtime['conf']):.2f}, "
                f"ys_conf={yolo_runtime.get('symbol_conf', yolo_runtime['conf']):.2f}, "
                f"nms_iou={yolo_runtime['iou']:.2f}, "
                f"overlap={yolo_runtime['overlap']:.2f}, "
                f"agnostic_nms={yolo_runtime['agnostic_nms']}, "
                f"seq_y={yolo_runtime['seq_center_y']:.2f}, "
                f"seq_h_min={yolo_runtime['seq_min_h']:.2f}, "
                f"seq_h_max={yolo_runtime['seq_max_h']:.2f}, "
                f"seq_w_max={yolo_runtime['seq_max_w']:.2f}, "
                f"seq_soft={yolo_runtime['seq_soft_overlap']:.2f}, "
                f"seq_hard={yolo_runtime['seq_hard_overlap']:.2f}",
                "INFO"
            )
        except Exception as e:
            messagebox.showwarning(
                "Błąd modelu YOLO",
                f"Nie udało się przygotować modelu YOLO:\n{e}"
            )
            self._log(self.test_log_text, f"[ERROR] {e}", "ERROR")
            return

    self._run_fast_ocr_test(guard_options=guard_options)


def unlock_ui_after_testing(host):
    self = host
    # NAJWAŻNIEJSZE: kończymy stan "processing"
    self.is_processing = False
    self.fast_test_running = False
    active_owner = str(getattr(self, "_active_test_operation_owner", "") or "").strip()
    self._active_test_operation_owner = None
    if active_owner and hasattr(self.app, "end_exclusive_operation"):
        self.app.end_exclusive_operation(active_owner)

    try:
        self.fast_test_stop.clear()
    except Exception:
        pass

    # lista tablic ma znowu działać zawsze po zakończeniu testu
    try:
        self.plates_listbox.config(state=tk.NORMAL)
    except Exception:
        pass

    # przyciski operacyjne
    for attr_name in ("btn_run_detection", "btn_rank_presets", "btn_ocr_lab"):
        widget = getattr(self, attr_name, None)
        if widget is not None:
            try:
                widget.config(state=tk.NORMAL)
            except Exception:
                pass

    # odśwież blokady/odblokowania pól ścieżek zgodnie z aktualnym trybem
    try:
        self._update_preview_path_lock()
    except Exception:
        pass

    try:
        self._update_step3_source_path_lock()
    except Exception:
        pass

    try:
        self._update_yolo_visibility()
    except Exception:
        pass

    # różne zachowanie dla kampanii i trybu swobodnego
    in_campaign = bool(getattr(self, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())

    if not in_campaign:
        # w trybie swobodnym wszystko ma działać
        for attr_name in ("btn_to_detect", "btn_to_dataset"):
            widget = getattr(self, attr_name, None)
            if widget is not None:
                try:
                    widget.config(state=tk.NORMAL)
                except Exception:
                    pass

        try:
            self._set_subtab_state(self.tab_extract, "normal")
            self._set_subtab_state(self.tab_detect, "normal")
            self._set_subtab_state(self.tab_dataset, "normal")
        except Exception:
            pass
    else:
        # w kampanii zostawiamy workflow tak, jak ustawiły go wcześniejsze kroki
        pass

    try:
        self.test_progress.update_idletasks()
    except Exception:
        pass

    try:
        self._refresh_detection_review_controls()
    except Exception:
        pass


def run_fast_ocr_test(host, guard_options: dict | None = None):
    self = host
    all_plate_ids = self._get_sorted_preview_plate_ids(list(getattr(self, "preview_metadata", {}).keys()))
    if not all_plate_ids:
        return messagebox.showinfo("Brak", "Wczytaj katalog wyodrębnionych tablic.")

    if self.fast_test_running:
        return

    guard_options = dict(guard_options or {})
    full_plate_total = len(all_plate_ids)
    process_scope = str(guard_options.get("process_scope", "all") or "all").strip().lower()
    if process_scope in {"perfect_only", "non_perfect_only"}:
        metadata = self.preview_metadata if isinstance(getattr(self, "preview_metadata", None), dict) else {}
        scoped_plate_ids: list[str] = []
        for pid in all_plate_ids:
            data = metadata.get(pid)
            chars = data.get("characters", []) if isinstance(data, dict) else []
            if not isinstance(chars, list):
                chars = []
            try:
                is_perfect = self._is_existing_plate_perfect(chars, data=data)
            except Exception:
                is_perfect = False
            if (process_scope == "perfect_only" and is_perfect) or (
                process_scope == "non_perfect_only" and not is_perfect
            ):
                scoped_plate_ids.append(pid)
        if not scoped_plate_ids:
            title = "Brak tablic perfect" if process_scope == "perfect_only" else "Brak tablic do korekty"
            message = (
                "W aktualnym katalogu PZ2 nie ma tablic ze statusem perfect do przetworzenia."
                if process_scope == "perfect_only"
                else "W aktualnym katalogu PZ2 wszystkie tablice mają status perfect."
            )
            return messagebox.showinfo(
                title,
                message,
            )
        all_plate_ids = scoped_plate_ids

    self._force_save_all()
    out_dir = Path(self.preview_dir_var.get().strip())
    imgs_dir = out_dir / "images"
    session_token = self._project_reset_token

    try:
        if hasattr(self, "preview_box_mode_var"):
            final_box_mode_label = (
                self._get_preview_box_mode_label("FINAL")
                if hasattr(self, "_get_preview_box_mode_label")
                else "FINAL"
            )
            self.preview_box_mode_var.set(final_box_mode_label)
            self._save_local_setting("char_preview_box_mode", "FINAL")
    except Exception:
        pass

    self.test_log_text.delete(1.0, tk.END)
    if not self._lock_ui_for_testing("pz2.fast_test.run", "PZ2: szybki test detekcji"):
        return
    detection_review_snapshot_created = snapshot_detection_metadata_before_run(self, all_plate_ids, out_dir)
    try:
        self._refresh_detection_review_controls()
    except Exception:
        pass
    self._set_test_status(self._compose_detection_method_status("start detekcji"), "info")
    self.test_progress.config(value=0)
    self._set_test_progress_counter(0, len(all_plate_ids), perfect_count=0)
    self._set_preview_processing_overlay(
        True,
        title="Trwa detekcja znaków",
        details="Ładuję modele OCR/YOLO i przygotowuję analizę tablic. Lista oraz podgląd są zablokowane do zakończenia procesu.",
        cancel_command=lambda: self.fast_test_stop.set(),
        cancel_visible=True,
        cancel_text="Anuluj",
    )
    self._update_preview_processing_overlay_progress(
        pct=0,
        current=0,
        total=len(all_plate_ids),
        meta_text="0% | przygotowanie detekcji",
    )
    try:
        self.preview_canvas_host.update_idletasks()
    except Exception:
        pass

    self.fast_test_stop.clear()
    self.fast_test_running = True

    self._log(self.test_log_text, "=======================================================", "HEADER")
    self._log(self.test_log_text, "START - Szybki Test Celności\n", "HEADER")

    method_key = self._get_detection_method_key()
    method_str = method_key.strip().lower()
    try:
        method = DetectionMethod(method_str)
    except Exception:
        method = DetectionMethod.OCR
        method_key = "OCR"
    detection_started_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        detection_pipeline_label = self._get_detection_pipeline_short_label(method_key)
    except Exception:
        detection_pipeline_label = self._get_detection_method_status_label()
    try:
        detection_method_label = self._get_detection_method_status_label()
    except Exception:
        detection_method_label = str(method_key or "OCR")

    protect_manual_requested = bool(guard_options.get("protect_manual_boxes", True))
    protect_manual_boxes = True
    protect_perfect_plates = bool(guard_options.get("protect_perfect_plates", True))
    use_perfect_box_refiner = bool(
        guard_options.get(
            "use_perfect_box_refiner",
            guard_options.get("allow_yolo_geometry_on_perfect", True),
        )
        and _method_uses_yolo_geometry(method)
    )
    use_perfect_refiner_continuity_guard = bool(
        use_perfect_box_refiner
        and guard_options.get("use_perfect_refiner_continuity_guard", True)
    )
    allow_yolo_geometry_on_perfect = use_perfect_box_refiner
    guard_counts = dict(guard_options.get("counts", {}) or {})
    scope_log_label = {
        "perfect_only": "tylko perfect",
        "non_perfect_only": "tylko do korekty",
    }.get(process_scope, "wszystkie tablice")
    self._log(
        self.test_log_text,
        (
            "[OCHRONA] "
            f"manual={'ON' if protect_manual_boxes else 'OFF'}"
            f"{' (wymuszone)' if not protect_manual_requested else ''}, "
            f"perfect={'ON' if protect_perfect_plates else 'OFF'}, "
            f"refiner perfect={'ON' if use_perfect_box_refiner else 'OFF'}, "
            f"ciągłość={'ON' if use_perfect_refiner_continuity_guard else 'OFF'}; "
            f"zakres={scope_log_label} ({len(all_plate_ids)}/{full_plate_total}); "
            f"perfect={int(guard_counts.get('perfect', 0) or 0)}, "
            f"manualne={int(guard_counts.get('manual', 0) or 0)}"
        ),
        "INFO",
    )

    effective_device_choice = self._get_effective_detection_device_choice()
    effective_yolo_device = self._device_to_ultralytics(effective_device_choice)
    effective_ocr_device = self._device_to_ocr(effective_device_choice)
    try:
        import torch
        cuda_state = (
            f"CUDA dostępne={bool(torch.cuda.is_available())}, "
            f"liczba GPU={int(torch.cuda.device_count())}"
        )
    except Exception as e:
        cuda_state = f"CUDA niedostępne do sprawdzenia ({e})"

    self._log(
        self.test_log_text,
        (
            "[INFO] Urządzenie detekcji: "
            f"global={effective_device_choice} | "
            f"YOLO={effective_yolo_device} | OCR={effective_ocr_device} | "
            f"{cuda_state}"
        ),
        "INFO",
    )

    yolo_model = None
    YoloClass = get_yolo_class() if _method_uses_yolo(method) else None
    if _method_uses_yolo(method) and YoloClass is not None:
        try:
            effective_model_path = self._get_effective_yolo_model_path()
            if not effective_model_path:
                raise RuntimeError("Brak aktywnej ścieżki modelu YOLO dla detekcji znaków.")
            yolo_model = YoloClass(str(effective_model_path))
        except Exception as e:
            self._log(self.test_log_text, f"Błąd YOLO: {e}", "ERROR")
            yolo_model = None

    prep_params = self._get_current_prep_params()
    ocr_engine = None
    if method in OCR_DETECTION_METHODS:
        try:
            ocr_engine = PlateOCR(
                device=effective_ocr_device,
                confidence_threshold=self.ocr_conf_var.get()
            )
            ocr_engine.custom_prep_params = prep_params
        except Exception as e:
            self._log(self.test_log_text, f"Błąd OCR Engine: {e}", "ERROR")
            ocr_engine = None

    try:
        yolo_runtime = self._get_yolo_runtime_settings()
    except Exception as e:
        self._log(self.test_log_text, f"Błąd parametrów YOLO: {e}", "ERROR")
        yolo_runtime = {
            "conf": 0.25,
            "box_conf": 0.25,
            "symbol_conf": 0.25,
            "iou": 0.45,
            "overlap": 0.70,
            "agnostic_nms": False,
            "seq_center_y": 0.60,
            "seq_min_h": 0.55,
            "seq_max_h": 1.80,
            "seq_max_w": 2.60,
            "seq_soft_overlap": 0.18,
            "seq_hard_overlap": 0.30,
        }

    detector = CharacterDetector(
        method=method,
        ocr_engine=ocr_engine,
        yolo_model=yolo_model,
        yolo_device=effective_yolo_device,
        yolo_confidence=yolo_runtime["conf"],
        yolo_box_confidence=yolo_runtime.get("box_conf", yolo_runtime["conf"]),
        yolo_symbol_confidence=yolo_runtime.get("symbol_conf", yolo_runtime["conf"]),
        yolo_iou=yolo_runtime["iou"],
        yolo_agnostic_nms=yolo_runtime["agnostic_nms"],
        yolo_overlap_threshold=yolo_runtime["overlap"],
        yolo_sequence_center_y_tolerance=yolo_runtime["seq_center_y"],
        yolo_sequence_min_height_ratio=yolo_runtime["seq_min_h"],
        yolo_sequence_max_height_ratio=yolo_runtime["seq_max_h"],
        yolo_sequence_max_width_ratio=yolo_runtime["seq_max_w"],
        yolo_sequence_soft_overlap=yolo_runtime["seq_soft_overlap"],
        yolo_sequence_hard_overlap=yolo_runtime["seq_hard_overlap"],
        ocr_min_height_ratio=self.ocr_min_height_ratio_var.get(),
    )

    def worker():
        local_meta = dict(self.preview_metadata) if isinstance(self.preview_metadata, dict) else {}
        total = len(all_plate_ids)
        stat_perfect = 0
        acc = 0.0
        processed_plate_ids: set[str] = set()
        detection_summary: dict = {}
        yolo_raw_total = 0
        yolo_nms_total = 0
        yolo_filtered_total = 0
        zero_backend_count = 0
        skipped_perfect_count = 0
        refined_perfect_count = 0
        perfect_refiner_box_total = 0
        perfect_refiner_failed_total = 0
        perfect_overwrite_count = 0
        manual_preserved_box_total = 0
        manual_skipped_auto_total = 0
        manual_overwrite_plate_count = 0
        yolo_runtime_device_logged = False
        try:
            detection_model_path = str(self.yolo_model_path_var.get() or "")
        except Exception:
            detection_model_path = ""

        def _mark_plate_detection(pid_value, *, result: str, characters: int = 0, status: str = "", extra: dict | None = None) -> None:
            pid_key = str(pid_value)
            if pid_key not in local_meta or not isinstance(local_meta.get(pid_key), dict):
                local_meta[pid_key] = {}
            payload = {
                "started_at": detection_started_iso,
                "method": str(method_key or "OCR"),
                "method_label": detection_method_label,
                "pipeline": detection_pipeline_label,
                "device": str(effective_device_choice or ""),
                "yolo_device": str(effective_yolo_device or ""),
                "ocr_device": str(effective_ocr_device or ""),
                "model": detection_model_path if _method_uses_yolo(method) else "",
                "characters": int(characters or 0),
                "status": str(status or ""),
                "result": str(result or ""),
            }
            if isinstance(extra, dict):
                payload.update(extra)
            local_meta[pid_key]["last_detection"] = payload
            processed_plate_ids.add(pid_key)

        try:
            for idx, pid in enumerate(all_plate_ids):
                if self.fast_test_stop.is_set() or session_token != self._project_reset_token:
                    break

                if pid not in local_meta or not isinstance(local_meta.get(pid), dict):
                    local_meta[pid] = {}

                manual_layout_state = self._capture_preview_manual_layout_state(local_meta.get(pid))
                existing_chars = list(local_meta[pid].get("characters", [])) if isinstance(local_meta[pid].get("characters"), list) else []
                existing_manual_chars = [
                    rec for rec in list(existing_chars or [])
                    if isinstance(rec, dict) and self._is_manual_character_record(rec, data=local_meta.get(pid))
                ]
                stale_auto_char_count = max(0, int(len(existing_chars)) - int(len(existing_manual_chars)))
                existing_is_perfect = self._is_existing_plate_perfect(
                    existing_chars,
                    data=local_meta.get(pid),
                )
                existing_has_manual = bool(existing_manual_chars)

                if (
                    existing_is_perfect
                    and protect_perfect_plates
                    and not allow_yolo_geometry_on_perfect
                ):
                    sorted_existing = self._sort_character_records_by_x(existing_chars, data=local_meta[pid])
                    sorted_existing = self._annotate_preview_character_reading_positions(
                        sorted_existing,
                        data=local_meta[pid],
                    )
                    local_meta[pid]["characters"] = self._serialize_character_records(
                        sorted_existing,
                        fusion_strategy=str(local_meta[pid].get("fusion_strategy", "") or ""),
                        fusion_details=local_meta[pid].get("fusion_details", {}) if isinstance(local_meta[pid].get("fusion_details"), dict) else None,
                        data=local_meta[pid],
                    )
                    local_meta[pid]["characters"] = self._annotate_preview_character_reading_positions(
                        local_meta[pid]["characters"],
                        data=local_meta[pid],
                    )
                    local_meta[pid]["status"] = "perfect"
                    self._update_preview_plate_layout_metadata(local_meta[pid], sorted_existing)
                    _mark_plate_detection(
                        pid,
                        result="pominięta przez ochronę perfect",
                        characters=len(local_meta[pid].get("characters", []) or []),
                        status="perfect",
                    )
                    skipped_perfect_count += 1
                    stat_perfect += 1
                    self._log(
                        self.test_log_text,
                        f"[OCHRONA] {pid}: pominięto detekcję, tablica ma status perfect.",
                        "INFO"
                    )
                    self.frame.after(
                        0,
                        lambda c=idx + 1, t=total, p=stat_perfect, token=session_token: self._update_detection_progress_ui(c, t, perfect_count=p, session_token=token)
                    )
                    continue

                img_path = imgs_dir / f"{pid}.jpg"
                if not img_path.exists():
                    _mark_plate_detection(
                        pid,
                        result="brak pliku obrazu",
                        characters=0,
                        status=str(local_meta[pid].get("status", "") or ""),
                    )
                    self.frame.after(
                        0,
                        lambda c=idx + 1, t=total, p=stat_perfect, token=session_token: self._update_detection_progress_ui(c, t, perfect_count=p, session_token=token)
                    )
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    _mark_plate_detection(
                        pid,
                        result="błąd odczytu obrazu",
                        characters=0,
                        status=str(local_meta[pid].get("status", "") or ""),
                    )
                    self.frame.after(
                        0,
                        lambda c=idx + 1, t=total, p=stat_perfect, token=session_token: self._update_detection_progress_ui(c, t, perfect_count=p, session_token=token)
                    )
                    continue

                source_image = local_meta[pid].get("source_image", "") or ""
                try:
                    true_texts = self._get_preview_expected_texts(local_meta[pid])
                except Exception:
                    true_texts = []
                if not true_texts:
                    true_texts = self._get_true_texts_from_filename(source_image)

                try:
                    expected_resolution = self._resolve_preview_expected_text_for_crop(local_meta[pid], [])
                except Exception:
                    expected_resolution = {}
                expected_char_count = int(expected_resolution.get("target_length", 0) or 0) if bool(
                    expected_resolution.get("count_resolved", False)
                ) else 0
                try:
                    detector.expected_character_count = int(expected_char_count)
                except Exception:
                    pass

                chars = detector.detect(img)
                if (
                    not yolo_runtime_device_logged
                    and _method_uses_yolo(method)
                ):
                    yolo_runtime_device_logged = True
                    runtime_device = str(getattr(detector, "last_yolo_runtime_device", "") or "brak wyniku YOLO")
                    requested_device = getattr(detector, "last_yolo_requested_device", effective_yolo_device)
                    self._log(
                        self.test_log_text,
                        (
                            "[INFO] Runtime YOLO: "
                            f"żądane device={requested_device}, "
                            f"tensor wynikowy={runtime_device}"
                        ),
                        "INFO",
                    )
                if session_token != self._project_reset_token:
                    break

                ocr_chars = self._sort_character_records_by_x(list(getattr(detector, "last_ocr_detections", [])))
                yolo_nms_chars = self._sort_character_records_by_x(list(getattr(detector, "last_yolo_nms_detections", [])))
                yolo_raw_chars = self._sort_character_records_by_x(list(getattr(detector, "last_yolo_raw_detections", [])))
                if method == DetectionMethod.YOLO_OCR:
                    yolo_chars = yolo_nms_chars
                else:
                    yolo_chars = self._sort_character_records_by_x(list(getattr(detector, "last_yolo_detections", [])))
                yolo_box_backend_chars = yolo_nms_chars or yolo_chars or yolo_raw_chars
                yolo_low_conf_recall_details = None
                if (
                    method == DetectionMethod.YOLO
                    and true_texts
                    and self._get_yolo_rescue_enabled()
                    and float(yolo_runtime.get("symbol_conf", yolo_runtime.get("conf", 0.25)) or 0.25) > 0.10
                    and hasattr(detector, "detect_yolo_candidate_pass")
                ):
                    try:
                        normal_text = self._characters_to_text(yolo_chars)
                        expected_text = self._pick_best_true_text(normal_text, true_texts, same_length_only=False)
                        expected_len = len(str(expected_text or "").strip())
                        normal_distance = self._best_text_distance(normal_text, true_texts)
                        normal_gap = abs(int(len(yolo_chars)) - int(expected_len)) if expected_len else 0
                        if expected_len > 0 and len(yolo_chars) < expected_len:
                            recall_conf = 0.10
                            recall_pass = detector.detect_yolo_candidate_pass(img, confidence=recall_conf)
                            recall_raw = self._sort_character_records_by_x(list(recall_pass.get("raw") or []))
                            recall_nms = self._sort_character_records_by_x(list(recall_pass.get("nms") or []))
                            recall_filtered = self._sort_character_records_by_x(list(recall_pass.get("filtered") or []))
                            recall_text = self._characters_to_text(recall_filtered)
                            recall_distance = self._best_text_distance(recall_text, true_texts)
                            recall_gap = abs(int(len(recall_filtered)) - int(expected_len))
                            improves_text = recall_distance < normal_distance
                            improves_count = recall_distance == normal_distance and recall_gap < normal_gap
                            if recall_filtered and (improves_text or improves_count):
                                detector.last_yolo_raw_detections = list(recall_raw)
                                detector.last_yolo_nms_detections = list(recall_nms)
                                detector.last_yolo_detections = list(recall_filtered)
                                detector.last_yolo_requested_device = recall_pass.get(
                                    "requested_device",
                                    getattr(detector, "last_yolo_requested_device", effective_yolo_device),
                                )
                                detector.last_yolo_runtime_device = str(
                                    recall_pass.get(
                                        "runtime_device",
                                        getattr(detector, "last_yolo_runtime_device", ""),
                                    )
                                    or ""
                                )
                                yolo_raw_chars = recall_raw
                                yolo_nms_chars = recall_nms
                                yolo_chars = recall_filtered
                                chars = list(recall_filtered)
                                yolo_box_backend_chars = yolo_nms_chars or yolo_chars or yolo_raw_chars
                                yolo_low_conf_recall_details = {
                                    "enabled": True,
                                    "conf": float(recall_conf),
                                    "normal_text": normal_text,
                                    "recall_text": recall_text,
                                    "expected_text": expected_text,
                                    "normal_count": int(len(normal_text)),
                                    "recall_count": int(len(recall_filtered)),
                                    "expected_count": int(expected_len),
                                    "normal_distance": int(normal_distance),
                                    "recall_distance": int(recall_distance),
                                }
                                self._log(
                                    self.test_log_text,
                                    (
                                        f"[YOLO RECALL] {pid}: ys_conf {float(yolo_runtime.get('symbol_conf', yolo_runtime.get('conf', 0.25)) or 0.25):.2f}"
                                        f" -> {recall_conf:.2f}, [{normal_text}] -> [{recall_text}]"
                                    ),
                                    "INFO",
                                )
                    except Exception as recall_error:
                        logger.debug(f"YOLO low-conf recall pominiety dla {pid}: {recall_error}")
                yolo_box_count_recall_details = None
                if (
                    method in (DetectionMethod.YOLO_BOX, DetectionMethod.YOLO)
                    and true_texts
                    and int(expected_char_count or 0) > int(len(yolo_box_backend_chars))
                    and hasattr(detector, "detect_yolo_candidate_pass")
                ):
                    try:
                        normal_count = int(len(yolo_box_backend_chars))
                        recall_pass = detector.detect_yolo_candidate_pass(
                            img,
                            confidence=YOLO_BOX_RECALL_CONFIDENCE,
                        )
                        recall_raw = self._sort_character_records_by_x(list(recall_pass.get("raw") or []))
                        recall_nms = self._sort_character_records_by_x(list(recall_pass.get("nms") or []))
                        recall_candidates = recall_nms or self._sort_character_records_by_x(
                            list(recall_pass.get("filtered") or [])
                        ) or recall_raw
                        selected_recall = recall_candidates
                        if hasattr(detector, "_select_sequence_best_count_candidates"):
                            selected_recall = detector._select_sequence_best_count_candidates(
                                list(recall_candidates),
                                int(expected_char_count),
                            )
                        selected_recall = self._sort_character_records_by_x(list(selected_recall or []))
                        if len(selected_recall) > normal_count:
                            detector.last_yolo_raw_detections = list(recall_raw)
                            detector.last_yolo_nms_detections = list(selected_recall)
                            if method == DetectionMethod.YOLO_BOX:
                                detector.last_yolo_detections = list(selected_recall)
                            detector.last_yolo_requested_device = recall_pass.get(
                                "requested_device",
                                getattr(detector, "last_yolo_requested_device", effective_yolo_device),
                            )
                            detector.last_yolo_runtime_device = str(
                                recall_pass.get(
                                    "runtime_device",
                                    getattr(detector, "last_yolo_runtime_device", ""),
                                )
                                or ""
                            )
                            yolo_raw_chars = recall_raw
                            yolo_nms_chars = selected_recall
                            if method == DetectionMethod.YOLO_BOX:
                                yolo_chars = selected_recall
                            yolo_box_backend_chars = selected_recall
                            yolo_box_count_recall_details = {
                                "enabled": True,
                                "conf": float(YOLO_BOX_RECALL_CONFIDENCE),
                                "normal_count": int(normal_count),
                                "recall_count": int(len(selected_recall)),
                                "expected_count": int(expected_char_count),
                                "raw_candidate_count": int(len(recall_raw)),
                                "nms_candidate_count": int(len(recall_nms)),
                            }
                            self._log(
                                self.test_log_text,
                                (
                                    f"[YB RECALL] {pid}: yb_conf "
                                    f"{float(yolo_runtime.get('box_conf', yolo_runtime.get('conf', 0.25)) or 0.25):.5f}"
                                    f" -> {YOLO_BOX_RECALL_CONFIDENCE:.5f}, "
                                    f"boxes {normal_count} -> {len(selected_recall)}"
                                ),
                                "INFO",
                            )
                    except Exception as recall_error:
                        logger.debug(f"YB low-conf recall pominiety dla {pid}: {recall_error}")
                if method == DetectionMethod.YOLO_BOX:
                    chars = build_yolo_box_only_records(self, yolo_box_backend_chars)
                    fusion_strategy = "yolo_box_only" if chars else "no_detection"
                    fusion_details = {
                        "source": "yolo_box_only",
                        "box_backend_count": int(len(chars)),
                        "box_backend_reference_count": int(len(yolo_box_backend_chars)),
                    }
                    if isinstance(yolo_box_count_recall_details, dict):
                        fusion_details["yolo_box_count_recall"] = yolo_box_count_recall_details
                elif method == DetectionMethod.YOLO:
                    yolo_box_records = build_yolo_box_only_records(self, yolo_box_backend_chars)
                    chars, fusion_strategy, fusion_details = apply_yolo_symbols_to_existing_boxes(
                        self,
                        yolo_box_records,
                        yolo_chars,
                        data=local_meta[pid],
                        protect_manual=False,
                    )
                    if chars:
                        fusion_strategy = "yolo_box_symbol"
                    fusion_details = dict(fusion_details or {})
                    fusion_details.update(
                        {
                            "source": "yolo_box_symbol",
                            "box_backend_count": int(len(yolo_box_records)),
                            "box_backend_reference_count": int(len(yolo_box_backend_chars)),
                            "symbol_reference_count": int(len(yolo_chars)),
                        }
                    )
                    if isinstance(yolo_box_count_recall_details, dict):
                        fusion_details["yolo_box_count_recall"] = yolo_box_count_recall_details
                elif method == DetectionMethod.YOLO_SYMBOL:
                    chars, fusion_strategy, fusion_details = apply_yolo_symbols_to_existing_boxes(
                        self,
                        existing_chars,
                        yolo_chars,
                        data=local_meta[pid],
                        protect_manual=protect_manual_boxes,
                    )
                else:
                    chars, fusion_strategy, fusion_details = self._resolve_canonical_detections(
                        method,
                        chars,
                        ocr_chars,
                        yolo_chars,
                        true_texts,
                        hybrid_rescue_max_chars=self._get_hybrid_rescue_max_chars(),
                        prefer_yolo_box_positions=(method == DetectionMethod.BOTH and self._use_hybrid_yolo_box_backend()),
                        yolo_box_backend_detections=yolo_box_backend_chars,
                        plate_image=img,
                    )
                if isinstance(yolo_low_conf_recall_details, dict):
                    fusion_details = dict(fusion_details or {})
                    fusion_details["yolo_low_conf_recall"] = yolo_low_conf_recall_details

                c_clean = self._serialize_character_records(
                    chars,
                    fusion_strategy=fusion_strategy,
                    fusion_details=fusion_details,
                    data=local_meta[pid],
                )
                yolo_clean = self._serialize_character_records(
                    getattr(detector, "last_yolo_detections", []),
                    data=local_meta[pid],
                )
                yolo_nms_clean = self._serialize_character_records(
                    getattr(detector, "last_yolo_nms_detections", []),
                    data=local_meta[pid],
                )
                yolo_raw_clean = self._serialize_character_records(
                    getattr(detector, "last_yolo_raw_detections", []),
                    data=local_meta[pid],
                )

                yolo_raw_total += len(getattr(detector, "last_yolo_raw_detections", []))
                yolo_nms_total += len(getattr(detector, "last_yolo_nms_detections", []))
                yolo_filtered_total += len(getattr(detector, "last_yolo_detections", []))

                if (
                    not ocr_chars
                    and not yolo_chars
                    and not list(getattr(detector, "last_yolo_raw_detections", []) or [])
                ):
                    zero_backend_count += 1

                preserve_perfect_existing = bool(existing_is_perfect and protect_perfect_plates)

                if preserve_perfect_existing:
                    preserved_chars, preserve_details = self._preserve_existing_perfect_plate_during_detection(
                        existing_chars,
                        yolo_box_backend_chars,
                        data=local_meta.get(pid),
                        plate_image=img,
                        enable_perfect_refiner=use_perfect_box_refiner,
                        perfect_refiner_continuity_guard=use_perfect_refiner_continuity_guard,
                    )
                    refiner_box_count = 0
                    refiner_failed_count = 0
                    if isinstance(preserve_details, dict):
                        refiner_box_count = int(preserve_details.get("perfect_refiner_count", 0) or 0)
                        refiner_failed_count = int(preserve_details.get("perfect_refiner_failed_count", 0) or 0)
                    perfect_refiner_box_total += int(refiner_box_count)
                    perfect_refiner_failed_total += int(refiner_failed_count)
                    if refiner_box_count > 0:
                        refined_perfect_count += 1
                    existing_fusion_details = local_meta[pid].get("fusion_details", {})
                    preserved_details = dict(existing_fusion_details) if isinstance(existing_fusion_details, dict) else {}
                    preserved_details["auto_strategy"] = str(fusion_strategy or "")
                    preserved_details["source"] = "pz2_detect_preserve_perfect"
                    preserved_details["locked_perfect_box_count"] = int(len(existing_chars))
                    ignored_extra_boxes = max(0, int(len(c_clean)) - int(len(existing_chars)))
                    if ignored_extra_boxes > 0:
                        preserved_details["auto_ignored_extra_boxes"] = int(ignored_extra_boxes)
                    if isinstance(preserve_details, dict) and preserve_details:
                        preserved_details.update(preserve_details)

                    merge_info = {
                        "manual_preserved_count": int(
                            preserve_details.get("manual_protected_count", 0) if isinstance(preserve_details, dict) else 0
                        ),
                        "auto_appended_count": 0,
                        "auto_skipped_due_manual": 0,
                        "perfect_preserved_count": int(len(existing_chars)),
                        "auto_ignored_extra_boxes": int(ignored_extra_boxes),
                    }
                    final_chars = preserved_chars
                    final_strategy = str(local_meta[pid].get("fusion_strategy", "") or fusion_strategy or "")
                    fusion_details = preserved_details
                else:
                    if existing_is_perfect and not protect_perfect_plates:
                        perfect_overwrite_count += 1

                    if protect_manual_boxes:
                        merged_chars, merge_info = self._merge_detected_characters_preserving_manual(
                            existing_manual_chars,
                            c_clean,
                            data=local_meta.get(pid),
                        )
                        if stale_auto_char_count > 0:
                            merge_info["stale_auto_removed_count"] = int(stale_auto_char_count)
                    else:
                        if existing_has_manual:
                            manual_overwrite_plate_count += 1
                        merged_chars = c_clean
                        merge_info = {
                            "manual_preserved_count": 0,
                            "auto_appended_count": int(len(c_clean)),
                            "auto_skipped_due_manual": 0,
                            "stale_auto_removed_count": int(stale_auto_char_count),
                        }

                    if int(merge_info.get("manual_preserved_count", 0) or 0) > 0:
                        manual_preserved_box_total += int(merge_info.get("manual_preserved_count", 0) or 0)
                        manual_skipped_auto_total += int(merge_info.get("auto_skipped_due_manual", 0) or 0)
                        fusion_details = dict(fusion_details or {})
                        fusion_details["auto_strategy"] = str(fusion_strategy or "")
                        fusion_details["manual_preserved_count"] = int(merge_info.get("manual_preserved_count", 0) or 0)
                        fusion_details["auto_appended_count"] = int(merge_info.get("auto_appended_count", 0) or 0)
                        fusion_details["auto_skipped_due_manual"] = int(merge_info.get("auto_skipped_due_manual", 0) or 0)
                        fusion_details["stale_auto_removed_count"] = int(merge_info.get("stale_auto_removed_count", 0) or 0)
                        fusion_details["source"] = "pz2_detect_merge"
                        final_chars = merged_chars
                        final_strategy = "manual_correction"
                    else:
                        final_chars = c_clean
                        final_strategy = str(fusion_strategy or "")

                # Ostateczny guard GT musi działać już po merge/preserve,
                # żeby stare manuale albo perfect-preserve nie przywracały
                # nadmiarowych boxów do finalnego metadata.
                final_chars, fusion_details = self._apply_final_truth_count_guard(
                    final_chars,
                    true_texts,
                    fusion_details if isinstance(fusion_details, dict) else None,
                    data=local_meta[pid],
                )

                # WAŻNE: znaki trafiają do metadata w kolejności czytania.

                self._update_preview_plate_layout_metadata(local_meta[pid], final_chars)
                final_chars = self._annotate_preview_character_reading_positions(final_chars, data=local_meta[pid])
                local_meta[pid]["characters"] = final_chars
                if manual_layout_state:
                    self._restore_preview_manual_layout_state(local_meta[pid], manual_layout_state)
                    final_chars = list(local_meta[pid].get("characters", []) or [])
                local_meta[pid]["yolo_detections"] = yolo_clean
                local_meta[pid]["yolo_nms_detections"] = yolo_nms_clean
                local_meta[pid]["yolo_raw_detections"] = yolo_raw_clean
                local_meta[pid]["fusion_strategy"] = str(final_strategy or "")
                if isinstance(fusion_details, dict) and fusion_details:
                    local_meta[pid]["fusion_details"] = fusion_details
                else:
                    local_meta[pid].pop("fusion_details", None)

                if fusion_strategy == "yolo_exact":
                    self._log(
                        self.test_log_text,
                        f"[HYBRID] {pid}: YOLO trafiło idealnie i przejęło finalne boxy.",
                        "INFO"
                    )
                elif fusion_strategy == "yolo_box_ocr":
                    final_text = self._characters_to_text(c_clean)
                    self._log(
                        self.test_log_text,
                        f"[HYBRID] {pid}: YOLO wyznaczyło boxy, a OCR odczytał cropy -> [{final_text}]",
                        "INFO"
                    )
                elif fusion_strategy == "ocr_yolo_rescue":
                    repaired_text = self._characters_to_text(c_clean)
                    ocr_text = ""
                    if isinstance(fusion_details, dict):
                        ocr_text = str(fusion_details.get("ocr_text", "") or "")
                    self._log(
                        self.test_log_text,
                        f"[HYBRID] {pid}: OCR=[{ocr_text}] -> naprawa YOLO -> [{repaired_text}]",
                        "INFO"
                    )

                if isinstance(fusion_details, dict) and int(fusion_details.get("box_backend_count", 0) or 0) > 0:
                    self._log(
                        self.test_log_text,
                        f"[HYBRID] {pid}: YOLO poprawiło pozycje {int(fusion_details.get('box_backend_count', 0))} boxów.",
                        "INFO"
                    )

                if isinstance(fusion_details, dict) and int(fusion_details.get("trimmed_extra_boxes", 0) or 0) > 0:
                    trim_stage = str(fusion_details.get("gt_count_guard_stage", "") or "").strip().lower()
                    stage_suffix = " po merge" if trim_stage == "final_characters" else ""
                    self._log(
                        self.test_log_text,
                        f"[GT] {pid}: przycięto nadmiarowe boxy do długości GT "
                        f"{stage_suffix}"
                        f"({int(fusion_details.get('original_box_count', 0) or 0)} -> "
                        f"{int(fusion_details.get('trimmed_box_count', 0) or 0)}).",
                        "INFO"
                    )

                if int(merge_info.get("manual_preserved_count", 0) or 0) > 0:
                    self._log(
                        self.test_log_text,
                        f"[MERGE] {pid}: zachowano ręczne boxy znaków={int(merge_info.get('manual_preserved_count', 0) or 0)}, "
                        f"dodano auto={int(merge_info.get('auto_appended_count', 0) or 0)}, "
                        f"pominięto auto przez kolizję z manual={int(merge_info.get('auto_skipped_due_manual', 0) or 0)}, "
                        f"usunięto stare auto={int(merge_info.get('stale_auto_removed_count', 0) or 0)}.",
                        "INFO"
                    )
                elif preserve_perfect_existing:
                    backend_count = 0
                    backend_ref_count = 0
                    if isinstance(fusion_details, dict):
                        backend_count = int(fusion_details.get("box_backend_count", 0) or 0)
                        backend_ref_count = int(fusion_details.get("box_backend_reference_count", 0) or 0)
                        manual_protected_count = int(fusion_details.get("manual_protected_count", 0) or 0)
                    else:
                        manual_protected_count = 0
                    preserve_message = (
                        f"[MERGE] {pid}: zachowano status perfect i liczbę boxów={int(len(existing_chars))}. "
                    )
                    if backend_count > 0:
                        refiner_count = int(fusion_details.get("perfect_refiner_count", 0) or 0) if isinstance(fusion_details, dict) else 0
                        if refiner_count > 0:
                            preserve_message += (
                                f"Refiner poprawił geometrię {refiner_count} boxów "
                                f"(kandydaci po NMS={backend_ref_count}). "
                            )
                        else:
                            preserve_message += (
                                f"YOLO dopasowało geometrię {backend_count} boxów "
                                f"(kandydaci po NMS={backend_ref_count}). "
                            )
                    else:
                        preserve_message += (
                            f"YOLO nie zmieniło geometrii perfecta "
                            f"(kandydaci po NMS={backend_ref_count}). "
                        )
                    if manual_protected_count > 0:
                        preserve_message += f"Manualne boxy nietknięte={manual_protected_count}. "
                    preserve_message += f"Zignorowano nowe boxy={int(merge_info.get('auto_ignored_extra_boxes', 0) or 0)}."
                    self._log(
                        self.test_log_text,
                        preserve_message,
                        "INFO"
                    )

                if final_chars:
                    txt = "".join(str(c.get("character", "")) for c in final_chars)
                    status_probe = dict(local_meta[pid])
                    status_probe["plate_id"] = str(pid)
                    derived_status = self._derive_preview_status_from_data(status_probe, final_chars)
                    local_meta[pid]["status"] = derived_status
                    if derived_status == "perfect":
                        stat_perfect += 1
                        self._log(self.test_log_text, f"✅ [{idx+1:03d}/{total}] {pid}: {txt}", "SUCCESS")
                    else:
                        expected_texts = self._get_preview_expected_texts(status_probe)
                        expected_str = " / ".join(expected_texts) if expected_texts else "Brak"
                        self._log(
                            self.test_log_text,
                            f"❌ [{idx+1:03d}/{total}] {pid}: Odczyt=[{txt}] (Oczek: [{expected_str}])",
                            "ERROR"
                        )
                else:
                    local_meta[pid]["status"] = "needs_fix"
                    if str(final_strategy or "").strip().lower() == "no_detection":
                        self._log(
                            self.test_log_text,
                            f"❌ [{idx+1:03d}/{total}] {pid}: BRAK KANDYDATÓW (OCR=0, YOLO=0)",
                            "ERROR"
                        )
                    else:
                        self._log(
                            self.test_log_text,
                            f"❌ [{idx+1:03d}/{total}] {pid}: NIC NIE ZNALEZIONO",
                            "ERROR"
                        )

                _mark_plate_detection(
                    pid,
                    result=str(local_meta[pid].get("status", "") or "needs_fix"),
                    characters=len(final_chars or []),
                    status=str(local_meta[pid].get("status", "") or ""),
                    extra={"fusion_strategy": str(final_strategy or "")},
                )

                self._ensure_plate_source_metadata(
                    local_meta[pid],
                    plate_id=str(pid),
                    meta_path=Path(out_dir) / "metadata.json",
                    default_bucket="auto_preview",
                    default_origin="pz2_detect",
                    modified_by="system",
                )

                self.frame.after(
                    0,
                    lambda c=idx + 1, t=total, p=stat_perfect, token=session_token: self._update_detection_progress_ui(c, t, perfect_count=p, session_token=token)
                )

            if session_token != self._project_reset_token:
                return

            finished_iso = datetime.now().astimezone().isoformat(timespec="seconds")
            for detected_pid in processed_plate_ids:
                plate_data = local_meta.get(str(detected_pid))
                if isinstance(plate_data, dict) and isinstance(plate_data.get("last_detection"), dict):
                    plate_data["last_detection"]["finished_at"] = finished_iso
            plates_with_chars = sum(
                1 for pid in all_plate_ids
                if isinstance(local_meta.get(pid), dict) and local_meta[pid].get("characters")
            )
            char_total = sum(
                len(local_meta[pid].get("characters") or [])
                for pid in all_plate_ids
                if isinstance(local_meta.get(pid), dict) and isinstance(local_meta[pid].get("characters"), list)
            )
            detection_summary = {
                "started_at": detection_started_iso,
                "finished_at": finished_iso,
                "method": str(method_key or "OCR"),
                "method_label": detection_method_label,
                "pipeline": detection_pipeline_label,
                "plate_scope": str(process_scope or "all"),
                "source_plates": int(full_plate_total),
                "scope_plates": int(total),
                "plates": int(total),
                "processed_plates": int(len(processed_plate_ids)),
                "plates_with_chars": int(plates_with_chars),
                "characters": int(char_total),
                "perfect": int(stat_perfect),
                "device": str(effective_device_choice or ""),
                "yolo_device": str(effective_yolo_device or ""),
                "ocr_device": str(effective_ocr_device or ""),
                "model": detection_model_path if _method_uses_yolo(method) else "",
                "yolo_raw": int(yolo_raw_total),
                "yolo_nms": int(yolo_nms_total),
                "yolo_filtered": int(yolo_filtered_total),
                "yolo_infer_conf": float(yolo_runtime.get("conf", 0.25) or 0.25),
                "yolo_box_conf": float(yolo_runtime.get("box_conf", yolo_runtime.get("conf", 0.25)) or 0.25),
                "yolo_symbol_conf": float(yolo_runtime.get("symbol_conf", yolo_runtime.get("conf", 0.25)) or 0.25),
                "review_snapshot": bool(detection_review_snapshot_created),
            }

            meta_file = out_dir / "metadata.json"
            self._atomic_write_json(meta_file, local_meta)
            self._save_last_detection_summary(detection_summary, out_dir)

            self._log(
                self.test_log_text,
                f"\n[DIAG] Tablice z wykrytymi znakami: {plates_with_chars}/{total}",
                "INFO"
            )
            self._log(
                self.test_log_text,
                f"[DIAG] Tablice bez jakichkolwiek kandydatów OCR/YOLO: {int(zero_backend_count)}/{total}",
                "INFO"
            )

            if _method_uses_yolo(method):
                self._log(
                    self.test_log_text,
                    f"[DIAG] YOLO boxy: raw={yolo_raw_total}, po NMS={yolo_nms_total}, po filtracji={yolo_filtered_total}",
                    "INFO"
                )

            strategy_counts = self._count_statuses_in_metadata_mapping(local_meta).get("strategy_counts", {})
            self._log(
                self.test_log_text,
                "[DIAG] Tablice perfect wg strategii: "
                f"OCR={int(strategy_counts.get('ocr_exact', 0))}, "
                f"YOLO={int(strategy_counts.get('yolo_exact', 0))}, "
                f"rescue={int(strategy_counts.get('ocr_yolo_rescue', 0))}, "
                f"yolo_box_ocr={int(strategy_counts.get('yolo_box_ocr', 0))}, "
                f"inne={int(strategy_counts.get('other_perfect', 0))}",
                "INFO"
            )
            try:
                flag_counts = {"MANUAL": 0, "O": 0, "YB": 0, "YS": 0}
                combo_counts = {}
                for plate_data in (local_meta or {}).values():
                    flags = list(self._get_plate_listbox_source_flags(plate_data))
                    if not flags:
                        continue
                    combo_key = "|".join(flags)
                    combo_counts[combo_key] = int(combo_counts.get(combo_key, 0) or 0) + 1
                    for flag in flags:
                        if flag in flag_counts:
                            flag_counts[flag] += 1
                combo_line = ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(combo_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
                ) or "brak"
                self._log(
                    self.test_log_text,
                    (
                        "[DIAG] Znaczniki listy po detekcji: "
                        f"MANUAL={flag_counts['MANUAL']}, O={flag_counts['O']}, "
                        f"YB={flag_counts['YB']}, YS={flag_counts['YS']} | "
                        f"kombinacje: {combo_line}"
                    ),
                    "INFO",
                )
                if (
                    _method_uses_yolo(method)
                    and (yolo_raw_total > 0 or yolo_nms_total > 0 or yolo_filtered_total > 0)
                    and int(flag_counts.get("YB", 0) or 0) <= 0
                    and int(flag_counts.get("YS", 0) or 0) <= 0
                ):
                    self._log(
                        self.test_log_text,
                        (
                            "[WARNING] YOLO zwróciło kandydatów, ale finalne etykiety listy "
                            "nie mają YB/YS. Sprawdź ochronę perfectów oraz wybrany układ pipeline."
                        ),
                        "WARNING",
                    )
            except Exception as flag_err:
                logger.debug(f"Nie udało się policzyć znaczników PZ2 po detekcji: {flag_err}")
            self._log(
                self.test_log_text,
                "[DIAG] Ochrona detekcji: "
                f"perfect pominięte={int(skipped_perfect_count)}, "
                f"perfect z refinerem={int(refined_perfect_count)}, "
                f"boxy poprawione refinerem={int(perfect_refiner_box_total)}, "
                f"próby odrzucone refinerem={int(perfect_refiner_failed_total)}, "
                f"perfect nadpisane={int(perfect_overwrite_count)}, "
                f"manualne boxy zachowane={int(manual_preserved_box_total)}, "
                f"auto pominięte przez manual={int(manual_skipped_auto_total)}, "
                f"tablice z manualem nadpisywane={int(manual_overwrite_plate_count)}",
                "INFO"
            )

            acc = (stat_perfect / total * 100) if total > 0 else 0
            self._log(
                self.test_log_text,
                f"\nSkuteczność: {acc:.1f}% ({stat_perfect}/{total} tablic)",
                "SUCCESS" if acc >= 80 else "WARNING"
            )

        except Exception as e:
            self._log(self.test_log_text, f"\n❌ BŁĄD: {e}", "ERROR")

        finally:
            try:
                if ocr_engine is not None:
                    try:
                        ocr_engine.unload()
                    except Exception:
                        pass
                detector.ocr_engine = None
            except Exception:
                pass
            try:
                detector.yolo_model = None
            except Exception:
                pass
            try:
                yolo_model = None
            except Exception:
                pass
            try:
                cleanup_gpu_memory()
            except Exception as cleanup_err:
                logger.debug(f"Nie udało się zwolnić GPU po PZ2: {cleanup_err}")

            def finalize():
                if session_token != self._project_reset_token:
                    return

                try:
                    self.fast_test_running = False
                    self.fast_test_stop.clear()

                    # Lista musi być aktywna przed przebudowa, inaczej repaint potrafi opoznic się
                    # do chwili kolejnej interakcji myszą lub klawiaturą.
                    try:
                        self.plates_listbox.config(state=tk.NORMAL)
                    except Exception:
                        pass

                    # Odśwież listę i preview na podstawie aktualnego metadata.
                    self._apply_preview_metadata_update(local_meta, preserve_selection=True)
                    self._loaded_meta_path = out_dir / "metadata.json"
                    try:
                        self._loaded_meta_mtime = self._loaded_meta_path.stat().st_mtime
                    except Exception:
                        self._loaded_meta_mtime = None

                    try:
                        method_name = self._get_detection_method_key()

                        if method_name == "OCR":
                            self._set_winner_name("Brak zwycięzcy turnieju", "neutral")
                            self._set_winner_acc("Uruchom turniej presetów OCR", "muted")
                        elif method_name in {"YOLO", "YOLO_BOX", "YOLO_SYMBOL"}:
                            self._set_winner_name("Brak rankingu OCR", "neutral")
                            self._set_winner_acc("Tryb YOLO nie bierze udziału w turnieju OCR", "muted")
                        elif method_name == "YOLO_OCR":
                            self._set_winner_name("Brak rankingu OCR", "neutral")
                            self._set_winner_acc("Tryb YOLO boxy + OCR nie ustala zwycięzcy turnieju OCR", "muted")
                        else:  # BOTH
                            self._set_winner_name("Brak rankingu OCR", "neutral")
                            self._set_winner_acc("Tryb hybrydowy nie ustala zwycięzcy turnieju OCR", "muted")
                    except Exception:
                        pass

                    try:
                        summary_msg = (
                            f"Podsumowanie detekcji: perfect={stat_perfect}/{total}, "
                            f"skuteczność={acc:.1f}%"
                        )
                        self._log(self.test_log_text, summary_msg, "INFO")
                    except Exception:
                        pass

                    self.test_progress.config(value=100)
                    self._set_test_progress_counter(total, total, perfect_count=stat_perfect)
                    self._update_preview_processing_overlay_progress(
                        pct=100,
                        current=total,
                        total=total,
                        meta_text=f"100% | OK {int(stat_perfect)}/{int(total)} tablic",
                    )
                    self._set_test_status(
                        self._compose_detection_method_status(
                            f"zakończona i zapisana | skuteczność {acc:.1f}%"
                        ),
                        "success"
                    )
                    try:
                        self._refresh_last_detection_status_label(detection_summary)
                    except Exception:
                        pass

                    try:
                        self.frame.update_idletasks()
                    except Exception:
                        pass

                    # 7. odblokuj dalszy krok
                    self.unlock_dataset_subtab()

                except Exception as e:
                    logger.error(f"Błąd finalize() po Szybkim Teście: {e}")
                    self._set_test_status(
                        self._compose_detection_method_status("błąd odświeżania UI"),
                        "error"
                    )

                finally:
                    self._set_preview_processing_overlay(False)
                    self._unlock_ui_after_testing()

            self.frame.after(0, finalize)

    threading.Thread(target=worker, daemon=True).start()

