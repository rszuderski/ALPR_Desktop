#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OCR filter laboratory UI extracted from CharacterAnnotationTab.

The function intentionally keeps the original method body shape and binds
``self`` to the host tab instance. This makes the extraction mechanical and
keeps widget/state interactions in one place while reducing the main tab file.
"""

import json
import random
import re
import shutil
import tkinter as tk
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - mirrors optional PIL handling in the tab.
    Image = None
    ImageTk = None

from .help_manager import HELP
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar
from ..campaign_manager import CAMPAIGN
from ..character_recognition import CharacterDetector, DetectionMethod
from ..config import CONFIG
from ..ocr import PlateOCR


def get_true_texts_from_filename(filename: str) -> list:
    stem = Path(str(filename or "")).stem.upper()
    if not stem:
        return []
    ignore_tokens = {
        "PLATE",
        "PLATES",
        "TABLICA",
        "TABLICE",
        "IMG",
        "IMAGE",
        "PHOTO",
        "RAW",
        "SOURCE",
        "RUN",
        "SAMPLE",
        "SAMPLES",
        "FRAME",
        "CAPTURE",
        "PREVIEW",
        "FILE",
        "PLIK",
        "CROP",
    }
    parts = [
        str(raw_part or "").strip().upper()
        for raw_part in re.findall(r"[A-Z0-9]+", stem)
        if str(raw_part or "").strip()
    ]
    filtered_parts = [part for part in parts if part not in ignore_tokens]
    if len(filtered_parts) > 1 and filtered_parts[-1].isdigit():
        previous_plate_like = any(
            3 <= len(part) <= 12
            and (any(ch.isalpha() for ch in part) or part.isdigit())
            for part in filtered_parts[:-1]
        )
        if previous_plate_like:
            filtered_parts = filtered_parts[:-1]

    candidates = []
    seen = set()
    for part in filtered_parts:
        if not part or part in seen:
            continue
        if len(part) < 3 or len(part) > 12:
            continue
        has_letter = any(ch.isalpha() for ch in part)
        has_digit = any(ch.isdigit() for ch in part)
        if not (has_letter or has_digit):
            continue
        seen.add(part)
        candidates.append(part)

    return candidates


def get_current_prep_params(host) -> dict:
    return {
        "target_height": host.prep_height_var.get(),
        "manual_angle": host.prep_angle_var.get(),
        "clip_thresh": host.prep_clip_var.get(),
        "denoise_h": host.prep_denoise_var.get(),
        "clahe_clip": host.prep_clahe_var.get() if host.do_clahe_var.get() else 0.0,
        "use_binarization": host.prep_use_bin_var.get(),
        "thresh_block": host.prep_block_var.get(),
        "thresh_c": host.prep_c_var.get(),
        "erode_iter": host.prep_erode_var.get(),
        "interpolation": host.interpolation_var.get(),
        "padding_pct": host.prep_padding_var.get(),
    }


def get_best_preset(host):
    cache_file = host.presets_dir / "global_ranking.json"
    if not cache_file.exists():
        return None, 0.0
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        best_name, best_acc, best_params = None, -1.0, {}
        for preset_key, stats in data.items():
            acc = float(stats.get("acc", 0.0))
            if acc > best_acc:
                best_acc, best_name, best_params = acc, str(stats.get("name", preset_key)), stats.get("params", {})
        if best_name:
            return {"name": best_name, "params": best_params}, best_acc
    except Exception:
        pass
    return None, 0.0


def build_live_ocr_sample_pool(host) -> list[dict]:
    preview_dir_raw = str(host.preview_dir_var.get() or "").strip()
    if not preview_dir_raw or not host.preview_plate_ids:
        return []

    images_dir = Path(preview_dir_raw) / "images"
    if not images_dir.exists() or not images_dir.is_dir():
        return []

    samples = []
    for pid in host.preview_plate_ids:
        pid_str = str(pid or "").strip()
        if not pid_str:
            continue
        img_path = host._find_preview_plate_image_path(images_dir, pid_str)
        if img_path is None:
            continue
        data = host.preview_metadata.get(pid_str, {}) if isinstance(host.preview_metadata, dict) else {}
        expected_texts = host._get_preview_expected_texts(data)
        label = f"ID: {pid_str}"
        if expected_texts:
            label = f"{label}  |  Oczek: {expected_texts[0]}"
        samples.append({
            "sample_id": pid_str,
            "image_path": str(img_path),
            "expected_texts": expected_texts,
            "label": label,
            "source": "preview",
        })
    return samples


def get_ocr_demo_dir(host) -> Path:
    demo_dir = host.session_file.parent / "ocr_demo_samples"
    demo_dir.mkdir(parents=True, exist_ok=True)
    return demo_dir


def load_cached_ocr_demo_samples(host, limit: int = 3) -> list[dict]:
    demo_dir = host._get_ocr_demo_dir()
    manifest_path = demo_dir / "manifest.json"
    manifest = host._load_json_file_safely(manifest_path)
    if not isinstance(manifest, dict):
        return []

    samples = []
    for idx, item in enumerate(manifest.get("samples") or [], start=1):
        if not isinstance(item, dict):
            continue
        rel_image = str(item.get("image", "") or "").strip()
        if not rel_image:
            continue
        img_path = demo_dir / rel_image
        if not img_path.exists() or not img_path.is_file():
            continue
        expected_texts = [
            str(text or "").strip().upper()
            for text in (item.get("expected_texts") or [])
            if str(text or "").strip()
        ]
        label = str(item.get("label", "") or "").strip()
        if not label:
            label = f"Demo {idx}"
            if expected_texts:
                label = f"{label}  |  Oczek: {expected_texts[0]}"
        samples.append({
            "sample_id": str(item.get("sample_id", f"demo_{idx:03d}") or f"demo_{idx:03d}"),
            "image_path": str(img_path),
            "expected_texts": expected_texts,
            "label": label,
            "source": "demo",
        })
    return samples[:limit]


def iter_ocr_demo_source_run_dirs(host) -> list[Path]:
    roots = []
    preview_dir_raw = str(host.preview_dir_var.get() or "").strip()
    if preview_dir_raw:
        roots.append(Path(preview_dir_raw))

    campaign_chars_dir = getattr(host, "_campaign_chars_dir", None)
    if campaign_chars_dir:
        roots.append(Path(campaign_chars_dir))

    roots.append(Path(CONFIG.DIR_3_CHARS))

    seen = set()
    run_dirs = []

    def add_run_dir(candidate: Path):
        try:
            key = candidate.resolve()
        except Exception:
            key = candidate
        if key in seen:
            return
        meta_path = candidate / "metadata.json"
        images_dir = candidate / "images"
        if meta_path.exists() and images_dir.exists() and images_dir.is_dir():
            seen.add(key)
            run_dirs.append(candidate)

    for root in roots:
        if root is None or not root.exists() or not root.is_dir():
            continue
        add_run_dir(root)
        try:
            for meta_path in root.rglob("metadata.json"):
                add_run_dir(meta_path.parent)
        except Exception:
            continue

    run_dirs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return run_dirs


def seed_ocr_demo_samples(host, limit: int = 3) -> list[dict]:
    selected = []
    selected_keys = set()

    for run_dir in host._iter_ocr_demo_source_run_dirs():
        metadata = host._load_json_file_safely(run_dir / "metadata.json")
        if not isinstance(metadata, dict):
            continue

        perfect_samples = []
        fallback_samples = []
        images_dir = run_dir / "images"

        for pid, data in metadata.items():
            if not isinstance(data, dict):
                continue
            img_path = host._find_preview_plate_image_path(images_dir, str(pid or "").strip())
            if img_path is None:
                continue
            expected_texts = host._get_preview_expected_texts(data)
            if not expected_texts:
                continue
            sample_key = tuple(expected_texts)
            if sample_key in selected_keys:
                continue
            label = f"Demo {len(selected) + len(perfect_samples) + len(fallback_samples) + 1}"
            label = f"{label}  |  Oczek: {expected_texts[0]}"
            sample = {
                "sample_id": str(pid or "").strip() or f"demo_{len(selected) + 1:03d}",
                "image_path": str(img_path),
                "expected_texts": expected_texts,
                "label": label,
                "source": "demo",
            }
            status = str(data.get("status", "") or "").strip().lower()
            if status == "perfect":
                perfect_samples.append(sample)
            else:
                fallback_samples.append(sample)

        for sample in perfect_samples + fallback_samples:
            sample_key = tuple(sample.get("expected_texts") or [])
            if sample_key in selected_keys:
                continue
            selected_keys.add(sample_key)
            selected.append(sample)
            if len(selected) >= limit:
                break

        if len(selected) >= limit:
            break

    if not selected:
        return []

    demo_dir = host._get_ocr_demo_dir()
    manifest_samples = []
    for index, sample in enumerate(selected[:limit], start=1):
        src_path = Path(sample["image_path"])
        ext = src_path.suffix if src_path.suffix else ".jpg"
        dst_name = f"sample_{index:03d}{ext.lower()}"
        dst_path = demo_dir / dst_name
        try:
            shutil.copy2(src_path, dst_path)
        except Exception:
            continue
        manifest_samples.append({
            "sample_id": str(sample.get("sample_id") or f"demo_{index:03d}"),
            "image": dst_name,
            "expected_texts": list(sample.get("expected_texts") or []),
            "label": str(sample.get("label") or f"Demo {index}"),
        })

    if not manifest_samples:
        return []

    host._atomic_write_json(
        demo_dir / "manifest.json",
        {"samples": manifest_samples},
    )
    return host._load_cached_ocr_demo_samples(limit=limit)


def get_ocr_demo_samples(host, limit: int = 3) -> list[dict]:
    cached = host._load_cached_ocr_demo_samples(limit=limit)
    if len(cached) >= limit:
        return cached[:limit]
    seeded = host._seed_ocr_demo_samples(limit=limit)
    return seeded[:limit] if seeded else cached[:limit]


def get_ocr_lab_sample_bundle(host) -> dict:
    live_samples = host._build_live_ocr_sample_pool()
    if live_samples:
        preview_dir = Path(str(host.preview_dir_var.get() or "").strip())
        meta_path = preview_dir / "metadata.json"
        try:
            preview_key = str(preview_dir.resolve())
        except Exception:
            preview_key = str(preview_dir)
        meta_mtime = meta_path.stat().st_mtime_ns if meta_path.exists() else 0
        return {
            "mode": "preview",
            "samples": live_samples,
            "allow_random": len(live_samples) > 3,
            "dataset_signature": f"preview:{preview_key}:{meta_mtime}:{len(live_samples)}",
            "description": "losowanie z wczytanej listy tablic",
        }

    demo_samples = host._get_ocr_demo_samples(limit=3)
    if demo_samples:
        manifest_path = host._get_ocr_demo_dir() / "manifest.json"
        manifest_mtime = manifest_path.stat().st_mtime_ns if manifest_path.exists() else 0
        return {
            "mode": "demo",
            "samples": demo_samples,
            "allow_random": False,
            "dataset_signature": f"demo:{manifest_mtime}:{len(demo_samples)}",
            "description": "3 próbki demo z danych aplikacji",
        }

    return {
        "mode": "empty",
        "samples": [],
        "allow_random": False,
        "dataset_signature": "",
        "description": "",
        "message": (
            "Brak wczytanej listy tablic i brak próbek demo OCR. "
            "Wykonaj przynajmniej jeden run wycinania PZ1 albo wczytaj katalog wyodrębnionych tablic."
        ),
    }


def get_ocr_ranking_sample_bundle(host) -> dict:
    bundle = host._get_ocr_lab_sample_bundle()
    if bundle.get("mode") == "empty":
        return bundle
    if bundle.get("mode") == "demo":
        return {
            "mode": "empty",
            "samples": [],
            "allow_random": False,
            "dataset_signature": "",
            "description": "",
            "message": (
                "Ranking OCR wymaga wczytanego katalogu wyodrębnionych tablic. "
                "Próbki demo są dostępne tylko w Laboratorium OCR."
            ),
        }

    ranked_samples = [
        sample
        for sample in (bundle.get("samples") or [])
        if sample.get("expected_texts")
    ]
    if not ranked_samples:
        return {
            "mode": "empty",
            "samples": [],
            "allow_random": False,
            "dataset_signature": "",
            "description": "",
            "message": "Brak próbek z oczekiwanym tekstem do rankingu OCR.",
        }

    return {
        "mode": bundle.get("mode"),
        "samples": ranked_samples,
        "allow_random": bundle.get("allow_random", False),
        "dataset_signature": bundle.get("dataset_signature", ""),
        "description": bundle.get("description", ""),
        "cache_file": host.presets_dir / "global_ranking.json",
    }


def run_preset_ranking(host) -> None:
    self = host
    sample_bundle = self._get_ocr_ranking_sample_bundle()
    if sample_bundle.get("mode") != "preview":
        self._set_ocr_ranking_modal_status(
            str(sample_bundle.get("message") or "Ranking OCR wymaga wczytanego katalogu wyodrębnionych tablic."),
            tone="warning",
        )
        messagebox.showinfo(
            "Ranking OCR",
            sample_bundle.get("message") or "Ranking OCR wymaga wczytanego katalogu wyodrębnionych tablic.",
        )
        return

    sample_records = list(sample_bundle.get("samples") or [])
    if not sample_records:
        self._set_ocr_ranking_modal_status(
            str(sample_bundle.get("message") or "Brak próbek do rankingu OCR."),
            tone="warning",
        )
        messagebox.showinfo("Brak", sample_bundle.get("message") or "Brak próbek do rankingu OCR.")
        return

    preset_files = list(self.presets_dir.glob("*.json"))
    preset_files = [f for f in preset_files if f.name != "global_ranking.json"]
    if not preset_files:
        self._set_ocr_ranking_modal_status(
            "Brak presetów OCR. Otwórz Laboratorium OCR i zapisz co najmniej jeden preset.",
            tone="warning",
        )
        messagebox.showinfo("Brak presetów", "Brak presetów! Otwórz Laboratorium OCR i zapisz co najmniej jeden preset.")
        return

    ranking_mode = str(sample_bundle.get("mode") or "preview")
    demo_mode = ranking_mode == "demo"
    total_imgs = len(sample_records)
    total_presets = len(preset_files)
    ranking_state = {"error": ""}

    self.test_log_text.delete(1.0, tk.END)
    if not self._lock_ui_for_testing("pz2.ocr_ranking.run", "PZ2: ranking presetów OCR"):
        return
    self._set_ocr_ranking_modal_running(True)
    self._set_ocr_ranking_modal_source(
        self._compose_ocr_ranking_source_text(source_desc="", total_imgs=total_imgs, ranking_mode=ranking_mode, preset_count=total_presets),
        tone="muted",
    )
    self._update_ocr_ranking_modal_progress(0, total_presets, sample_count=total_imgs, demo_mode=demo_mode)
    self._set_ocr_ranking_modal_status("Ranking presetów OCR w toku", "warning")
    self._log(self.test_log_text, "=======================================================", "HEADER")
    self._log(self.test_log_text, "ROZPOCZYNAM RANKING PRESETÓW OCR", "HEADER")

    cache_file = Path(sample_bundle.get("cache_file") or (self.presets_dir / "global_ranking.json"))
    dataset_signature = str(sample_bundle.get("dataset_signature") or "")
    source_desc = str(sample_bundle.get("description") or "wczytana lista tablic")
    session_token = self._project_reset_token
    self._set_ocr_ranking_modal_source(
        self._compose_ocr_ranking_source_text(source_desc, total_imgs, ranking_mode, total_presets),
        tone="muted",
    )
    self._log(self.test_log_text, f"Źródło próbek: {source_desc}", "INFO")
    effective_ocr_device = self._device_to_ocr(self._get_effective_detection_device_choice())

    def worker():
        try:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            ranking_cache = {}
            if cache_file.exists():
                try:
                    loaded_cache = self._load_json_file_safely(cache_file)
                    if isinstance(loaded_cache, dict):
                        ranking_cache = loaded_cache
                except Exception:
                    pass

            ocr_engine = PlateOCR(device=effective_ocr_device)
            detector = CharacterDetector(method=DetectionMethod.OCR, ocr_engine=ocr_engine)

            results_table = []

            for p_idx, p_file in enumerate(preset_files):
                if session_token != self._project_reset_token:
                    return

                preset_name = p_file.stem
                try:
                    with open(p_file, 'r', encoding='utf-8') as f:
                        preset_params = json.load(f)
                except Exception:
                    continue

                clean_params = {
                    k: v
                    for k, v in preset_params.items()
                    if not k.startswith("char_do_") and k not in {"char_ocr_conf", "char_ocr_min_height_ratio"}
                }
                ocr_min_height_ratio = float(preset_params.get("char_ocr_min_height_ratio", 0.58) or 0.58)
                param_signature = str(
                    sorted(clean_params.items())
                    + [
                        ("char_ocr_conf", float(preset_params.get("char_ocr_conf", 0.25) or 0.25)),
                        ("char_ocr_min_height_ratio", ocr_min_height_ratio),
                    ]
                )
                cached_entry = ranking_cache.get(preset_name, {}) if isinstance(ranking_cache, dict) else {}

                if (
                    isinstance(cached_entry, dict)
                    and cached_entry.get("signature") == param_signature
                    and str(cached_entry.get("dataset_signature") or "") == dataset_signature
                ):
                    results_table.append((cached_entry["acc"], cached_entry["matches"], preset_name, True))
                    self.frame.after(
                        0,
                        lambda processed=(p_idx + 1), total=total_presets, samples=total_imgs, is_demo=demo_mode: (
                            self._update_ocr_ranking_modal_progress(processed, total, sample_count=samples, demo_mode=is_demo)
                            if session_token == self._project_reset_token else None
                        )
                    )
                    continue

                ocr_engine.custom_prep_params = clean_params
                if "char_ocr_conf" in preset_params:
                    ocr_engine.confidence_threshold = float(preset_params.get("char_ocr_conf", 0.25))
                detector.ocr_min_height_ratio = max(0.20, min(1.00, ocr_min_height_ratio))

                perfect_matches = 0
                for sample in sample_records:
                    if session_token != self._project_reset_token:
                        return

                    img_path = Path(str(sample.get("image_path") or "").strip())
                    if not img_path.exists():
                        continue
                    plate_img = cv2.imread(str(img_path))
                    if plate_img is None:
                        continue

                    expected = [
                        str(text or "").strip().upper()
                        for text in (sample.get("expected_texts") or [])
                        if str(text or "").strip()
                    ]
                    if not expected:
                        continue
                    chars = detector.detect(plate_img)
                    detected_text = "".join([str(c.character) for c in chars]).strip().upper()
                    if detected_text and detected_text in expected:
                        perfect_matches += 1

                acc = (perfect_matches / total_imgs) * 100 if total_imgs > 0 else 0
                ranking_cache[preset_name] = {
                    "name": preset_name,
                    "acc": acc,
                    "matches": perfect_matches,
                    "signature": param_signature,
                    "dataset_signature": dataset_signature,
                    "params": preset_params
                }
                results_table.append((acc, perfect_matches, preset_name, False))
                self.frame.after(
                    0,
                    lambda processed=(p_idx + 1), total=total_presets, samples=total_imgs, is_demo=demo_mode: (
                        self._update_ocr_ranking_modal_progress(processed, total, sample_count=samples, demo_mode=is_demo)
                        if session_token == self._project_reset_token else None
                    )
                )

            if session_token != self._project_reset_token:
                return

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(ranking_cache, f, indent=4, ensure_ascii=False)

            results_table.sort(key=lambda x: x[0], reverse=True)

            self._log(self.test_log_text, "=" * 55, "HEADER")
            for i, (acc, matches, name, from_cache) in enumerate(results_table):
                self._log(self.test_log_text, f"#{i+1}. {name.ljust(22)} | {acc:5.1f}%  ({matches}/{total_imgs})", "SUCCESS" if i == 0 else "INFO")

            self.frame.after(
                0,
                lambda table=list(results_table), total=total_imgs, source=source_desc, mode=ranking_mode: (
                    None if mode == "demo" else self._update_winner_label(),
                    self._remember_ocr_ranking_results(table, total, source, mode),
                    self._refresh_ocr_ranking_modal_results()
                )
            )

        except Exception as e:
            ranking_state["error"] = str(e)
            self.frame.after(
                0,
                lambda err=str(e): self._set_ocr_ranking_modal_status(
                    f"Błąd rankingu OCR: {err}",
                    tone="error",
                )
            )
            self._log(self.test_log_text, f"\n❌ BŁĄD RANKINGU: {e}", "ERROR")
        finally:
            def finalize():
                if session_token != self._project_reset_token:
                    return

                if self._step3_linear_mode and CAMPAIGN.get_active_project_name():
                    self.unlock_dataset_subtab()

                if not ranking_state["error"]:
                    self._update_ocr_ranking_modal_progress(
                        total_presets,
                        total_presets,
                        sample_count=total_imgs,
                        demo_mode=demo_mode,
                        finished=True,
                    )
                    final_status = "Turniej demo zakończony." if ranking_mode == "demo" else "Turniej zakończony!"
                    self._set_ocr_ranking_modal_status(final_status, "success")
                self._set_ocr_ranking_modal_running(False)
                self._unlock_ui_after_testing()

            self.frame.after(0, finalize)

    threading.Thread(target=worker, daemon=True).start()


def open_ocr_ranking_modal(host, progress_bar_cls):
    self = host
    existing = getattr(self, "_ocr_ranking_modal", None)
    try:
        if existing is not None and existing.winfo_exists():
            try:
                existing.grab_release()
            except Exception:
                pass
            existing.destroy()
    except Exception:
        self._ocr_ranking_modal = None

    sample_bundle = self._get_ocr_ranking_sample_bundle()
    sample_records = list(sample_bundle.get("samples") or [])
    preset_files = [f for f in self.presets_dir.glob("*.json") if f.name != "global_ranking.json"]
    ranking_mode = str(sample_bundle.get("mode") or "preview")
    demo_mode = ranking_mode == "demo"
    total_imgs = len(sample_records)
    total_presets = len(preset_files)
    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", palette.get("bg", "#252526"))
    panel_alt = palette.get("panel_alt", "#2d2d30")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = self._get_readable_text_color(panel_bg, preferred=palette.get("fg", "#f3f3f3"))
    panel_fg = self._get_readable_text_color(panel_alt, preferred=palette.get("fg", "#f3f3f3"))
    muted = self._get_readable_text_color(panel_bg, preferred=palette.get("muted", "#c7c7c7"))
    panel_muted = self._get_readable_text_color(panel_alt, preferred=palette.get("muted", "#c7c7c7"))

    parent_modal = getattr(self, "_detection_pipeline_modal", None)
    try:
        if parent_modal is not None and not parent_modal.winfo_exists():
            parent_modal = None
    except Exception:
        parent_modal = None
    parent_window = parent_modal or self.frame.winfo_toplevel()

    win = tk.Toplevel(parent_window)
    self._ocr_ranking_modal = win
    win.title("Ranking presetów OCR")
    win.resizable(False, False)
    try:
        win.configure(bg=panel_bg)
    except Exception:
        pass

    try:
        win.transient(parent_window)
    except Exception:
        pass
    try:
        win.grab_set()
    except Exception:
        pass

    def _close(*_args):
        try:
            win.grab_release()
        except Exception:
            pass
        try:
            win.destroy()
        except Exception:
            pass
        if getattr(self, "_ocr_ranking_modal", None) is win:
            self._ocr_ranking_modal = None
        if parent_modal is not None:
            try:
                if parent_modal.winfo_exists():
                    parent_modal.grab_set()
                    parent_modal.lift()
                    parent_modal.focus_force()
            except Exception:
                pass
        return "break"

    win.protocol("WM_DELETE_WINDOW", _close)
    win.bind("<Escape>", _close, add="+")

    host_frame = tk.Frame(win, bg=panel_bg, bd=0, highlightthickness=0, padx=16, pady=16)
    host_frame.pack(fill=tk.BOTH, expand=True)

    header = SectionHeaderLabel(host_frame, self.app, text="Ranking presetów OCR")
    header.pack(fill=tk.X, pady=(0, 10))

    intro = tk.Label(
        host_frame,
        text="Tutaj uruchomisz turniej presetów OCR na aktualnym katalogu wyodrębnionych tablic. Próbki demo zostają tylko w Laboratorium OCR i nie biorą udziału w rankingu.",
        bg=panel_bg,
        fg=muted,
        justify=tk.LEFT,
        wraplength=430,
        anchor="w",
    )
    intro.pack(fill=tk.X, pady=(0, 12))

    control_card = tk.Frame(
        host_frame,
        bg=panel_alt,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        padx=12,
        pady=12,
    )
    control_card.pack(fill=tk.X)

    self._ocr_ranking_modal_source_lbl = tk.Label(
        control_card,
        text="",
        bg=panel_alt,
        fg=panel_muted,
        anchor="w",
        justify=tk.LEFT,
        wraplength=420,
        bd=0,
        highlightthickness=0,
    )
    self._ocr_ranking_modal_source_lbl.pack(fill=tk.X)

    self._ocr_ranking_modal_status_lbl = tk.Label(
        control_card,
        text="",
        bg=panel_alt,
        fg=panel_muted,
        anchor="w",
        justify=tk.LEFT,
        wraplength=420,
        bd=0,
        highlightthickness=0,
    )
    self._ocr_ranking_modal_status_lbl.pack(fill=tk.X, pady=(6, 0))

    progress_row = tk.Frame(control_card, bg=panel_alt, bd=0, highlightthickness=0)
    progress_row.pack(fill=tk.X, pady=(10, 0))

    self._ocr_ranking_modal_progress = progress_bar_cls(
        progress_row,
        maximum=100,
        value=0,
        thickness=2,
        trough_color=palette.get("panel", "#252526"),
        fill_color=palette.get("success", "#2ecc71"),
        bg=panel_alt,
        width=220,
        height=6,
    )
    self._ocr_ranking_modal_progress.pack(side=tk.LEFT)

    self._ocr_ranking_modal_progress_detail_lbl = tk.Label(
        progress_row,
        text="",
        bg=panel_alt,
        fg=panel_muted,
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self._ocr_ranking_modal_progress_detail_lbl.pack(side=tk.LEFT, padx=(8, 0))

    results_host = tk.Frame(host_frame, bg=panel_bg, bd=0, highlightthickness=0)
    results_host.pack(fill=tk.X, pady=(12, 0))
    self._ocr_ranking_modal_results_host = results_host

    footer = tk.Frame(host_frame, bg=panel_bg, bd=0, highlightthickness=0)
    footer.pack(fill=tk.X, pady=(12, 0))

    self._ocr_ranking_modal_start_btn = ttk.Button(
        footer,
        text="Start rankingu",
        command=self._run_preset_ranking,
        style="Accent.TButton",
    )
    self._ocr_ranking_modal_start_btn.pack(side=tk.LEFT)
    self._ocr_ranking_modal_start_btn.configure(padding=(8, 2))

    close_btn = ttk.Button(
        footer,
        text="Zamknij",
        command=_close,
        style="WorkflowCard.TButton",
    )
    close_btn.pack(side=tk.RIGHT)
    close_btn.configure(padding=(8, 2))

    source_text = self._compose_ocr_ranking_source_text(
        str(sample_bundle.get("description") or sample_bundle.get("message") or ""),
        total_imgs,
        ranking_mode,
        total_presets,
    )
    self._set_ocr_ranking_modal_source(source_text, tone="muted" if total_imgs > 0 else "warning")

    if self.is_processing:
        status_text = "Trwa inne zadanie. Poczekaj do jego zakończenia."
        status_tone = "warning"
        start_enabled = False
    elif not sample_records:
        status_text = str(sample_bundle.get("message") or "Brak próbek do rankingu OCR.")
        status_tone = "warning"
        start_enabled = False
    elif not preset_files:
        status_text = "Brak presetów OCR. Otwórz Laboratorium OCR i zapisz co najmniej jeden preset."
        status_tone = "warning"
        start_enabled = False
    else:
        status_text = "Gotowy do uruchomienia rankingu."
        status_tone = "neutral"
        start_enabled = True

    self._set_ocr_ranking_modal_status(status_text, tone=status_tone)
    self._update_ocr_ranking_modal_progress(0, total_presets, sample_count=total_imgs, demo_mode=demo_mode)
    self._refresh_ocr_ranking_modal_results()
    self._set_ocr_ranking_modal_running(not start_enabled)

    try:
        self.app.style_panel_surface(win, background=panel_bg)
    except Exception:
        pass

    try:
        win.update_idletasks()
        root = parent_window
        x = root.winfo_rootx() + max(40, int((root.winfo_width() - win.winfo_reqwidth()) / 2))
        y = root.winfo_rooty() + max(40, int((root.winfo_height() - win.winfo_reqheight()) / 2))
        win.geometry(f"+{x}+{y}")
    except Exception:
        pass


def remember_ocr_ranking_results(host, results_table, total_imgs: int, source_desc: str, ranking_mode: str):
    self = host
    prepared = []
    for idx, item in enumerate(results_table or [], start=1):
        try:
            acc, matches, name, from_cache = item
        except Exception:
            continue
        prepared.append({
            "rank": idx,
            "name": str(name or "").strip(),
            "acc": float(acc or 0.0),
            "matches": int(matches or 0),
            "from_cache": bool(from_cache),
        })
    self._ocr_ranking_last_results = prepared
    self._ocr_ranking_last_total = max(0, int(total_imgs or 0))
    self._ocr_ranking_last_source_desc = str(source_desc or "").strip()
    self._ocr_ranking_last_mode = str(ranking_mode or "").strip().lower()


def compose_ocr_ranking_source_text(
    source_desc: str = "",
    total_imgs: int = 0,
    ranking_mode: str = "",
    preset_count: int = 0,
) -> str:
    chunks = []
    source_value = str(source_desc or "").strip()
    if source_value:
        chunks.append(f"Źródło: {source_value}")
    if int(total_imgs or 0) > 0:
        chunks.append(f"Próbki: {int(total_imgs)}")
    if int(preset_count or 0) > 0:
        chunks.append(f"Presety: {int(preset_count)}")
    if str(ranking_mode or "").strip().lower() == "demo":
        chunks.append("Tryb demo")
    return " | ".join(chunks) or "Brak źródła rankingu OCR."


def set_winner_name(host, text: str, tone: str = "neutral"):
    label = getattr(host, "winner_name_lbl", None)
    if not host._set_inline_status_label_state(label, text=text, tone=tone, emphasis=True):
        host._set_themed_label_state(label, text=text, tone=tone, emphasis=True)


def set_winner_acc(host, text: str, tone: str = "muted"):
    label = getattr(host, "winner_acc_lbl", None)
    if not host._set_inline_status_label_state(label, text=text, tone=tone, emphasis=False):
        host._set_themed_label_state(label, text=text, tone=tone, emphasis=False)


def update_winner_label(host):
    best_preset_data, best_acc = host._get_best_preset()
    if best_preset_data and best_preset_data.get("name"):
        name = best_preset_data.get("name")
        host._set_winner_name(f"Lider: {name.upper()} .json", "success")
        host._set_winner_acc(f" Skuteczność najlepszego presetu: {best_acc:.1f}% ", "success")
    else:
        host._set_winner_name("BRAK DANYCH Z TURNIEJU", "neutral")
        host._set_winner_acc("Skuteczność detekcji OCR: 0.0%", "error")


def set_ocr_ranking_modal_label(host, attr_name: str, text: str, tone: str = "muted", emphasis: bool = False):
    self = host
    label = getattr(self, attr_name, None)
    if label is None:
        return
    try:
        if not label.winfo_exists():
            return
    except Exception:
        return
    if not self._set_inline_status_label_state(label, text=text, tone=tone, emphasis=emphasis):
        self._set_themed_label_state(label, text=text, tone=tone, emphasis=emphasis)


def set_ocr_ranking_modal_status(host, text: str, tone: str = "neutral"):
    set_ocr_ranking_modal_label(host, "_ocr_ranking_modal_status_lbl", text, tone=tone, emphasis=False)


def set_ocr_ranking_modal_source(host, text: str, tone: str = "muted"):
    set_ocr_ranking_modal_label(host, "_ocr_ranking_modal_source_lbl", text, tone=tone, emphasis=False)


def set_ocr_ranking_modal_running(host, running: bool):
    self = host
    btn = getattr(self, "_ocr_ranking_modal_start_btn", None)
    if btn is None:
        return
    try:
        if btn.winfo_exists():
            btn.config(state=tk.DISABLED if running else tk.NORMAL)
    except Exception:
        pass


def update_ocr_ranking_modal_progress(
    host,
    processed_presets: int,
    total_presets: int,
    *,
    sample_count: int = 0,
    demo_mode: bool = False,
    finished: bool = False,
):
    self = host
    total_value = max(0, int(total_presets or 0))
    processed_value = max(0, min(int(processed_presets or 0), total_value if total_value > 0 else int(processed_presets or 0)))
    sample_value = max(0, int(sample_count or 0))
    progress = getattr(self, "_ocr_ranking_modal_progress", None)
    try:
        if progress is not None and progress.winfo_exists():
            pct = ((processed_value / max(1, total_value)) * 100.0) if total_value > 0 else 0.0
            progress.config(value=100.0 if finished else pct)
    except Exception:
        pass

    if total_value <= 0 and sample_value <= 0:
        detail_text = ""
    else:
        sample_text = f"Demo: {sample_value} próbki" if demo_mode else f"Próbki: {sample_value}"
        detail_text = f"{sample_text} | Presety {processed_value}/{total_value}" if total_value > 0 else sample_text
    detail_tone = "success" if finished or (total_value > 0 and processed_value >= total_value) else "muted"
    set_ocr_ranking_modal_label(
        self,
        "_ocr_ranking_modal_progress_detail_lbl",
        detail_text,
        tone=detail_tone,
        emphasis=True,
    )


def populate_ocr_ranking_results_host(host_tab, results_host, *, panel_alt: str, border: str, fg: str, palette: dict):
    self = host_tab
    if results_host is None:
        return
    try:
        if not results_host.winfo_exists():
            return
    except Exception:
        return

    for child in list(results_host.winfo_children()):
        try:
            child.destroy()
        except Exception:
            pass

    last_results = list(getattr(self, "_ocr_ranking_last_results", []) or [])
    try:
        results_bg = str(results_host.cget("bg") or palette.get("panel", "#252526"))
    except Exception:
        results_bg = palette.get("panel", "#252526")
    results_muted = self._get_readable_text_color(results_bg, preferred=palette.get("muted", "#c7c7c7"))
    panel_fg = self._get_readable_text_color(panel_alt, preferred=fg)
    panel_muted = self._get_readable_text_color(panel_alt, preferred=palette.get("muted", "#c7c7c7"))
    if not last_results:
        empty_lbl = tk.Label(
            results_host,
            text="Brak zapisanych wyników rankingu OCR.",
            bg=results_bg,
            fg=results_muted,
            anchor="w",
            justify=tk.LEFT,
            wraplength=420,
            bd=0,
            highlightthickness=0,
        )
        empty_lbl.pack(fill=tk.X)
        return

    ranking_card = tk.Frame(
        results_host,
        bg=panel_alt,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        padx=12,
        pady=12,
    )
    ranking_card.pack(fill=tk.X)

    ranking_title = tk.Label(
        ranking_card,
        text="Ostatni ranking presetów OCR",
        bg=panel_alt,
        fg=panel_fg,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
    )
    ranking_title.pack(fill=tk.X, pady=(0, 4))

    source_label = tk.Label(
        ranking_card,
        text=compose_ocr_ranking_source_text(
            getattr(self, "_ocr_ranking_last_source_desc", ""),
            getattr(self, "_ocr_ranking_last_total", 0),
            getattr(self, "_ocr_ranking_last_mode", ""),
        ),
        bg=panel_alt,
        fg=panel_muted,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        wraplength=420,
    )
    source_label.pack(fill=tk.X, pady=(0, 10))

    total_imgs = int(getattr(self, "_ocr_ranking_last_total", 0) or 0)
    for item in last_results[:5]:
        suffix = " [cache]" if item.get("from_cache") else ""
        row_text = (
            f"#{item.get('rank', 0)}  {item.get('name', '').upper()}  "
            f"{item.get('acc', 0.0):.1f}% ({item.get('matches', 0)}/{max(1, total_imgs)}){suffix}"
        )
        row = tk.Label(
            ranking_card,
            text=row_text,
            bg=panel_alt,
            fg=panel_fg,
            font=("Consolas", 9),
            anchor="w",
            justify=tk.LEFT,
        )
        row.pack(fill=tk.X, pady=(0, 2))


def refresh_ocr_ranking_modal_results(host):
    self = host
    results_host = getattr(self, "_ocr_ranking_modal_results_host", None)
    palette = getattr(self.app, "palette", {})
    panel_alt = palette.get("panel_alt", "#2d2d30")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    populate_ocr_ranking_results_host(self, results_host, panel_alt=panel_alt, border=border, fg=fg, palette=palette)


def open_ocr_summary_modal(host):
    self = host
    existing = getattr(self, "_ocr_summary_modal", None)
    try:
        if existing is not None and existing.winfo_exists():
            existing.destroy()
    except Exception:
        self._ocr_summary_modal = None

    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", palette.get("bg", "#252526"))
    panel_alt = palette.get("panel_alt", "#2d2d30")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")

    win = tk.Toplevel(self.frame)
    self._ocr_summary_modal = win
    win.title("Podsumowanie OCR")
    win.resizable(False, False)

    try:
        win.transient(self.frame.winfo_toplevel())
    except Exception:
        pass
    try:
        win.grab_set()
    except Exception:
        pass

    def _close(*_args):
        try:
            win.grab_release()
        except Exception:
            pass
        try:
            win.destroy()
        except Exception:
            pass
        if getattr(self, "_ocr_summary_modal", None) is win:
            self._ocr_summary_modal = None
        return "break"

    win.protocol("WM_DELETE_WINDOW", _close)
    win.bind("<Escape>", _close, add="+")

    modal_host = tk.Frame(win, bg=panel_bg, bd=0, highlightthickness=0, padx=16, pady=16)
    modal_host.pack(fill=tk.BOTH, expand=True)

    header = SectionHeaderLabel(modal_host, self.app, text="Podsumowanie OCR")
    header.pack(fill=tk.X, pady=(0, 10))

    intro = tk.Label(
        modal_host,
        text="Tutaj znajdziesz lidera OCR i bieżący status rankingu bez zajmowania miejsca w prawym panelu.",
        bg=panel_bg,
        fg=palette.get("muted", "#c7c7c7"),
        justify=tk.LEFT,
        wraplength=360,
        anchor="w",
    )
    intro.pack(fill=tk.X, pady=(0, 12))

    info_card = tk.Frame(
        modal_host,
        bg=panel_alt,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        padx=12,
        pady=12,
    )
    info_card.pack(fill=tk.X)

    leader_title = tk.Label(
        info_card,
        text="Lider OCR",
        bg=panel_alt,
        fg=fg,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
    )
    leader_title.pack(fill=tk.X, pady=(0, 4))

    winner_name_lbl = getattr(self, "winner_name_lbl", None)
    leader_value = tk.Label(
        info_card,
        text=str(winner_name_lbl.cget("text") if winner_name_lbl is not None else "BRAK DANYCH"),
        bg=panel_alt,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0,
    )
    leader_value.pack(fill=tk.X, pady=(0, 10))
    self._set_inline_status_label_state(
        leader_value,
        text=leader_value.cget("text"),
        tone=getattr(winner_name_lbl, "_inline_status_tone", "neutral"),
        emphasis=getattr(winner_name_lbl, "_inline_status_emphasis", True),
    )

    status_title = tk.Label(
        info_card,
        text="Status rankingu OCR",
        bg=panel_alt,
        fg=palette.get("muted", "#c7c7c7"),
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
    )
    status_title.pack(fill=tk.X, pady=(0, 4))

    winner_acc_lbl = getattr(self, "winner_acc_lbl", None)
    status_value = tk.Label(
        info_card,
        text=str(winner_acc_lbl.cget("text") if winner_acc_lbl is not None else "0.0%"),
        bg=panel_alt,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
        bd=0,
        highlightthickness=0,
    )
    status_value.pack(fill=tk.X)
    self._set_inline_status_label_state(
        status_value,
        text=status_value.cget("text"),
        tone=getattr(winner_acc_lbl, "_inline_status_tone", "muted"),
        emphasis=getattr(winner_acc_lbl, "_inline_status_emphasis", False),
    )

    last_results = list(getattr(self, "_ocr_ranking_last_results", []) or [])
    if last_results:
        ranking_card = tk.Frame(
            modal_host,
            bg=panel_alt,
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
            padx=12,
            pady=12,
        )
        ranking_card.pack(fill=tk.X, pady=(12, 0))

        ranking_title = tk.Label(
            ranking_card,
            text="Ostatni ranking presetów OCR",
            bg=panel_alt,
            fg=fg,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
            justify=tk.LEFT,
        )
        ranking_title.pack(fill=tk.X, pady=(0, 4))

        source_chunks = []
        source_desc = str(getattr(self, "_ocr_ranking_last_source_desc", "") or "").strip()
        if source_desc:
            source_chunks.append(f"Źródło: {source_desc}")
        total_imgs = int(getattr(self, "_ocr_ranking_last_total", 0) or 0)
        if total_imgs > 0:
            source_chunks.append(f"Próbki: {total_imgs}")
        ranking_mode = str(getattr(self, "_ocr_ranking_last_mode", "") or "").strip()
        if ranking_mode == "demo":
            source_chunks.append("Tryb demo")

        source_label = tk.Label(
            ranking_card,
            text=" | ".join(source_chunks),
            bg=panel_alt,
            fg=palette.get("muted", "#c7c7c7"),
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=360,
        )
        source_label.pack(fill=tk.X, pady=(0, 10))

        for item in last_results[:5]:
            suffix = " [cache]" if item.get("from_cache") else ""
            row_text = (
                f"#{item.get('rank', 0)}  {item.get('name', '').upper()}  "
                f"{item.get('acc', 0.0):.1f}% ({item.get('matches', 0)}/{max(1, total_imgs)}){suffix}"
            )
            row = tk.Label(
                ranking_card,
                text=row_text,
                bg=panel_alt,
                fg=fg,
                font=("Consolas", 9),
                anchor="w",
                justify=tk.LEFT,
            )
            row.pack(fill=tk.X, pady=(0, 2))

    footer = ttk.Frame(modal_host)
    footer.pack(fill=tk.X, pady=(12, 0))

    close_btn = ttk.Button(
        footer,
        text="Zamknij",
        command=_close,
        style="WorkflowCard.TButton",
    )
    close_btn.pack(side=tk.RIGHT)
    close_btn.configure(padding=(8, 2))

    try:
        self.app.style_panel_surface(win, background=panel_bg)
    except Exception:
        pass

    try:
        win.update_idletasks()
        root = self.frame.winfo_toplevel()
        x = root.winfo_rootx() + max(40, int((root.winfo_width() - win.winfo_reqwidth()) / 2))
        y = root.winfo_rooty() + max(40, int((root.winfo_height() - win.winfo_reqheight()) / 2))
        win.geometry(f"+{x}+{y}")
    except Exception:
        pass


def open_filter_lab(host):
    self = host
    sample_bundle = self._get_ocr_lab_sample_bundle()
    sample_pool = list(sample_bundle.get("samples") or [])
    if not sample_pool:
        return messagebox.showinfo("Brak", sample_bundle.get("message") or "Brak próbek do laboratorium OCR.")
        return messagebox.showinfo("Brak", "Wczytaj najpierw listę tablic.")

    demo_mode = str(sample_bundle.get("mode") or "") == "demo"
    allow_random = bool(sample_bundle.get("allow_random", False))

    def roll_images():
        res = []
        if demo_mode:
            chosen_samples = list(sample_pool[: min(3, len(sample_pool))])
        else:
            s = min(3, len(sample_pool))
            chosen_samples = random.sample(sample_pool, s)
        for item in chosen_samples:
            i = cv2.imread(str(item.get("image_path") or ""))
            if i is not None:
                res.append((str(item.get("sample_id") or ""), i, item))
        return res

    self.lab_current_images = roll_images()
    if not self.lab_current_images:
        return

    best_preset_data, best_acc = self._get_best_preset()
    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    panel_alt_bg = palette.get("panel_alt", "#2d2d30")
    field_bg = palette.get("field", panel_alt_bg)
    preview_plate_bg = "#000000"
    fg = palette.get("fg", "#f3f3f3")
    muted_fg = palette.get("muted", "#c7c7c7")
    info_fg = palette.get("info", palette.get("accent", "#2980b9"))
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

    lab_win = tk.Toplevel(self.frame)
    lab_win.title("Laboratorium Filtrów OCR")
    lab_win.geometry("1100x850")
    try:
        parent_win = self.frame.winfo_toplevel()
    except Exception:
        parent_win = None
    try:
        lab_win.resizable(True, True)
    except Exception:
        pass
    try:
        anchor = parent_win or lab_win
        anchor.update_idletasks()
        screen_w = int(lab_win.winfo_screenwidth() or 1600)
        screen_h = int(lab_win.winfo_screenheight() or 1000)
        anchor_w = int(anchor.winfo_width() or anchor.winfo_reqwidth() or 1280)
        anchor_h = int(anchor.winfo_height() or anchor.winfo_reqheight() or 860)
        width = min(max(1180, int(anchor_w * 0.90)), max(1040, screen_w - 64))
        height = min(max(780, int(anchor_h * 0.88)), max(720, screen_h - 64))
        x = max(24, min(int(anchor.winfo_rootx() + max(0, (anchor_w - width) / 2)), screen_w - width - 24))
        y = max(24, min(int(anchor.winfo_rooty() + max(0, (anchor_h - height) / 2)), screen_h - height - 24))
        lab_win.minsize(1040, 720)
        lab_win.geometry(f"{width}x{height}+{x}+{y}")
    except Exception:
        try:
            lab_win.geometry("1180x820")
            lab_win.minsize(1040, 720)
        except Exception:
            pass
    try:
        lab_win.grab_set()
    except Exception:
        pass
    try:
        lab_win.configure(bg=palette.get("bg", panel_bg))
    except Exception:
        pass

    header_shell = tk.Frame(
        lab_win,
        bg=panel_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        padx=14,
        pady=10,
    )
    header_shell.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 8))

    header_text_col = tk.Frame(header_shell, bg=panel_bg, bd=0, highlightthickness=0)
    header_text_col.pack(side=tk.LEFT, fill=tk.X, expand=True)

    header_title_lbl = tk.Label(
        header_text_col,
        text="Laboratorium OCR",
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI", 12, "bold"),
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    header_title_lbl.pack(anchor=tk.W)

    header_subtitle_lbl = tk.Label(
        header_text_col,
        text="Dopasuj preprocessing i próg OCR na kilku próbkach, a potem zapisz ustawienia jako preset.",
        bg=panel_bg,
        fg=muted_fg,
        font=("Segoe UI", 9),
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    header_subtitle_lbl.pack(anchor=tk.W, pady=(4, 0))

    footer_shell = tk.Frame(
        lab_win,
        bg=panel_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        padx=14,
        pady=12,
    )
    footer_shell.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
    try:
        footer_shell.configure(height=112)
        footer_shell.pack_propagate(False)
    except Exception:
        pass

    footer_text_col = tk.Frame(footer_shell, bg=panel_bg, bd=0, highlightthickness=0)
    footer_text_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
    try:
        footer_text_col.configure(height=84)
        footer_text_col.pack_propagate(False)
    except Exception:
        pass

    footer_actions = tk.Frame(footer_shell, bg=panel_bg, bd=0, highlightthickness=0)
    footer_actions.pack(side=tk.RIGHT, anchor=tk.NE, padx=(28, 0))

    lab_help_default = "Najedź na etykietę suwaka, aby zobaczyć podpowiedź dla parametru."
    lab_help_lbl = tk.Label(
        footer_text_col,
        text=lab_help_default,
        justify=tk.LEFT,
        anchor="w",
        wraplength=560,
        height=3,
        bd=0,
        highlightthickness=0,
    )
    lab_help_lbl.pack(fill=tk.X, anchor=tk.W, padx=(0, 12))
    footer_text_col.bind(
        "<Configure>",
        lambda event: lab_help_lbl.configure(
            wraplength=max(320, int(getattr(event, "width", 0) or 0) - 18)
        ),
        add="+",
    )

    lab_status_row = tk.Frame(footer_text_col, bg=panel_bg, bd=0, highlightthickness=0)
    lab_status_row.pack(fill=tk.X, pady=(6, 0), padx=(0, 12))

    lab_source_lbl = tk.Label(
        lab_status_row,
        text="Tryb demo: 3 stałe wycinki z danych aplikacji." if demo_mode else "Próbka: losowanie z wczytanej listy tablic.",
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    lab_source_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

    lab_feedback_lbl = tk.Label(
        lab_status_row,
        text="Presety OCR są zapisywane do katalogu danych aplikacji.",
        justify=tk.RIGHT,
        anchor="e",
        bd=0,
        highlightthickness=0,
    )
    lab_feedback_lbl.pack(side=tk.RIGHT, padx=(24, 0))

    def _set_lab_footer_text(widget, text, tone="muted", emphasis=False):
        clean_text = str(text or "").strip() or " "
        self._set_inline_status_label_state(widget, text=clean_text, tone=tone, emphasis=emphasis)

    def set_lab_feedback(text, tone="muted", emphasis=False):
        if lab_win.winfo_exists():
            _set_lab_footer_text(lab_feedback_lbl, text, tone=tone, emphasis=emphasis)

    def update_lab_help(msg):
        if lab_win.winfo_exists():
            _set_lab_footer_text(lab_help_lbl, msg or lab_help_default, tone="muted", emphasis=False)

    self.old_status_updater = HELP.status_updater
    HELP.status_updater = update_lab_help

    main_content = ttk.Frame(lab_win)
    main_content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    left_container = ttk.Frame(main_content, width=350)
    left_container.pack(side=tk.LEFT, fill=tk.Y)
    left_container.pack_propagate(False)

    canvas_sliders = tk.Canvas(left_container, highlightthickness=0, bd=0, bg=panel_bg)
    scroll_sliders = WebSlimScrollbar(left_container, orient=tk.VERTICAL, command=canvas_sliders.yview)
    scrollable_frame = ttk.Frame(canvas_sliders, padding=10, style="Panel.TFrame")

    frame_id = canvas_sliders.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas_sliders.bind("<Configure>", lambda e: canvas_sliders.itemconfig(frame_id, width=e.width))
    scrollable_frame.bind("<Configure>", lambda e: canvas_sliders.configure(scrollregion=canvas_sliders.bbox("all")))
    canvas_sliders.configure(yscrollcommand=scroll_sliders.set)

    scroll_sliders.pack(side=tk.RIGHT, fill=tk.Y)
    canvas_sliders.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right_view_container = ttk.Frame(main_content, padding=10)
    right_view_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    view_canvas = tk.Canvas(right_view_container, bg=panel_alt_bg, highlightthickness=0, bd=0)
    view_scroll_v = WebSlimScrollbar(right_view_container, orient=tk.VERTICAL, command=view_canvas.yview)
    view_scroll_h = WebSlimScrollbar(right_view_container, orient=tk.HORIZONTAL, command=view_canvas.xview)

    view_scroll_h.pack(side=tk.BOTTOM, fill=tk.X)
    view_scroll_v.pack(side=tk.RIGHT, fill=tk.Y)
    view_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    view_canvas.configure(yscrollcommand=view_scroll_v.set, xscrollcommand=view_scroll_h.set)

    view_frame = tk.Frame(view_canvas, bg=panel_alt_bg, bd=0, highlightthickness=0)
    view_window_id = view_canvas.create_window((0, 0), window=view_frame, anchor="nw")
    view_frame.bind("<Configure>", lambda e: view_canvas.configure(scrollregion=view_canvas.bbox("all")))
    view_canvas.bind(
        "<Configure>",
        lambda e: view_canvas.itemconfigure(view_window_id, width=max(int(getattr(e, "width", 0) or 0), 760)),
        add="+",
    )

    def _lab_canvas_overflows(target_canvas) -> bool:
        try:
            bbox = target_canvas.bbox("all")
            if not bbox:
                return False
            content_height = int(bbox[3]) - int(bbox[1])
            viewport_height = int(target_canvas.winfo_height())
            return content_height > viewport_height + 1
        except Exception:
            return False

    def _on_lab_mousewheel(event):
        for target_canvas in (canvas_sliders, view_canvas):
            if self._inertial_scroll.scroll_canvas_if_targeted(
                target_canvas,
                event,
                pointer_widget=lab_win,
                overflow_checker=lambda c=target_canvas: _lab_canvas_overflows(c),
            ):
                return "break"
        return None

    lab_win.bind("<MouseWheel>", _on_lab_mousewheel, add="+")
    lab_win.bind("<Button-4>", _on_lab_mousewheel, add="+")
    lab_win.bind("<Button-5>", _on_lab_mousewheel, add="+")
    self._bind_scroll_canvas_children(
        scrollable_frame,
        canvas_sliders,
        lambda: _lab_canvas_overflows(canvas_sliders)
    )

    preview_grid = tk.Frame(view_frame, bg=panel_alt_bg, bd=0, highlightthickness=0)
    preview_grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    preview_grid.grid_columnconfigure(0, weight=0, minsize=180)
    preview_grid.grid_columnconfigure(1, weight=1, uniform="ocr_lab_cols")
    preview_grid.grid_columnconfigure(2, weight=1, uniform="ocr_lab_cols")

    def _make_preview_header(col: int, text: str, *, sticky: str = "ew", padx=(0, 0)):
        header_lbl = tk.Label(
            preview_grid,
            text=text,
            bg=panel_alt_bg,
            fg=fg,
            font=("Segoe UI", 10, "bold"),
            justify=tk.LEFT,
            anchor="w" if col == 0 else "center",
            bd=0,
            highlightthickness=0,
            padx=8,
            pady=6,
        )
        header_lbl.grid(row=0, column=col, sticky=sticky, padx=padx, pady=(0, 8))
        return header_lbl

    _make_preview_header(0, "Próbka", sticky="w", padx=(0, 8))
    _make_preview_header(1, "Przed", padx=(0, 8))
    _make_preview_header(2, "Po filtrach OCR")

    preview_plate_frame_pad = 16
    preview_plate_row_min_height = 220

    self.lab_image_labels = []
    for i in range(3):
        preview_grid.grid_rowconfigure(i + 1, minsize=preview_plate_row_min_height)
        sample_meta = tk.Frame(preview_grid, bg=panel_alt_bg, bd=0, highlightthickness=0)
        sample_meta.grid(row=i + 1, column=0, sticky="nw", padx=(0, 8), pady=(0, 12))

        lbl = tk.Label(
            sample_meta,
            text=f"Próbka {i+1}",
            bg=panel_alt_bg,
            fg=fg,
            font=("Segoe UI", 10, "bold"),
            justify=tk.LEFT,
            anchor="w",
            wraplength=170,
            bd=0,
            highlightthickness=0,
        )
        lbl.pack(fill=tk.X)

        orig_shell = tk.Frame(
            preview_grid,
            bg=preview_plate_bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
            padx=preview_plate_frame_pad,
            pady=preview_plate_frame_pad,
        )
        orig_shell.grid(row=i + 1, column=1, sticky="nsew", padx=(0, 8), pady=(0, 12))

        proc_shell = tk.Frame(
            preview_grid,
            bg=preview_plate_bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
            padx=preview_plate_frame_pad,
            pady=preview_plate_frame_pad,
        )
        proc_shell.grid(row=i + 1, column=2, sticky="nsew", pady=(0, 12))

        o_lbl = tk.Label(
            orig_shell,
            bg=preview_plate_bg,
            bd=0,
            highlightthickness=0,
            anchor="center",
            justify=tk.CENTER,
        )
        o_lbl.pack(fill=tk.BOTH, expand=True)

        p_lbl = tk.Label(
            proc_shell,
            bg=preview_plate_bg,
            bd=0,
            highlightthickness=0,
            anchor="center",
            justify=tk.CENTER,
        )
        p_lbl.pack(fill=tk.BOTH, expand=True)

        self.lab_image_labels.append({
            "lbl": lbl,
            "orig": o_lbl,
            "proc": p_lbl
        })

    self.lab_photo_refs = []
    active_traces = []

    def _fit_lab_preview_image(image_array, max_width: int = 360, max_height: int = 170):
        try:
            if image_array is None:
                return image_array
            h_img, w_img = image_array.shape[:2]
            if h_img <= 0 or w_img <= 0:
                return image_array
            scale = min(float(max_width) / float(w_img), float(max_height) / float(h_img), 1.0)
            if scale >= 0.999:
                return image_array
            resized_w = max(1, int(round(w_img * scale)))
            resized_h = max(1, int(round(h_img * scale)))
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            return cv2.resize(image_array, (resized_w, resized_h), interpolation=interpolation)
        except Exception:
            return image_array

    def update_preview(*args):
        if not lab_win.winfo_exists():
            return
        try:
            th = self.prep_height_var.get()
            ma = self.prep_angle_var.get()
            ct = self.prep_clip_var.get()
            dh = self.prep_denoise_var.get()
            cc = self.prep_clahe_var.get()
            use_bin = self.prep_use_bin_var.get()
            tb = self.prep_block_var.get()
            t_c = self.prep_c_var.get()
            if tb % 2 == 0:
                tb += 1
            ei = self.prep_erode_var.get()

            interp_str = self.interpolation_var.get()
            interp_map = {
                "nearest": cv2.INTER_NEAREST,
                "linear": cv2.INTER_LINEAR,
                "cubic": cv2.INTER_CUBIC,
                "lanczos4": cv2.INTER_LANCZOS4
            }
            cv2_interp = interp_map.get(interp_str.lower(), cv2.INTER_LANCZOS4)

            self.lab_photo_refs.clear()

            for idx, (pid, orig_img, sample_info) in enumerate(self.lab_current_images):
                if idx >= len(self.lab_image_labels):
                    break

                if abs(ma) > 0.1:
                    h, w = orig_img.shape[:2]
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), ma, 1.0)
                    rotated = cv2.warpAffine(
                        orig_img, M, (w, h),
                        flags=cv2_interp,
                        borderMode=cv2.BORDER_REPLICATE
                    )
                else:
                    rotated = orig_img.copy()

                gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY) if len(rotated.shape) == 3 else rotated.copy()
                h_g, w_g = gray.shape
                scale = float(th) / h_g if h_g > 0 else 1.0
                gray_scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2_interp)

                gray_display = _fit_lab_preview_image(gray_scaled)
                po = ImageTk.PhotoImage(Image.fromarray(gray_display))

                if ct < 255:
                    gray_scaled[gray_scaled > ct] = 255

                den = cv2.fastNlMeansDenoising(
                    gray_scaled, None,
                    h=dh, templateWindowSize=7, searchWindowSize=21
                ) if dh > 0 else gray_scaled

                if self.do_clahe_var.get() and cc > 0:
                    clahe = cv2.createCLAHE(clipLimit=cc, tileGridSize=(8, 8))
                    contrasted = clahe.apply(den)
                else:
                    contrasted = den

                blurred = cv2.GaussianBlur(contrasted, (3, 3), 0)

                if use_bin:
                    binary = cv2.adaptiveThreshold(
                        blurred, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        blockSize=max(3, tb),
                        C=t_c
                    )
                    if ei > 0:
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                        final = cv2.erode(binary, kernel, iterations=ei)
                    else:
                        final = binary
                else:
                    final = contrasted

                pad_pct = self.prep_padding_var.get()
                if pad_pct > 0:
                    py = int(final.shape[0] * (pad_pct / 100.0))
                    px = int(final.shape[1] * (pad_pct / 100.0))
                    final = cv2.copyMakeBorder(final, py, py, px, px, cv2.BORDER_CONSTANT, value=255)

                final_display = _fit_lab_preview_image(final)
                pp = ImageTk.PhotoImage(Image.fromarray(final_display))
                self.lab_photo_refs.extend([po, pp])

                ui_row = self.lab_image_labels[idx]
                sample_label = str(sample_info.get("label") or f"ID: {pid}").strip() or f"ID: {pid}"
                ui_row["lbl"].config(text=sample_label)
                ui_row["orig"].config(image=po)
                ui_row["proc"].config(image=pp)

            for idx in range(len(self.lab_current_images), len(self.lab_image_labels)):
                ui_row = self.lab_image_labels[idx]
                ui_row["lbl"].config(text="Brak próbki")
                ui_row["orig"].config(image="")
                ui_row["proc"].config(image="")

        except Exception:
            pass

    def add_slider(parent, label, var, from_, to_, res, ghost_key="", help_key=""):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=4)

        lbl_f = ttk.Frame(f)
        lbl_f.pack(fill=tk.X)

        main_label = ttk.Label(lbl_f, text=label)
        main_label.pack(side=tk.LEFT)

        if ghost_key and best_preset_data and isinstance(best_preset_data.get("params"), dict):
            params_dict = best_preset_data["params"]
            if ghost_key in params_dict:
                val = params_dict[ghost_key]
                ghost_str = f"{val:.1f}" if isinstance(val, float) else str(val)
                ttk.Label(
                    lbl_f,
                    text=f"[Zwycięzca: {ghost_str}]",
                    foreground=info_fg,
                    font=("Segoe UI", 9, "bold")
                ).pack(side=tk.RIGHT)

        s = ttk.Scale(f, from_=from_, to=to_, variable=var, command=update_preview)
        s.pack(side=tk.LEFT, fill=tk.X, expand=True)

        l = ttk.Label(f, width=5)
        l.pack(side=tk.RIGHT)

        def update_lbl(*a):
            if lab_win.winfo_exists():
                l.config(text=f"{var.get():.{res}f}")

        trace_id = var.trace_add("write", update_lbl)
        active_traces.append((var, trace_id))
        update_lbl()

        if help_key:
            HELP.bind_help(main_label, help_key)
            HELP.bind_help(s, help_key)

    intro_card = tk.Frame(
        scrollable_frame,
        bg=panel_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        padx=12,
        pady=10,
    )
    intro_card.pack(fill=tk.X, pady=(0, 15))
    tk.Label(
        intro_card,
        text="Laboratorium OCR",
        bg=panel_bg,
        fg=fg,
        font=("Segoe UI", 12, "bold"),
    ).pack(anchor=tk.W)
    tk.Label(
        intro_card,
        text="Dopasuj preprocessing i próg OCR na kilku próbkach, a potem zapisz ustawienia jako preset.",
        bg=panel_bg,
        fg=muted_fg,
        justify=tk.LEFT,
        wraplength=300,
        font=("Segoe UI", 9),
    ).pack(anchor=tk.W, pady=(4, 0))
    if best_preset_data and best_preset_data.get("name"):
        intro_meta = tk.Frame(intro_card, bg=panel_bg, bd=0, highlightthickness=0)
        intro_meta.pack(fill=tk.X, pady=(8, 0))
        tk.Label(
            intro_meta,
            text=f"Lider OCR: {best_preset_data.get('name').upper()}",
            bg=panel_bg,
            fg=fg,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            intro_meta,
            text=f"Ostatni najlepszy wynik: {best_acc:.1f}%",
            bg=panel_bg,
            fg=muted_fg,
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(2, 0))

    geom = ttk.LabelFrame(scrollable_frame, text=" 1. Geometria ", padding=10)
    geom.pack(fill=tk.X, pady=(0, 10))

    angle_row = ttk.Frame(geom)
    angle_row.pack(fill=tk.X)
    add_slider(angle_row, "Ręczna korekta kąta [°]:", self.prep_angle_var, -30, 30, 1, "manual_angle", "lab_angle")
    ttk.Button(geom, text="Reset Kąta", command=lambda: (self.prep_angle_var.set(0.0), update_preview())).pack(anchor=tk.E, pady=(0, 5))

    add_slider(geom, "Wysokość OCR (px):", self.prep_height_var, 40, 150, 0, "target_height", "lab_height")

    filt = ttk.LabelFrame(scrollable_frame, text=" 2. Filtry bazowe ", padding=10)
    filt.pack(fill=tk.X, pady=(0, 10))
    add_slider(filt, "Odcięcie odblasków (255=Wył):", self.prep_clip_var, 100, 255, 0, "clip_thresh", "lab_clip")
    add_slider(filt, "Usuwanie ziarna (0=Wył):", self.prep_denoise_var, 0, 50, 0, "denoise_h", "lab_denoise")
    cb2 = ttk.Checkbutton(filt, text="Wzmacniaj kontrast (CLAHE)", variable=self.do_clahe_var, command=update_preview)
    cb2.pack(anchor=tk.W)
    HELP.bind_help(cb2, "lab_clahe")
    add_slider(filt, "Siła CLAHE:", self.prep_clahe_var, 0.0, 10.0, 1, "clahe_clip", "lab_clahe")

    bina = ttk.LabelFrame(scrollable_frame, text=" 3. Binaryzacja ", padding=10)
    bina.pack(fill=tk.X, pady=(0, 10))
    ttk.Checkbutton(bina, text="Włącz pełną binaryzację", variable=self.prep_use_bin_var, command=update_preview).pack(anchor=tk.W)
    add_slider(bina, "Rozmiar bloku (nieparzyste):", self.prep_block_var, 3, 51, 0, "thresh_block", "lab_block")
    add_slider(bina, "Stała odcięcia (C):", self.prep_c_var, -20, 20, 0, "thresh_c", "lab_c")
    add_slider(bina, "Pogrubianie liter (Erozja):", self.prep_erode_var, 0, 5, 0, "erode_iter", "lab_erode")
    add_slider(bina, "Biała ramka - Padding [%]:", self.prep_padding_var, 0, 50, 0, "padding_pct", "lab_pad")

    ocr_f = ttk.LabelFrame(scrollable_frame, text=" 4. Parametry Sieci (OCR) ", padding=10)
    ocr_f.pack(fill=tk.X, pady=(0, 10))
    add_slider(ocr_f, "Wymagany próg pewności (0-1.0):", self.ocr_conf_var, 0.05, 0.95, 2, "char_ocr_conf", "lab_conf")

    add_slider(ocr_f, "Min. wysokość boxa OCR:", self.ocr_min_height_ratio_var, 0.20, 1.00, 2, "char_ocr_min_height_ratio", "lab_conf")

    def load_preset():
        preset_dir = Path(getattr(self, "presets_dir", CONFIG.get_presets_dir("ocr")))
        try:
            preset_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            preset_dir = preset_dir.resolve()
        except Exception:
            preset_dir = preset_dir.absolute()
        p = filedialog.askopenfilename(
            title="Wczytaj preset OCR",
            initialdir=str(preset_dir),
            initialfile="",
            filetypes=[("JSON", "*.json")],
            parent=lab_win,
        )
        if not p:
            return
        try:
            data = self._load_json_file_safely(Path(p))
            if not isinstance(data, dict):
                raise ValueError("invalid_preset")
            if "target_height" in data:
                self.prep_height_var.set(data["target_height"])
            if "manual_angle" in data:
                self.prep_angle_var.set(data["manual_angle"])
            if "clip_thresh" in data:
                self.prep_clip_var.set(data["clip_thresh"])
            if "denoise_h" in data:
                self.prep_denoise_var.set(data["denoise_h"])
            if "clahe_clip" in data:
                clahe_value = float(data["clahe_clip"] or 0.0)
                self.prep_clahe_var.set(clahe_value)
                self.do_clahe_var.set(clahe_value > 0.0)
            if "use_binarization" in data:
                self.prep_use_bin_var.set(bool(data["use_binarization"]))
            if "thresh_block" in data:
                self.prep_block_var.set(data["thresh_block"])
            if "thresh_c" in data:
                self.prep_c_var.set(data["thresh_c"])
            if "erode_iter" in data:
                self.prep_erode_var.set(data["erode_iter"])
            if "interpolation" in data and str(data["interpolation"]).strip():
                self.interpolation_var.set(str(data["interpolation"]))
            if "padding_pct" in data:
                self.prep_padding_var.set(data["padding_pct"])
            if "char_ocr_conf" in data:
                self.ocr_conf_var.set(data["char_ocr_conf"])
            if "char_ocr_min_height_ratio" in data:
                self.ocr_min_height_ratio_var.set(data["char_ocr_min_height_ratio"])
            update_preview()
            set_lab_feedback(f"Wczytano preset: {Path(p).stem}", tone="info")
        except Exception:
            set_lab_feedback("Nie udało się wczytać presetu OCR.", tone="error", emphasis=True)
            messagebox.showerror("Preset OCR", "Nie udało się wczytać wybranego presetu OCR.", parent=lab_win)

    def save_preset():
        raw_name = simpledialog.askstring("Preset OCR", "Podaj nazwę dla presetu:", parent=lab_win)
        if raw_name is None:
            return

        safe_name = re.sub(r'[<>:\"/\\\\|?*]+', "_", str(raw_name or "").strip()).strip(" .")
        if not safe_name:
            set_lab_feedback("Podaj poprawną nazwę presetu.", tone="warning", emphasis=True)
            messagebox.showinfo("Preset OCR", "Podaj poprawną nazwę presetu OCR.", parent=lab_win)
            return

        p = self.presets_dir / f"{safe_name}.json"
        if p.exists():
            overwrite = messagebox.askyesno(
                "Preset OCR",
                f"Preset '{safe_name}' już istnieje. Czy chcesz go nadpisać?",
                parent=lab_win,
            )
            if not overwrite:
                set_lab_feedback("Zapis presetu anulowany.", tone="muted")
                return

        data = self._get_current_prep_params()
        data["char_ocr_conf"] = self.ocr_conf_var.get()
        data["char_ocr_min_height_ratio"] = self.ocr_min_height_ratio_var.get()
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            set_lab_feedback(f"Zapisano preset: {safe_name}", tone="success", emphasis=True)
        except Exception:
            set_lab_feedback("Nie udało się zapisać presetu OCR.", tone="error", emphasis=True)
            messagebox.showerror("Preset OCR", "Nie udało się zapisać presetu OCR.", parent=lab_win)

    def safe_close():
        for var, tid in active_traces:
            try: var.trace_remove("write", tid)
            except: pass
        self._force_save_all()
        self.lab_photo_refs = []
        HELP.status_updater = self.old_status_updater
        lab_win.destroy()

    lab_win.protocol("WM_DELETE_WINDOW", safe_close)
    btn_roll = ttk.Button(
        footer_actions,
        text="Losuj próbki",
        style="WorkflowCard.TButton",
        command=lambda: (setattr(self, 'lab_current_images', roll_images()), update_preview())
    )
    btn_roll.grid(row=0, column=0, padx=(0, 8))
    if not allow_random:
        self._set_widget_state(btn_roll, "disabled")
    ttk.Button(
        footer_actions,
        text="Wczytaj preset",
        style="WorkflowCard.TButton",
        command=load_preset,
    ).grid(row=0, column=1, padx=8)
    ttk.Button(
        footer_actions,
        text="Zapisz preset",
        style="Accent.TButton",
        command=save_preset,
    ).grid(row=0, column=2, padx=8)
    ttk.Button(
        footer_actions,
        text="Zamknij",
        style="WorkflowCard.TButton",
        command=safe_close,
    ).grid(row=0, column=3, padx=(8, 0))

    try:
        self.app.style_panel_surface(lab_win, background=panel_bg)
        self.app.style_canvas_widget(canvas_sliders, background=panel_bg, bordercolor=border)
        self.app.style_canvas_widget(view_canvas, background=panel_alt_bg, bordercolor=border)
        self._set_inline_status_label_state(lab_help_lbl, text=lab_help_default, tone="muted", emphasis=False)
        self._set_inline_status_label_state(lab_source_lbl, text=lab_source_lbl.cget("text"), tone="muted", emphasis=False)
        self._set_inline_status_label_state(lab_feedback_lbl, text=lab_feedback_lbl.cget("text"), tone="muted", emphasis=False)
    except Exception:
        pass

    update_preview()
