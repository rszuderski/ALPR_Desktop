#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z3/PZ3 CVAT and dataset tab UI builder extracted from CharacterAnnotationTab."""

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog

from ..config import CONFIG, logger
from .help_manager import HELP
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .z3_free_mode_flow import set_pz3_dataset_source_mode, set_pz3_selected_path
from .z3_shared_ui import (
    refresh_pz3_cards_ui,
    refresh_pz3_cvat_card,
    refresh_pz3_dataset_card,
    refresh_pz3_dataset_mode_ui,
    refresh_pz3_dataset_source_card,
)


def pick_file(host, var) -> None:
    initial = host.preview_dir_var.get().strip()
    if not initial or not Path(initial).exists():
        initial = str(CONFIG.WORKSPACE_DIR.absolute())
    path = filedialog.askopenfilename(initialdir=initial, filetypes=[("XML", "*.xml")])
    if path:
        var.set(path)


def get_active_preview_context(host) -> dict:
    preview_dir_raw = str(host.preview_dir_var.get() or "").strip()
    if not host._preview_dir_has_plate_entries(preview_dir_raw):
        fallback_preview_dir = host._get_preferred_step3_preview_dir(require_plates=True, allow_fallback=True)
        if fallback_preview_dir:
            preview_dir_raw = str(fallback_preview_dir)
            try:
                host.preview_dir_var.set(preview_dir_raw)
                host._save_local_setting("char_preview_dir", preview_dir_raw)
            except Exception:
                pass

    if not preview_dir_raw:
        return {
            "ready": False,
            "reason": "missing_preview",
            "message": (
                "Brak aktywnego wyniku wyodrębniania tablic. Wróć do [Z3]/[PZ1], użyj „Wyodrębnij tablice”, "
                "a potem przejdź do [PZ2]. PZ2 korzysta z wyniku przygotowanego w PZ1."
            ),
            "preview_dir": None,
            "meta_path": None,
            "images_dir": None,
            "plate_count": 0,
        }

    preview_dir = Path(preview_dir_raw)
    meta_path = preview_dir / "metadata.json"
    images_dir = preview_dir / "images"

    if not preview_dir.exists() or not preview_dir.is_dir():
        return {
            "ready": False,
            "reason": "missing_directory",
            "message": f"Aktywny wynik wyodrębniania tablic nie istnieje:\n{preview_dir}",
            "preview_dir": preview_dir,
            "meta_path": meta_path,
            "images_dir": images_dir,
            "plate_count": 0,
        }

    if not meta_path.exists():
        return {
            "ready": False,
            "reason": "missing_metadata",
            "message": f"W aktywnym wyniku wyodrębniania tablic brakuje metadata.json:\n{preview_dir}",
            "preview_dir": preview_dir,
            "meta_path": meta_path,
            "images_dir": images_dir,
            "plate_count": 0,
        }

    if not images_dir.exists() or not images_dir.is_dir():
        return {
            "ready": False,
            "reason": "missing_images",
            "message": f"W aktywnym wyniku wyodrębniania tablic brakuje katalogu images/:\n{preview_dir}",
            "preview_dir": preview_dir,
            "meta_path": meta_path,
            "images_dir": images_dir,
            "plate_count": 0,
        }

    plate_count = int(host._get_preview_dir_plate_count(preview_dir) or 0)
    if plate_count <= 0:
        return {
            "ready": False,
            "reason": "empty_preview",
            "message": f"Wynik wyodrębniania jest pusty i nie zawiera tablic do eksportu:\n{preview_dir}",
            "preview_dir": preview_dir,
            "meta_path": meta_path,
            "images_dir": images_dir,
            "plate_count": 0,
        }

    return {
        "ready": True,
        "reason": "ok",
        "message": "",
        "preview_dir": preview_dir,
        "meta_path": meta_path,
        "images_dir": images_dir,
        "plate_count": plate_count,
    }


def refresh_preview_bound_action_states(host) -> None:
    context = host._get_active_preview_context()
    preview_ready = bool(context.get("ready"))
    button_state = "normal" if preview_ready else "disabled"

    for attr_name in ("btn_cvat_export", "btn_cvat_import"):
        host._set_widget_state(getattr(host, attr_name, None), button_state)

    export_hint = getattr(host, "cvat_export_state_lbl", None)
    if export_hint is not None:
        if preview_ready:
            preview_dir = context.get("preview_dir")
            plate_count = int(context.get("plate_count", 0) or 0)
            preview_name = preview_dir.name if isinstance(preview_dir, Path) else str(preview_dir or "")
            host._set_inline_status_label_state(
                export_hint,
                text=f"Aktywny preview z [PZ2]: {preview_name} | tablice do review: {plate_count}",
                tone="muted",
                emphasis=False,
            )
        else:
            host._set_inline_status_label_state(
                export_hint,
                text=(
                    "Preview run przygotujesz w [Z3]/[PZ1] przyciskiem „Wyodrębnij tablice do PZ2”. "
                    "PZ2 korzysta z aktywnego wyniku PZ1."
                ),
                tone="warning",
                emphasis=False,
            )

    import_hint = getattr(host, "cvat_import_target_hint_lbl", None)
    if import_hint is not None:
        if preview_ready:
            host._set_inline_status_label_state(
                import_hint,
                text="Import dotyczy aktywnego wyniku PZ2 wskazanego w sekcji źródła.",
                tone="muted",
                emphasis=False,
            )
        else:
            host._set_inline_status_label_state(
                import_hint,
                text=(
                    "Import wymaga aktywnego wyniku PZ2. Wróć do [PZ1] i wyodrębnij tablice, "
                    "a potem przejdź do [PZ2]."
                ),
                tone="warning",
                emphasis=False,
            )


def run_cvat_export(host) -> None:
    preview_context = host._get_active_preview_context()
    if not bool(preview_context.get("ready")):
        host._set_console_text(
            host.export_console,
            f"❌ BŁĄD: {str(preview_context.get('message') or 'Najpierw przygotuj poprawny wynik PZ2.')}"
        )
        return

    work_dir = preview_context["preview_dir"]
    export_dir = work_dir

    project_review_dir = host._get_project_review_dir()
    if project_review_dir is not None:
        try:
            preview_name = work_dir.name if work_dir.name else "review_pack"
        except Exception:
            preview_name = "review_pack"

        export_dir = project_review_dir / preview_name
        export_dir.mkdir(parents=True, exist_ok=True)
    meta_path = work_dir / "metadata.json"
    out_xml = export_dir / "annotations.xml"
    out_zip = export_dir / "cvat_export.zip"

    host._set_console_text(host.export_console, "⌛ Eksportowanie do CVAT w toku...")

    try:
        import datetime
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        if host.smart_export_var.get():
            filtered = {k: v for k, v in metadata.items() if v.get("status") != "perfect"}
            tmp_meta = work_dir / "temp_meta.json"
            with open(tmp_meta, 'w', encoding='utf-8') as f:
                json.dump(filtered, f, indent=2, ensure_ascii=False)
            source_meta = tmp_meta
        else:
            filtered = None
            source_meta = meta_path

        from ..cvat_tools.cvat_character_exporter import CVATCharacterExporter
        from ..cvat_tools.cvat_zip_manager import CVATZipManager

        if CVATCharacterExporter().export(source_meta, out_xml):
            exported_metadata = filtered if host.smart_export_var.get() else metadata
            xml_image_names = [f"{plate_id}.jpg" for plate_id in list(exported_metadata.keys())]
            zip_ok, zip_msg = CVATZipManager.create_cvat_import_zip(
                out_xml,
                work_dir / "images",
                out_zip,
                image_names_allowlist=xml_image_names,
            )
            review_manifest = {
                "review_manifest_id": f"review_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "source_preview_dir": str(work_dir),
                "source_metadata_path": str(meta_path),
                "smart_export": bool(host.smart_export_var.get()),
                "exported_plate_ids": list(exported_metadata.keys()),
                "exported_image_names": xml_image_names,
                "zip_path": str(out_zip),
                "xml_path": str(out_xml),
            }
            try:
                host._atomic_write_json(export_dir / "review_manifest.json", review_manifest)
            except Exception as e:
                logger.debug(f"Nie udało się zapisać review_manifest.json: {e}")

            summary = host._build_step3_local_success_message(
                "Przygotowano zestaw CVAT.",
                out_zip,
                "Otwórz CVAT, wykonaj poprawki i wróć do Z3 / PZ3 -> Import ręcznych poprawek.",
            )
            if not zip_ok:
                summary += f"\n\nUwaga: ZIP zgłosił ostrzeżenie: {zip_msg}"
            host._set_console_text(host.export_console, summary)
            host._show_pz3_operation_summary_modal(
                "Eksport CVAT zakończony",
                summary,
                tone=("warning" if not zip_ok else "success"),
            )
            try:
                host._log(
                    host.export_console,
                    f"[INFO] Zestaw do korekty CVAT zapisano w katalogu projektu: {export_dir}",
                    "INFO"
                )
                host._log(
                    host.export_console,
                    f"[INFO] review_manifest.json zapisano w: {export_dir / 'review_manifest.json'}",
                    "INFO"
                )
            except Exception:
                pass
            try:
                host._update_step3_finish_button_state()
            except Exception:
                pass

            if host.smart_export_var.get() and source_meta.exists():
                source_meta.unlink()

    except Exception as e:
        host._set_console_text(host.export_console, f"❌ BŁĄD EKSPORTU CVAT:\n{e}")


def run_cvat_import(host) -> None:
    xml_in = Path(host.import_cvat_xml_var.get().strip())
    preview_dir_raw = str(host.preview_dir_var.get() or "").strip()
    preview_dir = Path(preview_dir_raw) if preview_dir_raw else None
    meta_path = preview_dir / "metadata.json" if preview_dir is not None else None
    images_dir = preview_dir / "images" if preview_dir is not None else None

    if not xml_in.exists():
        host._set_console_text(host.import_console, "❌ BŁĄD: Wybrany plik XML nie istnieje.")
        return
    if preview_dir is None or not preview_dir.exists() or not preview_dir.is_dir():
        host._set_console_text(
            host.import_console,
            (
                "❌ BŁĄD: Brak aktywnego wyniku PZ2. Utwórz go w [Z3]/[PZ1] przyciskiem "
                "„Wyodrębnij tablice do PZ2”, a potem przejdź do [PZ2]."
            )
        )
        return
    if meta_path is None or not meta_path.exists():
        host._set_console_text(
            host.import_console,
            f"❌ BŁĄD: W aktywnym wyniku PZ2 brakuje metadata.json:\n{preview_dir}"
        )
        return
    if images_dir is None or not images_dir.exists() or not images_dir.is_dir():
        host._set_console_text(
            host.import_console,
            f"❌ BŁĄD: W aktywnym wyniku PZ2 brakuje katalogu images/:\n{preview_dir}"
        )
        return

    host._set_console_text(host.import_console, "⌛ Wczytywanie danych z XML...")

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        if not isinstance(metadata, dict):
            raise RuntimeError("metadata.json ma nieprawidłowy format.")

        from ..cvat_tools.cvat_character_importer import CVATCharacterImporter

        importer = CVATCharacterImporter()
        import_ok, import_msg, import_stats = importer.import_annotations(xml_in)
        if not import_ok:
            host._set_console_text(host.import_console, f"❌ BŁĄD IMPORTU:\n{import_msg}")
            return

        imported_annotations = importer.get_all_annotations()
        xml_images_total = len(imported_annotations)
        review_manifest_id = ""
        review_manifest_mismatch = False
        review_manifest_path = xml_in.parent / "review_manifest.json"
        if review_manifest_path.exists():
            try:
                review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
                if isinstance(review_manifest, dict):
                    review_manifest_id = str(review_manifest.get("review_manifest_id", "") or "").strip()
                    source_preview_dir = str(review_manifest.get("source_preview_dir", "") or "").strip()
                    if source_preview_dir:
                        try:
                            review_manifest_mismatch = Path(source_preview_dir).resolve() != preview_dir.resolve()
                        except Exception:
                            review_manifest_mismatch = str(Path(source_preview_dir)) != str(preview_dir)
            except Exception as e:
                logger.debug(f"Nie udało się odczytać review_manifest.json przy imporcie CVAT: {e}")
        matched_images = 0
        skipped_empty = 0
        skipped_unknown = 0
        updated = 0
        changed_characters = 0
        import_batch_id = f"cvat_import_{Path(xml_in).stem}"
        try:
            import datetime
            import_batch_id = f"cvat_import_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        except Exception:
            pass

        matched_plate_ids = []
        report_items = []

        for image_name, detections in imported_annotations.items():
            pid = Path(str(image_name or "")).stem
            if pid not in metadata:
                skipped_unknown += 1
                continue

            matched_images += 1
            clean_chars = []
            for det in list(detections or []):
                symbol = host._sanitize_preview_char_symbol(getattr(det, "character", ""))
                bbox = host._normalize_preview_char_bbox(getattr(det, "bbox", None))
                if not symbol or bbox is None:
                    continue
                try:
                    confidence = float(getattr(det, "confidence", 1.0) or 1.0)
                except Exception:
                    confidence = 1.0
                clean_chars.append(
                    {
                        "character": symbol,
                        "bbox": bbox,
                        "confidence": confidence,
                        "method": "cvat_manual",
                        "source_tag": "manual",
                        "source_kind": "cvat_manual",
                        "source_batch_id": import_batch_id,
                    }
                )

            if not clean_chars:
                skipped_empty += 1
                report_items.append(
                    {
                        "plate_id": pid,
                        "image_name": image_name,
                        "status": "skipped_empty",
                        "characters": 0,
                    }
                )
                continue

            clean_chars = host._sort_character_records_by_x(clean_chars)
            target_data = metadata.get(pid, {})
            previous_chars = list(target_data.get("characters", [])) if isinstance(target_data.get("characters"), list) else []
            previous_status = str(target_data.get("status", "unknown") or "unknown").strip().lower()
            host._update_preview_plate_layout_metadata(target_data, clean_chars)
            clean_chars = host._annotate_preview_character_reading_positions(clean_chars, data=target_data)
            target_data["characters"] = clean_chars
            target_data["fusion_strategy"] = "manual_correction"
            target_data["fusion_details"] = {
                "source": "cvat_import",
                "import_batch_id": import_batch_id,
                "xml_path": str(xml_in),
            }
            target_data["import_source_xml"] = str(xml_in)
            status_probe = dict(target_data)
            status_probe["plate_id"] = str(pid)
            status_now = host._derive_preview_status_from_data(status_probe, clean_chars)
            if (
                status_now == "perfect"
                and previous_status == "perfect"
                and len(clean_chars) != len(previous_chars)
                and not host._get_preview_expected_texts(status_probe)
            ):
                status_now = "needs_fix"
            target_data["status"] = status_now
            host._ensure_plate_source_metadata(
                target_data,
                plate_id=str(pid),
                meta_path=meta_path,
                default_bucket="cvat_manual",
                default_origin="cvat_import",
                import_batch_id=import_batch_id,
                review_manifest_id=review_manifest_id,
                modified_by="human",
            )
            metadata[pid] = target_data
            updated += 1
            changed_characters += len(clean_chars)
            matched_plate_ids.append(str(pid))
            report_items.append(
                {
                    "plate_id": pid,
                    "image_name": image_name,
                    "status": str(status_now),
                    "characters": len(clean_chars),
                }
            )

        if matched_images == 0:
            host._set_console_text(
                host.import_console,
                "❌ BŁĄD IMPORTU:\nWybrany XML nie pasuje do aktywnego wyniku PZ2. "
                "Nie znaleziono żadnej wspólnej tablicy po nazwie pliku."
            )
            return
        if updated == 0:
            host._set_console_text(
                host.import_console,
                "❌ BŁĄD IMPORTU:\nXML pasuje do aktywnego runu, ale żadna tablica nie zawiera poprawnych znaków do zapisania."
            )
            return

        backup_path = host._backup_json_before_import(meta_path)
        host._atomic_write_json(meta_path, metadata)
        host._apply_preview_metadata_update(metadata, preserve_selection=True)
        host._loaded_meta_path = meta_path
        try:
            host._loaded_meta_mtime = meta_path.stat().st_mtime
        except Exception:
            host._loaded_meta_mtime = None
        host._set_preview_import_focus(
            matched_plate_ids,
            import_batch_id=import_batch_id,
            activate=True,
        )

        report_payload = {
            "import_batch_id": import_batch_id,
            "source_xml": str(xml_in),
            "preview_dir": str(preview_dir),
            "matched_images": int(matched_images),
            "updated_images": int(updated),
            "skipped_empty": int(skipped_empty),
            "skipped_unknown": int(skipped_unknown),
            "xml_images_total": int(xml_images_total),
            "xml_characters_total": int(import_stats.get("characters", 0) or 0) if isinstance(import_stats, dict) else 0,
            "saved_characters": int(changed_characters),
            "review_manifest_id": review_manifest_id,
            "review_manifest_mismatch": bool(review_manifest_mismatch),
            "matched_plate_ids": list(matched_plate_ids),
            "items": report_items,
        }
        report_path = preview_dir / f"{import_batch_id}_report.json"
        try:
            host._atomic_write_json(report_path, report_payload)
        except Exception as e:
            logger.debug(f"Nie udało się zapisać raportu importu CVAT: {e}")

        import_summary = host._build_step3_local_success_message(
            "Zaimportowano ręczne poprawki z CVAT.",
            preview_dir,
            "Sprawdź tablice na liście po lewej albo przejdź do Budowy datasetu w Z3 / PZ3.",
        )
        final_import_summary = import_summary
        host._set_console_text(host.import_console, import_summary)
        try:
            stored_dir = host._store_current_preview_in_manual_char_pool(metadata)
            final_import_summary = host._build_step3_local_success_message(
                "Zaimportowano ręczne poprawki z CVAT.",
                stored_dir,
                "Sprawdź tablice na liście po lewej albo przejdź do Budowy datasetu w Z3 / PZ3.",
            )
            host._set_console_text(host.import_console, final_import_summary)
        except Exception as e:
            logger.debug(f"Nie udało się zapisać zestawu do manual_char_pool: {e}")
        host._show_pz3_operation_summary_modal(
            "Import CVAT zakończony",
            final_import_summary,
            tone="success",
        )
        try:
            host._update_step3_finish_button_state()
        except Exception:
            pass

    except Exception as e:
        host._set_console_text(host.import_console, f"❌ BŁĄD IMPORTU:\n{e}")


def build_cvat_tab(host, parent, nav_button_width, perfect_strategy_labels, gold_source_labels):
    self = host
    NAV_BUTTON_WIDTH = nav_button_width
    PERFECT_STRATEGY_LABELS = perfect_strategy_labels
    GOLD_SOURCE_LABELS = gold_source_labels
    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    card_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    card_bg = panel_alt
    card_hover_bg = palette.get("button_hover", panel_alt)
    card_active_bg = blend_hex_colors(panel_alt, card_hover_bg, 0.42)
    card_fg = palette.get("fg", "#f3f3f3")
    card_muted = palette.get("muted", "#c7c7c7")

    def ensure_wrap(widget, container, *, padding=28, min_wrap=180):
        try:
            self.app.ensure_adaptive_wrap(widget, container=container, padding=padding, min_wrap=min_wrap)
        except Exception:
            pass

    def _get_pz3_card_palette() -> dict:
        current_palette = getattr(self.app, "palette", {})
        current_card_bg = current_palette.get("panel_alt", current_palette.get("panel", "#252526"))
        current_hover_bg = current_palette.get("button_hover", current_card_bg)
        current_fg = current_palette.get("fg", "#f3f3f3")
        is_light_card = self._get_readable_text_color(current_card_bg, preferred="#111111") == "#111111"
        current_card_muted = blend_hex_colors(
            current_fg,
            current_card_bg,
            0.22 if is_light_card else 0.34,
        )
        return {
            "card_bg": current_card_bg,
            "card_hover_bg": current_hover_bg,
            "card_active_bg": blend_hex_colors(current_card_bg, current_hover_bg, 0.42),
            "card_border": current_palette.get("panel_border", current_palette.get("border", "#3c3c3c")),
            "card_fg": current_fg,
            "card_muted": current_card_muted,
        }

    def _get_pz3_action_card_palette() -> dict:
        current_palette = getattr(self.app, "palette", {})
        base = _get_pz3_card_palette()
        success = current_palette.get("success", current_palette.get("accent", "#2d7d46"))
        surface_success = current_palette.get(
            "surface_success",
            blend_hex_colors(success, base["card_bg"], 0.86),
        )
        action_bg = blend_hex_colors(surface_success, base["card_bg"], 0.32)
        action_hover_bg = blend_hex_colors(success, action_bg, 0.82)
        action_active_bg = blend_hex_colors(success, action_bg, 0.72)
        action_border = blend_hex_colors(success, base["card_border"], 0.28)
        action_fg = self._get_readable_text_color(action_bg, preferred=success)
        action_muted = blend_hex_colors(action_fg, action_bg, 0.34)
        return {
            "card_bg": action_bg,
            "card_hover_bg": action_hover_bg,
            "card_active_bg": action_active_bg,
            "card_border": action_border,
            "card_fg": action_fg,
            "card_muted": action_muted,
            "card_accent": success,
        }

    def make_pz3_card(parent_frame, badge_text: str, title_text: str, desc_text: str, *, clickable: bool = False):
        cursor = "hand2" if clickable else "arrow"
        card = tk.Frame(
            parent_frame,
            bd=0,
            highlightthickness=1,
            highlightbackground=card_border,
            highlightcolor=card_border,
            bg=card_bg,
            padx=14,
            pady=12,
            cursor=cursor,
        )
        top_row = tk.Frame(
            card,
            bd=0,
            highlightthickness=0,
            bg=card_bg,
            cursor=cursor,
        )
        top_row.pack(fill=tk.X)
        badge = tk.Label(
            top_row,
            text=badge_text,
            anchor="w",
            justify=tk.LEFT,
            font=("Segoe UI", 9, "bold"),
            bd=0,
            highlightthickness=0,
            bg=card_bg,
            fg=card_muted,
            cursor=cursor,
        )
        badge.pack(side=tk.LEFT, anchor=tk.W)
        state_badge = tk.Label(
            top_row,
            text="",
            anchor="e",
            justify=tk.CENTER,
            font=("Segoe UI", 8, "bold"),
            bd=0,
            highlightthickness=1,
            highlightbackground=card_border,
            highlightcolor=card_border,
            bg=card_bg,
            fg=card_fg,
            padx=8,
            pady=2,
            cursor=cursor,
        )
        title = tk.Label(
            card,
            text=title_text,
            anchor="w",
            justify=tk.LEFT,
            font=("Segoe UI Semibold", 10),
            bd=0,
            highlightthickness=0,
            bg=card_bg,
            fg=card_fg,
            cursor=cursor,
        )
        title.pack(anchor=tk.W, fill=tk.X, pady=(4, 0))
        desc = tk.Label(
            card,
            text=desc_text,
            anchor="w",
            justify=tk.LEFT,
            wraplength=320,
            bd=0,
            highlightthickness=0,
            bg=card_bg,
            fg=card_muted,
            cursor=cursor,
        )
        desc.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))
        self._set_inline_status_label_state(desc, text=desc_text, tone="muted", emphasis=False)
        self._mark_inline_status_contrast_boost(desc)
        ensure_wrap(desc, card, padding=34, min_wrap=220)
        return {
            "frame": card,
            "top_row": top_row,
            "badge": badge,
            "state_badge": state_badge,
            "title": title,
            "desc": desc,
        }

    def _normalize_pz3_card_badge_text(badge_text: str) -> str:
        text = str(badge_text or "").strip()
        if "|" in text:
            text = text.split("|", 1)[0].rstrip()
        return text

    def _refresh_pz3_card_state_badge(card_info, *, selected: bool, current_card_bg: str, card_palette: dict) -> None:
        badge_widget = card_info.get("state_badge")
        if badge_widget is None:
            return

        if not selected:
            try:
                if str(badge_widget.winfo_manager()):
                    badge_widget.pack_forget()
            except Exception:
                pass
            try:
                badge_widget.configure(text="", bg=current_card_bg)
            except Exception:
                pass
            return

        badge_fill = blend_hex_colors(
            palette.get("success", palette.get("accent", "#2d7d46")),
            current_card_bg,
            0.18,
        )
        badge_fg = self._get_readable_text_color(
            badge_fill,
            preferred=palette.get("success", card_palette.get("card_fg", "#f3f3f3")),
        )
        badge_outline = blend_hex_colors(
            palette.get("success", palette.get("accent", "#2d7d46")),
            current_card_bg,
            0.48,
        )
        try:
            badge_widget.configure(
                text="Wybrano",
                bg=badge_fill,
                fg=badge_fg,
                highlightbackground=badge_outline,
                highlightcolor=badge_outline,
            )
            if not str(badge_widget.winfo_manager()):
                badge_widget.pack(side=tk.RIGHT, anchor=tk.NE)
        except Exception:
            pass

    self._get_pz3_card_palette = _get_pz3_card_palette
    self._get_pz3_action_card_palette = _get_pz3_action_card_palette
    self._normalize_pz3_card_badge_text = _normalize_pz3_card_badge_text
    self._refresh_pz3_card_state_badge = _refresh_pz3_card_state_badge

    parent.grid_rowconfigure(0, weight=1)
    parent.grid_rowconfigure(1, weight=0)
    parent.grid_columnconfigure(0, weight=1)

    content_frame = ttk.Frame(parent, style="Panel.TFrame")
    content_frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=(10, 0))
    content_frame.grid_rowconfigure(0, weight=1)
    self._pz3_content_inset = 14
    self._pz3_content_max_width = 760
    self._pz3_status_panel_width = 330
    self._pz3_export_host_width = self._pz3_content_max_width + (2 * self._pz3_content_inset)
    content_frame.grid_columnconfigure(0, weight=1, minsize=620)
    content_frame.grid_columnconfigure(1, weight=0, minsize=self._pz3_status_panel_width)

    export_shell_border = blend_hex_colors(
        palette.get("accent", "#4f8de3"),
        card_border,
        0.62,
    )
    export_shell_fill = blend_hex_colors(
        palette.get("surface_info", panel_bg),
        panel_bg,
        0.80,
    )

    export_shell = tk.Frame(
        content_frame,
        bg=export_shell_border,
        bd=0,
        highlightthickness=0,
        padx=1,
        pady=1,
        width=self._pz3_export_host_width,
    )
    export_shell.grid(row=0, column=0, sticky="nsew")
    export_shell.grid_rowconfigure(0, weight=1)
    export_shell.grid_columnconfigure(0, weight=1)
    self.pz3_export_shell = export_shell

    status_shell_border = blend_hex_colors(
        palette.get("success", "#2fa36b"),
        card_border,
        0.56,
    )
    status_shell_fill = blend_hex_colors(
        palette.get("surface_success", panel_bg),
        panel_bg,
        0.84,
    )
    status_shell = tk.Frame(
        content_frame,
        bg=status_shell_border,
        bd=0,
        highlightthickness=0,
        padx=1,
        pady=1,
        width=self._pz3_status_panel_width,
    )
    status_shell.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
    status_shell.grid_propagate(False)
    status_shell.grid_rowconfigure(0, weight=1)
    status_shell.grid_columnconfigure(0, weight=1)
    self.pz3_status_shell = status_shell

    status_shell_inner = tk.Frame(
        status_shell,
        bg=status_shell_fill,
        bd=0,
        highlightthickness=0,
    )
    status_shell_inner.grid(row=0, column=0, sticky="nsew")
    status_shell_inner.grid_rowconfigure(0, weight=1)
    status_shell_inner.grid_columnconfigure(0, weight=1)
    self.pz3_status_shell_inner = status_shell_inner
    self._pz3_status_section_persistent = True

    export_shell_inner = tk.Frame(
        export_shell,
        bg=export_shell_fill,
        bd=0,
        highlightthickness=0,
    )
    export_shell_inner.grid(row=0, column=0, sticky="nsew")
    export_shell_inner.grid_rowconfigure(0, weight=1)
    export_shell_inner.grid_columnconfigure(0, weight=1)
    self.pz3_export_shell_inner = export_shell_inner

    export_host = ttk.Frame(export_shell_inner, style="Panel.TFrame", width=self._pz3_export_host_width)
    export_host.grid(row=0, column=0, sticky="nsew")
    export_host.grid_rowconfigure(0, weight=1)
    export_host.grid_columnconfigure(0, weight=1)
    self.pz3_export_host = export_host

    self.cvat_export_canvas = tk.Canvas(
        export_host,
        bg=export_shell_fill,
        bd=0,
        highlightthickness=0,
    )
    self.cvat_export_canvas.grid(row=0, column=0, sticky="nsew")

    self.cvat_export_scrollbar = WebSlimScrollbar(
        export_host,
        command=self.cvat_export_canvas.yview,
    )
    self.cvat_export_scrollbar.grid(row=0, column=1, sticky="ns", padx=(6, 0))
    self.cvat_export_canvas.configure(yscrollcommand=self.cvat_export_scrollbar.set)

    self.cvat_export_content = ttk.Frame(self.cvat_export_canvas, style="Panel.TFrame")
    self.cvat_export_content_window = self.cvat_export_canvas.create_window(
        (self._pz3_content_inset, 0),
        window=self.cvat_export_content,
        anchor="nw",
    )
    self.cvat_export_content.bind("<Configure>", self._sync_cvat_export_scrollregion, add="+")
    self.cvat_export_canvas.bind("<Configure>", self._sync_cvat_export_canvas_width, add="+")

    settings_col = ttk.Frame(self.cvat_export_content, style="Panel.TFrame")
    settings_col.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 10))

    overview_section = ttk.Frame(settings_col, style="Panel.TFrame")
    overview_section.pack(fill=tk.X)
    self._pz3_overview_section = overview_section
    self.dataset_title_lbl = SectionHeaderLabel(
        overview_section,
        self.app,
        text="PZ3: dataset znaków",
    )
    self.dataset_title_lbl.pack(anchor=tk.W, fill=tk.X)

    self.dataset_intro_lbl = tk.Label(
        overview_section,
        text=(
            "PZ3 zamyka pracę nad znakami: sprawdza materiał z PZ2, pozwala ustalić zakres "
            "poprawnych tablic i tworzy źródłowy dataset znaków. Wariant treningowy oraz split "
            "przygotujesz później w Z4."
        ),
        anchor="w",
        justify=tk.LEFT,
        wraplength=900,
        bd=0,
        highlightthickness=0,
    )
    self.dataset_intro_lbl.pack(anchor=tk.W, fill=tk.X, pady=(4, 8))
    self._set_inline_status_label_state(self.dataset_intro_lbl, text=self.dataset_intro_lbl.cget("text"), tone="muted", emphasis=False)
    ensure_wrap(self.dataset_intro_lbl, overview_section, padding=28, min_wrap=280)

    self.pz3_flow_lbl = tk.Label(
        overview_section,
        text="Flow: sprawdź materiał -> utwórz źródłowy dataset znaków -> opcjonalnie zawęź zakres lub użyj CVAT -> przejdź do Z4",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self._set_inline_status_label_state(self.pz3_flow_lbl, text=self.pz3_flow_lbl.cget("text"), tone="success", emphasis=True)
    self.pz3_flow_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))

    card_dataset = {
        "frame": None,
        "top_row": None,
        "badge": None,
        "state_badge": None,
        "title": None,
        "desc": None,
    }
    card_review = {
        "frame": None,
        "top_row": None,
        "badge": None,
        "state_badge": None,
        "title": None,
        "desc": None,
    }
    self._pz3_dataset_card = None
    self._pz3_review_card = None

    dataset_section = ttk.Frame(settings_col, style="Panel.TFrame")
    dataset_section.pack(fill=tk.X, pady=(18, 0))
    self.pz3_dataset_section = dataset_section
    self.dataset_export_title_lbl = SectionHeaderLabel(
        dataset_section,
        self.app,
        text="Budowa datasetu znaków",
    )
    self.dataset_export_title_lbl.pack(anchor=tk.W, fill=tk.X)

    self.cvat_option2_title_lbl = tk.Label(
        dataset_section,
        text="Stan materiału z PZ2",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.cvat_option2_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))
    self._set_inline_status_label_state(self.cvat_option2_title_lbl, tone="success", emphasis=True)

    self.cvat_option2_desc_lbl = tk.Label(
        dataset_section,
        text=(
            "Dataset powstaje z wyodrębnionych tablic, które w PZ2 mają status perfect. "
            "Jeśli brakuje tablic lub znaków, wróć do PZ2 i popraw boxy przed eksportem."
        ),
        wraplength=900,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.cvat_option2_desc_lbl.pack(anchor=tk.W, fill=tk.X, pady=(4, 14))
    self._set_inline_status_label_state(self.cvat_option2_desc_lbl, tone="muted", emphasis=False)
    ensure_wrap(self.cvat_option2_desc_lbl, dataset_section, padding=28, min_wrap=280)

    dataset_source_lf = ttk.LabelFrame(dataset_section, text="", padding=8)
    self._pz3_dataset_source_lf = dataset_source_lf
    dataset_source_lf.pack(fill=tk.X, pady=(0, 14))

    self.pz3_dataset_source_intro_lbl = tk.Label(
        dataset_source_lf,
        text="",
        wraplength=900,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self._set_inline_status_label_state(self.pz3_dataset_source_intro_lbl, tone="muted", emphasis=False)
    ensure_wrap(self.pz3_dataset_source_intro_lbl, dataset_source_lf, padding=28, min_wrap=280)

    self.pz3_source_preview_status_lbl = tk.Label(
        dataset_source_lf,
        text="Wynik PZ2: nie sprawdzono.",
        wraplength=900,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.pz3_source_preview_status_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 4))
    self._set_inline_status_label_state(self.pz3_source_preview_status_lbl, tone="muted", emphasis=False)
    ensure_wrap(self.pz3_source_preview_status_lbl, dataset_source_lf, padding=28, min_wrap=280)

    self.pz3_source_pool_status_lbl = tk.Label(
        dataset_source_lf,
        text="Materiał do datasetu: nie sprawdzono.",
        wraplength=900,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.pz3_source_pool_status_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 4))
    self._set_inline_status_label_state(self.pz3_source_pool_status_lbl, tone="muted", emphasis=False)
    ensure_wrap(self.pz3_source_pool_status_lbl, dataset_source_lf, padding=28, min_wrap=280)

    self.pz3_source_next_status_lbl = tk.Label(
        dataset_source_lf,
        text="Następny krok: utwórz źródłowy dataset znaków. Zakres poniżej zmieniaj tylko świadomie.",
        wraplength=900,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.pz3_source_next_status_lbl.pack(anchor=tk.W, fill=tk.X)
    self._set_inline_status_label_state(self.pz3_source_next_status_lbl, tone="muted", emphasis=False)
    ensure_wrap(self.pz3_source_next_status_lbl, dataset_source_lf, padding=28, min_wrap=280)

    dataset_source_cards = ttk.Frame(dataset_source_lf, style="Panel.TFrame")
    self._pz3_dataset_source_cards = dataset_source_cards
    dataset_source_cards.grid_rowconfigure(0, weight=1)
    dataset_source_cards.grid_columnconfigure(0, weight=1)
    dataset_source_cards.grid_columnconfigure(1, weight=1)

    dataset_source_perfect_card = make_pz3_card(
        dataset_source_cards,
        "PZ2",
        "Materiał z PZ2",
        "Źródłem są wyodrębnione tablice z PZ2 oznaczone jako perfect.",
        clickable=True,
    )
    self._pz3_dataset_source_perfect_card = dataset_source_perfect_card
    dataset_source_perfect_card["frame"].grid(row=0, column=0, sticky="nsew", padx=(0, 6))

    dataset_source_existing_card = make_pz3_card(
        dataset_source_cards,
        "Z4",
        "Warianty w Z4",
        "Gotowe datasety i ich warianty wybierzesz w Z4.",
        clickable=True,
    )
    self._pz3_dataset_source_existing_card = dataset_source_existing_card
    dataset_source_existing_card["frame"].grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    self.pz3_existing_dataset_panel = ttk.Frame(dataset_source_lf, style="Panel.TFrame")

    self.pz3_existing_dataset_title_lbl = tk.Label(
        self.pz3_existing_dataset_panel,
        text="Warianty datasetu obsługuje Z4",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.pz3_existing_dataset_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self._set_inline_status_label_state(self.pz3_existing_dataset_title_lbl, tone="default", emphasis=False)

    self.pz3_existing_dataset_desc_lbl = tk.Label(
        self.pz3_existing_dataset_panel,
        text="PZ3 tworzy tylko źródłowy dataset znaków. Warianty treningowe i split obsługuje Z4.",
        wraplength=900,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.pz3_existing_dataset_desc_lbl.pack(anchor=tk.W, fill=tk.X, pady=(4, 8))
    self._set_inline_status_label_state(self.pz3_existing_dataset_desc_lbl, tone="muted", emphasis=False)
    ensure_wrap(self.pz3_existing_dataset_desc_lbl, self.pz3_existing_dataset_panel, padding=28, min_wrap=260)

    existing_dataset_row = ttk.Frame(self.pz3_existing_dataset_panel, style="Panel.TFrame")
    existing_dataset_row.pack(fill=tk.X)
    existing_dataset_row.columnconfigure(0, weight=1)

    self.pz3_existing_dataset_entry = ttk.Entry(
        existing_dataset_row,
        textvariable=self.pz3_existing_dataset_var,
    )
    self.pz3_existing_dataset_entry.grid(row=0, column=0, sticky="ew")

    self.pz3_existing_dataset_pick_btn = ttk.Button(
        existing_dataset_row,
        text="Wybierz",
        command=self._pick_pz3_existing_dataset_dir,
        style="WorkflowCard.TButton",
    )
    self.pz3_existing_dataset_pick_btn.grid(row=0, column=1, sticky="e", padx=(6, 0))

    self.pz3_existing_dataset_project_btn = ttk.Button(
        existing_dataset_row,
        text="Ostatni projektowy",
        command=self._use_preferred_pz3_dataset_dir,
        style="WorkflowCard.TButton",
    )
    self.pz3_existing_dataset_project_btn.grid(row=0, column=2, sticky="e", padx=(6, 0))

    self.pz3_existing_dataset_status_lbl = tk.Label(
        self.pz3_existing_dataset_panel,
        text="Przejdź do Z4, jeśli chcesz wybrać gotowy wariant treningowy.",
        wraplength=900,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.pz3_existing_dataset_status_lbl.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))
    self._set_inline_status_label_state(self.pz3_existing_dataset_status_lbl, tone="muted", emphasis=False)
    ensure_wrap(self.pz3_existing_dataset_status_lbl, self.pz3_existing_dataset_panel, padding=28, min_wrap=260)

    self.pz3_goldpack_step_title_lbl = tk.Label(
        dataset_section,
        text="Zakres datasetu",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.pz3_goldpack_step_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(2, 0))
    self._set_inline_status_label_state(self.pz3_goldpack_step_title_lbl, tone="success", emphasis=True)

    self.pz3_goldpack_step_desc_lbl = tk.Label(
        dataset_section,
        text=(
            "Domyślnie do datasetu wejdą wszystkie dostępne tablice perfect. Zawężaj strategie "
            "i źródła tylko wtedy, gdy świadomie chcesz ograniczyć materiał treningowy."
        ),
        wraplength=900,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.pz3_goldpack_step_desc_lbl.pack(anchor=tk.W, fill=tk.X, pady=(4, 8))
    self._set_inline_status_label_state(self.pz3_goldpack_step_desc_lbl, tone="muted", emphasis=False)
    ensure_wrap(self.pz3_goldpack_step_desc_lbl, dataset_section, padding=28, min_wrap=260)

    dataset_grid = ttk.Frame(dataset_section, style="Panel.TFrame")
    dataset_grid.pack(fill=tk.X, pady=(0, 14))
    dataset_grid.grid_columnconfigure(0, weight=1)

    self.gold_export_goldpack_lf = ttk.LabelFrame(dataset_grid, text=" Zakres datasetu ", padding=8)
    self.gold_export_goldpack_lf.grid(row=0, column=0, sticky="ew", pady=(0, 2))

    gold_tables_grid = tk.Frame(self.gold_export_goldpack_lf, bd=0, highlightthickness=0)
    self.gold_export_tables_grid = gold_tables_grid
    gold_tables_grid.pack(fill=tk.X)
    gold_tables_grid.grid_columnconfigure(0, weight=1)

    gold_table_columns = (
        {"weight": 0, "minsize": 64, "anchor": "center"},
        {"weight": 6, "minsize": 300, "anchor": "w"},
        {"weight": 1, "minsize": 74, "anchor": "center"},
        {"weight": 1, "minsize": 74, "anchor": "center"},
        {"weight": 2, "minsize": 126, "anchor": "center"},
    )

    def _configure_gold_table_columns(container):
        for idx, spec in enumerate(gold_table_columns):
            container.grid_columnconfigure(
                idx,
                weight=int(spec.get("weight", 0) or 0),
                minsize=int(spec.get("minsize", 0) or 0),
            )

    def _make_gold_table(parent, *, column: int, title_attr: str, title: str, intro: str, headers: tuple[str, ...]):
        palette = getattr(self.app, "palette", {})
        panel_bg = palette.get("panel", "#252526")
        panel_alt = palette.get("panel_alt", panel_bg)
        border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
        shell_bg = blend_hex_colors(panel_alt, panel_bg, 0.38)
        header_bg = blend_hex_colors(palette.get("success", "#2ecc71"), shell_bg, 0.86)
        shell = tk.Frame(
            parent,
            bg=shell_bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
            padx=8,
            pady=10,
        )
        shell.grid(row=column, column=0, sticky="ew", pady=((0, 10) if column == 0 else (0, 0)))
        shell.grid_columnconfigure(0, weight=1)

        title_lbl = tk.Label(
            shell,
            text=title,
            anchor="w",
            justify=tk.LEFT,
            bg=shell_bg,
            bd=0,
            highlightthickness=0,
        )
        title_lbl.grid(row=0, column=0, sticky="ew")
        self._set_inline_status_label_state(title_lbl, tone="success", emphasis=True)
        setattr(self, title_attr, title_lbl)

        intro_lbl = tk.Label(
            shell,
            text=intro,
            anchor="w",
            justify=tk.LEFT,
            wraplength=420,
            bg=shell_bg,
            bd=0,
            highlightthickness=0,
        )
        intro_lbl.grid(row=1, column=0, sticky="ew", pady=(3, 8))
        self._set_inline_status_label_state(intro_lbl, tone="muted", emphasis=False)
        ensure_wrap(intro_lbl, shell, padding=28, min_wrap=220)

        table = tk.Frame(
            shell,
            bg=border,
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
        )
        table.grid(row=2, column=0, sticky="ew")
        table.grid_columnconfigure(0, weight=1)

        header = tk.Frame(table, bg=border, bd=0, highlightthickness=0)
        header.grid(row=0, column=0, sticky="ew")
        _configure_gold_table_columns(header)
        for idx, header_text in enumerate(headers):
            spec = gold_table_columns[idx] if idx < len(gold_table_columns) else {}
            header_lbl = tk.Label(
                header,
                text=header_text,
                anchor=str(spec.get("anchor", "center")),
                justify=tk.LEFT,
                bg=header_bg,
                bd=0,
                highlightthickness=0,
                padx=6,
                pady=5,
            )
            header_lbl.grid(row=0, column=idx, sticky="ew", padx=(0, 1), pady=(0, 1))

        return shell, table

    self.gold_export_filters_lf, gold_strategy_table = _make_gold_table(
        gold_tables_grid,
        column=0,
        title_attr="gold_export_filters_title_lbl",
        title="Strategie kwalifikacji",
        intro="Wybierz, które sposoby uzyskania statusu perfect mają wejść do datasetu.",
        headers=("Wybór", "Strategia", "Tablice", "Znaki", "Status"),
    )

    self.gold_export_sources_lf, gold_source_table = _make_gold_table(
        gold_tables_grid,
        column=1,
        title_attr="gold_export_sources_title_lbl",
        title="Pochodzenie tablic",
        intro="Wybierz pochodzenie rekordów. CVAT jest dołączany automatycznie po imporcie poprawek.",
        headers=("Wybór", "Źródło", "Tablice", "Znaki", "Status"),
    )

    def _add_gold_table_row(
        table,
        row_index: int,
        *,
        bucket_key: str,
        label_text: str,
        selected_getter,
        toggle_command=None,
        locked: bool = False,
        summary: bool = False,
    ):
        palette = getattr(self.app, "palette", {})
        panel_bg = palette.get("panel", "#252526")
        panel_alt = palette.get("panel_alt", panel_bg)
        row_bg = blend_hex_colors(panel_alt, panel_bg, 0.22)
        row_border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
        if summary:
            row_bg = blend_hex_colors(palette.get("accent", "#4f8de3"), panel_bg, 0.88)
            row_border = blend_hex_colors(palette.get("accent", "#4f8de3"), row_border, 0.42)
        row_cursor = "arrow" if locked or summary else "hand2"
        summary_font = None
        if summary:
            try:
                summary_font = self._get_preview_legend_font(9, "bold")
            except Exception:
                summary_font = None
        row = tk.Frame(
            table,
            bg=row_bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=row_border,
            highlightcolor=row_border,
            cursor=row_cursor,
        )
        row.grid(row=row_index, column=0, sticky="ew", pady=(0, 1))
        _configure_gold_table_columns(row)

        indicator = tk.Canvas(
            row,
            width=16,
            height=16,
            bd=0,
            highlightthickness=0,
            cursor=row_cursor,
        )
        indicator.grid(row=0, column=0, sticky="", padx=6, pady=6)

        label = tk.Label(
            row,
            text=label_text,
            anchor="w",
            justify=tk.LEFT,
            bg=row_bg,
            bd=0,
            highlightthickness=0,
            cursor=row_cursor,
            padx=4,
            pady=6,
            font=summary_font,
        )
        label.grid(row=0, column=1, sticky="ew")

        plate_lbl = tk.Label(row, text="0", anchor="center", bg=row_bg, bd=0, highlightthickness=0, padx=6, pady=6, font=summary_font)
        plate_lbl.grid(row=0, column=2, sticky="ew")

        char_lbl = tk.Label(row, text="0", anchor="center", bg=row_bg, bd=0, highlightthickness=0, padx=6, pady=6, font=summary_font)
        char_lbl.grid(row=0, column=3, sticky="ew")

        badge = tk.Label(row, text="", anchor="center", bg=row_bg, bd=0, highlightthickness=0, padx=8, pady=3, font=summary_font)
        badge.grid(row=0, column=4, sticky="ew", padx=(4, 8), pady=5)

        row_info = {
            "kind": "summary" if summary else "check",
            "frame": row,
            "indicator": indicator,
            "label": label,
            "bucket_key": bucket_key,
            "base_label": label_text,
            "plate_label": plate_lbl,
            "char_label": char_lbl,
            "badge": badge,
            "extra_widgets": [plate_lbl, char_lbl],
            "selected_getter": selected_getter,
            "hovered": False,
            "locked": bool(locked),
            "base_bg": row_bg,
            "hover_bg": blend_hex_colors(palette.get("accent", "#4f8de3"), panel_bg, 0.88),
            "border_color": row_border,
            "summary": bool(summary),
        }

        if callable(toggle_command) and not locked and not summary:
            for widget in (row, indicator, label, plate_lbl, char_lbl, badge):
                widget.bind("<Button-1>", toggle_command)
                widget.bind("<Enter>", lambda _event, info=row_info: self._set_selection_row_hover(info, True))
                widget.bind("<Leave>", lambda _event, info=row_info: self._set_selection_row_hover(info, False))

        return row_info

    self.gold_export_filter_rows = []
    for row_idx, (bucket_key, filter_label, filter_var) in enumerate((
        ("ocr_exact", PERFECT_STRATEGY_LABELS["ocr_exact"], self.gold_include_ocr_exact_var),
        ("yolo_exact", PERFECT_STRATEGY_LABELS["yolo_exact"], self.gold_include_yolo_exact_var),
        ("ocr_yolo_rescue", PERFECT_STRATEGY_LABELS["ocr_yolo_rescue"], self.gold_include_ocr_yolo_rescue_var),
        ("yolo_box_ocr", PERFECT_STRATEGY_LABELS["yolo_box_ocr"], self.gold_include_yolo_box_ocr_var),
        ("other_perfect", PERFECT_STRATEGY_LABELS["other_perfect"], self.gold_include_other_perfect_var),
    ), start=1):
        def _toggle_gold_filter(_event=None, target_var=filter_var):
            target_var.set(not bool(target_var.get()))
            self._on_gold_export_filter_change()

        self.gold_export_filter_rows.append(
            _add_gold_table_row(
                gold_strategy_table,
                row_idx,
                bucket_key=bucket_key,
                label_text=filter_label,
                selected_getter=(lambda target_var=filter_var: bool(target_var.get())),
                toggle_command=_toggle_gold_filter,
            )
        )

    self.gold_export_filter_total_row = _add_gold_table_row(
        gold_strategy_table,
        len(self.gold_export_filter_rows) + 1,
        bucket_key="__strategy_total__",
        label_text="Suma wybranych",
        selected_getter=lambda: True,
        locked=True,
        summary=True,
    )

    self.gold_export_source_rows = []
    for row_idx, (bucket_key, filter_label, filter_var) in enumerate((
        ("auto_preview", GOLD_SOURCE_LABELS["auto_preview"], self.gold_include_source_auto_var),
        ("local_manual", GOLD_SOURCE_LABELS["local_manual"], self.gold_include_source_local_manual_var),
    ), start=1):
        def _toggle_gold_source_filter(_event=None, target_var=filter_var):
            target_var.set(not bool(target_var.get()))
            self._on_gold_export_source_change()

        self.gold_export_source_rows.append(
            _add_gold_table_row(
                gold_source_table,
                row_idx,
                bucket_key=bucket_key,
                label_text=filter_label,
                selected_getter=(lambda target_var=filter_var: bool(target_var.get())),
                toggle_command=_toggle_gold_source_filter,
            )
        )

    self.gold_export_source_rows.append(
        _add_gold_table_row(
            gold_source_table,
            3,
            bucket_key="cvat_manual",
            label_text=GOLD_SOURCE_LABELS["cvat_manual"],
            selected_getter=lambda: True,
            locked=True,
        )
    )

    self.gold_export_source_total_row = _add_gold_table_row(
        gold_source_table,
        len(self.gold_export_source_rows) + 1,
        bucket_key="__source_total__",
        label_text="Suma wybranych",
        selected_getter=lambda: True,
        locked=True,
        summary=True,
    )

    self._apply_gold_export_filter_check_style()
    self._apply_gold_export_source_check_style()
    self._refresh_gold_export_filter_labels()
    self._refresh_gold_export_source_labels()

    self.gold_export_scope_lbl = tk.Label(
        self.gold_export_goldpack_lf,
        text="Zakres datasetu: wszystkie dostępne tablice perfect.",
        justify=tk.LEFT,
        wraplength=900,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.gold_export_scope_lbl.pack(anchor=tk.W, fill=tk.X, pady=(10, 0))
    self._set_inline_status_label_state(
        self.gold_export_scope_lbl,
        text=self.gold_export_scope_lbl.cget("text"),
        tone="muted",
        emphasis=False,
    )
    ensure_wrap(self.gold_export_scope_lbl, self.gold_export_goldpack_lf, padding=28, min_wrap=320)
    self._refresh_gold_export_scope_label()

    self.gold_cvat_source_status_lbl = tk.Label(
        self.gold_export_sources_lf,
        text="Poprawki CVAT po imporcie są dołączane jako ręczna korekta.",
        justify=tk.LEFT,
        wraplength=420,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.gold_cvat_source_status_lbl.grid(row=3, column=0, sticky="ew", pady=(8, 0))
    self._set_inline_status_label_state(
        self.gold_cvat_source_status_lbl,
        text=self.gold_cvat_source_status_lbl.cget("text"),
        tone="muted",
        emphasis=False,
    )
    ensure_wrap(self.gold_cvat_source_status_lbl, self.gold_export_sources_lf, padding=28, min_wrap=220)
    self._refresh_gold_export_source_labels()

    self.gold_export_sources_hint_lbl = tk.Label(
        self.gold_export_sources_lf,
        text="To ustawienie filtruje materiał przed eksportem datasetu. Jeśli nie masz pewności, zostaw domyślne wartości.",
        justify=tk.LEFT,
        wraplength=420,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.gold_export_sources_hint_lbl.grid(row=4, column=0, sticky="ew", pady=(8, 0))
    self._set_inline_status_label_state(
        self.gold_export_sources_hint_lbl,
        text=self.gold_export_sources_hint_lbl.cget("text"),
        tone="muted",
        emphasis=False,
    )
    ensure_wrap(self.gold_export_sources_hint_lbl, self.gold_export_sources_lf, padding=28, min_wrap=220)

    optional_cvat_lf = ttk.LabelFrame(dataset_section, text=" Opcjonalnie: korekta w CVAT ", padding=8)
    optional_cvat_lf.pack(fill=tk.X, pady=(18, 0))
    self.pz3_optional_cvat_lf = optional_cvat_lf

    self.pz3_optional_cvat_intro_lbl = tk.Label(
        optional_cvat_lf,
        text=(
            "To pętla pomocnicza: wyślij cropy tablic do CVAT, popraw boxy znaków poza aplikacją "
            "i wczytaj poprawiony XML z powrotem w tej samej sekcji."
        ),
        wraplength=900,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.pz3_optional_cvat_intro_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
    self._set_inline_status_label_state(self.pz3_optional_cvat_intro_lbl, tone="muted", emphasis=False)
    ensure_wrap(self.pz3_optional_cvat_intro_lbl, optional_cvat_lf, padding=28, min_wrap=260)

    self.gold_export_sources_import_host = ttk.Frame(optional_cvat_lf, style="Panel.TFrame")
    self.gold_export_sources_import_host.pack(fill=tk.X, pady=(12, 0))

    self._pz3_cvat_import_expanded = bool(getattr(self, "_pz3_cvat_import_expanded", False))
    self._pz3_cvat_import_card_state = {"hovered": False}
    cvat_import_card = make_pz3_card(
        self.gold_export_sources_import_host,
        "IMPORT",
        "Wczytaj poprawki z CVAT",
        "Wczytuje XML z korektami znaków i dołącza go do zakresu datasetu.",
        clickable=True,
    )
    self._pz3_cvat_import_card = cvat_import_card
    cvat_import_card["frame"].pack(fill=tk.X, pady=(12, 0))

    self.pz3_cvat_import_section = ttk.Frame(self.gold_export_sources_import_host, style="Panel.TFrame")

    self.cvat_import_title_lbl = tk.Label(
        self.pz3_cvat_import_section,
        text="Import kompatybilnego XML z CVAT",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.cvat_import_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(12, 0))
    self._set_inline_status_label_state(self.cvat_import_title_lbl, tone="default", emphasis=False)

    self.cvat_import_desc_lbl = tk.Label(
        self.pz3_cvat_import_section,
        text=(
            "Wskaż annotations.xml po poprawkach w CVAT. To musi być XML dla cropów tablic "
            "wyeksportowanych z PZ3, z boxami pojedynczych znaków wewnątrz każdej tablicy. "
            "Import aktualizuje aktywny wynik PZ2 i automatycznie zasila dataset poprawkami CVAT."
        ),
        wraplength=540,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.cvat_import_desc_lbl.pack(anchor=tk.W, fill=tk.X, pady=(4, 8))
    self._set_inline_status_label_state(self.cvat_import_desc_lbl, tone="muted", emphasis=False)
    ensure_wrap(self.cvat_import_desc_lbl, self.pz3_cvat_import_section, padding=28, min_wrap=220)

    import_lf = ttk.LabelFrame(self.pz3_cvat_import_section, text=" CVAT XML z poprawkami ", padding=8)
    import_lf.pack(fill=tk.X, pady=(0, 0))

    self.cvat_import_target_hint_lbl = tk.Label(
        import_lf,
        text="Import jest kompatybilny z eksportem PZ3: crop tablicy -> boxy znaków. Nie wybieraj XML-a z boxami tablic na pełnych zdjęciach.",
        wraplength=540,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.cvat_import_target_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
    self._set_inline_status_label_state(self.cvat_import_target_hint_lbl, tone="muted", emphasis=False)
    ensure_wrap(self.cvat_import_target_hint_lbl, import_lf, padding=28, min_wrap=220)

    row2 = ttk.Frame(import_lf)
    row2.pack(fill=tk.X)
    self.import_cvat_xml_var = tk.StringVar()
    ttk.Entry(row2, textvariable=self.import_cvat_xml_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Button(row2, text="Wybierz XML", command=lambda: self._pick_file(self.import_cvat_xml_var)).pack(side=tk.RIGHT, padx=(6, 0))

    btn_import = ttk.Button(import_lf, text="Wczytaj poprawki do datasetu", command=self._run_cvat_import, style="Accent.TButton")
    btn_import.pack(fill=tk.X, pady=(10, 0), ipady=4)
    self.btn_cvat_import = btn_import

    def _refresh_pz3_cvat_import_card():
        card_info = getattr(self, "_pz3_cvat_import_card", None)
        if card_info is None:
            return
        expanded = bool(getattr(self, "_pz3_cvat_import_expanded", False))
        hovered = bool((getattr(self, "_pz3_cvat_import_card_state", {}) or {}).get("hovered", False))
        card_palette = self._get_pz3_action_card_palette()
        current_card_bg = card_palette["card_active_bg"] if expanded else (
            card_palette["card_hover_bg"] if hovered else card_palette["card_bg"]
        )
        border_color = card_palette["card_border"]
        for widget in (
            card_info.get("frame"),
            card_info.get("top_row"),
            card_info.get("badge"),
            card_info.get("state_badge"),
            card_info.get("title"),
            card_info.get("desc"),
        ):
            if widget is None:
                continue
            try:
                widget.configure(
                    bg=current_card_bg,
                    highlightbackground=border_color,
                    highlightcolor=border_color,
                    cursor="hand2",
                )
            except Exception:
                pass
        try:
            card_info["badge"].configure(text="IMPORT", fg=(card_palette["card_fg"] if expanded else card_palette["card_accent"]))
            card_info["title"].configure(text="Wczytaj poprawki z CVAT", fg=card_palette["card_fg"])
        except Exception:
            pass
        self._set_inline_status_label_state(
            card_info["desc"],
            text="Wczytuje XML z korektami znaków i dołącza go do zakresu datasetu.",
            tone=("default" if expanded else "muted"),
            emphasis=False,
        )
        try:
            card_info["desc"].configure(fg=(card_palette["card_fg"] if expanded else card_palette["card_muted"]))
        except Exception:
            pass
        state_badge = card_info.get("state_badge")
        if state_badge is not None:
            try:
                if expanded:
                    state_badge.configure(
                        text="Otwarte",
                        bg=current_card_bg,
                        fg=card_palette["card_fg"],
                        highlightbackground=border_color,
                        highlightcolor=border_color,
                    )
                    if not str(state_badge.winfo_manager()):
                        state_badge.pack(side=tk.RIGHT, anchor=tk.NE)
                elif str(state_badge.winfo_manager()):
                    state_badge.pack_forget()
            except Exception:
                pass
        try:
            if expanded:
                if not str(self.pz3_cvat_import_section.winfo_manager()):
                    self.pz3_cvat_import_section.pack(fill=tk.X, pady=(10, 0))
            elif str(self.pz3_cvat_import_section.winfo_manager()):
                self.pz3_cvat_import_section.pack_forget()
        except Exception:
            pass

    def _toggle_pz3_cvat_import_section(_event=None):
        self._pz3_cvat_import_expanded = not bool(getattr(self, "_pz3_cvat_import_expanded", False))
        _refresh_pz3_cvat_import_card()
        return "break"

    def _set_pz3_cvat_import_hover(hovered: bool):
        if bool(getattr(self, "_selection_hover_suppressed", False)):
            return
        state = getattr(self, "_pz3_cvat_import_card_state", {}) or {}
        if bool(state.get("hovered", False)) == bool(hovered):
            return
        state["hovered"] = bool(hovered)
        self._pz3_cvat_import_card_state = state
        _refresh_pz3_cvat_import_card()

    for widget in (
        cvat_import_card.get("frame"),
        cvat_import_card.get("top_row"),
        cvat_import_card.get("badge"),
        cvat_import_card.get("state_badge"),
        cvat_import_card.get("title"),
        cvat_import_card.get("desc"),
    ):
        if widget is None:
            continue
        try:
            widget.bind("<Button-1>", _toggle_pz3_cvat_import_section)
            widget.bind("<Enter>", lambda _event: _set_pz3_cvat_import_hover(True))
            widget.bind("<Leave>", lambda _event: _set_pz3_cvat_import_hover(False))
        except Exception:
            pass
    self._refresh_pz3_cvat_import_card = _refresh_pz3_cvat_import_card
    _refresh_pz3_cvat_import_card()

    self.gold_export_split_lf = ttk.LabelFrame(dataset_grid, text=" Split train / val / test ", padding=8)
    self.gold_export_split_lf.grid(row=1, column=0, sticky="ew", pady=(8, 0))
    self.gold_export_split_lf.grid_remove()

    self.gold_export_split_row = tk.Frame(
        self.gold_export_split_lf,
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    self.gold_export_split_row.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))

    self.gold_export_split_indicator = tk.Canvas(
        self.gold_export_split_row,
        width=16,
        height=16,
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    self.gold_export_split_indicator.pack(side=tk.LEFT, padx=(0, 6))

    self.gold_export_split_label = tk.Label(
        self.gold_export_split_row,
        text="Wariant treningowy i split przygotujesz w Z4",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
        cursor="hand2",
    )
    self.gold_export_split_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _toggle_gold_export_split(_event=None):
        try:
            self.gold_export_split_var.set(not bool(self.gold_export_split_var.get()))
        except Exception:
            self.gold_export_split_var.set(False)
        return "break"

    for widget in (self.gold_export_split_row, self.gold_export_split_indicator, self.gold_export_split_label):
        widget.bind("<Button-1>", _toggle_gold_export_split)

    self.gold_export_split_row_info = {
        "kind": "check",
        "frame": self.gold_export_split_row,
        "indicator": self.gold_export_split_indicator,
        "label": self.gold_export_split_label,
        "selected_getter": lambda: bool(self.gold_export_split_var.get()),
        "hovered": False,
    }
    for widget in (self.gold_export_split_row, self.gold_export_split_indicator, self.gold_export_split_label):
        widget.bind("<Enter>", lambda _event, info=self.gold_export_split_row_info: self._set_selection_row_hover(info, True))
        widget.bind("<Leave>", lambda _event, info=self.gold_export_split_row_info: self._set_selection_row_hover(info, False))
    self._apply_gold_export_split_check_style()

    ratios = ttk.Frame(self.gold_export_split_lf)
    ratios.pack(fill=tk.X)
    ratios.columnconfigure(1, weight=1)

    ttk.Label(ratios, text="Train %").grid(row=0, column=0, sticky=tk.W)
    self.gold_export_train_scale = ttk.Scale(
        ratios,
        from_=50,
        to=90,
        variable=self.gold_export_train_pct_var,
        orient=tk.HORIZONTAL,
    )
    self.gold_export_train_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
    self.gold_export_train_pct_lbl = ttk.Label(ratios, text="80%")
    self.gold_export_train_pct_lbl.grid(row=0, column=2, sticky=tk.W)

    ttk.Label(ratios, text="Val %").grid(row=1, column=0, sticky=tk.W)
    self.gold_export_val_scale = ttk.Scale(
        ratios,
        from_=5,
        to=45,
        variable=self.gold_export_val_pct_var,
        orient=tk.HORIZONTAL,
    )
    self.gold_export_val_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
    self.gold_export_val_pct_lbl = ttk.Label(ratios, text="10%")
    self.gold_export_val_pct_lbl.grid(row=1, column=2, sticky=tk.W)

    ttk.Label(ratios, text="Test %").grid(row=2, column=0, sticky=tk.W)
    self.gold_export_test_hint_lbl = ttk.Label(ratios, text="liczony automatycznie")
    self.gold_export_test_hint_lbl.grid(row=2, column=1, sticky=tk.W, padx=5)
    self.gold_export_test_pct_lbl = ttk.Label(ratios, text="Test: 10%")
    self.gold_export_test_pct_lbl.grid(row=2, column=2, sticky=tk.W)
    self._update_gold_export_split_labels()

    primary_action_border = blend_hex_colors(palette.get("success", "#2fa36b"), card_border, 0.42)
    primary_action_bg = blend_hex_colors(palette.get("surface_success", panel_alt), panel_bg, 0.72)
    self.pz3_primary_dataset_action_shell = tk.Frame(
        dataset_section,
        bg=primary_action_border,
        padx=1,
        pady=1,
        bd=0,
        highlightthickness=0,
    )
    self.pz3_primary_dataset_action_inner = tk.Frame(
        self.pz3_primary_dataset_action_shell,
        bg=primary_action_bg,
        padx=12,
        pady=10,
        bd=0,
        highlightthickness=0,
    )
    self.pz3_primary_dataset_action_inner.pack(fill=tk.X, expand=True)

    self.pz3_export_step_title_lbl = tk.Label(
        self.pz3_primary_dataset_action_inner,
        text="Utwórz źródłowy dataset znaków",
        anchor="w",
        justify=tk.LEFT,
        bg=primary_action_bg,
        bd=0,
        highlightthickness=0,
    )
    self.pz3_export_step_title_lbl.pack(anchor=tk.W, fill=tk.X, pady=(2, 0))
    self._set_inline_status_label_state(self.pz3_export_step_title_lbl, tone="success", emphasis=True)

    self.pz3_export_step_desc_lbl = tk.Label(
        self.pz3_primary_dataset_action_inner,
        text=(
            "Jeśli stan materiału z PZ2 jest poprawny, utwórz źródłowy dataset znaków YOLO-Detect. "
            "Split i wariant treningowy ustawisz później w Z4."
        ),
        wraplength=900,
        justify=tk.LEFT,
        anchor="w",
        bg=primary_action_bg,
        bd=0,
        highlightthickness=0,
    )
    self.pz3_export_step_desc_lbl.pack(anchor=tk.W, fill=tk.X, pady=(4, 8))
    self._set_inline_status_label_state(self.pz3_export_step_desc_lbl, tone="muted", emphasis=False)
    ensure_wrap(self.pz3_export_step_desc_lbl, self.pz3_primary_dataset_action_inner, padding=28, min_wrap=260)

    export_actions = tk.Frame(
        self.pz3_primary_dataset_action_inner,
        bg=primary_action_bg,
        bd=0,
        highlightthickness=0,
    )
    self.pz3_export_actions = export_actions
    export_actions.pack(fill=tk.X, pady=(12, 0))
    export_actions.columnconfigure(0, weight=1)
    export_actions.columnconfigure(1, weight=1)

    self.btn_yolo_gold_export = ttk.Button(
        export_actions,
        text="UTWÓRZ ŹRÓDŁOWY DATASET ZNAKÓW",
        command=self._run_yolo_gold_export,
        style="Accent.TButton",
    )
    self.btn_yolo_gold_export.grid(row=0, column=0, sticky="ew", padx=(0, 4), ipady=4)

    self.btn_char_classifier_export = ttk.Button(
        export_actions,
        text="EKSPORT KLASYFIKACYJNY ZNAKÓW",
        command=self._run_char_classification_export,
        style="WorkflowCard.TButton",
    )
    self.btn_char_classifier_export.grid(row=0, column=1, sticky="ew", padx=(4, 0), ipady=4)

    self.pz3_dataset_action_hint_lbl = tk.Label(
        self.pz3_primary_dataset_action_inner,
        text="Po utworzeniu datasetu przejdź do Z4, aby ustawić wariant treningowy i split.",
        wraplength=900,
        justify=tk.LEFT,
        anchor="w",
        bg=primary_action_bg,
        bd=0,
        highlightthickness=0,
    )
    self.pz3_dataset_action_hint_lbl.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))
    self._set_inline_status_label_state(self.pz3_dataset_action_hint_lbl, tone="muted", emphasis=False)
    ensure_wrap(self.pz3_dataset_action_hint_lbl, self.pz3_primary_dataset_action_inner, padding=28, min_wrap=260)
    try:
        self.pz3_primary_dataset_action_shell.pack_forget()
        self.pz3_primary_dataset_action_shell.pack(
            fill=tk.X,
            pady=(14, 14),
            before=self.pz3_goldpack_step_title_lbl,
        )
    except Exception:
        pass

    optional_cvat_lf = self.pz3_optional_cvat_lf

    card_review = make_pz3_card(
        optional_cvat_lf,
        "OPCJA",
        "Korekta w CVAT",
        "Obieg korekty poza aplikacją: wyślij cropy tablic do CVAT, popraw boxy znaków i wczytaj XML z powrotem w PZ3.",
        clickable=True,
    )
    self._pz3_review_card = card_review
    card_review["frame"].pack(fill=tk.X)
    try:
        self.gold_export_sources_import_host.pack_forget()
        self.gold_export_sources_import_host.pack(fill=tk.X, pady=(12, 0), after=card_review["frame"])
    except Exception:
        pass

    self.pz3_cvat_section = ttk.Frame(optional_cvat_lf, style="Panel.TFrame")

    review_section = ttk.Frame(self.pz3_cvat_section, style="Panel.TFrame")
    review_section.pack(fill=tk.X)
    self.review_title_lbl = None

    self.review_intro_lbl = tk.Label(
        review_section,
        text=(
            "Ten blok obsługuje pełną pętlę CVAT: PZ3 eksportuje zestaw z cropami tablic, "
            "w CVAT poprawiasz boxy pojedynczych znaków, a potem w tej samej karcie PZ3 wczytujesz "
            "annotations.xml z poprawkami do datasetu."
        ),
        anchor="w",
        justify=tk.LEFT,
        wraplength=900,
        bd=0,
        highlightthickness=0,
    )
    self.review_intro_lbl.pack(anchor=tk.W, fill=tk.X, pady=(4, 8))
    self._set_inline_status_label_state(self.review_intro_lbl, text=self.review_intro_lbl.cget("text"), tone="muted", emphasis=False)
    ensure_wrap(self.review_intro_lbl, review_section, padding=28, min_wrap=280)

    review_pack_lf = ttk.LabelFrame(review_section, text=" Zestaw cropów tablic do poprawy ", padding=8)
    review_pack_lf.pack(fill=tk.X, pady=(4, 0))

    self.cvat_option1_title_lbl = tk.Label(
        review_pack_lf,
        text="Eksportuj cropy tablic do CVAT",
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    self.cvat_option1_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self._set_inline_status_label_state(self.cvat_option1_title_lbl, tone="error", emphasis=False)

    self.cvat_option1_desc_lbl = tk.Label(
        review_pack_lf,
        text=(
            "Eksport tworzy zestaw do wysłania do CVAT: annotations.xml + obrazy cropów tablic. "
            "Po korekcie w CVAT wracasz tutaj i importujesz poprawiony XML w bloku 'Wczytaj poprawki z CVAT'. "
            "Poprawiasz boxy znaków, nie boxy tablic na pełnych zdjęciach."
        ),
        wraplength=820,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.cvat_option1_desc_lbl.pack(anchor=tk.W, fill=tk.X, pady=(4, 8))
    self._set_inline_status_label_state(self.cvat_option1_desc_lbl, tone="muted", emphasis=False)
    ensure_wrap(self.cvat_option1_desc_lbl, review_pack_lf, padding=28, min_wrap=260)

    btn_cvat = ttk.Button(
        review_pack_lf,
        text="WYGENERUJ ZIP DLA CVAT",
        command=self._run_cvat_export,
        style="Accent.TButton",
    )
    btn_cvat.pack(fill=tk.X, ipady=4)
    self.btn_cvat_export = btn_cvat

    self.cvat_export_state_lbl = tk.Label(
        review_pack_lf,
        text=(
            "Do eksportu CVAT potrzebny jest aktywny wynik PZ2 z wyodrębnionymi tablicami. "
            "Jeśli go nie ma, wróć do PZ1/PZ2 i przygotuj materiał."
        ),
        wraplength=820,
        justify=tk.LEFT,
        anchor="w",
        bd=0,
        highlightthickness=0,
    )
    self.cvat_export_state_lbl.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))
    self._set_inline_status_label_state(self.cvat_export_state_lbl, tone="warning", emphasis=False)
    ensure_wrap(self.cvat_export_state_lbl, review_pack_lf, padding=28, min_wrap=260)

    selected_path = str(getattr(self, "_pz3_selected_path", "") or "").strip().lower()
    self._pz3_selected_path = selected_path if selected_path in ("dataset", "cvat") else ""
    self._pz3_cvat_expanded = self._pz3_selected_path == "cvat"
    dataset_card_state = {"hovered": False}
    pz3_cvat_card_state = {"hovered": False}
    dataset_source_card_state = {"perfect_hovered": False, "existing_hovered": False}
    self._pz3_dataset_card_state = dataset_card_state
    self._pz3_cvat_card_state = pz3_cvat_card_state
    self._pz3_dataset_source_card_state = dataset_source_card_state

    def _refresh_pz3_dataset_source_card(card_info, *, selected: bool, hovered: bool, badge_text: str, title_text: str, desc_text: str):
        refresh_pz3_dataset_source_card(
            self,
            card_info,
            selected=selected,
            hovered=hovered,
            badge_text=badge_text,
            title_text=title_text,
            desc_text=desc_text,
        )

    def _set_pz3_dataset_source_mode(mode: str):
        set_pz3_dataset_source_mode(self, mode)
        return "break"

    def _refresh_pz3_dataset_mode_ui():
        refresh_pz3_dataset_mode_ui(self)

    self._refresh_pz3_dataset_mode_ui = _refresh_pz3_dataset_mode_ui

    def _refresh_pz3_cvat_card():
        refresh_pz3_cvat_card(self)

    def _refresh_pz3_dataset_card():
        refresh_pz3_dataset_card(self)

    def _refresh_pz3_cvat_section():
        refresh_pz3_cards_ui(self)

    self._refresh_pz3_cards_ui = _refresh_pz3_cvat_section
    self._refresh_preview_bound_action_states()

    def _toggle_pz3_cvat_section(_event=None):
        set_pz3_selected_path(self, "cvat")
        return "break"

    def _toggle_pz3_dataset_section(_event=None):
        set_pz3_selected_path(self, "dataset")
        return "break"

    def _set_pz3_dataset_hover(hovered: bool):
        if bool(getattr(self, "_selection_hover_suppressed", False)):
            return
        if bool(dataset_card_state.get("hovered", False)) == bool(hovered):
            return
        dataset_card_state["hovered"] = bool(hovered)
        _refresh_pz3_dataset_card()

    def _set_pz3_cvat_hover(hovered: bool):
        if bool(getattr(self, "_selection_hover_suppressed", False)):
            return
        if bool(pz3_cvat_card_state.get("hovered", False)) == bool(hovered):
            return
        pz3_cvat_card_state["hovered"] = bool(hovered)
        _refresh_pz3_cvat_card()

    def _set_pz3_dataset_source_hover(key: str, hovered: bool):
        if bool(getattr(self, "_selection_hover_suppressed", False)):
            return
        state_key = f"{key}_hovered"
        if bool(dataset_source_card_state.get(state_key, False)) == bool(hovered):
            return
        dataset_source_card_state[state_key] = bool(hovered)
        _refresh_pz3_dataset_mode_ui()

    for widget in (
        card_dataset.get("frame"),
        card_dataset.get("top_row"),
        card_dataset.get("badge"),
        card_dataset.get("state_badge"),
        card_dataset.get("title"),
        card_dataset.get("desc"),
    ):
        if widget is None:
            continue
        try:
            widget.bind("<Button-1>", _toggle_pz3_dataset_section)
            widget.bind("<Enter>", lambda _event: _set_pz3_dataset_hover(True))
            widget.bind("<Leave>", lambda _event: _set_pz3_dataset_hover(False))
        except Exception:
            pass

    for widget in (
        card_review.get("frame"),
        card_review.get("top_row"),
        card_review.get("badge"),
        card_review.get("state_badge"),
        card_review.get("title"),
        card_review.get("desc"),
    ):
        if widget is None:
            continue
        try:
            widget.bind("<Button-1>", _toggle_pz3_cvat_section)
            widget.bind("<Enter>", lambda _event: _set_pz3_cvat_hover(True))
            widget.bind("<Leave>", lambda _event: _set_pz3_cvat_hover(False))
        except Exception:
            pass

    for widget in (
        dataset_source_perfect_card.get("frame"),
        dataset_source_perfect_card.get("top_row"),
        dataset_source_perfect_card.get("badge"),
        dataset_source_perfect_card.get("state_badge"),
        dataset_source_perfect_card.get("title"),
        dataset_source_perfect_card.get("desc"),
    ):
        if widget is None:
            continue
        try:
            widget.bind("<Button-1>", lambda _event: _set_pz3_dataset_source_mode("perfect"))
            widget.bind("<Enter>", lambda _event: _set_pz3_dataset_source_hover("perfect", True))
            widget.bind("<Leave>", lambda _event: _set_pz3_dataset_source_hover("perfect", False))
        except Exception:
            pass

    for widget in (
        dataset_source_existing_card.get("frame"),
        dataset_source_existing_card.get("top_row"),
        dataset_source_existing_card.get("badge"),
        dataset_source_existing_card.get("state_badge"),
        dataset_source_existing_card.get("title"),
        dataset_source_existing_card.get("desc"),
    ):
        if widget is None:
            continue
        try:
            widget.bind("<Button-1>", lambda _event: _set_pz3_dataset_source_mode("existing"))
            widget.bind("<Enter>", lambda _event: _set_pz3_dataset_source_hover("existing", True))
            widget.bind("<Leave>", lambda _event: _set_pz3_dataset_source_hover("existing", False))
        except Exception:
            pass

    _refresh_pz3_dataset_mode_ui()
    _refresh_pz3_cvat_section()

    status_section = ttk.Frame(status_shell_inner, style="Panel.TFrame", padding=(10, 8, 10, 10))
    self.pz3_status_section = status_section
    self.pz3_status_section.pack(fill=tk.BOTH, expand=True)
    self.dataset_status_title_lbl = SectionHeaderLabel(
        status_section,
        self.app,
        text="Status PZ3",
    )
    self.dataset_status_title_lbl.pack(anchor=tk.W, fill=tk.X)
    self.dataset_status_intro_lbl = None

    for widget_name in (
        "dataset_intro_lbl",
        "review_intro_lbl",
        "cvat_option1_desc_lbl",
        "cvat_option2_desc_lbl",
        "pz3_dataset_source_intro_lbl",
        "pz3_source_preview_status_lbl",
        "pz3_source_pool_status_lbl",
        "pz3_source_next_status_lbl",
        "pz3_existing_dataset_desc_lbl",
        "pz3_existing_dataset_status_lbl",
        "pz3_optional_cvat_intro_lbl",
        "pz3_goldpack_step_desc_lbl",
        "gold_export_filters_title_lbl",
        "gold_export_sources_title_lbl",
        "gold_export_scope_lbl",
        "gold_cvat_source_status_lbl",
        "gold_export_sources_hint_lbl",
        "cvat_import_desc_lbl",
        "pz3_export_step_desc_lbl",
        "dataset_status_intro_lbl",
        "run_console",
        "dataset_console",
        "readiness_console",
        "pz3_dataset_action_hint_lbl",
    ):
        self._mark_inline_status_contrast_boost(getattr(self, widget_name, None))

    status_table = tk.Frame(
        status_section,
        bg=blend_hex_colors(
            palette.get("panel_border", palette.get("border", "#3c3c3c")),
            palette.get("panel", "#252526"),
            0.34,
        ),
        bd=0,
        highlightthickness=1,
    )
    status_table.pack(fill=tk.X, pady=(8, 0))
    status_grid_border = blend_hex_colors(
        palette.get("panel_border", palette.get("border", "#3c3c3c")),
        palette.get("panel", "#252526"),
        0.34,
    )
    status_surface = blend_hex_colors(
        palette.get("panel_border", palette.get("border", "#3c3c3c")),
        palette.get("panel", "#252526"),
        0.92,
    )
    status_header_bg = blend_hex_colors(status_surface, palette.get("panel", "#252526"), 0.18)
    status_table.configure(
        highlightbackground=status_grid_border,
        highlightcolor=status_grid_border,
    )
    status_table.grid_columnconfigure(0, weight=0, minsize=92)
    status_table.grid_columnconfigure(1, weight=1)

    status_header_left = tk.Label(
        status_table,
        text="Obszar",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8),
        fg=palette.get("muted", "#c7c7c7"),
        bg=status_header_bg,
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground=status_grid_border,
        highlightcolor=status_grid_border,
        padx=10,
        pady=6,
    )
    status_header_left.grid(row=0, column=0, sticky="ew")

    status_header_right = tk.Label(
        status_table,
        text="Podsumowanie",
        anchor="w",
        justify=tk.LEFT,
        font=("Segoe UI", 8),
        fg=palette.get("muted", "#c7c7c7"),
        bg=status_header_bg,
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground=status_grid_border,
        highlightcolor=status_grid_border,
        padx=10,
        pady=6,
    )
    status_header_right.grid(row=0, column=1, sticky="ew")

    run_row_bg = blend_hex_colors(status_surface, palette.get("panel", "#252526"), 0.08)
    dataset_row_bg = blend_hex_colors(status_surface, palette.get("panel", "#252526"), 0.12)
    export_row_bg = blend_hex_colors(status_surface, palette.get("panel", "#252526"), 0.16)
    readiness_row_bg = blend_hex_colors(status_surface, palette.get("panel", "#252526"), 0.20)
    import_row_bg = blend_hex_colors(status_surface, palette.get("panel", "#252526"), 0.24)

    run_key_lbl = tk.Label(
        status_table,
        text="Run",
        anchor="nw",
        justify=tk.LEFT,
        width=10,
        font=("Segoe UI", 9),
        fg=palette.get("muted", "#c7c7c7"),
        bg=run_row_bg,
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground=status_grid_border,
        highlightcolor=status_grid_border,
        padx=10,
        pady=8,
    )
    run_key_lbl.grid(row=1, column=0, sticky="nsew")
    self.run_status_key_lbl = run_key_lbl

    self.run_console = tk.Label(
        status_table,
        text="â€”",
        anchor="nw",
        justify=tk.LEFT,
        wraplength=190,
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground=status_grid_border,
        highlightcolor=status_grid_border,
        bg=run_row_bg,
        padx=10,
        pady=8,
    )
    self.run_console.grid(row=1, column=1, sticky="nsew")
    self._set_inline_status_label_state(self.run_console, text="â€”", tone="muted", emphasis=False)

    dataset_key_lbl = tk.Label(
        status_table,
        text="Dataset",
        anchor="nw",
        justify=tk.LEFT,
        width=10,
        font=("Segoe UI", 9),
        fg=palette.get("muted", "#c7c7c7"),
        bg=dataset_row_bg,
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground=status_grid_border,
        highlightcolor=status_grid_border,
        padx=10,
        pady=8,
    )
    dataset_key_lbl.grid(row=2, column=0, sticky="nsew")
    self.dataset_status_key_lbl = dataset_key_lbl

    self.dataset_console = tk.Label(
        status_table,
        text="â€”",
        anchor="nw",
        justify=tk.LEFT,
        wraplength=190,
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground=status_grid_border,
        highlightcolor=status_grid_border,
        bg=dataset_row_bg,
        padx=10,
        pady=8,
    )
    self.dataset_console.grid(row=2, column=1, sticky="nsew")
    self._set_inline_status_label_state(self.dataset_console, text="â€”", tone="muted", emphasis=False)

    readiness_key_lbl = tk.Label(
        status_table,
        text="Gotowość",
        anchor="nw",
        justify=tk.LEFT,
        width=10,
        font=("Segoe UI", 9),
        fg=palette.get("muted", "#c7c7c7"),
        bg=readiness_row_bg,
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground=status_grid_border,
        highlightcolor=status_grid_border,
        padx=10,
        pady=8,
    )
    readiness_key_lbl.grid(row=4, column=0, sticky="nsew")
    self.readiness_status_key_lbl = readiness_key_lbl

    self.readiness_console = tk.Label(
        status_table,
        text="â€”",
        anchor="nw",
        justify=tk.LEFT,
        wraplength=190,
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground=status_grid_border,
        highlightcolor=status_grid_border,
        bg=readiness_row_bg,
        padx=10,
        pady=8,
    )
    self.readiness_console.grid(row=4, column=1, sticky="nsew")
    self._set_inline_status_label_state(self.readiness_console, text="â€”", tone="muted", emphasis=False)

    export_key_lbl = tk.Label(
        status_table,
        text="Eksport",
        anchor="nw",
        justify=tk.LEFT,
        width=10,
        font=("Segoe UI", 9),
        fg=palette.get("muted", "#c7c7c7"),
        bg=export_row_bg,
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground=status_grid_border,
        highlightcolor=status_grid_border,
        padx=10,
        pady=8,
    )
    export_key_lbl.grid(row=3, column=0, sticky="nsew")
    self.export_status_key_lbl = export_key_lbl

    self.export_console = tk.Label(
        status_table,
        text="—",
        anchor="nw",
        justify=tk.LEFT,
        wraplength=190,
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground=status_grid_border,
        highlightcolor=status_grid_border,
        bg=export_row_bg,
        padx=10,
        pady=8,
    )
    self.export_console.grid(row=3, column=1, sticky="nsew")
    self._set_inline_status_label_state(self.export_console, text="—", tone="muted", emphasis=False)

    import_key_lbl = tk.Label(
        status_table,
        text="Import",
        anchor="nw",
        justify=tk.LEFT,
        width=10,
        font=("Segoe UI", 9),
        fg=palette.get("muted", "#c7c7c7"),
        bg=import_row_bg,
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground=status_grid_border,
        highlightcolor=status_grid_border,
        padx=10,
        pady=8,
    )
    import_key_lbl.grid(row=5, column=0, sticky="nsew")
    self.import_status_key_lbl = import_key_lbl

    self.import_console = tk.Label(
        status_table,
        text="—",
        anchor="nw",
        justify=tk.LEFT,
        wraplength=190,
        bd=1,
        relief="solid",
        highlightthickness=1,
        highlightbackground=status_grid_border,
        highlightcolor=status_grid_border,
        bg=import_row_bg,
        padx=10,
        pady=8,
    )
    self.import_console.grid(row=5, column=1, sticky="nsew")
    self._set_inline_status_label_state(self.import_console, text="—", tone="muted", emphasis=False)

    HELP.bind_help(btn_cvat, "btn_export_cvat")
    HELP.bind_help(self._pz3_dataset_source_lf, "t2_pz3_source")
    HELP.bind_help(self.pz3_dataset_source_intro_lbl, "t2_pz3_source")
    HELP.bind_help(self.pz3_source_preview_status_lbl, "t2_pz3_source")
    HELP.bind_help(self.pz3_source_pool_status_lbl, "t2_pz3_source")
    HELP.bind_help(self.pz3_source_next_status_lbl, "t2_pz3_source")
    HELP.bind_help(self._pz3_dataset_source_perfect_card.get("frame"), "t2_pz3_source")
    HELP.bind_help(self._pz3_dataset_source_existing_card.get("frame"), "t2_pz3_existing_dataset")
    HELP.bind_help(self.pz3_existing_dataset_panel, "t2_pz3_existing_dataset")
    HELP.bind_help(self.pz3_existing_dataset_entry, "t2_pz3_existing_dataset")
    HELP.bind_help(self.pz3_existing_dataset_pick_btn, "t2_pz3_existing_dataset")
    HELP.bind_help(self.pz3_existing_dataset_project_btn, "t2_pz3_existing_dataset")
    HELP.bind_help(self.pz3_goldpack_step_title_lbl, "t2_pz3_goldpack")
    HELP.bind_help(self.pz3_goldpack_step_desc_lbl, "t2_pz3_goldpack")
    HELP.bind_help(self.btn_yolo_gold_export, "btn_export_yolo")
    HELP.bind_help(self.btn_char_classifier_export, "btn_export_yolo")
    HELP.bind_help(self.gold_export_goldpack_lf, "btn_export_yolo")
    HELP.bind_help(self.gold_export_split_lf, "btn_export_yolo")
    HELP.bind_help(btn_import, "btn_import_cvat")

    self._bind_scroll_canvas_children(
        self.cvat_export_content,
        self.cvat_export_canvas,
        self._cvat_export_canvas_overflows,
    )
    self.frame.after_idle(self._sync_cvat_export_scrollregion)
    self.frame.after_idle(self._sync_cvat_export_canvas_width)
    self.frame.bind_all("<MouseWheel>", self._on_cvat_export_global_mousewheel, add="+")
    self.frame.bind_all("<Button-4>", self._on_cvat_export_global_mousewheel, add="+")
    self.frame.bind_all("<Button-5>", self._on_cvat_export_global_mousewheel, add="+")

    nav = tk.Frame(
        export_shell_inner,
        bg=export_shell_fill,
        bd=0,
        highlightthickness=0,
    )
    nav.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 6))
    self.pz3_export_nav = nav

    self.pz3_back_to_detect_frame = tk.Frame(
        nav,
        bg=export_shell_fill,
        bd=0,
        highlightthickness=0,
    )
    self.pz3_back_to_detect_frame.pack(side=tk.LEFT, fill=tk.Y)

    self.btn_back_to_detect = ttk.Button(
        self.pz3_back_to_detect_frame,
        text="Popraw anotacje w PZ2",
        command=self.back_to_substep_2,
        style="WorkflowCard.TButton",
    )
    self.btn_back_to_detect.pack(side=tk.BOTTOM)
    self.btn_back_to_detect.configure(padding=(10, 3), width=max(NAV_BUTTON_WIDTH, 26))

    self.pz3_nav_actions_frame = tk.Frame(
        nav,
        bg=export_shell_fill,
        bd=0,
        highlightthickness=0,
    )
    self.pz3_nav_actions_frame.pack(side=tk.RIGHT, fill=tk.Y)
    self.step3_finish_action_card = None
    self.step3_finish_card = tk.Frame(
        self.pz3_nav_actions_frame,
        bg=export_shell_fill,
        bd=0,
        highlightthickness=0,
    )
    self.step3_finish_card.pack(anchor=tk.E)
    self.step3_finish_card_inner = self.step3_finish_card
    self.step3_finish_section_lbl = None
    self.step3_finish_hint_lbl = None

    self.btn_finish_step3_frame = tk.Frame(
        self.step3_finish_card,
        bg=export_shell_fill,
        bd=0,
        highlightthickness=0,
    )
    self.btn_finish_step3_frame.pack(fill=tk.X)
    self.btn_finish_step3_pulse_frame = tk.Frame(
        self.btn_finish_step3_frame,
        bg=export_shell_fill,
        bd=0,
        highlightthickness=0,
    )
    self.btn_finish_step3_pulse_frame.pack(fill=tk.X)
    self.btn_finish_step3 = ttk.Button(
        self.btn_finish_step3_pulse_frame,
        text="Zamknij pracę w PZ3",
        command=self._finalize_step3_from_existing_outputs,
        style="WorkflowCard.TButton",
        state=tk.DISABLED,
    )
    self.btn_finish_step3.pack(anchor=tk.E)
    self.btn_finish_step3.configure(padding=(10, 3), width=max(NAV_BUTTON_WIDTH + 2, 28))
    HELP.bind_help(self.step3_finish_card, "camp_step3")
    HELP.bind_help(self.btn_finish_step3, "camp_step3")
    try:
        self._update_step3_finish_button_state()
    except Exception:
        pass
