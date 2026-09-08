from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..campaign_manager import CAMPAIGN
from ..config import CONFIG
from .z4_campaign_flow import return_to_campaign_from_step4
from .web_slim_scrollbar import blend_hex_colors
from .dataset_display import build_dataset_display_ref

if TYPE_CHECKING:
    from .tab_training import TrainingTab


def _set_pack_visible(widget, visible: bool, **pack_kwargs):
    if widget is None:
        return
    try:
        manager = str(widget.winfo_manager())
    except Exception:
        manager = ""
    try:
        if visible:
            if manager != "pack":
                widget.pack(**pack_kwargs)
            elif pack_kwargs:
                widget.pack_configure(**pack_kwargs)
        elif manager == "pack":
            widget.pack_forget()
    except Exception:
        pass


def _configure_step4_next_button(host: "TrainingTab", label: str, *, state=None):
    button = getattr(host, "btn_step4_next", None)
    if button is None:
        return
    text = str(label or "Dalej do treningu")
    try:
        # ttk width is measured in text units; long campaign labels need more
        # room than the default compact navigation buttons.
        width = max(18, min(52, len(text) + 2))
        kwargs = {"text": text, "width": width}
        if state is not None:
            kwargs["state"] = state
        button.configure(**kwargs)
    except Exception:
        try:
            kwargs = {"text": text}
            if state is not None:
                kwargs["state"] = state
            button.configure(**kwargs)
        except Exception:
            pass


def refresh_step4_campaign_builder_inputs_ui(host: "TrainingTab"):
    vm = host._get_step4_dataset_workflow_view_model()

    def _dataset_ref_id(path_like, *, target_hint: str, counts: dict | None = None) -> str:
        raw = str(path_like or "").strip()
        if not raw:
            return "-"
        try:
            return build_dataset_display_ref(raw, target_hint=target_hint, counts=counts).id
        except Exception:
            try:
                return Path(raw).name
            except Exception:
                return raw

    def _char_yolo_dataset_brief(path_like) -> dict:
        raw = str(path_like or "").strip()
        result = {
            "ok": False,
            "created_at": "-",
            "plates": 0,
            "chars": 0,
            "missing_labels": 0,
        }
        if not raw:
            return result
        try:
            root = Path(raw)
        except Exception:
            return result
        if root.is_file() and root.name.lower() == "data.yaml":
            root = root.parent
        if not root.exists() or not root.is_dir():
            return result

        try:
            root_key = str(root.resolve())
        except Exception:
            root_key = str(root)

        layouts: list[tuple[Path, Path]] = []
        for split_name in ("train", "val", "test"):
            layouts.append((root / "images" / split_name, root / "labels" / split_name))
            layouts.append((root / split_name / "images", root / split_name / "labels"))
        layouts.append((root / "images", root / "labels"))

        layout_tokens: list[str] = []
        for image_dir, label_dir in layouts:
            for probe in (image_dir, label_dir):
                try:
                    layout_tokens.append(f"{probe}:{int(probe.stat().st_mtime) if probe.exists() else 0}")
                except Exception:
                    layout_tokens.append(f"{probe}:0")
        cache_key = f"{root_key}|" + "|".join(layout_tokens)
        cache = getattr(host, "_step4_char_dataset_brief_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            try:
                setattr(host, "_step4_char_dataset_brief_cache", cache)
            except Exception:
                pass
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            return dict(cached)

        created_timestamp = 0.0
        try:
            created_timestamp = float(root.stat().st_ctime or root.stat().st_mtime or 0)
        except Exception:
            created_timestamp = 0.0
        if created_timestamp <= 0:
            try:
                yaml_path = root / "data.yaml"
                created_timestamp = float(yaml_path.stat().st_ctime or yaml_path.stat().st_mtime or 0) if yaml_path.exists() else 0.0
            except Exception:
                created_timestamp = 0.0
        if created_timestamp > 0:
            try:
                result["created_at"] = datetime.fromtimestamp(created_timestamp).strftime("%Y-%m-%d %H:%M")
            except Exception:
                result["created_at"] = "-"

        image_exts = {str(ext or "").lower() for ext in getattr(CONFIG, "IMAGE_EXTENSIONS", ())}
        seen_images: set[str] = set()
        plate_count = 0
        char_count = 0
        missing_labels = 0

        for image_dir, label_dir in layouts:
            if not image_dir.exists() or not image_dir.is_dir():
                continue
            try:
                image_paths = [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in image_exts]
            except Exception:
                image_paths = []
            for image_path in image_paths:
                try:
                    image_key = str(image_path.resolve())
                except Exception:
                    image_key = str(image_path)
                if image_key in seen_images:
                    continue
                seen_images.add(image_key)
                plate_count += 1
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    missing_labels += 1
                    continue
                try:
                    with open(label_path, "r", encoding="utf-8", errors="ignore") as handle:
                        char_count += sum(1 for line in handle if str(line or "").strip())
                except Exception:
                    continue

        result.update(
            {
                "ok": bool(plate_count > 0),
                "plates": int(plate_count),
                "chars": int(char_count),
                "missing_labels": int(missing_labels),
            }
        )
        try:
            if len(cache) > 24:
                cache.clear()
            cache[cache_key] = dict(result)
        except Exception:
            pass
        return result

    def _format_char_dataset_brief(brief: dict, *, empty_text: str) -> str:
        if not bool(brief.get("ok")):
            return empty_text
        parts = [
            f"utworzono: {brief.get('created_at') or '-'}",
            f"tablice: {int(brief.get('plates', 0) or 0)}",
            f"znaki: {int(brief.get('chars', 0) or 0)}",
        ]
        missing = int(brief.get("missing_labels", 0) or 0)
        if missing > 0:
            parts.append(f"bez etykiet: {missing}")
        return " | ".join(parts)

    def _refresh_creator_campaign_summary_table() -> None:
        frame = getattr(host, "creator_campaign_summary_frame", None)
        rows = getattr(host, "_creator_campaign_summary_rows", None)
        if frame is None or not rows:
            return
        if not bool(getattr(vm, "in_campaign", False)) or str(getattr(vm, "mode", "") or "") != "plate":
            _set_pack_visible(frame, False)
            _set_pack_visible(getattr(host, "creator_campaign_summary_title", None), False)
            _set_pack_visible(getattr(host, "creator_flow_strip", None), False)
            return

        palette = getattr(host.app, "palette", {})
        panel = palette.get("panel", "#252526")
        panel_alt = palette.get("panel_alt", "#2d2d30")
        field = palette.get("field", panel_alt)
        border_base = palette.get("panel_border", palette.get("border", "#3c3c3c"))
        fg = palette.get("fg", "#f3f3f3")
        muted = palette.get("muted", "#c7c7c7")
        success = palette.get("success", "#4ec9b0")
        warning = palette.get("warning", "#d7ba7d")
        border = blend_hex_colors(success, border_base, 0.62)
        header_bg = blend_hex_colors(success, panel_alt, 0.84)
        row_alt = blend_hex_colors(panel_alt, panel, 0.45)

        readiness = {}
        try:
            readiness = dict(host.get_campaign_step4_readiness(iteration_target="plate") or {})
        except Exception:
            readiness = {}
        stats = {}
        try:
            stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
        except Exception:
            stats = {}

        approved_images = int(readiness.get("project_approved_images", stats.get("images", 0)) or 0)
        approved_plates = int(readiness.get("project_approved_plates", stats.get("plates", 0)) or 0)
        ready_dataset = str(readiness.get("ready_dataset") or "").strip()
        train_count = int(readiness.get("train_images", 0) or 0)
        val_count = int(readiness.get("val_images", 0) or 0)
        test_count = int(readiness.get("test_images", 0) or 0)

        xml_text = ""
        images_text = ""
        try:
            xml_text = host._shorten_training_text(
                host._format_workspace_relative_path(host.cvat_xml_var.get()),
                92,
            )
        except Exception:
            pass
        try:
            images_text = host._shorten_training_text(
                host._format_workspace_relative_path(host.cvat_images_var.get()),
                92,
            )
        except Exception:
            pass

        if ready_dataset and (train_count > 0 or val_count > 0 or test_count > 0):
            variant_text = f"Gotowy wariant: train={train_count}, val={val_count}, test={test_count}"
        else:
            variant_text = "Jeszcze nie utworzono wariantu"

        values = {
            "target": "Otworzy\u0107 trening modelu tablic YOLO Pose",
            "source": "Zatwierdzone tablice z projektu",
            "material": (
                f"{approved_images} obrazów, {approved_plates} tablic"
                + (f" | XML: {xml_text}" if xml_text else "")
                + (f" | obrazy: {images_text}" if images_text else "")
            ),
            "variant": variant_text,
        }

        try:
            frame.configure(bg=border, highlightbackground=border, highlightcolor=border)
            getattr(host, "creator_campaign_summary_grid", frame).configure(bg=border)
            title = getattr(host, "creator_campaign_summary_title", None)
            if title is not None:
                title.configure(bg=panel, fg=fg)
        except Exception:
            pass
        for widget in getattr(host, "_creator_campaign_summary_header_widgets", ()) or ():
            try:
                widget.configure(bg=header_bg, fg=success, highlightbackground=border, highlightcolor=border)
            except Exception:
                pass
        for index, (key, widgets) in enumerate(rows.items()):
            label_widget, value_widget = widgets
            bg = field if index % 2 == 0 else row_alt
            value_fg = success if key == "variant" and ready_dataset else fg
            if key == "variant" and not ready_dataset:
                value_fg = warning
            try:
                label_widget.configure(bg=bg, fg=muted, highlightbackground=border, highlightcolor=border)
                if key == "material" and isinstance(value_widget, tk.Frame):
                    value_widget.configure(bg=bg, highlightbackground=border, highlightcolor=border)
                    for child in value_widget.winfo_children():
                        child.destroy()

                    line = tk.Frame(value_widget, bd=0, highlightthickness=0, bg=bg)
                    line.pack(anchor=tk.W, fill=tk.X)

                    def _add_material_segment(text, font, color):
                        tk.Label(
                            line,
                            text=str(text),
                            font=font,
                            bg=bg,
                            fg=color,
                            bd=0,
                            highlightthickness=0,
                            padx=0,
                            pady=0,
                            anchor="w",
                        ).pack(side=tk.LEFT)

                    base_font = ("Segoe UI", 8)
                    counter_font = ("Segoe UI Semibold", 10)
                    _add_material_segment(approved_images, counter_font, success)
                    _add_material_segment(" obraz\u00f3w, ", base_font, fg)
                    _add_material_segment(approved_plates, counter_font, success)
                    _add_material_segment(" tablic", base_font, fg)

                    extra = []
                    if xml_text:
                        extra.append(f"XML: {xml_text}")
                    if images_text:
                        extra.append(f"obrazy: {images_text}")
                    if extra:
                        _add_material_segment(" | " + " | ".join(extra), base_font, fg)
                    continue
                value_widget.configure(
                    text=str(values.get(key, "-") or "-"),
                    bg=bg,
                    fg=value_fg,
                    highlightbackground=border,
                    highlightcolor=border,
                    wraplength=680,
                )
            except Exception:
                pass

    def _refresh_split_campaign_summary_table() -> None:
        frame = getattr(host, "split_campaign_summary_frame", None)
        rows = getattr(host, "_split_campaign_summary_rows", None)
        if frame is None or not rows:
            return
        if not bool(getattr(vm, "in_campaign", False)) or str(getattr(vm, "mode", "") or "") != "char":
            _set_pack_visible(frame, False)
            _set_pack_visible(getattr(host, "split_campaign_summary_title", None), False)
            return

        palette = getattr(host.app, "palette", {})
        panel = palette.get("panel", "#252526")
        panel_alt = palette.get("panel_alt", "#2d2d30")
        field = palette.get("field", panel_alt)
        border_base = palette.get("panel_border", palette.get("border", "#3c3c3c"))
        fg = palette.get("fg", "#f3f3f3")
        muted = palette.get("muted", "#c7c7c7")
        success = palette.get("success", "#4ec9b0")
        warning = palette.get("warning", "#d7ba7d")
        accent = palette.get("accent", "#0e639c")
        border = blend_hex_colors(success, border_base, 0.62)
        header_bg = blend_hex_colors(success, panel_alt, 0.84)
        row_alt = blend_hex_colors(panel_alt, panel, 0.45)

        readiness = {}
        try:
            readiness = dict(host.get_campaign_step4_readiness(iteration_target="char") or {})
        except Exception:
            readiness = {}
        source_dataset = str(readiness.get("source_dataset") or "").strip()
        if not source_dataset:
            try:
                source_dataset = str(host.split_src_var.get() or "").strip()
            except Exception:
                source_dataset = ""
        ready_dataset = str(readiness.get("ready_dataset") or "").strip()
        train_count = int(readiness.get("train_images", 0) or 0)
        val_count = int(readiness.get("val_images", 0) or 0)
        test_count = int(readiness.get("test_images", 0) or 0)
        counts = {
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "total": train_count + val_count + test_count,
        }
        has_ready_variant = bool(ready_dataset and train_count > 0 and val_count > 0)
        source_brief = _char_yolo_dataset_brief(source_dataset)
        variant_brief = _char_yolo_dataset_brief(ready_dataset) if has_ready_variant else {}
        values = {
            "source": (
                _dataset_ref_id(source_dataset, target_hint="char"),
                _format_char_dataset_brief(
                    source_brief,
                    empty_text="Brak źródłowego datasetu znaków z T05/PZ3.",
                ),
            ),
            "variant": (
                (
                    _dataset_ref_id(ready_dataset, target_hint="char", counts=counts)
                    if has_ready_variant
                    else "Jeszcze nie utworzono wariantu"
                ),
                (
                    _format_char_dataset_brief(
                        variant_brief,
                        empty_text="Utwórz wariant, aby odblokować trening w PZ2.",
                    )
                    if has_ready_variant
                    else "Utwórz wariant, aby odblokować trening w PZ2."
                ),
            ),
            "split": (
                "train / val / test",
                (
                    f"train {train_count} | val {val_count} | test {test_count}"
                    if has_ready_variant
                    else "Podział zostanie zapisany w tworzonym wariancie."
                ),
            ),
        }

        try:
            frame.configure(bg=border, highlightbackground=border, highlightcolor=border)
            getattr(host, "split_campaign_summary_grid", frame).configure(bg=border)
            title = getattr(host, "split_campaign_summary_title", None)
            if title is not None:
                title.configure(bg=panel, fg=fg)
        except Exception:
            pass
        for widget in getattr(host, "_split_campaign_summary_header_widgets", ()) or ():
            try:
                widget.configure(bg=header_bg, fg=success, highlightbackground=border, highlightcolor=border)
            except Exception:
                pass

        for index, (key, widgets) in enumerate(rows.items()):
            try:
                label_widget, id_widget, details_widget = widgets
            except Exception:
                continue
            bg = field if index % 2 == 0 else row_alt
            try:
                label_widget.configure(bg=bg, fg=muted, highlightbackground=border, highlightcolor=border)
                id_text, details_text = values.get(key, ("-", "-"))
                id_fg = fg
                details_fg = fg
                if key == "source":
                    id_fg = success if bool(source_brief.get("ok")) else warning
                    details_fg = success if bool(source_brief.get("ok")) else warning
                elif key == "variant":
                    id_fg = success if has_ready_variant else warning
                    details_fg = success if has_ready_variant else warning
                elif key == "split":
                    id_fg = accent
                    details_fg = success if has_ready_variant else muted
                id_widget.configure(
                    text=str(id_text or "-"),
                    bg=bg,
                    fg=id_fg,
                    highlightbackground=border,
                    highlightcolor=border,
                    wraplength=260,
                )
                details_widget.configure(
                    text=str(details_text or "-"),
                    bg=bg,
                    fg=details_fg,
                    highlightbackground=border,
                    highlightcolor=border,
                    wraplength=420,
                )
            except Exception:
                pass

    try:
        route_manager = str(host.step4_route_panel_frame.winfo_manager())
    except Exception:
        route_manager = ""

    if not bool(vm.show_route_panel):
        if route_manager == "pack":
            try:
                host.step4_route_panel_frame.pack_forget()
            except Exception:
                pass
    else:
        if route_manager != "pack":
            try:
                host.step4_route_panel_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8), before=host.step4_mode_right_host)
            except Exception:
                pass

    try:
        if vm.mode == "plate":
            if bool(vm.in_campaign):
                creator_intro = (
                    "Utwórz wariant datasetu tablic z materiału projektu. "
                    "PZ1 ustawia split i opcjonalnie powiększa część train."
                )
            else:
                creator_intro = (
                    "Wskaż XML anotacji i zgodny katalog zdjęć. PZ1 przygotuje wariant train / val / test dla YOLO Pose."
                )
            host._set_training_widget_text(host.creator_intro_lbl, creator_intro)
            host._set_training_widget_text(
                host.btn_step4_create,
                "Utwórz wariant datasetu tablic",
            )
            try:
                host._refresh_dataset_creator_cta_state()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if vm.mode == "plate":
            in_campaign = bool(vm.in_campaign)
            source_mode_frame = getattr(host, "creator_source_mode_frame", None)
            ready_label = getattr(host, "creator_ready_dataset_lbl", None)
            ready_radio = getattr(host, "creator_source_ready_radio", None)
            xml_radio = getattr(host, "creator_source_xml_radio", None)
            xml_row = getattr(host, "creator_xml_row", None)
            auto_match_hint = getattr(host, "creator_auto_match_hint_lbl", None)
            images_row = getattr(host, "creator_images_row", None)
            source_summary = getattr(host, "creator_source_summary_lbl", None)
            output_row = getattr(host, "creator_output_row", None)
            ratios_frame = getattr(host, "creator_ratios_frame", None)
            augmentation_frame = getattr(host, "creator_augmentation_frame", None)
            create_frame = getattr(host, "btn_step4_create_frame", None)
            progress = getattr(host, "ds_progress", None)
            status = getattr(host, "ds_status", None)

            if in_campaign:
                _set_pack_visible(source_mode_frame, False)
                try:
                    host._set_creator_source_mode("xml")
                except Exception:
                    pass
                if ready_label is not None:
                    _set_pack_visible(ready_label, False)
                _set_pack_visible(xml_row, False)
                _set_pack_visible(images_row, False)
                _refresh_creator_campaign_summary_table()
                _set_pack_visible(
                    getattr(host, "creator_flow_strip", None),
                    True,
                    fill=tk.X,
                    pady=(0, 12),
                    after=host.creator_intro_lbl,
                )
                _set_pack_visible(
                    getattr(host, "creator_campaign_summary_title", None),
                    True,
                    fill=tk.X,
                    pady=(8, 7),
                    after=getattr(host, "creator_flow_strip", None),
                )
                _set_pack_visible(
                    getattr(host, "creator_campaign_summary_frame", None),
                    True,
                    fill=tk.X,
                    pady=(0, 12),
                    after=getattr(host, "creator_campaign_summary_title", None),
                )
                if source_summary is not None:
                    _set_pack_visible(source_summary, False)
                _set_pack_visible(auto_match_hint, False)
                _set_pack_visible(output_row, False)
                _set_pack_visible(
                    ratios_frame,
                    True,
                    fill=tk.X,
                    pady=(12, 8),
                    after=getattr(host, "creator_campaign_summary_frame", None),
                )
                _set_pack_visible(augmentation_frame, True, fill=tk.X, pady=(12, 8), after=ratios_frame)
                _set_pack_visible(create_frame, True, fill=tk.X, pady=(12, 10), after=(augmentation_frame or ratios_frame))
                if getattr(progress, "master", None) is not getattr(host, "step4_creator_action_inner", None):
                    _set_pack_visible(progress, True, fill=tk.X, pady=2, after=create_frame)
                if getattr(status, "master", None) is not getattr(host, "step4_creator_action_inner", None):
                    _set_pack_visible(status, True, anchor=tk.W)
            else:
                _set_pack_visible(getattr(host, "creator_campaign_summary_frame", None), False)
                _set_pack_visible(getattr(host, "creator_campaign_summary_title", None), False)
                _set_pack_visible(getattr(host, "creator_flow_strip", None), False)
                try:
                    host._set_creator_source_mode("xml")
                except Exception:
                    pass
                _set_pack_visible(source_mode_frame, False)
                if ready_label is not None:
                    _set_pack_visible(ready_label, False)

                if source_summary is not None:
                    _set_pack_visible(source_summary, False)
                _set_pack_visible(xml_row, True, fill=tk.X, pady=2, after=host.creator_intro_lbl)
                _set_pack_visible(auto_match_hint, True, anchor=tk.W, fill=tk.X, pady=(2, 6), after=xml_row)
                _set_pack_visible(images_row, True, fill=tk.X, pady=2, after=auto_match_hint)
                _set_pack_visible(output_row, False)
                _set_pack_visible(ratios_frame, True, fill=tk.X, pady=(12, 8), after=images_row)
                _set_pack_visible(augmentation_frame, True, fill=tk.X, pady=(12, 8), after=ratios_frame)
                _set_pack_visible(create_frame, True, fill=tk.X, pady=(12, 10), after=(augmentation_frame or ratios_frame))
                if getattr(progress, "master", None) is not getattr(host, "step4_creator_action_inner", None):
                    _set_pack_visible(progress, True, fill=tk.X, pady=2, after=create_frame)
                if getattr(status, "master", None) is not getattr(host, "step4_creator_action_inner", None):
                    _set_pack_visible(status, True, anchor=tk.W, after=progress)

            try:
                host._refresh_dataset_creator_cta_state()
            except Exception:
                pass
        else:
            _set_pack_visible(getattr(host, "creator_source_mode_frame", None), False)
            _set_pack_visible(getattr(host, "creator_auto_match_hint_lbl", None), False)
    except Exception:
        pass

    try:
        if bool(vm.in_campaign) and vm.mode == "char":
            _set_pack_visible(getattr(host, "split_source_hint_lbl", None), False)
            host.split_source_row.pack_forget()
            host._set_training_widget_text(host.split_intro_lbl, str(vm.split_intro or ""))
            try:
                if bool(vm.show_split_toggle):
                    host._set_training_widget_text(host.btn_step4_split_toggle, str(vm.split_toggle_label or "Popraw split"))
                    if str(host.btn_step4_split_toggle.winfo_manager()) != "pack":
                        host.btn_step4_split_toggle.pack(anchor=tk.W, pady=(0, 8))
                elif str(host.btn_step4_split_toggle.winfo_manager()) == "pack":
                    host.btn_step4_split_toggle.pack_forget()
            except Exception:
                pass
            host._set_training_widget_text(host.btn_step4_split, str(vm.split_action_label or "Utwórz split treningowy"))
            try:
                host._refresh_dataset_split_cta_state()
            except Exception:
                pass
            host._set_training_widget_text(host.split_source_summary_lbl, str(vm.split_summary or ""))
            _refresh_split_campaign_summary_table()
            _set_pack_visible(host.split_source_summary_lbl, False)
            try:
                split_title_anchor = (
                    host.btn_step4_split_toggle
                    if bool(vm.show_split_toggle)
                    and str(host.btn_step4_split_toggle.winfo_manager()) == "pack"
                    else host.split_intro_lbl
                )
            except Exception:
                split_title_anchor = host.split_intro_lbl
            _set_pack_visible(
                getattr(host, "split_campaign_summary_title", None),
                True,
                fill=tk.X,
                pady=(8, 7),
                after=split_title_anchor,
            )
            _set_pack_visible(
                getattr(host, "split_campaign_summary_frame", None),
                True,
                fill=tk.X,
                pady=(0, 12),
                after=(getattr(host, "split_campaign_summary_title", None) or host.split_intro_lbl),
            )
            split_aug_frame = getattr(host, "split_augmentation_frame", None)
            try:
                split_summary_table = getattr(host, "split_campaign_summary_frame", None)
                if bool(vm.show_split_details):
                    _set_pack_visible(host.split_ratios_frame, True, fill=tk.X, pady=(12, 8), after=split_summary_table)
                    _set_pack_visible(split_aug_frame, True, fill=tk.X, pady=(12, 8), after=host.split_ratios_frame)
                    _set_pack_visible(host.btn_step4_split_frame, True, fill=tk.X, pady=(12, 10), after=(split_aug_frame or host.split_ratios_frame))
                else:
                    _set_pack_visible(host.split_ratios_frame, False)
                    _set_pack_visible(split_aug_frame, False)
                    _set_pack_visible(host.btn_step4_split_frame, False)
            except Exception:
                pass
            try:
                split_busy = bool(getattr(host, "dataset_split_is_running", False))
                split_progress = float(getattr(host, "split_progress_var", tk.DoubleVar(value=0.0)).get() or 0.0)
                if bool(vm.show_split_details) or split_busy or split_progress > 0.0:
                    host._set_split_feedback_visibility(True)
                elif str(host.split_feedback_frame.winfo_manager()) == "pack":
                    host.split_feedback_frame.pack_forget()
            except Exception:
                pass
        else:
            host._set_training_widget_text(
                host.split_intro_lbl,
                (
                    "Wskaż źródłowy dataset znaków. PZ1 przygotuje z niego wariant train / val / test dla YOLO Detect."
                ),
            )
            host._set_training_widget_text(host.btn_step4_split, "Utwórz split treningowy")
            try:
                host._refresh_dataset_split_cta_state()
            except Exception:
                pass
            _set_pack_visible(getattr(host, "split_campaign_summary_frame", None), False)
            _set_pack_visible(getattr(host, "split_campaign_summary_title", None), False)
            try:
                host._step4_char_split_details_visible = False
                if str(host.btn_step4_split_toggle.winfo_manager()) == "pack":
                    host.btn_step4_split_toggle.pack_forget()
            except Exception:
                pass
            _set_pack_visible(
                getattr(host, "split_source_hint_lbl", None),
                False,
                anchor=tk.W,
                fill=tk.X,
                pady=(0, 6),
                before=host.split_source_row,
            )
            if str(host.split_source_row.winfo_manager()) != "pack":
                host.split_source_row.pack(fill=tk.X, pady=2, before=host.split_source_summary_lbl)
            if str(host.split_source_summary_lbl.winfo_manager()) == "pack":
                host.split_source_summary_lbl.pack_forget()
            try:
                host.split_output_row.pack_forget()
            except Exception:
                pass
            split_aug_frame = getattr(host, "split_augmentation_frame", None)
            try:
                _set_pack_visible(host.split_ratios_frame, True, fill=tk.X, pady=(12, 8), after=host.split_source_row)
                _set_pack_visible(split_aug_frame, True, fill=tk.X, pady=(12, 8), after=host.split_ratios_frame)
                _set_pack_visible(host.btn_step4_split_frame, True, fill=tk.X, pady=(12, 10), after=(split_aug_frame or host.split_ratios_frame))
            except Exception:
                pass
            try:
                split_busy = bool(getattr(host, "dataset_split_is_running", False))
                split_progress = float(getattr(host, "split_progress_var", tk.DoubleVar(value=0.0)).get() or 0.0)
                if split_busy or split_progress > 0.0:
                    host._set_split_feedback_visibility(True)
            except Exception:
                pass
    except Exception:
        pass


def refresh_step4_training_inputs_mode_ui(host: "TrainingTab"):
    vm = host._get_step4_training_inputs_view_model()

    dataset_section = getattr(host, "train_dataset_section_frame", None)
    dataset_title = getattr(host, "train_dataset_title_lbl", None)
    dataset_intro = getattr(host, "train_dataset_intro_lbl", None)
    dataset_variant_row = getattr(host, "dataset_variant_row", None)
    dataset_variant_title = getattr(host, "dataset_variant_title_lbl", None)
    dataset_selected_path = getattr(host, "train_dataset_selected_path_lbl", None)
    dataset_hint = getattr(host, "train_dataset_hint_lbl", None)
    scope_hint = getattr(host, "train_scope_hint_lbl", None)
    pose_warning = getattr(host, "train_pose_warning_lbl", None)
    base_caption = getattr(host, "train_base_caption_lbl", None)
    base_combo = getattr(host, "base_combo", None)
    custom_row = getattr(host, "custom_row", None)
    custom_entry = getattr(host, "base_custom_entry", None)
    custom_btn = getattr(host, "base_custom_btn", None)

    _set_pack_visible(
        dataset_section,
        bool(vm.show_dataset_section),
        fill=tk.X,
        pady=(0, host._train_left_section_gap),
        before=scope_hint,
    )
    if dataset_title is not None:
        try:
            if not bool(vm.show_dataset_section):
                dataset_title.pack_forget()
            elif not str(dataset_title.winfo_manager()):
                dataset_title.pack(anchor=tk.W, fill=tk.X)
        except Exception:
            pass
    show_dataset_section = bool(vm.show_dataset_section)

    _set_pack_visible(dataset_intro, bool(vm.show_dataset_section), anchor=tk.W, fill=tk.X, pady=(0, 8), after=dataset_title)
    _set_pack_visible(dataset_variant_row, bool(vm.show_dataset_section), fill=tk.X, pady=(4, 8), after=dataset_intro)
    _set_pack_visible(dataset_selected_path, bool(vm.show_dataset_section), anchor=tk.W, fill=tk.X)
    _set_pack_visible(dataset_hint, False, anchor=tk.W, fill=tk.X, pady=(4, 4))
    _set_pack_visible(
        scope_hint,
        bool(vm.show_scope_hint and vm.in_campaign),
        anchor=tk.W,
        fill=tk.X,
        pady=(0, host._train_left_section_gap),
    )
    _set_pack_visible(
        pose_warning,
        (bool(vm.show_pose_warning) and bool(str(getattr(pose_warning, "cget", lambda _x: "")("text") or "").strip())),
        anchor=tk.W,
        fill=tk.X,
        pady=(0, host._train_left_section_gap),
    )

    if dataset_title is not None:
        host._set_training_widget_text(dataset_title, "Wybierz wariant splitu")

    if dataset_variant_title is not None:
        host._set_training_widget_text(dataset_variant_title, "Aktywny wariant")

    try:
        host._refresh_dataset_variant_choices()
    except Exception:
        pass

    base_caption_text = (
        "Od tego modelu zacznie się trening na wybranym datasecie. Możesz użyć modelu projektu, presetu albo własnego .pt."
        if bool(vm.in_campaign)
        else "Od tego modelu zacznie się trening na wybranym datasecie."
    )
    base_caption_text = (
        "Wybierz model, od którego zacznie się nowy run. "
        "To nie jest jeszcze wynik bramki."
    )
    host._set_training_widget_text(base_caption, base_caption_text)

    if base_combo is not None:
        try:
            base_combo.configure(state=str(vm.base_combo_state or "readonly"))
        except Exception:
            pass

    if custom_row is not None:
        try:
            if not bool(vm.show_custom_model):
                custom_row.pack_forget()
            elif not str(custom_row.winfo_manager()):
                custom_row.pack(fill=tk.X, pady=2, after=base_combo)
        except Exception:
            pass

    if custom_entry is not None:
        try:
            custom_entry.configure(state=str(vm.custom_entry_state or "disabled"))
        except Exception:
            pass

    if custom_btn is not None:
        try:
            if not bool(vm.show_custom_pick_button):
                custom_btn.pack_forget()
            elif not str(custom_btn.winfo_manager()):
                custom_btn.pack(side=tk.LEFT, padx=(8, 0))
            if bool(vm.show_custom_pick_button):
                custom_btn.configure(
                    state=tk.NORMAL
                    if host._is_custom_base_model_key(host.base_model_var.get())
                    else tk.DISABLED
                )
        except Exception:
            pass

    if not bool(vm.in_campaign):
        try:
            host._on_base_model_change()
        except Exception:
            pass


def _contrast_text_for_badge(bg_color: str, fallback: str = "#ffffff") -> str:
    raw = str(bg_color or "").strip().lstrip("#")
    if len(raw) != 6:
        return fallback
    try:
        r = int(raw[0:2], 16) / 255.0
        g = int(raw[2:4], 16) / 255.0
        b = int(raw[4:6], 16) / 255.0
    except Exception:
        return fallback

    luminance = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)
    return "#111827" if luminance > 0.58 else "#ffffff"


def _step4_route_panel_copy(campaign_active: bool) -> dict[str, str]:
    if campaign_active:
        return {
            "panel_title": " Wybór toru treningowego ",
            "header_title": " Aktywny tor ",
            "intro": "Wybierz tor pracy tej iteracji. PZ1 przygotuje wariant treningowy, a PZ2 uruchomi trening.",
            "plate_title": "Tor tablic",
            "plate_desc": "YOLO Pose. PZ1 tworzy wariant treningowy z zatwierdzonych anotacji tablic.",
            "char_title": "Tor znaków",
            "char_desc": "YOLO Detect. Źródło pochodzi z T05/PZ3; PZ1 tworzy z niego wariant treningowy.",
        }

    return {
        "panel_title": " Typ datasetu ",
        "header_title": " Aktywny typ datasetu ",
        "intro": "Wybierz typ datasetu. PZ1 buduje wariant treningowy YOLO, a PZ2 uruchamia trening na wybranym wariancie.",
        "plate_title": "Dataset tablic",
        "plate_desc": "YOLO Pose. Źródłem jest XML + zgodny katalog zdjęć albo gotowy dataset; po wyborze XML system spróbuje dobrać katalog zdjęć.",
        "char_title": "Dataset znaków",
        "char_desc": "YOLO Detect. Źródłem jest gotowy dataset znaków, z którego PZ1 utworzy wariant splitu.",
    }


def refresh_step4_route_choice_cards(host: "TrainingTab"):
    cards = getattr(host, "_step4_route_choice_cards", {})
    if not cards:
        return

    palette = getattr(host.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    accent = palette.get("accent", "#0e639c")
    success = palette.get("success", "#4ec9b0")
    accent_text = palette.get("accent_text", "#ffffff")
    surface_info = palette.get("surface_info", panel_alt)
    surface_success = palette.get("surface_success", panel_alt)

    campaign_active = bool(CAMPAIGN.get_active_project_name())
    copy = _step4_route_panel_copy(campaign_active)
    try:
        frame = getattr(host, "step4_route_choice_frame", None)
        if frame is not None:
            frame.configure(text=copy["panel_title"])
    except Exception:
        pass
    try:
        intro = getattr(host, "step4_route_intro_lbl", None)
        if intro is not None:
            intro.configure(text=copy["intro"])
    except Exception:
        pass
    try:
        header = getattr(host, "ds_mode_header_frame", None)
        if header is not None:
            header.configure(text=copy["header_title"])
    except Exception:
        pass

    mode = str(getattr(host, "_step4_dataset_mode", "char") or "char").strip().lower()
    route_selected = bool(getattr(host, "_step4_route_selected", False))

    for card_mode, widgets in cards.items():
        is_active = bool(route_selected and card_mode == mode)
        accent_color = accent if card_mode == "plate" else success
        active_bg = surface_info if card_mode == "plate" else surface_success
        card_bg = active_bg if is_active else panel_alt
        card_border = accent_color if is_active else border
        title_fg = accent_color if is_active else fg
        desc_fg = fg if is_active else muted
        badge_bg = accent_color if is_active else panel_bg
        badge_fg = _contrast_text_for_badge(badge_bg, accent_text) if is_active else fg
        badge_text = "WYBRANO" if is_active else "DO WYBORU"

        frame = widgets.get("frame")
        title_row = widgets.get("title_row")
        title = widgets.get("title")
        badge = widgets.get("badge")
        desc = widgets.get("desc")
        title_text = copy["plate_title"] if card_mode == "plate" else copy["char_title"]
        desc_text = copy["plate_desc"] if card_mode == "plate" else copy["char_desc"]

        for widget in (frame, title_row):
            try:
                if widget is not None:
                    widget.configure(bg=card_bg)
            except Exception:
                pass
        try:
            if frame is not None:
                frame.configure(highlightbackground=card_border, highlightcolor=card_border)
        except Exception:
            pass
        try:
            if title is not None:
                title.configure(text=title_text, bg=card_bg, fg=title_fg)
        except Exception:
            pass
        try:
            if desc is not None:
                desc.configure(text=desc_text, bg=card_bg, fg=desc_fg)
        except Exception:
            pass
        try:
            if badge is not None:
                badge.configure(
                    text=badge_text,
                    width=10,
                    anchor=tk.CENTER,
                    bg=badge_bg,
                    fg=badge_fg,
                    highlightbackground=card_border,
                    highlightcolor=card_border,
                )
        except Exception:
            pass


def _schedule_step4_mode_deferred_refresh(host: "TrainingTab"):
    frame = getattr(host, "frame", None)
    if frame is None:
        return

    pending_job = getattr(host, "_step4_mode_switch_refresh_job", None)
    if pending_job is not None:
        try:
            frame.after_cancel(pending_job)
        except Exception:
            pass

    def run_refresh():
        host._step4_mode_switch_refresh_job = None
        try:
            current_mode = str(getattr(host, "_step4_dataset_mode", "char") or "char").strip().lower()
            host._rebind_free_mode_training_storage(target=current_mode, reload_history=True)
        except Exception:
            pass
        try:
            host._schedule_step4_deferred_model_refresh()
        except Exception:
            try:
                host._refresh_base_model_choices()
                host._refresh_training_recommendation_table()
            except Exception:
                pass
        for callback_name in (
            "_refresh_step4_dataset_summary_table",
            "_refresh_step4_training_inputs_mode_ui",
            "_refresh_step4_campaign_navigation_ui",
            "_refresh_free_training_route_ui",
            "_update_training_dataset_hint",
            "_refresh_training_start_state",
        ):
            try:
                callback = getattr(host, callback_name, None)
                if callable(callback):
                    callback()
            except Exception:
                pass

    try:
        host._step4_mode_switch_refresh_job = frame.after(90, run_refresh)
    except Exception:
        host._step4_mode_switch_refresh_job = None


def refresh_step4_dataset_mode_ui(host: "TrainingTab", *, lightweight: bool = False):
    if not lightweight:
        try:
            host._refresh_free_training_route_ui()
        except Exception:
            pass

    try:
        host._refresh_step4_analysis_tab_visibility()
    except Exception:
        pass

    try:
        refresh_step4_route_choice_cards(host)
    except Exception:
        pass

    if not lightweight:
        try:
            host._refresh_step4_dataset_summary_table()
        except Exception:
            pass

    if not hasattr(host, "ds_mode_host"):
        return

    vm = host._get_step4_dataset_workflow_view_model()
    mode = getattr(host, "_step4_dataset_mode", "char")
    campaign_active = bool(CAMPAIGN.get_active_project_name())
    route_selected = bool(getattr(host, "_step4_route_selected", False))
    locked_target = host._get_locked_campaign_training_target()
    route_locked = campaign_active and locked_target in ("char", "plate")
    mode_header = getattr(host, "ds_mode_header_frame", None)
    hide_mode_header = bool(campaign_active and route_locked and str(mode).lower() == "plate")
    if hide_mode_header:
        try:
            mode_header.pack_forget()
        except Exception:
            pass
    else:
        try:
            if mode_header is not None and str(mode_header.winfo_manager()) != "pack":
                summary_frame = getattr(host, "step4_dataset_summary_frame", None)
                if summary_frame is not None:
                    mode_header.pack(fill=tk.X, before=summary_frame)
                else:
                    mode_header.pack(fill=tk.X)
        except Exception:
            try:
                if mode_header is not None and str(mode_header.winfo_manager()) != "pack":
                    mode_header.pack(fill=tk.X)
            except Exception:
                pass
    waiting_title = "Wybierz tor po lewej stronie" if campaign_active else "Wybierz typ datasetu po lewej stronie"
    waiting_next = "Wybierz tor" if campaign_active else "Wybierz typ datasetu"
    plate_title = "Tor tablic (YOLO Pose)" if campaign_active else "Dataset tablic (YOLO Pose)"
    char_title = "Tor znaków (YOLO Detect)" if campaign_active else "Dataset znaków (YOLO Detect)"

    try:
        host.ds_creator_frame.pack_forget()
    except Exception:
        pass
    try:
        host.ds_split_frame.pack_forget()
    except Exception:
        pass
    try:
        host.ds_mode_waiting_frame.pack_forget()
    except Exception:
        pass

    if bool(vm.show_waiting_panel):
        host.ds_mode_title_var.set(str(vm.title or waiting_title))
        host.ds_mode_desc_var.set(str(vm.description or ""))
        host.btn_choose_plate.configure(state=tk.NORMAL)
        host.btn_choose_char.configure(state=tk.NORMAL)
        refresh_step4_route_choice_cards(host)
        host.ds_mode_waiting_frame.pack(fill=tk.X, expand=False)
        _configure_step4_next_button(host, str(vm.next_label or waiting_next), state=tk.DISABLED)
        return

    if mode == "plate":
        host.ds_mode_title_var.set(str(vm.title or plate_title))
        host.ds_mode_desc_var.set(str(vm.description or ""))
        host.btn_choose_plate.configure(state=tk.DISABLED)
        host.btn_choose_char.configure(state=(tk.DISABLED if route_locked else tk.NORMAL))
        if bool(vm.show_creator_section):
            host.ds_creator_frame.pack(fill=tk.X, expand=False)
        _configure_step4_next_button(host, str(vm.next_label or "Dalej do treningu"))
    else:
        host.ds_mode_title_var.set(str(vm.title or char_title))
        host.ds_mode_desc_var.set(str(vm.description or ""))
        host.btn_choose_plate.configure(state=(tk.DISABLED if route_locked else tk.NORMAL))
        host.btn_choose_char.configure(state=tk.DISABLED)
        if bool(vm.show_split_section):
            host.ds_split_frame.pack(fill=tk.X, expand=False)
        _configure_step4_next_button(host, str(vm.next_label or "Dalej do treningu"))

    if route_locked:
        host.btn_choose_plate.configure(state=tk.DISABLED)
        host.btn_choose_char.configure(state=tk.DISABLED)

    refresh_step4_route_choice_cards(host)

    try:
        refresh_step4_campaign_builder_inputs_ui(host)
    except Exception:
        pass

    if not lightweight:
        try:
            refresh_step4_training_inputs_mode_ui(host)
        except Exception:
            pass

    try:
        host._sync_dataset_mode_canvas_width()
        host._sync_dataset_mode_scrollregion()
    except Exception:
        pass


def refresh_step4_campaign_navigation_ui(host: "TrainingTab"):
    vm = host._get_step4_campaign_navigation_view_model()

    try:
        host._update_step4_notebook_mode()
    except Exception:
        pass

    try:
        host._refresh_step4_analysis_tab_visibility()
    except Exception:
        pass

    try:
        if bool(vm.in_campaign) and bool(vm.dataset_tab_enabled):
            host.main_nb.tab(host.tab_dataset, state="normal")
        host.main_nb.tab(
            host.tab_train,
            state=("normal" if bool(vm.train_tab_enabled) else "disabled")
        )
    except Exception:
        pass

    try:
        _configure_step4_next_button(
            host,
            str(vm.next_label or "Dalej do treningu"),
            state=(tk.NORMAL if bool(vm.next_enabled) else tk.DISABLED),
        )
    except Exception:
        pass

    try:
        host.btn_step4_back.configure(text=str(vm.dataset_back_label or "Wstecz"))
        if not bool(vm.show_dataset_back):
            host.btn_step4_back.grid_remove()
        else:
            host.btn_step4_back.grid()
    except Exception:
        pass

    try:
        if bool(vm.force_dataset_tab_selection) and str(host.main_nb.select()) == str(host.tab_train):
            host.main_nb.select(host.tab_dataset)
    except Exception:
        pass

    if not hasattr(host, "step4_train_nav"):
        return

    try:
        if bool(vm.show_train_nav):
            host.step4_train_nav.grid()
        else:
            host.step4_train_nav.grid_remove()
    except Exception:
        pass

    try:
        host._update_training_dataset_hint()
    except Exception:
        pass

    try:
        train_back_label = str(vm.train_back_label or "Wstecz do toru")
        train_back_width = max(18, min(34, len(train_back_label) + 2))
        host.btn_step4_train_back.configure(text=train_back_label, width=train_back_width)
        if not bool(vm.show_train_back):
            host.btn_step4_train_back.pack_forget()
        elif not str(host.btn_step4_train_back.winfo_manager()):
            host.btn_step4_train_back.pack(side=tk.RIGHT, padx=(8, 0))
    except Exception:
        pass

    try:
        host.btn_step4_finish.configure(state=tk.DISABLED)
    except Exception:
        pass

    try:
        active_target = str(host.get_campaign_training_target() or "").strip().lower()
        show_dataset_adjust_cta = bool(
            CAMPAIGN.get_active_project_name()
            and active_target in {"char", "plate"}
            and bool(getattr(host, "_step4_train_unlocked", False))
        )
        if show_dataset_adjust_cta:
            host.btn_step4_finish.configure(
                text="Stwórz inny wariant datasetu",
                command=host._open_step4_dataset_stage,
                width=28,
                state=(tk.DISABLED if host._step4_has_active_operation() else tk.NORMAL),
            )
            if not str(host.btn_step4_finish.winfo_manager()):
                host.btn_step4_finish.pack(side=tk.LEFT)
        else:
            host.btn_step4_finish.pack_forget()
    except Exception:
        pass

    try:
        if bool(vm.show_complete_project):
            if not str(host.btn_step4_complete_project.winfo_manager()):
                host.btn_step4_complete_project.pack(side=tk.LEFT, padx=(0, 8))
        else:
            host.btn_step4_complete_project.pack_forget()
    except Exception:
        pass


def open_step4_dataset_stage(host: "TrainingTab"):
    try:
        host.main_nb.select(host.tab_dataset)
    except Exception:
        pass

    try:
        host._guide_step4_builder_action()
    except Exception:
        pass


def _normalize_training_context_target(target: str | None) -> str:
    value = str(target or "").strip().lower()
    return value if value in ("char", "plate") else "char"


def accept_training_input_context(
    host: "TrainingTab",
    *,
    source: str = "",
    target: str = "",
    dataset_path: str | Path | None = None,
    select_training: bool = False,
) -> bool:
    mode = _normalize_training_context_target(target or getattr(host, "_step4_dataset_mode", "char"))
    campaign_active = bool(CAMPAIGN.get_active_project_name())
    locked_target = host._get_locked_campaign_training_target()
    if campaign_active and locked_target in ("char", "plate"):
        mode = locked_target

    dataset_text = str(dataset_path or "").strip()
    if dataset_text:
        try:
            host.dataset_var.set(dataset_text)
        except Exception:
            pass

    try:
        host._step4_dataset_mode = mode
    except Exception:
        pass
    try:
        host.set_campaign_training_target(mode)
    except Exception:
        pass
    try:
        host._step4_route_selected = True
    except Exception:
        pass
    try:
        host._step4_train_unlocked = True
    except Exception:
        pass
    try:
        builder = getattr(host, "_build_step4_dataset_training_source", None)
        if callable(builder):
            host._last_training_source = builder(
                dataset_text,
                target=mode,
                provenance=str(source or "pz1"),
            )
    except Exception:
        pass

    if mode == "plate" and dataset_text:
        try:
            host._set_creator_source_mode("ready")
        except Exception:
            pass

    if not campaign_active:
        try:
            host._rebind_free_mode_training_storage(target=mode)
        except Exception:
            pass
        try:
            host.rank_models_dir.set(str(host._get_ranking_models_default_dir()))
        except Exception:
            pass

    for callback_name in (
        "_refresh_base_model_choices",
        "_refresh_training_recommendation_table",
        "_refresh_dataset_variant_choices",
        "_update_step4_notebook_mode",
        "_refresh_step4_dataset_mode_ui",
        "_refresh_step4_campaign_navigation_ui",
        "_refresh_free_training_route_ui",
        "_update_training_dataset_hint",
        "_refresh_training_start_state",
    ):
        try:
            callback = getattr(host, callback_name, None)
            if callable(callback):
                callback()
        except Exception:
            pass

    if select_training:
        try:
            host.main_nb.select(host.tab_train)
        except Exception:
            pass
        try:
            host._select_step4_analysis_tab(host.hist_tab)
        except Exception:
            pass

    return True


def mark_step4_dataset_ready(
    host: "TrainingTab",
    dataset_path: str | Path | None = None,
    *,
    target: str | None = None,
):
    if dataset_path:
        try:
            host.dataset_var.set(str(dataset_path))
        except Exception:
            pass

    try:
        dataset_text = str(dataset_path or host.dataset_var.get() or "").strip()
    except Exception:
        dataset_text = str(dataset_path or "").strip()

    target = _normalize_training_context_target(target or getattr(host, "_step4_dataset_mode", "char"))
    try:
        host._step4_dataset_mode = target
    except Exception:
        pass
    try:
        host.set_campaign_training_target(target)
    except Exception:
        pass
    try:
        builder = getattr(host, "_build_step4_dataset_training_source", None)
        if callable(builder):
            host._last_training_source = builder(
                dataset_text,
                target=target,
                provenance="pz1",
            )
    except Exception:
        pass

    if dataset_text and CAMPAIGN.get_active_project_name():
        try:
            root = Path(dataset_text)
            if root.is_file() and root.name.lower() == "data.yaml":
                yaml_path = root
                root = root.parent
            else:
                yaml_path = root / "data.yaml"
            try:
                dataset_dir = str(root.resolve())
            except Exception:
                dataset_dir = str(root)
            try:
                yaml_text = str(yaml_path.resolve()) if yaml_path.exists() else str(yaml_path)
            except Exception:
                yaml_text = str(yaml_path)
            counts_getter = getattr(host, "_get_dataset_split_image_counts", None)
            counts = dict(counts_getter(root) or {}) if callable(counts_getter) else {}
            payload = {
                "target": target,
                "dataset_path": dataset_dir,
                "yaml_path": yaml_text,
                "train_images": int(counts.get("train", 0) or 0),
                "val_images": int(counts.get("val", 0) or 0),
                "test_images": int(counts.get("test", 0) or 0),
                "total_images": int(counts.get("total", 0) or 0),
                "source_stage": "Z4/PZ1",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            if target == "char":
                source_dataset = ""
                source_yaml = ""
                source_stage = ""
                try:
                    source = getattr(host, "_pending_step4_input_training_source", None)
                    source_dataset = str(getattr(source, "dataset_dir", "") or "").strip()
                    source_yaml = str(getattr(source, "yaml_path", "") or "").strip()
                    source_stage = str(getattr(source, "source_stage", "") or "").strip()
                except Exception:
                    source_dataset = ""
                    source_yaml = ""
                    source_stage = ""
                if not source_dataset and not source_yaml:
                    try:
                        source_dataset = str(host.split_src_var.get() or "").strip()
                    except Exception:
                        source_dataset = ""
                if source_dataset or source_yaml:
                    try:
                        source_root = Path(source_yaml or source_dataset)
                        if source_root.is_file() and source_root.name.lower() == "data.yaml":
                            source_yaml = str(source_root)
                            source_root = source_root.parent
                        elif source_root:
                            source_yaml = source_yaml or str(source_root / "data.yaml")
                        source_dataset = str(source_root.resolve()) if source_root.exists() else str(source_root)
                    except Exception:
                        pass
                    payload["source_dataset"] = source_dataset
                    payload["source_yaml"] = source_yaml
                    payload["source_dataset_stage"] = source_stage or "Z3/PZ3"
            if target == "plate":
                try:
                    manifest_loader = getattr(host, "_load_plate_dataset_source_manifest", None)
                    manifest = dict(manifest_loader(root) or {}) if callable(manifest_loader) else {}
                except Exception:
                    manifest = {}
                if manifest:
                    payload["source_manifest"] = {
                        "project": str(manifest.get("project", "") or ""),
                        "iteration": int(manifest.get("iteration", 0) or 0),
                        "approved_set_images": int(manifest.get("approved_set_images", 0) or 0),
                        "approved_set_plates": int(manifest.get("approved_set_plates", 0) or 0),
                        "source_kind": str(manifest.get("source_kind", "") or ""),
                    }
            try:
                iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
            except Exception:
                iteration_num = 1
            payload["iteration"] = int(iteration_num)
            payload["dataset_iteration"] = int(iteration_num)
            CAMPAIGN.upsert_iteration_state(
                iteration_num=iteration_num,
                updates={"step4_dataset": payload},
            )
            CAMPAIGN.upsert_iteration_artifact_bundle(
                iteration_num=iteration_num,
                updates={"step4_dataset": payload},
            )
        except Exception:
            pass

    if target == "plate" and dataset_text and CAMPAIGN.get_active_project_name():
        try:
            remember_source = getattr(host, "_remember_campaign_plate_training_source", None)
            if callable(remember_source):
                remember_source(dataset_text)
        except Exception:
            pass
        try:
            resolver = getattr(host, "_resolve_plate_training_source_from_dataset", None)
            source_info = resolver(dataset_text) if callable(resolver) else {"dataset_path": dataset_text}
            CAMPAIGN.set_last_plate_training_source(
                dataset_path=str(source_info.get("dataset_path", dataset_text) or dataset_text),
                source_run_path=str(source_info.get("source_run_path", "") or ""),
                source_xml_path=str(source_info.get("source_xml_path", "") or ""),
            )
        except Exception:
            pass

    if target == "plate" and dataset_text:
        try:
            host._set_creator_source_mode("ready")
        except Exception:
            pass

    try:
        host._step4_route_selected = True
    except Exception:
        pass
    try:
        host._step4_train_unlocked = True
    except Exception:
        pass

    try:
        host._refresh_step4_campaign_navigation_ui()
    except Exception:
        pass

    try:
        host._guide_step4_next_action()
    except Exception:
        pass
    try:
        host._refresh_step4_dataset_mode_ui()
    except Exception:
        pass
    try:
        host._refresh_dataset_variant_choices()
    except Exception:
        pass
    try:
        host._update_training_dataset_hint()
    except Exception:
        pass
    try:
        host._refresh_training_start_state()
    except Exception:
        pass


def set_step4_dataset_mode(host: "TrainingTab", mode: str, *, show_locked_message: bool = True):
    mode = (mode or "char").strip().lower()
    if mode not in ("char", "plate"):
        mode = "char"

    previous_mode = str(getattr(host, "_step4_dataset_mode", "char") or "char").strip().lower()
    if previous_mode not in ("char", "plate"):
        previous_mode = "char"
    route_changed = mode != previous_mode
    clear_dataset_for_route_change = False
    campaign_active = bool(CAMPAIGN.get_active_project_name())
    mode_name = "tor treningu" if campaign_active else "typ datasetu"

    if route_changed:
        active_label = host._get_active_step4_operation_label()
        if active_label:
            messagebox.showinfo(
                "Proces w toku",
                f"Nie można teraz zmienić ustawienia: {mode_name}.\n\n"
                f"Najpierw poczekaj na zakończenie: {active_label}.",
            )
            return

    locked_target = host._get_locked_campaign_training_target()
    if campaign_active and locked_target in ("char", "plate"):
        if mode != locked_target:
            if show_locked_message:
                messagebox.showinfo(
                    "Tor iteracji jest stały",
                    "Tor treningu jest stały w bieżącej iteracji.\n\n"
                    f"Ta iteracja pozostaje w torze {host._format_training_target_label(locked_target)}."
                )
        mode = locked_target
        route_changed = mode != previous_mode

    if route_changed and not campaign_active:
        try:
            current_dataset = str(host.dataset_var.get() or "").strip()
        except Exception:
            current_dataset = ""
        if current_dataset:
            try:
                inferred_target = str(host._infer_dataset_target(current_dataset) or "").strip().lower()
            except Exception:
                inferred_target = ""
            if inferred_target in {"char", "plate", "vehicle"} and inferred_target != mode:
                previous_label = host._format_training_target_label(previous_mode)
                new_label = host._format_training_target_label(mode)
                dataset_label = host._format_training_target_label(inferred_target)
                try:
                    display_path = host._format_workspace_relative_path(current_dataset)
                except Exception:
                    display_path = current_dataset
                confirm_switch = messagebox.askyesno(
                    "Zmiana typu datasetu",
                    "Wybrany dataset wygląda na inny typ danych niż ten, który chcesz teraz przygotować.\n\n"
                    f"Obecny typ: {previous_label}\n"
                    f"Nowy typ: {new_label}\n"
                    f"Dataset: {display_path}\n"
                    f"Rozpoznany jako: {dataset_label}\n\n"
                    "Po zmianie typu odłączymy ten dataset, wyczyścimy wariant splitu i kontekst PZ2. "
                    "Eksporty oraz pliki na dysku pozostaną bez zmian.\n\n"
                    "Kontynuować zmianę typu datasetu?",
                    parent=getattr(host, "frame", None),
                )
                if not confirm_switch:
                    return
                clear_dataset_for_route_change = True

    host._step4_dataset_mode = mode
    host.set_campaign_training_target(mode)
    if campaign_active:
        host._step4_route_selected = True
        host._step4_train_unlocked = False
    else:
        if route_changed:
            try:
                host._pending_step4_input_training_source = None
            except Exception:
                pass
            try:
                host._step4_dataset_summary_split_counts = {}
            except Exception:
                pass
        if clear_dataset_for_route_change:
            try:
                host.dataset_var.set("")
            except Exception:
                pass
            try:
                host.dataset_variant_var.set("")
            except Exception:
                pass
            try:
                host._last_training_source = None
            except Exception:
                pass
        elif mode == "plate":
            try:
                current_dataset = str(host.dataset_var.get() or "").strip()
            except Exception:
                current_dataset = ""
            if current_dataset:
                try:
                    inferred_target = str(host._infer_dataset_target(current_dataset) or "").strip().lower()
                except Exception:
                    inferred_target = ""
                if not inferred_target or inferred_target == "plate":
                    try:
                        host._set_creator_source_mode("ready")
                    except Exception:
                        pass
        if route_changed and not clear_dataset_for_route_change:
            try:
                current_dataset = str(host.dataset_var.get() or "").strip()
            except Exception:
                current_dataset = ""
            if not current_dataset:
                try:
                    host._last_training_source = None
                except Exception:
                    pass

        host._rebind_free_mode_training_storage(target=mode, reload_history=False)
        try:
            host.rank_models_dir.set(str(host._get_ranking_models_default_dir()))
        except Exception:
            pass
    if campaign_active:
        try:
            host._refresh_base_model_choices()
        except Exception:
            pass
        try:
            host._refresh_training_recommendation_table()
        except Exception:
            pass
        try:
            host._refresh_dataset_variant_choices()
        except Exception:
            pass
        host._refresh_step4_dataset_mode_ui()
        host._refresh_step4_campaign_navigation_ui()
        try:
            host._refresh_free_training_route_ui()
        except Exception:
            pass
        try:
            host._update_training_dataset_hint()
        except Exception:
            pass
    else:
        refresh_step4_dataset_mode_ui(host, lightweight=True)
        try:
            host._refresh_step4_dataset_summary_table()
        except Exception:
            pass
        _schedule_step4_mode_deferred_refresh(host)

    try:
        label = "tablic (YOLO Pose)" if mode == "plate" else "znaków (YOLO Detect)"
        if campaign_active:
            host._append_step4_builder_log(f"[TRYB] Wybrano tor budowy datasetu dla modelu {label}.")
        else:
            host._append_step4_builder_log(f"[TYP] Wybrano typ datasetu dla modelu {label}.")
    except Exception:
        pass

    if campaign_active:
        try:
            host._guide_step4_builder_action()
        except Exception:
            pass


def step4_dataset_go_next(host: "TrainingTab"):
    if CAMPAIGN.get_active_project_name() and not getattr(host, "_step4_train_unlocked", False):
        return

    try:
        mode = getattr(host, "_step4_dataset_mode", "char")
        host.set_campaign_training_target(mode)
    except Exception:
        pass

    try:
        host._ensure_step4_train_tab_built()
    except Exception:
        pass

    try:
        host._refresh_step4_campaign_navigation_ui()
    except Exception:
        pass

    try:
        host.main_nb.select(host.tab_train)
    except Exception:
        pass

    try:
        host._select_step4_analysis_tab(host.hist_tab)
    except Exception:
        pass

    if CAMPAIGN.get_active_project_name():
        try:
            if getattr(host, "_step4_campaign_finish_ready", False):
                host._guide_step4_finish_action()
            else:
                host._guide_step4_training_action()
        except Exception:
            pass


def step4_dataset_go_back(host: "TrainingTab"):
    if CAMPAIGN.get_active_project_name():
        return_to_campaign_from_step4(host)
        return

    host._append_step4_builder_log(
        "[NAWIGACJA] Tryb swobodny: brak poprzedniego kroku wizardowego do otwarcia."
    )


def step4_train_go_back(host: "TrainingTab"):
    if CAMPAIGN.get_active_project_name():
        try:
            return_to_campaign_from_step4(host)
            return
        except Exception:
            pass

    try:
        open_step4_dataset_stage(host)
    except Exception:
        try:
            host.main_nb.select(host.tab_dataset)
        except Exception:
            pass

    try:
        host._refresh_step4_dataset_mode_ui()
    except Exception:
        pass

    try:
        host._refresh_free_training_route_ui()
    except Exception:
        pass


def refresh_step4_analysis_tab_visibility(host: "TrainingTab"):
    if not hasattr(host, "right_nb") or not hasattr(host, "ranking_tab"):
        return

    ranking_enabled = host._is_ranking_available_for_selected_target()

    if ranking_enabled:
        if not getattr(host, "_step4_ranking_tab_visible", False):
            try:
                host.right_nb.add(host.ranking_tab, text="Ranking")
            except Exception:
                try:
                    host.right_nb.insert("end", host.ranking_tab, text="Ranking")
                except Exception:
                    pass
            host._step4_ranking_tab_visible = True
        else:
            try:
                host.right_nb.tab(host.ranking_tab, text="Ranking", state="normal")
            except Exception:
                pass
        if host._is_ranking_tab_active():
            try:
                host._prefill_ranking_reference_if_empty()
            except Exception:
                pass
            try:
                host._refresh_ranking_reference_ui()
            except Exception:
                pass
        return

    try:
        if str(host.right_nb.select()) == str(host.ranking_tab):
            host.right_nb.select(host.hist_tab)
    except Exception:
        pass

    if getattr(host, "_step4_ranking_tab_visible", False):
        try:
            host.right_nb.hide(host.ranking_tab)
        except Exception:
            pass
        host._step4_ranking_tab_visible = False


def sync_step4_analysis_nav_buttons(host: "TrainingTab", event=None):
    try:
        refresh_step4_analysis_tab_visibility(host)
    except Exception:
        pass


def resolve_step4_guidance_buttons(host: "TrainingTab", attr_name: str):
    if attr_name == "step4_route_panel_frame":
        return []
    if attr_name == "btn_step4_start_train_frame":
        return [getattr(host, "btn_start_train", None)]
    if attr_name == "btn_step4_finish_frame":
        return [
            getattr(host, "btn_step4_finish", None),
            getattr(host, "btn_step4_complete_project", None),
        ]

    candidates = [attr_name]
    if attr_name.endswith("_frame"):
        candidates.append(attr_name[:-6])

    resolved = []
    for candidate in candidates:
        widget = getattr(host, candidate, None)
        try:
            import tkinter.ttk as ttk  # local import to avoid unused at top
            if isinstance(widget, ttk.Button):
                resolved.append(widget)
        except Exception:
            pass

    unique_buttons = []
    seen = set()
    for btn in resolved:
        if btn is None:
            continue
        btn_id = str(btn)
        if btn_id not in seen:
            unique_buttons.append(btn)
            seen.add(btn_id)
    return unique_buttons


def resolve_step4_guidance_frame(host: "TrainingTab", attr_name: str):
    if not attr_name:
        return None

    if attr_name == "step4_route_panel_frame":
        return getattr(host, "step4_route_panel_frame", None)
    if attr_name == "btn_step4_start_train_frame":
        return getattr(host, "btn_step4_start_train_pulse_frame", None)

    candidates = [attr_name]

    for candidate in candidates:
        widget = getattr(host, candidate, None)
        if isinstance(widget, tk.Frame):
            return widget

    return None


def set_step4_emphasis(host: "TrainingTab", frame_attr: str, enabled: bool, color: str = "#f39c12"):
    frame = resolve_step4_guidance_frame(host, frame_attr)
    if frame is not None:
        try:
            bg = host.app.palette.get("bg", "#1e1e1e") if frame_attr == "step4_route_panel_frame" else host.app.palette.get("panel", "#252526")
            host.app.set_frame_emphasis(frame, enabled, background=bg)
        except Exception:
            pass

    buttons = resolve_step4_guidance_buttons(host, frame_attr)
    for btn in buttons:
        try:
            host.app.set_button_emphasis(btn, enabled)
        except Exception:
            pass


def clear_step4_guidance(host: "TrainingTab"):
    for attr_name in (
        "step4_route_panel_frame",
        "btn_step4_create_frame",
        "btn_step4_split_frame",
        "btn_step4_next_frame",
        "btn_step4_start_train_frame",
        "btn_step4_finish_frame",
        "btn_step4_train_back",
    ):
        try:
            set_step4_emphasis(host, attr_name, False)
        except Exception:
            pass


def guide_step4_route_selection(host: "TrainingTab"):
    if not CAMPAIGN.get_active_project_name():
        return
    clear_step4_guidance(host)
    set_step4_emphasis(host, "step4_route_panel_frame", True)


def guide_step4_builder_action(host: "TrainingTab"):
    if not CAMPAIGN.get_active_project_name():
        return
    clear_step4_guidance(host)
    frame_attr = "btn_step4_create_frame" if host._step4_dataset_mode == "plate" else "btn_step4_split_frame"
    set_step4_emphasis(host, frame_attr, True)


def guide_step4_next_action(host: "TrainingTab"):
    if not CAMPAIGN.get_active_project_name():
        return
    clear_step4_guidance(host)
    set_step4_emphasis(host, "btn_step4_next_frame", True)


def guide_step4_training_action(host: "TrainingTab"):
    if not CAMPAIGN.get_active_project_name():
        return
    clear_step4_guidance(host)
    set_step4_emphasis(host, "btn_step4_start_train_frame", True)


def guide_step4_finish_action(host: "TrainingTab"):
    if not CAMPAIGN.get_active_project_name():
        return
    clear_step4_guidance(host)
    set_step4_emphasis(host, "btn_step4_train_back", True)
