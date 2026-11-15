#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Osadzony podgląd próbek AT z historii runów Z2."""

from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk

MARGIN_RATIO = 0.15


def select_evenly_spaced(items, limit=8):
    items = list(items or [])
    limit = max(1, int(limit or 8))
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    last = len(items) - 1
    indices = []
    for slot in range(limit):
        idx = int(round(slot * last / (limit - 1)))
        if idx not in indices:
            indices.append(idx)
    return [items[idx] for idx in indices]


def detection_bounds(det):
    polygon = list(getattr(det, "polygon", None) or [])
    if len(polygon) >= 3:
        pts = [(float(x), float(y)) for x, y in polygon]
        return (
            min(x for x, _ in pts), min(y for _, y in pts),
            max(x for x, _ in pts), max(y for _, y in pts),
        )
    bbox = list(getattr(det, "bbox", None) or [])
    if len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def compute_crop_box(image_size, bounds, margin_ratio=MARGIN_RATIO):
    iw, ih = [max(1, int(v)) for v in image_size]
    x1, y1, x2, y2 = [float(v) for v in bounds]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    mx = max(8.0, bw * float(margin_ratio))
    my = max(8.0, bh * float(margin_ratio))
    left = max(0, int(x1 - mx))
    top = max(0, int(y1 - my))
    right = min(iw, int(x2 + mx + 0.999))
    bottom = min(ih, int(y2 + my + 0.999))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def collect_run_history_preview_items(host, run_dir, limit=8):
    safe = host._resolve_safe_annotation_run_dir(run_dir, require_xml=True)
    if safe is None:
        return {"ok": False, "items": [], "reason": "invalid_run"}
    safe = Path(safe)
    try:
        annotations = list(host._parse_cvat_preview_annotations(safe / "annotations.xml") or [])
    except Exception as exc:
        return {"ok": False, "items": [], "reason": "xml_read_failed", "error": str(exc)}

    try:
        manifest = dict(host._load_annotation_run_manifest(safe) or {})
    except Exception:
        manifest = {}

    try:
        image_dir = host._resolve_run_image_dir_for_annotations(annotations, manifest, safe)
    except Exception:
        image_dir = None

    candidates = []
    for ann in annotations:
        filename = str(getattr(ann, "filename", "") or "").strip()
        if not filename:
            continue
        try:
            plates = list(host._get_plate_detections(ann) or [])
        except Exception:
            plates = []
        for local_index, det in enumerate(plates, 1):
            if detection_bounds(det) is None:
                continue
            candidates.append({
                "filename": filename,
                "detection": det,
                "local_index": local_index,
                "image_path": _resolve_image_path(safe, image_dir, filename),
            })

    try:
        images_with_plates, total_plates = host._get_run_plate_annotation_counts(safe)
    except Exception:
        total_plates = len(candidates)
        images_with_plates = len({x["filename"] for x in candidates})

    return {
        "ok": True,
        "run_dir": safe,
        "items": select_evenly_spaced(candidates, limit),
        "images_with_plates": int(images_with_plates or 0),
        "total_plates": int(total_plates or 0),
    }


def clear_run_history_preview_inline(host, hide=True):
    grid = getattr(host, "manual_history_preview_grid", None)
    if grid is not None:
        try:
            for child in list(grid.winfo_children()):
                child.destroy()
        except Exception:
            pass
    host._manual_history_preview_photo_refs = []
    host._manual_history_preview_loaded = False
    if hide:
        panel = getattr(host, "manual_history_preview_host", None)
        if panel is not None:
            try:
                panel.pack_forget()
            except Exception:
                pass


def render_run_history_preview_inline(host, run_dir, limit=8):
    summary = collect_run_history_preview_items(host, run_dir, limit=limit)
    parent = getattr(host, "frame", None)
    parent = parent.winfo_toplevel() if parent is not None else None

    if not summary.get("ok"):
        messagebox.showwarning(
            "Podgląd anotacji",
            "Nie udało się odczytać wybranego runu Z2.",
            parent=parent,
        )
        return False

    items = list(summary.get("items") or [])
    if not items:
        messagebox.showinfo(
            "Podgląd anotacji",
            "Wybrany run nie zawiera anotacji tablic do pokazania.",
            parent=parent,
        )
        return False

    panel = getattr(host, "manual_history_preview_host", None)
    grid = getattr(host, "manual_history_preview_grid", None)
    title_var = getattr(host, "manual_history_preview_title_var", None)
    if panel is None or grid is None:
        return False

    for child in list(grid.winfo_children()):
        child.destroy()

    palette = getattr(getattr(host, "app", None), "palette", {}) or {}
    field = palette.get("field", "#1a1a1a")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    accent = palette.get("accent", "#4f8de3")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

    if title_var is not None:
        title_var.set(
            f"{Path(summary['run_dir']).name}  •  "
            f"AT: {summary['total_plates']}  •  próbka: {len(items)}"
        )

    refs = []
    cols = 2
    rows = max(1, (len(items) + cols - 1) // cols)
    for col in range(cols):
        grid.grid_columnconfigure(col, weight=1, uniform="history_at")
    for row in range(rows):
        grid.grid_rowconfigure(row, weight=1, uniform="history_at")

    for idx, item in enumerate(items):
        card = tk.Frame(
            grid,
            bg=field,
            highlightthickness=1,
            highlightbackground=border,
        )
        card.grid(
            row=idx // cols,
            column=idx % cols,
            sticky="nsew",
            padx=6,
            pady=6,
        )

        photo = None
        error = ""
        if item.get("image_path") is not None:
            try:
                image = _render_detection_crop(
                    Path(item["image_path"]),
                    item["detection"],
                    outline=accent,
                    max_size=(520, 190),
                )
                photo = ImageTk.PhotoImage(image)
                refs.append(photo)
            except Exception as exc:
                error = str(exc)
        else:
            error = "Nie odnaleziono obrazu źródłowego."

        if photo is not None:
            tk.Label(card, image=photo, bg=field, bd=0).pack(
                fill=tk.BOTH, expand=True, padx=6, pady=(6, 2)
            )
        else:
            tk.Label(
                card,
                text="Brak podglądu",
                bg=field,
                fg=muted,
            ).pack(fill=tk.BOTH, expand=True, padx=6, pady=(10, 2))

        caption = (
            f"{item['filename']}  •  "
            f"AT #{int(item.get('local_index') or 1)}"
        )
        if error:
            caption += f"\n{error}"
        tk.Label(
            card,
            text=caption,
            bg=field,
            fg=fg if not error else muted,
            anchor="w",
            justify=tk.LEFT,
            wraplength=500,
            font=("Segoe UI", 8),
        ).pack(fill=tk.X, padx=8, pady=(2, 7))

    host._manual_history_preview_photo_refs = refs
    host._manual_history_preview_loaded = True

    try:
        host._refresh_preview_workspace_visibility(manual_review_active=False)
    except Exception:
        panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=0)

    return True


def open_run_history_preview(host, run_dir, limit=8):
    return render_run_history_preview_inline(host, run_dir, limit=limit)


def _resolve_image_path(run_dir, image_dir, filename):
    candidates = []
    if image_dir is not None:
        candidates.append(Path(image_dir) / Path(filename))
    candidates.extend([
        Path(run_dir) / "images" / Path(filename),
        Path(run_dir) / Path(filename),
    ])
    for path in candidates:
        try:
            if path.exists() and path.is_file():
                return path
        except Exception:
            pass
    return None


def _render_detection_crop(image_path, detection, outline, max_size):
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    bounds = detection_bounds(detection)
    if bounds is None:
        raise ValueError("AT nie ma poprawnej geometrii.")

    left, top, right, bottom = compute_crop_box(image.size, bounds)
    crop = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(crop)
    stroke = max(2, int(round(max(crop.size) / 220.0)))

    polygon = list(getattr(detection, "polygon", None) or [])
    if len(polygon) >= 3:
        pts = [(float(x) - left, float(y) - top) for x, y in polygon]
        draw.line(pts + [pts[0]], fill=outline, width=stroke)
    else:
        x1, y1, x2, y2 = bounds
        draw.rectangle(
            (x1-left, y1-top, x2-left, y2-top),
            outline=outline,
            width=stroke,
        )

    crop.thumbnail(max_size, Image.Resampling.LANCZOS)
    return crop
