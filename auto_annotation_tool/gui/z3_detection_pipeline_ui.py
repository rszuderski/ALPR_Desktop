from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING

from ..campaign_manager import CAMPAIGN
from .z3_detection_controls_ui import normalize_detection_method_key
from .z3_model_metadata_dialog import show_yolo_model_metadata_dialog
from .web_slim_scrollbar import blend_hex_colors

if TYPE_CHECKING:
    from .tab_character_annotation import CharacterAnnotationTab


def _is_step3_campaign_runtime(host: "CharacterAnnotationTab") -> bool:
    try:
        return bool(getattr(host, "_step3_linear_mode", False) and CAMPAIGN.get_active_project_name())
    except Exception:
        return False


def _pipeline_requires_yolo(blocks_or_compiled) -> bool:
    if isinstance(blocks_or_compiled, dict):
        if bool(blocks_or_compiled.get("requires_yolo")):
            return True
        blocks = blocks_or_compiled.get("blocks", [])
    else:
        blocks = blocks_or_compiled
    try:
        return any(str(block or "").strip().lower().startswith("yolo") for block in (blocks or []))
    except Exception:
        return False


def _method_requires_yolo(method_key: str | None) -> bool:
    return str(method_key or "").strip().upper() in {"YOLO", "BOTH", "YOLO_OCR", "YOLO_BOX", "YOLO_SYMBOL"}


def _host_has_configured_yolo_detection_model(host) -> bool:
    checker = getattr(host, "_has_configured_yolo_detection_model", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    try:
        model_path = str(getattr(host, "_get_effective_yolo_model_path")() or "").strip()
        return bool(model_path and Path(model_path).exists())
    except Exception:
        return False


def _pipeline_allowed_in_current_context(host, blocks) -> bool:
    compiled = compile_detection_pipeline_blocks(blocks)
    if not bool(compiled.get("valid")):
        return False
    if _pipeline_requires_yolo(compiled) and not _host_has_configured_yolo_detection_model(host):
        return False
    return True


def get_detection_pipeline_blocks(host, method_key: str | None, method_labels: dict, key_by_label: dict) -> list[str]:
    current_method = normalize_detection_method_key(
        host._get_detection_method_key(),
        method_labels,
        key_by_label,
    )
    requested_method = normalize_detection_method_key(
        method_key or current_method,
        method_labels,
        key_by_label,
    )
    saved_blocks = get_saved_detection_pipeline_blocks(host)
    if saved_blocks:
        compiled_saved = compile_detection_pipeline_blocks(saved_blocks)
        if bool(compiled_saved.get("valid")):
            explicit_other_method = method_key is not None and requested_method != current_method
            if not explicit_other_method or str(compiled_saved.get("method_key") or "") == requested_method:
                return saved_blocks

    resolved_method = requested_method
    if _method_requires_yolo(resolved_method) and not _host_has_configured_yolo_detection_model(host):
        resolved_method = "OCR"
    if resolved_method == "YOLO":
        return ["yolo_box", "yolo_symbol"]
    if resolved_method == "YOLO_BOX":
        return ["yolo_box"]
    if resolved_method == "YOLO_SYMBOL":
        return ["yolo_symbol"]
    if resolved_method == "BOTH":
        blocks = ["ocr_symbol", "yolo_box"]
        if host._get_hybrid_rescue_max_chars() > 0:
            blocks.append("yolo_symbol")
        return blocks
    if resolved_method == "YOLO_OCR":
        return ["yolo_box", "ocr_symbol"]
    return ["ocr_symbol"]


def compile_detection_pipeline_blocks(blocks=None, method_card_meta: dict | None = None) -> dict:
    prepared = normalize_detection_pipeline_blocks(blocks)
    valid_patterns = {
        ("ocr_symbol",): ("OCR", False),
        ("yolo_box",): ("YOLO_BOX", False),
        ("yolo_symbol",): ("YOLO_SYMBOL", False),
        ("yolo_box", "yolo_symbol"): ("YOLO", False),
        ("ocr_symbol", "yolo_box"): ("BOTH", False),
        ("ocr_symbol", "yolo_box", "yolo_symbol"): ("BOTH", True),
        ("yolo_box", "ocr_symbol"): ("YOLO_OCR", False),
    }
    compiled = valid_patterns.get(tuple(prepared))
    if compiled is None:
        return {
            "valid": False,
            "blocks": prepared,
            "method_key": None,
            "rescue_enabled": False,
            "requires_yolo": any(block.startswith("yolo") for block in prepared),
            "status_text": (
                "Ten łańcuch nie jest jeszcze wspierany. "
                "Dozwolone układy: OCR | YB | YS | YB->YS | OCR->YB | OCR->YB->YS | YB->OCR."
            ),
        }

    method_key, rescue_enabled = compiled
    method_card_meta = method_card_meta or {}
    label = method_card_meta.get(method_key, {}).get("title", method_key)
    status_text = f"Builder złoży ten pipeline jako tryb: {label}"
    if method_key == "BOTH":
        status_text += " z rescue" if rescue_enabled else " bez rescue"
    return {
        "valid": True,
        "blocks": prepared,
        "method_key": method_key,
        "rescue_enabled": bool(rescue_enabled),
        "requires_yolo": method_key in {"YOLO", "BOTH", "YOLO_OCR", "YOLO_BOX", "YOLO_SYMBOL"},
        "status_text": status_text,
    }


def normalize_detection_pipeline_blocks(blocks=None) -> list[str]:
    allowed = {"ocr_symbol", "yolo_box", "yolo_symbol"}
    prepared = []
    if isinstance(blocks, str):
        raw_blocks = blocks.replace(">", ",").replace("|", ",").split(",")
    else:
        raw_blocks = list(blocks or [])
    for block in raw_blocks:
        normalized = str(block or "").strip().lower().replace("-", "_")
        if normalized in allowed:
            prepared.append(normalized)
    return prepared


def get_saved_detection_pipeline_blocks(host) -> list[str]:
    try:
        raw = (getattr(host, "local_session", {}) or {}).get("char_detection_pipeline_blocks", [])
    except Exception:
        raw = []
    blocks = normalize_detection_pipeline_blocks(raw)
    compiled = compile_detection_pipeline_blocks(blocks)
    if bool(compiled.get("valid")):
        if _pipeline_requires_yolo(compiled) and not _host_has_configured_yolo_detection_model(host):
            return []
        return blocks
    return []


def save_detection_pipeline_blocks(host, blocks) -> None:
    prepared = normalize_detection_pipeline_blocks(blocks)
    compiled = compile_detection_pipeline_blocks(prepared)
    if not bool(compiled.get("valid")):
        return
    if _pipeline_requires_yolo(compiled) and not _host_has_configured_yolo_detection_model(host):
        prepared = ["ocr_symbol"]
        compiled = compile_detection_pipeline_blocks(prepared)
    method_key = str(compiled.get("method_key") or "").strip().upper()
    try:
        host._detection_pipeline_last_blocks = list(prepared)
    except Exception:
        pass
    try:
        host._save_local_setting("char_detection_pipeline_blocks", prepared)
    except Exception:
        pass
    if method_key:
        try:
            host._save_local_setting("char_det_method", method_key)
        except Exception:
            pass


def get_detection_workflow_text(host, method_key: str | None, method_labels: dict, key_by_label: dict) -> str:
    resolved_method = normalize_detection_method_key(
        method_key or host._get_detection_method_key(),
        method_labels,
        key_by_label,
    )
    if _method_requires_yolo(resolved_method) and not _host_has_configured_yolo_detection_model(host):
        resolved_method = "OCR"
    if resolved_method == "YOLO":
        return "Pipeline: model detekcji YOLO wykrywa ramki i klasy znaków."
    if resolved_method == "YOLO_BOX":
        return "Pipeline: YB wykrywa tylko ramki znakow; tekst pozostaje pusty."
    if resolved_method == "YOLO_SYMBOL":
        return "Pipeline: YS wpisuje znaki w istniejace ramki; nie tworzy boxow."
    if resolved_method == "BOTH":
        rescue_chars = host._get_hybrid_rescue_max_chars()
        rescue_text = "rescue AUTO" if rescue_chars > 0 else "rescue wyłączone"
        return f"Pipeline: OCR czyta, model detekcji YOLO dopasowuje ramki; {rescue_text}."
    if resolved_method == "YOLO_OCR":
        return "Pipeline: model detekcji YOLO daje boxy, OCR czyta cropy."
    return "Pipeline: OCR czyta i segmentuje znaki."


def apply_detection_pipeline_blocks(host, blocks, *, save: bool = True, prompt_for_yolo_model: bool = True) -> bool:
    compiled = host._compile_detection_pipeline_blocks(blocks)
    if not bool(compiled.get("valid")):
        return False

    method_key = str(compiled.get("method_key") or "OCR")
    requires_yolo = bool(compiled.get("requires_yolo"))
    if requires_yolo and not host._has_configured_yolo_detection_model():
        if prompt_for_yolo_model:
            selected_model = str(host._pick_yolo_model() or "").strip()
            if not selected_model or not host._has_configured_yolo_detection_model():
                messagebox.showinfo(
                    "Brak modelu YOLO",
                    "Ten pipeline wymaga modelu YOLO znaków. Wskaż poprawny plik .pt, aby go zatwierdzić.",
                )
                return False
        else:
            return False

    if method_key == "BOTH":
        rescue_enabled = bool(compiled.get("rescue_enabled"))
        try:
            host.hybrid_rescue_max_chars_var.set(1 if rescue_enabled else 0)
        except Exception:
            pass
        try:
            host.hybrid_yolo_box_backend_var.set(True)
        except Exception:
            pass

    host._set_detection_method_key(method_key, save=save)
    if save:
        try:
            host._save_detection_pipeline_blocks(blocks)
        except Exception:
            pass
    host._refresh_detection_workflow_info_label()
    return True


def refresh_detection_workflow_info_label(host) -> None:
    label = getattr(host, "detect_workflow_info_lbl", None)
    if label is None:
        return
    try:
        label.configure(text=host._get_detection_workflow_text())
    except Exception:
        pass
    try:
        host._refresh_detection_active_model_label()
    except Exception:
        pass


def get_detection_pipeline_block_meta(block_key: str, block_library: dict) -> dict:
    normalized = str(block_key or "").strip().lower().replace("-", "_")
    fallback = block_library.get("ocr_symbol", {})
    return dict(block_library.get(normalized, fallback))


def get_detection_pipeline_block_style(host, block_key: str) -> dict:
    normalized = str(block_key or "").strip().lower().replace("-", "_")
    palette = getattr(host.app, "palette", {})
    component_style = host._get_preview_badge_component_style(normalized)
    panel = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    badge_fill = str(component_style.get("badge_fill", component_style.get("outline", "#3c3c3c")))
    badge_outline = str(component_style.get("badge_outline", badge_fill))
    fill = blend_hex_colors(badge_fill, panel_alt, 0.22)
    fill_selected = blend_hex_colors(badge_fill, panel_alt, 0.42)
    text_color = host._get_readable_text_color(fill, preferred=palette.get("fg", "#f3f3f3"))
    muted_color = host._get_readable_text_color(fill, preferred=palette.get("muted", "#c7c7c7"))
    selected_text_color = host._get_readable_text_color(fill_selected, preferred=text_color)
    selected_muted_color = host._get_readable_text_color(fill_selected, preferred=muted_color)
    selected_outline = palette.get("accent", palette.get("success", badge_outline))
    return {
        "badge_fill": badge_fill,
        "badge_outline": badge_outline,
        "badge_fg": host._get_readable_text_color(
            badge_fill,
            preferred=str(component_style.get("badge_fg", "#ffffff")),
        ),
        "fill": fill,
        "fill_selected": fill_selected,
        "outline": badge_outline,
        "text": text_color,
        "muted": muted_color,
        "text_selected": selected_text_color,
        "muted_selected": selected_muted_color,
        "selected_outline": selected_outline,
        "selected_glow": blend_hex_colors(selected_outline, panel, 0.42),
        "shadow": blend_hex_colors(badge_fill, panel, 0.55),
    }


def get_detection_pipeline_builder_blocks(host) -> list[str]:
    state = getattr(host, "_detection_pipeline_state", {}) or {}
    return list(state.get("blocks", []) or [])


def set_detection_pipeline_builder_blocks(host, blocks, *, selected_index: int | None = None) -> None:
    prepared = normalize_detection_pipeline_blocks(blocks)
    if selected_index is None:
        selected_index = 0 if prepared else None
    elif prepared:
        selected_index = max(0, min(len(prepared) - 1, int(selected_index)))
    else:
        selected_index = None
    host._detection_pipeline_state = {
        "blocks": prepared,
        "selected_index": selected_index,
    }


def select_detection_pipeline_builder_block(host, index: int | None) -> None:
    blocks = host._get_detection_pipeline_builder_blocks()
    if not blocks:
        host._set_detection_pipeline_builder_blocks([], selected_index=None)
    else:
        safe_index = 0 if index is None else max(0, min(len(blocks) - 1, int(index)))
        host._set_detection_pipeline_builder_blocks(blocks, selected_index=safe_index)
    host._refresh_detection_pipeline_builder()


def set_detection_pipeline_builder_preset(host, method_key: str) -> None:
    resolved_method = host._normalize_detection_method_key(method_key)
    if resolved_method == "YOLO":
        blocks = ["yolo_box", "yolo_symbol"]
    elif resolved_method == "YOLO_BOX":
        blocks = ["yolo_box"]
    elif resolved_method == "YOLO_SYMBOL":
        blocks = ["yolo_symbol"]
    elif resolved_method == "BOTH":
        blocks = ["ocr_symbol", "yolo_box"]
        if host._get_hybrid_rescue_max_chars() > 0:
            blocks.append("yolo_symbol")
    elif resolved_method == "YOLO_OCR":
        blocks = ["yolo_box", "ocr_symbol"]
    else:
        blocks = ["ocr_symbol"]
    host._set_detection_pipeline_builder_blocks(blocks, selected_index=0 if blocks else None)
    host._refresh_detection_pipeline_builder()


def append_detection_pipeline_builder_block(host, block_key: str) -> None:
    blocks = host._get_detection_pipeline_builder_blocks()
    blocks.append(str(block_key or "").strip().lower().replace("-", "_"))
    host._set_detection_pipeline_builder_blocks(blocks, selected_index=len(blocks) - 1)
    host._refresh_detection_pipeline_builder()


def move_detection_pipeline_builder_selected_block(host, direction: int) -> None:
    blocks = host._get_detection_pipeline_builder_blocks()
    state = getattr(host, "_detection_pipeline_state", {}) or {}
    selected_index = state.get("selected_index")
    if selected_index is None or not blocks:
        return
    try:
        current = int(selected_index)
        target = int(current) + int(direction)
    except Exception:
        return
    if target < 0 or target >= len(blocks):
        return
    blocks[current], blocks[target] = blocks[target], blocks[current]
    host._set_detection_pipeline_builder_blocks(blocks, selected_index=target)
    host._refresh_detection_pipeline_builder()


def remove_detection_pipeline_builder_selected_block(host) -> None:
    blocks = host._get_detection_pipeline_builder_blocks()
    state = getattr(host, "_detection_pipeline_state", {}) or {}
    selected_index = state.get("selected_index")
    if selected_index is None or not blocks:
        return
    try:
        current = int(selected_index)
    except Exception:
        return
    if current < 0 or current >= len(blocks):
        return
    blocks.pop(current)
    next_index = min(current, len(blocks) - 1) if blocks else None
    host._set_detection_pipeline_builder_blocks(blocks, selected_index=next_index)
    host._refresh_detection_pipeline_builder()


def clear_detection_pipeline_builder(host) -> None:
    host._set_detection_pipeline_builder_blocks([], selected_index=None)
    host._refresh_detection_pipeline_builder()


def close_detection_pipeline_builder(host) -> None:
    advanced_modal = getattr(host, "_detection_pipeline_advanced_modal", None)
    try:
        if advanced_modal is not None and advanced_modal.winfo_exists():
            try:
                advanced_modal.grab_release()
            except Exception:
                pass
            advanced_modal.destroy()
    except Exception:
        pass
    modal = getattr(host, "_detection_pipeline_modal", None)
    try:
        if modal is not None and modal.winfo_exists():
            try:
                modal.grab_release()
            except Exception:
                pass
            modal.destroy()
    except Exception:
        pass
    host._detection_pipeline_modal = None
    host._detection_pipeline_canvas = None
    host._detection_pipeline_property_body = None
    host._detection_pipeline_runtime = {}
    host._detection_pipeline_confirm_btn = None
    host._detection_pipeline_status_var = None
    host._detection_pipeline_hint_var = None
    host._detection_pipeline_model_frame = None
    host._detection_pipeline_model_status_lbl = None
    host._detection_pipeline_model_btn = None
    host._detection_pipeline_model_details_btn = None
    host._detection_pipeline_advanced_open = {}
    host._detection_pipeline_advanced_modal = None
    host._detection_pipeline_advanced_modal_block = None


def show_detection_pipeline_model_details(host, model_path: str | None = None, *, title: str | None = None) -> None:
    resolved_path = str(model_path or host._get_effective_yolo_model_path() or "").strip()
    if not resolved_path:
        messagebox.showinfo(
            "Parametry modelu detekcji",
            "Najpierw wskaż model detekcji znaków .pt. Parametry zostaną odczytane z pliku JSON obok modelu.",
            parent=getattr(host, "_detection_pipeline_modal", None) or getattr(host, "frame", None),
        )
        return
    show_yolo_model_metadata_dialog(host, resolved_path, title=title)


def pick_detection_pipeline_yolo_model(host) -> str:
    selected = str(host._pick_yolo_model() or "").strip()
    try:
        host._refresh_detection_pipeline_builder()
    except Exception:
        pass
    if selected:
        try:
            host._show_detection_pipeline_model_details(selected, title="Parametry wybranego modelu detekcji")
        except Exception:
            pass
    return selected


def open_detection_pipeline_advanced_modal(host, block_key: str) -> None:
    self = host
    normalized = str(block_key or "").strip().lower().replace("-", "_")
    if normalized not in {"ocr_symbol", "yolo_box", "yolo_symbol"}:
        return

    existing = getattr(self, "_detection_pipeline_advanced_modal", None)
    existing_block = str(getattr(self, "_detection_pipeline_advanced_modal_block", "") or "")
    try:
        if existing is not None and existing.winfo_exists():
            if existing_block == normalized:
                existing.deiconify()
                existing.lift()
                try:
                    existing.grab_set()
                except Exception:
                    pass
                existing.focus_force()
                return
            try:
                existing.grab_release()
            except Exception:
                pass
            existing.destroy()
    except Exception:
        pass

    meta = self._get_detection_pipeline_block_meta(normalized)
    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#c7c7c7")
    accent = palette.get("accent", palette.get("success", "#5ac878"))
    success = palette.get("success", accent)
    card_bg = blend_hex_colors(panel_alt, panel_bg, 0.22)
    card_inner_bg = blend_hex_colors(card_bg, panel_bg, 0.12)
    card_border = blend_hex_colors(border, accent, 0.18)
    header_bg = blend_hex_colors(panel_alt, panel_bg, 0.16)
    header_badge_bg = blend_hex_colors(header_bg, accent, 0.18)
    muted_soft = blend_hex_colors(muted, card_bg, 0.22)

    parent = getattr(self, "_detection_pipeline_modal", None) or getattr(self, "frame", None)
    win = tk.Toplevel(parent)
    title = f"Zaawansowane: {meta.get('badge', '?')} {meta.get('title', normalized)}"
    win.title(title)
    try:
        win.geometry("760x640")
        win.minsize(640, 520)
        win.resizable(True, True)
    except Exception:
        pass

    windowing_system = ""
    try:
        windowing_system = str(win.tk.call("tk", "windowingsystem") or "")
    except Exception:
        windowing_system = ""
    if windowing_system != "win32":
        try:
            win.transient(parent)
        except Exception:
            pass

    def _close():
        try:
            self._force_save_all()
        except Exception:
            pass
        try:
            win.grab_release()
        except Exception:
            pass
        try:
            win.destroy()
        except Exception:
            pass
        if getattr(self, "_detection_pipeline_advanced_modal", None) is win:
            self._detection_pipeline_advanced_modal = None
            self._detection_pipeline_advanced_modal_block = None
        try:
            self._refresh_detection_pipeline_builder_property_panel()
        except Exception:
            pass

    def _release_grab_if_iconic(_event=None):
        try:
            if str(win.state() or "") == "iconic":
                win.grab_release()
        except Exception:
            pass

    def _restore_grab_if_visible(_event=None):
        try:
            if str(win.state() or "") != "iconic":
                win.grab_set()
        except Exception:
            pass

    try:
        win.grab_set()
    except Exception:
        pass
    try:
        win.bind("<Unmap>", _release_grab_if_iconic, add="+")
        win.bind("<Map>", _restore_grab_if_visible, add="+")
    except Exception:
        pass
    win.configure(bg=panel_bg)
    win.protocol("WM_DELETE_WINDOW", _close)

    self._detection_pipeline_advanced_modal = win
    self._detection_pipeline_advanced_modal_block = normalized

    root = tk.Frame(win, bg=panel_bg, padx=16, pady=16, bd=0, highlightthickness=0)
    root.pack(fill=tk.BOTH, expand=True)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    header = tk.Frame(
        root,
        bg=header_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=card_border,
        highlightcolor=card_border,
        padx=14,
        pady=12,
    )
    header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
    header.columnconfigure(1, weight=1)
    tk.Label(
        header,
        text=str(meta.get("badge", "?")),
        bg=header_badge_bg,
        fg=success,
        font=("Segoe UI", 18, "bold"),
        width=4,
        anchor="center",
        bd=0,
        highlightthickness=0,
    ).grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 10))
    tk.Label(
        header,
        text=str(meta.get("title", normalized)),
        bg=header_bg,
        fg=fg,
        font=("Segoe UI", 12, "bold"),
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    ).grid(row=0, column=1, sticky="ew")
    tk.Label(
        header,
        text=str(meta.get("desc", "")),
        bg=header_bg,
        fg=muted_soft,
        anchor="w",
        justify=tk.LEFT,
        wraplength=620,
        bd=0,
        highlightthickness=0,
    ).grid(row=1, column=1, sticky="ew", pady=(3, 0))

    canvas_shell = tk.Frame(root, bg=panel_bg, bd=0, highlightthickness=0)
    canvas_shell.grid(row=1, column=0, sticky="nsew")
    canvas_shell.grid_columnconfigure(0, weight=1)
    canvas_shell.grid_rowconfigure(0, weight=1)

    canvas = tk.Canvas(canvas_shell, bg=panel_bg, bd=0, highlightthickness=0)
    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(canvas_shell, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    canvas.configure(yscrollcommand=scrollbar.set)

    body = tk.Frame(canvas, bg=panel_bg, padx=2, pady=2, bd=0, highlightthickness=0)
    body_window = canvas.create_window((0, 0), window=body, anchor="nw")

    def _sync_scrollregion(_event=None):
        try:
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(body_window, width=max(1, int(canvas.winfo_width() or 1)))
        except Exception:
            pass

    body.bind("<Configure>", _sync_scrollregion, add="+")
    canvas.bind("<Configure>", _sync_scrollregion, add="+")

    def _wheel(event):
        try:
            delta = int(getattr(event, "delta", 0) or 0)
            if delta:
                canvas.yview_scroll(-1 * int(delta / 120), "units")
                return "break"
        except Exception:
            pass
        return None

    canvas.bind("<MouseWheel>", _wheel, add="+")
    body.bind("<MouseWheel>", _wheel, add="+")

    def add_section(text, note=None):
        shell = tk.Frame(
            body,
            bg=card_bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=card_border,
            highlightcolor=card_border,
            padx=0,
            pady=0,
        )
        shell.pack(fill=tk.X, pady=(0, 12), padx=(0, 2))
        head = tk.Frame(shell, bg=card_bg, bd=0, highlightthickness=0, padx=12, pady=9)
        head.pack(fill=tk.X)
        accent_line = tk.Frame(head, bg=success, width=3, height=20, bd=0, highlightthickness=0)
        accent_line.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 9))
        title_stack = tk.Frame(head, bg=card_bg, bd=0, highlightthickness=0)
        title_stack.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            title_stack,
            text=str(text),
            bg=card_bg,
            fg=fg,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
        ).pack(anchor=tk.W, fill=tk.X)
        if note:
            tk.Label(
                title_stack,
                text=note,
                bg=card_bg,
                fg=muted_soft,
                font=("Segoe UI", 8),
                wraplength=650,
                justify=tk.LEFT,
                anchor="w",
                bd=0,
                highlightthickness=0,
            ).pack(anchor=tk.W, fill=tk.X, pady=(2, 0))
        content = tk.Frame(shell, bg=card_inner_bg, bd=0, highlightthickness=0, padx=12, pady=10)
        content.pack(fill=tk.X)
        return content

    def widget_bg(widget, fallback=card_inner_bg):
        try:
            return str(widget.cget("bg") or fallback)
        except Exception:
            return fallback

    def add_spin(parent, text, variable, from_, to_, increment, width=9, hint=None):
        try:
            if variable in (self.yolo_box_conf_var, self.yolo_symbol_conf_var):
                from_ = max(0.00001, float(from_))
        except Exception:
            pass
        row_bg = widget_bg(parent)
        row = tk.Frame(parent, bg=row_bg, bd=0, highlightthickness=0)
        row.pack(fill=tk.X, pady=(0, 9))
        row.columnconfigure(1, weight=1)
        tk.Label(
            row,
            text=text,
            bg=row_bg,
            fg=fg,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
        ).grid(row=0, column=0, sticky="w")
        spin = ttk.Spinbox(row, from_=from_, to=to_, increment=increment, textvariable=variable, width=width, format="%.5f")
        spin.grid(row=0, column=2, sticky="e")
        try:
            spin.configure(command=save_pipeline_advanced_options)
            spin.bind("<FocusOut>", lambda _event: save_pipeline_advanced_options(), add="+")
            spin.bind("<Return>", lambda _event: save_pipeline_advanced_options(), add="+")
        except Exception:
            pass
        if hint:
            tk.Label(
                parent,
                text=hint,
                bg=row_bg,
                fg=muted_soft,
                font=("Segoe UI", 8),
                wraplength=650,
                justify=tk.LEFT,
                anchor="w",
                bd=0,
                highlightthickness=0,
            ).pack(anchor=tk.W, fill=tk.X, pady=(-4, 9))
        return spin

    def add_check(parent, text, variable, *, command=None, onvalue=True, offvalue=False, hint=None):
        row_bg = widget_bg(parent)
        row = tk.Frame(parent, bg=row_bg, bd=0, highlightthickness=0)
        row.pack(fill=tk.X, pady=(0, 9))
        check = tk.Checkbutton(
            row,
            text=text,
            variable=variable,
            command=command,
            onvalue=onvalue,
            offvalue=offvalue,
            bg=row_bg,
            fg=fg,
            activebackground=row_bg,
            activeforeground=fg,
            selectcolor=card_bg,
            disabledforeground=muted_soft,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            cursor="hand2",
            bd=0,
            highlightthickness=0,
        )
        check.pack(anchor=tk.W, fill=tk.X)
        if hint:
            tk.Label(
                parent,
                text=hint,
                bg=row_bg,
                fg=muted_soft,
                font=("Segoe UI", 8),
                wraplength=650,
                justify=tk.LEFT,
                anchor="w",
                bd=0,
                highlightthickness=0,
            ).pack(anchor=tk.W, fill=tk.X, padx=(24, 0), pady=(-6, 9))
        return check

    def add_choice(parent, text, variable, values, *, width=18, hint=None):
        row_bg = widget_bg(parent)
        row = tk.Frame(parent, bg=row_bg, bd=0, highlightthickness=0)
        row.pack(fill=tk.X, pady=(0, 9))
        row.columnconfigure(1, weight=1)
        tk.Label(
            row,
            text=text,
            bg=row_bg,
            fg=fg,
            font=("Segoe UI", 9),
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
        ).grid(row=0, column=0, sticky="w")
        combo = ttk.Combobox(
            row,
            textvariable=variable,
            values=tuple(values),
            state="readonly",
            width=width,
        )
        combo.grid(row=0, column=2, sticky="e")
        try:
            combo.bind("<FocusOut>", lambda _event: self._force_save_all(), add="+")
            combo.bind("<<ComboboxSelected>>", lambda _event: self._force_save_all(), add="+")
        except Exception:
            pass
        if hint:
            tk.Label(
                parent,
                text=hint,
                bg=row_bg,
                fg=muted_soft,
                font=("Segoe UI", 8),
                wraplength=650,
                justify=tk.LEFT,
                anchor="w",
                bd=0,
                highlightthickness=0,
            ).pack(anchor=tk.W, fill=tk.X, pady=(-4, 9))
        return combo

    def save_pipeline_advanced_options():
        try:
            self.detect_protect_manual_var.set(True)
        except Exception:
            pass
        try:
            self._on_yolo_option_var_write()
        except Exception:
            pass
        try:
            self._force_save_all()
        except Exception:
            pass
        try:
            self._refresh_detection_refiner_guard_label()
        except Exception:
            pass

    def add_yolo_common_section(note: str):
        common = add_section("Wspólny przebieg YOLO", note)
        add_spin(
            common,
            "Próg YB - ramki:",
            self.yolo_box_conf_var,
            0.0,
            1.0,
            0.05,
            hint="Minimalna pewność, od której detekcja może zostać użyta jako ramka znaku. Wyżej: mniej szumu i dubli. Niżej: większa szansa odzyskania trudnych znaków.",
        )
        add_spin(
            common,
            "Próg YS - znak:",
            self.yolo_symbol_conf_var,
            0.0,
            1.0,
            0.05,
            hint="Minimalna pewność klasy znaku. Ten próg decyduje, czy YOLO może podpowiedzieć literę lub cyfrę, a nie tylko położenie ramki.",
        )
        add_spin(
            common,
            "NMS IoU:",
            self.yolo_iou_var,
            0.01,
            0.99,
            0.05,
            hint="Steruje usuwaniem podobnych detekcji z samego modelu. Niżej: agresywniej usuwa duble. Wyżej: zostawia więcej bliskich ramek przy ciasnych znakach.",
        )
        add_spin(
            common,
            "Nakładanie ramek:",
            self.yolo_overlap_var,
            0.0,
            1.0,
            0.05,
            hint="Dodatkowy filtr po YOLO. Niżej: szybciej uznaje dwie ramki za dubel. Wyżej: pozwala im mocniej na siebie nachodzić.",
        )
        add_check(
            common,
            "Class agnostic NMS",
            self.yolo_agnostic_nms_var,
            command=save_pipeline_advanced_options,
            hint="Przydatne, gdy ten sam fragment znaku dostaje kilka różnych klas. Model usuwa wtedy duble bez przywiązywania się do etykiety klasy.",
        )

    def add_yolo_sequence_section(note: str):
        sequence = add_section("Filtr sekwencji znaków", note)
        add_spin(
            sequence,
            "Tolerancja osi Y:",
            self.yolo_seq_center_y_var,
            0.10,
            1.50,
            0.05,
            hint="Jak bardzo znaki mogą odbiegać od wspólnej linii. Niżej: ostrzej pilnuje rzędu. Wyżej: lepiej znosi krzywe lub nierówne tablice.",
        )
        add_spin(
            sequence,
            "Min. zgodność wysokości:",
            self.yolo_seq_min_h_ratio_var,
            0.20,
            1.00,
            0.05,
            hint="Odrzuca podejrzanie niskie kandydaty. Wyżej: bezpieczniej przeciw śmieciom. Niżej: większa tolerancja dla uszkodzonych lub słabych znaków.",
        )
        add_spin(
            sequence,
            "Max. wysokość:",
            self.yolo_seq_max_h_ratio_var,
            1.00,
            3.50,
            0.05,
            hint="Odrzuca kandydaty zbyt wysokie względem reszty znaków. Pomaga, gdy YOLO obejmie część ramki tablicy albo tło.",
        )
        add_spin(
            sequence,
            "Max. szerokość:",
            self.yolo_seq_max_w_ratio_var,
            1.00,
            4.50,
            0.05,
            hint="Odrzuca ramki podejrzanie szerokie. To główny bezpiecznik przeciw boxom obejmującym dwa sąsiednie znaki.",
        )
        add_spin(
            sequence,
            "Miękki konflikt:",
            self.yolo_seq_soft_overlap_var,
            0.00,
            1.00,
            0.05,
            hint="Próg ostrzegawczy dla nachodzących ramek. Niżej: filtr szybciej zaczyna wybierać lepszego kandydata.",
        )
        add_spin(
            sequence,
            "Twardy konflikt:",
            self.yolo_seq_hard_overlap_var,
            0.00,
            1.00,
            0.05,
            hint="Próg krytyczny dla nachodzących ramek. Po jego przekroczeniu słabsza detekcja jest traktowana jak dubel lub błąd.",
        )

    def add_yolo_box_protection_section():
        protection = add_section(
            "Ochrona i refiner ramek",
            "Te opcje decydują, czy YB może poprawiać istniejące wyniki. Zakres przebiegu, np. tylko tablice bez perfect, wybierasz w modalu startu detekcji.",
        )
        try:
            self.detect_protect_manual_var.set(True)
        except Exception:
            pass
        manual_guard = add_check(
            protection,
            "Manualne boxy znaków: zawsze chronione",
            self.detect_protect_manual_var,
            command=save_pipeline_advanced_options,
            hint="Ręczne korekty nie są nadpisywane przez OCR ani YOLO. To stała zasada pracy, a nie opcja eksperymentalna.",
        )
        try:
            manual_guard.configure(state=tk.DISABLED)
        except Exception:
            pass
        add_check(
            protection,
            "Zachowaj status i odczyt tablic perfect",
            self.detect_protect_perfect_var,
            command=save_pipeline_advanced_options,
            hint="Perfect chroni gotowy odczyt tablicy. Po wyłączeniu tej ochrony automat może przebudować niemanualne wyniki.",
        )
        add_check(
            protection,
            "Refiner geometrii boxów perfect",
            self.detect_refine_perfect_yolo_var,
            command=save_pipeline_advanced_options,
            hint="YOLO może poprawić położenie ramek na tablicach perfect, ale bez zmiany odczytu znaków i bez ruszania manuali.",
        )
        add_check(
            protection,
            "Pilnuj ciągłości znaku w refinerze",
            self.detect_refiner_continuity_guard_var,
            command=save_pipeline_advanced_options,
            hint="Dodatkowy bezpiecznik: poprawiana ramka nie powinna przejmować sąsiedniego znaku ani rozjechać się na dwa znaki.",
        )

    def add_yolo_box_backend_section():
        hybrid = add_section(
            "YB w pipeline hybrydowym",
            "Dotyczy układu OCR -> YB: OCR czyta znak, a YB może podmienić samą geometrię ramki.",
        )
        add_check(
            hybrid,
            "Końcowe ramki bierz z YB",
            self.hybrid_yolo_box_backend_var,
            command=save_pipeline_advanced_options,
            hint="Włącz, gdy OCR dobrze czyta znaki, ale ramki po OCR są gorzej dopasowane. Manualne boxy nadal pozostają chronione.",
        )

    def add_yolo_symbol_rescue_section():
        rescue = add_section(
            "YS: rescue brakujących znaków",
            "Rescue jest dodatkową próbą YOLO wtedy, gdy odczyt OCR/YB nie domyka tablicy.",
        )
        add_check(
            rescue,
            "Włącz rescue niskopewnych znaków",
            self.hybrid_rescue_max_chars_var,
            command=save_pipeline_advanced_options,
            onvalue=1,
            offvalue=0,
            hint="System może wykonać ostrożny przebieg pomocniczy dla brakujących znaków. Nie służy do masowego nadpisywania gotowych manuali.",
        )

    if normalized == "yolo_box":
        add_yolo_common_section(
            "YB odpowiada za geometrię ramek, ale korzysta z tego samego przebiegu modelu co YS. Dlatego poniżej widać pełny dawny zestaw parametrów YOLO."
        )
        add_yolo_box_protection_section()
        add_yolo_box_backend_section()
        add_yolo_sequence_section(
            "Filtr porządkuje kandydatów YOLO w logiczny ciąg znaków. Przydaje się zwłaszcza, gdy model próbuje zostawić duble albo ramki obejmujące dwa znaki."
        )

    elif normalized == "yolo_symbol":
        add_yolo_common_section(
            "YS odpowiada za klasę znaku, ale progi i NMS działają na wspólnym przebiegu YOLO. Dzięki temu nie trzeba szukać ustawień w innym miejscu."
        )
        add_yolo_symbol_rescue_section()
        add_yolo_sequence_section(
            "Te parametry pilnują, żeby znaki YOLO tworzyły sensowną sekwencję na tablicy zamiast przypadkowego zbioru ramek."
        )

    else:
        base = add_section(
            "OCR: przygotowanie odczytu",
            "Zaawansowane parametry OCR dotyczą przygotowania obrazu znaku przed odczytem.",
        )
        add_spin(base, "Min. wysokość boxa OCR:", self.ocr_min_height_ratio_var, 0.20, 1.00, 0.05)
        add_spin(
            base,
            "Ręczna korekta kąta [°]:",
            self.prep_angle_var,
            -30,
            30,
            1,
            hint="Stała korekta obrotu przed OCR. Zwykle zostaw 0, używaj tylko dla serii z powtarzalnym przekoszeniem.",
        )
        add_spin(base, "Wysokość OCR [px]:", self.prep_height_var, 40, 150, 1)
        add_spin(base, "Padding [%]:", self.prep_padding_var, 0, 50, 1)
        add_choice(
            base,
            "Interpolacja:",
            self.interpolation_var,
            ("nearest", "linear", "cubic", "area", "lanczos4"),
            hint="Sposób skalowania znaku przed odczytem. Lanczos4 jest najostrzejszy, area bywa lepsza przy mocnym zmniejszaniu.",
        )
        add_check(
            base,
            "Pełna binaryzacja OCR",
            self.prep_use_bin_var,
            command=save_pipeline_advanced_options,
        )
        add_spin(base, "Odcięcie odblasków:", self.prep_clip_var, 100, 255, 1)
        add_spin(base, "Usuwanie ziarna:", self.prep_denoise_var, 0, 50, 1)
        add_check(
            base,
            "Wzmacniaj kontrast CLAHE",
            self.do_clahe_var,
            command=save_pipeline_advanced_options,
            hint="Włącza lokalne wzmocnienie kontrastu. Siła CLAHE poniżej działa tylko wtedy, gdy ta opcja jest aktywna.",
        )
        add_spin(base, "Siła CLAHE:", self.prep_clahe_var, 0.0, 10.0, 0.5)
        add_spin(base, "Blok binaryzacji:", self.prep_block_var, 3, 51, 2)
        add_spin(base, "Stała C:", self.prep_c_var, -20, 20, 1)
        add_spin(base, "Erozja:", self.prep_erode_var, 0, 5, 1)

    footer = ttk.Frame(root)
    footer.grid(row=2, column=0, sticky="ew", pady=(12, 0))
    footer.columnconfigure(0, weight=1)
    ttk.Label(
        footer,
        text="Zmiany są zapisywane od razu i użyte przy następnym uruchomieniu detekcji.",
        style="PanelMuted.TLabel",
        wraplength=520,
        justify=tk.LEFT,
    ).grid(row=0, column=0, sticky="w")
    ttk.Button(footer, text="Zamknij", command=_close).grid(row=0, column=1, sticky="e")

    try:
        self._on_yolo_option_var_write()
    except Exception:
        pass
    try:
        win.focus_force()
    except Exception:
        pass


def commit_detection_pipeline_builder(host) -> None:
    blocks = host._get_detection_pipeline_builder_blocks()
    if not host._apply_detection_pipeline_blocks(blocks, save=True, prompt_for_yolo_model=True):
        host._refresh_detection_pipeline_builder()
        return
    try:
        host._refresh_detection_workflow_info_label()
    except Exception:
        pass
    try:
        host._refresh_detection_active_model_label()
    except Exception:
        pass
    try:
        host._refresh_yolo_model_picker_state()
    except Exception:
        pass
    host._close_detection_pipeline_builder()


def refresh_detection_pipeline_builder(host) -> None:
    modal = getattr(host, "_detection_pipeline_modal", None)
    if modal is None:
        return
    try:
        if not modal.winfo_exists():
            host._close_detection_pipeline_builder()
            return
    except Exception:
        host._close_detection_pipeline_builder()
        return

    blocks = host._get_detection_pipeline_builder_blocks()
    compiled = host._compile_detection_pipeline_blocks(blocks)
    if (
        bool(compiled.get("valid"))
        and bool(compiled.get("requires_yolo"))
        and not host._has_configured_yolo_detection_model()
    ):
        compiled["status_text"] = (
            str(compiled.get("status_text") or "")
            + " | Brak modelu YOLO: zatwierdzenie poprosi o wskazanie pliku .pt."
        )
    status_var = getattr(host, "_detection_pipeline_status_var", None)
    if status_var is not None:
        try:
            status_var.set(str(compiled.get("status_text") or ""))
        except Exception:
            pass

    hint_var = getattr(host, "_detection_pipeline_hint_var", None)
    if hint_var is not None:
        if blocks:
            badges = [host._get_detection_pipeline_block_meta(block).get("badge", "?") for block in blocks]
            hint_text = "Aktualny łańcuch: " + " -> ".join(badges)
        else:
            hint_text = "Dodaj klocki OCR, YB i YS, aby złożyć metodę detekcji."
        try:
            hint_var.set(hint_text)
        except Exception:
            pass

    host._refresh_detection_pipeline_model_row(compiled)

    confirm_btn = getattr(host, "_detection_pipeline_confirm_btn", None)
    if confirm_btn is not None:
        host._set_widget_state(confirm_btn, "normal" if bool(compiled.get("valid")) else "disabled")

    host._draw_detection_pipeline_builder_canvas()
    host._refresh_detection_pipeline_builder_property_panel()


def refresh_detection_pipeline_model_row(host, compiled: dict | None = None) -> None:
    frame = getattr(host, "_detection_pipeline_model_frame", None)
    status_lbl = getattr(host, "_detection_pipeline_model_status_lbl", None)
    browse_btn = getattr(host, "_detection_pipeline_model_btn", None)
    details_btn = getattr(host, "_detection_pipeline_model_details_btn", None)
    if frame is None:
        return
    # Model selection now belongs to the selected YOLO block inspector.
    try:
        frame.grid_remove()
    except Exception:
        pass
    return

    compiled = compiled if isinstance(compiled, dict) else {}
    requires_yolo = bool(compiled.get("requires_yolo"))
    try:
        if requires_yolo:
            frame.grid()
        elif str(frame.winfo_manager()):
            frame.grid_remove()
    except Exception:
        pass

    if not requires_yolo:
        return

    project_mode = _is_step3_campaign_runtime(host)
    model_path = str(host._get_effective_yolo_model_path() or "").strip()
    yolo_ready = bool(model_path and Path(model_path).exists())

    if yolo_ready:
        model_name = Path(model_path).name
        version, size = host._infer_yolo_arch_from_model_path(model_path)
        if version and size:
            model_name = f"{model_name} (YOLOv{version}{size})"
        text = (
            f"Model detekcji znaków z projektu: {model_name}"
            if project_mode
            else f"Model detekcji znaków dla tego pipeline: {model_name}"
        )
        tone = "neutral"
    elif project_mode:
        text = "Projekt nie ma przypiętego modelu detekcji znaków. Wskaż wytrenowany .pt dla PZ2."
        tone = "warning"
    else:
        text = "Ten pipeline używa YOLO do detekcji. Wybierz wytrenowany model znaków .pt."
        tone = "warning"

    if status_lbl is not None:
        if not host._set_inline_status_label_state(status_lbl, text=text, tone=tone, emphasis=False):
            host._set_themed_label_state(status_lbl, text=text, tone=tone, emphasis=False)

    if browse_btn is not None:
        try:
            browse_btn.configure(text=("Zmień model detekcji" if yolo_ready else "Wybierz model detekcji"))
        except Exception:
            pass
        try:
            browse_btn.grid()
        except Exception:
            pass
        host._set_widget_state(browse_btn, "normal")

    if details_btn is not None:
        try:
            details_btn.grid()
        except Exception:
            pass
        host._set_widget_state(details_btn, "normal" if yolo_ready else "disabled")


def refresh_detection_pipeline_builder_property_panel(host):
    self = host
    body = getattr(self, "_detection_pipeline_property_body", None)
    if body is None:
        return
    try:
        for child in list(body.winfo_children()):
            child.destroy()
    except Exception:
        return

    def bind_property_mousewheel(widget):
        handler = getattr(self, "_detection_pipeline_property_mousewheel_handler", None)
        if handler is None or widget is None:
            return
        try:
            widget.bind("<MouseWheel>", handler, add="+")
        except Exception:
            pass
        try:
            for child in widget.winfo_children():
                bind_property_mousewheel(child)
        except Exception:
            pass

    def finish_property_panel_refresh():
        bind_property_mousewheel(body)
        canvas = getattr(self, "_detection_pipeline_property_canvas", None)
        if canvas is not None:
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass

    blocks = self._get_detection_pipeline_builder_blocks()
    state = getattr(self, "_detection_pipeline_state", {}) or {}
    selected_index = state.get("selected_index")
    if selected_index is None or not blocks:
        ttk.Label(
            body,
            text="Wybierz klocek na canvasie, aby edytować jego właściwości.",
            style="Muted.TLabel",
            wraplength=280,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X)
        finish_property_panel_refresh()
        return

    try:
        block_key = blocks[int(selected_index)]
    except Exception:
        ttk.Label(
            body,
            text="Nie udało się odczytać zaznaczonego klocka.",
            style="Muted.TLabel",
            wraplength=280,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X)
        finish_property_panel_refresh()
        return

    meta = self._get_detection_pipeline_block_meta(block_key)
    compiled = self._compile_detection_pipeline_blocks(blocks)

    ttk.Label(
        body,
        text=f"{meta.get('badge', '?')}  {meta.get('title', block_key)}",
        style="PanelHeading.TLabel",
    ).pack(anchor=tk.W, fill=tk.X)
    ttk.Label(
        body,
        text=str(meta.get("desc", "")),
        style="PanelMuted.TLabel",
        wraplength=300,
        justify=tk.LEFT,
    ).pack(anchor=tk.W, fill=tk.X, pady=(4, 10))

    def add_spin(parent, text, variable, from_, to_, increment, width=8, hint=None, command=None):
        try:
            if variable in (self.yolo_box_conf_var, self.yolo_symbol_conf_var):
                from_ = max(0.00001, float(from_))
        except Exception:
            pass
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row, text=text).pack(side=tk.LEFT)
        spin = ttk.Spinbox(row, from_=from_, to=to_, increment=increment, textvariable=variable, width=width, format="%.5f")
        spin.pack(side=tk.RIGHT)
        save_command = command or self._on_yolo_option_var_write
        try:
            spin.configure(command=save_command)
            spin.bind("<FocusOut>", lambda _event: save_command(), add="+")
            spin.bind("<Return>", lambda _event: save_command(), add="+")
        except Exception:
            pass
        if hint:
            add_hint(parent, hint)
        return spin

    def add_check(parent, text, variable, *, command=None, onvalue=True, offvalue=False, hint=None):
        check = ttk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            command=command,
            onvalue=onvalue,
            offvalue=offvalue,
        )
        check.pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
        if hint:
            add_hint(parent, hint)
        return check

    def add_hint(parent, text):
        ttk.Label(
            parent,
            text=text,
            style="PanelMuted.TLabel",
            wraplength=300,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X, pady=(0, 8))

    def add_section(parent, text, note=None):
        ttk.Label(parent, text=text, style="PanelHeading.TLabel").pack(anchor=tk.W, fill=tk.X, pady=(8, 6))
        if note:
            add_hint(parent, note)

    def save_property_option_vars():
        try:
            self.detect_protect_manual_var.set(True)
        except Exception:
            pass
        try:
            self._on_yolo_option_var_write()
        except Exception:
            pass
        for key, attr_name, default in (
            ("char_detect_protect_manual", "detect_protect_manual_var", True),
            ("char_detect_protect_perfect", "detect_protect_perfect_var", True),
            ("char_detect_refine_perfect_yolo", "detect_refine_perfect_yolo_var", False),
            ("char_detect_refiner_continuity_guard", "detect_refiner_continuity_guard_var", True),
        ):
            try:
                value = bool(getattr(self, attr_name).get())
            except Exception:
                value = bool(default)
            try:
                self._save_local_setting(key, value)
            except Exception:
                pass
        try:
            self._force_save_all()
        except Exception:
            pass

    def add_yolo_model_choice(parent):
        model_path = str(self._get_effective_yolo_model_path() or "").strip()
        model_ready = bool(model_path and Path(model_path).exists())
        model_name = Path(model_path).name if model_ready else "Nie wybrano modelu"
        if model_ready:
            try:
                version, size = self._infer_yolo_arch_from_model_path(model_path)
            except Exception:
                version, size = ("", "")
            if version and size:
                model_name = f"{model_name} (YOLOv{version}{size})"

        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row, text="Model detekcji:").pack(anchor=tk.W)
        ttk.Label(row, text=model_name, style="PanelMuted.TLabel", wraplength=300, justify=tk.LEFT).pack(
            anchor=tk.W,
            fill=tk.X,
            pady=(2, 6),
        )
        btn_row = ttk.Frame(row)
        btn_row.pack(fill=tk.X)
        ttk.Button(
            btn_row,
            text=("Zmień model" if model_ready else "Wybierz model"),
            command=self._pick_detection_pipeline_yolo_model,
        ).pack(side=tk.LEFT)
        ttk.Button(
            btn_row,
            text="Szczegóły",
            command=self._show_detection_pipeline_model_details,
        ).pack(side=tk.LEFT, padx=(8, 0))

    def add_advanced_section(parent, key: str, builder):
        label = "Filtry OCR"
        if key == "yolo_box":
            label = "Zaawansowane YB - ramki"
        elif key == "yolo_symbol":
            label = "Zaawansowane YS - znaki"

        add_section(parent, label)
        builder(parent)
        return

    def build_yolo_shared_advanced(parent):
        add_spin(
            parent,
            "NMS IoU:",
            self.yolo_iou_var,
            0.01,
            0.99,
            0.05,
            hint="Niżej: szybciej usuwa podobne duble. Wyżej: zostawia więcej bliskich ramek przy ciasnych znakach.",
        )
        add_spin(
            parent,
            "Nakładanie boxów:",
            self.yolo_overlap_var,
            0.0,
            1.0,
            0.05,
            hint="Dodatkowy próg usuwania ramek nachodzących na siebie. Niżej: ostrzej przeciw dublom.",
        )
        add_check(
            parent,
            "Class agnostic NMS",
            self.yolo_agnostic_nms_var,
            command=save_property_option_vars,
            hint="Pomaga, gdy ten sam fragment znaku dostaje kilka klas naraz, np. B i 8.",
        )

    def build_yolo_sequence_advanced(parent):
        add_section(parent, "Filtr sekwencji")
        add_spin(
            parent,
            "Tolerancja osi Y:",
            self.yolo_seq_center_y_var,
            0.10,
            1.50,
            0.05,
            hint="Jak bardzo znaki mogą odbiegać od wspólnej linii. Wyżej: większa tolerancja dla krzywych tablic.",
        )
        add_spin(
            parent,
            "Min. zgodność wysokości:",
            self.yolo_seq_min_h_ratio_var,
            0.20,
            1.00,
            0.05,
            hint="Odcina podejrzanie niskie ramki. Wyżej: bezpieczniej, ale łatwiej zgubić trudny znak.",
        )
        add_spin(
            parent,
            "Max. wysokość:",
            self.yolo_seq_max_h_ratio_var,
            1.00,
            3.50,
            0.05,
            hint="Odcina kandydaty zbyt wysokie względem reszty znaków.",
        )
        add_spin(
            parent,
            "Max. szerokość:",
            self.yolo_seq_max_w_ratio_var,
            1.00,
            4.50,
            0.05,
            hint="Najważniejszy bezpiecznik przeciw ramkom obejmującym dwa znaki.",
        )
        add_spin(
            parent,
            "Miękki konflikt:",
            self.yolo_seq_soft_overlap_var,
            0.00,
            1.00,
            0.05,
            hint="Próg ostrzegawczy dla nachodzących ramek.",
        )
        add_spin(
            parent,
            "Twardy konflikt:",
            self.yolo_seq_hard_overlap_var,
            0.00,
            1.00,
            0.05,
            hint="Próg krytyczny. Po jego przekroczeniu słabsza ramka jest traktowana jak dubel lub błąd.",
        )

    def build_yolo_box_advanced(parent):
        build_yolo_shared_advanced(parent)
        add_section(parent, "Ochrona i refiner")
        try:
            self.detect_protect_manual_var.set(True)
        except Exception:
            pass
        manual_guard = add_check(
            parent,
            "Manualne boxy: zawsze chronione",
            self.detect_protect_manual_var,
            command=save_property_option_vars,
            hint="Ręczna korekta nie jest nadpisywana przez automat.",
        )
        try:
            manual_guard.configure(state=tk.DISABLED)
        except Exception:
            pass
        add_check(
            parent,
            "Chroń tablice perfect",
            self.detect_protect_perfect_var,
            command=save_property_option_vars,
            hint="Gotowe tablice nie są przebudowywane, o ile świadomie nie wyłączysz tej ochrony w starcie detekcji.",
        )
        add_check(
            parent,
            "Refiner ramek perfect",
            self.detect_refine_perfect_yolo_var,
            command=save_property_option_vars,
            hint="Pozwala poprawić geometrię ramek bez zmiany odczytu znaków i bez ruszania manuali.",
        )
        add_check(
            parent,
            "Pilnuj ciągłości znaku",
            self.detect_refiner_continuity_guard_var,
            command=save_property_option_vars,
            hint="Chroni przed ramką, która zaczyna obejmować dwa sąsiednie znaki.",
        )
        add_check(
            parent,
            "Końcowe ramki bierz z YB",
            self.hybrid_yolo_box_backend_var,
            command=save_property_option_vars,
            hint="W hybrydzie OCR czyta znak, a YB może podmienić geometrię ramki.",
        )
        build_yolo_sequence_advanced(parent)

    def build_yolo_symbol_advanced(parent):
        build_yolo_shared_advanced(parent)
        add_section(parent, "Rescue YS")
        add_check(
            parent,
            "Rescue brakujących znaków",
            self.hybrid_rescue_max_chars_var,
            command=save_property_option_vars,
            onvalue=1,
            offvalue=0,
            hint="Dodatkowa, ostrożna próba YOLO dla znaków, których brakuje albo które nie domykają odczytu tablicy.",
        )
        build_yolo_sequence_advanced(parent)

    def save_ocr_option_vars():
        try:
            self._save_local_setting("char_ocr_conf", max(0.05, min(0.95, float(self.ocr_conf_var.get()))))
        except Exception:
            pass
        try:
            self._save_local_setting(
                "char_ocr_min_height_ratio",
                max(0.20, min(1.00, float(self.ocr_min_height_ratio_var.get()))),
            )
        except Exception:
            pass
        try:
            self._force_save_all()
        except Exception:
            pass

    def build_ocr_advanced(parent):
        add_spin(
            parent,
            "Min. wysokość znaku:",
            self.ocr_min_height_ratio_var,
            0.20,
            1.00,
            0.05,
            command=save_ocr_option_vars,
            hint="Filtr odrzuca zbyt niskie odczyty OCR, które zwykle są artefaktami albo fragmentami tła.",
        )

    if block_key == "ocr_symbol":
        add_section(body, "Odczyt OCR")
        add_spin(
            body,
            "Próg pewności OCR:",
            self.ocr_conf_var,
            0.05,
            0.95,
            0.05,
            command=save_ocr_option_vars,
            hint="Minimalna pewność odczytu, od której OCR może podać znak do pipeline.",
        )
        add_advanced_section(body, "ocr_symbol", build_ocr_advanced)
        add_section(body, "Narzędzia OCR")
        ttk.Button(body, text="Laboratorium OCR", command=self._open_filter_lab).pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
        ttk.Button(body, text="Ranking presetów OCR", command=self._open_ocr_ranking_modal).pack(anchor=tk.W, fill=tk.X, pady=(0, 8))
    elif block_key == "yolo_box":
        add_section(body, "Podstawowe")
        add_yolo_model_choice(body)
        add_spin(
            body,
            "Próg YB - ramki:",
            self.yolo_box_conf_var,
            0.0,
            1.0,
            0.05,
            hint="YB decyduje, które detekcje YOLO mogą stać się ramkami znaków.",
        )
        add_hint(body, "YB odpowiada tylko za ramki znaków. Jeśli YOLO ma też podpowiadać znaki, dodaj osobny klocek YS.")
        add_advanced_section(body, "yolo_box", build_yolo_box_advanced)
    elif block_key == "yolo_symbol":
        add_section(body, "Podstawowe")
        add_yolo_model_choice(body)
        add_spin(
            body,
            "Próg YS - znak:",
            self.yolo_symbol_conf_var,
            0.0,
            1.0,
            0.05,
            hint="YS decyduje, kiedy klasa znaku z YOLO jest wystarczająco wiarygodna.",
        )
        if bool(compiled.get("valid")) and compiled.get("method_key") == "YOLO_SYMBOL":
            add_hint(body, "YS dziala na istniejacych ramkach: wpisuje symbole, ale nie tworzy nowych boxow.")
        elif bool(compiled.get("valid")) and compiled.get("method_key") == "BOTH":
            if self._get_hybrid_rescue_max_chars() <= 0:
                try:
                    self.hybrid_rescue_max_chars_var.set(1)
                except Exception:
                    pass
            add_hint(body, "W tej konfiguracji YS działa jako rescue AUTO: pomaga tylko tam, gdzie odczyt nie pasuje do oczekiwanego napisu.")
        else:
            add_hint(body, "YS odpowiada za klasę znaku z modelu YOLO. Ramki nadal pochodzą z klocka YB.")
        add_advanced_section(body, "yolo_symbol", build_yolo_symbol_advanced)

    finish_property_panel_refresh()


def draw_detection_pipeline_builder_canvas(host):
    self = host
    canvas = getattr(self, "_detection_pipeline_canvas", None)
    if canvas is None:
        return
    try:
        canvas.update_idletasks()
    except Exception:
        pass

    width = max(360, int(canvas.winfo_width() or 0))
    height = max(220, int(canvas.winfo_height() or 0))
    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))
    muted = palette.get("muted", "#c7c7c7")
    fg = palette.get("fg", "#f3f3f3")
    canvas.configure(bg=panel_bg)
    canvas.delete("all")
    self._detection_pipeline_runtime = {}

    blocks = self._get_detection_pipeline_builder_blocks()
    state = getattr(self, "_detection_pipeline_state", {}) or {}
    selected_index = state.get("selected_index")

    canvas.create_rectangle(1, 1, width - 2, height - 2, outline=border, width=1)
    center_y = float(height // 2)
    canvas.create_line(28, center_y, width - 28, center_y, fill=blend_hex_colors(border, panel_bg, 0.35), dash=(3, 4))

    if not blocks:
        canvas.create_text(
            width / 2.0,
            center_y - 16.0,
            text="Pusty pipeline",
            fill=fg,
            font=("Segoe UI", 14, "bold"),
            anchor="center",
        )
        canvas.create_text(
            width / 2.0,
            center_y + 14.0,
            text="Dodaj klocki OCR, YB i YS lub kliknij preset u góry.",
            fill=muted,
            font=("Segoe UI", 10),
            anchor="center",
        )
        return

    block_w = 168.0
    block_h = 86.0
    gap = 34.0
    total_w = (len(blocks) * block_w) + (max(0, len(blocks) - 1) * gap)
    start_x = max(18.0, (float(width) - total_w) / 2.0)
    top_y = center_y - (block_h / 2.0)

    for idx, block_key in enumerate(blocks):
        meta = self._get_detection_pipeline_block_meta(block_key)
        style = self._get_detection_pipeline_block_style(block_key)
        x1 = float(start_x + idx * (block_w + gap))
        x2 = x1 + block_w
        y1 = top_y
        y2 = y1 + block_h
        is_selected = idx == selected_index
        block_fill = str(style.get("fill_selected" if is_selected else "fill", panel_bg))
        title_fill = str(style.get("text_selected" if is_selected else "text", fg))
        subtitle_fill = str(style.get("muted_selected" if is_selected else "muted", muted))
        outline_fill = str(style.get("selected_outline" if is_selected else "outline", border))

        if idx < len(blocks) - 1:
            arrow_y = center_y
            arrow_x1 = x2 + 8.0
            arrow_x2 = arrow_x1 + gap - 16.0
            canvas.create_line(
                arrow_x1,
                arrow_y,
                arrow_x2,
                arrow_y,
                fill=str(style.get("outline", border)),
                width=2,
                arrow=tk.LAST,
                arrowshape=(10, 12, 4),
            )

        if is_selected:
            canvas.create_rectangle(
                x1 - 5,
                y1 - 5,
                x2 + 5,
                y2 + 5,
                outline=str(style.get("selected_glow", style.get("shadow", outline_fill))),
                width=4,
            )

        rect_id = canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            fill=block_fill,
            outline=outline_fill,
            width=3 if is_selected else 1,
            tags=(f"det_pipeline_block::{idx}", "det_pipeline_block"),
        )
        badge_text = str(meta.get("badge", "?"))
        self._draw_preview_text_badge(
            canvas,
            x1 + 10,
            y1 + 10,
            badge_text,
            fill_color=str(style.get("badge_fill", style.get("outline", border))),
            outline_color=str(style.get("badge_outline", style.get("outline", border))),
            text_color=str(style.get("badge_fg", "#ffffff")),
            font=("Segoe UI", 9, "bold"),
            anchor=tk.NW,
            pad_x=6,
            pad_y=2,
            tags=(f"det_pipeline_block::{idx}", "det_pipeline_block"),
        )
        canvas.create_text(
            x1 + 12,
            y1 + 38,
            text=str(meta.get("title", block_key)),
            fill=title_fill,
            font=("Segoe UI", 10, "bold"),
            anchor=tk.NW,
            tags=(f"det_pipeline_block::{idx}", "det_pipeline_block"),
        )
        canvas.create_text(
            x1 + 12,
            y1 + 60,
            text=str(meta.get("subtitle", "")),
            fill=subtitle_fill,
            font=("Segoe UI", 8),
            anchor=tk.NW,
            tags=(f"det_pipeline_block::{idx}", "det_pipeline_block"),
        )
        self._detection_pipeline_runtime[idx] = {
            "bbox": (x1, y1, x2, y2),
            "rect_id": rect_id,
            "block_key": block_key,
        }

    for idx in self._detection_pipeline_runtime.keys():
        canvas.tag_bind(
            f"det_pipeline_block::{idx}",
            "<Button-1>",
            lambda _event, picked=idx: self._select_detection_pipeline_builder_block(picked),
        )
        canvas.tag_bind(
            f"det_pipeline_block::{idx}",
            "<Enter>",
            lambda _event: canvas.configure(cursor="hand2"),
        )
        canvas.tag_bind(
            f"det_pipeline_block::{idx}",
            "<Leave>",
            lambda _event: canvas.configure(cursor=""),
        )


def open_detection_pipeline_builder(host, initial_method=None, detection_pipeline_preset_meta=None):
    self = host
    preset_meta = detection_pipeline_preset_meta or {}
    initial_mode = self._normalize_detection_method_key(initial_method or self._get_detection_method_key())
    if _method_requires_yolo(initial_mode) and not _host_has_configured_yolo_detection_model(self):
        initial_mode = "OCR"
    existing = getattr(self, "_detection_pipeline_modal", None)
    try:
        if existing is not None and existing.winfo_exists():
            if not self._get_detection_pipeline_builder_blocks():
                self._set_detection_pipeline_builder_blocks(self._get_detection_pipeline_blocks(initial_mode), selected_index=0)
            existing.deiconify()
            existing.lift()
            try:
                existing.grab_set()
            except Exception:
                pass
            existing.focus_force()
            return
    except Exception:
        self._close_detection_pipeline_builder()

    palette = getattr(self.app, "palette", {})
    panel_bg = palette.get("panel", "#252526")
    panel_alt = palette.get("panel_alt", "#2d2d30")
    border = palette.get("panel_border", palette.get("border", "#3c3c3c"))

    win = tk.Toplevel(self.frame)
    win.title("Budowniczy pipeline detekcji")
    try:
        win.geometry("1160x760")
        win.minsize(980, 680)
        win.resizable(True, True)
    except Exception:
        pass
    windowing_system = ""
    try:
        windowing_system = str(win.tk.call("tk", "windowingsystem") or "")
    except Exception:
        windowing_system = ""
    # Na Windowsie transient często odbiera natywne przyciski minimalizacji i maksymalizacji.
    if windowing_system != "win32":
        try:
            win.transient(self.frame.winfo_toplevel())
        except Exception:
            pass

    def _release_pipeline_grab_if_iconic(_event=None):
        try:
            if str(win.state() or "") == "iconic":
                win.grab_release()
        except Exception:
            pass

    def _restore_pipeline_grab_if_visible(_event=None):
        try:
            if str(win.state() or "") != "iconic":
                win.grab_set()
        except Exception:
            pass

    try:
        win.grab_set()
    except Exception:
        pass
    try:
        win.bind("<Unmap>", _release_pipeline_grab_if_iconic, add="+")
        win.bind("<Map>", _restore_pipeline_grab_if_visible, add="+")
    except Exception:
        pass
    win.configure(bg=panel_bg)
    win.protocol("WM_DELETE_WINDOW", self._close_detection_pipeline_builder)

    self._detection_pipeline_modal = win
    self._detection_pipeline_status_var = tk.StringVar(value="")
    self._detection_pipeline_hint_var = tk.StringVar(value="")
    self._detection_pipeline_advanced_open = {}
    remembered_blocks = normalize_detection_pipeline_blocks(getattr(self, "_detection_pipeline_last_blocks", []) or [])
    if not _pipeline_allowed_in_current_context(self, remembered_blocks):
        remembered_blocks = []
    initial_blocks = remembered_blocks or self._get_detection_pipeline_blocks(initial_mode)
    self._set_detection_pipeline_builder_blocks(initial_blocks, selected_index=0)

    root = ttk.Frame(win, padding=14)
    root.pack(fill=tk.BOTH, expand=True)
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=0)
    root.rowconfigure(4, weight=1)

    ttk.Label(root, text="Budowniczy pipeline detekcji", style="PanelHeading.TLabel").grid(
        row=0,
        column=0,
        columnspan=2,
        sticky="w",
    )
    intro = ttk.Label(
        root,
        text=(
            "Ułóż liniowy łańcuch klocków OCR, YB i YS. System na żywo sprawdzi, "
            "czy taki pipeline jest wspierany przez obecny backend i do jakiego trybu się mapuje. "
            "Jeżeli pipeline używa YOLO, najpierw wskaż model detekcji znaków."
        ),
        style="PanelMuted.TLabel",
        wraplength=960,
        justify=tk.LEFT,
    )
    intro.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 12))

    presets_row = ttk.Frame(root)
    presets_row.grid(row=2, column=0, sticky="ew", pady=(0, 10))
    ttk.Label(presets_row, text="Gotowe presety:").pack(side=tk.LEFT, padx=(0, 8))
    for preset_key in ("OCR", "YOLO_BOX", "YOLO_SYMBOL", "YOLO", "BOTH", "YOLO_OCR"):
        meta = preset_meta.get(preset_key, {})
        ttk.Button(
            presets_row,
            text=str(meta.get("label", preset_key)),
            command=lambda key=preset_key: self._set_detection_pipeline_builder_preset(key),
        ).pack(side=tk.LEFT, padx=(0, 6))

    model_frame = tk.Frame(
        root,
        bg=panel_alt,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        padx=10,
        pady=8,
    )
    model_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    model_frame.grid_columnconfigure(1, weight=1)
    self._detection_pipeline_model_frame = model_frame

    model_title_lbl = tk.Label(
        model_frame,
        text="Model YOLO pipeline",
        bg=panel_alt,
        fg=palette.get("fg", "#f3f3f3"),
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        justify=tk.LEFT,
        bd=0,
        highlightthickness=0,
    )
    model_title_lbl.grid(row=0, column=0, sticky="w", padx=(0, 10))

    self._detection_pipeline_model_status_lbl = tk.Label(
        model_frame,
        text="",
        bg=panel_alt,
        fg=palette.get("muted", "#c7c7c7"),
        anchor="w",
        justify=tk.LEFT,
        wraplength=780,
        bd=0,
        highlightthickness=0,
    )
    self._detection_pipeline_model_status_lbl.grid(row=0, column=1, sticky="ew")

    self._detection_pipeline_model_btn = ttk.Button(
        model_frame,
        text="Wybierz model",
        command=self._pick_detection_pipeline_yolo_model,
    )
    self._detection_pipeline_model_btn.grid(row=0, column=2, sticky="e", padx=(10, 0))

    self._detection_pipeline_model_details_btn = ttk.Button(
        model_frame,
        text="Parametry",
        command=self._show_detection_pipeline_model_details,
    )
    self._detection_pipeline_model_details_btn.grid(row=0, column=3, sticky="e", padx=(8, 0))

    main = ttk.Frame(root)
    main.grid(row=4, column=0, columnspan=2, sticky="nsew")
    main.columnconfigure(0, weight=1)
    main.columnconfigure(1, weight=0)
    main.rowconfigure(0, weight=1)

    canvas_shell = tk.Frame(main, bg=panel_bg, bd=0, highlightthickness=1, highlightbackground=border, highlightcolor=border)
    canvas_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
    canvas_shell.grid_rowconfigure(1, weight=1)
    canvas_shell.grid_columnconfigure(0, weight=1)

    builder_hint_lbl = tk.Label(
        canvas_shell,
        textvariable=self._detection_pipeline_hint_var,
        anchor="w",
        justify=tk.LEFT,
        bg=panel_bg,
        fg=palette.get("fg", "#f3f3f3"),
        bd=0,
        highlightthickness=0,
        padx=12,
        pady=10,
    )
    builder_hint_lbl.grid(row=0, column=0, sticky="ew")

    self._detection_pipeline_canvas = tk.Canvas(
        canvas_shell,
        height=320,
        bg=panel_bg,
        bd=0,
        highlightthickness=0,
    )
    self._detection_pipeline_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
    self._detection_pipeline_canvas.bind(
        "<Configure>",
        lambda _event: self._draw_detection_pipeline_builder_canvas(),
        add="+",
    )

    controls_row = ttk.Frame(canvas_shell)
    controls_row.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 12))
    ttk.Button(controls_row, text="+ OCR", command=lambda: self._append_detection_pipeline_builder_block("ocr_symbol")).pack(side=tk.LEFT)
    ttk.Button(controls_row, text="+ YB", command=lambda: self._append_detection_pipeline_builder_block("yolo_box")).pack(side=tk.LEFT, padx=(6, 0))
    ttk.Button(controls_row, text="+ YS", command=lambda: self._append_detection_pipeline_builder_block("yolo_symbol")).pack(side=tk.LEFT, padx=(6, 0))
    ttk.Button(controls_row, text="Przesuń w lewo", command=lambda: self._move_detection_pipeline_builder_selected_block(-1)).pack(side=tk.LEFT, padx=(16, 0))
    ttk.Button(controls_row, text="Przesuń w prawo", command=lambda: self._move_detection_pipeline_builder_selected_block(1)).pack(side=tk.LEFT, padx=(6, 0))
    ttk.Button(controls_row, text="Usuń blok", command=self._remove_detection_pipeline_builder_selected_block).pack(side=tk.LEFT, padx=(16, 0))
    ttk.Button(controls_row, text="Wyczyść", command=self._clear_detection_pipeline_builder).pack(side=tk.LEFT, padx=(6, 0))

    inspector = tk.Frame(main, bg=panel_alt, bd=0, highlightthickness=1, highlightbackground=border, highlightcolor=border)
    inspector.grid(row=0, column=1, sticky="ns")
    inspector.configure(width=340)
    inspector.grid_propagate(False)
    inspector_canvas = tk.Canvas(
        inspector,
        bg=panel_alt,
        bd=0,
        highlightthickness=0,
        yscrollincrement=18,
    )
    inspector_scrollbar = ttk.Scrollbar(inspector, orient=tk.VERTICAL, command=inspector_canvas.yview)
    inspector_canvas.configure(yscrollcommand=inspector_scrollbar.set)
    inspector_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    inspector_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    inspector_body = ttk.Frame(inspector_canvas, padding=(12, 12))
    inspector_body_window = inspector_canvas.create_window((0, 0), window=inspector_body, anchor="nw")

    def _sync_pipeline_inspector_scroll(_event=None):
        try:
            inspector_canvas.configure(scrollregion=inspector_canvas.bbox("all"))
            inspector_canvas.itemconfigure(inspector_body_window, width=max(1, int(inspector_canvas.winfo_width() or 1)))
        except Exception:
            pass

    def _on_pipeline_inspector_mousewheel(event):
        try:
            delta = int(getattr(event, "delta", 0) or 0)
            if delta:
                inspector_canvas.yview_scroll(int(-1 * (delta / 120)), "units")
        except Exception:
            pass
        return "break"

    inspector_body.bind("<Configure>", _sync_pipeline_inspector_scroll, add="+")
    inspector_canvas.bind("<Configure>", _sync_pipeline_inspector_scroll, add="+")
    inspector_canvas.bind("<MouseWheel>", _on_pipeline_inspector_mousewheel, add="+")
    inspector_body.bind("<MouseWheel>", _on_pipeline_inspector_mousewheel, add="+")
    self._detection_pipeline_property_canvas = inspector_canvas
    self._detection_pipeline_property_mousewheel_handler = _on_pipeline_inspector_mousewheel
    self._detection_pipeline_property_body = inspector_body

    footer = ttk.Frame(root)
    footer.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 0))
    footer.columnconfigure(0, weight=1)

    ttk.Label(
        footer,
        textvariable=self._detection_pipeline_status_var,
        style="PanelMuted.TLabel",
        wraplength=900,
        justify=tk.LEFT,
    ).grid(row=0, column=0, sticky="w")

    footer_buttons = ttk.Frame(footer)
    footer_buttons.grid(row=0, column=1, sticky="e")
    ttk.Button(footer_buttons, text="Anuluj", command=self._close_detection_pipeline_builder).pack(side=tk.RIGHT)
    self._detection_pipeline_confirm_btn = ttk.Button(
        footer_buttons,
        text="Zatwierdź pipeline",
        command=self._commit_detection_pipeline_builder,
    )
    self._detection_pipeline_confirm_btn.pack(side=tk.RIGHT, padx=(0, 8))

    self._refresh_detection_pipeline_builder()
