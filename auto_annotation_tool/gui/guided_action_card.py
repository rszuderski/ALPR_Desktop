#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lekka, wielorazowa karta akcji do paneli workflow.

Wzorzec:
- tytuł sekcji,
- krótki opis / wskazówka,
- jedno główne CTA w zwartym kontenerze.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .web_slim_scrollbar import blend_hex_colors


_UNSET = object()


class GuidedActionCard(tk.Frame):
    def __init__(
        self,
        master,
        app,
        *,
        title: str = "",
        description: str = "",
        description_tone: str = "muted",
        button_text: str = "",
        button_command=None,
        button_style: str = "WorkflowCard.TButton",
        button_state: str = tk.NORMAL,
        button_width: int = 20,
        button_padding: tuple[int, int] = (10, 3),
        body_wraplength: int = 320,
        container_background: str | None = None,
        emphasized: bool = False,
        title_font=("Segoe UI", 9),
        **kwargs,
    ):
        self.app = app
        self._container_background = container_background
        self._description_tone = str(description_tone or "muted")
        self._button_style = str(button_style or "WorkflowCard.TButton")
        self._button_width = int(button_width or 20)
        self._button_padding = button_padding
        self._body_wraplength = int(body_wraplength or 320)
        self._emphasized = bool(emphasized)

        outer_bg = self._resolve_outer_background()
        kwargs.setdefault("bg", outer_bg)
        kwargs.setdefault("bd", 0)
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(master, **kwargs)

        self.inner = tk.Frame(
            self,
            bd=0,
            highlightthickness=0,
            padx=12,
            pady=10,
        )
        self.inner.pack(fill=tk.BOTH, expand=True)

        self.title_label = tk.Label(
            self.inner,
            text=str(title or ""),
            bd=0,
            highlightthickness=0,
            anchor="w",
            justify=tk.LEFT,
            font=title_font,
        )
        self.title_label.pack(anchor=tk.W)

        self.body_label = tk.Label(
            self.inner,
            text=str(description or ""),
            bd=0,
            highlightthickness=0,
            anchor="w",
            justify=tk.LEFT,
            wraplength=self._body_wraplength,
        )
        self.body_label.pack(anchor=tk.W, pady=(6, 8))

        self.button_frame = tk.Frame(
            self.inner,
            bd=0,
            highlightthickness=0,
        )
        self.button_frame.pack(fill=tk.X)

        self.button_pulse_frame = tk.Frame(
            self.button_frame,
            bd=0,
            highlightthickness=0,
        )
        self.button_pulse_frame.pack(fill=tk.X)

        self.button = ttk.Button(
            self.button_pulse_frame,
            text=str(button_text or ""),
            command=button_command,
            style=self._button_style,
            state=button_state,
        )
        self.button.pack(fill=tk.X)
        self.button.configure(width=self._button_width, padding=self._button_padding)

        self.refresh_theme(container_background=container_background)
        self.set_title(title)
        self.set_description(description, tone=description_tone)
        self.configure_action(
            text=button_text,
            command=button_command,
            state=button_state,
            style=button_style,
            width=button_width,
            padding=button_padding,
        )
        self.set_emphasis(emphasized)

    def _resolve_outer_background(self) -> str:
        palette = getattr(self.app, "palette", {})
        return str(
            self._container_background
            or palette.get("surface_info", palette.get("panel", "#252526"))
        )

    @staticmethod
    def _tone_to_color_key(tone: str) -> str:
        normalized = str(tone or "muted").strip().lower()
        mapping = {
            "success": "success",
            "warning": "warning",
            "danger": "error",
            "error": "error",
            "info": "accent",
            "accent": "accent",
            "neutral": "fg",
            "muted": "muted",
        }
        return mapping.get(normalized, "muted")

    def refresh_theme(self, *, container_background: str | None = None) -> None:
        if container_background is not None:
            self._container_background = container_background

        palette = getattr(self.app, "palette", {})
        outer_bg = self._resolve_outer_background()
        border = blend_hex_colors(
            palette.get("panel_border", palette.get("border", "#3c3c3c")),
            outer_bg,
            0.35,
        )
        fill = blend_hex_colors(
            palette.get("surface_info", palette.get("panel", "#252526")),
            outer_bg,
            0.45,
        )
        if self._emphasized:
            border = blend_hex_colors(
                palette.get("accent", "#4ec9b0"),
                border,
                0.22,
            )

        try:
            self.configure(
                bg=border,
                highlightthickness=1,
                highlightbackground=border,
                highlightcolor=border,
            )
        except Exception:
            pass

        for widget in (self.inner, self.button_frame, self.button_pulse_frame):
            try:
                widget.configure(bg=fill)
            except Exception:
                pass

        desc_key = self._tone_to_color_key(self._description_tone)
        desc_color = palette.get(desc_key, palette.get("muted", "#c7c7c7"))
        title_color = palette.get("fg", "#f3f3f3")

        try:
            self.title_label.configure(bg=fill, fg=title_color)
        except Exception:
            pass
        try:
            self.body_label.configure(bg=fill, fg=desc_color)
        except Exception:
            pass

    def set_title(self, text: str) -> None:
        try:
            self.title_label.configure(text=str(text or ""))
        except Exception:
            pass

    def set_description(self, text: str, *, tone: str | None = None) -> None:
        if tone is not None:
            self._description_tone = str(tone or "muted")
        try:
            self.body_label.configure(text=str(text or ""))
        except Exception:
            pass
        self.refresh_theme()

    def configure_action(
        self,
        *,
        text: str | None | object = _UNSET,
        command=_UNSET,
        state: str | None | object = _UNSET,
        style: str | None = None,
        width: int | None = None,
        padding: tuple[int, int] | None = None,
    ) -> None:
        if style is not None:
            self._button_style = str(style or "WorkflowCard.TButton")
        if width is not None:
            self._button_width = int(width or 20)
        if padding is not None:
            self._button_padding = padding

        cfg = {
            "style": self._button_style,
            "width": self._button_width,
            "padding": self._button_padding,
        }
        if text is not _UNSET:
            cfg["text"] = str(text or "")
        if command is not _UNSET:
            cfg["command"] = command
        if state is not _UNSET:
            cfg["state"] = state
        try:
            self.button.configure(**cfg)
        except Exception:
            pass

    def set_emphasis(self, enabled: bool) -> None:
        self._emphasized = bool(enabled)
        self.refresh_theme()
        try:
            if hasattr(self.app, "set_button_emphasis"):
                self.app.set_button_emphasis(self.button, self._emphasized, base_style=self._button_style)
        except Exception:
            pass
