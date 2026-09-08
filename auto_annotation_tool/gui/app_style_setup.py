#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TTK/Tk style setup for the main application shell."""

import tkinter as tk
import tkinter.font as tkfont

from .app_theme_definitions import DEFAULT_THEME_KEY, get_theme_palette
from .web_slim_scrollbar import blend_hex_colors


def setup_style(app, theme_key: str = None):
    self = app
    try:
        theme_key = theme_key or self.current_theme_key
        if theme_key not in self.themes:
            theme_key = DEFAULT_THEME_KEY

        self.current_theme_key = theme_key
        self.current_theme_name = self.themes[theme_key]["label"]
        self.palette = get_theme_palette(theme_key, self.themes)
        if hasattr(self, "theme_var"):
            self.theme_var.set(theme_key)
        palette = self.palette
        dark_theme = str(theme_key or "").strip().lower().startswith("dark")
        dropdown_select_bg = palette["selection_bg"]
        dropdown_select_fg = palette["selection_fg"]
        list_select_bg, list_select_fg = self.get_list_selection_colors()
        scrollbar_track = palette["scrollbar_track"]
        _scroll_track, scrollbar_thumb, scrollbar_thumb_hover = self._get_scrollbar_colors(
            track_color=scrollbar_track,
        )

        self.root.configure(bg=palette["bg"])

        named_fonts = {
            "TkDefaultFont": ("Segoe UI", 10, "normal"),
            "TkTextFont": ("Segoe UI", 10, "normal"),
            "TkMenuFont": ("Segoe UI", 10, "normal"),
            "TkHeadingFont": ("Segoe UI", 10, "bold"),
            "TkCaptionFont": ("Segoe UI", 10, "bold"),
            "TkSmallCaptionFont": ("Segoe UI", 9, "normal"),
            "TkIconFont": ("Segoe UI", 10, "normal"),
            "TkTooltipFont": ("Segoe UI", 10, "normal"),
            "TkFixedFont": ("Consolas", 10, "normal"),
        }
        for font_name, (family, size, weight) in named_fonts.items():
            try:
                font_obj = tkfont.nametofont(font_name)
                font_obj.configure(family=family, size=size, weight=weight, slant="roman")
            except Exception:
                pass

        self.root.option_add("*Font", "{Segoe UI} 10")
        self.root.option_add("*Menu.Font", "{Segoe UI} 10")
        self.root.option_add("*Label.background", palette["bg"])
        self.root.option_add("*Label.foreground", palette["fg"])
        self.root.option_add("*Frame.background", palette["bg"])
        self.root.option_add("*Canvas.background", palette["panel"])
        self.root.option_add("*Entry.background", palette["field"])
        self.root.option_add("*Entry.foreground", palette["fg"])
        self.root.option_add("*Entry.insertBackground", palette["fg"])
        self.root.option_add("*Entry.selectBackground", list_select_bg)
        self.root.option_add("*Entry.selectForeground", list_select_fg)
        self.root.option_add("*Listbox.background", palette["field"])
        self.root.option_add("*Listbox.foreground", palette["fg"])
        self.root.option_add("*Listbox.selectBackground", list_select_bg)
        self.root.option_add("*Listbox.selectForeground", list_select_fg)
        self.root.option_add("*TCombobox*Listbox.background", palette["field"])
        self.root.option_add("*TCombobox*Listbox.foreground", palette["fg"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", dropdown_select_bg)
        self.root.option_add("*TCombobox*Listbox.selectForeground", dropdown_select_fg)
        self.root.option_add("*Text.background", palette["field"])
        self.root.option_add("*Text.foreground", palette["fg"])
        self.root.option_add("*Text.insertBackground", palette["fg"])
        self.root.option_add("*Text.selectBackground", list_select_bg)
        self.root.option_add("*Text.selectForeground", list_select_fg)
        self.root.option_add("*Button.background", palette["panel_alt"])
        self.root.option_add("*Button.foreground", palette["fg"])
        self.root.option_add("*Button.activeBackground", palette["button_hover"])
        self.root.option_add("*Button.activeForeground", palette["fg"])
        self.root.option_add("*Button.highlightBackground", palette["panel_border"])
        self.root.option_add("*Button.highlightColor", palette["accent"])
        self.root.option_add("*Scrollbar.background", scrollbar_thumb)
        self.root.option_add("*Scrollbar.activeBackground", scrollbar_thumb_hover)
        self.root.option_add("*Scrollbar.troughColor", scrollbar_track)
        self.root.option_add("*Menu.background", palette["panel"])
        self.root.option_add("*Menu.foreground", palette["fg"])
        self.root.option_add("*Menu.activeBackground", palette["accent"])
        self.root.option_add("*Menu.activeForeground", palette["accent_text"])

        self.style.theme_use('clam')

        def safe_configure(style_name, **kwargs):
            try:
                self.style.configure(style_name, **kwargs)
            except Exception:
                pass

        def safe_map(style_name, **kwargs):
            try:
                self.style.map(style_name, **kwargs)
            except Exception:
                pass

        control_arrow = palette["success"]
        control_arrow_disabled = palette["muted_dim"]

        safe_configure(
            '.',
            background=palette["bg"],
            foreground=palette["fg"],
            fieldbackground=palette["field"],
            font=('Segoe UI', 10)
        )
        safe_configure('TFrame', background=palette["bg"])
        safe_configure('Panel.TFrame', background=palette["panel"])
        safe_configure(
            'Card.TFrame',
            background=palette["panel"],
            bordercolor=palette["panel_border"],
            lightcolor=palette["panel_border"],
            darkcolor=palette["panel_border"],
            borderwidth=1,
            relief=tk.SOLID
        )
        safe_configure('TPanedwindow', background=palette["panel"])
        safe_configure('TLabel', background=palette["bg"], foreground=palette["fg"], padding=2)
        safe_configure(
            'Panel.TLabel',
            background=palette["panel"],
            foreground=palette["fg"],
            padding=2
        )
        safe_configure(
            'TCheckbutton',
            background=palette["bg"],
            foreground=palette["fg"],
            focuscolor=palette["bg"],
            indicatorcolor=palette["field"],
        )
        safe_map(
            'TCheckbutton',
            background=[
                ('active', palette["bg"]),
                ('disabled', palette["bg"]),
            ],
            foreground=[('disabled', palette["muted_dim"])],
            indicatorcolor=[
                ('selected', palette["success"]),
                ('active', palette["field"]),
                ('!selected', palette["field"]),
                ('disabled', palette["panel_alt"]),
            ]
        )
        safe_configure(
            'Panel.TCheckbutton',
            background=palette["panel"],
            foreground=palette["fg"],
            focuscolor=palette["panel"],
            indicatorcolor=palette["field"],
        )
        safe_map(
            'Panel.TCheckbutton',
            background=[
                ('active', palette["panel"]),
                ('disabled', palette["panel"]),
            ],
            foreground=[('disabled', palette["muted_dim"])],
            indicatorcolor=[
                ('selected', palette["success"]),
                ('active', palette["field"]),
                ('!selected', palette["field"]),
                ('disabled', palette["panel_alt"]),
            ]
        )
        safe_configure(
            'TRadiobutton',
            background=palette["bg"],
            foreground=palette["fg"],
            focuscolor=palette["bg"],
            indicatorcolor=palette["field"],
        )
        safe_map(
            'TRadiobutton',
            background=[
                ('active', palette["bg"]),
                ('disabled', palette["bg"]),
            ],
            foreground=[('disabled', palette["muted_dim"])],
            indicatorcolor=[
                ('selected', palette["success"]),
                ('active', palette["field"]),
                ('!selected', palette["field"]),
                ('disabled', palette["panel_alt"]),
            ]
        )
        safe_configure(
            'Panel.TRadiobutton',
            background=palette["panel"],
            foreground=palette["fg"],
            focuscolor=palette["panel"],
            indicatorcolor=palette["field"],
        )
        safe_map(
            'Panel.TRadiobutton',
            background=[
                ('active', palette["panel"]),
                ('disabled', palette["panel"]),
            ],
            foreground=[('disabled', palette["muted_dim"])],
            indicatorcolor=[
                ('selected', palette["success"]),
                ('active', palette["field"]),
                ('!selected', palette["field"]),
                ('disabled', palette["panel_alt"]),
            ]
        )
        safe_configure(
            'Info.TLabel',
            background=palette["bg"],
            foreground=palette["info"],
            padding=2
        )
        safe_configure(
            'Muted.TLabel',
            background=palette["bg"],
            foreground=palette["muted"],
            padding=2
        )
        safe_configure(
            'PanelInfo.TLabel',
            background=palette["panel"],
            foreground=palette["info"],
            padding=2
        )
        safe_configure(
            'PanelSuccess.TLabel',
            background=palette["panel"],
            foreground=palette["success"],
            padding=2
        )
        safe_configure(
            'PanelError.TLabel',
            background=palette["panel"],
            foreground=palette["error"],
            padding=2
        )
        safe_configure(
            'PanelMuted.TLabel',
            background=palette["panel"],
            foreground=palette["muted"],
            padding=2
        )
        safe_configure(
            'PanelStatusNeutral.TLabel',
            background=palette["panel"],
            foreground=palette["muted"],
            padding=2,
            font=('Segoe UI', 10, 'bold')
        )
        safe_configure(
            'PanelStatusInfo.TLabel',
            background=palette["panel"],
            foreground=palette["info"],
            padding=2,
            font=('Segoe UI', 10, 'bold')
        )
        safe_configure(
            'PanelStatusSuccess.TLabel',
            background=palette["panel"],
            foreground=palette["success"],
            padding=2,
            font=('Segoe UI', 10, 'bold')
        )
        safe_configure(
            'PanelStatusWarning.TLabel',
            background=palette["panel"],
            foreground=palette["warning"],
            padding=2,
            font=('Segoe UI', 10, 'bold')
        )
        safe_configure(
            'PanelStatusError.TLabel',
            background=palette["panel"],
            foreground=palette["error"],
            padding=2,
            font=('Segoe UI', 10, 'bold')
        )
        safe_configure(
            'TLabelframe',
            background=palette["panel"],
            bordercolor=palette["panel_border"],
            lightcolor=palette["panel_border"],
            darkcolor=palette["panel_border"],
            borderwidth=1,
            relief=tk.SOLID
        )
        safe_configure(
            'TLabelframe.Label',
            background=palette["panel"],
            foreground=palette["fg"],
            font=('Segoe UI', 10, 'bold')
        )
        safe_configure(
            'AccentPanel.Horizontal.TSeparator',
            background=palette["surface_info"],
            troughcolor=palette["panel"],
            bordercolor=palette["surface_info"],
            lightcolor=palette["surface_info"],
            darkcolor=palette["surface_info"],
        )
        safe_configure(
            'TNotebook',
            background=palette["panel"],
            borderwidth=1,
            bordercolor=palette["panel_border"],
            lightcolor=palette["panel_border"],
            darkcolor=palette["panel_border"],
            tabmargins=[0, 0, 0, 0]
        )
        safe_configure(
            'TNotebook.Tab',
            background=palette["panel_alt"],
            foreground=palette["muted"],
            borderwidth=1,
            bordercolor=palette["panel_border"],
            lightcolor=palette["panel_border"],
            darkcolor=palette["panel_border"],
            relief=tk.SOLID,
            padding=[12, 5],
            font=('Segoe UI', 9, 'normal')
        )
        safe_map(
            'TNotebook.Tab',
            background=[
                ('disabled', palette["tab_disabled_bg"]),
                ('selected', palette["panel"]),
                ('active', palette["panel_alt"])
            ],
            foreground=[
                ('disabled', palette["tab_disabled_fg"]),
                ('selected', palette["fg"]),
                ('active', palette["fg"])
            ],
            bordercolor=[
                ('disabled', palette["panel_border"]),
                ('selected', palette["accent"]),
                ('active', palette["panel_border"])
            ],
            lightcolor=[
                ('disabled', palette["panel_border"]),
                ('selected', palette["accent"]),
                ('active', palette["panel_border"])
            ],
            darkcolor=[
                ('disabled', palette["panel_border"]),
                ('selected', palette["accent"]),
                ('active', palette["panel_border"])
            ],
            padding=[
                ('disabled', [12, 5]),
                ('selected', [12, 5]),
                ('active', [12, 5])
            ],
            expand=[
                ('disabled', [0, 0, 0, 0]),
                ('selected', [0, 0, 0, 0]),
                ('active', [0, 0, 0, 0])
            ]
        )
        nav_button_font = ('Segoe UI Semibold', 10)
        cta_outline = blend_hex_colors(
            palette["success"],
            palette["panel_border"],
            0.18,
        )
        cta_outline_hover = blend_hex_colors(
            palette["success"],
            palette["accent_hover"],
            0.24,
        )
        safe_configure(
            'TButton',
            background=palette["panel_alt"],
            foreground=palette["fg"],
            bordercolor=cta_outline,
            lightcolor=cta_outline,
            darkcolor=cta_outline,
            padding=6,
            borderwidth=1,
            relief=tk.SOLID,
        )
        safe_map(
            'TButton',
            background=[
                ('active', palette["button_hover"]),
                ('pressed', palette["accent_selected"]),
                ('disabled', palette["panel"])
            ],
            foreground=[('disabled', palette["muted_dim"])],
            bordercolor=[
                ('active', cta_outline_hover),
                ('pressed', cta_outline_hover),
                ('disabled', palette["border"])
            ],
            lightcolor=[
                ('active', cta_outline_hover),
                ('pressed', cta_outline_hover),
                ('disabled', palette["border"])
            ],
            darkcolor=[
                ('active', cta_outline_hover),
                ('pressed', cta_outline_hover),
                ('disabled', palette["border"])
            ]
        )
        safe_configure(
            'Accent.TButton',
            background=palette["panel_alt"],
            foreground=palette["fg"],
            bordercolor=cta_outline,
            lightcolor=cta_outline,
            darkcolor=cta_outline,
            padding=6,
            borderwidth=1,
            relief=tk.SOLID,
            font=nav_button_font,
        )
        safe_map(
            'Accent.TButton',
            background=[
                ('active', palette["surface_info"]),
                ('pressed', palette["surface_info"]),
                ('disabled', palette["panel"])
            ],
            foreground=[('disabled', palette["muted_dim"])],
            bordercolor=[
                ('active', cta_outline_hover),
                ('pressed', cta_outline_hover),
                ('disabled', palette["border"])
            ],
            lightcolor=[
                ('active', cta_outline_hover),
                ('pressed', cta_outline_hover),
                ('disabled', palette["border"])
            ],
            darkcolor=[
                ('active', cta_outline_hover),
                ('pressed', cta_outline_hover),
                ('disabled', palette["border"])
            ]
        )
        safe_configure(
            'GuidedNeutral.TButton',
            background=palette["panel_alt"],
            foreground=palette["fg"],
            bordercolor=cta_outline,
            lightcolor=cta_outline,
            darkcolor=cta_outline,
            padding=6,
            borderwidth=1,
            relief=tk.SOLID,
            font=nav_button_font,
        )
        safe_map(
            'GuidedNeutral.TButton',
            background=[
                ('active', palette["surface_info"]),
                ('pressed', palette["surface_info"]),
                ('disabled', palette["panel"])
            ],
            foreground=[('disabled', palette["muted_dim"])],
            bordercolor=[
                ('active', cta_outline_hover),
                ('pressed', cta_outline_hover),
                ('disabled', palette["border"])
            ],
            lightcolor=[
                ('active', cta_outline_hover),
                ('pressed', cta_outline_hover),
                ('disabled', palette["border"])
            ],
            darkcolor=[
                ('active', cta_outline_hover),
                ('pressed', cta_outline_hover),
                ('disabled', palette["border"])
            ]
        )
        safe_configure(
            'GuidedAccent.TButton',
            background=palette["panel_alt"],
            foreground=palette["fg"],
            bordercolor=cta_outline,
            lightcolor=cta_outline,
            darkcolor=cta_outline,
            padding=6,
            borderwidth=1,
            relief=tk.SOLID,
            font=nav_button_font,
        )
        safe_map(
            'GuidedAccent.TButton',
            background=[
                ('active', palette["surface_info"]),
                ('pressed', palette["surface_info"]),
                ('disabled', palette["panel"])
            ],
            foreground=[('disabled', palette["muted_dim"])],
            bordercolor=[
                ('active', cta_outline_hover),
                ('pressed', cta_outline_hover),
                ('disabled', palette["border"])
            ],
            lightcolor=[
                ('active', cta_outline_hover),
                ('pressed', cta_outline_hover),
                ('disabled', palette["border"])
            ],
            darkcolor=[
                ('active', cta_outline_hover),
                ('pressed', cta_outline_hover),
                ('disabled', palette["border"])
            ]
        )
        safe_configure(
            'TEntry',
            fieldbackground=palette["field"],
            foreground=palette["fg"],
            bordercolor=palette["border"],
            lightcolor=palette["border"],
            darkcolor=palette["border"]
        )
        safe_map(
            'TEntry',
            fieldbackground=[
                ('readonly', palette["field"]),
                ('disabled', palette["panel"])
            ],
            foreground=[
                ('readonly', palette["fg"]),
                ('disabled', palette["muted_dim"])
            ],
            selectbackground=[
                ('readonly', palette["field"]),
                ('disabled', palette["panel"])
            ],
            selectforeground=[
                ('readonly', palette["fg"]),
                ('disabled', palette["muted_dim"])
            ]
        )
        safe_configure(
            'TCombobox',
            fieldbackground=palette["field"],
            background=palette["panel_alt"],
            foreground=palette["fg"],
            bordercolor=palette["border"],
            lightcolor=palette["border"],
            darkcolor=palette["border"],
            selectbackground=dropdown_select_bg,
            selectforeground=dropdown_select_fg,
            arrowsize=14,
            arrowcolor=control_arrow,
        )
        safe_map(
            'TCombobox',
            fieldbackground=[('readonly', palette["field"])],
            selectbackground=[
                ('disabled', palette["panel"]),
                ('readonly', dropdown_select_bg),
                ('focus', dropdown_select_bg),
            ],
            selectforeground=[
                ('disabled', palette["muted_dim"]),
                ('readonly', dropdown_select_fg),
                ('focus', dropdown_select_fg),
            ],
            foreground=[('disabled', palette["muted_dim"])],
            arrowcolor=[
                ('readonly', control_arrow),
                ('active', control_arrow),
                ('disabled', control_arrow_disabled),
            ],
        )
        safe_configure(
            'TSpinbox',
            fieldbackground=palette["field"],
            foreground=palette["fg"],
            bordercolor=palette["border"],
            lightcolor=palette["border"],
            darkcolor=palette["border"],
            arrowsize=14,
            arrowcolor=control_arrow,
        )
        safe_map(
            'TSpinbox',
            arrowcolor=[
                ('active', control_arrow),
                ('disabled', control_arrow_disabled),
            ],
        )
        safe_configure(
            'Treeview',
            background=palette["field"],
            fieldbackground=palette["field"],
            foreground=palette["fg"],
            bordercolor=palette["border"],
            rowheight=24
        )
        safe_map(
            'Treeview',
            background=[('selected', list_select_bg)],
            foreground=[('selected', list_select_fg)]
        )
        safe_configure(
            'Treeview.Heading',
            background=palette["panel_alt"],
            foreground=palette["fg"],
            bordercolor=palette["border"],
            font=('Segoe UI', 10, 'bold')
        )
        safe_map(
            'Treeview.Heading',
            background=[
                ('active', palette["button_hover"]),
                ('pressed', palette["button_hover"]),
            ],
            foreground=[
                ('active', palette["fg"]),
                ('pressed', palette["fg"]),
                ('disabled', palette["tab_disabled_fg"]),
            ],
        )
        safe_configure(
            'Horizontal.TProgressbar',
            background=palette["progress_fill"],
            troughcolor=palette["progress_trough"],
            bordercolor=palette["border"],
            lightcolor=palette["progress_fill"],
            darkcolor=palette["progress_fill"]
        )
        self._ensure_horizontal_scale_style_assets(background=palette["panel"])
        safe_configure(
            'Vertical.TScrollbar',
            background=scrollbar_thumb,
            troughcolor=scrollbar_track,
            bordercolor=palette["border"],
            lightcolor=scrollbar_thumb,
            darkcolor=scrollbar_thumb,
            arrowcolor=control_arrow
        )
        safe_map(
            'Vertical.TScrollbar',
            background=[('active', scrollbar_thumb_hover), ('pressed', scrollbar_thumb_hover)],
            lightcolor=[('active', scrollbar_thumb_hover), ('pressed', scrollbar_thumb_hover)],
            darkcolor=[('active', scrollbar_thumb_hover), ('pressed', scrollbar_thumb_hover)],
            arrowcolor=[('active', control_arrow), ('disabled', control_arrow_disabled)]
        )
        safe_configure(
            'Horizontal.TScrollbar',
            background=scrollbar_thumb,
            troughcolor=scrollbar_track,
            bordercolor=palette["border"],
            lightcolor=scrollbar_thumb,
            darkcolor=scrollbar_thumb,
            arrowcolor=control_arrow
        )
        safe_map(
            'Horizontal.TScrollbar',
            background=[('active', scrollbar_thumb_hover), ('pressed', scrollbar_thumb_hover)],
            lightcolor=[('active', scrollbar_thumb_hover), ('pressed', scrollbar_thumb_hover)],
            darkcolor=[('active', scrollbar_thumb_hover), ('pressed', scrollbar_thumb_hover)],
            arrowcolor=[('active', control_arrow), ('disabled', control_arrow_disabled)]
        )
    except: pass
