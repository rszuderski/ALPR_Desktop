#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kontekstowy system pomocy dla całej aplikacji.
Wyświetla opis aktualnie wskazanego elementu w globalnym panelu na dole okna.
"""

import json
import tkinter as tk
from pathlib import Path

from ..config import logger


class HelpSystem:
    def __init__(self):
        self.db = {}
        self.status_updater = None
        self.overlay_presenter = None
        self.overlay_dismisser = None
        self.default_message = (
            "Gotowy. Najważniejsze komunikaty i wskazówki pojawią się tutaj podczas pracy."
        )
        self._hover_widget = None
        self._hover_help_key = ""
        self._hover_full_text = ""
        self._load_database()

    def _load_database(self):
        db_path = Path(__file__).parent.parent / "help_db.json"
        if not db_path.exists():
            return

        try:
            with open(db_path, "r", encoding="utf-8") as f:
                self.db = json.load(f)
        except Exception as e:
            logger.error(f"Nie udało się załadować bazy pomocy: {e}")

    def _push_status(self, message: str, icon: str):
        if not self.status_updater:
            return

        try:
            self.status_updater(message, icon)
        except TypeError:
            self.status_updater(message)

    def _show_overlay(self, message: str, icon: str = "help"):
        if not self.overlay_presenter:
            return

        try:
            self.overlay_presenter(message, icon)
        except TypeError:
            try:
                self.overlay_presenter(message)
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _get_widget_cursor(widget) -> str:
        try:
            return str(widget.cget("cursor") or "")
        except Exception:
            return ""

    @staticmethod
    def _should_show_help_cursor(widget) -> bool:
        try:
            return not isinstance(widget, tk.Canvas)
        except Exception:
            return True

    def _set_help_cursor(self, widget, enabled: bool):
        if widget is None:
            return

        if enabled and not self._should_show_help_cursor(widget):
            return

        if enabled:
            if not hasattr(widget, "_help_original_cursor"):
                try:
                    widget._help_original_cursor = self._get_widget_cursor(widget)
                except Exception:
                    widget._help_original_cursor = ""

            for candidate in ("question_arrow", "help", "hand2"):
                try:
                    widget.configure(cursor=candidate)
                    widget._help_context_cursor = candidate
                    return
                except Exception:
                    continue
            return

        try:
            widget.configure(cursor=str(getattr(widget, "_help_original_cursor", "") or ""))
        except Exception:
            pass

    def bind_help(self, widget, index_key: str):
        # Kontekstowy hover-help jest obecnie wyłączony, żeby nie dokładać
        # globalnych listenerów do ciężkich ekranów edycji. Zostawiamy metodę
        # jako stabilny no-op dla istniejących wywołań HELP.bind_help(...).
        return


HELP = HelpSystem()
