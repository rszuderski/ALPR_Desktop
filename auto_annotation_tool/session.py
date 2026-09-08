#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zarządzanie sesją - zapamiętywanie ostatnio używanych ścieżek i ustawień.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("AutoAnnotationTool")


class SessionManager:
    """Zarządza zapisywaniem i wczytywaniem ustawień sesji (ostatnie ścieżki, parametry itp.)."""
    
    def __init__(self, session_file: Optional[Path] = None):
        """
        Inicjalizuj menedżer sesji.
        
        Args:
            session_file: Ścieżka do pliku sesji. Jeśli None, używa domyślnego.
        """
        if session_file is None:
            # Domyślnie: ~/.auto_annotation_tool/session.json
            app_data_dir = Path.home() / ".auto_annotation_tool"
            app_data_dir.mkdir(parents=True, exist_ok=True)
            session_file = app_data_dir / "session.json"
        
        self.session_file = session_file
        self.data: Dict[str, Any] = self._load_session()
    
    def _load_session(self) -> Dict[str, Any]:
        """Wczytaj dane sesji z pliku."""
        if not self.session_file.exists():
            logger.info(f"Plik sesji nie istnieje: {self.session_file}")
            return self._get_default_session()
        
        try:
            with open(self.session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Wczytano sesję z: {self.session_file}")
            return data
        except Exception as e:
            logger.error(f"Błąd wczytywania sesji: {e}")
            return self._get_default_session()
    
    def _get_default_session(self) -> Dict[str, Any]:
        """Zwróć domyślne ustawienia sesji."""
        default_output = str(Path.home() / "Auto-Annotation-Tool-Output")
        
        return {
            "rectification": {
                "images_dir": "",
                "xml_path": "",
                "output_dir": str(Path(default_output) / "rectified_plates"),
            },
            "annotation": {
                "images_dir": "",
                "input_dir": "",
                "output_dir": str(Path(default_output) / "annotations"),
                "last_preview_run_dir": "",
                "last_preview_index": -1,
                "last_preview_filename": "",
            },
            "training": {
                "dataset_dir": "",
                "output_dir": str(Path(default_output) / "training"),
            },
            "ui": {
                "theme": "dark_visual_cs",
                "active_main_tab": "annotation",
                "global_yolo_device": "auto",
            },
        }
    
    def save_session(self):
        """Zapisz dane sesji do pliku."""
        try:
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            logger.info(f"Sesja zapisana do: {self.session_file}")
        except Exception as e:
            logger.error(f"Błąd zapisywania sesji: {e}")
    
    def get(self, tab: str, key: str, default: str = "") -> str:
        """
        Pobierz wartość z sesji.
        
        Args:
            tab: Nazwa zakładki (np. "rectification")
            key: Klucz wartości (np. "images_dir")
            default: Wartość domyślna jeśli klucz nie istnieje
        
        Returns:
            Wartość lub domyślna
        """
        return self.data.get(tab, {}).get(key, default)
    
    def set(self, tab: str, key: str, value: str):
        """
        Ustaw wartość w sesji.
        
        Args:
            tab: Nazwa zakładki
            key: Klucz wartości
            value: Nowa wartość
        """
        if tab not in self.data:
            self.data[tab] = {}
        self.data[tab][key] = value
    
    def get_rectification(self) -> Dict[str, str]:
        """Pobierz wszystkie ustawienia dla zakładki Prostowania."""
        defaults = self._get_default_session()
        return self.data.get("rectification", defaults["rectification"])
    
    def set_rectification(self, images_dir: str = "", xml_path: str = "", output_dir: str = ""):
        """Ustaw ustawienia dla zakładki Prostowania."""
        if "rectification" not in self.data:
            self.data["rectification"] = {}
        
        if images_dir:
            self.data["rectification"]["images_dir"] = images_dir
        if xml_path:
            self.data["rectification"]["xml_path"] = xml_path
        if output_dir:
            self.data["rectification"]["output_dir"] = output_dir
    
    def __repr__(self):
        return f"SessionManager(file={self.session_file})"
