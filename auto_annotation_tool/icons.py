#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Menedżer ikon dla GUI.
Wersja sterylna - nie używa plików PNG ani zasobów z dysku.
"""

class IconManager:
    """Prosty manager do ew. tekstowych modyfikatorów, bez tworzenia folderów."""
    
    @classmethod
    def get_image(cls, icon_name: str, size=(24, 24)):
        """Dummy funkcja - zapobiega błędom w starych kodach."""
        return ""

    @classmethod
    def get(cls, name):
        """Zwraca puste stringi lub absolutnie minimalne znaczniki."""
        mapping = {
            "car": "[ AUTO ]",
            "cut": "[ WYTNIJ ]", 
            "training": "[ TRENING ]", 
            "check": "[ WALIDACJA ]", 
            "trophy": "[ RANKING ]", 
            "info": "", 
            "stop": "", 
            "play": "", 
            "save": "[ EKSPORT ]", 
            "eye": "[ PODGLĄD ]"
        }
        return mapping.get(name, "")

    @classmethod
    def test_emoji_support(cls, root):
        pass