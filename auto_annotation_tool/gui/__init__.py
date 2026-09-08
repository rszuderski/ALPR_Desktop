#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Punkt wejścia pakietu GUI.

Ten moduł celowo nie importuje ciężkich zakładek eager na starcie.
Pozwala to ograniczyć koszt pamięci przy samym imporcie pakietu
`auto_annotation_tool.gui`.
"""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "AutoAnnotationApp",
    "AnnotationTab",
    "CharacterAnnotationTab",
    "TrainingTab",
    "RectificationTab",
    "HelpTab",
    "GuidedActionCard",
]


_LAZY_IMPORTS = {
    "AutoAnnotationApp": (".app", "AutoAnnotationApp"),
    "AnnotationTab": (".tab_annotation", "AnnotationTab"),
    "CharacterAnnotationTab": (".tab_character_annotation", "CharacterAnnotationTab"),
    "TrainingTab": (".tab_training", "TrainingTab"),
    "RectificationTab": (".tab_rectification", "RectificationTab"),
    "HelpTab": (".tab_help", "HelpTab"),
    "GuidedActionCard": (".guided_action_card", "GuidedActionCard"),
}


def __getattr__(name: str):
    target = _LAZY_IMPORTS.get(str(name or ""))
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
