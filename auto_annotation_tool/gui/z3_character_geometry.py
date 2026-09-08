#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z3 character-record geometry helpers extracted from tab_character_annotation.py."""

from __future__ import annotations

from ..character_recognition import CharacterDetection


def _char_record_bbox(self, rec):
    bbox = None

    if isinstance(rec, dict):
        bbox = rec.get("bbox")
    else:
        try:
            bbox = getattr(rec, "bbox", None)
        except Exception:
            bbox = None

    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    try:
        return tuple(float(v) for v in bbox[:4])
    except Exception:
        return None

def _char_record_width(self, rec) -> float:
    bbox = self._char_record_bbox(rec)
    if not bbox:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0]))

def _char_record_height(self, rec) -> float:
    bbox = self._char_record_bbox(rec)
    if not bbox:
        return 0.0
    return max(0.0, float(bbox[3]) - float(bbox[1]))

def _char_record_center_y(self, rec) -> float:
    bbox = self._char_record_bbox(rec)
    if not bbox:
        return 0.0
    return (float(bbox[1]) + float(bbox[3])) / 2.0

def _build_char_record_geometry_stats(self, records):
    prepared = list(records or [])
    widths = [self._char_record_width(rec) for rec in prepared if self._char_record_width(rec) > 0.0]
    heights = [self._char_record_height(rec) for rec in prepared if self._char_record_height(rec) > 0.0]
    centers_y = [self._char_record_center_y(rec) for rec in prepared if self._char_record_bbox(rec)]

    median_width = float(sorted(widths)[len(widths) // 2]) if widths else 0.0
    median_height = float(sorted(heights)[len(heights) // 2]) if heights else 0.0
    median_center_y = float(sorted(centers_y)[len(centers_y) // 2]) if centers_y else 0.0
    return {
        "median_width": median_width,
        "median_height": median_height,
        "median_center_y": median_center_y,
        "count": int(len(prepared)),
    }

def _char_record_confidence(self, rec) -> float:
    if isinstance(rec, dict):
        try:
            return float(rec.get("confidence", 0.0))
        except Exception:
            return 0.0

    try:
        return float(getattr(rec, "confidence", 0.0))
    except Exception:
        return 0.0

def _clone_character_detection(self, rec, character=None, method=None):
    symbol, _ = self._char_record_to_symbol_and_x(rec)
    bbox = self._char_record_bbox(rec) or (0.0, 0.0, 1.0, 1.0)
    confidence = self._char_record_confidence(rec)

    return CharacterDetection(
        character=str(character if character is not None else symbol or ""),
        bbox=tuple(float(v) for v in bbox[:4]),
        confidence=float(confidence),
        method=str(method if method is not None else getattr(rec, "method", "ocr") if not isinstance(rec, dict) else rec.get("method", "ocr")),
    )
