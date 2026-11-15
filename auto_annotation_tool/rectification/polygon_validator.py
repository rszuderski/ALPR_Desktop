#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walidacja i naprawianie poligonów (ramek wokół tablic).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import logger
from .quad_geometry import analyze_quad, canonicalize_polygon


class PolygonValidator:
    """Geometria obrysu użytkowego; nie certyfikuje indeksów modelu Pose."""

    analyze_quad = staticmethod(analyze_quad)
    canonicalize_polygon = staticmethod(canonicalize_polygon)

    @staticmethod
    def sort_points_clockwise(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Sortuj punkty w porządku zgodnym z ruchem wskazówek zegara.
        """
        result = canonicalize_polygon(points)
        return result.polygon if result.polygon is not None else points

    @staticmethod
    def order_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Stabilny cykl ekranowy. Kotwica min(X+Y) nie poświadcza znaczenia TL.
        """
        result = canonicalize_polygon(points)
        if result.polygon is None:
            return points
        anchor = min(range(4), key=lambda i: (sum(result.polygon[i]),
                                              result.polygon[i][1], result.polygon[i][0]))
        return result.polygon[anchor:] + result.polygon[:anchor]

    @staticmethod
    def is_valid_quad(points: List[Tuple[float, float]]) -> bool:
        """
        Wypukły prosty cykl w dowolnym obiegu, bez modyfikowania punktów.
        """
        return analyze_quad(points).is_geometrically_valid

    @staticmethod
    def fix_polygon(points: List[Tuple[float, float]], *, quiet: bool = False) -> List[Tuple[float, float]]:
        """
        Obrys do wyświetlania; nie naprawia złego zbioru lokalizacji punktów.

        Kod zachowujący indeksy modelu/GT powinien użyć analyze_quad albo
        zapisać jawną permutację z canonicalize_polygon zamiast tego wrappera.
        """
        ordered = PolygonValidator.order_points(points)
        if PolygonValidator.is_valid_quad(ordered):
            return ordered
        if not quiet:
            logger.warning("Nieprawidłowa geometria polygonu: %s", analyze_quad(points).status)
        return points

    @staticmethod
    def points_to_vector(points: List[Tuple[float, float]]) -> np.ndarray:
        """Konwertuj listę punktów na numpy array."""
        return np.array(points, dtype=np.float32)
