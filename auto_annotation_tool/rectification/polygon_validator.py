#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walidacja i naprawianie poligonów (ramek wokół tablic).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import logger


class PolygonValidator:
    """Waliduje i naprawia poligony (ramki tablic)."""

    @staticmethod
    def sort_points_clockwise(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Sortuj punkty w porządku zgodnym z ruchem wskazówek zegara.
        """
        if len(points) != 4:
            logger.warning(f"Oczekiwano 4 punktów, otrzymano {len(points)}")
            return points

        pts = np.array(points, dtype=np.float32)
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        sorted_pts = pts[sorted_indices]
        return [tuple(float(v) for v in p) for p in sorted_pts]

    @staticmethod
    def order_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Porządkuj punkty: top-left, top-right, bottom-right, bottom-left.
        """
        if len(points) != 4:
            return points

        pts = np.array(points, dtype=np.float32)
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        ordered = pts[np.argsort(angles)]

        start_idx = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
        ordered = np.roll(ordered, -start_idx, axis=0)

        if ordered[1][0] < ordered[-1][0]:
            ordered = np.array([ordered[0], ordered[-1], ordered[-2], ordered[-3]], dtype=np.float32)

        return [tuple(float(v) for v in p) for p in ordered]

    @staticmethod
    def is_valid_quad(points: List[Tuple[float, float]]) -> bool:
        """
        Sprawdź czy cztery punkty tworzą prawidłowy czworokąt (bez kokard).
        """
        if len(points) != 4:
            return False

        unique_points = {(float(x), float(y)) for x, y in points}
        if len(unique_points) != 4:
            return False

        pts = np.array(points, dtype=np.float32)
        x = pts[:, 0]
        y = pts[:, 1]
        area = 0.5 * abs(
            x[0] * (y[1] - y[3])
            + x[1] * (y[2] - y[0])
            + x[2] * (y[3] - y[1])
            + x[3] * (y[0] - y[2])
        )
        if area < 10:
            return False

        try:
            import cv2

            poly = cv2.contourArea(pts.reshape(-1, 1, 2))
            if poly < 10:
                return False
        except Exception:
            pass

        return True

    @staticmethod
    def fix_polygon(points: List[Tuple[float, float]], *, quiet: bool = False) -> List[Tuple[float, float]]:
        """
        Napraw poligon (usuwa kokardę).

        `quiet=True` przydaje się przy masowym wczytywaniu XML do podglądu,
        żeby nie zalewać logu ostrzeżeniami dla historycznych, już zapisanych polygonów.
        """
        if len(points) != 4:
            if not quiet:
                logger.warning(f"Poligon ma {len(points)} punktów, oczekiwano 4")
            return points

        ordered = PolygonValidator.order_points(points)
        if PolygonValidator.is_valid_quad(ordered):
            logger.debug("Poligon uporzadkowany (zmiana kolejnosci punktow)")
            return ordered

        clockwise = PolygonValidator.sort_points_clockwise(points)
        if PolygonValidator.is_valid_quad(clockwise):
            logger.debug("Poligon uporzadkowany (sortowanie po kacie)")
            return clockwise

        if not quiet:
            logger.warning("Nie udało się naprawić poligonu - może być kokarda")
        return points

    @staticmethod
    def points_to_vector(points: List[Tuple[float, float]]) -> np.ndarray:
        """Konwertuj listę punktów na numpy array."""
        return np.array(points, dtype=np.float32)
