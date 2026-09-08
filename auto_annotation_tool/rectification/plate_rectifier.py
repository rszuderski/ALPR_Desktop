#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Silnik rektyfikacji tablic z odpornym deskewingiem konturowym.
"""

from __future__ import annotations

from typing import List, Tuple

from ..config import CV2_AVAILABLE, cv2, np


class PlateRectifier:
    INTERPOLATIONS = {"nearest": 0, "linear": 1, "cubic": 2, "lanczos4": 4}

    @staticmethod
    def polygon_wh_px(pts: List[Tuple[float, float]]) -> Tuple[float, float]:
        if len(pts) != 4:
            return 0.0, 0.0
        p = np.array(pts, dtype="float32")
        w = np.linalg.norm(p[0] - p[1])
        h = np.linalg.norm(p[0] - p[3])
        return w, h

    @staticmethod
    def order_quad_points(pts: List[Tuple[float, float]]) -> np.ndarray:
        """
        Ustawia punkty w kolejności: top-left, top-right, bottom-right, bottom-left.

        Dla mocno skośnych tablic podział po samym Y bywa zawodny, bo obie lewe
        albo obie prawe krawędzie mogą wpaść do różnych połówek obrazu. Stabilniej
        działa najpierw rozdzielenie lewej i prawej pary po osi X, a dopiero potem
        ułożenie każdej pary po osi Y.
        """
        p = np.array(pts, dtype="float32")
        if p.shape != (4, 2):
            raise ValueError("Do rektyfikacji wymagane są dokładnie 4 punkty.")

        x_sorted = p[np.argsort(p[:, 0])]
        left_pair = x_sorted[:2][np.argsort(x_sorted[:2, 1])]
        right_pair = x_sorted[2:][np.argsort(x_sorted[2:, 1])]

        top_left, bottom_left = left_pair
        top_right, bottom_right = right_pair
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

    @classmethod
    def deskew(cls, image_bgr: np.ndarray) -> np.ndarray:
        """
        Szuka konturów liter i wyrównuje obraz do mediany ich kąta.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        angles = []
        h_img, w_img = image_bgr.shape[:2]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > h_img * 0.3 and h < h_img * 0.9 and w < w_img * 0.5:
                rect = cv2.minAreaRect(contour)
                angle = rect[-1]

                if angle < -45:
                    angle = -(90 + angle)
                elif angle > 45:
                    angle = 90 - angle

                angles.append(angle)

        if len(angles) < 3:
            return image_bgr

        best_angle = np.median(angles)
        if abs(best_angle) > 15:
            return image_bgr

        return cls.rotate_image(image_bgr, best_angle)

    @staticmethod
    def rotate_image(image, angle):
        if abs(angle) < 0.01:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    @classmethod
    def rectify(
        cls,
        image_bgr,
        pts: List[Tuple[float, float]],
        out_w_px: int,
        out_h_px: int,
        interpolation: str = "lanczos4",
        sharpen: float = 0.0,
        enhance_contrast: bool = False,
        do_deskew: bool = False,
        manual_angle: float = 0.0,
    ):
        if not CV2_AVAILABLE:
            return image_bgr

        rect_pts = cls.order_quad_points(pts)
        dst = np.array(
            [[0, 0], [out_w_px - 1, 0], [out_w_px - 1, out_h_px - 1], [0, out_h_px - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect_pts, dst)
        interp_idx = cls.INTERPOLATIONS.get(interpolation, cv2.INTER_LANCZOS4)
        warped = cv2.warpPerspective(image_bgr, matrix, (out_w_px, out_h_px), flags=interp_idx)

        if do_deskew:
            warped = cls.deskew(warped)

        if abs(manual_angle) > 0:
            warped = cls.rotate_image(warped, manual_angle)

        if enhance_contrast:
            lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            warped = cv2.cvtColor(
                cv2.merge((clahe.apply(l_chan), a_chan, b_chan)),
                cv2.COLOR_LAB2BGR,
            )

        if sharpen > 0:
            blur = cv2.GaussianBlur(warped, (0, 0), 3)
            warped = cv2.addWeighted(warped, 1.0 + sharpen, blur, -sharpen, 0)

        h_out, w_out = warped.shape[:2]
        if h_out > w_out * 1.1:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped
