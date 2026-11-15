#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Silnik rektyfikacji tablic z odpornym deskewingiem konturowym.
"""

from __future__ import annotations

from typing import List, Tuple

from ..config import CV2_AVAILABLE, cv2, np
from .polygon_validator import PolygonValidator


class PlateRectifier:
    INTERPOLATIONS = {"nearest": 0, "linear": 1, "cubic": 2, "lanczos4": 4}

    @staticmethod
    def predicted_quad_geometry(raw_points, image_size):
        """Deterministic homography plan, reusable to validate persisted evidence."""
        from math import floor, hypot
        from .quad_geometry import analyze_quad, canonicalize_polygon

        if not CV2_AVAILABLE or cv2 is None:
            raise RuntimeError("E2E_OPENCV_REQUIRED")
        width, height = image_size
        analysis = analyze_quad(raw_points, (width, height))
        evidence = {"policy": "predicted_quad_anchor_k0_aspect320_v1",
                    "raw_quad_status": analysis.status, "status": "QUAD_INVALID",
                    "permutation": None, "homography": None, "size": None, "pixel_sha256": None}
        if not analysis.is_geometrically_valid:
            return evidence
        ordered = canonicalize_polygon(raw_points, anchor_index=0, image_size=(width, height))
        if ordered.polygon is None:
            return evidence
        q = ordered.polygon
        evidence["permutation"] = ordered.permutation
        w = max(hypot(q[1][0]-q[0][0], q[1][1]-q[0][1]), hypot(q[2][0]-q[3][0], q[2][1]-q[3][1]))
        h = max(hypot(q[3][0]-q[0][0], q[3][1]-q[0][1]), hypot(q[2][0]-q[1][0], q[2][1]-q[1][1]))
        scale = 320 / max(w, h)
        out_w, out_h = floor(scale*w + .5), floor(scale*h + .5)
        evidence["size"] = [out_w, out_h]
        evidence["status"] = "RECTIFICATION_FAILED"
        if min(out_w, out_h) < 2:
            evidence["reason"] = "CROP_TOO_THIN"
            return evidence
        try:
            source = np.array(q, dtype=np.float32)
            destination = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(source, destination)
            if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
                evidence["reason"] = "SINGULAR_HOMOGRAPHY"
                return evidence
        except cv2.error as exc:
            # Resource/backend failures must stop execution, never become NO_READ.
            if getattr(exc, "code", None) == getattr(getattr(cv2, "Error", None), "StsNoMem", -4):
                raise
            evidence["reason"] = "HOMOGRAPHY_ERROR: " + str(exc)
            return evidence
        evidence.update(status="RECTIFIED", homography=matrix.tolist())
        return evidence

    @classmethod
    def rectify_predicted_quad(cls, image_bgr, raw_points):
        """Strict E2E: k0 anchor, aspect320, no legacy sorting or auto rotation."""
        import hashlib
        if not CV2_AVAILABLE or cv2 is None:
            raise RuntimeError("E2E_OPENCV_REQUIRED")
        if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
                or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) < 1):
            raise ValueError("E2E_INVALID_BGR_INPUT")
        height, width = image_bgr.shape[:2]
        evidence = cls.predicted_quad_geometry(raw_points, (width, height))
        if evidence["status"] != "RECTIFIED":
            return None, evidence
        out_w, out_h = evidence["size"]
        try:
            crop = cv2.warpPerspective(image_bgr, np.asarray(evidence["homography"], dtype=np.float64), (out_w, out_h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        except cv2.error as exc:
            if getattr(exc, "code", None) == getattr(getattr(cv2, "Error", None), "StsNoMem", -4):
                raise
            evidence.update(status="RECTIFICATION_FAILED", reason="HOMOGRAPHY_ERROR: " + str(exc))
            return None, evidence
        if crop is None or crop.shape != (out_h, out_w, 3):
            evidence["status"] = "RECTIFICATION_FAILED"
            evidence["reason"] = "INVALID_WARP_RESULT"
            return None, evidence
        evidence["pixel_sha256"] = hashlib.sha256(crop.tobytes()).hexdigest()
        return crop, evidence

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
        Buduje bezpieczny obwód do transformacji, z kotwicą ekranową min(X+Y).

        Jest to orientacja obrazu użytkowego, a nie certyfikat tożsamości
        surowych keypointów modelu. Nigdy nie dzieli punktów na pary po X/Y.
        """
        ordered = PolygonValidator.order_points(pts)
        if not PolygonValidator.is_valid_quad(ordered):
            raise ValueError("Do rektyfikacji wymagany jest wypukły, niesamoprzecinający czworokąt.")
        return np.array(ordered, dtype="float32")

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
