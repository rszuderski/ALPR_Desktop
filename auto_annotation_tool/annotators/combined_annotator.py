#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined annotator: vehicle-first plate detection (mode C).
"""

from pathlib import Path
from typing import List, Optional, Tuple

from ..config import CONFIG, CV2_AVAILABLE, YOLO_AVAILABLE, cv2, get_yolo_class, logger
from ..data_models import AnnotationStatus, Detection, ImageAnnotation
from ..quality_metrics import compute_plate_polygon_fit_metrics
from ..utils import cleanup_gpu_memory, get_image_size
from .base import BaseAnnotator


class CombinedAnnotator(BaseAnnotator):
    """
    Mode C:
    1. Detect vehicles on full image.
    2. For each detected vehicle, run plate detection only inside vehicle crop.
    3. Keep the best plate that fits inside the vehicle bbox.

    Output remains helper-friendly for Z2:
    - <box label="vehicle">
    - <polygon label="plate">

    Final YOLO export still uses only plates.
    """

    COCO_VEHICLE_CLASSES = {2, 3, 5, 7}
    VEHICLE_CROP_PADDING_RATIO = 0.06
    VEHICLE_CROP_MIN_PADDING_PX = 8
    PLATE_PAIR_OVERLAP_THRESHOLD = 0.82
    PLATE_PAIR_IOU_THRESHOLD = 0.58

    def __init__(
        self,
        vehicle_model_path: Path,
        plate_model_path: Path,
        vehicle_confidence: float = 0.25,
        plate_confidence: float = 0.25,
        plate_inside_threshold: float = 0.95,
        device: str = "auto",
    ):
        super().__init__(vehicle_confidence, device)

        self.vehicle_model_path = Path(vehicle_model_path)
        self.plate_model_path = Path(plate_model_path)
        self.vehicle_confidence = vehicle_confidence
        self.plate_confidence = plate_confidence
        self.plate_inside_threshold = plate_inside_threshold

        self.vehicle_model: Optional[object] = None
        self.plate_model: Optional[object] = None
        self.is_plate_pose_model = False
        self.vehicle_class_names = {}

    def load_models(self) -> Tuple[bool, str]:
        if not YOLO_AVAILABLE:
            return False, "YOLO niedostepny"
        YoloClass = get_yolo_class()
        if YoloClass is None:
            return False, "YOLO niedostepny"

        try:
            logger.info(f"Ladowanie modelu pojazdow: {self.vehicle_model_path}")
            self.vehicle_model = YoloClass(str(self.vehicle_model_path))
            if hasattr(self.vehicle_model, "names"):
                self.vehicle_class_names = self.vehicle_model.names

            logger.info(f"Ladowanie modelu tablic: {self.plate_model_path}")
            self.plate_model = YoloClass(str(self.plate_model_path))
            if hasattr(self.plate_model, "model") and hasattr(self.plate_model.model, "kpt_shape"):
                self.is_plate_pose_model = True
                logger.info("Model tablic: POSE")
            else:
                logger.warning("Model tablic nie jest typu POSE")

            return True, "Modele zaladowane"
        except Exception as e:
            return False, f"Blad: {e}"

    def unload_models(self):
        if self.vehicle_model:
            del self.vehicle_model
            self.vehicle_model = None
        if self.plate_model:
            del self.plate_model
            self.plate_model = None
        cleanup_gpu_memory()

    def process_image(self, image_path: Path) -> ImageAnnotation:
        width, height = get_image_size(image_path)

        annotation = ImageAnnotation(
            filename=image_path.name,
            width=width,
            height=height,
        )

        try:
            image_source = self._read_image_for_yolo(image_path)
            if image_source is None:
                return self._make_image_error_annotation(
                    image_path,
                    self._describe_image_read_error(image_path),
                )

            yolo_source = str(image_path) if image_source is True else image_source
            image = None if image_source is True else image_source

            vehicles = self._detect_vehicles(yolo_source)
            if not vehicles:
                annotation.status = AnnotationStatus.NO_VEHICLE
                annotation.status_message = "Nie wykryto zadnego pojazdu"
                return annotation

            if image is None and CV2_AVAILABLE:
                try:
                    image = cv2.imread(str(image_path))
                except Exception:
                    image = None

            if image is None:
                plates = self._detect_plates(str(image_path), width, height)
                if not plates:
                    annotation.status = AnnotationStatus.NO_PLATE
                    annotation.status_message = "Nie wykryto zadnej tablicy"
                    return annotation

                matched_pairs = self._match_plates_to_vehicles(vehicles, plates)
                if not matched_pairs:
                    annotation.status = AnnotationStatus.PARTIAL_PLATE
                    annotation.status_message = "Tablice nie mieszcza sie w calosci wewnatrz pojazdow"
                    return annotation
            else:
                matched_pairs: List[Tuple[Detection, Detection]] = []
                had_any_plate_candidate = False

                for vehicle in vehicles:
                    if self.is_stopped():
                        annotation.status = AnnotationStatus.SKIPPED
                        annotation.status_message = "Przetwarzanie przerwane"
                        return annotation

                    crop_plates = self._detect_plates_for_vehicle(
                        image,
                        vehicle,
                        width,
                        height,
                    )
                    if crop_plates:
                        had_any_plate_candidate = True

                    best_plate = self._select_best_plate_for_vehicle(vehicle, crop_plates)
                    if best_plate is not None:
                        matched_pairs.append((vehicle, best_plate))

                if not matched_pairs:
                    annotation.status = (
                        AnnotationStatus.PARTIAL_PLATE if had_any_plate_candidate else AnnotationStatus.NO_PLATE
                    )
                    annotation.status_message = (
                        "Wykryte tablice nie mieszcza sie w pojazdach lub sa tylko czesciowo widoczne"
                        if had_any_plate_candidate
                        else "Nie wykryto zadnej tablicy w obrebie wykrytych pojazdow"
                    )
                    return annotation

            original_pair_count = len(matched_pairs)
            matched_pairs = self._deduplicate_matched_plate_pairs(matched_pairs)
            suppressed_pair_count = max(0, original_pair_count - len(matched_pairs))

            for vehicle, plate in matched_pairs:
                annotation.detections.append(vehicle)
                annotation.detections.append(plate)

            annotation.status = AnnotationStatus.SUCCESS
            annotation.status_message = f"Znaleziono {len(matched_pairs)} par pojazd-tablica"
            if suppressed_pair_count:
                annotation.status_message += f" (odrzucono {suppressed_pair_count} duplikatow tablic)"
            return annotation
        except Exception as e:
            annotation.status = AnnotationStatus.ERROR
            annotation.status_message = str(e)
            logger.error(f"Blad: {image_path.name}: {e}")
            return annotation

    def _detect_vehicles(self, image_source) -> List[Detection]:
        vehicles: List[Detection] = []

        results = self.vehicle_model(
            image_source,
            conf=self.vehicle_confidence,
            device=self.device,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return vehicles

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, classes):
            class_name = self.vehicle_class_names.get(cls_id, "").lower()
            is_vehicle = cls_id in self.COCO_VEHICLE_CLASSES or class_name in CONFIG.VEHICLE_LABELS
            if not is_vehicle:
                continue
            x1, y1, x2, y2 = map(float, box)
            vehicles.append(
                Detection(
                    label="vehicle",
                    confidence=float(conf),
                    bbox=(x1, y1, x2, y2),
                )
            )

        return vehicles

    def _detect_plates(
        self,
        image_source,
        width: int,
        height: int,
        *,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> List[Detection]:
        plates: List[Detection] = []

        results = self.plate_model(
            image_source,
            conf=self.plate_confidence,
            device=self.device,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return plates

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        keypoints = None
        if self.is_plate_pose_model and hasattr(result, "keypoints") and result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()

        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = map(float, box)
            x1 += float(offset_x)
            y1 += float(offset_y)
            x2 += float(offset_x)
            y2 += float(offset_y)

            polygon = None
            kpts_list = None

            if keypoints is not None and i < len(keypoints):
                raw_kpts = self._normalize_keypoints(keypoints[i])
                if raw_kpts:
                    translated_kpts = []
                    for kp_x, kp_y, kp_conf in raw_kpts:
                        translated_kpts.append(
                            (
                                float(kp_x) + float(offset_x),
                                float(kp_y) + float(offset_y),
                                float(kp_conf),
                            )
                        )
                    kpts_list = translated_kpts
                    if len(kpts_list) >= 4:
                        corners = [(float(kpts_list[j][0]), float(kpts_list[j][1])) for j in range(4)]
                        valid = all(0 <= p[0] <= width and 0 <= p[1] <= height for p in corners)
                        if valid:
                            polygon = self._sort_corners_clockwise(corners)

            if polygon is None:
                polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

            detection = Detection(
                label="plate",
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                keypoints=kpts_list,
                polygon=polygon,
            )
            try:
                fit_metrics = compute_plate_polygon_fit_metrics(
                    float(conf),
                    polygon,
                    (x1, y1, x2, y2),
                    keypoints=kpts_list,
                    image_size=(width, height),
                )
                detection.attributes.update(
                    {
                        "fit_score": f"{float(fit_metrics.get('fit_score', 0.0) or 0.0):.3f}",
                        "fit_label": str(fit_metrics.get("fit_label") or "").strip(),
                        "fit_keypoint_score": f"{float(fit_metrics.get('keypoint_score', 0.0) or 0.0):.3f}",
                        "fit_shape_score": f"{float(fit_metrics.get('shape_score', 0.0) or 0.0):.3f}",
                        "fit_bbox_score": f"{float(fit_metrics.get('bbox_alignment_score', 0.0) or 0.0):.3f}",
                        "fit_size_score": f"{float(fit_metrics.get('size_score', 0.0) or 0.0):.3f}",
                    }
                )
            except Exception:
                pass
            plates.append(detection)

        deduplicated = self._suppress_overlapping_detections(
            plates,
            overlap_threshold=0.82,
            iou_threshold=0.58,
        )
        if len(deduplicated) != len(plates):
            logger.debug(
                f"[CombinedAnnotator] Odrzucono {len(plates) - len(deduplicated)} nakladajacych sie detekcji tablic."
            )
        return deduplicated

    def _get_vehicle_crop_bounds(
        self,
        vehicle_bbox: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int] | None:
        try:
            x1, y1, x2, y2 = [float(v) for v in vehicle_bbox[:4]]
        except Exception:
            return None

        vehicle_w = max(1.0, x2 - x1)
        vehicle_h = max(1.0, y2 - y1)
        pad_x = max(float(self.VEHICLE_CROP_MIN_PADDING_PX), vehicle_w * float(self.VEHICLE_CROP_PADDING_RATIO))
        pad_y = max(float(self.VEHICLE_CROP_MIN_PADDING_PX), vehicle_h * float(self.VEHICLE_CROP_PADDING_RATIO))

        crop_x1 = max(0, int(round(x1 - pad_x)))
        crop_y1 = max(0, int(round(y1 - pad_y)))
        crop_x2 = min(int(image_width), int(round(x2 + pad_x)))
        crop_y2 = min(int(image_height), int(round(y2 + pad_y)))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None
        return crop_x1, crop_y1, crop_x2, crop_y2

    def _detect_plates_for_vehicle(
        self,
        image,
        vehicle: Detection,
        image_width: int,
        image_height: int,
    ) -> List[Detection]:
        crop_bounds = self._get_vehicle_crop_bounds(vehicle.bbox, image_width, image_height)
        if crop_bounds is None:
            return []

        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
        try:
            vehicle_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        except Exception:
            vehicle_crop = None
        if vehicle_crop is None or getattr(vehicle_crop, "size", 0) == 0:
            return []

        crop_plates = self._detect_plates(
            vehicle_crop,
            image_width,
            image_height,
            offset_x=float(crop_x1),
            offset_y=float(crop_y1),
        )
        return [
            plate
            for plate in crop_plates
            if plate.is_inside(vehicle.bbox, threshold=self.plate_inside_threshold)
        ]

    @staticmethod
    def _select_best_plate_for_vehicle(vehicle: Detection, plates: List[Detection]) -> Optional[Detection]:
        if not plates:
            return None
        _ = vehicle
        best_plate = None
        best_conf = -1.0
        for plate in plates:
            try:
                plate_conf = float(getattr(plate, "confidence", 0.0) or 0.0)
            except Exception:
                plate_conf = 0.0
            if plate_conf > best_conf:
                best_conf = plate_conf
                best_plate = plate
        return best_plate

    def _deduplicate_matched_plate_pairs(
        self,
        matched_pairs: List[Tuple[Detection, Detection]],
    ) -> List[Tuple[Detection, Detection]]:
        if len(matched_pairs or []) <= 1:
            return list(matched_pairs or [])

        ordered = sorted(
            list(enumerate(matched_pairs or [])),
            key=lambda item: (
                -float(getattr(item[1][1], "confidence", 0.0) or 0.0),
                -float(getattr(item[1][0], "confidence", 0.0) or 0.0),
                -float(getattr(item[1][1], "get_area", lambda: 0.0)() or 0.0),
            ),
        )

        kept: List[Tuple[int, Detection, Detection]] = []
        for original_index, (candidate_vehicle, candidate_plate) in ordered:
            duplicate = False
            for _existing_index, _existing_vehicle, existing_plate in kept:
                overlap = self._bbox_overlap_over_smaller(candidate_plate.bbox, existing_plate.bbox)
                iou = self._bbox_iou(candidate_plate.bbox, existing_plate.bbox)
                if (
                    overlap >= float(self.PLATE_PAIR_OVERLAP_THRESHOLD)
                    or iou >= float(self.PLATE_PAIR_IOU_THRESHOLD)
                ):
                    duplicate = True
                    break
            if not duplicate:
                kept.append((original_index, candidate_vehicle, candidate_plate))

        if len(kept) != len(matched_pairs):
            logger.debug(
                "[CombinedAnnotator] Odrzucono %s duplikatow par pojazd-tablica po detekcji w cropach.",
                len(matched_pairs) - len(kept),
            )
        return [(vehicle, plate) for _index, vehicle, plate in sorted(kept, key=lambda item: item[0])]

    def _match_plates_to_vehicles(
        self,
        vehicles: List[Detection],
        plates: List[Detection],
    ) -> List[Tuple[Detection, Detection]]:
        matched: List[Tuple[Detection, Detection]] = []
        used_plates = set()

        for vehicle in vehicles:
            best_plate = None
            best_plate_idx = -1
            best_conf = 0.0

            for i, plate in enumerate(plates):
                if i in used_plates:
                    continue
                if plate.is_inside(vehicle.bbox, threshold=self.plate_inside_threshold):
                    if float(plate.confidence) > best_conf:
                        best_plate = plate
                        best_plate_idx = i
                        best_conf = float(plate.confidence)

            if best_plate is not None:
                matched.append((vehicle, best_plate))
                used_plates.add(best_plate_idx)

        return matched

    @staticmethod
    def _sort_corners_clockwise(corners: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(corners) != 4:
            return corners

        cx = sum(p[0] for p in corners) / 4
        cy = sum(p[1] for p in corners) / 4

        top = [p for p in corners if p[1] < cy]
        bottom = [p for p in corners if p[1] >= cy]

        if len(top) != 2 or len(bottom) != 2:
            sorted_by_y = sorted(corners, key=lambda p: p[1])
            top = sorted(sorted_by_y[:2], key=lambda p: p[0])
            bottom = sorted(sorted_by_y[2:], key=lambda p: p[0], reverse=True)
        else:
            top = sorted(top, key=lambda p: p[0])
            bottom = sorted(bottom, key=lambda p: p[0], reverse=True)

        return [top[0], top[1], bottom[0], bottom[1]]
