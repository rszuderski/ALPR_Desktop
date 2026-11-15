"""Catch legacy AT in which automatic plate polygons repeat vehicle boxes."""
from .config import CONFIG
from .data_models import Detection, ImageAnnotation
from functools import lru_cache
from pathlib import Path
import json
import xml.etree.ElementTree as ET


def conflict_shape_key(filename, size, points):
    return (str(filename).strip().casefold(), tuple(size),
            tuple(sorted((round(float(x), 2), round(float(y), 2)) for x, y in points)))


def is_unchanged_vehicle_plate(filename, size, points, attrs, index, *, source=""):
    if (str(source).lower() == "manual" or str(attrs.get("manually_edited", "")).lower() == "true"
            or attrs.get("manual_source") or str(attrs.get("ground_truth_text") or "").strip()):
        return False
    conflict = index.get(conflict_shape_key(filename, size, points))
    return bool(conflict and (not conflict["plate_annotation_id"]
                or conflict["plate_annotation_id"] == attrs.get("plate_annotation_id")))


@lru_cache(maxsize=8)
def _read_source_conflicts(path, mtime_ns, size):
    annotations = []
    for image in ET.parse(path).getroot().findall(".//image"):
        detections = []
        for shape in image:
            label = str(shape.get("label") or "").strip().lower()
            attrs = {str(a.get("name") or ""): str(a.text or "") for a in shape.findall("attribute")}
            try:
                if shape.tag == "box" and label in CONFIG.VEHICLE_LABELS:
                    det = Detection("vehicle", 1., tuple(float(shape.get(k)) for k in ("xtl","ytl","xbr","ybr")), attributes=attrs)
                elif shape.tag == "polygon" and label in CONFIG.PLATE_LABELS:
                    points = [tuple(map(float,p.split(","))) for p in shape.get("points", "").split(";")]
                    if len(points) != 4:
                        continue
                    det = Detection("plate", 1., (min(x for x,y in points),min(y for x,y in points),
                                    max(x for x,y in points),max(y for x,y in points)), polygon=points, attributes=attrs)
                else:
                    continue
            except (TypeError, ValueError):
                continue
            det._cvat_source = shape.get("source", "")
            detections.append(det)
        annotations.append(ImageAnnotation(image.get("name", ""), int(image.get("width", 0)),
                                           int(image.get("height", 0)), detections))
    return tuple(plate_vehicle_conflicts(annotations))


def project_plate_conflicts(path):
    location = Path(path).resolve()
    for parent in (location, *location.parents):
        project_path = parent / "_campaign_state" / "project.json"
        if not project_path.is_file():
            continue
        project = json.loads(project_path.read_text(encoding="utf-8-sig")).get("project", {})
        raw_source = project.get("project_start_plate_source_xml")
        if not raw_source:
            return []
        source = Path(raw_source).resolve()
        if not source.is_relative_to(parent) or not source.is_file():
            return []
        stat = source.stat()
        try:
            return list(_read_source_conflicts(str(source), stat.st_mtime_ns, stat.st_size))
        except ET.ParseError as exc:
            raise ValueError(f"Nie można odczytać źródłowego AT: {source}") from exc
    return []


def plate_vehicle_conflicts(annotations) -> list[dict]:
    conflicts = []
    for annotation in annotations:
        vehicles = [det for det in annotation.detections
                    if str(det.label).strip().lower() in CONFIG.VEHICLE_LABELS]
        for index, plate in enumerate(annotation.detections):
            if str(plate.label).strip().lower() not in CONFIG.PLATE_LABELS:
                continue
            attrs = plate.attributes or {}
            if (str(getattr(plate, "_cvat_source", "")).lower() == "manual"
                    or str(attrs.get("manually_edited", "")).lower() == "true"
                    or attrs.get("manual_source") or attrs.get("ground_truth_text")):
                continue
            x1, y1, x2, y2 = plate.bbox
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if not area:
                continue
            # This legacy failure turned a detector bbox into a rectangular
            # plate polygon. Do not infer semantic errors from aspect alone.
            points = plate.polygon or []
            if len(points) != 4:
                continue
            polygon_area = abs(sum(points[i][0] * points[(i + 1) % 4][1]
                                   - points[(i + 1) % 4][0] * points[i][1] for i in range(4))) / 2
            if polygon_area / area < .97:
                continue
            for vehicle in vehicles:
                vx1, vy1, vx2, vy2 = vehicle.bbox
                other_area = max(0, vx2 - vx1) * max(0, vy2 - vy1)
                intersection = max(0, min(x2, vx2) - max(x1, vx1)) * max(0, min(y2, vy2) - max(y1, vy1))
                union = area + other_area - intersection
                iou = intersection / union if union else 0
                if iou >= .9:
                    conflicts.append({"filename": annotation.filename, "detection_index": index,
                                      "image_size": [annotation.width, annotation.height],
                                      "plate_annotation_id": attrs.get("plate_annotation_id", ""),
                                      "polygon": [list(point) for point in points],
                                      "plate_bbox": list(plate.bbox), "vehicle_bbox": list(vehicle.bbox),
                                      "iou": round(iou, 6)})
                    break
    return conflicts


def plate_import_error(annotations) -> str:
    conflicts = plate_vehicle_conflicts(annotations)
    if not conflicts:
        return ""
    files = list(dict.fromkeys(item["filename"] for item in conflicts))
    examples = "\n".join(files[:5])
    return (
        f"AT zawiera {len(conflicts)} automatycznych ramek tablic niemal pokrywających całe pojazdy "
        f"na {len(files)} obrazach. Etykiety tego AT są niespójne — mogły powstać przy użyciu modelu pojazdów "
        "jako modelu tablic.\n\n"
        f"Przykłady:\n{examples}\n\n"
        "Popraw AT lub wybierz inny zbiór anotacji. Możesz też dodać same obrazy "
        "i uruchomić w Z2 detekcję modelem tablic. Plik źródłowy nie został zmieniony."
    )
