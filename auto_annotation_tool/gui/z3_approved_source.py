"""Materialize the exact approved plate geometry for campaign extraction.

An image-level OK flag on a historical XML does not approve every polygon in
that XML. The project's approved entries contain the reviewed plate snapshots.
"""
import hashlib
import json
import math
import os
from pathlib import Path

from ..data_models import AnnotationStatus, Detection, ImageAnnotation
from ..exporters import CVATExporter
from ..plate_import_validation import project_plate_conflicts, conflict_shape_key, is_unchanged_vehicle_plate


def build_approved_plate_source(entries, state_dir) -> dict:
    conflicts = project_plate_conflicts(state_dir)
    conflict_index = {conflict_shape_key(c["filename"], c["image_size"], c["polygon"]): c for c in conflicts}
    records = []
    for entry in entries:
        source = Path(str(entry.get("source_image_path") or "")).resolve()
        if not source.is_file():
            raise ValueError(f"Brakuje zatwierdzonego obrazu: {entry.get('image_name') or source}")
        plates = []
        for plate in entry.get("plates") or []:
            points = [(float(x), float(y)) for x, y in plate.get("polygon", [])]
            area = abs(sum(x1*y2-x2*y1 for (x1,y1),(x2,y2) in zip(points, points[1:]+points[:1])))
            if len(points) != 4 or not all(math.isfinite(v) for p in points for v in p) or area <= 0:
                raise ValueError(f"Nieprawidłowa zatwierdzona geometria tablicy: {entry.get('image_name')}")
            attributes = dict(plate.get("attributes") or {})
            if plate.get("ground_truth_text"):
                attributes.setdefault("ground_truth_text", plate["ground_truth_text"])
            if is_unchanged_vehicle_plate(entry.get("image_name"), (entry.get("width"), entry.get("height")),
                                          points, attributes, conflict_index):
                continue
            plates.append({"polygon": points, "confidence": float(plate.get("confidence", 1.) or 1.),
                           "attributes": attributes})
        if plates:
            records.append({"source": str(source), "width": int(entry.get("width", 0)),
                            "height": int(entry.get("height", 0)), "plates": plates})
    if not records:
        raise ValueError("Brak zatwierdzonych tablic do wycięcia. Zatwierdź tablice w Z2 i wróć do grafu.")
    records.sort(key=lambda item: item["source"].casefold())
    try:
        images_dir = Path(os.path.commonpath([str(Path(item["source"]).parent) for item in records]))
    except ValueError as exc:
        raise ValueError("Zatwierdzone obrazy muszą być dostępne we wspólnym drzewie katalogów.") from exc
    signature = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    run_dir = Path(state_dir) / "z3_approved_sources" / signature
    xml_path = run_dir / "annotations.xml"
    manifest_path = run_dir / "run_manifest.json"
    result = {"run_dir": run_dir, "xml_path": xml_path, "images_dir": images_dir,
              "approved_images": len(records), "approved_plates": sum(len(r["plates"]) for r in records),
              "approved_geometry_signature": signature}
    if xml_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("xml_sha256") == hashlib.sha256(xml_path.read_bytes()).hexdigest():
                return result
        except (OSError, ValueError):
            pass
    annotations = []
    for item in records:
        detections = []
        for plate in item["plates"]:
            points = plate["polygon"]
            detections.append(Detection(
                "plate", plate["confidence"],
                (min(x for x,y in points), min(y for x,y in points), max(x for x,y in points), max(y for x,y in points)),
                polygon=points, attributes=plate["attributes"],
            ))
        annotations.append(ImageAnnotation(
            Path(item["source"]).relative_to(images_dir).as_posix(), item["width"], item["height"],
            detections, status=AnnotationStatus.SUCCESS,
        ))
    run_dir.mkdir(parents=True, exist_ok=True)
    if not CVATExporter(task_name="Campaign approved plate geometry").export(
            annotations, xml_path, include_confidence=True, only_successful=False):
        raise OSError("Nie udało się przygotować zatwierdzonego źródła tablic dla Z3.")
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in result.items()}
    manifest.update(annotation_run_type="campaign_approved_for_z3", derived_from_approved_only=True,
                    input_dir=str(images_dir), approved_filenames=[a.filename.lower() for a in annotations],
                    xml_sha256=hashlib.sha256(xml_path.read_bytes()).hexdigest())
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    return result


def source_for_campaign(campaign) -> dict:
    state_dir = campaign.get_project_state_dir()
    if state_dir is None:
        raise ValueError("Nie znaleziono aktywnego projektu dla wycinania tablic.")
    return build_approved_plate_source(campaign.list_plate_approved_entries() or [], state_dir)


def bind_approved_source(host, source):
    host._source_binding_sync_in_progress = True
    try:
        host.annotation_run_dir_var.set(str(source["run_dir"]))
        host.xml_path_var.set(str(source["xml_path"]))
        host.images_dir_var.set(str(source["images_dir"]))
    finally:
        host._source_binding_sync_in_progress = False
