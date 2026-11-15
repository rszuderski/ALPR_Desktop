"""Exclude proven legacy vehicle/plate duplicates when reopening an existing run.

The source AT stays intact. Only automatic shapes with unchanged conflicting
geometry are excluded; human geometry and explicit GT survive. Image-level OK
does not turn a proven vehicle/plate duplicate into valid plate geometry.
"""
from ..config import CONFIG, logger
from ..data_models import AnnotationStatus
from ..plate_import_validation import (plate_vehicle_conflicts, project_plate_conflicts,
                                       conflict_shape_key, is_unchanged_vehicle_plate)


def exclude_restored_vehicle_plate_conflicts(owner, annotations, run_dir, manifest):
    """Run once on restore, before the list, counters and GT editor see shapes."""
    conflicts = plate_vehicle_conflicts(annotations)
    try:
        conflicts += project_plate_conflicts(run_dir)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("[Z2 RESTORE] Nie można sprawdzić źródłowego AT: %s", exc)
    index = {conflict_shape_key(item["filename"], item["image_size"], item["polygon"]): item for item in conflicts}
    if not index:
        return 0
    removed = 0
    for ann in annotations:
        rejected, retained = [], []
        for det in ann.detections:
            attrs = det.attributes or {}
            if str(det.label).strip().lower() in CONFIG.PLATE_LABELS and is_unchanged_vehicle_plate(
                    ann.filename, (ann.width, ann.height), det.polygon or [], attrs, index,
                    source=getattr(det, "_cvat_source", "")):
                rejected.append(det)
            else:
                retained.append(det)
        if rejected:
            # Retain the rejected objects for diagnostics during this session.
            ann._excluded_vehicle_plate_conflicts = rejected
            ann.detections = retained
            ann.status = AnnotationStatus.SUCCESS if ann.plates else AnnotationStatus.NO_PLATE
            ann.status_message = f"Pominięto {len(rejected)} błędnych ramek pojazdów oznaczonych w starym AT jako tablice."
            removed += len(rejected)
    if removed:
        logger.warning("[Z2 RESTORE] Pominięto %s błędnych automatycznych ramek plate z AT pojazdów: %s", removed, run_dir)
    return removed
