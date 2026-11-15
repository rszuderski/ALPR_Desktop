"""Result rules shared by Z2's worker and the final preview merge."""

from ..config import CONFIG
import copy
from pathlib import Path


def remove_vehicle_assistance(annotations) -> int:
    """Remove auxiliary vehicle detections from a new plate-only result.

    Call only on the new result, never on the pre-run rollback snapshot.
    Imported/manual plate geometry and its attributes remain unchanged.
    """
    removed = 0
    for annotation in annotations:
        detections = list(annotation.detections or [])
        retained = [
            detection for detection in detections
            if str(detection.label or "").strip().lower() not in CONFIG.VEHICLE_LABELS
        ]
        removed += len(detections) - len(retained)
        annotation.detections = retained
    return removed


def merge_completed_view(annotations, image_map, previous_state):
    """Keep unprocessed rows before committing the run, outside the Tk thread."""
    previous = list(previous_state.get("annotations") or [])
    if not previous:
        return annotations, image_map
    by_name = {ann.filename.lower(): ann for ann in annotations}
    result, seen = [], set()
    merged_map = dict(previous_state.get("image_map") or {})
    merged_map.update(image_map)
    for old in previous:
        name = old.filename.lower()
        if name in seen:
            continue
        seen.add(name)
        result.append(by_name[name] if name in by_name else copy.deepcopy(old))
    result.extend(ann for ann in annotations if ann.filename.lower() not in seen)
    return result, merged_map


def protected_view_bundle(previous_state, protected_names, *, build_manual_override=None):
    """Preserve reviewed images, but only manual shapes in unfinished edits."""
    paths = {str(name).lower(): path for name, path in (previous_state.get("image_map") or {}).items()}
    input_dir = previous_state.get("input_dir")
    approved = {str(name).lower() for name in previous_state.get("approved_filenames", ())}
    bundle = {}
    for ann in previous_state.get("annotations") or []:
        name = ann.filename.lower()
        if name not in protected_names:
            continue
        path = paths.get(name)
        if path is None and input_dir:
            path = Path(input_dir) / ann.filename
        if path is not None:
            is_approved = bool(getattr(ann, "_approved_for_training", False) or name in approved)
            if not is_approved and build_manual_override is not None:
                # Match the persisted manual-overlay policy. Protecting an
                # entire edited image also kept stale auto detections alive.
                preserved = build_manual_override(ann)
            elif is_approved and not getattr(ann, "_approved_for_training", False):
                preserved = copy.deepcopy(ann)
                preserved._approved_for_training = True
            else:
                preserved = ann
            bundle[ann.filename] = (preserved, Path(path))
    return bundle


def vehicle_assistance_visible(owner) -> bool:
    """Imported vehicle geometry is auxiliary; the active choice controls it."""
    if getattr(owner, "_manual_xml_template_enabled", lambda: False)():
        return bool(getattr(owner, "_manual_vehicle_assist_enabled", lambda: False)())
    return getattr(owner, "_get_auto_vehicle_choice", lambda: "skip")() == "use"
