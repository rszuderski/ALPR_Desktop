"""Operator-authorized metric eligibility, separate from immutable GT geometry."""
from copy import deepcopy
import json
from pathlib import Path

from .eval396 import digest, require

METRIC_POLICY_ID = "EVAL396_POSE478_BBOX479_v1"
METRIC_POLICY_SHA = "4bbb605401c0751e1c4118eadd6f1da3a1b458ba87a17c51fc1ca2a1885c877a"
METRIC_POLICY_FILE = Path(__file__).with_name("protocols") / "eval396_pose478_v1.json"


def load_metric_policy(reader=None):
    payload = reader.read(METRIC_POLICY_FILE) if reader else METRIC_POLICY_FILE.read_bytes()
    policy = json.loads(payload)
    # Canonical JSON SHA is portable across LF/CRLF checkouts. The evidence
    # receipt separately pins the actual bytes used for this review.
    require(digest(policy) == METRIC_POLICY_SHA, "METRIC_POLICY_CHANGED")
    return policy


def bound_pose_exclusion(scene, plate, geometry_issue, policy):
    """An ID alone cannot authorize an exception to a geometry check."""
    for exclusion in policy["exclusions"]:
        if plate["plate_id"] != exclusion["plate_id"]:
            continue
        source_geometry = {
            "source_file_sha256": scene["sha256"],
            "source_annotation_id": plate["source_annotation_id"],
            "source_bbox": plate["bbox"], "source_polygon": plate["quad"],
        }
        matches = (
            geometry_issue == exclusion["expected_geometry_issue"]
            and scene["file"] == exclusion["scene_file"]
            and scene["sha256"] == exclusion["source_scene_sha256"]
            and scene["domain"] == exclusion["domain"]
            and scene["image"]["oriented_size"] == exclusion["oriented_size"]
            and plate["source_annotation_id"] == exclusion["source_annotation_id"]
            and plate["bbox"] == exclusion["original_bbox"]
            and plate["quad"] == exclusion["original_quad"]
            and digest(source_geometry) == exclusion["source_geometry_sha256"]
        )
        return deepcopy(exclusion) if matches else None
    return None


def metric_population(scenes):
    per_domain = {}
    for domain in ("ALL", "DAY", "NIGHT"):
        selected = [s for s in scenes if domain == "ALL" or s["domain"] == domain]
        plates = [p for s in selected for p in s["plates"]]
        per_domain[domain] = {"scenes": len(selected),
            "bbox": sum(p["bbox_eligible"] for p in plates),
            "pose": sum(p["pose_eligible"] for p in plates),
            "ocr": sum(p["ocr_eligible"] for p in plates)}
    return {**per_domain["ALL"], "pose_exclusions": sum(bool(p["pose_exclusion"]) for s in scenes for p in s["plates"]),
            "per_domain": per_domain}
