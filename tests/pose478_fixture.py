"""Synthetic full-cohort structure, never real evaluation inputs or approvals."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from auto_annotation_tool.ranking.mt_metric_policy import load_metric_policy


def cohort(first_scene):
    policy = load_metric_policy()
    exception = policy["exclusions"][0]
    scenes = []
    plate_number = 0
    for index in range(396):
        domain = "DAY" if index < 199 else "NIGHT"
        local_index = index if domain == "DAY" else index - 199
        extra = 36 if domain == "DAY" else 47
        scene = deepcopy(first_scene)
        if index:
            scene.update(file=f"{domain}/fixture_{index}.jpg",
                         sha256=hashlib.sha256(f"fixture_scene_{index}".encode()).hexdigest())
            scene["image"]["exif_orientation"] = 1
        scene["domain"] = domain
        scene["plates"] = []
        for _ in range(1 + (local_index < extra)):
            p = deepcopy(first_scene["plates"][0])
            p.update(plate_id=f"fixture_{plate_number}", source_annotation_id=f"ann-fixture-{plate_number}",
                     text_decision="approved")
            scene["plates"].append(p)
            plate_number += 1
        scenes.append(scene)
    # One exact operator-bound exception; every other record is synthetic.
    scene = scenes[198]
    scene.update(file=exception["scene_file"], sha256=exception["source_scene_sha256"])
    scene["image"]["oriented_size"] = exception["oriented_size"]
    p = scene["plates"][0]
    p.update(plate_id=exception["plate_id"], source_annotation_id=exception["source_annotation_id"],
             bbox=deepcopy(exception["original_bbox"]), quad=deepcopy(exception["original_quad"]),
             text_decision="excluded_unreadable")
    # Five DAY and twenty-five NIGHT unreadable; they still belong to MT.
    for domain, count in (("DAY", 4), ("NIGHT", 25)):
        plates = [p for s in scenes if s["domain"] == domain for p in s["plates"]
                  if p["plate_id"] != exception["plate_id"]]
        for p in plates[:count]:
            p["text_decision"] = "excluded_unreadable"
    return scenes


def bind_source_fixture(report, tmp_path, monkeypatch):
    """Fixture-specific trusted manifest, using the real byte-binding path."""
    from auto_annotation_tool.ranking import mt_geometry_review as gate
    selection = Path(report["selection_dir"])
    freeze = Path(report["freeze_dir"])
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        path.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        report["source_files"][str(path.resolve())] = sha
        return sha
    scenes = [{k: s[k] for k in ("file", "sha256", "domain")} for s in report["scenes"]]
    plates, metadata = [], {}
    for scene in report["scenes"]:
        for p in scene["plates"]:
            plates.append({"plate_id": p["plate_id"], "source_annotation_id": p["source_annotation_id"],
                           "source_file_sha256": scene["sha256"], "domain": scene["domain"], "decision": p["text_decision"]})
            metadata[p["plate_id"]] = {"source_bbox": p["bbox"], "source_polygon": p["quad"]}
    files = {name: write(selection / name, value) for name, value in (
        ("included_scenes.json", scenes), ("included_plates.json", plates), ("model_lineage.json", report["models"]))}
    manifest = {"selection_fingerprint_sha256": gate.SELECTION_SHA, "files": files,
                "references": {"source_scene_root": ".", "source_snapshot_folder": "freeze",
                               "parent_metadata_sha256": write(freeze / "metadata.json", metadata)}}
    monkeypatch.setattr(gate, "SELECTION_MANIFEST_SHA", write(selection / "selection_manifest.json", manifest))
