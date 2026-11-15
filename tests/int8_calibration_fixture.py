"""Synthetic RGB inputs; val/test colors must never reach calibration."""
import hashlib
import json
from pathlib import Path

from PIL import Image
from auto_annotation_tool.exporters import int8_calibration as cal


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def holdout(path, purpose, hashes):
    if purpose == "EVAL396":
        hashes = list(hashes) + [hashlib.sha256(f"synthetic-eval-{i}".encode()).hexdigest()
                                  for i in range(396 - len(hashes))]
    p = {"schema": cal.HOLDOUT_SCHEMA, "purpose": purpose, "status": "frozen", "cohort_id": "synthetic-" + purpose,
         "approved_by": "synthetic test operator", "source_scene_hashes": hashes, "scene_count": len(hashes)}
    p["inventory_fingerprint"] = cal.fingerprint(p)
    write_json(path, p)


def fixture(root, role="plate", duplicate=False):
    root = Path(root)
    dataset = root / "dataset"
    rows, scenes, artifacts = [], [], []
    paths = {}
    for split, colors in (("train", [(30, 40, 50), (80, 90, 100), (130, 140, 150)]),
                          ("val", [(230, 0, 0)]), ("test", [(0, 230, 0)])):
        for index, color in enumerate(colors):
            p = dataset / f"images/{split}/{index}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 10), color).save(p)
            sha = cal.file_sha256(p)
            source = sha if role == "plate" else hashlib.sha256(f"scene-{split}-{index}".encode()).hexdigest()
            paths[split + str(index)] = p
            scenes.append({"sha256": source, "splits": [split]})
            artifacts.append({"sha256": sha, "split": split, "kind": "crop" if role == "character" else "image"})
            rows.append({"image_path": p.relative_to(dataset).as_posix(), "split": split,
                         "provenance": {"source_image_id": "img-sha256-" + source},
                         "group_id": "source_image_id:img-sha256-" + source})
    if duplicate:
        p = dataset / "images/train/duplicate.png"
        p.write_bytes(paths["train0"].read_bytes())
        rows.append({**rows[0], "image_path": "images/train/duplicate.png"})
        artifacts.append(dict(artifacts[0]))
    (dataset / "data.yaml").write_text("train: images/train\nval: images/val\ntest: images/test\nnames: [plate]\n", encoding="utf-8")
    manifests = []
    if role == "character":
        for name in ("metadata_manifest.json", "split_assignment_manifest.json"):
            p = write_json(dataset / name, {"items": rows})
            manifests.append({"name": name, "sha256": cal.file_sha256(p)})
    checkpoint = root / "run/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint; never loaded")
    inv = {"schema": "alpr.training_source_hashes.v1", "role": role, "dataset_id": "DS-synthetic",
           "status": "complete", "source_scene_hashes": scenes, "artifact_hashes": artifacts,
           "sample_count": len(artifacts), "unresolved_source_count": 0, "unresolved_dataset_count": 0,
           "enumeration_error_count": 0}
    run = {"id": "synthetic", "status": "completed", "dataset_path": str(dataset), "best_weights": str(checkpoint),
           "training_dataset_snapshot": {"dataset_id": "DS-synthetic", "snapshot_source": "frozen_at_training_start",
               "data_yaml_sha256": cal.file_sha256(dataset / "data.yaml"), "training_source_hash_inventory": inv,
               "manifests": manifests}}
    history = write_json(root / "history/training_history.json", {"runs": {"synthetic": run}})
    eval_path, final_path = root / "exclusions/eval.json", root / "exclusions/final.json"
    holdout(eval_path, "EVAL396", ["a" * 64])
    holdout(final_path, "final_test", ["b" * 64])
    config = dict(history_path=str(history), run_id="synthetic", role=role, eval_inventory=str(eval_path),
                  final_inventory=str(final_path), image_size=32, limit=2, approved_by="synthetic operator")
    return dict(config=config, dataset=dataset, checkpoint=checkpoint, paths=paths, history=history,
                run=run, rows=rows, inv=inv, eval_path=eval_path, final_path=final_path)


def save_plan(f, target):
    manifest = cal.build_manifest(**f["config"])
    write_json(target, manifest)
    return manifest
