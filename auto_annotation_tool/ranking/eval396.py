"""Read-only adapter for the accepted EVAL396 selection and completed MZ runs.

This module never loads a model, runs inference or mutates the source artifacts.
The accepted fingerprints are trust anchors, not values inferred from an input.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from ..training.source_inventory import inventory_for_export, validate_inventory
from .mz_whole_plate_evaluation import evaluate_plate_records, aggregate_plate_metrics

SELECTION_SHA = "1f2db18cd073114f230ca2dd70a76ea9d1257d624dddf7ff39f51ada8e66aefe"
FREEZE_SHA = "33add3fda705416bb88ed9b1ab71e275d2c4f36dd8f5b1bb2f2cbc59130ff77b"
COMPARISON_SHA = "91d2beac538c732b2ce18ca23a1003861180342815d443657acf7aca14b44b30"
SELECTION_MANIFEST_SHA = "06e3b1c1f9e5c3891d53038b381fe909f5dac65215ead0358a47b7830f56116a"
RUN_MANIFEST_SHAS = {
    "MZ-s": "16a427e7aea4bbbc6ab973dfb0296be556d9f277260f510e02e91937657dfd38",
    "MZ-DAY": "4d9dd641bce0f43824700e4b793471ade0203e4e8aa29f20839023f9c5b94ce2",
    "MZ-NIGHT": "87175415207bc87dc219f3e4123c2bf236e2eeae3165c2771151a4e733d3ae11",
}
SELECTION_NAME = "E-MZ-DN-01_EVAL396_20261009"
EXPERIMENT_ID = "E-MZ-DN-01_EVAL396_MZ_v1"
MODELS = ("MT-n", "MT-DAY", "MT-NIGHT", "MZ-s", "MZ-DAY", "MZ-NIGHT")
VARIANTS = MODELS[3:]
DOMAINS = ("ALL", "DAY", "NIGHT")
INFERENCE = {"augment": False, "confidence": .25, "device": 0, "end2end": True,
             "half": False, "imgsz": 320, "iou": .45, "max_det": 300}
COUNTS = {"scenes": {"ALL": 396, "DAY": 199, "NIGHT": 197}, "plates": {
    "ALL": {"all": 479, "approved": 449, "excluded_unreadable": 30},
    "DAY": {"all": 235, "approved": 230, "excluded_unreadable": 5},
    "NIGHT": {"all": 244, "approved": 219, "excluded_unreadable": 25}}}


class EvaluationImportError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvaluationImportError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    path = root / str(relative)
    require(not Path(str(relative)).is_absolute() and ":" not in str(relative)
            and ".." not in Path(str(relative)).parts and path.resolve().is_relative_to(root),
            f"UNSAFE_PATH: {relative}")
    return path


class EvidenceReader:
    def __init__(self, progress=None):
        self.receipt = {}
        self.progress = progress or (lambda message: None)

    def read(self, path, expected=None, *, json_value=False):
        path = Path(path).resolve()
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        require(expected is None or sha == expected, f"SHA_MISMATCH: {path.name}")
        previous = self.receipt.setdefault(str(path), sha)
        require(previous == sha, f"SOURCE_CHANGED_DURING_IMPORT: {path.name}")
        return json.loads(data) if json_value else data

    def json(self, path, expected=None):
        return self.read(path, expected, json_value=True)

    def finish(self):
        self.progress("Sprawdzam niezmienność źródeł po odczycie…")
        for path, sha in list(self.receipt.items()):
            self.read(path, sha)


def default_selection(repo_root):
    return Path(repo_root) / "Workspace/E-MZ-DN-01/evaluation_selections" / SELECTION_NAME


def is_selection(path):
    """UI routing only: a known but unreadable selection still opens its gate.

    Recognition never certifies data. load_selection/load_eval396 retain all
    source SHA checks and propagate read failures to the native status panel.
    """
    path = Path(str(path or ""))
    if path.name in {"selection_manifest.json", SELECTION_NAME}:
        return True
    try:
        return (path / "selection_manifest.json").is_file()
    except OSError:
        return False


def selection_reference(path):
    """Cheap discovery for UI; explicitly not an independence certificate."""
    path = Path(path)
    if path.name == "selection_manifest.json":
        path = path.parent
    manifest = json.loads((path / "selection_manifest.json").read_text(encoding="utf-8-sig"))
    require(manifest.get("schema") == "emzdn01.eval_scene_selection.v1"
            and manifest.get("selection_fingerprint_sha256") == SELECTION_SHA,
            "Nieznana selekcja; wymagany zaakceptowany kontrakt EVAL396.")
    return {"ok": True, "kind": "eval396", "selected_path": str(path), "reference_dir": str(path),
            "reference_name": "EVAL396 — zamrożona selekcja", "split_name": "selekcyjny EVAL",
            "image_count": 396, "status": "INCOMPLETE", "message":
            "396 scen (199 DAY / 197 NIGHT) • MT: 479 tablic • MZ: 449 tekstów • 30 nieczytelnych. "
            "WYNIKI: zweryfikuj i odczytaj zapisane MZ. MT: wymagana decyzja EXIF."}


def _verify_freeze(reader, root, scene_root, references):
    manifest = reader.json(root / "freeze_manifest.json")
    require(manifest.get("freeze_fingerprint_sha256") == FREEZE_SHA, "GT_FREEZE_MISMATCH")
    data = {}
    for name in ("source_inventory.json", "plates.json", "crop_inventory.json",
                 "gt_pack_inventory.json", "sidecar_inventory.json"):
        data[name] = reader.json(root / name)
        require(digest(data[name]) == manifest["files"][name], "GT_CANONICAL_SHA_MISMATCH: " + name)
    reader.read(root / "metadata.json", manifest["files"]["metadata.json"])
    require(manifest["metadata_sha256"] == references["parent_metadata_sha256"], "GT_METADATA_MISMATCH")
    for name, ref in (("source_inventory.json", "parent_source_inventory_sha256"),
                      ("plates.json", "parent_plates_sha256")):
        require(manifest["files"][name] == references[ref], "GT_REFERENCE_MISMATCH: " + name)
    fingerprint = {"schema": manifest["schema"], "metadata_sha256": manifest["metadata_sha256"],
                   "source_inventory_digest": manifest["source_inventory_audit_sha256"]}
    for key, name in (("sources", "source_inventory"), ("plates", "plates"), ("crops", "crop_inventory"),
                      ("gt_pack_files", "gt_pack_inventory"), ("sidecars", "sidecar_inventory")):
        fingerprint[key + "_sha256"] = digest(data[name + ".json"])
    require(digest(fingerprint) == FREEZE_SHA, "GT_FINGERPRINT_MISMATCH")
    for name in ("crop_inventory.json", "gt_pack_inventory.json", "sidecar_inventory.json"):
        for row in data[name]:
            reader.read(contained(root, row["file"]), row["sha256"])
    for row in data["source_inventory.json"]:
        reader.read(contained(scene_root, row["file"]), row["sha256"])
    return data


def _verify_models(reader, repo_root, lineage, scenes):
    require(len(lineage) == 6 and {m["label"] for m in lineage} == set(MODELS), "SIX_MODELS_REQUIRED")
    by_id = {m["run_id"]: m for m in lineage}
    require(len(by_id) == 6, "DUPLICATE_RUN_ID")
    for model in lineage:
        reader.progress("Sprawdzam źródła i rodowód: " + model["label"])
        reader.read(repo_root / model["checkpoint_file"], model["checkpoint_sha256"])
        history = reader.json(repo_root / model["history_file"], model["history_sha256"])
        runs = history.get("runs", []) if isinstance(history, dict) else history
        if isinstance(runs, dict):
            runs = list(runs.values())
        matching = [run for run in runs if run.get("id") == model["run_id"]]
        require(len(matching) == 1, "AMBIGUOUS_TRAINING_RUN")
        run = matching[0]
        require(run.get("status") == "completed", "TRAINING_RUN_NOT_COMPLETED")
        snapshot = run.get("training_dataset_snapshot") or {}
        require(snapshot.get("split_sha256") == model["split_sha256"], "TRAINING_SPLIT_CHANGED")
        if not snapshot.get("training_source_hash_inventory"):
            # Legacy inventories are reconstructed from verified dataset bytes.
            # Keep those inputs in the same before/after/export receipt as GT.
            dataset = Path(snapshot.get("local_path_hint") or run.get("dataset_path") or "")
            require(dataset.is_dir(), "LEGACY_DATASET_MISSING")
            for path in sorted(dataset.rglob("*")):
                if path.is_file():
                    reader.read(path)
        inventory = inventory_for_export({"training": {"dataset": snapshot,
            "dataset_path": run.get("dataset_path", "")}}, role=model["role"])
        validate_inventory(inventory, role=model["role"])
        require(inventory["status"] == "complete", "SOURCE_INVENTORY_INCOMPLETE: " + model["label"])
        hashes = {row["sha256"] for row in inventory["source_scene_hashes"]}
        require(len(hashes) == model["source_scene_count"] and digest(sorted(hashes)) == model["source_set_sha256"],
                "TRAINING_SOURCE_SET_CHANGED: " + model["label"])
        require(not hashes & scenes, "TRAIN_EVAL_OVERLAP: " + model["label"])
        parent = model["parent_run_id"]
        require(str(run.get("parent_run_id") or "") == parent, "MODEL_PARENT_CHANGED")
        if parent:
            require(parent in by_id and by_id[parent]["role"] == model["role"], "MISSING_PARENT_INVENTORY")
            require((run.get("input_checkpoint_snapshot") or {}).get("sha256") == by_id[parent]["checkpoint_sha256"],
                    "PARENT_CHECKPOINT_MISMATCH")


def _read_selection(reader, selection_dir, repo_root):
    manifest = reader.json(selection_dir / "selection_manifest.json", SELECTION_MANIFEST_SHA)
    require(manifest.get("schema") == "emzdn01.eval_scene_selection.v1"
            and manifest.get("status") == "EVALUATION_SELECTION_CREATED", "INVALID_SELECTION")
    names = {"included_scenes.json", "excluded_scenes.json", "included_plates.json", "model_lineage.json"}
    require(set(manifest["files"]) == names, "SELECTION_FILES_CHANGED")
    data = {name: reader.json(selection_dir / name, sha) for name, sha in manifest["files"].items()}
    fp = digest({"schema": manifest["schema"], "counts": manifest["counts"],
                 "references": manifest["references"], "files_sha256": manifest["files"]})
    require(fp == manifest["selection_fingerprint_sha256"] == SELECTION_SHA, "SELECTION_FINGERPRINT_MISMATCH")
    require(manifest["counts"] == COUNTS, "SELECTION_COUNTS_CHANGED")
    refs = manifest["references"]
    require(refs["parent_freeze_fingerprint_sha256"] == FREEZE_SHA, "WRONG_PARENT_FREEZE")
    require(digest(data["model_lineage.json"]) == refs["model_lineage_digest"], "LINEAGE_DIGEST_MISMATCH")
    freeze = contained(repo_root, refs["source_snapshot_folder"])
    frozen = _verify_freeze(reader, freeze, contained(repo_root, refs["source_scene_root"]), refs)
    scenes = data["included_scenes.json"]
    included = {row["sha256"] for row in scenes}
    excluded = {row["sha256"] for row in data["excluded_scenes.json"]}
    require(len(included) == len(scenes) == 396 and len(excluded) == 4 and not included & excluded,
            "INVALID_SCENE_PARTITION")
    require(included | excluded == {row["sha256"] for row in frozen["source_inventory.json"]}, "SCENE_PARTITION_CHANGED")
    require(scenes == sorted([r for r in frozen["source_inventory.json"] if r["sha256"] in included],
                             key=lambda r: (r["domain"], r["sha256"])), "SCENES_DIFFER_FROM_GT")
    plates = data["included_plates.json"]
    require(plates == sorted([r for r in frozen["plates.json"] if r["source_file_sha256"] in included],
                             key=lambda r: r["plate_id"]), "PLATES_DIFFER_FROM_GT")
    require(len({row["plate_id"] for row in plates}) == 479, "DUPLICATE_OR_MISSING_PLATE")
    for domain in DOMAINS:
        population = [p for p in plates if domain == "ALL" or p["domain"] == domain]
        counts = dict(Counter(p["decision"] for p in population))
        require({"all": len(population), **counts} == COUNTS["plates"][domain], "PLATE_DOMAIN_COUNTS_CHANGED")
        require(sum(domain == "ALL" or s["domain"] == domain for s in scenes) == COUNTS["scenes"][domain],
                "SCENE_DOMAIN_COUNTS_CHANGED")
    _verify_models(reader, repo_root, data["model_lineage.json"], included)
    return manifest, data


def load_selection(selection_dir, *, repo_root=None, progress=None):
    """Verify frozen inputs and six-model lineage without reading old MZ results.

    This is the same strict selection/freeze/independence gate as the MZ
    importer. PermissionError and incomplete inventories are never waived.
    """
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    directory = Path(selection_dir).resolve()
    if directory.name == "selection_manifest.json":
        directory = directory.parent
    reader = EvidenceReader(progress)
    manifest, data = _read_selection(reader, directory, root)
    reader.finish()
    return {"selection_dir": str(directory), "repo_root": str(root),
            "selection_sha": SELECTION_SHA, "gt_freeze_sha": FREEZE_SHA,
            "manifest": manifest, "scenes": data["included_scenes.json"],
            "plates": data["included_plates.json"], "models": data["model_lineage.json"],
            "source_files": dict(reader.receipt)}


def _read_mz_results(reader, runs_dir, selection):
    approved = {p["plate_id"]: p for p in selection["included_plates.json"] if p["decision"] == "approved"}
    lineage = {m["label"]: m for m in selection["model_lineage.json"]}
    comparison = reader.json(runs_dir / "comparison.json", COMPARISON_SHA)
    require(comparison.get("schema") == "emzdn01.eval396.mz_plate_comparison.v1"
            and comparison.get("selection_fingerprint_sha256") == SELECTION_SHA
            and comparison.get("inference") == INFERENCE, "INVALID_COMPARISON")
    summaries = {}
    for label in VARIANTS:
        reader.progress("Weryfikuję zapisane odczyty: " + label)
        folder = runs_dir / label
        manifest = reader.json(folder / "run_manifest.json", RUN_MANIFEST_SHAS[label])
        require(manifest.get("schema") == "emzdn01.eval396.mz_plate_recognition_run.v1"
                and manifest.get("variant") == label
                and manifest.get("selection_fingerprint_sha256") == SELECTION_SHA
                and manifest.get("checkpoint_sha256") == lineage[label]["checkpoint_sha256"], "INVALID_MZ_MANIFEST")
        summary = reader.json(folder / "summary.json", manifest["files"]["summary.json"])
        rows = [json.loads(line) for line in reader.read(folder / "rows.jsonl", manifest["files"]["rows.jsonl"]).splitlines() if line.strip()]
        require(summary.get("schema") == manifest["schema"] and summary.get("model") == label
                and summary.get("run_id") == lineage[label]["run_id"]
                and summary.get("checkpoint_sha256") == lineage[label]["checkpoint_sha256"]
                and summary.get("selection_fingerprint_sha256") == SELECTION_SHA
                and summary.get("gt_freeze_fingerprint_sha256") == FREEZE_SHA
                and summary.get("inference") == INFERENCE
                and summary.get("source") == "approved_GT_crops_only_no_visual_unreadables", "INVALID_MZ_SUMMARY")
        require(len(rows) == len(approved) and {r["plate_id"] for r in rows} == set(approved), "MZ_POPULATION_MISMATCH")
        for row in rows:
            gt = approved[row["plate_id"]]
            require(row.get("checkpoint_label") == label and row.get("domain") == gt["domain"]
                    and row.get("source_file_sha256") == gt["source_file_sha256"]
                    and row.get("crop_sha256") == gt["crop_sha256"]
                    and row.get("ground_truth") == gt["ground_truth_text"], "MZ_ROW_GT_MISMATCH")
            actual = json.loads(canonical(evaluate_plate_records(gt["ground_truth_text"], row["ordered_predictions"])))
            require(all(row.get(key) == value for key, value in actual.items()), "MZ_ROW_METRICS_MISMATCH")
        metrics = {domain: aggregate_plate_metrics([r for r in rows if domain == "ALL" or r["domain"] == domain])
                   for domain in DOMAINS}
        require(metrics == summary["per_domain"] == comparison["model_summaries"][label], "MZ_AGGREGATION_MISMATCH")
        summaries[label] = summary
    require(set(comparison["model_summaries"]) == set(VARIANTS), "COMPARISON_MODELS_CHANGED")
    return summaries


def load_eval396(selection_dir, *, repo_root=None, runs_dir=None, progress=None):
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    selection_dir = Path(selection_dir).resolve()
    if selection_dir.name == "selection_manifest.json":
        selection_dir = selection_dir.parent
    runs_dir = Path(runs_dir or repo_root / "Workspace/E-MZ-DN-01/evaluation_runs" / EXPERIMENT_ID)
    reader = EvidenceReader(progress)
    try:
        reader.progress("Weryfikuję selekcję i pliki GT…")
        manifest, selection = _read_selection(reader, selection_dir, repo_root)
        summaries = _read_mz_results(reader, runs_dir, selection)
        reader.finish()
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise EvaluationImportError("FAIL — import EVAL396: " + str(exc)) from exc
    return {"schema": "alpr.z4.imported_evaluation.v1", "status": "PASS", "experiment_id": EXPERIMENT_ID,
            "source_kind": "import_existing_measurements", "selection_sha": SELECTION_SHA, "gt_freeze_sha": FREEZE_SHA,
            "comparison_sha": COMPARISON_SHA, "selection_dir": str(selection_dir), "runs_dir": str(runs_dir),
            "counts": manifest["counts"], "models": selection["model_lineage.json"], "summaries": summaries,
            "inference": INFERENCE, "source_files": reader.receipt,
            "limitations": manifest["limitations"] + ["Izolowane MZ na GT cropach; bez MT, E2E i pomiaru Androida.",
                "MT: polityka współrzędnych EXIF wymaga osobnego zatwierdzenia."]}
