"""Train-only, content-addressed calibration; no GT, model or dataset writes.

Approval is an operator statement, not a cryptographic identity signature.
Holdout inventories contain hashes only. Their images are never opened here.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re

from ..training.source_inventory import file_sha256, validate_inventory, _source_sha

SCHEMA = "alpr.int8_calibration.v1"
HOLDOUT_SCHEMA = "alpr.holdout_source_hashes.v1"
POLICY = "train_sha_order_letterbox_v1"
LITERT_ADAPTER_VERSION = "8.4.19"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class CalibrationError(ValueError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _check(condition, message):
    if not condition:
        raise CalibrationError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _ref(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": file_sha256(p)}


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def preprocessing(image_size):
    width, height = (image_size, image_size) if isinstance(image_size, int) else image_size
    _check(type(width) is int and type(height) is int and width > 0 and height > 0,
           "Nieprawidłowy rozmiar wejścia kalibracji.")
    return {"policy": POLICY, "width": width, "height": height, "color": "RGB",
            "exif": "stored_pixels_no_transpose", "resize": "letterbox_center",
            "interpolation": "cv2.INTER_LINEAR", "scale_up": True, "padding": 114,
            "rounding": "python_round; border +/-0.1", "augmentation": False,
            "onnx": "NCHW float32 / 255", "litert_loader": "NCHW uint8",
            "litert_converter": "NHWC float32; mean=0 std=255",
            "libraries": {n: importlib.metadata.version(n) for n in ("Pillow", "numpy", "opencv-python", "ultralytics")}}


def exclusion_inventory(path, purpose):
    """Accept a frozen hash-only contract, or the native EVAL396 selection."""
    _check(bool(path), f"BLOCKED: brak inwentarza SHA {purpose}.")
    p = Path(path).resolve()
    payload = _read(p)
    refs = [_ref(p)]
    if purpose == "EVAL396" and payload.get("schema") == "emzdn01.eval_scene_selection.v1":
        _check(payload.get("status") == "EVALUATION_SELECTION_CREATED", "EVAL396 nie jest zamrożony.")
        scenes = p.parent / "included_scenes.json"
        refs.append(_ref(scenes))
        _check(refs[-1]["sha256"] == payload.get("files", {}).get(scenes.name), "Niezgodne SHA EVAL396.")
        hashes = [row["sha256"] for row in _read(scenes)]
        count = payload["counts"]["scenes"]["ALL"]
        _check(count == 396, "EVAL396 musi zawierać 396 scen.")
    else:
        _check(payload.get("schema") == HOLDOUT_SCHEMA and payload.get("purpose") == purpose,
               f"Nieobsługiwany inwentarz {purpose} (wymagany {HOLDOUT_SCHEMA}).")
        _check(payload.get("status") == "frozen" and bool(str(payload.get("approved_by", "")).strip())
               and bool(str(payload.get("cohort_id", "")).strip()), f"Brak zatwierdzenia kohorty {purpose}.")
        _check(payload.get("inventory_fingerprint") == fingerprint(
            {k: v for k, v in payload.items() if k != "inventory_fingerprint"}), f"Niezgodny fingerprint {purpose}.")
        hashes, count = payload["source_scene_hashes"], payload["scene_count"]
    _check(type(count) is int and count > 0 and isinstance(hashes, list) and len(hashes) == count
           and all(_sha(h) for h in hashes) and len(set(hashes)) == count,
           f"Niepełny lub powielony inwentarz {purpose}.")
    _check(purpose != "EVAL396" or count == 396, "EVAL396 wymaga kompletnej kohorty 396 scen.")
    return set(hashes), {"purpose": purpose, "scene_count": count, "files": refs,
                         "hashes_fingerprint": fingerprint(sorted(hashes))}


def _train_paths(root):
    import yaml
    cfg = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8-sig"))
    _check(isinstance(cfg, dict) and cfg.get("train"), "Brak jawnego splitu train w data.yaml.")
    data_root = (root / str(cfg.get("path") or ".")).resolve()
    sources = cfg["train"] if isinstance(cfg["train"], list) else [cfg["train"]]
    paths = set()
    # Calibration is restricted to physical images/train, including resolved links.
    train_root = (root / "images/train").resolve()
    for raw in sources:
        source = (data_root / str(raw)).resolve()
        if source.is_dir():
            candidates = list(source.rglob("*"))
        elif source.suffix.lower() == ".txt":
            candidates = [(source.parent / line[2:] if line.startswith("./") else data_root / line)
                          for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        else:
            candidates = [source]
        for p in candidates:
            if p.suffix.lower() not in IMAGE_SUFFIXES or not p.is_file():
                continue
            resolved = p.resolve()
            _check(resolved.is_relative_to(train_root), "Split train wskazuje poza images/train: " + str(p))
            paths.add(resolved)
    _check(bool(paths), "Pusty zbiór obrazów train.")
    return sorted(paths, key=str)


def build_manifest(*, history_path, run_id, role, eval_inventory, final_inventory,
                   image_size, limit=300, approved_by=""):
    """Read-only plan. Missing holdout proof returns BLOCKED with candidate evidence."""
    _check(role in {"plate", "character"}, "F4-B2 obsługuje sceny MT i cropy MZ.")
    _check(type(limit) is int and limit > 0, "Liczba próbek musi być dodatnia.")
    history = Path(history_path).resolve()
    run = _read(history)["runs"][run_id]
    _check(run.get("status") == "completed", "Źródło musi pochodzić z ukończonego treningu.")
    snapshot = run["training_dataset_snapshot"]
    inv = snapshot["training_source_hash_inventory"]
    validate_inventory(inv, role=role)
    _check(inv["status"] == "complete" and inv["dataset_id"] == snapshot["dataset_id"]
           and snapshot.get("snapshot_source") == "frozen_at_training_start",
           "Wymagany kompletny inwentarz zamrożony przed treningiem.")
    _check(len(inv["artifact_hashes"]) == inv["sample_count"], "Niepełne SHA artefaktów treningu.")
    run_root = Path(run["dataset_path"]).resolve()
    refs = [_ref(history), _ref(run_root / "data.yaml")]
    _check(refs[-1]["sha256"] == snapshot["data_yaml_sha256"], "Zmieniono data.yaml treningu.")
    frozen_names = set()
    for entry in snapshot.get("manifests", []):
        p = (run_root / entry["name"]).resolve()
        _check(p.is_relative_to(run_root), "Nieprawidłowy manifest treningu.")
        ref = _ref(p)
        _check(ref["sha256"] == entry["sha256"], "Zmieniono manifest treningu: " + entry["name"])
        refs.append(ref)
        frozen_names.add(entry["name"])
    root = run_root
    aug = run_root / "augmentation_scope.json"
    if aug.exists():
        refs.append(_ref(aug))
        root = Path(_read(aug)["source_dataset_dir"]).resolve()
    refs.append(_ref(root / "data.yaml"))
    scene_splits = {r["sha256"]: set(r["splits"]) for r in inv["source_scene_hashes"]}
    artifact_splits = {}
    for row in inv["artifact_hashes"]:
        artifact_splits.setdefault(row["sha256"], set()).add(row["split"])
    maps = []
    if role == "character":
        for name in ("metadata_manifest.json", "split_assignment_manifest.json"):
            _check(name in frozen_names, "Brak zamrożonego powiązania crop–scena: " + name)
            rows = {}
            for row in _read(run_root / name)["items"]:
                key = str(row.get("image_path") or "").replace("\\", "/")
                _check(key not in rows, "Powielone powiązanie cropa w " + name)
                rows[key] = row
            maps.append(rows)
    blockers = []
    holdouts, exclusions = {}, []
    for purpose, path in (("EVAL396", eval_inventory), ("final_test", final_inventory)):
        try:
            hashes, evidence = exclusion_inventory(path, purpose)
            holdouts[purpose] = hashes
            exclusions.append(evidence)
        except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
            code = "FINAL_TEST_INVENTORY_NOT_VERIFIED" if purpose == "final_test" else "EVAL396_INVENTORY_NOT_VERIFIED"
            blockers.append(f"{code}: {exc}")
    if not str(approved_by).strip():
        blockers.append("Brak jawnego zatwierdzenia źródła przez operatora.")
    candidates, rejected = [], []
    for path in _train_paths(root):
        sha = file_sha256(path)
        rel = path.relative_to(root).as_posix()
        scene = sha
        if maps:
            matched = [mapping.get(rel, {}) for mapping in maps]
            identities = {_source_sha(row) for row in matched}
            _check(all(row.get("split") == "train" for row in matched)
                   and len(identities) == 1 and "" not in identities,
                   "Brak jednoznacznego pochodzenia train dla cropa: " + rel)
            scene = identities.pop()
        _check("train" in artifact_splits.get(sha, set()) and "train" in scene_splits.get(scene, set()),
               "Próbka lub scena poza zatwierdzonym treningiem: " + rel)
        row = {"path": str(path), "relative_path": rel, "sha256": sha, "source_scene_sha256": scene,
               "split": "train", "kind": "crop" if role == "character" else "scene"}
        if artifact_splits[sha] != {"train"} or scene_splits[scene] != {"train"}:
            rejected.append({**row, "reason": "source_or_artifact_also_val_test"})
            continue
        for purpose, hashes in holdouts.items():
            if sha in hashes or scene in hashes:
                blockers.append(f"Nakładanie SHA z {purpose}: {rel}")
        candidates.append(row)
    candidates.sort(key=lambda r: (r["sha256"], r["relative_path"]))
    unique = {}
    for row in candidates:
        previous = unique.setdefault(row["sha256"], row)
        _check(previous["source_scene_sha256"] == row["source_scene_sha256"], "Sprzeczne pochodzenie duplikatu SHA.")
    selected = list(unique.values())[:limit]
    if not selected:
        blockers.append("Brak niezależnych próbek train.")
    checkpoint = _ref(run["best_weights"])
    config = {"history_path": str(history), "run_id": run_id, "role": role,
              "eval_inventory": str(Path(eval_inventory).resolve()) if eval_inventory else "",
              "final_inventory": str(Path(final_inventory).resolve()) if final_inventory else "",
              "image_size": list(image_size) if isinstance(image_size, (list, tuple)) else image_size,
              "limit": limit, "approved_by": str(approved_by).strip()}
    result = {"schema": SCHEMA, "policy": POLICY, "status": "BLOCKED" if blockers else "READY",
              "blockers": blockers, "config": config, "role": role, "split": "train",
              "dataset_id": inv["dataset_id"], "source_dataset": str(root), "checkpoint": checkpoint,
              "source_files": refs, "run_fingerprint": fingerprint(run),
              "training_inventory_fingerprint": fingerprint(inv), "exclusions": exclusions,
              "gates": {"final_test_inventory": "VERIFIED" if "final_test" in holdouts else "FINAL_TEST_INVENTORY_NOT_VERIFIED",
                        "eval396_inventory": "VERIFIED" if "EVAL396" in holdouts else "EVAL396_INVENTORY_NOT_VERIFIED"},
              "selection": "ascending sample SHA256, relative path; exact sample dedup; first limit",
              "counts": {"train_candidates": len(candidates) + len(rejected), "cross_split_rejected": len(rejected),
                         "duplicates_removed": len(candidates) - len(unique), "eligible_unique": len(unique),
                         "selected": len(selected), "selected_source_scenes": len({r["source_scene_sha256"] for r in selected})},
              "candidate_fingerprint": fingerprint(candidates), "rejected": rejected, "samples": selected,
              "preprocessing": preprocessing(image_size), "execution_precision": "UNVERIFIED"}
    # Detect changes while inventorying; no plan certifies an unstable input.
    for ref in refs + [checkpoint] + [f for e in exclusions for f in e["files"]]:
        _check(file_sha256(ref["path"]) == ref["sha256"], "Plik zmienił się podczas kontroli: " + ref["path"])
    result["input_fingerprint"] = fingerprint(result)
    return result


def validate_embedded_manifest(payload, *, role=None):
    """Portable evidence integrity; does not assert live source availability."""
    _check(payload.get("schema") == SCHEMA, "INT8 wymaga manifestu bezpiecznej kalibracji z Z4; samo data.yaml nie wystarcza.")
    _check(payload.get("status") == "READY", "BLOCKED: " + "; ".join(payload.get("blockers", [])))
    _check(payload.get("input_fingerprint") == fingerprint({k: v for k, v in payload.items() if k != "input_fingerprint"}),
           "Zmieniono fingerprint manifestu kalibracji.")
    if role is not None:
        _check(payload["role"] == role, "Manifest kalibracji dotyczy innego toru modelu.")
    samples = payload["samples"]
    _check(payload["split"] == "train" and not payload["blockers"] and bool(samples)
           and len(samples) == payload["counts"]["selected"] and bool(payload["config"]["approved_by"])
           and all(r["split"] == "train" and _sha(r["sha256"]) and _sha(r["source_scene_sha256"]) for r in samples)
           and len({r["sha256"] for r in samples}) == len(samples), "Niespójne próbki lub zatwierdzenie manifestu.")
    _check(payload["gates"] == {"final_test_inventory": "VERIFIED", "eval396_inventory": "VERIFIED"}
           and {r["purpose"] for r in payload["exclusions"]} == {"EVAL396", "final_test"}
           and len(payload["exclusions"]) == 2
           and all(r["scene_count"] > 0 and _sha(r["hashes_fingerprint"]) for r in payload["exclusions"]),
           "Brak dowodów niezależności EVAL396/final_test.")


def validate_manifest(path, *, role=None, checkpoint=None, image_size=None):
    payload = _read(path)
    validate_embedded_manifest(payload, role=role)
    if checkpoint is not None:
        _check(file_sha256(checkpoint) == payload["checkpoint"]["sha256"], "Inny checkpoint niż w manifestie kalibracji.")
    if image_size is not None:
        _check(preprocessing(image_size) == payload["preprocessing"], "Zmieniono preprocessing / imgsz kalibracji.")
    current = build_manifest(**payload["config"])
    _check(current == payload, "Dane lub konfiguracja zmieniły się od zatwierdzenia kalibracji. Przygotuj nowy manifest.")
    return payload


def sample_rgb(row, config):
    """Decode the exact hashed bytes; no labels, split discovery or fallback."""
    import cv2
    import numpy as np
    from PIL import Image
    data = Path(row["path"]).read_bytes()
    _check(hashlib.sha256(data).hexdigest() == row["sha256"], "Zmieniono próbkę kalibracji: " + row["path"])
    with Image.open(io.BytesIO(data)) as image:
        array = np.asarray(image.convert("RGB"))
    h, w = array.shape[:2]
    width, height = config["width"], config["height"]
    ratio = min(width / w, height / h)
    new_w, new_h = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = cv2.resize(array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (width - new_w) / 2, (height - new_h) / 2
    return cv2.copyMakeBorder(resized, round(dh - .1), round(dh + .1), round(dw - .1), round(dw + .1),
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


class ApprovedOnnxReader:
    def __init__(self, manifest, input_name):
        self.manifest, self.input_name = manifest, input_name
        self.index = 0
        self.consumed = []

    def get_next(self):
        import numpy as np
        if self.index == len(self.manifest["samples"]):
            return None
        row = self.manifest["samples"][self.index]
        array = sample_rgb(row, self.manifest["preprocessing"])
        self.index += 1
        self.consumed.append(row["sha256"])
        return {self.input_name: np.ascontiguousarray(array.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0}

    def rewind(self):
        self.index = 0
        self.consumed = []

    def assert_complete(self):
        _check(self.consumed == [r["sha256"] for r in self.manifest["samples"]], "Kalibrator nie zużył dokładnie zatwierdzonych próbek.")


class ApprovedLiteRTLoader:
    def __init__(self, manifest):
        self.manifest = manifest
        self.consumed = []

    def __len__(self):
        return len(self.manifest["samples"])

    def __iter__(self):
        import numpy as np
        import torch
        self.consumed = []
        for row in self.manifest["samples"]:
            array = sample_rgb(row, self.manifest["preprocessing"])
            self.consumed.append(row["sha256"])
            yield {"img": torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)[None])),
                   "im_file": [row["path"]]}

    def assert_complete(self):
        _check(self.consumed == [r["sha256"] for r in self.manifest["samples"]], "LiteRT nie zużył dokładnie zatwierdzonych próbek.")


@contextmanager
def bind_litert_loader(model, manifest):
    """Per-exporter callback: never patch a global Ultralytics class or read YAML."""
    _check(importlib.metadata.version("ultralytics") == LITERT_ADAPTER_VERSION,
           "Wersja Ultralytics wymaga odbioru adaptera kalibracji LiteRT.")
    callbacks = getattr(model, "callbacks", None)
    _check(isinstance(callbacks, dict), "Runtime nie udostępnia kontroli loadera LiteRT.")
    loader = ApprovedLiteRTLoader(manifest)
    called = []

    def install(exporter):
        _check(exporter.args.split == "train" and exporter.args.batch == 1 and exporter.args.int8
               and exporter.args.format == "tflite" and exporter.args.fraction == 1.0,
               "Runtime zmienił parametry kalibracji train.")
        exporter.get_int8_calibration_dataloader = lambda prefix="": loader
        called.append(True)

    model.callbacks = {k: list(v) for k, v in callbacks.items()}
    # Install last, after existing callbacks.
    model.callbacks.setdefault("on_export_start", []).append(install)
    try:
        yield loader
        _check(len(called) == 1, "Nie uruchomiono chronionego loadera LiteRT.")
        loader.assert_complete()
    finally:
        model.callbacks = callbacks
