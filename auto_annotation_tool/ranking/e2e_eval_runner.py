"""Native Z4 quality experiment: verified MT cache -> predicted crop -> MZ.

No CLI or MT fallback. Preparation never infers. Only explicit signed Start or
Resume may call a MZ backend. The prediction worker receives no ground truth.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import math
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from .eval396 import canonical, contained, digest, require, load_selection, SELECTION_SHA, FREEZE_SHA
from . import mt_eval_runner as mt
from . import e2e_artifacts as storage
from . import e2e_io_compat as io_compat
from .e2e_eval_metrics import CHAR_COUNTS, SCORING_POLICY, validate_reference, score_pair
from .mz_whole_plate_evaluation import read_plate_records, READING_ORDER_POLICY
from ..registration_text import NORMALIZATION_POLICY
from ..rectification.plate_rectifier import PlateRectifier
from ..image_orientation import read_oriented_bgr

SCHEMA = "alpr.e2e.protocol.v1"
CAMPAIGN_SCHEMA = "alpr.e2e.campaign.v1"
APPROVAL_SCHEMA = "alpr.e2e.operator_contract.v1"
MT_PROTOCOL_SHA = "33e85df6c9a9cffd134127fc72f97c120b5583326d6eafd0a42d2bb646ae2da6"
MT_COMPLETION_SHA = "5903f3cdb11de65b67b3c0757797018ed0540e6b848145b391821c67dcdd074c"
MT_LABELS = ("MT-n", "MT-DAY", "MT-NIGHT")
MZ_LABELS = ("MZ-s", "MZ-DAY", "MZ-NIGHT")
ALPHABET = {str(i): ch for i, ch in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
RECTIFICATION = {"id": "predicted_quad_anchor_k0_aspect320_v1", "long_side": 320,
    "anchor": "raw_k0", "invalid_raw": "NO_READ_no_bbox_fallback", "reversed": "positive_cycle_anchor_k0_record_permutation",
    "dimensions": "max_opposite_edges;floor(s*edge+0.5);min2", "interpolation": "INTER_LINEAR",
    "border": "BORDER_CONSTANT_black", "enhancements": False, "auto_rotate": False}
MIN_FREE_BYTES = 2 * 1024**3
IO_COMPAT_DOCUMENT_SHA = "f4c4d70b7b8d60e9cc626492895174b2e121bba418d28f987d2e34b10c21e231"


def _emit(callback, message, **values):
    if callback:
        callback({"message": message, **values})


def _merge_receipts(*receipts):
    merged = {}
    for receipt in receipts:
        for path, sha in receipt.items():
            require(path not in merged or merged[path] == sha, "E2E_SOURCE_CHANGED_DURING_PREFLIGHT")
            merged[path] = sha
    return merged


def _check_receipt(receipt):
    for path, sha in receipt.items():
        require(mt._sha(path) == sha, "E2E_SOURCE_CHANGED: " + str(path))


def _source_context(selection_dir, cache_dir, progress=None):
    """Strict production gate; exceptions propagate, including PermissionError."""
    _emit(progress, "Weryfikuję selekcję, freeze i rodowód sześciu modeli…")
    selection = load_selection(selection_dir, progress=lambda value: _emit(progress, value))
    cache_dir = Path(cache_dir).resolve()
    output_root = Path(selection["selection_dir"]).parent.parent / "evaluation_runs"
    require(cache_dir.parent == output_root and cache_dir.name.startswith("EVAL396_MT_run_"), "E2E_CACHE_OUTSIDE_EXPERIMENT")
    require(mt._sha(cache_dir / "completion_manifest.json") == MT_COMPLETION_SHA, "E2E_MT_COMPLETION_ANCHOR_CHANGED")
    cache = mt.load_campaign(cache_dir)
    require(cache["status"] == "COMPLETE" and not cache["recovery_required"], "E2E_MT_CACHE_INCOMPLETE")
    require(cache["protocol"]["protocol_sha256"] == MT_PROTOCOL_SHA, "E2E_MT_PROTOCOL_ANCHOR_CHANGED")
    require(cache["preflight"]["selection_dir"] == selection["selection_dir"], "E2E_SELECTION_CACHE_MISMATCH")
    receipt = _merge_receipts(selection["source_files"], cache["source_receipt"], cache["source_files"])
    _check_receipt(receipt)
    require(selection["selection_sha"] == SELECTION_SHA and selection["gt_freeze_sha"] == FREEZE_SHA, "E2E_FROZEN_SOURCE_CHANGED")
    plates = {p["plate_id"]: p for p in selection["plates"]}
    reference = deepcopy(cache["preflight"]["scenes"])
    for scene in reference:
        for plate in scene["plates"]:
            original = plates[plate["plate_id"]]
            require(original["source_file_sha256"] == scene["sha256"] and original["decision"] == plate["text_decision"],
                    "E2E_GT_SOURCE_CHANGED")
            text = original.get("ground_truth_text")
            if original["decision"] == "approved":
                require(isinstance(text, str), "E2E_GT_TEXT_MISSING")
            elif original["decision"] == "excluded_unreadable":
                require(text is None, "E2E_UNREADABLE_GT_TEXT_INVALID")
            else:
                require(False, "E2E_GT_DECISION_INVALID")
            plate["ground_truth_text"] = text
    counts = validate_reference(reference)
    for domain, char_count in CHAR_COUNTS.items():
        require(counts[domain] == {**cache["protocol"]["metric_counts"]["per_domain"][domain], "characters": char_count},
                "E2E_POPULATION_OR_TEXT_DENOMINATOR_CHANGED: " + domain)
    inputs = {}
    for model in cache["models"]:
        inputs[model["label"]] = [{"scene_sha256": row["scene_sha256"], "file": row["file"],
            "width": row["width"], "height": row["height"], "decoded_pixel_sha256": row["decoded_pixel_sha256"],
            "mt_raw_sha256": mt._sha(cache_dir / model["label"] / "raw" / (row["scene_sha256"] + ".json")),
            "predictions": deepcopy(row["predictions"])} for row in model["rows"]]
    source_root = cache["preflight"]["source_scene_root"]
    # An EXIF tag/decoder change may preserve file SHA but change model pixels.
    for index, scene in enumerate(inputs[MT_LABELS[0]]):
        _emit(progress, f"Sprawdzam piksele EXIF {index+1}/{len(reference)}…")
        _decode(scene, source_root)
    _check_receipt(receipt)
    models = deepcopy(selection["models"])
    for model in models:
        model["checkpoint_file"] = str((Path(selection["repo_root"]) / model["checkpoint_file"]).resolve())
    return {"selection_dir": selection["selection_dir"], "cache_dir": str(cache_dir),
        "output_root": str(output_root), "source_root": source_root, "models": models,
        "inputs": inputs, "reference": reference, "counts": counts, "source_files": receipt,
        "selection_sha": selection["selection_sha"], "gt_freeze_sha": selection["gt_freeze_sha"],
        "geometry_approval_sha": cache["geometry_approval"]["contract_sha256"],
        "mt_protocol_sha": cache["protocol"]["protocol_sha256"], "mt_completion_sha": MT_COMPLETION_SHA,
        "evidence_kind": "EVAL396_MEASUREMENT", "protected_roots": [selection["selection_dir"],
            cache["preflight"]["freeze_dir"], source_root, str(cache_dir), str(output_root / "E-MZ-DN-01_EVAL396_MZ_v1")]}


def _decode(scene, source_root):
    path = contained(source_root, scene["file"])
    require(mt._sha(path) == scene["scene_sha256"], "E2E_SCENE_SHA_CHANGED")
    pixels = read_oriented_bgr(path)
    require([pixels.shape[1], pixels.shape[0]] == [scene["width"], scene["height"]]
            and hashlib.sha256(pixels.tobytes()).hexdigest() == scene["decoded_pixel_sha256"], "E2E_EXIF_PIXELS_CHANGED")
    return pixels


def _pairs(values):
    require(isinstance(values, (list, tuple)) and 1 <= len(values) <= 9, "E2E_SELECT_PAIRS")
    rows = []
    for value in values:
        require(isinstance(value, (list, tuple)) and len(value) == 2, "E2E_PAIR_INVALID")
        plate, chars = value
        require(plate in MT_LABELS and chars in MZ_LABELS, "E2E_PAIR_UNKNOWN_MODEL")
        rows.append({"id": plate + "__" + chars, "mt": plate, "mz": chars})
    require(len({r["id"] for r in rows}) == len(rows), "E2E_DUPLICATE_PAIR")
    return rows


def _inference(device, end2end):
    settings = mt._inference_parameters(device, end2end)
    settings.update(imgsz=320, iou=.45)
    return settings


def inspect_mz_checkpoint(row):
    """CPU-only contract inspection, no forward and no predict."""
    from ultralytics import YOLO
    require(mt._sha(row["checkpoint_file"]) == row["checkpoint_sha256"], "E2E_CHECKPOINT_SHA_CHANGED")
    model = YOLO(row["checkpoint_file"], verbose=False)
    try:
        names = {str(k): str(v) for k, v in model.names.items()}
        require(model.task == "detect" and names == ALPHABET, "E2E_MZ_REQUIRES_DETECT_36_CLASSES")
        result = {**deepcopy(row), "task": model.task, "names": names,
            "native_end2end": bool(getattr(model.model, "end2end", False)),
            "loaded_parameter_dtype": str(next(model.model.parameters()).dtype)}
        require(mt._sha(row["checkpoint_file"]) == row["checkpoint_sha256"], "E2E_CHECKPOINT_CHANGED_DURING_INSPECTION")
        return result
    finally:
        del model
        gc.collect()


def _code_receipt():
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    paths = [Path(__file__).resolve().parents[1] / name for name in (
        "ranking/e2e_eval_runner.py", "ranking/e2e_artifacts.py", "ranking/e2e_eval_metrics.py",
        "ranking/e2e_io_compat.py", "ranking/e2e_eval_report.py",
        "ranking/protocols/eval396_e2e_io_compat_20261010/compatibility.json",
        "ranking/eval396.py", "ranking/mt_geometry_review.py", "ranking/mt_metric_policy.py",
        "ranking/mz_whole_plate_evaluation.py", "ranking/mobile_human_review.py",
        "rectification/plate_rectifier.py", "rectification/polygon_validator.py",
        "character_recognition/reading_order.py", "registration_text.py")]
    paths.append(Path(inspect.getfile(DetectionPredictor)))
    return _merge_receipts(mt._code_receipt(), {str(p.resolve()): mt._sha(p) for p in paths})


def _git_state():
    root = Path(__file__).resolve().parents[2]
    kwargs = {"cwd": root, "text": True, "timeout": 10, "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], **kwargs).strip(),
            "status": subprocess.check_output(["git", "status", "--porcelain=v1", "-uall"], **kwargs).splitlines()}


def _disk_check(root):
    path = Path(root)
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    require(free >= MIN_FREE_BYTES, "E2E_DISK_SPACE_REQUIRES_2_GIB")
    return free


def _validate_protocol(protocol):
    require(protocol.get("schema") == SCHEMA and protocol.get("protocol_sha256") == digest(mt._without(protocol, "protocol_sha256")),
            "E2E_PROTOCOL_CHANGED")
    require(protocol["pairs"] == _pairs([(p["mt"], p["mz"]) for p in protocol["pairs"]]), "E2E_PAIRS_CHANGED")
    require({m["label"] for m in protocol["mt_models"]} == {p["mt"] for p in protocol["pairs"]}
            and len(protocol["mt_models"]) == len({m["label"] for m in protocol["mt_models"]}), "E2E_MT_MODEL_SET_CHANGED")
    require(protocol["rectification"] == RECTIFICATION and protocol["scoring"] == SCORING_POLICY
            and protocol["reading_order"] == READING_ORDER_POLICY and protocol["normalization"] == NORMALIZATION_POLICY,
            "E2E_FIXED_POLICY_CHANGED")
    models = protocol["mz_models"]
    require({m["label"] for m in models} == {p["mz"] for p in protocol["pairs"]}
            and len(models) == len({m["label"] for m in models}), "E2E_MZ_MODEL_SET_CHANGED")
    require(all(m["task"] == "detect" and m["names"] == ALPHABET and type(m["native_end2end"]) is bool for m in models),
            "E2E_MZ_CONTRACT_CHANGED")
    require(len({m["native_end2end"] for m in models}) == 1, "E2E_MZ_BACKENDS_DIFFER")
    require(protocol["inference"] == _inference(protocol["inference"]["device"], models[0]["native_end2end"])
            and protocol["inference"]["device"] in {"0", "cpu"}, "E2E_INFERENCE_CHANGED")
    require(protocol["rng"] == {**mt.RNG_PROTOCOL, "per_scene_seed": "42 + scene_index*300 + prediction_index"}, "E2E_RNG_CHANGED")


def prepare_campaign(selection_dir, cache_dir, *, pairs, device="0", progress=None):
    """Build an unsigned proposal. No directories, approvals or inference."""
    selected_pairs = _pairs(pairs)
    from ..rectification import plate_rectifier
    require(plate_rectifier.CV2_AVAILABLE and plate_rectifier.cv2 is not None, "E2E_OPENCV_REQUIRED")
    context = _source_context(selection_dir, cache_dir, progress)
    rows = {r["label"]: r for r in context["models"]}
    models = [inspect_mz_checkpoint(rows[label]) for label in MZ_LABELS if any(p["mz"] == label for p in selected_pairs)]
    require(len({m["native_end2end"] for m in models}) == 1, "E2E_MZ_BACKENDS_DIFFER")
    inputs = {label: context["inputs"][label] for label in MT_LABELS if any(p["mt"] == label for p in selected_pairs)}
    protocol = {"schema": SCHEMA, "evidence_kind": context["evidence_kind"], "pairs": selected_pairs,
        "selection_dir": context["selection_dir"], "cache_dir": context["cache_dir"], "output_root": context["output_root"],
        "source_root": context["source_root"], "protected_roots": context["protected_roots"],
        "selection_sha256": context["selection_sha"], "gt_freeze_sha256": context["gt_freeze_sha"],
        "mt_protocol_sha256": context["mt_protocol_sha"], "mt_completion_sha256": context["mt_completion_sha"],
        "geometry_approval_sha256": context["geometry_approval_sha"], "counts": context["counts"],
        "mt_models": [deepcopy(rows[label]) for label in inputs], "mz_models": models,
        "inference": _inference(device, models[0]["native_end2end"]), "runtime": mt._runtime(device),
        "rng": {**mt.RNG_PROTOCOL, "per_scene_seed": "42 + scene_index*300 + prediction_index"},
        "rectification": deepcopy(RECTIFICATION), "reading_order": READING_ORDER_POLICY,
        "normalization": NORMALIZATION_POLICY, "scoring": deepcopy(SCORING_POLICY),
        "input_sha256": digest(inputs), "reference_sha256": digest(context["reference"]),
        "source_receipt_sha256": digest(context["source_files"]), "code_sha256": _code_receipt(),
        "git": _git_state(), "mt_inference": "reused_not_executed", "full_pipeline_latency": "NOT_MEASURED"}
    protocol["protocol_sha256"] = digest(protocol)
    _validate_protocol(protocol)
    free = _disk_check(context["output_root"])
    _check_receipt(context["source_files"])
    _emit(progress, "PASS preflight — propozycja oczekuje na podpis operatora. Pomiar nie jest uruchomiony.")
    return {"protocol": protocol, "inputs": inputs, "source_receipt": context["source_files"], "free_disk_bytes": free}


def _verify_prepared(prepared, *, runtime=True, resume_dir=None):
    p = prepared["protocol"]
    _validate_protocol(p)
    context = _source_context(p["selection_dir"], p["cache_dir"])
    for key, value in (("selection_sha256", context["selection_sha"]), ("gt_freeze_sha256", context["gt_freeze_sha"]),
        ("mt_protocol_sha256", context["mt_protocol_sha"]), ("mt_completion_sha256", context["mt_completion_sha"]),
        ("geometry_approval_sha256", context["geometry_approval_sha"]), ("counts", context["counts"]),
        ("reference_sha256", digest(context["reference"])), ("output_root", context["output_root"]),
        ("source_root", context["source_root"]), ("protected_roots", context["protected_roots"]), ("evidence_kind", context["evidence_kind"])):
        require(p[key] == value, "E2E_SOURCE_BINDING_CHANGED: " + key)
    inputs = {label: context["inputs"][label] for label in MT_LABELS if any(pair["mt"] == label for pair in p["pairs"])}
    require(prepared["inputs"] == inputs and digest(inputs) == p["input_sha256"], "E2E_INPUT_CATALOG_CHANGED")
    require(prepared["source_receipt"] == context["source_files"]
            and digest(context["source_files"]) == p["source_receipt_sha256"], "E2E_SOURCE_RECEIPT_CHANGED")
    sources = {r["label"]: r for r in context["models"]}
    for model in p["mt_models"] + p["mz_models"]:
        require(all(model.get(k) == v for k, v in sources[model["label"]].items()), "E2E_MODEL_LINEAGE_CHANGED")
    if runtime:
        context["io_compatibility"] = io_compat.verify_runtime_code(
            p, _code_receipt(), resume_dir, IO_COMPAT_DOCUMENT_SHA)
        require(mt._runtime(p["inference"]["device"]) == p["runtime"], "E2E_RUNTIME_CHANGED")
    return context


def verify_approval(protocol, approval):
    require(approval.get("schema") == APPROVAL_SCHEMA and approval.get("decision") == "APPROVED"
            and bool(str(approval.get("operator_signature", "")).strip()) and approval.get("acknowledged") is True,
            "E2E_OPERATOR_SIGNATURE_REQUIRED")
    require(approval.get("protocol_sha256") == protocol["protocol_sha256"]
            and approval.get("contract_sha256") == digest(mt._without(approval, "contract_sha256")), "E2E_APPROVAL_CHANGED")


def save_approval(prepared, *, operator, acknowledged):
    require(isinstance(operator, str) and bool(operator.strip()) and acknowledged is True, "E2E_OPERATOR_SIGNATURE_REQUIRED")
    _verify_prepared(prepared)
    p = prepared["protocol"]
    approval = {"schema": APPROVAL_SCHEMA, "decision": "APPROVED", "acknowledged": True,
        "operator_signature": operator.strip(), "signed_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": p["protocol_sha256"], "signature_kind": "local_operator_attestation_with_sha256_integrity",
        "scope": "E2E_quality_cached_MT_explicit_Start_required", "measurement_status": "NOT_RUN"}
    approval["contract_sha256"] = digest(approval)
    path = Path(p["output_root"]) / ("EVAL396_E2E_protocol_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    mt._write_json(path / "protocol.json", p, exclusive=True)
    mt._write_json(path / "approval.json", approval, exclusive=True)
    mt._write_json(path / "status.json", {"status": "APPROVED_NOT_RUN", "measurement_status": "NOT_RUN"}, exclusive=True)
    return {"directory": str(path), "approval": approval}


def _effective_matches(actual, model, protocol):
    require(actual.get("backend_names") == ALPHABET and actual.get("backend_fp16") is False
            and actual.get("backend_end2end") == model["native_end2end"]
            and actual.get("checked_imgsz") == [320, 320] and actual.get("device") == protocol["runtime"]["device"],
            "E2E_EFFECTIVE_BACKEND_CHANGED")
    for k, v in protocol["inference"].items():
        if k not in {"imgsz", "device"}:
            require(actual.get(k) == v, "E2E_EFFECTIVE_PARAMETER_CHANGED: " + k)
    require(actual.get("determinism") == {"deterministic_algorithms": True, "warn_only": True,
        "cudnn_benchmark": False, "cudnn_deterministic": True, "matmul_tf32": False, "cudnn_tf32": False,
        "cublas_workspace_config": ":4096:8"}, "E2E_DETERMINISM_CHANGED")


class MZBackend:
    def __init__(self, model, protocol):
        from ultralytics import YOLO
        require(mt._sha(model["checkpoint_file"]) == model["checkpoint_sha256"], "E2E_CHECKPOINT_SHA_CHANGED")
        self.model = YOLO(model["checkpoint_file"], verbose=False)
        require(self.model.task == "detect" and {str(k): v for k, v in self.model.names.items()} == ALPHABET,
                "E2E_MZ_REQUIRES_DETECT_36_CLASSES")
        self.model_row, self.protocol = model, protocol

    def predict(self, crop):
        import torch
        with torch.inference_mode():
            results = self.model.predict(source=crop, stream=False, **self.protocol["inference"])
        require(len(results) == 1 and tuple(results[0].orig_shape) == crop.shape[:2], "E2E_MZ_COORDINATE_SPACE_CHANGED")
        predictor = self.model.predictor
        actual = {key: mt._portable(getattr(predictor.args, key, None)) for key in self.protocol["inference"]}
        actual.update(device=str(predictor.model.device), checked_imgsz=list(predictor.imgsz),
            backend_fp16=bool(predictor.model.fp16), backend_end2end=bool(predictor.model.end2end),
            backend_names={str(k): str(v) for k, v in predictor.model.names.items()}, determinism=mt._actual_determinism())
        _effective_matches(actual, self.model_row, self.protocol)
        boxes, chars = results[0].boxes, []
        if boxes is not None:
            xyxy, classes, confidences = boxes.xyxy.detach().cpu().tolist(), boxes.cls.detach().cpu().tolist(), boxes.conf.detach().cpu().tolist()
            require(len(xyxy) == len(classes) == len(confidences), "E2E_MZ_ARRAY_LENGTH_CHANGED")
            for index, (box, cls, conf) in enumerate(zip(xyxy, classes, confidences)):
                require(int(cls) == cls and str(int(cls)) in ALPHABET, "E2E_MZ_CLASS_INVALID")
                chars.append({"index": index, "class_id": int(cls), "character": ALPHABET[str(int(cls))],
                              "bbox": box, "confidence": conf})
        return {"characters": chars, "effective_inference": actual, "speed_ms": mt._portable(dict(results[0].speed))}

    def close(self):
        del self.model
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def predict_detection(pixels, prediction, backend):
    """GT-free boundary: pixels, one raw MT detection, a MZ-only backend."""
    require(set(prediction) == {"index", "class_id", "bbox", "confidence", "keypoints"}, "E2E_PREDICTION_INPUT_CONTRACT")
    crop, geometry = PlateRectifier.rectify_predicted_quad(pixels, prediction["keypoints"])
    result = {"mt": deepcopy(prediction), "rectification": geometry, "characters": [],
              "effective_inference": None, "speed_ms": None, "mz_elapsed_ms": None}
    if crop is not None:
        started = time.monotonic()
        result.update(backend.predict(crop))
        result["mz_elapsed_ms"] = (time.monotonic() - started) * 1000
    reading = read_plate_records(result["characters"])
    failure = geometry["status"] if crop is None else "MZ_NO_CHARACTERS" if reading["no_read"] else ""
    result.update(reading=reading, status="NO_READ" if reading["no_read"] else "READ", failure=failure)
    return result


def _validate_detection(row, scene, prediction, pair, p, model):
    storage.check_record(row)
    mt_model = next(m for m in p["mt_models"] if m["label"] == pair["mt"])
    require(row.get("schema") == "alpr.e2e.prediction.v1" and row["protocol_sha256"] == p["protocol_sha256"]
            and row["pair_id"] == pair["id"] and row["scene_sha256"] == scene["scene_sha256"]
            and row["decoded_pixel_sha256"] == scene["decoded_pixel_sha256"] and row["mt_raw_sha256"] == scene["mt_raw_sha256"]
            and row["mz_checkpoint_sha256"] == model["checkpoint_sha256"] and row["mt"] == prediction
            and row["mt_checkpoint_sha256"] == mt_model["checkpoint_sha256"]
            and row["mt_model"] == pair["mt"] and row["mz_model"] == pair["mz"], "E2E_PREDICTION_BINDING_CHANGED")
    require(not {"domain", "ground_truth", "plate_id", "gt_bbox", "gt_quad"} & set(row), "E2E_GT_IN_PREDICTION")
    geom = row["rectification"]
    plan = PlateRectifier.predicted_quad_geometry(prediction["keypoints"], (scene["width"], scene["height"]))
    require(all(geom.get(k) == v for k, v in plan.items() if k not in {"pixel_sha256", "status", "reason"}), "E2E_HOMOGRAPHY_OR_QUAD_CHANGED")
    if plan["status"] != "RECTIFIED":
        require(geom == plan, "E2E_INVALID_QUAD_REPAIRED")
    else:
        require(geom["status"] == "RECTIFIED" or (geom["status"] == "RECTIFICATION_FAILED"
                and (str(geom.get("reason", "")).startswith("HOMOGRAPHY_ERROR:") or geom.get("reason") == "INVALID_WARP_RESULT")
                and geom["pixel_sha256"] is None), "E2E_RECTIFICATION_STATUS_CHANGED")
    chars = row["characters"]
    require(isinstance(chars, list) and len(chars) <= 300, "E2E_MZ_CHARACTER_LIST_INVALID")
    require([c["index"] for c in chars] == list(range(len(chars))), "E2E_MZ_INDICES_CHANGED")
    for char in chars:
        box, conf, cls = char["bbox"], char["confidence"], char["class_id"]
        require(type(cls) is int and char["character"] == ALPHABET.get(str(cls)), "E2E_MZ_CLASS_INVALID")
        require(isinstance(conf, (int, float)) and not isinstance(conf, bool) and math.isfinite(conf) and .25 < conf <= 1,
                "E2E_MZ_CONFIDENCE_INVALID")
        require(len(box) == 4 and all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) for x in box)
                and box[2] > box[0] and box[3] > box[1], "E2E_MZ_BBOX_INVALID")
        require(geom["size"] is not None and 0 <= box[0] < box[2] <= geom["size"][0]
                and 0 <= box[1] < box[3] <= geom["size"][1], "E2E_MZ_BBOX_OUTSIDE_CROP")
    require(row["reading"] == read_plate_records(chars), "E2E_READING_CHANGED")
    if geom["status"] == "RECTIFIED":
        require(isinstance(geom["size"], list) and len(geom["size"]) == 2 and min(geom["size"]) >= 2
                and max(geom["size"]) == 320 and isinstance(geom["pixel_sha256"], str) and len(geom["pixel_sha256"]) == 64,
                "E2E_CROP_CONTRACT_CHANGED")
        require(geom["homography"] is not None and len(geom["homography"]) == 3, "E2E_HOMOGRAPHY_MISSING")
        _effective_matches(row["effective_inference"], model, p)
        expected_failure = "MZ_NO_CHARACTERS" if not chars else ""
        require(isinstance(row["mz_elapsed_ms"], (int, float)) and math.isfinite(row["mz_elapsed_ms"]) and row["mz_elapsed_ms"] >= 0,
                "E2E_MZ_TIMING_INVALID")
    else:
        require(not chars and row["effective_inference"] is None and row["mz_elapsed_ms"] is None, "E2E_FAILED_RECTIFICATION_INFERRED")
        expected_failure = geom["status"]
    require(row["failure"] == expected_failure and row["status"] == ("READ" if chars else "NO_READ"), "E2E_PREDICTION_STATUS_CHANGED")


def _validate_store(prepared, inspected):
    p, inputs = prepared["protocol"], prepared["inputs"]
    models = {m["label"]: m for m in p["mz_models"]}
    rows, assembled = inspected["rows"], {pair["id"]: [] for pair in p["pairs"]}
    for pair in p["pairs"]:
        for scene_index, scene in enumerate(inputs[pair["mt"]]):
            refs, detections = {}, []
            for prediction in scene["predictions"]:
                name = storage.detection_path(pair["id"], scene["scene_sha256"], prediction["index"])
                if name in rows:
                    row = rows[name]
                    _validate_detection(row, scene, prediction, pair, p, models[pair["mz"]])
                    require(row["seed"] == 42 + scene_index*300 + prediction["index"], "E2E_SEED_CHANGED")
                    detections.append(row)
                    # Pending records are not committed scene dependencies yet.
                    if name in inspected["ledger"]["records"]:
                        refs[name] = inspected["ledger"]["records"][name]
            name = storage.scene_path(pair["id"], scene["scene_sha256"])
            if name in rows:
                row = rows[name]
                require(len(refs) == len(scene["predictions"]) and row.get("schema") == "alpr.e2e.scene.v1"
                        and row["protocol_sha256"] == p["protocol_sha256"] and row["pair_id"] == pair["id"]
                        and row["scene_sha256"] == scene["scene_sha256"] and row["detections"] == refs
                        and row["width"] == scene["width"] and row["height"] == scene["height"]
                        and row["status"] == "COMPLETE", "E2E_SCENE_LEDGER_CHANGED")
                assembled[pair["id"]].append({**row, "detections": detections})
    return assembled


def _new_campaign(prepared, approval):
    p = prepared["protocol"]
    root = Path(p["output_root"])
    name = "EVAL396_E2E_run_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    stage, destination = root / (name + ".partial"), root / name
    stage.mkdir()
    provenance = {"schema": "alpr.e2e.provenance.v1", "evidence_kind": p["evidence_kind"],
        "git": p["git"], "code_sha256": p["code_sha256"], "source_receipt": prepared["source_receipt"],
        "mt_cache": p["cache_dir"], "mt_inference_performed": False, "full_pipeline_latency": "NOT_MEASURED"}
    files = {}
    for filename, value in (("protocol.json", p), ("approval.json", approval), ("inputs.json", prepared["inputs"]), ("provenance.json", provenance)):
        files[filename] = mt._write_json(stage / filename, value, exclusive=True)
    mt._write_json(stage / "manifest.json", {"schema": CAMPAIGN_SCHEMA, "protocol_sha256": p["protocol_sha256"],
        "created_at": datetime.now(timezone.utc).isoformat(), "files": files}, exclusive=True)
    mt._write_json(stage / "progress.json", {"schema": "alpr.e2e.progress.v1", "records": {}}, exclusive=True)
    mt._write_json(stage / "status.json", {"status": "PARTIAL", "reason": "CREATED_NOT_STARTED"}, exclusive=True)
    require(not destination.exists(), "E2E_DESTINATION_EXISTS")
    stage.rename(destination)
    return destination


def _read_snapshot(run_dir):
    path = Path(run_dir).resolve()
    manifest = storage.read_json(path / "manifest.json")
    require(manifest.get("schema") == CAMPAIGN_SCHEMA and set(manifest["files"]) == {"protocol.json", "approval.json", "inputs.json", "provenance.json"},
            "E2E_MANIFEST_INVALID")
    payload = {}
    for name, sha in manifest["files"].items():
        file = contained(path, name)
        require(mt._sha(file) == sha, "E2E_SNAPSHOT_CHANGED: " + name)
        payload[name] = storage.read_json(file)
    p = payload["protocol.json"]
    _validate_protocol(p)
    require(manifest["protocol_sha256"] == p["protocol_sha256"] and path.parent == Path(p["output_root"])
            and path.name.startswith("EVAL396_E2E_run_") and not path.name.endswith(".partial"), "E2E_RUN_PATH_OR_PROTOCOL_CHANGED")
    verify_approval(p, payload["approval.json"])
    require(digest(payload["inputs.json"]) == p["input_sha256"]
            and digest(payload["provenance.json"]["source_receipt"]) == p["source_receipt_sha256"]
            and payload["provenance.json"]["code_sha256"] == p["code_sha256"], "E2E_PROVENANCE_CHANGED")
    prepared = {"protocol": p, "inputs": payload["inputs.json"], "source_receipt": payload["provenance.json"]["source_receipt"]}
    return prepared, payload["approval.json"], manifest


def _completion_files(path, allowed):
    names = allowed | {"protocol.json", "approval.json", "inputs.json", "provenance.json", "manifest.json", "progress.json",
                       "prediction_manifest.json", "scoring_reference.json", "scoring.json", "summary.json"}
    if (path / io_compat.APPLIED_FILE).exists():
        names.add(io_compat.APPLIED_FILE)
    return {name: mt._sha(contained(path, name)) for name in sorted(names)}


def load_campaign(run_dir):
    """Validate persisted results without inference, recovery or writes."""
    path = Path(run_dir).resolve()
    prepared, approval, manifest = _read_snapshot(path)
    io_application = io_compat.read_application(path, prepared["protocol"], IO_COMPAT_DOCUMENT_SHA)
    context = _verify_prepared(prepared, runtime=False)
    p = prepared["protocol"]
    allowed = storage.expected_paths(prepared["inputs"], p["pairs"])
    inspected = storage.inspect_ledger(path, allowed)
    assembled = _validate_store(prepared, inspected)
    status = storage.read_json(path / "status.json")
    require(status.get("status") in {"PARTIAL", "RUNNING", "COMPLETE"}, "E2E_STATUS_INVALID")
    seal_file = path / "prediction_manifest.json"
    scoring = None
    if seal_file.exists():
        seal = storage.read_json(seal_file)
        require(not inspected["pending"] and set(inspected["ledger"]["records"]) == allowed
                and seal == {"schema": "alpr.e2e.prediction_manifest.v1", "protocol_sha256": p["protocol_sha256"],
                             "files": inspected["ledger"]["records"]}, "E2E_PREDICTION_SEAL_CHANGED")
        if (path / "scoring.json").exists():
            require(storage.read_json(path / "scoring_reference.json") == context["reference"], "E2E_SCORING_REFERENCE_CHANGED")
            scoring = storage.read_json(path / "scoring.json")
            expected = {pair["id"]: score_pair(assembled[pair["id"]], context["reference"]) for pair in p["pairs"]}
            require(scoring == expected, "E2E_SCORING_DIFFERS_FROM_RAW")
            if (path / "summary.json").exists():
                require(storage.read_json(path / "summary.json") == {key: value["domains"] for key, value in scoring.items()},
                        "E2E_SUMMARY_CHANGED")
    else:
        require(not any((path / name).exists() for name in ("scoring_reference.json", "scoring.json", "summary.json")),
                "E2E_GT_SCORING_BEFORE_RAW_SEAL")
    if status["status"] == "COMPLETE":
        require(scoring is not None and all(s["status"] == "COMPLETE" for s in scoring.values()), "E2E_CAMPAIGN_INCOMPLETE")
        expected = {"protocol_sha256": p["protocol_sha256"], "files": _completion_files(path, allowed)}
        require(storage.read_json(path / "completion_manifest.json") == expected, "E2E_COMPLETION_CHANGED")
    return {"run_dir": str(path), "status": status["status"], "status_detail": status,
        "prepared": prepared, "protocol": p, "approval": approval, "created_at": manifest["created_at"],
        "ledger": inspected, "pairs": assembled, "scoring": scoring, "io_compatibility": io_application,
        "recovery_required": inspected["recovery_required"], "reference": context["reference"],
        "completed_scenes": sum(len(rows) for rows in assembled.values()),
        "total_scenes": sum(len(prepared["inputs"][pair["mt"]]) for pair in p["pairs"])}


def list_campaigns(output_root):
    """Cheap history discovery. Full validation is mandatory on opening/resuming."""
    rows = []
    for path in sorted(Path(output_root).glob("EVAL396_E2E_run_*"), reverse=True):
        if not path.is_dir() or path.name.endswith(".partial"):
            continue
        try:
            p, _, manifest = _read_snapshot(path)
            status = storage.read_json(path / "status.json")
            rows.append({"run_dir": str(path), "status": "DO WERYFIKACJI: " + status["status"],
                         "created_at": manifest["created_at"], "pairs": len(p["protocol"]["pairs"])})
        except Exception as exc:
            rows.append({"run_dir": str(path), "status": "BŁĄD: " + str(exc), "created_at": "", "pairs": 0})
    return rows


def run_campaign(prepared, approval, *, progress=None, cancelled=None, resume_dir=None):
    """Only explicit Start/Resume. Never called by opening UI, preflight or export."""
    verify_approval(prepared["protocol"], approval)
    _emit(progress, "Weryfikuję podpis, źródła, kod i runtime przed pomiarem…")
    context = _verify_prepared(prepared, resume_dir=resume_dir)
    p, inputs = prepared["protocol"], prepared["inputs"]
    _disk_check(p["output_root"])
    cancelled = cancelled or (lambda: False)
    allowed = storage.expected_paths(inputs, p["pairs"])
    models = {m["label"]: m for m in p["mz_models"]}
    total = sum(len(inputs[pair["mt"]]) for pair in p["pairs"])
    with mt._execution_lock(p["output_root"]), mt._deterministic_runtime(p):
        if resume_dir:
            existing = load_campaign(resume_dir)
            require(existing["protocol"] == p and existing["approval"] == approval, "E2E_RESUME_PROTOCOL_CHANGED")
            if existing["status"] == "COMPLETE":
                return existing
            path = Path(existing["run_dir"])
            io_compat.record_application(path, context.get("io_compatibility"))
            ledger = storage.recover(path, existing["ledger"])
        else:
            path = _new_campaign(prepared, approval)
            ledger = storage.read_json(path / "progress.json")
        mt._write_json(path / "status.json", {"status": "RUNNING", "measurement_status": "RUNNING"})
        _emit(progress, "Rozpoczęto zapis E2E.", run_dir=str(path))
        try:
            for pair in p["pairs"]:
                backend = None
                try:
                    for scene_index, scene in enumerate(inputs[pair["mt"]]):
                        scene_name = storage.scene_path(pair["id"], scene["scene_sha256"])
                        if scene_name in ledger["records"]:
                            continue
                        if cancelled():
                            raise InterruptedError("CANCELLED")
                        pixels = None
                        refs = {}
                        for prediction in scene["predictions"]:
                            name = storage.detection_path(pair["id"], scene["scene_sha256"], prediction["index"])
                            if name not in ledger["records"]:
                                if cancelled():
                                    raise InterruptedError("CANCELLED")
                                _disk_check(p["output_root"])
                                if pixels is None:
                                    pixels = _decode(scene, p["source_root"])
                                if backend is None:
                                    # No GT, domain, selection manifest or reference is passed to the backend.
                                    backend = MZBackend(models[pair["mz"]], {"inference": p["inference"], "runtime": p["runtime"]})
                                mt._seed_scene(scene_index*300 + prediction["index"])
                                prediction_result = predict_detection(pixels, prediction, backend)
                                require(mt._sha(contained(p["source_root"], scene["file"])) == scene["scene_sha256"],
                                        "E2E_IMAGE_CHANGED_DURING_PREDICTION")
                                row = storage.signed_record({"schema": "alpr.e2e.prediction.v1", "protocol_sha256": p["protocol_sha256"],
                                    "pair_id": pair["id"], "scene_sha256": scene["scene_sha256"],
                                    "mt_model": pair["mt"], "mz_model": pair["mz"],
                                    "mt_checkpoint_sha256": next(m["checkpoint_sha256"] for m in p["mt_models"] if m["label"] == pair["mt"]),
                                    "decoded_pixel_sha256": scene["decoded_pixel_sha256"], "mt_raw_sha256": scene["mt_raw_sha256"],
                                    "mz_checkpoint_sha256": models[pair["mz"]]["checkpoint_sha256"],
                                    "seed": 42 + scene_index*300 + prediction["index"], **prediction_result})
                                _validate_detection(row, scene, prediction, pair, p, models[pair["mz"]])
                                storage.publish(path, ledger, name, row)
                            refs[name] = ledger["records"][name]
                            _emit(progress, f"{pair['id']}: scena {scene_index+1}/{len(inputs[pair['mt']])}, detekcja {prediction['index']+1}",
                                  detections=sum('/detections/' in name for name in ledger["records"]), run_dir=str(path))
                        row = storage.signed_record({"schema": "alpr.e2e.scene.v1", "protocol_sha256": p["protocol_sha256"],
                            "pair_id": pair["id"], "scene_sha256": scene["scene_sha256"], "width": scene["width"], "height": scene["height"],
                            "status": "COMPLETE", "detections": refs})
                        storage.publish(path, ledger, scene_name, row)
                        completed = sum('/scenes/' in name for name in ledger["records"])
                        _emit(progress, f"{pair['id']}: zapisano scenę {scene_index+1}/{len(inputs[pair['mt']])}",
                              completed=completed, total=total, run_dir=str(path))
                        del pixels
                finally:
                    if backend is not None:
                        backend.close()
            if cancelled():
                raise InterruptedError("CANCELLED")
            storage.seal_predictions(path, p["protocol_sha256"], ledger, allowed)
            # Load GT for the scorer only after all pairs' raw predictions are sealed.
            _emit(progress, "Surowe predykcje zamknięte. Weryfikuję źródła i wykonuję osobny scoring GT…")
            context = _verify_prepared(prepared, resume_dir=path)
            inspected = storage.inspect_ledger(path, allowed)
            assembled = _validate_store(prepared, inspected)
            scoring = {pair["id"]: score_pair(assembled[pair["id"]], context["reference"]) for pair in p["pairs"]}
            require(all(value["status"] == "COMPLETE" for value in scoring.values()), "E2E_SCORING_INCOMPLETE")
            storage.immutable_json(path / "scoring_reference.json", context["reference"])
            storage.immutable_json(path / "scoring.json", scoring)
            storage.immutable_json(path / "summary.json", {key: value["domains"] for key, value in scoring.items()})
            _check_receipt(prepared["source_receipt"])
            storage.immutable_json(path / "completion_manifest.json", {"protocol_sha256": p["protocol_sha256"], "files": _completion_files(path, allowed)})
            mt._write_json(path / "status.json", {"status": "COMPLETE", "measurement_status": "COMPLETE", "completed_scenes": total})
        except Exception as exc:
            mt._write_json(path / "status.json", {"status": "PARTIAL", "measurement_status": "PARTIAL",
                "reason": "CANCELLED" if isinstance(exc, InterruptedError) else "ERROR", "error": str(exc)})
            _emit(progress, "PARTIAL: " + str(exc), run_dir=str(path))
            if not isinstance(exc, InterruptedError):
                raise RuntimeError(f"E2E_PARTIAL: {path}: {exc}") from exc
    return load_campaign(path)
