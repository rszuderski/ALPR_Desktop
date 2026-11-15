"""Native Z4 MT experiment service. No CLI, training, GT-guided inference or MZ calls."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import threading
import time
import uuid

import numpy as np

from ..image_orientation import read_oriented_bgr
from .eval396 import canonical, contained, digest, require, EvidenceReader
from .mt_geometry_review import preflight_mt_geometry, require_current_approval, load_operator_decision
from .mt_eval_metrics import aggregate, METRIC_PROTOCOL
from . import mt_geometry_review as geometry_gate

PROTOCOL_SCHEMA = "alpr.mt_eval_protocol.v1"
CAMPAIGN_SCHEMA = "alpr.mt_eval_campaign.v1"
ROW_SCHEMA = "alpr.mt_prediction_scene.v1"
MODEL_LABELS = ("MT-n", "MT-DAY", "MT-NIGHT")
_RUN_LOCK = threading.Lock()


def _inference_parameters(device, end2end):
    return {"imgsz": 512, "conf": .25, "iou": .7, "max_det": 300, "batch": 1,
        "rect": False, "half": False, "augment": False, "device": str(device),
        "end2end": end2end, "agnostic_nms": False, "classes": None,
        "single_cls": False, "compile": False, "dnn": False, "embed": None,
        "visualize": False, "show": False, "save": False, "save_txt": False,
        "save_crop": False, "save_frames": False, "verbose": False, "data": None,
        "seed": 42, "deterministic": True}


RNG_PROTOCOL = {"seed": 42, "per_scene_seed": "42 + scene_index", "deterministic_algorithms": "enabled_warn_only",
                "cudnn_benchmark": False, "cudnn_deterministic": True, "tf32": False, "cublas_workspace_config": ":4096:8"}


def _validate_protocol_parameters(protocol):
    models = protocol["models"]
    require([m["label"] for m in models] == list(MODEL_LABELS), "MT_MODEL_SET_CHANGED")
    require(all(m.get("task") == "pose" and m.get("kpt_shape") in ([4, 2], [4, 3])
                and m.get("names") == {"0": "plate"} and isinstance(m.get("native_end2end"), bool)
                for m in models), "MT_MODEL_HEAD_CHANGED")
    require(len({tuple(m["kpt_shape"]) for m in models}) == 1
            and len({m["native_end2end"] for m in models}) == 1, "MODELS_REQUIRE_DIFFERENT_INFERENCE_PROTOCOLS")
    device = protocol["inference"]["device"]
    require(device in {"0", "cpu"}, "MT_DEVICE_MUST_BE_0_OR_CPU")
    require(protocol["inference"] == _inference_parameters(device, models[0]["native_end2end"])
            and protocol["rng"] == RNG_PROTOCOL and protocol["metrics"] == METRIC_PROTOCOL,
            "MT_FIXED_PARAMETERS_CHANGED")


def _actual_determinism():
    import torch
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def _validate_effective(actual, model, protocol):
    expected = protocol["inference"]
    require(not actual["backend_fp16"] and actual["backend_end2end"] == expected["end2end"]
            and actual["backend_kpt_shape"] == model["kpt_shape"]
            and actual["backend_names"] == model["names"]
            and actual["checked_imgsz"] == [expected["imgsz"], expected["imgsz"]]
            and actual["device"] == protocol["runtime"]["device"], "MT_EFFECTIVE_BACKEND_CHANGED")
    for key, value in expected.items():
        if key not in {"device", "imgsz"}:
            require(actual[key] == value, "MT_EFFECTIVE_PARAMETER_CHANGED: " + key)
    require(actual["determinism"] == {"deterministic_algorithms": True, "warn_only": True,
            "cudnn_benchmark": False, "cudnn_deterministic": True, "matmul_tf32": False,
            "cudnn_tf32": False, "cublas_workspace_config": protocol["rng"]["cublas_workspace_config"]},
            "MT_EFFECTIVE_DETERMINISM_CHANGED")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _without(value, key):
    return {k: v for k, v in value.items() if k != key}


def _portable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": repr(value)}
    if isinstance(value, dict):
        return {str(k): _portable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_portable(v) for v in value]
    return value


def _write_json(path, payload, *, exclusive=False):
    """Flush before publication; exclusive targets are never partly visible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(payload) + b"\n"
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if not exclusive:
            for attempt in range(6):
                try:
                    os.replace(temporary, path)
                    break
                except OSError as exc:
                    # Windows readers/filters can briefly deny delete/rename.
                    # Retry only publication of the same closed, fsynced file.
                    if os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 5:
                        raise
                    time.sleep(0.025 * (2 ** attempt))
        elif os.name == "nt":
            os.rename(temporary, path)  # Windows rename fails if target exists.
        else:
            os.link(temporary, path)  # Atomic no-replace publication on POSIX.
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(data).hexdigest()


def _emit(progress, message, stage="prepare", **extra):
    if progress:
        progress({"stage": stage, "message": message, "model": None, "scene": 0,
                  "total_scenes": 396, "completed": 0, "total": 1188, **extra})


def _runtime(device):
    import torch
    require(str(device) in {"0", "cpu"}, "MT_DEVICE_MUST_BE_0_OR_CPU")
    gpu = str(device) == "0"
    require(not gpu or torch.cuda.is_available(), "CUDA_NOT_AVAILABLE")
    result = {"python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "ultralytics": importlib.metadata.version("ultralytics"),
        "numpy": np.__version__, "opencv": importlib.metadata.version("opencv-python"),
        "pillow": importlib.metadata.version("Pillow"), "device": "cuda:0" if gpu else "cpu"}
    if gpu:
        props = torch.cuda.get_device_properties(0)
        result.update(device_name=props.name, device_memory=props.total_memory,
                      compute_capability=[props.major, props.minor])
        try:
            result["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "--id=0"],
                text=True, timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).strip()
        except (OSError, subprocess.SubprocessError):
            result["driver"] = "UNKNOWN"
    return result


def _code_receipt():
    from ultralytics.engine.predictor import BasePredictor
    from ultralytics.engine.results import Results
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.models.yolo.pose.predict import PosePredictor
    from ultralytics.utils import nms, ops
    root = Path(__file__).resolve().parents[2]
    paths = [root / "auto_annotation_tool" / p for p in (
        "ranking/mt_eval_runner.py", "ranking/mt_eval_metrics.py", "image_orientation.py",
        "rectification/quad_geometry.py")]
    paths += [Path(inspect.getfile(item)) for item in (BasePredictor, AutoBackend, PosePredictor, Results, nms, ops)]
    return {str(path.resolve()): _sha(path) for path in paths}


def inspect_checkpoint(model_row):
    """Load trusted weights on CPU to inspect task/head; never calls forward/predict."""
    from ultralytics import YOLO
    path = Path(model_row["checkpoint_file"])
    require(_sha(path) == model_row["checkpoint_sha256"], "CHECKPOINT_SHA_CHANGED")
    model = YOLO(str(path), verbose=False)
    try:
        shape = list(getattr(model.model, "kpt_shape", getattr(model.model.model[-1], "kpt_shape", [])))
        names = {str(k): str(v) for k, v in model.model.names.items()}
        require(model.task == "pose" and shape in ([4, 2], [4, 3]), "MT_CHECKPOINT_POSE_CONTRACT_INVALID")
        require(names == {"0": "plate"}, "MT_CHECKPOINT_CLASS_CONTRACT_INVALID")
        info = {**deepcopy(model_row), "task": model.task, "kpt_shape": shape, "names": names,
                "native_end2end": bool(getattr(model.model, "end2end", False)),
                "stored_parameter_dtype": str(next(model.model.parameters()).dtype)}
        require(_sha(path) == model_row["checkpoint_sha256"], "CHECKPOINT_CHANGED_DURING_INSPECTION")
        return info
    finally:
        del model
        gc.collect()


def prepare_campaign(selection_dir, *, approval_dir=None, device="0", progress=None):
    """Verify real approval and sources, inspect CPU weights, freeze parameters. No inference."""
    _emit(progress, "Sprawdzam aktualny kontrakt geometrii i źródła…")
    preflight = preflight_mt_geometry(selection_dir, progress=lambda s: _emit(progress, s))
    if approval_dir is None:
        saved = load_operator_decision(preflight, Path(preflight["selection_dir"]).parent.parent / "evaluation_runs")
        require(saved is not None, "EXIF_OPERATOR_APPROVAL_REQUIRED")
        approval_dir, decision = saved
    else:
        approval_dir = Path(approval_dir).resolve()
        decision = json.loads((approval_dir / "geometry_policy.json").read_text(encoding="utf-8"))
    require_current_approval(preflight, decision)
    approval_dir = Path(approval_dir).resolve()
    approval_files = {str(approval_dir / name): _sha(approval_dir / name) for name in
                      ("geometry_policy.json", "preflight.json", "metric_eligibility_policy.json", "status.json")}
    source_receipt = {**preflight["source_files"], **approval_files}
    models = []
    for label in MODEL_LABELS:
        row = next(m for m in preflight["models"] if m["label"] == label)
        _emit(progress, "Sprawdzam nagłówek checkpointu na CPU: " + label, model=label)
        models.append(inspect_checkpoint(row))
    require(len({tuple(m["kpt_shape"]) for m in models}) == 1
            and len({m["native_end2end"] for m in models}) == 1, "MODELS_REQUIRE_DIFFERENT_INFERENCE_PROTOCOLS")
    inference = _inference_parameters(device, models[0]["native_end2end"])
    protocol = {"schema": PROTOCOL_SCHEMA, "geometry_review_sha256": preflight["review_sha256"],
        "geometry_approval_contract_sha256": decision["contract_sha256"],
        "metric_policy_id": preflight["metric_policy"]["id"], "metric_policy_sha256": preflight["metric_policy_sha256"],
        "metric_counts": deepcopy(preflight["metric_counts"]), "models": models,
        "inference": inference, "metrics": deepcopy(METRIC_PROTOCOL), "runtime": _runtime(device),
        "postprocessing": "native_end2end_topk_conf_filter; iou_parameter_inactive" if inference["end2end"] else "Ultralytics_NMS_iou_0.7",
        "rng": deepcopy(RNG_PROTOCOL),
        "code_sha256": _code_receipt(), "source_receipt_sha256": digest(source_receipt),
        "scene_sha256": [s["sha256"] for s in preflight["scenes"]],
        "input": "full_EXIF_oriented_BGR_uint8; no_GT_coordinates_to_predictor; no_application_dedupe_or_fallback",
        "keypoints": "native Results.keypoints.data in oriented source pixels; native scaling/clipping; original output indices; no application sorting/repair",
        "official_metrics": {"box_ap": "NOT_COMPUTED", "pose_ap": "NOT_COMPUTED"}}
    protocol["protocol_sha256"] = digest(protocol)
    output_root = Path(preflight["selection_dir"]).parent.parent / "evaluation_runs"
    prepared = {"protocol": protocol, "preflight": preflight, "decision": decision,
                "approval_dir": str(approval_dir), "source_receipt": source_receipt, "output_root": str(output_root)}
    _verify_prepared(prepared)
    _emit(progress, "GOTOWE — trzy checkpointy Pose zweryfikowane; oczekiwanie na Uruchom pomiar.")
    return prepared


def _verify_prepared(prepared):
    protocol = prepared["protocol"]
    require(protocol.get("schema") == PROTOCOL_SCHEMA
            and protocol.get("protocol_sha256") == digest(_without(protocol, "protocol_sha256")), "MT_PROTOCOL_CHANGED")
    _validate_protocol_parameters(protocol)
    require_current_approval(prepared["preflight"], prepared["decision"])
    require(protocol["geometry_review_sha256"] == prepared["preflight"]["review_sha256"]
            and protocol["geometry_approval_contract_sha256"] == prepared["decision"]["contract_sha256"]
            and protocol["metric_counts"] == prepared["preflight"]["metric_counts"]
            and protocol["metric_policy_sha256"] == prepared["preflight"]["metric_policy_sha256"], "MT_GEOMETRY_BINDING_CHANGED")
    require(protocol["scene_sha256"] == [s["sha256"] for s in prepared["preflight"]["scenes"]], "MT_SCENE_SELECTION_CHANGED")
    require([m["label"] for m in protocol["models"]] == list(MODEL_LABELS), "MT_MODEL_SET_CHANGED")
    for model, source in zip(protocol["models"], [next(m for m in prepared["preflight"]["models"] if m["label"] == label) for label in MODEL_LABELS]):
        require(all(model.get(k) == v for k, v in source.items()), "MT_MODEL_LINEAGE_CHANGED")
    require(protocol["metrics"] == METRIC_PROTOCOL, "MT_METRIC_PROTOCOL_CHANGED")
    require(digest(prepared["source_receipt"]) == protocol["source_receipt_sha256"], "MT_SOURCE_RECEIPT_CHANGED")
    for path, sha in {**prepared["source_receipt"], **protocol["code_sha256"]}.items():
        require(_sha(path) == sha, "MT_SOURCE_OR_CODE_CHANGED: " + str(path))
    require(_runtime(protocol["inference"]["device"]) == protocol["runtime"], "MT_RUNTIME_CHANGED")


@contextmanager
def _execution_lock(output_root):
    require(_RUN_LOCK.acquire(blocking=False), "MT_MEASUREMENT_ALREADY_RUNNING")
    stream = None
    locked = False
    try:
        Path(output_root).mkdir(parents=True, exist_ok=True)
        stream = (Path(output_root) / ".mt_eval396_execution.lock").open("a+b")
        if stream.seek(0, 2) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    finally:
        if stream:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()
        _RUN_LOCK.release()


class UltralyticsBackend:
    def __init__(self, model_row, protocol):
        from ultralytics import YOLO
        require(_sha(model_row["checkpoint_file"]) == model_row["checkpoint_sha256"], "CHECKPOINT_SHA_CHANGED")
        self.model = YOLO(model_row["checkpoint_file"], verbose=False)
        self.protocol, self.model_row = protocol, model_row
        require(self.model.task == "pose", "MT_BACKEND_TASK_CHANGED")

    def predict(self, pixels):
        import torch
        with torch.inference_mode():
            results = self.model.predict(source=pixels, stream=False, **self.protocol["inference"])
        require(len(results) == 1, "MT_BACKEND_SCENE_COUNT_INVALID")
        result = results[0]
        require(tuple(result.orig_shape) == pixels.shape[:2], "MT_BACKEND_COORDINATE_SPACE_CHANGED")
        predictor = self.model.predictor
        actual = {key: _portable(getattr(predictor.args, key, None)) for key in self.protocol["inference"]}
        actual["device"] = str(predictor.model.device)
        actual["checked_imgsz"] = list(predictor.imgsz)
        actual["backend_fp16"] = bool(predictor.model.fp16)
        actual["backend_end2end"] = bool(predictor.model.end2end)
        actual["backend_kpt_shape"] = list(predictor.model.kpt_shape)
        actual["backend_names"] = {str(k): str(v) for k, v in predictor.model.names.items()}
        actual["fused_backend"] = True
        actual["determinism"] = _actual_determinism()
        _validate_effective(actual, self.model_row, self.protocol)
        boxes = result.boxes
        keypoints = result.keypoints.data.detach().cpu().tolist() if result.keypoints is not None else None
        predictions = []
        if boxes is not None:
            xyxy, confs, classes = boxes.xyxy.detach().cpu().tolist(), boxes.conf.detach().cpu().tolist(), boxes.cls.detach().cpu().tolist()
            require(len(xyxy) == len(confs) == len(classes), "MT_BACKEND_ARRAY_LENGTH_MISMATCH")
            require(keypoints is None or len(keypoints) == len(xyxy), "MT_BACKEND_KEYPOINT_COUNT_MISMATCH")
            for index, (bbox, confidence, class_id) in enumerate(zip(xyxy, confs, classes)):
                require(math.isfinite(confidence) and 0 <= confidence <= 1 and class_id == 0, "MT_BACKEND_CONFIDENCE_OR_CLASS_INVALID")
                predictions.append({"index": index, "bbox": bbox, "confidence": confidence,
                                    "class_id": int(class_id), "keypoints": keypoints[index] if keypoints is not None else None})
        return {"predictions": _portable(predictions), "effective_inference": actual,
                "speed_ms": _portable(dict(result.speed))}

    def close(self):
        import torch
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@contextmanager
def _deterministic_runtime(protocol):
    import torch
    previous = (random.getstate(), np.random.get_state(), torch.random.get_rng_state(),
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = protocol["rng"]["cublas_workspace_config"]
    gpu_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        random.setstate(previous[0]); np.random.set_state(previous[1]); torch.random.set_rng_state(previous[2])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[3:5]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[5:7]
        torch.use_deterministic_algorithms(previous[7], warn_only=previous[8])
        if gpu_states is not None:
            torch.cuda.set_rng_state_all(gpu_states)
        if previous[9] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous[9]


def _seed_scene(index):
    import torch
    random.seed(42 + index)
    np.random.seed(42 + index)
    torch.manual_seed(42 + index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + index)


def _decode_scene(scene, source_root):
    path = contained(source_root, scene["file"])
    require(_sha(path) == scene["sha256"], "MT_INPUT_IMAGE_SHA_CHANGED")
    pixels = read_oriented_bgr(path)
    require([pixels.shape[1], pixels.shape[0]] == scene["image"]["oriented_size"]
            and hashlib.sha256(pixels.tobytes()).hexdigest() == scene["image"]["decoded_pixel_sha256"], "MT_INPUT_PIXELS_CHANGED")
    return pixels


def _new_campaign(prepared):
    root = Path(prepared["output_root"]).resolve()
    expected_root = Path(prepared["preflight"]["selection_dir"]).resolve().parent.parent / "evaluation_runs"
    require(root == expected_root, "MT_OUTPUT_ROOT_CHANGED")
    name = "EVAL396_MT_run_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    destination = root / name
    path = root / (name + ".partial")
    path.mkdir()
    files = {}
    for name, payload in (("protocol.json", prepared["protocol"]), ("reference.json", prepared["preflight"]),
                          ("geometry_approval.json", prepared["decision"]), ("source_receipt.json", prepared["source_receipt"])):
        files[name] = _write_json(path / name, payload, exclusive=True)
    manifest = {"schema": CAMPAIGN_SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": prepared["protocol"]["protocol_sha256"], "files": files}
    _write_json(path / "manifest.json", manifest, exclusive=True)
    for model in prepared["protocol"]["models"]:
        _write_json(path / model["label"] / "progress.json", {"status": "PENDING", "records": {}, "summary_sha256": None}, exclusive=True)
    _write_json(path / "status.json", {"status": "PARTIAL", "reason": "CREATED_NOT_STARTED", "measurement_status": "NOT_RUN"})
    require(not destination.exists(), "MT_CAMPAIGN_DESTINATION_EXISTS")
    path.rename(destination)
    return destination


def _validate_row(row, scene, model, protocol):
    require(row.get("schema") == ROW_SCHEMA and row.get("record_sha256") == digest(_without(row, "record_sha256")), "MT_RAW_RECORD_CHANGED")
    require(row["protocol_sha256"] == protocol["protocol_sha256"] and row["model"] == model["label"]
            and row["checkpoint_sha256"] == model["checkpoint_sha256"], "MT_RAW_MODEL_OR_PROTOCOL_CHANGED")
    require(row["scene_sha256"] == scene["sha256"] and row["file"] == scene["file"]
            and row["domain"] == scene["domain"] and [row["width"], row["height"]] == scene["image"]["oriented_size"]
            and row["decoded_pixel_sha256"] == scene["image"]["decoded_pixel_sha256"], "MT_RAW_SOURCE_CHANGED")
    index = protocol["scene_sha256"].index(scene["sha256"])
    require(row.get("scene_index") == index and row.get("seed") == 42 + index, "MT_RAW_SCENE_SEED_CHANGED")
    _validate_effective(row["effective_inference"], model, protocol)


def _verify_campaign_geometry(protocol, reference, decision, source_receipt, reader):
    """Check stored evidence, including frozen XY, without requiring old code to be current."""
    require(reference.get("schema") == geometry_gate.REVIEW_SCHEMA
            and reference.get("review_sha256") == geometry_gate.review_fingerprint(reference), "MT_REFERENCE_FINGERPRINT_CHANGED")
    require(reference.get("status") == "READY_FOR_REVIEW" and not reference.get("blocking_geometry_issues"), "MT_REFERENCE_NOT_APPROVED_FOR_METRICS")
    require(reference["policy"] == geometry_gate.POLICY
            and reference["metric_policy"] == geometry_gate.load_metric_policy()
            and reference["metric_policy_sha256"] == geometry_gate.METRIC_POLICY_SHA, "MT_REFERENCE_POLICY_CHANGED")
    derived_scenes = deepcopy(reference["scenes"])
    derived = geometry_gate.assess_metric_scope(derived_scenes, reference["metric_policy"])
    require(derived_scenes == reference["scenes"] and all(reference[key] == value for key, value in derived.items())
            and reference["counts"] == geometry_gate.cohort_counts(derived_scenes)
            and reference["review_scene_hashes"] == geometry_gate.required_review_scenes(derived_scenes, derived["geometry_issues"]),
            "MT_REFERENCE_ELIGIBILITY_CHANGED")
    require(decision.get("schema") == geometry_gate.APPROVAL_SCHEMA and decision.get("decision") == "APPROVED"
            and bool(str(decision.get("operator_signature") or "").strip()), "MT_STORED_GEOMETRY_APPROVAL_REQUIRED")
    require(decision.get("contract_sha256") == digest(_without(decision, "contract_sha256")), "MT_STORED_APPROVAL_CHANGED")
    require(decision.get("review_sha256") == reference["review_sha256"]
            and decision.get("policy") == reference["policy"]
            and decision.get("selection_sha") == reference["selection_sha"] == geometry_gate.SELECTION_SHA
            and decision.get("gt_freeze_sha") == reference["gt_freeze_sha"] == geometry_gate.FREEZE_SHA
            and set(decision.get("reviewed_scene_hashes", [])) == set(reference["review_scene_hashes"])
            and decision.get("metric_counts") == reference["metric_counts"]
            and decision.get("metric_exclusions") == reference["metric_exclusions"]
            and decision.get("metric_policy_sha256") == reference["metric_policy_sha256"], "MT_STORED_APPROVAL_BINDING_CHANGED")
    require(protocol["metric_counts"] == reference["metric_counts"]
            and protocol["metric_policy_sha256"] == reference["metric_policy_sha256"]
            and protocol["metric_policy_id"] == reference["metric_policy"]["id"], "MT_CAMPAIGN_METRIC_SCOPE_CHANGED")
    require(digest(source_receipt) == protocol["source_receipt_sha256"]
            and all(source_receipt.get(path) == sha for path, sha in reference["source_files"].items()), "MT_SOURCE_RECEIPT_CHANGED")
    sources = {model["label"]: model for model in reference["models"]}
    require(set(sources) == set(MODEL_LABELS), "MT_REFERENCE_MODEL_SET_CHANGED")
    for model in protocol["models"]:
        require(all(model.get(key) == value for key, value in sources[model["label"]].items()), "MT_MODEL_LINEAGE_CHANGED")
    geometry_gate.verify_report_source_binding(reference, reader)


def _read_campaign_inputs(path, reader):
    manifest = reader.json(path / "manifest.json")
    require(manifest.get("schema") == CAMPAIGN_SCHEMA, "MT_CAMPAIGN_SCHEMA_INVALID")
    require(set(manifest.get("files", {})) == {"protocol.json", "reference.json", "geometry_approval.json", "source_receipt.json"},
            "MT_CAMPAIGN_INPUT_FILES_CHANGED")
    payload = {name: reader.json(contained(path, name), sha) for name, sha in manifest["files"].items()}
    protocol, reference, decision = payload["protocol.json"], payload["reference.json"], payload["geometry_approval.json"]
    require(protocol.get("schema") == PROTOCOL_SCHEMA and protocol["protocol_sha256"] == manifest["protocol_sha256"]
            == digest(_without(protocol, "protocol_sha256")), "MT_PROTOCOL_CHANGED")
    _validate_protocol_parameters(protocol)
    require([m["label"] for m in protocol["models"]] == list(MODEL_LABELS), "MT_MODEL_SET_CHANGED")
    require(protocol["geometry_review_sha256"] == reference["review_sha256"]
            and protocol["geometry_approval_contract_sha256"] == decision["contract_sha256"], "MT_CAMPAIGN_APPROVAL_BINDING_CHANGED")
    require(protocol["scene_sha256"] == [s["sha256"] for s in reference["scenes"]], "MT_SCENE_SELECTION_CHANGED")
    _verify_campaign_geometry(protocol, reference, decision, payload["source_receipt.json"], reader)
    return manifest, payload


def _read_pending(path, reader, scenes, model, protocol):
    if not path.exists():
        return None
    pending = reader.json(path)
    require(pending.get("schema") == "alpr.mt_pending_scene.v1"
            and pending.get("transaction_sha256") == digest(_without(pending, "transaction_sha256")), "MT_PENDING_RECORD_CHANGED")
    row = pending["row"]
    sha = row.get("scene_sha256")
    require(sha in scenes, "MT_PENDING_UNKNOWN_SCENE")
    _validate_row(row, scenes[sha], model, protocol)
    aggregate([row], [scenes[sha]], protocol["metrics"])
    require(pending.get("raw_file_sha256") == hashlib.sha256(canonical(row)+b"\n").hexdigest(), "MT_PENDING_RAW_SHA_CHANGED")
    return pending


def _completion_files(path, protocol, models):
    names = {"manifest.json", "protocol.json", "reference.json", "geometry_approval.json", "source_receipt.json"}
    for model in models:
        label = model["label"]
        names.update({label+"/progress.json", label+"/summary.json"})
        names.update(label+"/raw/"+sha+".json" for sha in protocol["scene_sha256"])
    return {name: _sha(contained(path, name)) for name in sorted(names)}


def _publish_row(path, model, scene, protocol, ledger, row):
    """Write-ahead full row allows recovery after any publication boundary."""
    label, sha = model["label"], scene["sha256"]
    _validate_row(row, scene, model, protocol)
    aggregate([row], [scene], protocol["metrics"])
    require(sha not in ledger["records"], "MT_RAW_ALREADY_INDEXED")
    pending_path = path / label / "pending.json"
    pending = {"schema": "alpr.mt_pending_scene.v1", "row": row,
               "raw_file_sha256": hashlib.sha256(canonical(row)+b"\n").hexdigest()}
    pending["transaction_sha256"] = digest(pending)
    _write_json(pending_path, pending, exclusive=True)
    raw_sha = _write_json(path / label / "raw" / (sha+".json"), row, exclusive=True)
    updated = deepcopy(ledger)
    updated["records"][sha] = raw_sha
    updated["summary_sha256"] = None
    _write_json(path / label / "progress.json", updated)
    ledger.clear(); ledger.update(updated)
    pending_path.unlink()  # Only this owned, successfully committed journal.


def _summarize_model(path, model, reference, protocol):
    label = model["label"]
    ledger_path = path / label / "progress.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger["status"] == "COMPLETE":
        return ledger
    rows = []
    for scene in reference["scenes"]:
        sha = scene["sha256"]
        if sha not in ledger["records"]:
            continue
        raw_path = path / label / "raw" / (sha+".json")
        require(_sha(raw_path) == ledger["records"][sha], "MT_RAW_RECORD_CHANGED")
        row = json.loads(raw_path.read_text(encoding="utf-8"))
        _validate_row(row, scene, model, protocol)
        rows.append(row)
    summary = aggregate(rows, reference["scenes"], protocol["metrics"])
    ledger["summary_sha256"] = _write_json(path / label / "summary.json", summary)
    ledger["status"] = "PARTIAL" if (path / label / "pending.json").exists() else summary["status"]
    _write_json(ledger_path, ledger)
    return ledger


def _recover_campaign(path, prepared):
    """Explicit Resume only: adopt proven journal/orphan rows, never rerun them."""
    existing = load_campaign(path)
    require(existing["protocol"] == prepared["protocol"]
            and existing["preflight"]["review_sha256"] == prepared["preflight"]["review_sha256"]
            and existing["geometry_approval"] == prepared["decision"], "MT_RESUME_PROTOCOL_OR_REFERENCE_CHANGED")
    if existing["status"] == "COMPLETE":
        return existing
    for model_info, model in zip(existing["models"], prepared["protocol"]["models"]):
        if not model_info["recovery_required"]:
            continue
        label = model["label"]
        ledger_path = path / label / "progress.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        require(ledger["status"] != "COMPLETE", "MT_COMPLETED_MODEL_RECOVERY_FORBIDDEN")
        for item in model_info["recoverable_rows"]:
            row, sha = item["row"], item["row"]["scene_sha256"]
            raw_path = path / label / "raw" / (sha+".json")
            if raw_path.exists():
                require(_sha(raw_path) == item["sha256"], "MT_RECOVERY_RAW_CHANGED")
            else:
                require(_write_json(raw_path, row, exclusive=True) == item["sha256"], "MT_RECOVERY_RAW_CHANGED")
            ledger["records"][sha] = item["sha256"]
        if model_info["recoverable_rows"]:
            ledger.update(status="PARTIAL", summary_sha256=None)
            _write_json(ledger_path, ledger)
        pending = path / label / "pending.json"
        if pending.exists():
            require(_sha(pending) == model_info["pending_sha256"], "MT_PENDING_CHANGED_DURING_RECOVERY")
            pending.unlink()
        _summarize_model(path, model, prepared["preflight"], prepared["protocol"])
    return load_campaign(path)


def load_campaign(run_dir):
    """Read/verify evidence and journals; never repair or infer during loading."""
    path = Path(run_dir).resolve()
    reader = EvidenceReader()
    manifest, payload = _read_campaign_inputs(path, reader)
    protocol, reference = payload["protocol.json"], payload["reference.json"]
    status = reader.json(path / "status.json")
    require(status.get("status") in {"PARTIAL", "RUNNING", "COMPLETE"}, "MT_CAMPAIGN_STATUS_INVALID")
    scenes = {s["sha256"]: s for s in reference["scenes"]}
    models = []
    for model in protocol["models"]:
        label = model["label"]
        require(label in MODEL_LABELS, "MT_MODEL_LABEL_INVALID")
        ledger = reader.json(path / label / "progress.json")
        require(ledger.get("status") in {"PENDING", "RUNNING", "PARTIAL", "COMPLETE"}
                and isinstance(ledger.get("records"), dict), "MT_MODEL_PROGRESS_INVALID")
        raw_dir = path / label / "raw"
        disk_records = {p.stem for p in raw_dir.glob("*.json")}
        require(set(ledger["records"]) <= disk_records, "MT_MISSING_INDEXED_RAW_RECORD")
        require(disk_records <= set(scenes), "MT_UNKNOWN_RAW_SCENE")
        pending_path = path / label / "pending.json"
        pending = _read_pending(pending_path, reader, scenes, model, protocol)
        recoverable = {}
        orphan_shas = disk_records-set(ledger["records"])
        require(not orphan_shas or (pending is not None and orphan_shas == {pending["row"]["scene_sha256"]}),
                "MT_UNJOURNALED_ORPHAN_RAW_RECORD")
        for sha in sorted(orphan_shas):
            raw_path = raw_dir / (sha+".json")
            row = reader.json(raw_path)
            _validate_row(row, scenes[sha], model, protocol)
            aggregate([row], [scenes[sha]], protocol["metrics"])
            recoverable[sha] = {"row": row, "sha256": reader.receipt[str(raw_path.resolve())]}
        if pending:
            sha = pending["row"]["scene_sha256"]
            if sha in disk_records:
                raw_path = raw_dir / (sha+".json")
                disk_row = reader.json(raw_path, pending["raw_file_sha256"])
                require(disk_row == pending["row"], "MT_PENDING_RAW_CONTENT_CHANGED")
            if sha not in ledger["records"]:
                recoverable[sha] = {"row": pending["row"], "sha256": pending["raw_file_sha256"]}
        rows = []
        for sha in protocol["scene_sha256"]:
            if sha not in ledger["records"]:
                continue
            row = reader.json(raw_dir / (sha + ".json"), ledger["records"][sha])
            _validate_row(row, scenes[sha], model, protocol)
            aggregate([row], [scenes[sha]], protocol["metrics"])
            rows.append(row)
        require(set(ledger["records"]) <= set(scenes), "MT_UNKNOWN_RAW_SCENE")
        metrics = None
        if ledger["summary_sha256"]:
            metrics = reader.json(path / label / "summary.json", ledger["summary_sha256"])
            # Recalculate from immutable raw evidence, not from model predictions.
            require(metrics == aggregate(rows, reference["scenes"], protocol["metrics"]), "MT_SUMMARY_DIFFERS_FROM_RAW")
        if ledger["status"] == "COMPLETE":
            require(len(rows) == len(scenes) and metrics is not None and metrics["status"] == "COMPLETE"
                    and not recoverable and not pending, "MT_MODEL_INCOMPLETE")
        models.append({"label": label, "status": ledger["status"], "metrics": metrics, "rows": rows,
                       "recovery_required": bool(recoverable or pending),
                       "recoverable_rows": [recoverable[key] for key in sorted(recoverable)],
                       "pending_sha256": reader.receipt.get(str(pending_path.resolve()))})
    require([m["label"] for m in models] == list(MODEL_LABELS), "MT_MODEL_SET_CHANGED")
    if status["status"] == "COMPLETE":
        require(all(m["status"] == "COMPLETE" for m in models), "MT_CAMPAIGN_INCOMPLETE")
        completion = reader.json(path / "completion_manifest.json")
        require(completion["protocol_sha256"] == protocol["protocol_sha256"], "MT_COMPLETION_PROTOCOL_CHANGED")
        require(completion.get("files") == _completion_files(path, protocol, models), "MT_COMPLETION_FILE_COVERAGE_CHANGED")
        for filename, sha in completion["files"].items():
            reader.read(contained(path, filename), sha)
    reader.finish()
    return {"run_dir": str(path), "protocol": protocol, "preflight": reference,
            "geometry_approval": payload["geometry_approval.json"], "source_receipt": payload["source_receipt.json"],
            "status": status["status"], "status_detail": status, "created_at": manifest["created_at"],
            "models": models, "source_files": dict(reader.receipt),
            "recovery_required": any(m["recovery_required"] for m in models)}


def list_campaigns(output_root):
    rows = []
    for path in sorted(Path(output_root).glob("EVAL396_MT_run_*"), reverse=True):
        if not path.is_dir() or path.name.endswith(".partial"):
            continue
        try:
            data = load_campaign(path)
            rows.append({"run_dir": str(path), "status": "PARTIAL" if data["recovery_required"] else data["status"], "created_at": data["created_at"],
                "models_completed": sum(m["status"] == "COMPLETE" for m in data["models"]),
                "scenes_completed": sum(len(m["rows"]) for m in data["models"]),
                "protocol_sha256": data["protocol"]["protocol_sha256"], "recovery_required": data["recovery_required"]})
        except Exception as exc:
            rows.append({"run_dir": str(path), "status": "CORRUPT", "created_at": "", "models_completed": 0,
                         "scenes_completed": 0, "protocol_sha256": "", "error": str(exc)})
    return rows


def run_campaign(prepared, *, progress=None, cancelled=None, resume_dir=None):
    """Called by the explicit native Start/Resume action, one model at a time."""
    cancelled = cancelled or (lambda: False)
    _emit(progress, "Ponownie weryfikuję podpis, źródła, kod i środowisko…", stage="verify")
    _verify_prepared(prepared)
    protocol, reference = prepared["protocol"], prepared["preflight"]
    root = Path(prepared["output_root"]).resolve()
    require(root == Path(reference["selection_dir"]).resolve().parent.parent / "evaluation_runs", "MT_OUTPUT_ROOT_CHANGED")
    path = None
    with _execution_lock(root), _deterministic_runtime(protocol):
        if resume_dir:
            path = Path(resume_dir).resolve()
            require(path.parent == root and path.name.startswith("EVAL396_MT_run_"), "MT_RESUME_OUTSIDE_EXPERIMENT")
            existing = _recover_campaign(path, prepared)
            if existing["status"] == "COMPLETE":
                return existing  # A complete model/campaign is never inferred again.
        else:
            path = _new_campaign(prepared)
        started = time.monotonic()
        _write_json(path / "status.json", {"status": "RUNNING", "measurement_status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat()})
        stop_reason = None
        try:
            total = len(reference["scenes"]) * len(protocol["models"])
            completed = sum(len(json.loads((path / m["label"] / "progress.json").read_text(encoding="utf-8"))["records"]) for m in protocol["models"])
            for model in protocol["models"]:
                label = model["label"]
                ledger_path = path / label / "progress.json"
                ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
                if ledger["status"] == "COMPLETE":
                    continue
                if len(ledger["records"]) == len(reference["scenes"]):
                    _summarize_model(path, model, reference, protocol)
                    continue  # Fully recorded model: no backend reload or forward.
                if cancelled():
                    stop_reason = "CANCELLED"
                    break
                _emit(progress, "Ładuję " + label, stage="inference", model=label, completed=completed, total=total)
                backend = UltralyticsBackend(model, protocol)
                ledger["status"] = "RUNNING"
                _write_json(ledger_path, ledger)
                try:
                    for index, scene in enumerate(reference["scenes"]):
                        if scene["sha256"] in ledger["records"]:
                            continue
                        if cancelled():
                            stop_reason = "CANCELLED"
                            break
                        _seed_scene(index)
                        pixels = _decode_scene(scene, reference["source_scene_root"])
                        wall_start = time.monotonic()
                        result = backend.predict(pixels)  # Only pixels. No GT, file name, boxes or corners.
                        elapsed = (time.monotonic() - wall_start) * 1000
                        require(_sha(contained(reference["source_scene_root"], scene["file"])) == scene["sha256"], "MT_INPUT_CHANGED_DURING_PREDICTION")
                        row = {"schema": ROW_SCHEMA, "protocol_sha256": protocol["protocol_sha256"],
                               "model": label, "checkpoint_sha256": model["checkpoint_sha256"],
                               "scene_index": index, "scene_sha256": scene["sha256"], "file": scene["file"], "domain": scene["domain"],
                               "width": pixels.shape[1], "height": pixels.shape[0],
                               "decoded_pixel_sha256": scene["image"]["decoded_pixel_sha256"],
                               "seed": 42 + index, "elapsed_ms": elapsed, **result}
                        row = _portable(row)
                        # Validate raw semantics before writing; evaluation cannot affect inference already completed.
                        row["record_sha256"] = digest(row)
                        _publish_row(path, model, scene, protocol, ledger, row)
                        completed += 1
                        _emit(progress, f"{label}: scena {index + 1}/{len(reference['scenes'])}", stage="inference", model=label,
                              scene=index + 1, total_scenes=len(reference["scenes"]), completed=completed, total=total)
                        del pixels
                finally:
                    backend.close()
                _summarize_model(path, model, reference, protocol)
                if stop_reason:
                    break
            if not stop_reason:
                _emit(progress, "Weryfikuję kompletność wyników i niezmienność źródeł…", stage="verify", completed=completed, total=total)
                _verify_prepared(prepared)
                data = load_campaign(path)
                require(all(m["status"] == "COMPLETE" for m in data["models"]), "MT_CAMPAIGN_INCOMPLETE")
                files = _completion_files(path, protocol, data["models"])
                completion_payload = {"protocol_sha256": protocol["protocol_sha256"], "files": files}
                completion_path = path / "completion_manifest.json"
                if completion_path.exists():
                    require(json.loads(completion_path.read_text(encoding="utf-8")) == completion_payload,
                            "MT_EXISTING_COMPLETION_MANIFEST_CHANGED")
                else:
                    _write_json(completion_path, completion_payload, exclusive=True)
                _write_json(path / "status.json", {"status": "COMPLETE", "measurement_status": "COMPLETE",
                    "completed_scenes": completed, "completed_models": 3, "elapsed_seconds": time.monotonic() - started})
            else:
                _write_json(path / "status.json", {"status": "PARTIAL", "measurement_status": "PARTIAL", "reason": stop_reason})
        except Exception as exc:
            summary_errors = []
            for model in protocol["models"]:
                try:
                    ledger = json.loads((path / model["label"] / "progress.json").read_text(encoding="utf-8"))
                    if ledger["status"] != "PENDING":
                        _summarize_model(path, model, reference, protocol)
                except Exception as summary_exc:
                    summary_errors.append({"model": model["label"], "error": str(summary_exc)})
            _write_json(path / "status.json", {"status": "PARTIAL", "measurement_status": "PARTIAL", "reason": "ERROR",
                                               "error": str(exc), "summary_errors": summary_errors})
            # Keep interrupted raw evidence. The UI receives its path and explicit failure.
            _emit(progress, "Przebieg częściowy: " + str(exc), stage="error", run_dir=str(path))
    return load_campaign(path)
