"""Local evidence for a rejected conversion; never publishes a model package."""
from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile

from .tflite_precision import load_schema


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _custom_codes(path):
    """Supplement the unchanged inspector; unknown/malformed graphs stay unknown."""
    try:
        data = path.read_bytes()
        if data[4:8] != b"TFL3":
            return {"status": "NOT_IDENTIFIED", "operators": []}
        model = load_schema().Model.GetRootAsModel(data, 0)
        if not 0 <= model.OperatorCodesLength() <= 100_000:
            raise ValueError("operator limit")
        rows = []
        for i in range(model.OperatorCodesLength()):
            raw = model.OperatorCodes(i).CustomCode()
            if raw:
                name = raw.decode("utf-8", errors="replace")
                rows.append({"opcode_index": i, "custom_code": name,
                             "flex_prefix_observed": name.startswith("Flex")})
        return {"status": "OBSERVED", "operators": rows}
    except Exception:
        return {"status": "NOT_IDENTIFIED", "operators": []}


def write_rejected_precision(source, report, *, root, request=None):
    """Retain identical bytes, then atomically commit JSON in a unique directory.

    A partial directory on I/O failure is not a successful evidence record. It is
    deliberately not advertised by the caller. No source or historical evidence
    is removed, moved, or overwritten; inherited local ACLs remain unchanged.
    """
    source, root = Path(source), Path(root)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Expected a regular generated TFLite file")
    source_size = source.stat().st_size
    source_sha = _digest(source)
    if source_sha != report["artifact_sha256"] or source_size != report["artifact_bytes"]:
        raise ValueError("Generated file changed after graph inspection")
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="rejected_int8_", dir=root))
    retained = folder / "rejected_model.tflite"
    with source.open("rb") as src, retained.open("xb") as dst:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    retained_sha = _digest(retained)
    if (retained_sha != source_sha or retained.stat().st_size != source_size
            or _digest(source) != source_sha or source.stat().st_size != source_size):
        raise ValueError("Retained TFLite byte verification failed")
    versions = {"python": platform.python_version()}
    for name in ("ultralytics", "onnx2tf", "tensorflow", "tensorflow-cpu", "onnx",
                 "onnxruntime", "ai-edge-litert", "flatbuffers", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    operators = [{"subgraph": sg["index"], **op}
                 for sg in report.get("subgraphs", []) for op in sg["operators"]]
    io = [{"subgraph": sg["index"], "boundary": key, **tensor}
          for sg in report.get("subgraphs", []) for key in ("inputs", "outputs")
          for tensor in sg[key]]
    payload = {
        "schema": "alpr.rejected_tflite_precision.v1",
        "diagnostic_id": folder.name,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "INT8_NOT_PROVEN", "execution_precision": "UNVERIFIED",
        "diagnostic_only": True, "publication_status": "NOT_PUBLISHED",
        "source_converted_file_sha256": source_sha,
        "source_converted_file_bytes": source_size,
        "retained_tflite_file": retained.name,
        "retained_tflite_sha256": retained_sha,
        "retained_tflite_bytes": retained.stat().st_size,
        "byte_verification": "MATCH", "graph": report,
        "runtime_versions": versions,
        "non_integer_operators": [op for op in operators
            if op["mode"] not in {"INT8", "STRUCTURAL"} or not op["verified_signature"]],
        "boundary_io": io,
        "diagnosis": {
            "scope": "Static graph observations, not converter root cause or runtime execution",
            "signed_int8_io": bool(io) and all(t["dtype"] == "INT8" for t in io),
            "non_int8_boundary_tensors": [t for t in io if t["dtype"] != "INT8"],
            "float_compute_operators": [op for op in operators if op["mode"] in {"FP32", "MIXED"}
                and op["operator"] not in {"QUANTIZE", "DEQUANTIZE"}],
            "quantize_dequantize_operators": [op for op in operators
                if op["operator"] in {"QUANTIZE", "DEQUANTIZE"}],
            "unverified_signatures": [op for op in operators if not op["verified_signature"]],
            "custom_codes": _custom_codes(retained),
            "converter_root_cause": "NOT_IDENTIFIED",
        },
        "access_policy": "Local only; inherited filesystem permissions, no ACL changes",
    }
    if request is not None:
        identity = {"role": request.role, "image_size": request.image_size,
                    "confidence_threshold": request.confidence_threshold,
                    "iou_threshold": request.iou_threshold,
                    "calibration_mode": request.calibration_mode,
                    "calibration_fingerprint_requested": request.calibration_fingerprint}
        for key, path in (("checkpoint", request.checkpoint),
                          ("calibration_manifest", request.calibration_manifest)):
            try:
                identity[key + "_sha256"] = _digest(path) if path else None
                if key == "calibration_manifest" and path:
                    # Whitelist identity only; never dump sample paths, images or GT.
                    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
                    identity["run_id"] = manifest.get("config", {}).get("run_id")
                    identity["calibration_schema"] = manifest.get("schema")
                    identity["calibration_input_fingerprint"] = manifest.get("input_fingerprint")
                    if _digest(path) != identity[key + "_sha256"]:
                        raise ValueError("Manifest changed while reading identity")
            except Exception as exc:
                identity[key + "_read_error"] = type(exc).__name__
        payload["request"] = identity
    pending = folder / "rejected_precision.json.tmp"
    target = folder / "rejected_precision.json"
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    # os.rename (not replace): an existing report is never overwritten on Windows.
    os.rename(pending, target)
    return target
