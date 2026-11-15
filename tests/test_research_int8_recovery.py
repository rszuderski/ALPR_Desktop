"""F4 research-only MT Pose: fail-closed rewrite and narrow integer CAST policy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest

from auto_annotation_tool.exporters import research_int8_recovery as recovery
from auto_annotation_tool.exporters import tflite_precision


def fixture():
    root = Path(__file__).resolve().parents[1]
    return (root / "output" / "mobile_export_diagnostics" / "rejected_int8_rrr27nuo"
            / "rejected_model.tflite")


def test_recovery_reference_model_byte_identical_and_static_int8():
    src = fixture()
    if not src.is_file():
        pytest.skip("Historical rejected MT-DAY fixture exists only on Desktop operator PC")
    original = src.read_bytes()
    assert hashlib.sha256(original).hexdigest() == recovery.REFERENCE_SOURCE_SHA
    result, changes = recovery.patch(original)
    assert hashlib.sha256(result).hexdigest() == "1cc49f7c0ff72d56301c7ea560f7fa32b2e3f6915f28b4ac794c3a96bf7d115f"
    assert not recovery.audit(result)["float32_tensor_indices"]
    assert len(changes) >= 10
    report = tflite_precision.analyze_tflite_bytes(result)
    assert report["classification"] == "INT8", report["issues"]
    assert report["full_integer_graph"]
    assert not report["issues"]
    assert all(t["dtype"] == "INT8" for sg in report["subgraphs"]
               for key in ("inputs", "outputs") for t in sg[key])


def test_rejects_malformed_or_wrong_model():
    for payload in (b"", b"not a FlatBuffer", b"\x00" * 1_000_000):
        with pytest.raises((ValueError, IndexError, struct.error, OverflowError)):
            recovery.patch(payload)


def test_full_desktop_recovery_offline_from_saved_graph(tmp_path):
    """No 18-minute export: real FlatBuffer plus actual TensorFlow CPU proof."""
    src = fixture()
    previous_report = src.parent / "rejected_precision.json"
    if not src.is_file() or not previous_report.is_file():
        pytest.skip("Historical rejected MT-DAY and its JSON exist only on Desktop PC")
    report = json.loads(previous_report.read_text(encoding="utf-8-sig"))["graph"]
    copied = tmp_path / "original.tflite"
    copied.write_bytes(src.read_bytes())  # original never opened for writing
    result = recovery.recover_research_int8(
        copied, report=report, diagnostics_root=tmp_path / "evidence",
    )
    assert hashlib.sha256(result.read_bytes()).hexdigest() == (
        "1cc49f7c0ff72d56301c7ea560f7fa32b2e3f6915f28b4ac794c3a96bf7d115f"
    )
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == recovery.REFERENCE_SOURCE_SHA
    evidence = list((tmp_path / "evidence").glob("recovered_int8_*/recovery_report.json"))
    assert len(evidence) == 1
    payload = json.loads(evidence[0].read_text(encoding="utf-8"))
    assert payload["cpu_equivalence"]["status"] == "CPU_EXACT_EQ_PASS_NOT_ANDROID_PROOF"
    assert payload["cpu_equivalence"]["cases_run"] == 8
    assert payload["cpu_equivalence"]["total_values_compared"] == 33_600


def test_rejects_unsupported_quant_scale_or_class_signature():
    src = fixture()
    if not src.is_file():
        pytest.skip("Historical rejected MT-DAY fixture exists only on Desktop operator PC")
    original = bytearray(src.read_bytes())
    f, root, sg, tensors, ops, codes, bufs = recovery.analyze(original)
    # Break terminal output quantization scale; safe in-memory mutation only.
    q = f.obj(tensors[763], 4)
    vec, n = f.vector(q, 2)
    assert n == 1
    struct.pack_into("<f", original, vec + 4, 0.1)
    with pytest.raises(ValueError, match="qparams differ"):
        recovery.patch(bytes(original))

    original2 = bytearray(src.read_bytes())
    f, root, sg, tensors, ops, codes, bufs = recovery.analyze(original2)
    p, n = f.vector(bufs[f.scalar(tensors[48], 2)], 0)
    assert n == 8
    struct.pack_into("<q", original2, p + 4, 2)  # not nc=1
    with pytest.raises(ValueError, match="nc=1"):
        recovery.patch(bytes(original2))


def test_narrow_integer_cast_signature_not_a_mixed_graph_waiver():
    src = {"index": 736, "dtype": "INT64", "shape": [1, 300, 1],
           "scale": [], "zero_point": []}
    output = {"index": 737, "dtype": "INT8", "shape": [1, 300, 1],
              "scale": [1.0], "zero_point": [0]}
    issues = []
    mode, weights, bias = tflite_precision._operator_mode("CAST", [src], [output],
                                                         "test CAST: ", issues)
    assert mode == "INT8" and not issues and not weights and not bias
    # Wrong qparams must remain unsupported.
    other = {**output, "zero_point": [-91]}
    issues = []
    mode, weights, bias = tflite_precision._operator_mode("CAST", [src], [other],
                                                         "test CAST: ", issues)
    assert mode == "UNVERIFIED" and issues
    # Original float class conversion must remain FP32.
    fl = {**output, "dtype": "FLOAT32", "scale": [], "zero_point": []}
    issues = []
    mode, _, _ = tflite_precision._operator_mode("CAST", [src], [fl], "test: ", issues)
    assert mode == "FP32"


def test_gui_scope_and_existing_strict_gate_preserved():
    root = Path(__file__).resolve().parents[1]
    exporter = (root / "auto_annotation_tool/exporters/mobile_model_exporter.py").read_text(encoding="utf-8-sig")
    assert "# F4_RESEARCH_INT8_RECOVERY_GUI_HOOK_V1" in exporter
    assert 'request.calibration_mode == exploratory.MODE and request.role == "plate"' in exporter
    assert 'def _approved_calibration(request: MobileExportRequest)' in exporter
    assert "write_rejected_precision(full, report," in exporter
    assert "RESEARCH_INT8_RECOVERY_FAILED" in exporter
