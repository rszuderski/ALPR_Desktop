#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research-only, structural MT Pose INT8 recovery (not TensorFlow converter fix).

This module may run ONLY after the normal Ultralytics INT8 conversion,
ONLY for a clearly matched single-class Pose graph, and ONLY in the
Desktop GUI exploratory calibration workflow. No strict-policy bypass.
A CPU interpreter comparison and strict static precision check are required
before returning a recoverable variant to the existing package exporter.
"""
import collections
import hashlib
import json
from pathlib import Path
import os
import shutil
import tempfile
import uuid
import warnings
import struct

REFERENCE_SOURCE_SIZE = 2916342
REFERENCE_SOURCE_SHA = "564030f439b306d9d59c2ca148ff5792cacdc4bc4a40a3bd31b02893432fcc56"
FLOAT_TENSORS = (718, 719, 737, 750, 755, 761, 762)
FIRST_OPS = (428, 458, 463, 469)
FLOAT_VTABLE_TYPE_RESERVED_OFFSET = 5
TENSOR_TYPE_INT8 = 9


class FBuf:
    def __init__(self, b):
        self.b = b

    def u8(self, p):
        return self.b[p]

    def u16(self, p):
        return struct.unpack_from("<H", self.b, p)[0]

    def u32(self, p):
        return struct.unpack_from("<I", self.b, p)[0]

    def i32(self, p):
        return struct.unpack_from("<i", self.b, p)[0]

    def vtable(self, t):
        v = t - self.i32(t)
        if v < 8 or v >= len(self.b) - 4:
            raise ValueError("Invalid vtable pointer")
        return v, self.u16(v), self.u16(v + 2)

    def field(self, t, i):
        v, n, z = self.vtable(t)
        if n <= 4 + 2 * i:
            return None
        off = self.u16(v + 4 + 2 * i)
        return t + off if off else None

    def obj(self, t, i):
        p = self.field(t, i)
        return p + self.u32(p) if p is not None else None

    def scalar(self, t, i, fmt="I", default=0):
        p = self.field(t, i)
        return struct.unpack_from("<" + fmt, self.b, p)[0] if p is not None else default

    def vector(self, t, i):
        p = self.obj(t, i)
        return (p, self.u32(p)) if p is not None else (None, 0)

    def vals(self, t, i, fmt="i"):
        p, n = self.vector(t, i)
        z = struct.calcsize("<" + fmt)
        return [struct.unpack_from("<" + fmt, self.b, p + 4 + z * j)[0] for j in range(n)] if p is not None else None

    def tables(self, t, i):
        p, n = self.vector(t, i)
        return [p + 4 + 4 * j + self.u32(p + 4 + 4 * j) for j in range(n)] if p is not None else None

    def constbuf(self, root, tensor, buffers):
        bidx = self.scalar(tensor, 2)
        p, n = self.vector(buffers[bidx], 0)
        return bytes(self.b[p + 4:p + 4 + n]) if p is not None else b""


def s_u8(b, pos, val):
    struct.pack_into("<B", b, pos, val)


def s_u16(b, pos, val):
    struct.pack_into("<H", b, pos, val)


def s_u32(b, pos, val):
    struct.pack_into("<I", b, pos, val)


def s_i32(b, pos, val):
    struct.pack_into("<i", b, pos, val)


def align(b, n, modulo=0):
    while len(b) % n != modulo:
        b.append(0)


def append_class_quantization(b):
    """Add an immutable Q table: scale=[1.0], zero_point=[0].

    The physical value of INT8 index 0 must be zero before the explicit
    Quantize op maps it into the existing output qparams (0.0078669, -91).
    """
    align(b, 2)
    vpos = len(b)
    # QuantizationParameters: min(0), max(1), scale(2), zero_point(3).
    # No min/max/details/quantized_dimension fields are needed.
    b.extend(struct.pack("<HHHHHH", 12, 12, 0, 0, 4, 8))
    align(b, 4)
    tpos = len(b)
    b.extend(b"\x00" * 12)
    s_i32(b, tpos, tpos - vpos)
    align(b, 4)
    scale_pos = len(b)
    b.extend(struct.pack("<If", 1, 1.0))
    align(b, 8, 4)  # int64 vector payload must begin at 8-byte alignment
    zero_pos = len(b)
    b.extend(struct.pack("<Iq", 1, 0))
    s_u32(b, tpos + 4, scale_pos - (tpos + 4))
    s_u32(b, tpos + 8, zero_pos - (tpos + 8))
    return tpos


def ptr_to(b, field_position, target):
    if not target > field_position:
        raise ValueError("FlatBuffer uoffset target must be forward")
    s_u32(b, field_position, target - field_position)


def analyze(b):
    f = FBuf(b)
    if bytes(b[4:8]) != b"TFL3":
        raise ValueError("Not a TFL3 flatbuffer")
    root = f.u32(0)
    ss = f.tables(root, 2)
    if len(ss) != 1:
        raise ValueError("Expected a single subgraph")
    sg = ss[0]
    t = f.tables(sg, 0)
    op = f.tables(sg, 3)
    codes = f.tables(root, 1)
    bufs = f.tables(root, 4)
    return f, root, sg, t, op, codes, bufs


def assert_source(b):
    if not 500_000 <= len(b) <= 30_000_000:
        raise ValueError("UNSUPPORTED_RESEARCH_GRAPH: source size is outside the bounded template")
    f, root, sg, t, op, codes, bufs = analyze(b)
    if len(t) != 764 or len(op) != 472:
        raise ValueError("Unexpected model tensor/operator counts")
    flt = {i for i, x in enumerate(t) if f.scalar(x, 1, "b", 0) == 0}
    if flt != set(FLOAT_TENSORS):
        raise ValueError(f"Unexpected FLOAT32 tensor indices: {sorted(flt)}")
    if f.vals(t[0], 0) != [1, 512, 512, 3] or f.vals(t[763], 0) != [1, 300, 14]:
        raise ValueError("Input or output shape mismatch")
    if f.scalar(t[0], 1, "b") != TENSOR_TYPE_INT8 or f.scalar(t[763], 1, "b") != TENSOR_TYPE_INT8:
        raise ValueError("Input or output is not signed INT8")
    if f.vals(t[750], 0) != [1, 300, 8]:
        raise ValueError("Tensor 750 original shape mismatch")
    if f.vals(t[737], 0) != [1, 300, 1]:
        raise ValueError("Tensor 737 original shape mismatch")
    if struct.unpack("<q", f.constbuf(root, t[48], bufs)) != (1,):
        raise ValueError("Model is NOT confirmed nc=1")
    if struct.unpack("<i", f.constbuf(root, t[57], bufs)) != (300,):
        raise ValueError("TopK k mismatch")
    expected = {
        428: ([717], [718], 6), 429: ([718, 57], [719, 720], 48),
        441: ([731, 57], [732, 733], 48), 445: ([736], [737], 53),
        458: ([749], [750], 6), 463: ([754], [755], 6),
        469: ([760], [761], 6), 470: ([761, 755, 737, 750], [762], 2),
        471: ([762], [763], 114),
    }
    for i, (inputs, outputs, builtin) in expected.items():
        o=op[i]; ci=f.scalar(o, 0)
        if f.vals(o,1)!=inputs or f.vals(o,2)!=outputs or f.scalar(codes[ci],3,"i",f.scalar(codes[ci],0,"b"))!=builtin:
            raise ValueError(f"Unexpected original operator {i}")
    if f.scalar(op[471],0) == f.scalar(op[428],0):
        raise ValueError("Expected different original Quantize/Dequantize opcodes")
    # Empty FLOAT32 Tensor tables share a single vtable; only the seven
    # intended tables may reference it. Its unused byte +5 is safe to claim.
    vt = f.vtable(t[718])[0]
    if {i for i, p in enumerate(t) if f.vtable(p)[0] == vt} != set(FLOAT_TENSORS):
        raise ValueError("Unexpected shared FLOAT32 tensor vtable")
    if f.field(t[718],1) is not None:
        raise ValueError("FLOAT32 Tensor type field unexpectedly present")
    if any(f.vtable(t[i])[0]!=vt or b[t[i]+FLOAT_VTABLE_TYPE_RESERVED_OFFSET]!=0 for i in FLOAT_TENSORS):
        raise ValueError("Tensor type reserved-byte invariant failed")
    for i in FLOAT_TENSORS:
        if f.obj(t[i],4) is None or f.vals(f.obj(t[i],4),2,"f"):
            raise ValueError(f"Tensor {i} not using expected empty quantization")
    if any(f.scalar(op[i],3,"b",0)!=0 or f.obj(op[i],4) is not None for i in FIRST_OPS):
        raise ValueError("Unexpected Dequantize builtin options")
    q0 = f.obj(t[717], 4)
    scales, zero = f.vals(q0, 2, "f"), f.vals(q0, 3, "q")
    if (not scales or len(scales) != 1 or len(zero) != 1 or not 0 < scales[0] < 1
            or not -128 <= zero[0] <= 127):
        raise ValueError("UNSUPPORTED_RESEARCH_GRAPH: invalid shared qparams")
    for ti in (749, 754, 760, 763):
        qi = f.obj(t[ti], 4)
        if (f.vals(qi, 2, "f"), f.vals(qi, 3, "q")) != (scales, zero):
            raise ValueError("UNSUPPORTED_RESEARCH_GRAPH: terminal qparams differ")
    # The value output of first TopK is dead; the second TopK values are live.
    uses = collections.Counter(v for o in op for v in (f.vals(o, 1) or []) if v >= 0)
    if uses[719] != 0 or uses[720] != 1 or uses[732] == 0:
        raise ValueError("UNSUPPORTED_RESEARCH_GRAPH: TopK consumer signature changed")
    return f, root, sg, t, op, codes, bufs, vt


def patch(source_bytes):
    b = bytearray(source_bytes)
    f, root, sg, t, op, codes, bufs, vt = assert_source(b)
    changed = []
    # Change TensorType defaulted FLOAT32 -> signed INT8 in ONLY the seven
    # tables sharing this vtable. The object already has byte padding +5.
    s_u16(b, vt + 4 + 2 * 1, FLOAT_VTABLE_TYPE_RESERVED_OFFSET)
    for i in FLOAT_TENSORS:
        s_u8(b, t[i]+FLOAT_VTABLE_TYPE_RESERVED_OFFSET, TENSOR_TYPE_INT8)
        changed.append(f"tensor {i} INT8")
    # Reuse exactly the pre-existing and proven qparams on tensor 717.
    qpos = f.obj(t[717], 4)
    for i in FLOAT_TENSORS:
        if i==737:
            continue
        ptr_to(b, f.field(t[i],4), qpos)
    # Cast INT64 class index 0 to int8 physical 0, then quantize it with
    # scale=1, zero_point=0 -> output scale 0.007866919, zero_point=-91.
    class_q = append_class_quantization(b)
    ptr_to(b, f.field(t[737],4), class_q)
    changed.append("class int8 qparams 1/0")
    # Reinterpret the unused old keypoint dequantized tensor as class output.
    shape_pos, n = f.vector(t[750],0)
    if n!=3:
        raise ValueError("Tensor750 shape vector length mismatch")
    s_i32(b, shape_pos+4+2*4, 1)
    changed.append("tensor 750 shape [1,300,1]")
    # Convert four Dequantize nodes to quantize nodes. The nodes are already
    # at valid topological positions, and Quantize INT8->INT8 is used by this
    # very model at ops 159/195/408. No operators need to be inserted.
    q_opcode = f.scalar(op[471],0)
    for i in FIRST_OPS:
        p=f.field(op[i],0)
        if p is None:raise ValueError(f"Operator {i} missing opcode_index")
        s_u32(b,p,q_opcode)
        changed.append(f"op {i} Dequantize -> Quantize")
    # Rewire op458 to remap the integer class ID to output qparams.
    v,n=f.vector(op[458],1)
    if n!=1:raise ValueError("op458 unexpected inputs count")
    s_i32(b,v+4,737)
    changed.append("op 458 input 749 -> 737 (class mapping)")
    # Compose [box(4), conf(1), class(1), keypoints(8)] in original order.
    v,n=f.vector(op[470],1)
    if n!=4:raise ValueError("op470 unexpected inputs count")
    s_i32(b,v+4+2*4,750) # class quantized
    s_i32(b,v+4+3*4,749) # keypoints still from original tensor
    changed.append("op 470 input -> [761,755,750,749]")
    return bytes(b), changed


def audit(b):
    f,root,sg,t,op,codes,bufs=analyze(b)
    types=collections.Counter(f.scalar(x,1,"b") for x in t)
    if types.get(0,0) or any(f.scalar(t[i],1,"b")!=9 for i in FLOAT_TENSORS):
        raise ValueError("Some FLOAT32 tensors still present")
    reference_q = f.obj(t[717], 4)
    reference_scales = f.vals(reference_q, 2, "f")
    reference_zero = f.vals(reference_q, 3, "q")
    for i in [718,719,750,755,761,762]:
        q=f.obj(t[i],4)
        if (f.vals(q,2,"f"),f.vals(q,3,"q"))!=(reference_scales, reference_zero):
            raise ValueError(f"Wrong INT8 qparams on tensor {i}")
    if (f.vals(f.obj(t[737],4),2,"f"),f.vals(f.obj(t[737],4),3,"q"))!=([1.0],[0]):
        raise ValueError("Wrong INT8 class qparams")
    if f.vals(t[750],0)!=[1,300,1]:raise ValueError("Wrong class output shape")
    for i in FIRST_OPS:
        if f.scalar(op[i],0)!=f.scalar(op[471],0):raise ValueError("Quantize opcode not updated")
    if f.vals(op[458],1)!=[737] or f.vals(op[470],1)!=[761,755,750,749]:
        raise ValueError("Rewiring not applied")
    if f.vals(sg,1)!=[0] or f.vals(sg,2)!=[763]:
        raise ValueError("Changed public I/O")
    op_types=collections.Counter(f.scalar(codes[f.scalar(o,0)],3,"i",f.scalar(codes[f.scalar(o,0)],0,"b")) for o in op)
    return {"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b),"tensor_types":dict(types),
            "float32_tensor_indices":[i for i,el in enumerate(t) if f.scalar(el,1,"b")==0],
            "operator_counts":dict(op_types),"input_shape":f.vals(t[0],0),"output_shape":f.vals(t[763],0),
            "first_topk_value_dtype":"INT8", "last_concat_dtype":"INT8", "nc":1,"max_det":300,
            "static_only":True, "runtime_precision":"UNVERIFIED","output_equivalence":"UNVERIFIED"}




def _test_inputs(np):
    """Eight reproducible synthetic signed INT8 inputs, never holdout images."""
    shape = (1, 512, 512, 3)
    cases = [(name, np.full(shape, value, dtype=np.int8)) for name, value in (
        ("black", -128), ("midgray", 0), ("white", 127),
    )]
    gradient = np.linspace(-128, 127, 512, dtype=np.int16).astype(np.int8)
    cases.append(("gradient_x", np.broadcast_to(gradient[None, None, :, None], shape).copy()))
    stripes = np.where((np.arange(512) // 16) % 2 == 0, -128, 127).astype(np.int8)
    cases.append(("stripe_y", np.broadcast_to(stripes[None, :, None, None], shape).copy()))
    checker = np.where((np.arange(512)[:, None] // 32 + np.arange(512)[None, :] // 32) % 2 == 0,
                       -128, 127).astype(np.int8)
    cases.append(("checkerboard", np.broadcast_to(checker[None, :, :, None], shape).copy()))
    rng = np.random.default_rng(20261012)
    for i in (1, 2):
        cases.append((f"noise_{i}", rng.integers(-128, 128, size=shape, dtype=np.int16).astype(np.int8)))
    return cases


def _compare_cpu(original: Path, candidate: Path):
    """Exact output equivalence on CPU; neither device precision nor quality proof."""
    import numpy as np
    import tensorflow as tf

    if tf.__version__.split("+")[0] != "2.19.0":
        raise RuntimeError("RESEARCH_INT8_UNSUPPORTED_RUNTIME: expected TensorFlow 2.19.0")

    def interpreter(path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")
            warnings.filterwarnings("ignore", message=".*experimental_preserve_all_tensors.*")
            it = tf.lite.Interpreter(model_path=str(path), num_threads=2,
                                     experimental_preserve_all_tensors=True)
        it.allocate_tensors()
        return it

    x, y = interpreter(original), interpreter(candidate)
    xi, xo = x.get_input_details(), x.get_output_details()
    yi, yo = y.get_input_details(), y.get_output_details()
    if len(xi) != 1 or len(yi) != 1 or len(xo) != 1 or len(yo) != 1:
        raise RuntimeError("RESEARCH_INT8_IO_MISMATCH: expected exactly 1 input/1 output")
    for a, b, shape in ((xi[0], yi[0], (1, 512, 512, 3)),
                        (xo[0], yo[0], (1, 300, 14))):
        if (tuple(a["shape"]) != shape or tuple(b["shape"]) != shape
                or a["dtype"] != np.int8 or b["dtype"] != np.int8
                or a["quantization"] != b["quantization"]):
            raise RuntimeError("RESEARCH_INT8_IO_MISMATCH: shapes, dtype or scale differ")

    comparisons = []
    original_output_hashes = set()
    for label, data in _test_inputs(np):
        x.set_tensor(int(xi[0]["index"]), data)
        x.invoke()
        original_output = x.get_tensor(int(xo[0]["index"])).copy()
        y.set_tensor(int(yi[0]["index"]), data)
        y.invoke()
        candidate_output = y.get_tensor(int(yo[0]["index"])).copy()
        original_hash = hashlib.sha256(original_output.tobytes()).hexdigest()
        candidate_hash = hashlib.sha256(candidate_output.tobytes()).hexdigest()
        original_output_hashes.add(original_hash)
        if not np.array_equal(original_output, candidate_output):
            diffs = int(np.count_nonzero(original_output != candidate_output))
            raise RuntimeError(f"RESEARCH_INT8_CPU_MISMATCH: {label}, {diffs} differing values")
        comparisons.append({"name": label, "input_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                            "original_output_sha256": original_hash, "candidate_output_sha256": candidate_hash,
                            "values_compared": int(original_output.size), "differing_values": 0})
    if len(original_output_hashes) < 2:
        raise RuntimeError("RESEARCH_INT8_UNINFORMATIVE_CPU_TEST: all outputs identical")
    return {"status": "CPU_EXACT_EQ_PASS_NOT_ANDROID_PROOF", "tensorflow": tf.__version__,
            "cases_run": len(comparisons), "total_values_compared": sum(c["values_compared"] for c in comparisons),
            "cases": comparisons, "android_integer_execution": "UNVERIFIED", "final_test": "NOT_USED",
            "real_scene_equivalence": "NOT_TESTED", "accuracy_with_gt": "NOT_TESTED"}


def recover_research_int8(original: Path, *, report: dict, diagnostics_root: Path) -> Path:
    """Recover a matched MT one-class Pose from MIXED, fail closed on ANY doubt.

    The calling exporter MUST gate this to the exploratory plate INT8 path.
    On success returns a new file in the same temporary working directory as
    the original. The existing exporter still applies its original precision
    inspector before packaging; strict mode never calls this function.
    """
    from .tflite_precision import analyze_tflite_bytes

    original = Path(original)
    if report.get("classification") != "MIXED" or report.get("issues"):
        raise ValueError("RESEARCH_INT8_UNSUPPORTED_SOURCE: expected clean MIXED graph")
    source = original.read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    if report.get("artifact_sha256") != source_sha or report.get("artifact_bytes") != len(source):
        raise ValueError("RESEARCH_INT8_SOURCE_CHANGED: forensic source hash/length mismatch")

    candidate_bytes, edits = patch(source)  # strict structural checks before touching a file
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    if (source_sha == REFERENCE_SOURCE_SHA and len(source) == REFERENCE_SOURCE_SIZE
            and candidate_sha != "1cc49f7c0ff72d56301c7ea560f7fa32b2e3f6915f28b4ac794c3a96bf7d115f"):
        raise ValueError("RESEARCH_INT8_REFERENCE_DRIFT: historical candidate checksum changed")
    simple_audit = audit(candidate_bytes)
    candidate_graph = analyze_tflite_bytes(candidate_bytes)
    if (candidate_graph.get("classification") != "INT8" or not candidate_graph.get("full_integer_graph")
            or candidate_graph.get("issues") or candidate_graph.get("artifact_sha256") != candidate_sha):
        raise ValueError("RESEARCH_INT8_STATIC_GATE_FAILED: candidate still not proven signed INT8")
    for sub in candidate_graph.get("subgraphs", []):
        if any(t.get("dtype") != "INT8" for key in ("inputs", "outputs") for t in sub.get(key, [])):
            raise ValueError("RESEARCH_INT8_IO_GATE_FAILED: public I/O is not signed INT8")

    suffix = uuid.uuid4().hex[:10]
    target = original.with_name(original.stem + f"_research_recovered_{suffix}.tflite")
    if target.exists():
        raise FileExistsError("RESEARCH_INT8_OUTPUT_COLLISION: refusing overwrite")
    with target.open("xb") as stream:
        stream.write(candidate_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    if hashlib.sha256(target.read_bytes()).hexdigest() != candidate_sha:
        raise IOError("RESEARCH_INT8_OUTPUT_NOT_VERIFIED: persisted bytes differ")

    equivalence = _compare_cpu(original, target)

    # Preserve original plus candidate and a complete CPU parity record before
    # allowing package publication. Directory lies under ignored local output/.
    diagnostics_root = Path(diagnostics_root)
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    evidence_dir = diagnostics_root / ("recovered_int8_" + uuid.uuid4().hex[:12])
    evidence_dir.mkdir(exist_ok=False)
    originals = (original, target)
    for src, dest_name, expected_sha in zip(originals, ("source_mixed.tflite", "candidate_int8.tflite"),
                                            (source_sha, candidate_sha)):
        dest = evidence_dir / dest_name
        with src.open("rb") as fin, dest.open("xb") as fout:
            shutil.copyfileobj(fin, fout, length=1024 * 1024)
            fout.flush()
            os.fsync(fout.fileno())
        if hashlib.sha256(dest.read_bytes()).hexdigest() != expected_sha:
            raise IOError("RESEARCH_INT8_EVIDENCE_COPY_MISMATCH: candidate/original not preserved")
    evidence = {
        "status": "EXPLORATORY_CPU_PARITY_ONLY",
        "source_sha256": source_sha, "source_bytes": len(source),
        "candidate_sha256": candidate_sha, "candidate_bytes": len(candidate_bytes),
        "source_signature": "KNOWN_REFERENCE" if source_sha == REFERENCE_SOURCE_SHA else "STRUCTURAL_MATCH_ONLY",
        "graph_audit": simple_audit, "static_classification": candidate_graph["classification"],
        "patch_steps": edits, "cpu_equivalence": equivalence,
        "limitations": ["NO_ANDROID_INT8_EXECUTION_PROOF", "NO_FINAL_TEST_INDEPENDENCE_PROOF",
                        "NO_REAL_SCENE_EQUIVALENCE_PROOF", "RESEARCH_ONLY_NOT_PRODUCTION_APPROVED"],
    }
    report_path = evidence_dir / "recovery_report.json"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                     dir=evidence_dir, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(evidence, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp_path, report_path)
    return target
