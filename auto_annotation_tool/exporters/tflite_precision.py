"""Conservative, static TFLite precision evidence. Never loads an interpreter.

INT8 means signed 8-bit data/weights with integer bias and structural tensors.
Float boundary quantize/dequantize, hybrid weights, and float fallback are MIXED.
Unknown operators/control flow/types or invalid quantizers are UNVERIFIED.
This is not a kernel/delegate execution certificate or a full TFLite verifier.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path

SCHEMA = "alpr.tflite_graph_precision.v1"
POLICY = "signed_int8_static_graph_v1"
MAX_BYTES = 512 * 1024 * 1024
MAX_ITEMS = 100_000

# Explicit static type signatures, not claims about Android kernel support.
WEIGHTED = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}
UNARY = {"RELU", "RELU6", "RELU_N1_TO_1", "LOGISTIC", "TANH", "SOFTMAX",
         "LEAKY_RELU", "ABS", "NEG", "EXP", "LOG", "SQRT", "RSQRT", "SQUARE",
         "AVERAGE_POOL_2D", "MAX_POOL_2D", "L2_NORMALIZATION"}
BINARY = {"ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM", "POW", "SQUARED_DIFFERENCE", "FLOOR_MOD"}
LAYOUT = {"RESHAPE", "TRANSPOSE", "SQUEEZE", "EXPAND_DIMS", "STRIDED_SLICE", "SLICE",
          "PAD", "MIRROR_PAD", "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR",
          "SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND", "SPACE_TO_DEPTH", "DEPTH_TO_SPACE",
          "TILE", "GATHER", "GATHER_ND", "BROADCAST_TO", "REVERSE_V2"}
REDUCE = {"MEAN", "SUM", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD"}
COMPARE = {"LESS", "LESS_EQUAL", "GREATER", "GREATER_EQUAL", "EQUAL", "NOT_EQUAL"}
STRUCTURAL = {"INT32", "INT64", "BOOL"}
DATA = {"FLOAT32", "INT8"}
KNOWN = WEIGHTED | UNARY | BINARY | LAYOUT | REDUCE | COMPARE | {
    "CONCATENATION", "PACK", "UNPACK", "SPLIT", "SPLIT_V", "SHAPE", "RANK", "SIZE",
    "ARG_MAX", "ARG_MIN", "TOPK_V2", "CAST", "QUANTIZE", "DEQUANTIZE", "SELECT", "SELECT_V2",
    "LOGICAL_AND", "LOGICAL_OR", "LOGICAL_NOT", "FILL", "BATCH_MATMUL", "PADV2",
}
# Bounded, explicit policy. Other versions need a policy revision and tests.
VERSIONS = {name: {1, 2} for name in KNOWN}
VERSIONS.update(CONV_2D={1, 2, 3, 4, 5, 6}, DEPTHWISE_CONV_2D={1, 2, 3, 4, 5, 6},
                FULLY_CONNECTED={1, 2, 3, 4, 5, 6, 7, 8, 9}, ADD={1, 2, 3, 4},
                MUL={1, 2, 3, 4, 5}, RESIZE_NEAREST_NEIGHBOR={1, 2, 3},
                BATCH_MATMUL={1, 2, 3, 4}, GATHER={1, 2, 3, 4, 5},
                STRIDED_SLICE={1, 2, 3, 4}, TRANSPOSE={1, 2, 3, 4, 5})
FLOAT_ONLY = {"EXP", "LOG", "SQRT", "RSQRT", "SQUARE", "POW", "DIV", "FLOOR_MOD"}


@lru_cache(maxsize=1)
def load_schema():
    """Load only the generated schema file, without importing TF/LiteRT runtime."""
    for dist_name, relative in (
        ("ai-edge-litert", "ai_edge_litert/schema_py_generated.py"),
        ("tensorflow", "tensorflow/lite/python/schema_py_generated.py"),
        ("tensorflow-intel", "tensorflow/lite/python/schema_py_generated.py"),
        ("tensorflow-cpu", "tensorflow/lite/python/schema_py_generated.py"),
    ):
        try:
            dist = importlib.metadata.distribution(dist_name)
            path = Path(dist.locate_file(relative))
            if not path.is_file():
                continue
            spec = importlib.util.spec_from_file_location("_alpr_tflite_schema", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except importlib.metadata.PackageNotFoundError:
            continue
    raise ImportError("Brak lokalnego schematu FlatBuffers TFLite (ai-edge-litert/TensorFlow).")


def report_digest(report: dict) -> str:
    payload = {k: v for k, v in report.items() if k != "analysis_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _finish(report):
    report["analysis_sha256"] = report_digest(report)
    return report


def analyze_tflite_file(path: Path) -> dict:
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Model TFLite przekracza limit 512 MiB.")
    return analyze_tflite_bytes(path.read_bytes())


def analyze_tflite_bytes(data: bytes) -> dict:
    report = {"schema": SCHEMA, "policy": POLICY, "artifact_sha256": hashlib.sha256(data).hexdigest(),
              "artifact_bytes": len(data), "classification": "UNVERIFIED", "scope": "static_graph_only",
              "execution_precision": "UNVERIFIED", "full_integer_graph": False,
              "issues": [], "subgraphs": [], "operator_counts": {}, "tensor_type_counts": {}}
    issues = report["issues"]
    try:
        if len(data) < 12 or len(data) > MAX_BYTES or data[4:8] != b"TFL3":
            raise ValueError("Nieprawidłowy identyfikator/rozmiar FlatBuffers TFL3.")
        if not 8 <= int.from_bytes(data[:4], "little") < len(data):
            raise ValueError("Nieprawidłowy offset korzenia FlatBuffers.")
        schema = load_schema()
        types = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
        opcodes = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
        model = schema.Model.GetRootAsModel(data, 0)
        if model.Version() != 3:
            raise ValueError("Nieobsługiwana wersja schematu modelu.")
        if not 1 <= model.SubgraphsLength() <= 100:
            raise ValueError("Nieprawidłowa liczba podgrafów.")
        if not 1 <= model.BuffersLength() <= MAX_ITEMS or model.OperatorCodesLength() > MAX_ITEMS:
            raise ValueError("Nieprawidłowa liczba buforów/operator codes.")
        all_modes = set()
        all_ops, all_types = Counter(), Counter()
        for si in range(model.SubgraphsLength()):
            sub = model.Subgraphs(si)
            if not 1 <= sub.TensorsLength() <= MAX_ITEMS or not 1 <= sub.OperatorsLength() <= MAX_ITEMS:
                raise ValueError("Pusty lub zbyt duży podgraf.")
            tensors = []
            for ti in range(sub.TensorsLength()):
                t = sub.Tensors(ti)
                if t.ShapeLength() > 32 or not 0 <= t.Buffer() < model.BuffersLength():
                    raise ValueError("Nieprawidłowy kształt lub indeks bufora tensora.")
                shape = [int(t.Shape(i)) for i in range(t.ShapeLength())]
                buf = model.Buffers(t.Buffer())
                q = t.Quantization()
                if q and max(q.ScaleLength(), q.ZeroPointLength()) > MAX_ITEMS:
                    raise ValueError("Zbyt wiele parametrów kwantyzacji.")
                scales = [float(q.Scale(i)) for i in range(q.ScaleLength())] if q else []
                zero = [int(q.ZeroPoint(i)) for i in range(q.ZeroPointLength())] if q else []
                dtype = types.get(t.Type(), "UNKNOWN_" + str(t.Type()))
                row = {"index": ti, "name": (t.Name() or b"").decode("utf-8", errors="replace"),
                       "dtype": dtype, "shape": shape, "buffer": int(t.Buffer()),
                       "constant": bool(buf.DataLength()), "buffer_bytes": int(buf.DataLength()),
                       "scale": [v if math.isfinite(v) else None for v in scales],
                       "zero_point": zero, "quantized_dimension": int(q.QuantizedDimension()) if q else 0}
                tensors.append(row)
                all_types[dtype] += 1
                prefix = f"subgraph={si} tensor={ti}: "
                if t.ShapeSignatureLength() > 32:
                    raise ValueError("Zbyt duża sygnatura kształtu tensora.")
                if t.IsVariable() or any(d < 0 for d in shape) or any(t.ShapeSignature(i) < 0 for i in range(t.ShapeSignatureLength())):
                    issues.append(prefix + "VARIABLE_OR_DYNAMIC_TENSOR")
                if q and q.DetailsType() != 0:
                    issues.append(prefix + "CUSTOM_QUANTIZATION_UNSUPPORTED")
                if getattr(buf, "Offset", lambda: 0)() or getattr(buf, "Size", lambda: 0)():
                    issues.append(prefix + "EXTERNAL_BUFFER_UNSUPPORTED")
                if dtype not in DATA | STRUCTURAL | {"UINT8"}:
                    issues.append(prefix + "UNSUPPORTED_DTYPE " + dtype)
                if dtype in {"INT8", "UINT8"} or scales or zero:
                    limit = (-128, 127) if dtype == "INT8" else (0, 255) if dtype == "UINT8" else (-2**31, 2**31-1)
                    axis = row["quantized_dimension"]
                    valid = (len(scales) == len(zero) > 0 and all(math.isfinite(v) and v > 0 for v in scales)
                             and all(limit[0] <= v <= limit[1] for v in zero)
                             and (len(scales) == 1 or (0 <= axis < len(shape) and shape[axis] == len(scales))))
                    if not valid or dtype == "FLOAT32":
                        issues.append(prefix + "INVALID_QUANTIZATION")
                if row["constant"] and dtype in {"FLOAT32", "INT8", "UINT8", "INT32", "INT64", "BOOL"}:
                    size = {"FLOAT32": 4, "INT8": 1, "UINT8": 1, "INT32": 4, "INT64": 8, "BOOL": 1}[dtype]
                    # Sparse buffers need their own verified interpretation.
                    if t.Sparsity() is not None or math.prod(shape) * size != row["buffer_bytes"]:
                        issues.append(prefix + "UNSUPPORTED_BUFFER_LAYOUT")

            def refs(values):
                result = []
                for index in values:
                    if not 0 <= index < len(tensors):
                        raise ValueError("Nieprawidłowy indeks tensora.")
                    result.append(tensors[index])
                return result

            inputs = refs([int(sub.Inputs(i)) for i in range(sub.InputsLength())])
            outputs = refs([int(sub.Outputs(i)) for i in range(sub.OutputsLength())])
            if not inputs or not outputs:
                issues.append(f"subgraph={si}: MISSING_IO")
            operators, used = [], set()
            for oi in range(sub.OperatorsLength()):
                op = sub.Operators(oi)
                if not 0 <= op.OpcodeIndex() < model.OperatorCodesLength():
                    raise ValueError("Nieprawidłowy indeks operator code.")
                code = model.OperatorCodes(op.OpcodeIndex())
                builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
                name = opcodes.get(builtin, "UNKNOWN_" + str(builtin))
                raw_inputs = [int(op.Inputs(i)) for i in range(op.InputsLength())]
                if any(i < -1 for i in raw_inputs) or max(op.InputsLength(), op.OutputsLength()) > MAX_ITEMS:
                    raise ValueError("Nieprawidłowe argumenty operatora.")
                if -1 in raw_inputs and not (name in WEIGHTED and raw_inputs == raw_inputs[:2] + [-1]):
                    issues.append(f"subgraph={si} op={oi}: UNSUPPORTED_OPTIONAL_OPERAND")
                ins = refs([i for i in raw_inputs if i >= 0])
                outs = refs([int(op.Outputs(i)) for i in range(op.OutputsLength())])
                used.update(t["index"] for t in ins + outs)
                prefix = f"subgraph={si} op={oi} {name}: "
                before = len(issues)
                if code.Version() not in VERSIONS.get(name, set()):
                    issues.append(prefix + "UNSUPPORTED_OPERATOR_OR_VERSION")
                if code.CustomCode() or op.CustomOptionsLength():
                    issues.append(prefix + "CUSTOM_OPERATOR_UNSUPPORTED")
                mode, weights, bias = _operator_mode(name, ins, outs, prefix, issues)
                if len(issues) == before:
                    all_modes.add(mode)
                all_ops[name] += 1
                operators.append({"index": oi, "operator": name, "version": int(code.Version()),
                                  "inputs": raw_inputs, "outputs": [t["index"] for t in outs],
                                  "input_types": [t["dtype"] for t in ins], "output_types": [t["dtype"] for t in outs],
                                  "activation_inputs": [t["index"] for t in ins if t["index"] not in weights + bias
                                                        and t["dtype"] in DATA | {"UINT8"}],
                                  "activation_outputs": [t["index"] for t in outs if t["dtype"] in DATA | {"UINT8"}],
                                  "weights": weights, "bias": bias, "mode": mode,
                                  "verified_signature": len(issues) == before})
            # All buffers/tensors and all subgraphs are accounted for, including dead branches.
            for t in tensors:
                if t["dtype"] in DATA and t["index"] not in used:
                    issues.append(f"subgraph={si} tensor={t['index']}: UNUSED_DATA_TENSOR")
            report["subgraphs"].append({"index": si, "inputs": inputs, "outputs": outputs,
                                        "tensors": tensors, "operators": operators})
        report["operator_counts"] = dict(sorted(all_ops.items()))
        report["tensor_type_counts"] = dict(sorted(all_types.items()))
        report["observed_modes"] = sorted(all_modes)
        if not issues:
            modes = all_modes - {"STRUCTURAL"}
            if modes == {"FP32"}:
                report["classification"] = "FP32"
            elif modes == {"INT8"}:
                boundary = {t["dtype"] for sg in report["subgraphs"] for t in sg["inputs"] + sg["outputs"]}
                report["classification"] = "INT8" if boundary <= {"INT8"} else "UNVERIFIED"
                if report["classification"] == "UNVERIFIED":
                    issues.append("UNSUPPORTED_INTEGER_BOUNDARY")
            elif "MIXED" in modes or {"FP32", "INT8"} <= modes:
                report["classification"] = "MIXED"
            else:
                issues.append("NO_VERIFIED_NUMERIC_COMPUTATION")
        report["full_integer_graph"] = report["classification"] == "INT8"
    except Exception as exc:
        issues.append(f"ANALYSIS_UNAVAILABLE: {type(exc).__name__}: {exc}")
        report["classification"] = "UNVERIFIED"
        report["full_integer_graph"] = False
    return _finish(report)


def _operator_mode(name, ins, outs, prefix, issues):
    weights, bias = [], []
    if not ins or not outs:
        issues.append(prefix + "MISSING_OPERAND")
        return "UNVERIFIED", weights, bias
    data_in, data_out = ins, outs
    if name in UNARY and (len(ins) != 1 or len(outs) != 1):
        issues.append(prefix + "INVALID_UNARY_SIGNATURE")
    if name in BINARY | COMPARE | {"BATCH_MATMUL"} and (len(ins) != 2 or len(outs) != 1):
        issues.append(prefix + "INVALID_BINARY_SIGNATURE")
    if name in {"CAST", "QUANTIZE", "DEQUANTIZE"} and (len(ins) != 1 or len(outs) != 1):
        issues.append(prefix + "INVALID_CONVERSION_SIGNATURE")
    if name in WEIGHTED:
        if len(ins) not in {2, 3} or len(outs) != 1:
            issues.append(prefix + "INVALID_WEIGHTED_SIGNATURE")
            return "UNVERIFIED", weights, bias
        weights = [ins[1]["index"]]
        bias = [ins[2]["index"]] if len(ins) == 3 else []
        data_in = ins[:2]
        kinds = {t["dtype"] for t in data_in + outs}
        if kinds == {"INT8"}:
            w = ins[1]
            if not w["constant"]:
                issues.append(prefix + "DYNAMIC_INTEGER_WEIGHTS_UNSUPPORTED")
            if any(w["zero_point"]) or (len(w["scale"]) > 1 and w["quantized_dimension"] != (3 if name == "DEPTHWISE_CONV_2D" else 0)):
                issues.append(prefix + "INVALID_WEIGHT_QUANTIZATION")
            if len(ins[0]["scale"]) != 1 or len(outs[0]["scale"]) != 1:
                issues.append(prefix + "ACTIVATION_MUST_BE_PER_TENSOR")
            if bias:
                b = ins[2]
                valid = (b["constant"] and b["dtype"] == "INT32" and b["scale"]
                         and len(b["scale"]) == len(w["scale"]) and not any(b["zero_point"]))
                if valid:
                    valid = all(x is not None and y is not None and ins[0]["scale"][0] is not None
                                and math.isclose(x, ins[0]["scale"][0] * y, rel_tol=1e-5, abs_tol=1e-12)
                                for x, y in zip(b["scale"], w["scale"]))
                if not valid:
                    issues.append(prefix + "INVALID_INTEGER_BIAS")
            if len(ins[0]["shape"]) != (2 if name == "FULLY_CONNECTED" else 4) or len(w["shape"]) != (2 if name == "FULLY_CONNECTED" else 4):
                issues.append(prefix + "UNSUPPORTED_WEIGHTED_SHAPE")
        elif bias and ins[2]["dtype"] != "FLOAT32":
            issues.append(prefix + "INVALID_FLOAT_BIAS")
    elif name in LAYOUT | REDUCE | {"ARG_MAX", "ARG_MIN", "TOPK_V2", "SHAPE", "RANK", "SIZE"}:
        data_in = ins[:1]
        if any(t["dtype"] not in STRUCTURAL for t in ins[1:]):
            issues.append(prefix + "INVALID_INDEX_DTYPE")
        if name in {"ARG_MAX", "ARG_MIN", "SHAPE", "RANK", "SIZE"}:
            data_out = []
            if any(t["dtype"] not in {"INT32", "INT64"} for t in outs):
                issues.append(prefix + "INVALID_INDEX_OUTPUT")
            if name in {"SHAPE", "RANK", "SIZE"}:
                return "STRUCTURAL", weights, bias
        elif name == "TOPK_V2":
            data_out = outs[:1]
            if len(outs) != 2 or outs[1]["dtype"] not in {"INT32", "INT64"}:
                issues.append(prefix + "INVALID_TOPK_SIGNATURE")
    elif name in {"SPLIT", "SPLIT_V"}:
        pos = 1 if name == "SPLIT" else 0
        data_in = ins[pos:pos+1]
        if not data_in or any(t["dtype"] not in {"INT32", "INT64"} for i, t in enumerate(ins) if i != pos):
            issues.append(prefix + "INVALID_SPLIT_SIGNATURE")
    elif name in {"SELECT", "SELECT_V2"}:
        data_in = ins[1:]
        if len(ins) != 3 or ins[0]["dtype"] != "BOOL":
            issues.append(prefix + "INVALID_SELECT_SIGNATURE")
    elif name == "PADV2":
        data_in = ins[:1] + ins[2:]
        if len(ins) != 3 or ins[1]["dtype"] not in {"INT32", "INT64"}:
            issues.append(prefix + "INVALID_PAD_SIGNATURE")
    elif name == "FILL":
        data_in = ins[1:]
        if len(ins) != 2 or ins[0]["dtype"] not in {"INT32", "INT64"}:
            issues.append(prefix + "INVALID_FILL_SIGNATURE")
    elif name in COMPARE:
        data_out = []
        if any(t["dtype"] != "BOOL" for t in outs):
            issues.append(prefix + "INVALID_COMPARE_OUTPUT")
    kinds = {t["dtype"] for t in data_in + data_out}
    if name in {"LOGICAL_AND", "LOGICAL_OR", "LOGICAL_NOT"}:
        if kinds != {"BOOL"} or len(ins) != (1 if name == "LOGICAL_NOT" else 2) or len(outs) != 1:
            issues.append(prefix + "INVALID_LOGICAL_SIGNATURE")
    if kinds == {"INT8"}:
        activations = [t for t in data_in + data_out if t["index"] not in weights]
        if any(len(t["scale"]) != 1 for t in activations):
            issues.append(prefix + "ACTIVATION_MUST_BE_PER_TENSOR")
        fixed = {"SOFTMAX": (1/256, -128), "LOGISTIC": (1/256, -128),
                 "TANH": (1/128, 0), "L2_NORMALIZATION": (1/128, 0)}
        if name in fixed:
            scale, zero = fixed[name]
            if any(t["scale"] != [scale] or t["zero_point"] != [zero] for t in data_out):
                issues.append(prefix + "INVALID_FIXED_OUTPUT_QUANTIZATION")
        same_scale = LAYOUT | {"AVERAGE_POOL_2D", "MAX_POOL_2D", "CONCATENATION", "PACK", "UNPACK", "SPLIT", "SPLIT_V", "PADV2"}
        if name in same_scale and activations:
            if any((t["scale"], t["zero_point"]) != (activations[0]["scale"], activations[0]["zero_point"]) for t in activations[1:]):
                issues.append(prefix + "INCONSISTENT_OPERATOR_QUANTIZATION")
    if kinds <= STRUCTURAL:
        return "STRUCTURAL", weights, bias
    if name in {"QUANTIZE", "DEQUANTIZE"} and kinds <= {"FLOAT32", "INT8", "UINT8"}:
        return ("MIXED" if "FLOAT32" in kinds else "INT8"), weights, bias
    if kinds <= DATA:
        mode = "MIXED" if len(kinds) == 2 else next(iter(kinds))
        if mode == "MIXED" and name not in WEIGHTED | {"CAST", "BATCH_MATMUL"}:
            issues.append(prefix + "UNSUPPORTED_MIXED_SIGNATURE")
        if name in FLOAT_ONLY and "INT8" in kinds:
            issues.append(prefix + "UNSUPPORTED_INTEGER_SIGNATURE")
        return ("FP32" if mode == "FLOAT32" else mode), weights, bias
    # CAST of index/shape data to real arithmetic remains float computation.
    # F4_RESEARCH_INT8_CLASS_INDEX_CAST_V1
    # CAST INT64->INT8 is a supported integer TFLite kernel.
    # This is the narrow one-class YOLO Pose ID=0, not a MIXED waiver.
    if (name == "CAST" and len(ins) == len(outs) == 1
            and ins[0]["dtype"] == "INT64" and outs[0]["dtype"] == "INT8"
            and ins[0]["shape"] == outs[0]["shape"] == [1, 300, 1]
            and outs[0]["scale"] == [1.0] and outs[0]["zero_point"] == [0]):
        return "INT8", weights, bias
    if name == "CAST" and kinds <= STRUCTURAL | {"FLOAT32"}:
        return "FP32", weights, bias
    issues.append(prefix + "UNSUPPORTED_TYPE_SIGNATURE " + ",".join(sorted(kinds)))
    return "UNVERIFIED", weights, bias


def manifest_precision_issues(report: dict, precision: str, input_spec: dict | None = None) -> list[str]:
    """Compare metadata with evidence; metadata never sets graph classification."""
    issues = []
    expected = {"fp32": "FP32", "int8": "INT8"}.get(str(precision).lower())
    if expected is None or report["classification"] != expected:
        issues.append(f"PRECISION_NOT_PROVEN: deklaracja={precision}, graf={report['classification']}")
    if input_spec and report["subgraphs"] and report["subgraphs"][0]["inputs"]:
        actual = report["subgraphs"][0]["inputs"][0]
        if input_spec.get("data_type") != actual["dtype"]:
            issues.append("INPUT_DTYPE_MISMATCH")
        if all(k in input_spec for k in ("height", "width", "channels", "layout")):
            expected_shape = ([1, input_spec["height"], input_spec["width"], input_spec["channels"]]
                              if input_spec["layout"] == "NHWC" else
                              [1, input_spec["channels"], input_spec["height"], input_spec["width"]])
            if expected_shape != actual["shape"]:
                issues.append("INPUT_SHAPE_MISMATCH")
        q = input_spec.get("quantization")
        if q is not None:
            try:
                valid = (len(actual["scale"]) == len(actual["zero_point"]) == 1
                         and math.isclose(float(q["scale"]), actual["scale"][0], rel_tol=1e-5, abs_tol=1e-12)
                         and q["zero_point"] == actual["zero_point"][0])
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                issues.append("INPUT_QUANTIZATION_MISMATCH")
    return issues


def compact_evidence(report: dict) -> dict:
    return {key: report[key] for key in ("schema", "policy", "artifact_sha256", "analysis_sha256",
            "classification", "scope", "execution_precision", "full_integer_graph")}
