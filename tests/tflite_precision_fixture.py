"""Tiny serialized graphs, built from the schema without TF/conversion/inference."""
import flatbuffers
import numpy as np

from auto_annotation_tool.exporters.tflite_precision import load_schema


def graph_bytes(kind="fp32", *, opcode="CONV_2D", version=1, bad_quant=None,
                per_axis=False, extra_subgraph=False, float_boundary=False):
    s = load_schema()
    model = s.ModelT()
    model.version = 3
    model.buffers = []

    def buffer(data=b""):
        b = s.BufferT()
        b.data = np.frombuffer(data, dtype=np.uint8)
        model.buffers.append(b)
        return len(model.buffers) - 1

    empty = buffer()
    dims = [[1, 2, 2, 3], [2, 1, 1, 3], [2], [1, 2, 2, 2]]
    dtypes = ([s.TensorType.FLOAT32] * 4 if kind == "fp32" else
              [s.TensorType.INT8, s.TensorType.INT8, s.TensorType.INT32, s.TensorType.INT8] if kind == "int8" else
              [s.TensorType.FLOAT32, s.TensorType.INT8, s.TensorType.FLOAT32, s.TensorType.FLOAT32])
    tensors = []
    for i, (shape, dtype) in enumerate(zip(dims, dtypes)):
        t = s.TensorT()
        t.name = f"tensor_{i}"
        t.shape = np.array(shape, dtype=np.int32)
        t.type = dtype
        size = 1 if dtype == s.TensorType.INT8 else 4
        t.buffer = buffer(b"\x00" * (int(np.prod(shape)) * size)) if i in (1, 2) else empty
        if dtype in (s.TensorType.INT8, s.TensorType.INT32):
            q = s.QuantizationParametersT()
            vals = [0.5, 0.25] if per_axis and i in (1, 2) else [0.5]
            if i == 2:
                vals = [v * 0.5 for v in vals]
            q.scale = np.array(vals, dtype=np.float32)
            q.zeroPoint = np.zeros(len(vals), dtype=np.int64)
            q.quantizedDimension = 0
            if bad_quant == i:
                q.scale = np.array([0.0], dtype=np.float32)
            t.quantization = q
        tensors.append(t)
    op = s.OperatorT()
    op.opcodeIndex = 0
    op.inputs = np.array([0, 1, 2], dtype=np.int32)
    op.outputs = np.array([3], dtype=np.int32)
    if opcode in {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}:
        # Only CONV_2D fixtures need options for a real serialized conv signature.
        options = s.Conv2DOptionsT()
        options.strideW = options.strideH = 1
        options.dilationWFactor = options.dilationHFactor = 1
        op.builtinOptionsType = s.BuiltinOptions.Conv2DOptions
        op.builtinOptions = options
    code = s.OperatorCodeT()
    code.builtinCode = getattr(s.BuiltinOperator, opcode)
    code.deprecatedBuiltinCode = min(code.builtinCode, 127)
    code.version = version
    if opcode == "CUSTOM":
        code.customCode = "unknown_custom"
    model.operatorCodes = [code]
    sub = s.SubGraphT()
    sub.name = "synthetic_precision_test"
    sub.tensors = tensors
    sub.inputs = np.array([0], dtype=np.int32)
    sub.outputs = np.array([3], dtype=np.int32)
    sub.operators = [op]
    if float_boundary:
        t = s.TensorT()
        t.name = "float_boundary"
        t.shape = np.array(dims[0], dtype=np.int32)
        t.type = s.TensorType.FLOAT32
        t.buffer = empty
        tensors.append(t)
        qcode = s.OperatorCodeT()
        qcode.builtinCode = qcode.deprecatedBuiltinCode = s.BuiltinOperator.QUANTIZE
        qcode.version = 1
        model.operatorCodes.append(qcode)
        qop = s.OperatorT()
        qop.opcodeIndex = 1
        qop.inputs = np.array([4], dtype=np.int32)
        qop.outputs = np.array([0], dtype=np.int32)
        sub.operators.insert(0, qop)
        sub.inputs = np.array([4], dtype=np.int32)
    model.subgraphs = [sub]
    if extra_subgraph:
        import copy
        other = copy.deepcopy(sub)
        for t in other.tensors:
            t.type = s.TensorType.FLOAT32
            t.quantization = None
            if t.buffer:
                t.buffer = buffer(b"\x00" * (int(np.prod(t.shape)) * 4))
        model.subgraphs.append(other)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())
