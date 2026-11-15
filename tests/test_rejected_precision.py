"""SYNTHETIC graph bytes only. Tests production rejection code; no converter or inference."""
import hashlib, json, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import flatbuffers
import numpy as np
import pytest
from auto_annotation_tool.exporters import mobile_model_exporter as mobile
from auto_annotation_tool.exporters import tflite_precision as precision
from tflite_precision_fixture import graph_bytes


def signed_io_float_island():
    s = precision.load_schema()
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(graph_bytes('int8'), 0))
    sg = model.subgraphs[0]
    for name, dtype in [('float_dequant', s.TensorType.FLOAT32), ('float_activation', s.TensorType.FLOAT32),
                        ('signed_result', s.TensorType.INT8)]:
        t = s.TensorT(); t.name = name; t.shape = np.array([1,2,2,2], dtype=np.int32); t.type = dtype; t.buffer = 0
        if dtype == s.TensorType.INT8:
            q = s.QuantizationParametersT(); q.scale = np.array([.5],dtype=np.float32)
            q.zeroPoint = np.array([0],dtype=np.int64); t.quantization = q
        sg.tensors.append(t)
    for name, src, dst in [('DEQUANTIZE',3,4),('LOGISTIC',4,5),('QUANTIZE',5,6)]:
        code = s.OperatorCodeT(); code.builtinCode = getattr(s.BuiltinOperator,name)
        code.deprecatedBuiltinCode = min(127,code.builtinCode); code.version = 1
        op = s.OperatorT(); op.opcodeIndex = len(model.operatorCodes)
        op.inputs = np.array([src],dtype=np.int32); op.outputs = np.array([dst],dtype=np.int32)
        model.operatorCodes.append(code); sg.operators.append(op)
    sg.outputs = np.array([6],dtype=np.int32)
    builder = flatbuffers.Builder(2048); builder.Finish(model.Pack(builder),file_identifier=b'TFL3')
    return bytes(builder.Output())



@pytest.mark.parametrize('kind,classification', [('hybrid','MIXED'),('float_boundary','MIXED'),
    ('float_island','MIXED'),('custom','UNVERIFIED'),('flex','UNVERIFIED'),('unknown','UNVERIFIED'),
    ('bad_quant','UNVERIFIED'),('fp32','FP32'),('invalid','UNVERIFIED')])
def test_rejection_retains_exact_bytes_and_complete_private_safe_report(tmp_path,kind,classification):
    data = (graph_bytes('mixed') if kind=='hybrid' else graph_bytes('int8',float_boundary=True) if kind=='float_boundary'
        else signed_io_float_island() if kind=='float_island' else graph_bytes('int8',opcode='CUSTOM') if kind in {'custom','flex'}
        else graph_bytes('int8',opcode='IF') if kind=='unknown' else graph_bytes('int8',bad_quant=0) if kind=='bad_quant'
        else b'SYNTHETIC invalid graph' if kind=='invalid' else graph_bytes('fp32'))
    if kind=='flex':
        schema=precision.load_schema(); model=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(data,0))
        model.operatorCodes[0].customCode=b'FlexSyntheticOnly'
        builder=flatbuffers.Builder(2048);builder.Finish(model.Pack(builder),file_identifier=b'TFL3');data=bytes(builder.Output())
    full=tmp_path/'model_full_integer_quant.tflite';full.write_bytes(data)
    checkpoint=tmp_path/'weights.pt';checkpoint.write_bytes(b'SYNTHETIC never loaded')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'config':{'run_id':'SYNTHETIC'},
        'input_fingerprint':'a'*64,'schema':'alpr.int8_calibration.exploratory.v2','private_images':'DO_NOT_RETAIN'}))
    req=mobile.MobileExportRequest(checkpoint=checkpoint,destination=tmp_path/'NEVER.alprmodel',role='plate',
        formats=('litert',),image_size=512,calibration_manifest=manifest,calibration_mode='exploratory',
        metadata={'private_GT':'DO_NOT_RETAIN'})
    sink=tmp_path/'persistent'
    with patch.object(mobile.MobileModelExporter,'_precision_diagnostics_root',return_value=sink):
        with pytest.raises(mobile.MobileExportError,match='FULL_INT8_OUTPUT_NOT_PROVEN') as fail:
            mobile.MobileModelExporter._exploratory_signed_output(tmp_path/'model_int8.tflite',request=req)
    path,=sink.glob('*/rejected_precision.json');raw=path.read_text(encoding='utf-8');e=json.loads(raw)
    assert str(path) in str(fail.value) and 'DO_NOT_RETAIN' not in raw and str(tmp_path) not in raw
    assert e['graph']==precision.analyze_tflite_bytes(data) and e['graph']['classification']==classification
    assert (path.parent/'rejected_model.tflite').read_bytes()==data==full.read_bytes()
    assert e['source_converted_file_sha256']==e['retained_tflite_sha256']==hashlib.sha256(data).hexdigest()
    assert e['source_converted_file_bytes']==e['retained_tflite_bytes']==len(data)
    assert e['diagnostic_only'] and e['publication_status']=='NOT_PUBLISHED' and e['byte_verification']=='MATCH'
    assert e['execution_precision']=='UNVERIFIED' and e['status']=='INT8_NOT_PROVEN'
    assert e['request']['run_id']=='SYNTHETIC' and e['request']['calibration_input_fingerprint']=='a'*64
    assert e['request']['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert e['request']['calibration_manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert not req.destination.exists() and not list(path.parent.glob('*.tmp'))
    assert e['diagnosis']['converter_root_cause']=='NOT_IDENTIFIED'
    if kind=='float_island':
        assert e['diagnosis']['signed_int8_io']
        assert {o['operator'] for o in e['non_integer_operators']}=={'DEQUANTIZE','LOGISTIC','QUANTIZE'}
        assert [o['operator'] for o in e['diagnosis']['float_compute_operators']]==['LOGISTIC']
    if kind=='float_boundary':assert e['diagnosis']['non_int8_boundary_tensors']
    if kind=='flex':assert e['diagnosis']['custom_codes']['operators'][0]['flex_prefix_observed']


def test_int8_success_has_no_diagnostic_side_effect(tmp_path):
    from auto_annotation_tool.exporters import rejected_precision as diag
    full=tmp_path/'model_full_integer_quant.tflite';full.write_bytes(graph_bytes('int8'))
    with patch.object(diag,'write_rejected_precision') as write:
        assert mobile.MobileModelExporter._exploratory_signed_output(tmp_path/'model_int8.tflite')==full
        write.assert_not_called()


@pytest.mark.parametrize('missing', [True,False])
def test_unavailable_or_unknown_output_is_not_claimed_retained(tmp_path,missing):
    sink=tmp_path/'persistent'
    with patch.object(mobile.MobileModelExporter,'_precision_diagnostics_root',return_value=sink):
        with pytest.raises(mobile.MobileExportError,match='FULL_INT8_OUTPUT_NOT_AVAILABLE' if missing else 'FULL_INT8_OUTPUT_NOT_VERIFIED'):
            mobile.MobileModelExporter._exploratory_signed_output(tmp_path/('model_int8.tflite' if missing else 'unexpected.tflite'))
    assert not sink.exists()


@pytest.mark.parametrize('fault', ['directory','copy','fsync','json','rename','changed_source'])
def test_write_failures_keep_original_refusal_without_committed_report(tmp_path,fault,monkeypatch):
    import errno
    from auto_annotation_tool.exporters import rejected_precision as diag
    full=tmp_path/'model_full_integer_quant.tflite';full.write_bytes(signed_io_float_island())
    sink=tmp_path/'persistent';monkeypatch.setattr(mobile.MobileModelExporter,'_precision_diagnostics_root',staticmethod(lambda:sink))
    def failed(*a,**kw):raise OSError(errno.ENOSPC,'SYNTHETIC disk full')
    if fault=='directory':sink.write_text('existing file, never overwrite')
    elif fault=='copy':
        original=Path.open
        def denied(self,*a,**kw):
            if self.name=='rejected_model.tflite':raise PermissionError('synthetic denied')
            return original(self,*a,**kw)
        monkeypatch.setattr(Path,'open',denied)
    elif fault=='fsync':monkeypatch.setattr(diag.os,'fsync',failed)
    elif fault=='json':monkeypatch.setattr(diag.json,'dump',failed)
    elif fault=='rename':monkeypatch.setattr(diag.os,'rename',failed)
    else:
        original=mobile.analyze_tflite_file
        def changed(path):
            report=original(path);full.write_bytes(b'changed after analysis');return report
        monkeypatch.setattr(mobile,'analyze_tflite_file',changed)
    with pytest.raises(mobile.MobileExportError,match='FULL_INT8_OUTPUT_NOT_PROVEN: graf=MIXED.*DIAGNOSTICS_WRITE_FAILED') as fail:
        mobile.MobileModelExporter._exploratory_signed_output(tmp_path/'model_int8.tflite')
    assert 'Diagnostyka:' not in str(fail.value)
    assert not list(sink.glob('*/rejected_precision.json'))
    if fault=='directory':assert sink.read_text()=='existing file, never overwrite'


def test_nested_cleanup_and_concurrent_rejections_never_overwrite(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    sink=tmp_path/'persistent';data=signed_io_float_island()
    with patch.object(mobile.MobileModelExporter,'_precision_diagnostics_root',return_value=sink):
        with tempfile.TemporaryDirectory(dir=tmp_path,prefix='outer_') as outer:
            with tempfile.TemporaryDirectory(dir=outer,prefix='inner_') as inner:
                full=Path(inner)/'model_full_integer_quant.tflite';full.write_bytes(data)
                def reject(_):
                    with pytest.raises(mobile.MobileExportError):
                        mobile.MobileModelExporter._exploratory_signed_output(Path(inner)/'model_int8.tflite')
                reject(0)
                first,=sink.glob('*/rejected_precision.json');original=first.read_bytes()
                with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(reject,range(2)))
        assert not Path(outer).exists() and first.read_bytes()==original
        reports=list(sink.glob('*/rejected_precision.json'));assert len(reports)==3
        assert all((x.parent/'rejected_model.tflite').read_bytes()==data for x in reports)
