"""Z4LIB013: explicit, no-fallback MZ FP32/INT8 package selection for F4."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from auto_annotation_tool.gui.z4_mobile_package_role_reuse import (
    ReusedMzPackageError, validate_existing_mz_package,
)
from auto_annotation_tool.gui.z4_mobile_artifact_library import MobileArtifactLibrary
from auto_annotation_tool.gui.z4_mobile_artifact_registry import ArtifactRow
from auto_annotation_tool.gui.z4_mobile_conversion_journal import build_conversion_plan

GUI = Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui/z4_model_export.py'
LIB = Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui/z4_mobile_artifact_library.py'


def fake_model(tmp_path, *, precision='int8', dtype=None, variants=None, source_sha=None,
               role='character', task='detect', calibration=True):
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'freeze-mz-F4-checkpoint')
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    package = tmp_path / 'MZ_ready.alprmodel'
    if variants is None:
        variants = [{
            'runtime': 'tflite', 'precision': precision,
            'id': 'tflite-' + precision,
            'graph_precision': {'test_graph_evidence': 'not_real_graph_verification'},
        }]
    manifest = {
        'schema': 'alpr.model.v1', 'role': role, 'task': task,
        'model_id': 'MZ-DAY-TEST', 'labels': ['A', 'B', '0'],
        'source': {'checkpoint_sha256': source_sha or sha},
        'input': {'width': 320, 'height': 320, 'data_type': dtype or ('INT8' if precision == 'int8' else 'FLOAT32')},
        'output': {'class_count': 3, 'confidence_threshold': 0.25, 'iou_threshold': 0.45},
        'variants': variants,
    }
    if calibration:
        manifest['calibration_evidence'] = {
            'file': 'calibration_manifest.json', 'mode': 'exploratory', 'status': 'EXPLORATORY_ONLY',
        }
        manifest['research_calibration'] = {'final_test_status': 'NOT_VERIFIED'}
    with zipfile.ZipFile(package, 'w') as f:
        f.writestr('manifest.json', json.dumps(manifest))
    return package, checkpoint, manifest


def assert_package(tmp_path, *, expected='int8', validator=None, **kwargs):
    package, checkpoint, _ = fake_model(tmp_path, **kwargs)
    def mock_full_verification(path, *, require_precision=False):
        assert require_precision is True
        assert Path(path) == package.resolve()
    result = validate_existing_mz_package(
        package, checkpoint, expected_image_size=320,
        validate_package=validator or mock_full_verification,
        expected_precision=expected,
    )
    return result


@pytest.mark.parametrize('precision,input_type', [
    ('int8','INT8'),('fp32','FLOAT32'),
])
def test_explicit_precision_accepts_exact_one_variant(tmp_path,precision,input_type):
    result = assert_package(tmp_path,expected=precision,precision=precision,dtype=input_type,
                            calibration=precision=='int8')
    assert result['precision'] == precision
    assert result['precisions'] == [precision]
    assert result['execution_precision_android'] == 'UNVERIFIED'
    assert result['calibration_status'] == ('EXPLORATORY_ONLY' if precision=='int8' else 'NOT_APPLICABLE_FP32')


@pytest.mark.parametrize('expected,actual', [('int8','fp32'),('fp32','int8')])
def test_reject_silent_precision_fallback(tmp_path,expected,actual):
    with pytest.raises(ReusedMzPackageError,match='jeden wariant.*'+expected.upper()):
        assert_package(tmp_path,expected=expected,precision=actual)


def test_reject_mixed_package_for_both_single_precision_modes(tmp_path):
    mixed = [
        {'id':'tflite-int8', 'runtime':'tflite','precision':'int8','graph_precision': {'evidence':True}},
        {'id':'tflite-fp32', 'runtime':'tflite','precision':'fp32'},
    ]
    for desired in ('int8','fp32'):
        with pytest.raises(ReusedMzPackageError,match='paczki mieszanej'):
            assert_package(tmp_path,expected=desired,precision='int8',variants=mixed)


@pytest.mark.parametrize('precision,dtype', [('int8','FLOAT32'),('fp32','INT8'),('fp32','')])
def test_reject_wrong_input_dtype(tmp_path,precision,dtype):
    with pytest.raises(ReusedMzPackageError,match='wymaga wejścia'):
        assert_package(tmp_path,expected=precision,precision=precision,dtype=dtype or 'UNKNOWN')


def test_reject_invalid_expected_precision_even_for_valid_package(tmp_path):
    with pytest.raises(ReusedMzPackageError,match='jawnie'):
        assert_package(tmp_path, expected='auto')


def test_pinned_mz_still_calls_authoritative_precision_validator(tmp_path):
    calls=[]
    def reject(path, *, require_precision=False):
        calls.append(require_precision)
        raise RuntimeError('STATIC_GRAPH_MIXED')
    with pytest.raises(ReusedMzPackageError,match='STATIC_GRAPH_MIXED'):
        assert_package(tmp_path, expected='int8',validator=reject)
    assert calls==[True]


def test_pinned_mz_legacy_inspection_compatible(tmp_path):
    pkg, checkpoint, _=fake_model(tmp_path, precision='fp32',calibration=False)
    result=validate_existing_mz_package(pkg,checkpoint, expected_image_size=320,
                                        validate_package=lambda *_a,**_kw:None)
    assert result['precisions']==['fp32']
    assert result['precision']=='fp32'


def fake_picker(tmp_path, *, selected_precision='int8', actual_precision='int8', data_type=None,variants=None):
    checkpoint=tmp_path/'candidate.pt';checkpoint.write_bytes(b'pinned-source')
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if variants is None:
        variants=[{'runtime':'tflite','precision':actual_precision}]
    fake=object.__new__(MobileArtifactLibrary)
    fake.on_choose=lambda _:None
    fake.choose_role='MZ'
    fake.choose_checkpoint=checkpoint
    fake.choose_image_size=320
    fake.choose_mz_precision=selected_precision
    row=ArtifactRow(path=str(tmp_path/'MZ.alprmodel'),name='MZ.alprmodel',
                    schema='alpr.model.v1',kind='MODEL',role='MZ',model_id='MZ-DAY',
                    variants='TFLITE/'+actual_precision.upper(),size_bytes=123,modified_utc='2026-10-14',
                    state='DISCOVERED_NOT_VERIFIED',manifest={
                        'source':{'checkpoint_sha256':sha},
                        'input':{'width':320,'height':320, 'data_type':data_type or ('INT8' if actual_precision=='int8' else 'FLOAT32')},
                        'variants':variants,
                    })
    return fake,row


def test_picker_rejects_mz_precision_mismatch_and_mixed(tmp_path):
    for wanted, actual in [('int8','fp32'),('fp32','int8')]:
        picker,row=fake_picker(tmp_path,selected_precision=wanted,actual_precision=actual)
        assert wanted.upper() in picker._choice_problem(row)
    picker,row=fake_picker(tmp_path,selected_precision='int8',variants=[
        {'runtime':'tflite','precision':'int8'},{'runtime':'tflite','precision':'fp32'}])
    assert 'pojedynczy wariant' in picker._choice_problem(row)


@pytest.mark.parametrize('precision', ['int8','fp32'])
def test_picker_accepts_matching_precision(tmp_path, precision):
    picker,row=fake_picker(tmp_path,selected_precision=precision,actual_precision=precision)
    assert picker._choice_problem(row)==''


def test_reuse_not_a_conversion_in_journal(tmp_path):
    pkg=tmp_path/'MZ_READY.alprmodel'
    req=SimpleNamespace(destination=tmp_path/'F4.alprmodel',
                        vehicle_request=None, plate_request=None, character_request=None,
                        vehicle_package=None, plate_package=tmp_path/'MT_READY.alprmodel',
                        character_package=pkg)
    assert build_conversion_plan(req,parent_operation_id='Z4LIB013TEST')==[]


def test_z4_contracts_pin_selected_mz_precision_from_ui_to_preflight(tmp_path):
    text=GUI.read_text(encoding='utf-8-sig')
    ast.parse(text)
    lib=LIB.read_text(encoding='utf-8-sig');ast.parse(lib)
    for token in (
        '"reuse_mz_precision": tk.StringVar(value="int8")',
        'variable=state["reuse_mz_precision"]',
        'choose_mz_precision=(current_state["reuse_mz_precision"].get() if marker == "MZ" else "int8")',
        'expected_precision=str(char_state["reuse_mz_precision"].get()).lower()',
        'mz_precision = verified_existing_mz["precision"]',
        'quantizations=[mz_precision]',
        '"reuse_mz_precision"].trace_add("write", _mark_executor_dirty)',
        'character_package=character_package',
        'with tempfile.TemporaryDirectory(prefix="alpr_mz_source_snapshot_"',
    ):
        assert token in text
    assert 'TFLite (precyzja z manifestu)' not in text
    assert 'choose_mz_precision=choose_mz_precision' in lib
    assert 'def _choice_problem(self, row)' in lib
