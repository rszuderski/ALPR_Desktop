"""Z4LIB012: MT FP32 baseline reuse; INT8 still default. No conversions."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from auto_annotation_tool.gui.z4_mobile_artifact_library import MobileArtifactLibrary
from auto_annotation_tool.gui.z4_mobile_artifact_registry import ArtifactRow
from auto_annotation_tool.gui.z4_mobile_package_reuse import (
    ReusedMtPackageError, validate_existing_mt_package,
)
from auto_annotation_tool.gui.z4_mobile_conversion_journal import build_conversion_plan
from types import SimpleNamespace

GUI = Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui/z4_model_export.py'


def fixture(tmp_path, *, precision='fp32', input_type='FLOAT32', calib=False,
            role='plate', source=None, variants=1, manifest_dup=False):
    checkpoint=tmp_path/'day_best.pt'
    checkpoint.write_bytes(b'checkpoint-F4-frozen')
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    package=tmp_path/'ready_mt.alprmodel'
    manifest={
        'schema': 'alpr.model.v1', 'role': role, 'task': 'pose',
        'model_id':'MT-DAY-MOCK', 'labels':['plate'],
        'source':{'checkpoint_sha256':source or sha},
        'input':{'width':512,'height':512,'data_type':input_type},
        'output':{'class_count':1,'keypoint_count':4,'confidence_threshold':.25,'iou_threshold':.45},
        'variants':[{'id':f'tflite-{precision}', 'runtime':'tflite','precision':precision,
                     'graph_precision':{'mock':True}} for _ in range(variants)]
    }
    if calib:
        manifest['calibration_evidence']={'file':'cal.json','mode':'exploratory','status':'EXPLORATORY_ONLY'}
        manifest['research_calibration']={'final_test_status':'NOT_VERIFIED'}
    with zipfile.ZipFile(package,'w') as out:
        out.writestr('manifest.json',json.dumps(manifest))
        if manifest_dup:
            with pytest.warns(UserWarning, match='Duplicate name'):
                out.writestr('manifest.json',json.dumps(manifest))
    return package,checkpoint,manifest


def check(package, checkpoint, *, expected_precision='int8', callback=None):
    def full_validator(path, *, require_precision=False):
        assert require_precision is True
    return validate_existing_mt_package(
        package,checkpoint,expected_image_size=512,
        expected_precision=expected_precision,validate_package=callback or full_validator,
    )


def test_fp32_reuse_is_explicit_and_full_validation_called(tmp_path):
    package,ckpt,_=fixture(tmp_path)
    calls=[]
    def full_validator(path, *, require_precision=False):
        calls.append((str(path),require_precision))
        assert require_precision
    proof=check(package,ckpt,expected_precision='fp32',callback=full_validator)
    assert calls==[(str(package.resolve()),True)]
    assert proof['precision']=='fp32'
    assert proof['format']=='tflite'
    assert proof['calibration_status']=='NOT_APPLICABLE_FP32'
    assert proof['package_sha256']==hashlib.sha256(package.read_bytes()).hexdigest()
    assert proof['execution_precision_android']=='UNVERIFIED'


def test_default_mt_validator_still_rejects_fp32(tmp_path):
    pkg,ckpt,_=fixture(tmp_path)
    with pytest.raises(ReusedMtPackageError,match='tflite/int8'):
        check(pkg,ckpt)


def test_existing_mt_int8_path_unchanged(tmp_path):
    pkg,ckpt,_=fixture(tmp_path,precision='int8',input_type='INT8',calib=True)
    assert check(pkg,ckpt)['precision']=='int8'
    with pytest.raises(ReusedMtPackageError,match='tflite/fp32'):
        check(pkg,ckpt,expected_precision='fp32')


@pytest.mark.parametrize('kwargs,pattern',[
    ({'input_type':'INT8'}, 'FLOAT32'),
    ({'source':'f'*64}, 'SHA checkpointu'),
    ({'variants':2}, 'jeden wariant'),
    ({'role':'character'}, 'role=plate'),
    ({'precision':'int8','input_type':'INT8','calib':True},'tflite/fp32'),
    ({'manifest_dup':True}, 'jeden poprawny manifest'),
])
def test_fp32_selection_blocks_invalid_model(tmp_path,kwargs,pattern):
    pkg,ckpt,_=fixture(tmp_path,**kwargs)
    with pytest.raises(ReusedMtPackageError,match=pattern):
        check(pkg,ckpt,expected_precision='fp32')


def test_fp32_selection_blocks_graph_verifier_failure(tmp_path):
    pkg,ckpt,_=fixture(tmp_path)
    def reject(*_a,**_kw): raise RuntimeError('GRAPH_NOT_FP32')
    with pytest.raises(ReusedMtPackageError,match='GRAPH_NOT_FP32'):
        check(pkg,ckpt,expected_precision='fp32',callback=reject)


def test_invalid_precision_value_fail_closed(tmp_path):
    pkg,ckpt,_=fixture(tmp_path)
    with pytest.raises(ReusedMtPackageError,match='precyzja'):
        check(pkg,ckpt,expected_precision='auto')


def picker(tmp_path,*, selected='fp32', actual='fp32', dtype='FLOAT32'):
    pkg,ckpt,manifest=fixture(tmp_path, precision=actual,input_type=dtype)
    row=ArtifactRow(str(pkg),pkg.name,'alpr.model.v1','MODEL','MT','MT-DAY',
                    f'TFLITE/{actual.upper()}',pkg.stat().st_size,'2026-10-14T14:00:00',
                    'DISCOVERED_NOT_VERIFIED',manifest)
    owner=object.__new__(MobileArtifactLibrary)
    owner.choose_role='MT'
    owner.choose_checkpoint=ckpt
    owner.choose_image_size=512
    owner.choose_mt_precision=selected
    owner.on_choose=lambda _: None
    return owner,row


def test_library_fp32_choice_and_int8_nonfallback(tmp_path):
    obj,row=picker(tmp_path)
    assert obj._choice_problem(row)==''
    obj.choose_mt_precision='int8'
    assert 'TFLite INT8' in obj._choice_problem(row)


def test_library_fp32_requires_float32_tensor(tmp_path):
    obj,row=picker(tmp_path,dtype='INT8')
    assert 'FLOAT32' in obj._choice_problem(row)


def test_reused_mt_fp32_creates_no_conversion_plan(tmp_path):
    pkg,_,_=fixture(tmp_path)
    req=SimpleNamespace(destination=tmp_path/'result.alprmodel',
        vehicle_request=None,plate_request=None,character_request=None,
        vehicle_package=None,plate_package=pkg,character_package=pkg)
    assert build_conversion_plan(req,parent_operation_id='F4TEST123456') == []


def test_ui_routes_precision_to_library_validator_snapshot_and_no_reconversion():
    text=GUI.read_text(encoding='utf-8-sig')
    assert '"reuse_mt_precision": tk.StringVar(value="int8")' in text
    assert 'value="fp32"' in text
    assert 'choose_mt_precision=(current_state["reuse_mt_precision"].get() if marker == "MT" else "int8")' in text
    assert 'expected_precision=str(plate_state["reuse_mt_precision"].get()).lower()' in text
    assert 'mt_precision = verified_existing_mt["precision"]' in text
    assert '"calibration_mode", "reuse_mt_precision")' in text
    assert 'plate_package=plate_package' in text
    assert 'with tempfile.TemporaryDirectory(prefix="alpr_mt_source_snapshot_"' in text
