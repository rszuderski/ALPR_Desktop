"""Z4LIB005: roles, checkpoint SHA and safe package-reuse UI contracts."""
from pathlib import Path
import hashlib
import json
import zipfile

import pytest

from auto_annotation_tool.gui.z4_mobile_artifact_registry import ArtifactRow
from auto_annotation_tool.gui.z4_mobile_artifact_library import MobileArtifactLibrary
from auto_annotation_tool.gui.z4_mobile_package_role_reuse import (
    ReusedMzPackageError, validate_existing_mz_package,
)

GUI = Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui/z4_model_export.py'


def package_fixture(tmp_path, *, role='character', task='detect', checkpoint_sha=None,
                    variants=None, source=True, calib=True, input_type='INT8',
                    imgsz=320, labels=None, class_count=None, research=True):
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'fixed-mz-checkpoint')
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    path = tmp_path / 'MZ-verified.alprmodel'
    classes = labels if labels is not None else ['A', '1', 'Z']
    if variants is None:
        variants = [dict(id='tflite-int8', runtime='tflite', precision='int8',
                         graph_precision={'evidence': True})]
    manifest = dict(
        schema='alpr.model.v1', role=role, task=task, model_id='MZ-TEST',
        labels=classes,
        input={'width': imgsz, 'height': imgsz, 'data_type': input_type},
        output={'class_count': len(classes) if class_count is None else class_count,
                'confidence_threshold': .25, 'iou_threshold': .45},
        variants=variants,
    )
    if source:
        manifest['source'] = {'checkpoint_sha256': checkpoint_sha or sha}
    if calib:
        manifest['calibration_evidence'] = {'file': 'calib.json',
                'mode':'exploratory' if research else 'strict',
                'status':'EXPLORATORY_ONLY' if research else 'READY'}
        if research:
            manifest['research_calibration'] = {'final_test_status': 'NOT_VERIFIED'}
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
    return path, checkpoint, manifest


def validate(path, checkpoint, *, expected_image_size=320, validator=None):
    def default_validator(path, *, require_precision=False):
        assert require_precision is True
    return validate_existing_mz_package(path, checkpoint, expected_image_size=expected_image_size,
                                        validate_package=validator or default_validator)


def test_mz_int8_reuse_full_validator_and_sha(tmp_path):
    path, checkpoint, _ = package_fixture(tmp_path)
    called=[]
    def callback(p, *, require_precision=False):
        called.append((p,require_precision))
        assert require_precision
    result=validate(path,checkpoint,validator=callback)
    assert called == [(path.resolve(),True)]
    assert result['precisions']==['int8']
    assert result['calibration_status']=='EXPLORATORY_ONLY'
    assert result['execution_precision_android']=='UNVERIFIED'
    assert result['package_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_mz_fp32_reuse_does_not_invent_int8(tmp_path):
    path, checkpoint,_ = package_fixture(tmp_path,variants=[dict(id='tflite-fp32',runtime='tflite',precision='fp32')],
                                       calib=False,input_type='FLOAT32')
    result=validate(path,checkpoint)
    assert result['precisions']==['fp32']
    assert result['calibration_status']=='NOT_APPLICABLE_FP32'

@pytest.mark.parametrize('modifier,pattern', [
    (dict(role='plate'), 'role=character'),
    (dict(task='pose'), 'task=detect'),
    (dict(checkpoint_sha='f'*64), 'SHA checkpointu'),
    (dict(imgsz=512), 'imgsz'),
    (dict(class_count=2), 'lista klas'),
    (dict(source=False), 'SHA checkpointu'),
    (dict(calib=False), 'dowodu kalibracji'),
    (dict(input_type='FLOAT32'), 'signed INT8'),
    (dict(variants=[dict(runtime='tflite', precision='int8')]), 'grafu INT8'),
    (dict(variants=[dict(runtime='onnx', precision='fp32')],calib=False), 'TFLite'),
])
def test_mz_fail_closed(tmp_path,modifier,pattern):
    path,checkpoint,_ = package_fixture(tmp_path,**modifier)
    with pytest.raises(ReusedMzPackageError, match=pattern):
        validate(path,checkpoint)


def test_mz_rejects_raw_research_tflite(tmp_path):
    file=tmp_path/'MZ_DAY_RAW_INT8_CLEAN_EXPLORATORY.tflite'
    file.write_bytes(b'INT8_RESEARCH_ONLY')
    checkpoint=tmp_path/'best.pt';checkpoint.write_bytes(b'x')
    with pytest.raises(ReusedMzPackageError,match='badawczy .tflite'):
        validate(file,checkpoint)


def test_mz_validator_failure_blocks(tmp_path):
    path,checkpoint,_=package_fixture(tmp_path)
    def refuse(*_args,**_kwargs):raise RuntimeError('STATIC_GRAPH_MIXED')
    with pytest.raises(ReusedMzPackageError,match='STATIC_GRAPH_MIXED'):
        validate(path,checkpoint,validator=refuse)


def test_mz_no_validator_blocks(tmp_path):
    path,checkpoint,_=package_fixture(tmp_path)
    with pytest.raises(ReusedMzPackageError,match='pełnego walidatora'):
        validate_existing_mz_package(path, checkpoint, expected_image_size=320, validate_package=None)


def test_mz_duplicate_manifest_blocks(tmp_path):
    path,checkpoint,manifest=package_fixture(tmp_path)
    with pytest.warns(UserWarning, match='Duplicate name'):
        with zipfile.ZipFile(path,'a') as archive:
            archive.writestr('manifest.json',json.dumps(manifest))
    with pytest.raises(ReusedMzPackageError,match='powielony'):
        validate(path,checkpoint)


def picker(tmp_path, *, actual_role='MZ', expected_role='MZ', kind='MODEL',
           pack_source=None, expected_imgsz=320, actual_imgsz=320, state='DISCOVERED_NOT_VERIFIED'):
    checkpoint=tmp_path/'selection.pt';checkpoint.write_bytes(b'target-checkpoint')
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest={'source':{'checkpoint_sha256': pack_source or sha},
              'input': {'width':actual_imgsz, 'height':actual_imgsz},
              'variants':[{'runtime':'tflite','precision':'int8'}]}
    row=ArtifactRow(path=str(tmp_path/'model.alprmodel'),name='model.alprmodel',schema='alpr.model.v1',
                    kind=kind, role=actual_role,model_id='MZ-TEST',variants='TFLITE/INT8',
                    size_bytes=20,modified_utc='2026-10-14',state=state,manifest=manifest)
    fake=object.__new__(MobileArtifactLibrary)
    fake.choose_role=expected_role; fake.choose_checkpoint=checkpoint
    fake.choose_image_size=expected_imgsz; fake.on_choose=lambda p: p
    return fake,row


def test_picker_accepts_only_pinned_single_role_model(tmp_path):
    browser,row=picker(tmp_path)
    assert browser._choice_problem(row)==''
    browser,row=picker(tmp_path,actual_role='MT')
    assert 'pojedynczy model MZ' in browser._choice_problem(row)
    browser,row=picker(tmp_path,kind='PACKAGE')
    assert 'pojedynczy model MZ' in browser._choice_problem(row)


def test_picker_sha_and_size_prechecks(tmp_path):
    browser,row=picker(tmp_path,pack_source='a'*64)
    assert 'SHA checkpointu' in browser._choice_problem(row)
    browser,row=picker(tmp_path,actual_imgsz=512)
    assert 'imgsz' in browser._choice_problem(row)
    browser,row=picker(tmp_path,state='REGISTERED_SHA_MISMATCH')
    assert 'integralności' in browser._choice_problem(row)


def test_picker_mt_rejects_not_int8(tmp_path):
    browser,row=picker(tmp_path,actual_role='MT',expected_role='MT')
    assert browser._choice_problem(row)==''
    row.manifest['variants']=[{'runtime':'tflite','precision':'fp32'}]
    assert 'INT8' in browser._choice_problem(row)


def test_source_contract_uses_authoritative_preflight_and_atomic_snapshots():
    source=GUI.read_text(encoding='utf-8-sig')
    for token in (
        'open_mobile_artifact_library(', 'on_choose=selected_package',
        'choose_checkpoint=Path(current_state["candidate"]["best_weights"])',
        'validate_existing_mt_package(', 'validate_existing_mz_package(',
        'plate_package=plate_package', 'character_package=character_package',
        'character_request=_build_state_request(',
        'with tempfile.TemporaryDirectory(prefix="alpr_mt_source_snapshot_"',
        'with tempfile.TemporaryDirectory(prefix="alpr_mz_source_snapshot_"',
        'reused_verified_mt', 'reused_verified_mz',
        'package_exporter.preflight(request)',
    ):
        assert token in source
    assert 'def export_with_pinned_mz(' in source
    assert 'package_exporter.export(\n                                    replace(active_request, character_package=snapshot)' in source


def test_source_syntax_and_library_is_readonly_on_selection():
    import ast
    ast.parse(GUI.read_text(encoding='utf-8-sig'))
    lib=Path(__file__).resolve().parents[1]/'auto_annotation_tool/gui/z4_mobile_artifact_library.py'
    source=lib.read_text(encoding='utf-8-sig')
    ast.parse(source)
    assert 'self.on_choose(Path(row.path))' in source
    assert 'def choose_for_export(self):' in source
    assert 'self.on_choose is not None' in source
