"""Research v2 without human confirmation; all model/image bytes are synthetic."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from auto_annotation_tool.exporters import exploratory_calibration as exp, int8_calibration as strict
from auto_annotation_tool.exporters import mobile_model_exporter as mobile
from int8_calibration_fixture import fixture, write_json
import test_exploratory_calibration as legacy


def prepared_v2(tmp_path, role="plate", **overrides):
    f=fixture(tmp_path/'inputs',role,duplicate=True)
    cfg={k:v for k,v in f['config'].items() if k not in {'final_inventory','approved_by'}}
    cfg.update(checkpoint_sha256=strict.file_sha256(f['checkpoint']),**overrides)
    m=exp.build_manifest(**cfg);p=write_json(tmp_path/'research.json',m)
    req=mobile.MobileExportRequest(checkpoint=f['checkpoint'],destination=tmp_path/'unused.alprmodel',
        role=role,formats=('litert',),image_size=cfg['image_size'],quantizations=('int8',),
        calibration_manifest=p,calibration_mode='exploratory')
    return f,m,p,req


@pytest.mark.parametrize('role,size',[('plate',512),('character',320)])
def test_v2_400_samples_without_approval_keeps_strict_blocked(tmp_path,role,size):
    f=fixture(tmp_path/'inputs',role)
    for i in range(3,403):
        path=f['dataset']/f'images/train/{i}.png';Image.new('RGB',(20,10),(i%256,i//256,17)).save(path)
        digest=strict.file_sha256(path);source=digest if role=='plate' else hashlib.sha256(f'SYNTHETIC scene {i}'.encode()).hexdigest()
        f['rows'].append({'image_path':path.relative_to(f['dataset']).as_posix(),'split':'train',
            'provenance':{'source_image_id':'img-sha256-'+source},'group_id':'SYNTHETIC group '+str(i)})
        f['inv']['source_scene_hashes'].append({'sha256':source,'splits':['train']})
        f['inv']['artifact_hashes'].append({'sha256':digest,'split':'train','kind':'crop' if role=='character' else 'image'})
    f['inv']['sample_count']=len(f['inv']['artifact_hashes'])
    if role=='character':
        for ref in f['run']['training_dataset_snapshot']['manifests']:
            path=write_json(f['dataset']/ref['name'],{'items':f['rows']});ref['sha256']=strict.file_sha256(path)
    write_json(f['history'],{'runs':{'synthetic':f['run']}})
    cfg={k:v for k,v in f['config'].items() if k not in {'final_inventory','approved_by'}}
    cfg.update(image_size=size,limit=400,checkpoint_sha256=strict.file_sha256(f['checkpoint']))
    m=exp.build_manifest(**cfg);p=write_json(tmp_path/'v2.json',m)
    assert m['schema']==exp.SCHEMA and m['status']=='EXPLORATORY_ONLY' and len(m['samples'])==400
    forbidden={'approved_by','operator_confirmation','confirmation_fingerprint'}
    assert not forbidden.intersection(m) and not forbidden.intersection(m['config'])
    assert m['source_audit']['status']=='BLOCKED' and m['source_audit']['config']['approved_by']==''
    assert len(m['source_audit']['blockers'])==2 and m['execution_precision']=='UNVERIFIED'
    assert exp.validate_manifest(p,role=role,image_size=size)==m
    req=mobile.MobileExportRequest(checkpoint=f['checkpoint'],destination=tmp_path/'NEVER.alprmodel',
        role=role,image_size=size,formats=('litert',),calibration_manifest=p,calibration_mode='exploratory')
    assert mobile.MobileModelExporter()._calibration_for_request(req)==m
    with pytest.raises(mobile.MobileExportError):mobile.MobileModelExporter()._approved_calibration(req)
    assert not req.destination.exists()


@pytest.mark.parametrize('change', ['sample','checkpoint','run','metadata','inventory','eval','preprocessing','fingerprint','group','versions'])
def test_v2_rechecks_live_sources(tmp_path,change):
    with patch.object(legacy,'prepared',prepared_v2):legacy.test_changed_inputs_reject_live_validation(tmp_path,change)


@pytest.mark.parametrize('fault',['partial','unfrozen','missing_mz_binding','unknown_augmentation'])
def test_v2_rejects_incomplete_provenance(tmp_path,fault):
    with patch.object(legacy,'prepared',prepared_v2):legacy.test_incomplete_provenance_never_authorizes_research(tmp_path,fault)


@pytest.mark.parametrize('split',['val','test'])
def test_v2_rejects_holdout_split(tmp_path,split):
    with patch.object(legacy,'prepared',prepared_v2):legacy.test_yaml_cannot_redirect_research_to_holdout(tmp_path,split)


@pytest.mark.parametrize('damage',['mode','role','source_blocker','status','fingerprint','fake_operator','schema_to_strict'])
def test_v2_forged_declarations_do_not_bypass_controls(tmp_path,damage):
    _,m,p,req=prepared_v2(tmp_path)
    if damage=='mode':req=replace(req,calibration_mode='strict')
    elif damage=='role':m['role']='vehicle'
    elif damage=='source_blocker':
        m['source_audit']['blockers'].append('SOURCE_SHA_ERROR')
        m['source_audit']['input_fingerprint']=strict.fingerprint({k:v for k,v in m['source_audit'].items() if k!='input_fingerprint'})
    elif damage=='status':m['source_status']='BLOCKED'
    elif damage=='fingerprint':m['training_source_fingerprint']='a'*64
    elif damage=='fake_operator':m['config']['approved_by']='auto'
    elif damage=='schema_to_strict':m['schema']=strict.SCHEMA;m['status']='READY'
    m['input_fingerprint']=strict.fingerprint({k:v for k,v in m.items() if k!='input_fingerprint'});write_json(p,m)
    with pytest.raises(mobile.MobileExportError):mobile.MobileModelExporter()._calibration_for_request(req)


@pytest.mark.parametrize('field,value',[('image_size',64),('confidence_threshold',.3),('iou_threshold',.5),('calibration_fingerprint','a'*64)])
def test_v2_rejects_changed_request(tmp_path,field,value):
    _,_,_,req=prepared_v2(tmp_path)
    with pytest.raises(mobile.MobileExportError):mobile.MobileModelExporter()._calibration_for_request(replace(req,**{field:value}))


@pytest.mark.parametrize('role',['plate','character'])
def test_v2_exact_loaders_and_cross_split(tmp_path,role):
    with patch.object(legacy,'prepared',prepared_v2):legacy.test_cross_split_removed_and_exact_pixels_reach_both_loaders(tmp_path,role)


def test_v2_actual_export_path_with_stubbed_converter_and_usage(tmp_path):
    with patch.object(legacy,'prepared',prepared_v2):legacy.test_native_export_writes_research_manifest_usage_and_both_variant_shas(tmp_path)


def test_v2_nested_package_compatibility(tmp_path):
    with patch.object(legacy,'prepared',prepared_v2):legacy.test_research_nested_package_preserves_role_proof_and_sha(tmp_path)


@pytest.mark.parametrize('damage',['mixed_graph','usage_sha','usage_rehashed','missing_warning','wrong_mode'])
def test_v2_package_rejects_tampered_precision_and_consumption(tmp_path,damage):
    with patch.object(legacy,'prepared',prepared_v2):legacy.test_archive_cannot_claim_research_int8_with_bad_graph_or_proof(tmp_path,damage)


@pytest.mark.parametrize('role',['plate','character'])
def test_v1_rebuild_is_unchanged_and_second_confirmation_not_required(tmp_path,role):
    _,m,p,req=legacy.prepared(tmp_path,role)
    before=p.read_bytes()
    assert m['schema']==exp.SCHEMA_V1 and exp.validate_manifest(p)==m
    assert mobile.MobileModelExporter()._calibration_for_request(replace(req,exploratory_confirmation=False))==m
    assert p.read_bytes()==before
    for field,value in [('image_size',64),('checkpoint_sha256','a'*64)]:
        changed=copy.deepcopy(m);changed['config'][field]=value
        write_json(p,changed)
        with pytest.raises(mobile.MobileExportError):mobile.MobileModelExporter()._calibration_for_request(req)
