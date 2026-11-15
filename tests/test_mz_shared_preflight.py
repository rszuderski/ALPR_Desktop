from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from auto_annotation_tool.dataset_split_assignment import canonical_sha256
from auto_annotation_tool.training.dataset_splitter import DatasetSplitter
from auto_annotation_tool.training.model_provenance import build_training_dataset_snapshot
from auto_annotation_tool.training import mz_experiment_runtime as runtime
from test_mz_strict_training_protocol import recipe


def resign(protocol):
    protocol['protocol_sha256'] = canonical_sha256({k:v for k,v in protocol.items() if k != 'protocol_sha256'})
    return protocol


@pytest.fixture
def frozen(tmp_path):
    source=tmp_path/'source'
    (source/'images').mkdir(parents=True)
    (source/'labels').mkdir()
    items=[]
    for index in range(30):
        row={'pid':f'p{index}', 'source_pid':f'plate_{index}', 'image_path':f'images/p{index}.jpg',
            'label_path':f'labels/p{index}.txt','characters':[{'character':'A','class_id':10}],
            'provenance':{'source_image_id':str(index//3)}}
        items.append(row)
        (source/row['image_path']).write_bytes(f'fixture-{index}'.encode())
        (source/row['label_path']).write_text('10 .5 .5 .2 .2',encoding='utf-8')
    (source/'data.yaml').write_text('nc: 36\nnames: '+json.dumps(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'))+'\ntrain: images\nval: images\n',encoding='utf-8')
    (source/'metadata_manifest.json').write_text(json.dumps({'dataset_type':'char_yolo_detect',
        'items':items,'plate_count':30,'character_count':30,'gold_source_contract_sha256':'a'*64}),encoding='utf-8')
    root=tmp_path/'split'
    assert DatasetSplitter(42).split_dataset(source,root,{'train':.8,'val':.1,'test':.1})[0]
    snapshot=build_training_dataset_snapshot(root,target='char')
    assignment=json.loads((root/'split_assignment_manifest.json').read_text(encoding='utf-8'))
    code=tmp_path/'code.py';code.write_text('frozen code')
    code_hashes={'code.py':runtime.file_sha256(code)}
    models={}
    for variant,name in [('MZ-n','yolo26n'),('MZ-s','yolo26s')]:
        weight=tmp_path/(name+'.pt');weight.write_bytes(name.encode())
        models[variant]={'architecture':name,'checkpoint':str(weight),'checkpoint_sha256':runtime.file_sha256(weight)}
    protocol={'schema':'alpr.experiment.mz_ns.v1','experiment_id':'E-MZ-NS-01',
        'dataset':{'path':str(root),'dataset_id':snapshot['dataset_id'],
            'training_dataset_snapshot':snapshot,'split_sha256':snapshot['split_sha256'],
            'data_yaml_sha256':snapshot['data_yaml_sha256'],'assignment_sha256':assignment['assignment_sha256'],
            'source_dataset_id':'source','source_manifest_sha256':runtime.file_sha256(source/'metadata_manifest.json'),
            'source_contract_sha256':'a'*64,
            **{name+'_count':assignment['counts'][name+'_images'] for name in ['train','val','test']}},
        'code':{'file_sha256':code_hashes,'working_code_sha256':canonical_sha256(code_hashes)},
        'models':models,'training':recipe(),'environment':{'torch':'test-torch','ultralytics':'test-ultra',
            'albumentations':'test-albu','gpu':'test-GPU','cuda_runtime':'test-cuda'}}
    env={'torch':SimpleNamespace(__version__='test-torch',version=SimpleNamespace(cuda='test-cuda'),
            cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda index:'test-GPU')),
        'ultralytics':SimpleNamespace(__version__='test-ultra'),
        'albumentations':SimpleNamespace(__version__='test-albu')}
    with patch.dict('sys.modules',env):
        yield resign(protocol),tmp_path,env


def test_shared_checker_accepts_frozen_input_without_loading_model_or_writing_files(frozen):
    protocol,repo,_=frozen
    before={str(path):path.read_bytes() for path in repo.rglob('*') if path.is_file()}
    checked=runtime.check_mz_experiment_protocol(protocol,repo_root=repo)
    assert checked['ok'] and sum(checked['counts'][name+'_images'] for name in ['train','val','test'])==30
    assert before=={str(path):path.read_bytes() for path in repo.rglob('*') if path.is_file()}


def test_bad_protocol_sha_is_rejected(frozen):
    protocol,repo,_=frozen
    protocol['training']['batch']=2
    with pytest.raises(ValueError,match='fingerprint mismatch'):
        runtime.check_mz_experiment_protocol(protocol,repo_root=repo)


@pytest.mark.parametrize('kind',['assignment','label','code','source','MZ-n','MZ-s'])
def test_changed_frozen_artifact_is_rejected(frozen,kind):
    protocol,repo,_=frozen
    root=Path(protocol['dataset']['path'])
    paths={'assignment':root/'split_assignment_manifest.json','label':next((root/'labels/test').glob('*.txt')),
        'code':repo/'code.py','source':repo/'source/metadata_manifest.json',
        'MZ-n':Path(protocol['models']['MZ-n']['checkpoint']),'MZ-s':Path(protocol['models']['MZ-s']['checkpoint'])}
    path=paths[kind]
    path.write_bytes(path.read_bytes()+b'\nchanged')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        runtime.check_mz_experiment_protocol(protocol,repo_root=repo)


@pytest.mark.parametrize('library',['torch','ultralytics','albumentations'])
def test_environment_version_drift_is_rejected(frozen,library):
    protocol,repo,env=frozen
    env[library].__version__='changed'
    with pytest.raises(ValueError,match='library version'):
        runtime.check_mz_experiment_protocol(protocol,repo_root=repo)


def test_missing_declared_gpu_is_rejected(frozen):
    protocol,repo,env=frozen
    env['torch'].cuda.is_available=lambda:False
    with pytest.raises(ValueError,match='CUDA device'):
        runtime.check_mz_experiment_protocol(protocol,repo_root=repo)


def test_wrong_declared_split_fingerprint_is_rejected_even_if_protocol_is_resigned(frozen):
    protocol,repo,_=frozen
    protocol['dataset']['split_sha256']='0'*64
    resign(protocol)
    with pytest.raises(ValueError,match='split_sha256'):
        runtime.check_mz_experiment_protocol(protocol,repo_root=repo)


def test_both_variants_build_same_recipe_and_isolate_the_input_json(frozen):
    protocol,repo,_=frozen
    first=runtime.prepare_mz_experiment(protocol,'MZ-n',repo_root=repo)
    second=runtime.prepare_mz_experiment(protocol,'MZ-s',repo_root=repo)
    n,s=first.request,second.request
    assert {k:v for k,v in n.items() if k not in ['name','base_model']}=={k:v for k,v in s.items() if k not in ['name','base_model']}
    assert n['base_model'].endswith('yolo26n.pt') and s['base_model'].endswith('yolo26s.pt')
    protocol['training']['batch']=99
    assert first.request['batch_size']==4 and first.request['training_protocol']['batch']==4
