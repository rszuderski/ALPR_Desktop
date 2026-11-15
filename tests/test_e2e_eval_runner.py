from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from auto_annotation_tool.ranking import e2e_eval_runner as runner
from auto_annotation_tool.ranking import e2e_artifacts as store
from auto_annotation_tool.ranking.e2e_eval_report import export_report
from auto_annotation_tool.ranking.eval396 import digest
from e2e_fixture import build_env


@pytest.fixture
def env(tmp_path, monkeypatch):
    return build_env(tmp_path, monkeypatch)


def run(env, **kwargs):
    return runner.run_campaign(env.prepared, env.approval, **kwargs)


def latest(env):
    return next(Path(env.context['output_root']).glob('EVAL396_E2E_run_*'))


def test_preflight_and_signature_never_call_predict_or_create_run(env):
    assert env.calls == [] and env.inspections == ['MZ-s']
    assert not list(Path(env.context['output_root']).glob('EVAL396_E2E_run_*'))
    assert not any('domain' in s for rows in env.prepared['inputs'].values() for s in rows)


def test_full_prediction_graph_keeps_fp_duplicates_zero_and_invalid_quad(env):
    report = run(env)
    assert report['status'] == 'COMPLETE' and report['completed_scenes'] == 3
    assert len(env.calls) == 3 and env.max_active == 1 and env.active == 0
    rows = report['pairs']['MT-n__MZ-s']
    assert [len(s['detections']) for s in rows] == [0,3,1]
    assert rows[-1]['detections'][0]['failure'] == 'QUAD_INVALID'
    metrics = report['scoring']['MT-n__MZ-s']['domains']['ALL']
    assert (metrics['box_tp'],metrics['box_fp'],metrics['box_fn']) == (2,2,1)
    assert metrics['plates'] == 2 and metrics['exact_matches'] == 1 and metrics['cer'] == .5
    assert metrics['conditional_plates'] == 1 and metrics['fp_ocr'] == 1
    assert (latest(env)/'prediction_manifest.json').exists()
    assert run(env,resume_dir=latest(env))['status'] == 'COMPLETE' and len(env.calls) == 3


def test_gt_scoring_occurs_only_after_all_raw_is_sealed(env, monkeypatch):
    real = runner.score_pair
    def score(*args):
        assert (latest(env)/'prediction_manifest.json').exists()
        assert len(env.calls) == 3
        return real(*args)
    monkeypatch.setattr(runner,'score_pair',score)
    run(env)


def test_changing_gt_changes_score_but_not_prediction_content(env):
    first=run(env)
    env.context['reference'][1]['plates'][0]['ground_truth_text']='XY'
    env.prepared=runner.prepare_campaign(env.context['selection_dir'],env.context['cache_dir'],pairs=[('MT-n','MZ-s')],device='cpu')
    env.approval=runner.save_approval(env.prepared,operator='TEST ONLY',acknowledged=True)['approval']
    second=run(env)
    for a,b in zip(first['pairs']['MT-n__MZ-s'][1]['detections'],second['pairs']['MT-n__MZ-s'][1]['detections']):
        assert {k:a[k] for k in ['mt','characters','reading','rectification']} == {k:b[k] for k in ['mt','characters','reading','rectification']}
    assert first['scoring']['MT-n__MZ-s']['domains']['ALL']['exact_matches']==1
    assert second['scoring']['MT-n__MZ-s']['domains']['ALL']['exact_matches']==0


def test_cancel_inside_multi_plate_scene_and_resume_skips_recorded_forward(env):
    env.after_predict=lambda: None
    first=run(env,cancelled=lambda:len(env.calls)==1)
    assert first['status']=='PARTIAL' and first['scoring'] is None
    assert first['completed_scenes']==1
    second=run(env,resume_dir=first['run_dir'])
    assert second['status']=='COMPLETE' and len(env.calls)==3


@pytest.mark.parametrize('boundary',['raw','ledger','cleanup'])
def test_atomic_journal_recovers_without_repeating_persisted_forward(env,monkeypatch,boundary):
    write=store._write_json; unlink=Path.unlink; tripped=False
    def broken(path,payload,**kwargs):
        nonlocal tripped
        path=Path(path)
        hit=('/detections/' in path.as_posix() if boundary=='raw' else path.name=='progress.json' and env.calls)
        if boundary!='cleanup' and hit and not tripped:
            tripped=True;raise OSError('TEST_CRASH')
        return write(path,payload,**kwargs)
    def broken_unlink(path,*args,**kwargs):
        nonlocal tripped
        if boundary=='cleanup' and path.name=='pending.json' and env.calls and not tripped:
            tripped=True;raise OSError('TEST_CRASH')
        return unlink(path,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(store,'_write_json',broken);patch.setattr(Path,'unlink',broken_unlink)
        with pytest.raises(RuntimeError,match='E2E_PARTIAL'):run(env)
    report=runner.load_campaign(latest(env))
    assert report['recovery_required'] and len(env.calls)==1
    assert run(env,resume_dir=latest(env))['status']=='COMPLETE' and len(env.calls)==3


def test_cuda_error_remains_partial_and_does_not_become_no_read(env):
    env.fail_on=2
    with pytest.raises(RuntimeError,match='TEST_CUDA_ERROR'):run(env)
    report=runner.load_campaign(latest(env)); assert report['scoring'] is None and report['status']=='PARTIAL'
    env.fail_on=None
    assert run(env,resume_dir=latest(env))['status']=='COMPLETE' and len(env.calls)==4


@pytest.mark.parametrize('which',['source','checkpoint','code'])
def test_changed_inputs_block_before_any_predict(env,which):
    path=env.marker if which=='source' else env.code if which=='code' else Path(env.context['models'][-1]['checkpoint_file'])
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='CHANGED'):run(env)
    assert env.calls==[]


def test_changed_image_during_forward_never_commits_that_result(env):
    path=Path(env.context['source_root'])/env.context['inputs']['MT-n'][1]['file']
    env.after_predict=lambda:path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(RuntimeError,match='IMAGE_CHANGED'):run(env)
    assert not list(latest(env).glob('raw/*/detections/**/*.json'))


@pytest.mark.parametrize('key,value',[('conf',.1),('half',True),('rect',True),('max_det',1)])
def test_resealed_parameter_change_does_not_pass_fixed_protocol(env,key,value):
    p=env.prepared['protocol'];p['inference'][key]=value;p['protocol_sha256']=digest(runner.mt._without(p,'protocol_sha256'))
    with pytest.raises(ValueError):run(env)
    assert not env.calls


@pytest.mark.parametrize('field,value',[('backend_fp16',True),('backend_end2end',False),('checked_imgsz',[640,640]),('conf',.1)])
def test_actual_backend_changes_fail_without_record(env,field,value):
    env.effective_override[field]=value
    with pytest.raises(RuntimeError,match='EFFECTIVE'):run(env)
    assert not list(latest(env).glob('raw/*/detections/**/*.json'))


def test_missing_signature_is_not_authorization(env):
    env.approval['operator_signature']=''
    with pytest.raises(ValueError,match='SIGNATURE'):run(env)
    assert not env.calls


def test_three_explicit_pairs_are_sequential_and_all_domains(env):
    env.prepared=runner.prepare_campaign(env.context['selection_dir'],env.context['cache_dir'],
        pairs=list(zip(runner.MT_LABELS,runner.MZ_LABELS)),device='cpu')
    env.approval=runner.save_approval(env.prepared,operator='SYNTHETIC TEST ONLY',acknowledged=True)['approval']
    report=run(env)
    assert report['completed_scenes']==9 and env.max_active==1 and len(env.calls)==9
    assert all(set(score['domains'])=={'ALL','DAY','NIGHT'} for score in report['scoring'].values())


def test_original_open_errors_are_never_waived(tmp_path,monkeypatch):
    def denied(*args,**kwargs):raise PermissionError('S/selection_manifest.json ACCESS DENIED')
    monkeypatch.setattr(runner,'load_selection',denied)
    monkeypatch.setattr(runner,'inspect_mz_checkpoint',lambda *_:pytest.fail('Must not load a model after failed source gate'))
    with pytest.raises(PermissionError):runner.prepare_campaign(tmp_path,tmp_path,pairs=[('MT-n','MZ-s')])


def test_wrong_class_map_rejected_without_forward(tmp_path,monkeypatch):
    path=tmp_path/'fake.pt';path.write_bytes(b'NOT_WEIGHTS')
    model=SimpleNamespace(task='detect',names={0:'car'})
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=lambda *a,**k:model))
    with pytest.raises(ValueError,match='36_CLASSES'):
        runner.inspect_mz_checkpoint({'checkpoint_file':str(path),'checkpoint_sha256':runner.mt._sha(path)})


def test_no_read_mz_is_counted_as_deletions(env):
    env.chars='';report=run(env);m=report['scoring']['MT-n__MZ-s']['domains']['ALL']
    assert m['no_read']==2 and m['cer']==1 and m['conditional_plates']==1


def test_unknown_raw_and_missing_indexed_are_not_rerun(env):
    report=run(env,cancelled=lambda:len(env.calls)==1);path=Path(report['run_dir'])
    extra=path/'raw/foreign.json';extra.write_text('{}')
    with pytest.raises(ValueError,match='UNKNOWN_RAW'):runner.load_campaign(path)
    extra.unlink()
    first=next(path.glob('raw/*/detections/**/*.json'));first.unlink()
    with pytest.raises(ValueError,match='MISSING_INDEXED'):run(env,resume_dir=path)


def test_completed_export_is_new_svg_png_csv_md_with_provenance(env,tmp_path):
    report=run(env)
    output=export_report(report['run_dir'],tmp_path/'exports')
    assert len(list(output.glob('*.svg')))==3 and len(list(output.glob('*.png')))==3
    assert len(list(output.glob('*.csv')))==6
    status=json.loads((output/'status.json').read_text());assert status['status']=='COMPLETE'
    assert all(runner.mt._sha(output/name)==sha for name,sha in status['files_sha256'].items())
    assert 'SYNTHETIC_TEST_ONLY' in (output/'report.md').read_text(encoding='utf8')
    with pytest.raises(ValueError,match='PROTECTED'):export_report(report['run_dir'],env.context['source_root'])
    with pytest.raises(ValueError,match='PROTECTED'):export_report(report['run_dir'],tmp_path/'4_training_datasets'/'old_dataset')
    with pytest.raises(ValueError,match='INSIDE_EXISTING'):export_report(report['run_dir'],tmp_path/'ranking_report_20260919_114308'/'nested')


def test_unsigned_preparation_policy_is_not_silently_changed(env):
    assert env.prepared['protocol']['rectification']==runner.RECTIFICATION
    assert env.prepared['protocol']['mt_inference']=='reused_not_executed'
    assert env.prepared['protocol']['counts']['ALL']['ocr']==2


def test_crashed_running_status_can_resume_under_execution_lock(env):
    report=run(env,cancelled=lambda:len(env.calls)==1)
    path=Path(report['run_dir'])
    runner.mt._write_json(path/'status.json',{'status':'RUNNING','measurement_status':'RUNNING'})
    assert runner.load_campaign(path)['status']=='RUNNING'
    assert run(env,resume_dir=path)['status']=='COMPLETE' and len(env.calls)==3


def test_completion_seal_publication_then_status_failure_is_idempotent(env,monkeypatch):
    original=runner.mt._write_json
    def fail_status(path,value,**kwargs):
        if Path(path).name=='status.json' and value.get('status')=='COMPLETE':raise OSError('TEST_FINAL_STATUS_CRASH')
        return original(path,value,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(runner.mt,'_write_json',fail_status)
        with pytest.raises(RuntimeError,match='E2E_PARTIAL'):run(env)
    assert (latest(env)/'completion_manifest.json').exists()
    before=len(env.calls)
    assert run(env,resume_dir=latest(env))['status']=='COMPLETE' and len(env.calls)==before


@pytest.mark.parametrize('change',['homography','permutation','raw_mt','char_class','reading','pixel_sha'])
def test_resealed_pending_semantic_corruption_is_rejected(env,monkeypatch,change):
    original=store._write_json
    def crash(path,payload,**kwargs):
        if '/detections/' in Path(path).as_posix():raise OSError('TEST_CRASH')
        return original(path,payload,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(store,'_write_json',crash)
        with pytest.raises(RuntimeError):run(env)
    pending_file=latest(env)/'pending.json';pending=store.read_json(pending_file);row=pending['record']
    if change=='homography':row['rectification']['homography'][0][0]+=1
    elif change=='permutation':row['rectification']['permutation']=[3,0,1,2]
    elif change=='raw_mt':row['mt']['keypoints'].reverse()
    elif change=='char_class':row['characters'][0]['class_id']=35
    elif change=='reading':row['reading']['prediction']='GT_ASSIST'
    else:row['decoded_pixel_sha256']='0'*64
    row['record_sha256']=digest(runner.mt._without(row,'record_sha256'))
    import hashlib
    from auto_annotation_tool.ranking.eval396 import canonical
    pending['sha256']=hashlib.sha256(canonical(row)+b'\n').hexdigest()
    pending['transaction_sha256']=digest(runner.mt._without(pending,'transaction_sha256'))
    original(pending_file,pending)
    with pytest.raises(ValueError):runner.load_campaign(latest(env))
    assert len(env.calls)==1


def test_partial_export_has_raw_evidence_but_no_fake_final_metrics(env,tmp_path):
    report=run(env,cancelled=lambda:len(env.calls)==1)
    output=export_report(report['run_dir'],tmp_path/'partial_reports')
    assert store.read_json(output/'status.json')['status']=='PARTIAL'
    assert store.read_json(output/'summary.json')==[]
    assert 'PARTIAL' in (output/'report.md').read_text(encoding='utf8')
    assert len(env.calls)==1


def test_selection_adapter_does_not_read_historical_mz_results(monkeypatch,tmp_path):
    from auto_annotation_tool.ranking import eval396
    data={'included_scenes.json':[],'included_plates.json':[],'model_lineage.json':[]}
    monkeypatch.setattr(eval396,'_read_selection',lambda *a:({'TEST_ONLY':True},data))
    monkeypatch.setattr(eval396,'_read_mz_results',lambda *a:pytest.fail('Old MZ must not gate source-only adapter'))
    result=eval396.load_selection(tmp_path)
    assert result['manifest']=={'TEST_ONLY':True} and result['models']==[]


def test_native_cache_trust_anchor_is_checked_before_cache_loader(tmp_path,monkeypatch):
    selection=tmp_path/'evaluation_selections/test';selection.mkdir(parents=True)
    cache=tmp_path/'evaluation_runs/EVAL396_MT_run_TEST';cache.mkdir(parents=True)
    (cache/'completion_manifest.json').write_text('{}')
    monkeypatch.setattr(runner,'load_selection',lambda *a,**k:{'selection_dir':str(selection)})
    monkeypatch.setattr(runner.mt,'load_campaign',lambda *a:pytest.fail('Untrusted cache must not be loaded'))
    with pytest.raises(ValueError,match='COMPLETION_ANCHOR'):
        runner._source_context(selection,cache)


def test_low_disk_blocks_without_creating_run(env,monkeypatch):
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda *a:SimpleNamespace(free=10))
    with pytest.raises(ValueError,match='DISK_SPACE'):run(env)
    assert not env.calls


def test_known_unreadable_selection_routes_to_gui_but_not_past_preflight(monkeypatch,tmp_path):
    from auto_annotation_tool.ranking import eval396
    def denied(*a,**kw):raise PermissionError('TEST_ACCESS_DENIED')
    monkeypatch.setattr(Path,'is_file',denied)
    assert eval396.is_selection(tmp_path/eval396.SELECTION_NAME)
    assert not eval396.is_selection(tmp_path/'unknown')
    monkeypatch.setattr(Path,'read_bytes',denied)
    with pytest.raises(PermissionError):eval396.load_selection(tmp_path/eval396.SELECTION_NAME)
