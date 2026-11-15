from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_annotation_tool.gui import z4_training_metrics as metrics, z4_history_runtime as history
from auto_annotation_tool.training.training_history import TrainingRun


@pytest.fixture
def detail_host(monkeypatch):
    monkeypatch.setattr(metrics.CAMPAIGN,'get_active_project_name',lambda:'')
    value=SimpleNamespace(_shorten_training_text=lambda text,length:text,
        _format_history_run_status_label=lambda run:run.status,
        _infer_dataset_target=lambda path:'char',_get_dataset_split_image_counts=lambda *args:{},
        _does_history_run_match_active_campaign_target=lambda run:True)
    value._is_history_run_resumable=history._is_history_run_resumable
    value._is_history_run_resume_allowed=lambda run:history._is_history_run_resume_allowed(value,run)
    return value


@pytest.mark.parametrize('status',['completed','cancelled','running','paused','failed'])
@pytest.mark.parametrize('checkpoint',[True,False])
def test_resume_copy_respects_status_and_actual_checkpoint(detail_host,tmp_path,status,checkpoint):
    weight=tmp_path/'last.pt'
    if checkpoint: weight.write_bytes(b'fixture')
    run=TrainingRun('one','one','2026-10-05T10:00:00',status=status,last_weights=str(weight))
    rows=dict(metrics._build_training_run_detail_rows(detail_host,run))
    hint=rows['Wznowienie']
    if status=='completed':
        assert hint=='NIE - trening został zakończony.'
    elif status=='cancelled':
        assert 'anulowany' in hint
    elif status in {'paused','failed'} and checkpoint:
        assert hint.startswith('TAK')
    if checkpoint:
        assert 'brak' not in hint.lower()
    elif status in {'paused','failed'}:
        assert 'brak checkpointu last.pt' in hint.lower()


@pytest.mark.parametrize('device,actual,requested,expected',[
    ('', '0', 'cpu', 'CUDA device 0'),('-',0,'cpu','CUDA device 0'),
    (None,None,'0','CUDA device 0'),(0,'1','cpu','CUDA device 0'),
    ('cpu','0','1','cpu'),('auto','0','1','CUDA device 0'),('',None,None,'-')])
def test_device_display_falls_back_without_writing_or_guessing_GPU(detail_host,device,actual,requested,expected):
    run=TrainingRun('one','one','2026-10-05T10:00:00',status='completed',device=device,strict_experiment=True,
        training_protocol_snapshot={'actual':{'device':actual},'requested':{'device':requested}})
    before=run.to_dict()
    assert dict(metrics._build_training_run_detail_rows(detail_host,run))['Urządzenie']==expected
    assert run.to_dict()==before
