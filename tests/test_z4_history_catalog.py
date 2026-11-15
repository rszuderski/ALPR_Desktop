import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_annotation_tool.gui import z4_history_runtime as catalog, z4_training_runtime as runtime
from auto_annotation_tool.gui import z4_training_compare as compare, z4_validation_panel as validation
from auto_annotation_tool.training.training_history import TrainingHistory, TrainingRun


class Tree:
    def __init__(self):
        self.rows={}
        self.selected=()
        self.focused=''
    def selection(self): return self.selected
    def selection_set(self,items): self.selected=tuple(items) if isinstance(items,(tuple,list)) else (items,)
    def focus(self,item=None):
        if item is not None: self.focused=item
        return self.focused
    def get_children(self): return tuple(self.rows)
    def delete(self,*items):
        for item in items: self.rows.pop(item,None)
    def insert(self,parent,index,iid,values,tags=()):
        assert iid not in self.rows
        self.rows[iid]={'values':values,'tags':tags}
        return iid
    def item(self,item,option): return self.rows[item][option]
    def tag_configure(self,*args,**kwargs): pass
    def exists(self,item): return item in self.rows
    def see(self,item): pass


@pytest.fixture
def host(tmp_path,monkeypatch):
    directories={target:tmp_path/name for target,name in [('plate','plates'),('char','chars'),('vehicle','vehicles')]}
    for target,directory in directories.items():
        directory.mkdir()
        output=directory/'same_timestamp'
        weights=output/'train/weights'
        weights.mkdir(parents=True)
        (weights/'best.pt').write_bytes(target.encode())
        (weights/'last.pt').write_bytes(target.encode())
        run=TrainingRun(id='same_timestamp',name=target,created_at='2026-10-05T10:00:00',status='completed',
            training_target=target,dataset_path=str(tmp_path/('dataset_'+target)),output_dir=str(output),
            best_weights=str(weights/'best.pt'),last_weights=str(weights/'last.pt'))
        (directory/'training_history.json').write_text(json.dumps({'runs':{run.id:run.to_dict()}}),encoding='utf-8')
    monkeypatch.setattr(catalog.CAMPAIGN,'get_active_project_name',lambda:'')
    monkeypatch.setattr(catalog.CONFIG,'get_training_runs_dir',lambda target:directories[target])
    monkeypatch.setattr(runtime,'_get_pinned_history_result_keys',lambda host:(set(),set()))
    monkeypatch.setattr(runtime,'_refresh_campaign_training_result_selector',lambda host:None)
    active=TrainingHistory(directories['plate'],reconcile_on_load=False)
    value=SimpleNamespace(history=active,trainer=SimpleNamespace(history=active,is_training=False),tree=Tree(),
        app=SimpleNamespace(palette={}),_get_selected_training_target=lambda:'plate',
        get_campaign_training_target=lambda:'plate',_infer_history_run_target=lambda run:run.training_target,
        _infer_dataset_target=lambda path:'char' if 'char' in path else 'plate',
        _format_history_run_target_label=lambda target:{'plate':'Tablice','char':'Znaki','vehicle':'Pojazdy'}.get(target,target),
        _format_history_run_status_label=lambda run:run.status,
        _resolve_history_run_best_weights=lambda run:Path(run.best_weights),
        _set_history_run_tables=Mock(),_autofill_validation_inputs_from_run=Mock(),_open_path=Mock(),
        _open_model_validation_modal=Mock(),_open_run_details_modal=Mock(),_open_run_analysis_window=Mock())
    value._selected_run=lambda:catalog._selected_run(value)
    value._reload_history_snapshot_from_disk=lambda:runtime._reload_history_snapshot_from_disk(value)
    value._on_run_selected=lambda:runtime._on_run_selected(value)
    value._load_history=lambda:runtime._load_history(value)
    value._open_selected_run_validation_modal=lambda:validation._open_selected_run_validation_modal(value)
    value._load_history()
    return value,directories


def test_restart_at_plate_shows_all_targets_with_collision_safe_ids_and_no_writes(host,monkeypatch):
    value,directories=host
    before={target:(path/'training_history.json').read_bytes() for target,path in directories.items()}
    active,trainer_history=value.history,value.trainer.history
    monkeypatch.setattr(TrainingHistory,'_save',Mock(side_effect=AssertionError('catalog must not write')))
    monkeypatch.setattr(TrainingHistory,'_reconcile_checkpoint_paths',Mock(side_effect=AssertionError('no repairs')))
    value._load_history()
    assert set(value.tree.rows)=={'plate:same_timestamp','char:same_timestamp','vehicle:same_timestamp'}
    assert {iid:row['values'][0] for iid,row in value.tree.rows.items()}=={
        'plate:same_timestamp':'Tablice','char:same_timestamp':'Znaki','vehicle:same_timestamp':'Pojazdy'}
    assert value.trainer.history is value.history
    assert value.history.history_dir==active.history_dir==trainer_history.history_dir
    assert before=={target:(path/'training_history.json').read_bytes() for target,path in directories.items()}


def test_selected_foreign_run_routes_details_validation_folder_and_analysis(host):
    value,_=host
    value.tree.selection_set('char:same_timestamp')
    run=value._selected_run()
    assert run.training_target=='char' and run.id=='same_timestamp'
    runtime._on_run_selected(value)
    value._set_history_run_tables.assert_called_with(run)
    runtime._open_selected_run_details(value)
    value._open_run_details_modal.assert_called_with(run)
    validation._open_selected_run_validation_modal(value)
    kwargs=value._open_model_validation_modal.call_args.kwargs
    assert kwargs['model_path']==Path(run.best_weights)
    assert kwargs['dataset_path']==run.dataset_path and kwargs['split']=='test'
    runtime._open_run_folder(value)
    value._open_path.assert_called_with(Path(run.output_dir))
    value._run_details_current_run=run
    value._run_details_current_run_id=run.id
    value.tree.selection_set('plate:same_timestamp')
    runtime._open_current_run_details_folder(value)
    value._open_path.assert_called_with(Path(run.output_dir))
    runtime._open_current_run_details_analysis(value)
    value._open_run_analysis_window.assert_called_with(run)
    validation._open_current_run_validation_modal(value)
    assert value._open_model_validation_modal.call_args.kwargs['model_path']==Path(run.best_weights)
    assert value._open_model_validation_modal.call_args.kwargs['split']=='test'


def test_delete_updates_only_selected_storage_even_when_ids_collide(host,monkeypatch):
    value,directories=host
    before=(directories['plate']/'training_history.json').read_bytes()
    vehicle_before=(directories['vehicle']/'training_history.json').read_bytes()
    value.tree.selection_set('char:same_timestamp')
    monkeypatch.setattr(runtime.messagebox,'askyesno',lambda *args:True)
    runtime._delete_selected(value)
    assert not json.loads((directories['char']/'training_history.json').read_text())['runs']
    assert (directories['plate']/'training_history.json').read_bytes()==before
    assert (directories['vehicle']/'training_history.json').read_bytes()==vehicle_before
    assert not (directories['char']/'same_timestamp').exists()
    assert (directories['plate']/'same_timestamp').exists()


def test_compare_keeps_same_timestamp_from_distinct_storages(host):
    value,_=host
    value.tree.selection_set(['plate:same_timestamp','char:same_timestamp'])
    value._reload_history_snapshot_from_disk=Mock(side_effect=AssertionError('no reload on selection'))
    runs=compare._selected_training_compare_runs(value)
    assert [run.training_target for run in runs]==['plate','char']
    assert value._selected_run() is runs[0]
    value._reload_history_snapshot_from_disk.assert_not_called()


def test_compare_modal_and_legend_keep_distinct_source_ids(host,monkeypatch):
    value,_=host
    value.tree.selection_set(['plate:same_timestamp','char:same_timestamp'])
    runs=compare._selected_training_compare_runs(value)
    monkeypatch.setattr(compare,'_format_compare_run_label',lambda host,run,index:run.training_target)
    monkeypatch.setattr(compare,'_compare_dataset_label',lambda host,run:run.dataset_path)
    data=compare._build_training_compare_runs_data(value,runs)
    assert len(data)==2
    assert [item['run_id'] for item in data]==['same_timestamp','same_timestamp']
    assert [item['row_id'] for item in data]==['plate:same_timestamp','char:same_timestamp']
    value._training_compare_runs_data=data
    value._training_compare_summary_tree=Tree()
    compare._populate_training_compare_summary(value)
    assert len(value._training_compare_summary_tree.rows)==2
    value._training_compare_summary_tree.selection_set('char:same_timestamp')
    assert compare._selected_training_compare_modal_run(value).training_target=='char'


@pytest.mark.parametrize('action',[runtime._resume_selected_run,runtime._select_selected_run_as_fine_tune_base])
def test_foreign_resume_and_fine_tune_block_before_active_storage_or_parameters(host,monkeypatch,action):
    value,_=host
    value.tree.selection_set('char:same_timestamp')
    warning=Mock()
    monkeypatch.setattr(runtime.messagebox,'showwarning',warning)
    value._begin_step4_operation=Mock(side_effect=AssertionError('no run may start'))
    value.base_model_var=Mock(side_effect=AssertionError('do not change parameters'))
    action(value)
    assert warning.called and 'Znaki' in warning.call_args.args[1]
    value._begin_step4_operation.assert_not_called()


def test_campaign_catalog_remains_scoped_and_keeps_legacy_row_ids(host,monkeypatch):
    value,_=host
    monkeypatch.setattr(catalog.CAMPAIGN,'get_active_project_name',lambda:'project')
    value._does_history_run_match_active_campaign_target=lambda run:run.training_target=='plate'
    value._load_history()
    assert set(value.tree.rows)=={'same_timestamp'}
    assert catalog._get_visible_training_history_sources(value)==[('plate',value.history)]


def test_reload_during_training_keeps_active_objects(host):
    value,_=host
    value.trainer.is_training=True
    history=value.history
    assert not runtime._reload_history_snapshot_from_disk(value)
    assert value.history is value.trainer.history is history


def test_real_treeview_shows_char_at_plate_and_preserves_selected_row(host):
    import tkinter as tk
    from tkinter import ttk
    value,_=host
    root=tk.Tk()
    root.withdraw()
    try:
        value.tree=ttk.Treeview(root,columns=('Tor','Model','Dataset','Run','Start','Status','Epoki','mAP50-95'))
        value._load_history()
        value.tree.selection_set('char:same_timestamp')
        value.tree.focus('char:same_timestamp')
        value._load_history()
        root.update_idletasks()
        assert value.tree.selection()==('char:same_timestamp',)
        assert value.tree.focus()=='char:same_timestamp'
        assert value.tree.set('char:same_timestamp','Tor')=='Znaki'
        assert value._selected_run().training_target=='char'
        assert value.history.history_dir.name=='plates'
    finally:
        root.destroy()
