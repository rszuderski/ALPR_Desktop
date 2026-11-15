import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from auto_annotation_tool.gui import z4_training_runtime as runtime
from auto_annotation_tool.gui import z4_training_metrics as metrics
from auto_annotation_tool.gui import campaign_graph_actions as actions
from auto_annotation_tool.training.trainer import YOLOPoseTrainer


class HistoryTree:
    def __init__(self):
        self.owner = threading.get_ident()
        self.rows = {"live": {"Status": "oczekuje", "Epoki": "0/10"},
                     "other": {"Status": "completed", "Epoki": "10/10"}}
        self.tags = ()
        self.writes = []

    def exists(self, row):
        assert threading.get_ident() == self.owner
        return row in self.rows

    def selection(self):
        return getattr(self,'selected',())

    def item(self, row, option):
        assert threading.get_ident() == self.owner
        assert option == "tags"
        return self.tags

    def set(self, row, column, value=None):
        assert threading.get_ident() == self.owner
        if value is not None:
            self.writes.append((row, column, value))
            self.rows[row][column] = value
        return self.rows[row].get(column, "")


def live_owner():
    host = Mock()
    host.tree = HistoryTree()
    host.current_run_id = "live"
    host.trainer = YOLOPoseTrainer(history=Mock())
    host.trainer.current_run = SimpleNamespace(id="live", epochs=10, current_epoch=0)
    host.trainer.is_training = True
    host._last_training_batch_ui_emit_at = 0
    host._training_started_monotonic = time.perf_counter() - 1
    host._training_last_total_batches = 5
    host._current_training_metric_history = []
    host.get_campaign_training_target.return_value = "char"
    host._build_training_best_epoch_summary.return_value = "Test summary"
    host._metric_float.side_effect = float
    queued = []
    host._ui.side_effect = queued.append
    runtime._bind_trainer_callbacks(host)
    return host, queued


def test_worker_events_update_history_on_ui_without_reloading_or_disk_reads():
    host, queued = live_owner()
    host.trainer._emit_worker_event({"type": "batch_progress", "epoch": 1,
                                    "batch_idx": 1, "total_batches": 5, "batch_pct": 20})
    assert host.tree.rows["live"]["Status"] == "oczekuje"
    queued.pop(0)()
    assert host.tree.rows["live"] == {"Status": "trwa trening", "Epoki": "1/10"}
    assert host.trainer.current_run.current_epoch == 0  # A running epoch is not a completed epoch.

    worker = threading.Thread(target=host.trainer._emit_worker_event, args=({
        "type": "epoch_end", "epoch": 2, "metrics": {"map50": 0.5, "map50_95": 0.3, "loss": 1.0},
    },))
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert host.tree.rows["live"]["Epoki"] == "1/10"
    queued.pop(0)()
    assert host.tree.rows["live"]["Epoki"] == "2/10"
    assert host.tree.rows["other"] == {"Status": "completed", "Epoki": "10/10"}
    host._load_history.assert_not_called()
    host._reload_history_snapshot_from_disk.assert_not_called()


def test_repeated_batches_do_not_rewrite_same_cells():
    host, _ = live_owner()
    for _ in range(50):
        runtime._update_training_history_progress(host, run_id="live", epoch=3, total_epochs=10)
    assert len(host.tree.writes) == 2


@pytest.mark.parametrize('selected',[True,False])
def test_epoch_metrics_are_best_to_date_and_refresh_selected_panel_without_disk(selected):
    host,queued=live_owner()
    run=host.trainer.current_run
    run.dataset_path='chars'
    host.tree.selected=('char:live',) if selected else ('other',)
    host.tree.rows['char:live']=host.tree.rows.pop('live')
    source=SimpleNamespace(history_dir='chars')
    host.history=source
    host._history_row_refs={'char:live':{'run':run,'history':source,'target':'char'}}
    host._infer_dataset_target.return_value='char'
    host._iter_training_metric_keys.return_value=['map50_95']
    host._format_training_metric_band.return_value=('Test grade','')
    host._format_training_metric_name.side_effect=lambda key:key
    host._format_training_metric_value.side_effect=lambda key,value:f'{value:.3f}'
    host._build_training_run_detail_rows.return_value=[('Postęp','live')]
    host._build_training_run_metric_rows.side_effect=lambda run:metrics._build_training_run_metric_rows(host,run)
    host._set_history_run_tables.side_effect=lambda run:metrics._set_history_run_tables(host,run)
    host.hist_detail_tree=object()
    host.hist_metrics_tree=object()
    host._set_metric_table_rows=Mock()
    host._history_selected_detail_rows=[('Postęp','0/10 epok'),('Status','running')]
    host._format_history_run_status_label.return_value='trwa trening'
    host._infer_dataset_target.side_effect=AssertionError('no dataset inspection after an epoch')
    for epoch,value,best in [(1,.30,.30),(2,.25,.30),(3,.42,.42)]:
        host.trainer._emit_worker_event({'type':'epoch_end','epoch':epoch,'metrics':{'map50_95':value,'map50':.5,'loss':1.0}})
        queued.pop(0)()
        assert host.tree.rows['char:live']['mAP50-95']==f'{best:.3f}'
        assert host.tree.rows['char:live']['Epoki']==f'{epoch}/10'
        if selected:
            host._set_metric_table_rows.assert_any_call(host.hist_metrics_tree,
                [('map50_95',f'{value:.3f}',f'{best:.3f}','Test grade')])
    assert len(run.metrics_history)==3
    host._set_history_run_tables.assert_not_called()
    host._build_training_run_detail_rows.assert_not_called()
    host._infer_dataset_target.assert_not_called()
    assert host.tree.selected==(('char:live',) if selected else ('other',))
    host._load_history.assert_not_called()
    host._reload_history_snapshot_from_disk.assert_not_called()


def test_epoch_update_does_not_replace_details_of_first_selected_other_run():
    host,queued=live_owner()
    host.tree.selected=('other','live')
    with patch.object(metrics,'_refresh_history_epoch_tables') as refresh:
        host.trainer._emit_worker_event({'type':'epoch_end','epoch':1,'metrics':{'map50_95':.3}})
        queued.pop(0)()
    assert host.tree.rows['live']['mAP50-95']=='0.300'
    refresh.assert_not_called()
    assert host.tree.selected==('other','live')


def test_resume_pause_stop_and_pinned_marker():
    host, _ = live_owner()
    host.tree.tags = ("pinned_result",)
    host.tree.rows["live"]["Status"] = "paused (resume)"
    runtime._update_training_history_progress(host, run_id="live", epoch=4, total_epochs=10)
    assert host.tree.rows["live"]["Status"] == "★ PODPIĘTY | trwa trening"
    host.trainer.should_pause = True
    runtime._update_training_history_progress(host, run_id="live")
    assert host.tree.rows["live"]["Status"] == "★ PODPIĘTY | wstrzymywanie"
    host.trainer.should_stop = True
    runtime._update_training_history_progress(host, run_id="live")
    assert host.tree.rows["live"]["Status"] == "★ PODPIĘTY | zatrzymywanie"
    assert host.tree.rows["live"]["Epoki"] == "4/10"


@pytest.mark.parametrize("scenario", ["ended", "other_run", "other_context", "row_removed"])
def test_late_events_cannot_change_finished_or_other_run(scenario):
    host, _ = live_owner()
    if scenario == "ended":
        host.trainer.is_training = False
    elif scenario == "other_run":
        host.trainer.current_run = SimpleNamespace(id="new")
    elif scenario == "other_context":
        host.current_run_id = "new"
    else:
        host.tree.rows.pop("live")
    assert not runtime._update_training_history_progress(host, run_id="live", epoch=4, total_epochs=10)
    assert not host.tree.writes


@pytest.mark.parametrize("target", ["plate", "char"])
@pytest.mark.parametrize("iteration", [1, 7])
def test_skip_training_copy_uses_t06_and_only_prepares_decision(target, iteration):
    host = Mock()
    campaign = Mock()
    campaign.get_active_project_name.return_value = "test"
    campaign.get_current_step.return_value = 4
    campaign.get_current_iteration_num.return_value = iteration
    campaign.get_iteration_target.return_value = target
    with patch.object(actions, "CAMPAIGN", campaign), \
         patch.object(actions, "_current_training_stage_label", return_value="E4"):
        result = actions._execute_prepare_step4_without_training(host, {})
    assert result.ok
    title, message = host.app.themed_info.call_args.args[:2]
    assert "T06" in title and "T06" in message and "T06" in result.message
    assert "T07" not in title + message + result.message
    campaign.set_step4_without_training_decision.assert_called_once_with(True, target=target, iteration_num=iteration)
    host._finish_step4_without_training.assert_not_called()
