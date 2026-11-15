"""Real native Tk widgets with synthetic services; never production inference."""
from pathlib import Path
import gc
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import pytest

from auto_annotation_tool.gui import z4_eval396, z4_e2e_results
from auto_annotation_tool.ranking import e2e_eval_runner as runner
from e2e_fixture import build_env


def spin(root, predicate, timeout=12):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        root.update()
        if predicate():return
        time.sleep(.01)
    raise AssertionError('Tk operation timed out')


@pytest.fixture
def ui(tmp_path,monkeypatch):
    gc.collect()  # Reclaim closed Tk fixtures on the UI thread, never a worker.
    env=build_env(tmp_path,monkeypatch)
    root=tk.Tk();root.withdraw();errors=[]
    root.report_callback_exception=lambda *args:errors.append(args)
    host=SimpleNamespace(frame=ttk.Frame(root),rank_data_dir=tk.StringVar(root,value=env.context['selection_dir']),_get_ranking_task_target=lambda:'plate')
    dialog=z4_eval396.open_results(host)
    spin(root,lambda:hasattr(dialog,'e2e_panel'))
    panel=dialog.e2e_panel
    dialog.evaluation_notebook.select(panel)
    panel.device.set('cpu')
    for pair,var in panel.pair_vars.items():var.set(pair==('MT-n','MZ-s'))
    root.update()
    yield root,dialog,panel,env,errors
    root.destroy()
    assert not errors


@pytest.mark.parametrize('error_class',[PermissionError,FileNotFoundError,ValueError])
def test_native_notebook_no_autostart_and_source_failure_keeps_start_disabled(ui,monkeypatch,error_class):
    root,dialog,panel,env,_=ui
    assert len(dialog.evaluation_notebook.tabs())==4
    assert str(panel.start_button['state'])=='disabled' and env.calls==[]
    def denied(*a,**kw):raise error_class('S/selection_manifest.json')
    monkeypatch.setattr(runner,'prepare_campaign',denied)
    panel.prepare_button.invoke();spin(root,lambda:not panel.busy)
    assert 'BLOCKED' in panel.status.get() and panel.prepared is None
    assert str(panel.start_button['state'])=='disabled' and env.calls==[]


def prepare_and_sign(root,panel,env):
    panel.prepare_button.invoke();spin(root,lambda:not panel.busy)
    assert panel.prepared and not panel.can_start and env.calls==[]
    panel.operator.set('TEST FIXTURE ONLY - NOT A REAL APPROVAL')
    panel.ack_button.invoke();panel.approve_button.invoke();spin(root,lambda:not panel.busy)
    assert panel.can_start and env.calls==[]


def test_explicit_signature_then_start_results_domain_and_preview(ui):
    root,_,panel,env,_=ui
    prepare_and_sign(root,panel,env)
    panel.start_button.invoke();spin(root,lambda:not panel.busy)
    assert panel.campaign['status']=='COMPLETE' and len(env.calls)==3
    assert len(panel.table.get_children())==1 and not panel.can_start
    root.update()
    ticks=[item for item in panel.chart.find_all() if panel.chart.type(item)=='text' and panel.chart.itemcget(item,'text') in {'0%','25%','50%','75%','100%'}]
    assert len(ticks)==5
    assert all(panel.chart.bbox(item)[3]<=panel.chart.winfo_height() for item in ticks)
    assert float(panel.progress['value'])==3
    panel.domain.set('NIGHT');panel.refresh_results()
    assert '0/0' in str(panel.table.item(panel.table.get_children()[0])['values'])
    panel.views.select(2)
    chosen=next(str(i) for i,r in enumerate(panel.error_rows) if r['prediction_index'] is not None and r['prediction'])
    panel.errors.selection_set(chosen);panel.select_error();root.update()
    assert panel.scene_canvas.original_image is not None and panel.crop_canvas.original_image is not None
    assert 'Błąd' not in panel.preview_note.get()
    for canvas in (panel.scene_canvas, panel.crop_canvas):
        expected = min(canvas.winfo_width()/canvas.original_image.width,
                       canvas.winfo_height()/canvas.original_image.height)
        expected = max(canvas.min_zoom, min(canvas.max_zoom, expected))
        assert canvas.zoom_level == pytest.approx(expected)


@pytest.mark.parametrize('field',['pair','device','cache'])
def test_changing_inputs_invalidates_prepared_and_signature(ui,field):
    root,_,panel,env,_=ui
    prepare_and_sign(root,panel,env)
    if field=='pair':panel.pair_vars[('MT-n','MZ-DAY')].set(True)
    elif field=='device':panel.device.set('0')
    else:panel.cache.set('different-cache')
    assert panel.prepared is None and panel.approval is None and not panel.can_start
    assert 'Brak aktualnego' in panel.protocol_text.get('1.0','end')


def test_cancel_history_resume_is_native_and_preserves_finished_forward(ui):
    root,_,panel,env,_=ui
    prepare_and_sign(root,panel,env)
    ready,release=threading.Event(),threading.Event()
    def after():
        if len(env.calls)==1:ready.set();assert release.wait(10)
    env.after_predict=after
    panel.start_button.invoke();spin(root,ready.is_set)
    assert str(panel.cancel_button['state'])=='normal'
    panel.cancel_button.invoke();release.set();spin(root,lambda:not panel.busy)
    assert panel.campaign['status']=='PARTIAL' and panel.can_resume
    panel.refresh_history();spin(root,lambda:not panel.busy)
    assert len(panel.history)==1
    panel.resume_button.invoke();spin(root,lambda:not panel.busy)
    assert panel.campaign['status']=='COMPLETE' and len(env.calls)==3


@pytest.mark.skipif(sys.platform!='win32',reason='Windows native window acceptance')
def test_window_minimize_maximize_restore_and_footer_layout(ui):
    root,dialog,panel,_,_=ui
    assert not str(dialog.transient())
    dialog.state('normal');dialog.geometry('1160x790+10+10');root.update()
    normal=panel.table.winfo_width()
    dialog.state('zoomed');spin(root,lambda:dialog.state()=='zoomed');root.update()
    assert panel.table.winfo_width()>=normal
    assert panel.cancel_button.winfo_rooty()+panel.cancel_button.winfo_height()<=dialog.winfo_rooty()+dialog.winfo_height()+2
    assert panel.table.master.grid_info()['sticky']=='ew'
    dialog.iconify();spin(root,lambda:dialog.state()=='iconic')
    dialog.deiconify();dialog.state('normal');spin(root,lambda:dialog.state()=='normal')


def test_export_uses_selected_existing_run_not_runner(ui,monkeypatch,tmp_path):
    root,_,panel,env,_=ui
    prepare_and_sign(root,panel,env)
    panel.start_button.invoke();spin(root,lambda:not panel.busy)
    monkeypatch.setattr(z4_e2e_results.filedialog,'askdirectory',lambda **k:str(tmp_path/'reports'))
    before=len(env.calls)
    panel.export_button.invoke();spin(root,lambda:not panel.busy,30)
    assert 'Raport E2E zapisany' in panel.status.get() and len(env.calls)==before
    assert len(list((tmp_path/'reports').glob('eval396_e2e_report_*/*.png')))==3


def test_native_runner_final_scoring_is_busy_until_verified_complete(ui, monkeypatch):
    root, _, panel, env, _ = ui
    prepare_and_sign(root, panel, env)
    entered, release = threading.Event(), threading.Event()
    score_pair = runner.score_pair
    def waiting_score(*args, **kwargs):
        entered.set()
        assert release.wait(20)
        return score_pair(*args, **kwargs)
    monkeypatch.setattr(runner, 'score_pair', waiting_score)
    try:
        panel.start_button.invoke()
        spin(root, entered.is_set)
        spin(root, lambda: 'scoring GT' in panel.status.get())
        assert len(env.calls) == 3 and panel.busy and panel.running
        assert str(panel.progress['mode']) == 'indeterminate'
        assert '3/3' in panel.status.get()
        assert str(panel.export_button['state']) == 'disabled'
        before = float(panel.progress['value'])
        spin(root, lambda: float(panel.progress['value']) >= before+3)
    finally:
        release.set()
        spin(root, lambda: not panel.busy)
    assert panel.campaign['status'] == 'COMPLETE'
    assert str(panel.progress['mode']) == 'determinate'
    assert float(panel.progress['value']) == float(panel.progress['maximum']) == 3
    assert str(panel.export_button['state']) == 'normal'


@pytest.mark.parametrize('status', ['PARTIAL', 'RUNNING'])
def test_all_raw_scenes_without_complete_scoring_do_not_display_finished_bar(ui, status):
    root, _, panel, env, _ = ui
    prepare_and_sign(root, panel, env)
    panel.start_button.invoke(); spin(root, lambda: not panel.busy)
    # UI-only transition: leave all persisted fixture results intact.
    incomplete = dict(panel.campaign, status=status, scoring=None)
    panel.work('loaded', lambda: incomplete, 'TEST ONLY — incomplete scoring')
    spin(root, lambda: not panel.busy)
    assert panel.campaign['completed_scenes'] == panel.campaign['total_scenes'] == 3
    assert str(panel.progress['mode']) == 'determinate'
    assert float(panel.progress['value']) < float(panel.progress['maximum'])
    assert status in panel.status.get() and '3/3' in panel.status.get()
    assert 'ocena nieukończona' in panel.status.get()
