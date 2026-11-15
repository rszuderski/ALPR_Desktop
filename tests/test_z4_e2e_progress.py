"""Real Tk marker positions across successive E2E operations; no models."""
import gc
import threading
import time
import tkinter as tk
from tkinter import ttk

import pytest

from auto_annotation_tool.gui.z4_e2e_results import E2EMeasurementPanel


def pump(root, predicate, timeout=3):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        root.update()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('Progress operation timed out')


@pytest.fixture
def panel(tmp_path):
    gc.collect()
    root = tk.Tk(); root.geometry('1100x820+10+10')
    ttk.Style(root).theme_use('clam')
    errors, releases = [], []
    root.report_callback_exception = lambda *args: errors.append(args)
    notebook = ttk.Notebook(root); notebook.pack(fill='both', expand=True)
    view = E2EMeasurementPanel(notebook, tmp_path/'evaluation_selections'/'TEST_ONLY')
    notebook.add(view, text='TEST ONLY'); root.update()
    yield root, view, releases
    for release in releases:
        release.set()
    pump(root, lambda: not view.busy)
    root.destroy()
    assert not errors


def begin_waiting(panel, kind='history'):
    root, view, releases = panel
    release = threading.Event(); releases.append(release)
    def operation():
        assert release.wait(10)
        if kind == 'run':
            raise InterruptedError('TEST ONLY — no executor was called')
        return []
    view.work(kind, operation, 'TEST ONLY — waiting for I/O')
    root.update()
    return release


def marker_center(bar):
    xs = [x for x in range(bar.winfo_width()) if 'pbar' in bar.identify(x, bar.winfo_height()//2)]
    assert xs, 'Native progress marker must be visible'
    return (min(xs)+max(xs))/2


@pytest.mark.parametrize('previous_maximum', [1, 3, 1188])
def test_next_operation_moves_smoothly_regardless_of_previous_counter(panel, previous_maximum):
    root, view, _ = panel
    view.work('history', lambda: [], 'TEST ONLY — finished operation')
    pump(root, lambda: not view.busy)
    view.progress.configure(maximum=previous_maximum, value=previous_maximum)
    release = begin_waiting(panel)
    view.progress.stop()  # Deterministic single animation ticks below.
    centers = []
    for _ in range(16):
        root.update()
        centers.append(marker_center(view.progress))
        view.progress.step()
    width = view.progress.winfo_width()
    assert str(view.progress['mode']) == 'indeterminate'
    assert all(0 <= b-a < width*.02 for a, b in zip(centers, centers[1:]))
    assert centers[-1]-centers[0] > width*.04  # Neither flickering nor stalled.
    release.set(); pump(root, lambda: not view.busy)


def test_real_counts_stop_animation_and_never_drift(panel):
    root, view, _ = panel
    release = begin_waiting(panel)
    view.messages.put(('progress', view.epoch, {'message':'TEST 4/10', 'completed':4, 'total':10}))
    pump(root, lambda: float(view.progress['value']) == 4 and str(view.progress['mode']) == 'determinate')
    until = time.monotonic()+.16
    while time.monotonic() < until:
        root.update(); time.sleep(.005)
    assert float(view.progress['maximum']) == 10 and float(view.progress['value']) == 4
    release.set(); pump(root, lambda: not view.busy)


def test_run_validation_animates_before_first_measured_count(panel):
    root, view, _ = panel
    release = begin_waiting(panel, kind='run')
    assert str(view.progress['mode']) == 'indeterminate'
    initial = float(view.progress['value'])
    pump(root, lambda: float(view.progress['value']) >= initial+3)
    release.set(); pump(root, lambda: not view.busy)


def test_last_prediction_and_repeated_finalization_messages_keep_animating(panel):
    root, view, _ = panel
    release = begin_waiting(panel, kind='run')
    view.messages.put(('progress', view.epoch, {'message':'Last prediction', 'completed':1188, 'total':1188}))
    pump(root, lambda: 'Last prediction' in view.status.get())
    assert view.busy and str(view.progress['mode']) == 'indeterminate'
    pump(root, lambda: float(view.progress['value']) >= 8)
    for message in ('Verifying sources', 'Scoring GT', 'Loading verified results'):
        before = float(view.progress['value'])
        view.messages.put(('progress', view.epoch, {'message':message}))
        pump(root, lambda: message in view.status.get())
        assert str(view.progress['mode']) == 'indeterminate'
        assert float(view.progress['value']) >= before  # Do not restart every message.
        assert '1188/1188' in view.status.get()
    release.set(); pump(root, lambda: not view.busy)


def test_detection_updates_preserve_scene_count_but_uncounted_stage_animates(panel):
    root, view, _ = panel
    release = begin_waiting(panel, kind='run')
    view.messages.put(('progress', view.epoch, {'message':'Scenes', 'completed':4, 'total':10}))
    pump(root, lambda: view.status.get() == 'Scenes')
    view.messages.put(('progress', view.epoch, {'message':'Detection', 'detections':7}))
    pump(root, lambda: view.status.get() == 'Detection')
    assert str(view.progress['mode']) == 'determinate'
    assert float(view.progress['value']) == 4 and float(view.progress['maximum']) == 10
    view.messages.put(('progress', view.epoch, {'message':'Verifying sources'}))
    pump(root, lambda: 'Verifying sources' in view.status.get())
    assert str(view.progress['mode']) == 'indeterminate'
    release.set(); pump(root, lambda: not view.busy)
