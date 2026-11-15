from pathlib import Path
from threading import Event, get_ident
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import campaign_ingest_identity as runtime
from auto_annotation_tool.gui import campaign_step1_ingest as ui
from test_z3_extraction_performance import Frame


def test_hash_preparation_runs_off_tk_and_only_saved_manifest_is_approved(monkeypatch, tmp_path):
    ui_thread = get_ident()
    frame = Frame()
    ready = Event()
    entered = Event()
    prepared = {"selected_images": [{"name": "a.jpg"}], "selected_count": 1,
                "identity_summary": {"duplicate_sha256": 1}}
    def record(**kwargs):
        assert get_ident() != ui_thread
        assert kwargs["project_name"] == "demo" and kwargs["iteration_num"] == 1
        entered.set()
        assert ready.wait(3)
        for index in range(1000):
            kwargs["progress_callback"](index / 10, "hash", detail="file")
        return tmp_path / "manifest.json"
    manager = SimpleNamespace(get_active_project_name=lambda: "demo", get_current_iteration_num=lambda: 1,
                              record_iteration_ingest=record, load_ingest_manifest=lambda *args: prepared)
    monkeypatch.setattr(runtime, "CAMPAIGN", manager)
    def update(*args, **kwargs):
        assert get_ident() == ui_thread
    monkeypatch.setattr(ui, "_update_ingest_plan_progress_dialog", Mock(side_effect=update))
    monkeypatch.setattr(ui, "_hide_ingest_plan_progress_dialog", Mock(side_effect=update))
    def approve(**kwargs):
        assert get_ident() == ui_thread
        assert kwargs["prepared_manifest"] == prepared
        return 1
    host = SimpleNamespace(frame=frame, app=SimpleNamespace(update_status=Mock()),
                           _approve_current_iteration_package=Mock(side_effect=approve))
    result = runtime.approve_ingest_async(host, target_iter_dir=tmp_path, source_dir=tmp_path,
                                          selected_source_files=[tmp_path / "a.jpg", tmp_path / "copy.jpg"])
    assert entered.wait(2)
    host._approve_current_iteration_package.assert_not_called()
    assert not result["done"] and host._ingest_identity_running
    ready.set()
    deadline = monotonic() + 3
    while not result["done"] and monotonic() < deadline:
        frame.tick()
        sleep(.002)
    assert result["ok"] and result["count"] == 1
    host._approve_current_iteration_package.assert_called_once()
    assert ui._update_ingest_plan_progress_dialog.call_count <= 3
    assert not frame.jobs and frame.binding is None


def test_switching_project_does_not_approve_new_project_after_old_worker(monkeypatch, tmp_path):
    frame = Frame()
    release = Event()
    project = ["demo"]
    def record(**kwargs):
        release.wait(2)
        return tmp_path / "manifest.json"
    manager = SimpleNamespace(get_active_project_name=lambda: project[0], get_current_iteration_num=lambda: 1,
                              record_iteration_ingest=record,
                              load_ingest_manifest=lambda *args: {"selected_images": [{"name": "a.jpg"}]})
    monkeypatch.setattr(runtime, "CAMPAIGN", manager)
    host = SimpleNamespace(frame=frame, _approve_current_iteration_package=Mock())
    result = runtime.approve_ingest_async(host, target_iter_dir=tmp_path, source_dir=tmp_path,
                                          selected_source_files=[tmp_path / "a.jpg"])
    project[0] = "new_project"
    frame.tick()
    release.set()
    assert result["cancelled"] and result["done"]
    host._approve_current_iteration_package.assert_not_called()
    assert not frame.jobs
