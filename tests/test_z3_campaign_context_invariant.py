from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import z3_shared_ui


def test_active_project_overrides_stale_free_mode_flags(monkeypatch):
    monkeypatch.setattr(
        z3_shared_ui.CAMPAIGN,
        "get_active_project_name",
        lambda: "train_yolo26n_pose",
    )

    app = SimpleNamespace(
        campaign_free_mode=True,
        set_campaign_mode=Mock(),
    )
    host = SimpleNamespace(
        app=app,
        _step3_linear_mode=False,
    )

    assert z3_shared_ui.is_step3_campaign_runtime(host) is True
    assert host._step3_linear_mode is True
    assert app.campaign_free_mode is False
    app.set_campaign_mode.assert_called_once_with(True)


def test_no_active_project_remains_free_mode(monkeypatch):
    monkeypatch.setattr(
        z3_shared_ui.CAMPAIGN,
        "get_active_project_name",
        lambda: "",
    )

    app = SimpleNamespace(
        campaign_free_mode=True,
        set_campaign_mode=Mock(),
    )
    host = SimpleNamespace(
        app=app,
        _step3_linear_mode=False,
    )

    assert z3_shared_ui.is_step3_campaign_runtime(host) is False
    assert host._step3_linear_mode is False
    assert app.campaign_free_mode is True
    app.set_campaign_mode.assert_not_called()


def test_z3_init_binds_initial_runtime_to_active_project():
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "auto_annotation_tool"
        / "gui"
        / "z3_init_runtime.py"
    ).read_text(encoding="utf-8-sig")

    assert "_active_campaign_project" in source
    assert "self._step3_linear_mode = bool(_active_campaign_project)" in source
    assert "self.app.campaign_free_mode = False" in source


def test_campaign_entry_defensively_repairs_context():
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "auto_annotation_tool"
        / "gui"
        / "z3_campaign_flow.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def open_campaign_step3_entry(")
    end = source.find("\ndef ", start + 1)
    body = source[start:] if end < 0 else source[start:end]

    assert "host._step3_linear_mode = True" in body
    assert "host.app.campaign_free_mode = False" in body
