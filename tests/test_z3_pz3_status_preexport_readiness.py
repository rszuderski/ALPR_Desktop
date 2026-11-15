from pathlib import Path
from types import SimpleNamespace

from auto_annotation_tool.gui import z3_campaign_flow as flow


def test_pz3_status_before_export_uses_annotation_readiness_not_training_readiness(monkeypatch, tmp_path):
    host = SimpleNamespace(
        _read_step3_export_summary=lambda: {},
        _get_campaign_step3_annotation_readiness=lambda: {
            "ok": True,
            "exportable_plate_count": 10,
            "exportable_char_count": 70,
            "perfect_count": 10,
        },
        _get_campaign_step3_training_readiness=lambda: (_ for _ in ()).throw(
            AssertionError("pre-export status must not ask for training readiness")
        ),
        _get_active_preview_context=lambda: {
            "ready": True,
            "preview_dir": tmp_path / "run_001_20260926_183227",
        },
        _get_inline_status_widget_snapshot=lambda _widget, *, fallback_text, fallback_tone: (
            fallback_text,
            fallback_tone,
        ),
        export_console=None,
        import_console=None,
    )

    monkeypatch.setattr(
        flow,
        "build_step3_pz3_path_selection_view_model",
        lambda _host: SimpleNamespace(show_status_section=True),
    )
    monkeypatch.setattr(
        flow,
        "build_step3_finish_action_view_model",
        lambda _host: SimpleNamespace(enabled=False),
    )
    monkeypatch.setattr(
        flow,
        "_build_step3_pz3_dataset_iteration_context",
        lambda _host, _summary: {
            "has_dataset": False,
            "current_iteration": 1,
            "origin_iteration": 0,
            "is_current_iteration": False,
            "interrupted_pz3": False,
        },
    )

    vm = flow.build_step3_pz3_status_panel_view_model(host)

    assert vm.dataset_row.text == "Dataset: można go utworzyć z gotowego runu PZ2."
    assert vm.dataset_row.tone == "success"
    assert vm.export_row.text == "Eksport: uruchom tworzenie źródłowego datasetu znaków."
    assert vm.readiness_row.text == "Run jest gotowy; brakuje utworzenia i zapisania datasetu znaków."
    assert "popraw anotacje znaków w PZ2" not in (
        vm.dataset_row.text + vm.export_row.text + vm.readiness_row.text
    ).lower()
