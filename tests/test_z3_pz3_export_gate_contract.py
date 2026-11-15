from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_annotation_tool.gui import z3_goldpack_ui as gold
from auto_annotation_tool.gui import z3_shared_ui as shared


class _Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value


def test_runtime_export_is_blocked_by_canonical_readiness_before_writing_dataset():
    messages = []
    dataset_root = Mock(side_effect=AssertionError("dataset root must not be touched"))
    collect = Mock(side_effect=AssertionError("candidate export must not run"))

    host = SimpleNamespace(
        export_console=object(),
        app=SimpleNamespace(update_status=Mock()),
        _get_selected_gold_export_strategy_buckets=lambda: {"yolo_exact"},
        _format_selected_gold_export_strategy_labels=lambda: "YOLO exact",
        _get_selected_gold_export_source_buckets=lambda: {"auto_preview"},
        _format_selected_gold_export_source_labels=lambda: "Auto",
        _get_step3_yolo_export_readiness_snapshot=lambda **kwargs: {
            "ok": False,
            "reason": "no_exportable_yolo_candidates",
            "message": "Bramka E3 pozostaje zamknięta. Minimum kampanii to 10, obecnie 1.",
            "selected_plate_count": 1,
            "selected_char_count": 7,
            "min_exportable_plate_count": 10,
        },
        _set_console_text=lambda _console, text: messages.append(text),
        _get_step3_datasets_root_dir=dataset_root,
        _collect_gold_export_plate_candidates=collect,
        _get_gold_export_split_percentages=Mock(
            side_effect=AssertionError("split must not be read before readiness")
        ),
    )

    gold.run_yolo_gold_export(
        host,
        char_class_alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        gold_source_buckets=(),
        gold_source_labels={},
    )

    assert messages
    assert "zablokowany" in messages[-1].lower()
    assert "minimum kampanii" in messages[-1].lower()
    dataset_root.assert_not_called()
    collect.assert_not_called()


def test_pz3_view_model_uses_canonical_readiness_not_positive_count_only():
    host = SimpleNamespace(
        pz3_existing_dataset_var=_Var(""),
        _get_selected_gold_export_strategy_buckets=lambda: {"yolo_exact"},
        _get_selected_gold_export_source_buckets=lambda: {"auto_preview"},
        _build_campaign_aware_gold_export_counts=lambda **kwargs: {
            "selected_plate_count": 1,
            "selected_char_count": 7,
        },
        _get_step3_yolo_export_readiness_snapshot=lambda **kwargs: {
            "ok": False,
            "message": "Bramka E3 pozostaje zamknięta. Minimum kampanii to 10, obecnie 1.",
            "selected_plate_count": 1,
            "selected_char_count": 7,
            "min_exportable_plate_count": 10,
        },
        _get_active_preview_context=lambda: {
            "ready": True,
            "preview_dir": None,
            "message": "",
            "plate_count": 1,
        },
        _read_step3_export_summary=lambda: {},
    )

    with patch.object(shared, "is_step3_campaign_runtime", return_value=True):
        vm = shared.build_step3_pz3_dataset_mode_view_model(host)

    assert vm.primary_export_enabled is False
    assert "minimum kampanii" in vm.action_hint.lower()
    assert "minimum kampanii" in vm.source_next_text.lower()


def test_pz3_view_model_enables_export_when_canonical_readiness_is_ok():
    host = SimpleNamespace(
        pz3_existing_dataset_var=_Var(""),
        _get_selected_gold_export_strategy_buckets=lambda: {"yolo_exact"},
        _get_selected_gold_export_source_buckets=lambda: {"auto_preview"},
        _build_campaign_aware_gold_export_counts=lambda **kwargs: {
            "selected_plate_count": 10,
            "selected_char_count": 70,
        },
        _get_step3_yolo_export_readiness_snapshot=lambda **kwargs: {
            "ok": True,
            "message": "",
            "selected_plate_count": 10,
            "selected_char_count": 70,
            "min_exportable_plate_count": 10,
        },
        _get_active_preview_context=lambda: {
            "ready": True,
            "preview_dir": None,
            "message": "",
            "plate_count": 10,
        },
        _read_step3_export_summary=lambda: {},
    )

    with patch.object(shared, "is_step3_campaign_runtime", return_value=True):
        vm = shared.build_step3_pz3_dataset_mode_view_model(host)

    assert vm.primary_export_enabled is True
