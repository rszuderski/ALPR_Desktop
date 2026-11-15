from types import SimpleNamespace
from unittest.mock import patch

from auto_annotation_tool.gui import z3_preview_status_ui as status_ui


def _host(*, plates: int, chars: int, minimum: int, ready: bool):
    return SimpleNamespace(
        _step3_linear_mode=True,
        _get_selected_gold_export_strategy_buckets=lambda: set(),
        _get_selected_gold_export_source_buckets=lambda: set(),
        _get_step3_yolo_export_readiness_snapshot=lambda **_kwargs: {
            "selected_plate_count": plates,
            "selected_char_count": chars,
            "min_exportable_plate_count": minimum,
            "ok": ready,
        },
    )


def test_campaign_pz2_ready_points_only_to_immediate_return_to_graph():
    host = _host(plates=10, chars=70, minimum=10, ready=True)
    counts = {
        "perfect": 10,
        "needs_fix": 0,
        "unknown": 0,
        "total": 10,
    }

    with patch.object(status_ui.CAMPAIGN, "get_active_project_name", return_value="AZ007C3_SOURCE"):
        snapshot = status_ui.get_preview_repair_progress_snapshot(host, counts)

    assert snapshot["summary"] == "PZ2 gotowe: wszystkie tablice zostały zatwierdzone."
    assert "Zapisz PZ2 i wróć do pracy T05" in snapshot["details"]

    combined = f'{snapshot["summary"]} {snapshot["details"]}'
    for forbidden in ("PZ3", "Z4", "T06", "Krok 2"):
        assert forbidden not in combined


def test_campaign_pz2_incomplete_keeps_guidance_inside_pz2():
    host = _host(plates=9, chars=63, minimum=10, ready=False)
    counts = {
        "perfect": 9,
        "needs_fix": 1,
        "unknown": 0,
        "total": 10,
    }

    with patch.object(status_ui.CAMPAIGN, "get_active_project_name", return_value="AZ007C3_SOURCE"):
        snapshot = status_ui.get_preview_repair_progress_snapshot(host, counts)

    assert snapshot["summary"] == (
        "PZ2 w toku: wymagane minimum zatwierdzonych tablic nie jest jeszcze spełnione."
    )
    assert "Kontynuuj sprawdzanie i poprawianie tablic w PZ2." in snapshot["details"]

    combined = f'{snapshot["summary"]} {snapshot["details"]}'
    for forbidden in ("PZ3", "Z4", "T06", "Krok 2"):
        assert forbidden not in combined

def test_campaign_pz2_return_label_names_the_immediate_t05_work_modal():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    detection_ui = (
        root
        / "auto_annotation_tool"
        / "gui"
        / "z3_detection_tab_ui.py"
    ).read_text(encoding="utf-8-sig")
    status_ui = (
        root
        / "auto_annotation_tool"
        / "gui"
        / "z3_preview_status_ui.py"
    ).read_text(encoding="utf-8-sig")

    expected = "Zapisz PZ2 i wróć do pracy T05"
    assert expected in detection_ui
    assert expected in status_ui
    assert "Zapisz PZ2 i wróć do grafu" not in detection_ui
    assert "Następna akcja: „Zapisz PZ2 i wróć do grafu”." not in status_ui
