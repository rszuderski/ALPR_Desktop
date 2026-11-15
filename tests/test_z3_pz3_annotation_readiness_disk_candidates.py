from types import SimpleNamespace

from auto_annotation_tool.gui import z3_readiness as readiness


def _host(*, selected_plates: int, selected_chars: int, preview_perfect: int):
    return SimpleNamespace(
        _get_selected_gold_export_strategy_buckets=lambda: {"other_perfect"},
        _get_selected_gold_export_source_buckets=lambda: {"local_manual"},
        _get_step3_yolo_export_readiness_snapshot=lambda **_kwargs: {
            "ok": bool(selected_plates >= 10 and selected_chars > 0),
            "selected_plate_count": selected_plates,
            "selected_char_count": selected_chars,
            "min_exportable_plate_count": 10,
            "missing_exportable_plate_count": max(0, 10 - selected_plates),
            "message": "",
            "gold_source_contract_schema": "gold-source.v1",
            "gold_source_contract_sha256": "abc123",
        },
        _count_preview_statuses=lambda: {
            "perfect": preview_perfect,
            "needs_fix": 0,
            "unknown": 0,
            "total": preview_perfect,
        },
    )


def test_annotation_readiness_uses_exportable_disk_candidates_when_preview_memory_is_empty():
    host = _host(selected_plates=10, selected_chars=70, preview_perfect=0)

    result = readiness.get_campaign_step3_annotation_readiness(host)

    assert result["ok"] is True
    assert result["reason"] == ""
    assert result["perfect_count"] == 10
    assert result["exportable_plate_count"] == 10
    assert result["exportable_char_count"] == 70


def test_annotation_readiness_does_not_invent_missing_perfects():
    host = _host(selected_plates=8, selected_chars=56, preview_perfect=0)

    result = readiness.get_campaign_step3_annotation_readiness(host)

    assert result["ok"] is False
    assert result["reason"] == "missing_perfect_plates"
    assert result["perfect_count"] == 8
