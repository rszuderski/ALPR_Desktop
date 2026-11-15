from types import SimpleNamespace

from auto_annotation_tool.gui.z3_preview_list_ui import (
    get_preview_sort_priority,
    get_sorted_preview_plate_ids,
)


def _host(metadata):
    return SimpleNamespace(
        preview_metadata=metadata,
        _get_preview_sort_mode_key=lambda: "DEFAULT",
        _get_perfect_strategy_bucket=lambda _data: "other_perfect",
        _get_review_quality_status=lambda _data: "perfect",
    )


def test_default_sort_separates_regular_red_and_n_into_three_groups():
    metadata = {
        "n_first": {
            "status": "needs_fix",
            "gold_state": {"excluded": True},
        },
        "red_first": {
            "status": "needs_fix",
            "gold_state": {"excluded": False},
        },
        "regular_ok": {
            "status": "perfect",
            "gold_state": {"excluded": False},
        },
        "regular_ready": {
            "status": "needs_fix",
            "review_state": {"status": "in_progress"},
            "gold_state": {"excluded": False},
        },
        "red_second": {
            "status": "needs_fix",
            "gold_state": {"excluded": False},
        },
        "n_second": {
            "status": "needs_fix",
            "gold_state": {"excluded": True},
        },
    }
    host = _host(metadata)
    source_order = list(metadata)

    assert get_sorted_preview_plate_ids(host, source_order) == [
        "regular_ok",
        "regular_ready",
        "red_first",
        "red_second",
        "n_first",
        "n_second",
    ]


def test_default_sort_priority_is_regular_then_red_then_n():
    host = _host({})

    regular = {
        "status": "perfect",
        "gold_state": {"excluded": False},
    }
    red = {
        "status": "needs_fix",
        "gold_state": {"excluded": False},
    }
    excluded = {
        "status": "needs_fix",
        "gold_state": {"excluded": True},
    }

    assert get_preview_sort_priority(host, "regular", regular, 7)[0] == 0
    assert get_preview_sort_priority(host, "red", red, 7)[0] == 1
    assert get_preview_sort_priority(host, "n", excluded, 7)[0] == 2


def test_default_sort_preserves_source_order_inside_each_group():
    metadata = {
        "red_a": {"status": "needs_fix"},
        "regular_a": {"status": "perfect"},
        "n_a": {"status": "needs_fix", "gold_state": {"excluded": True}},
        "regular_b": {"status": "perfect"},
        "red_b": {"status": "needs_fix"},
        "n_b": {"status": "needs_fix", "gold_state": {"excluded": True}},
    }
    host = _host(metadata)

    assert get_sorted_preview_plate_ids(host, list(metadata)) == [
        "regular_a",
        "regular_b",
        "red_a",
        "red_b",
        "n_a",
        "n_b",
    ]
