from types import SimpleNamespace

from auto_annotation_tool.gui.z3_preview_list_ui import (
    get_preview_sort_priority,
    preview_has_inherited_z2_gt,
)


def test_inherited_z2_gt_requires_manual_z2_and_nonempty_gt():
    assert preview_has_inherited_z2_gt({
        "ground_truth_text": "WI1234A",
        "ground_truth_source": "manual_z2",
    }) is True
    assert preview_has_inherited_z2_gt({
        "ground_truth_text": "WI1234A",
        "ground_truth_source": "manual_z3",
    }) is False
    assert preview_has_inherited_z2_gt({
        "source_expected_text": "WI1234A",
        "source_expected_text_source": "filename_order",
    }) is False
    assert preview_has_inherited_z2_gt({
        "ground_truth_text": "",
        "ground_truth_source": "manual_z2",
    }) is False


def test_nested_manual_z2_gt_is_recognized():
    data = {
        "plate_attributes": {
            "ground_truth_text": "KR12345",
            "ground_truth_source": "manual_z2",
        }
    }
    assert preview_has_inherited_z2_gt(data) is True


def test_gt_z2_sort_puts_inherited_gt_first_and_preserves_source_order():
    host = SimpleNamespace(
        _get_preview_sort_mode_key=lambda: "GT_Z2",
        _get_perfect_strategy_bucket=lambda _data: "",
    )
    rows = [
        ("a", {"ground_truth_text": "", "ground_truth_source": ""}),
        ("b", {"ground_truth_text": "WX1234", "ground_truth_source": "manual_z2"}),
        ("c", {"ground_truth_text": "AB1234", "ground_truth_source": "manual_z3"}),
        ("d", {"ground_truth_text": "PO9876", "ground_truth_source": "manual_z2"}),
    ]
    ordered = sorted(
        enumerate(rows),
        key=lambda item: get_preview_sort_priority(
            host,
            item[1][0],
            item[1][1],
            item[0],
        ),
    )
    assert [row[1][0] for row in ordered] == ["b", "d", "a", "c"]


def test_sort_contract_contains_gt_z2_option():
    from auto_annotation_tool.gui.tab_character_annotation import (
        PREVIEW_SORT_COLOR_KEYS,
        PREVIEW_SORT_LABELS,
    )
    assert PREVIEW_SORT_LABELS["GT_Z2"] == "Po GT z Z2"
    assert PREVIEW_SORT_COLOR_KEYS["GT_Z2"] == "info"
