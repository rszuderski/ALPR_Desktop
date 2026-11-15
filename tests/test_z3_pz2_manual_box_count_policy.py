from types import SimpleNamespace

from auto_annotation_tool.gui.z3_gt_box_policy import can_add_character_box


def _host():
    messages = []
    host = SimpleNamespace()
    host.messages = messages
    host._update_preview_edit_status = (
        lambda message, tone=None: messages.append((message, tone))
    )
    return host


def _chars(count):
    return [
        {"character": "A", "bbox": [i * 10, 0, i * 10 + 8, 20]}
        for i in range(count)
    ]


def test_manual_z3_gt_does_not_block_operator_from_adding_extra_box():
    host = _host()
    data = {
        "ground_truth_text": "B955ET",
        "ground_truth_source": "manual_z3",
        "characters": _chars(6),
    }
    assert can_add_character_box(host, data) is True
    assert host.messages == []


def test_filename_or_unprotected_gt_does_not_become_hard_box_limit():
    host = _host()
    data = {
        "ground_truth_text": "B955ET",
        "ground_truth_source": "filename_order",
        "characters": _chars(6),
    }
    assert can_add_character_box(host, data) is True
    assert host.messages == []


def test_z2_gt_does_not_block_manual_add_at_reference_count():
    host = _host()
    data = {
        "ground_truth_text": "ABC123",
        "ground_truth_source": "manual_z2",
        "characters": _chars(6),
    }
    assert can_add_character_box(host, data) is True
    assert host.messages == []


def test_z2_gt_allows_box_until_expected_count_is_reached():
    host = _host()
    data = {
        "ground_truth_text": "ABC123",
        "ground_truth_source": "manual_z2",
        "characters": _chars(5),
    }
    assert can_add_character_box(host, data) is True
    assert host.messages == []


def test_nested_z2_gt_source_also_allows_manual_addition():
    host = _host()
    data = {
        "ground_truth_text": "ABC123",
        "plate_attributes": {"ground_truth_source": "manual_z2"},
        "characters": _chars(6),
    }
    assert can_add_character_box(host, data, notify=False) is True
    assert host.messages == []
