from types import SimpleNamespace
from pathlib import Path

from auto_annotation_tool.gui.z3_preview_ui import get_preview_status_presentation
from auto_annotation_tool.gui.z3_review_runtime import get_review_quality_status


def _chars(text):
    return [
        {"character": ch, "bbox": [idx * 10, 0, idx * 10 + 8, 20]}
        for idx, ch in enumerate(text)
    ]


def _presentation_host(*, live_status="needs_fix", quality="perfect"):
    return SimpleNamespace(
        _get_preview_active_data=lambda create=False: {},
        _get_preview_live_status=lambda **kwargs: live_status,
        _sanitize_preview_char_symbol=lambda value: str(value or "").strip().upper()[:1],
        _characters_to_text=lambda chars, data=None: "".join(
            str(item.get("character", "")) for item in chars
        ),
        _get_preview_ground_truth_text=lambda data=None: str(
            (data or {}).get("ground_truth_text", "") or ""
        ),
        _get_review_quality_status=lambda data, chars=None: quality,
    )


def test_no_z2_gt_never_infers_missing_frame_count():
    host = _presentation_host()
    data = {
        "ground_truth_text": "OLD12345",
        "ground_truth_source": "manual_z3",
        "review_state": {"status": "in_progress"},
        "characters": _chars("NEW1234"),
    }

    state = get_preview_status_presentation(host, data, data["characters"], "plate_1")

    assert state["operator_decides_frame_completeness"] is True
    assert state["frame_expected_count"] == 0
    assert state["decision_text"] == "DO KONTROLI"
    assert state["ready_for_approval"] is True
    assert "Brak GT z Z2" in state["action_text"]


def test_z2_gt_is_the_only_hard_frame_count_reference():
    host = _presentation_host(quality="needs_fix")
    data = {
        "ground_truth_text": "RCT962EE",
        "ground_truth_source": "manual_z2",
        "review_state": {"status": "in_progress"},
        "characters": _chars("CT9662E"),
    }

    state = get_preview_status_presentation(host, data, data["characters"], "plate_1")

    assert state["operator_decides_frame_completeness"] is False
    assert state["frame_expected_count"] == 8
    assert state["decision_text"] == "DO KOREKTY"
    assert "GT z Z2 ma 8 znaków" in state["action_text"]


def test_existing_frame_without_symbol_is_program_detectable():
    host = _presentation_host(quality="needs_fix")
    chars = _chars("ABC")
    chars[1]["character"] = ""
    data = {
        "ground_truth_text": "",
        "ground_truth_source": "",
        "review_state": {"status": "in_progress"},
        "characters": chars,
    }

    state = get_preview_status_presentation(host, data, chars, "plate_1")

    assert state["unlabeled_boxes"] == 1
    assert state["decision_text"] == "DO KOREKTY"
    assert "istniejących ramkach" in state["action_text"]


def test_review_quality_uses_operator_reading_when_gt_is_not_from_z2():
    chars = _chars("ABC123")
    data = {
        "ground_truth_text": "OLD999",
        "ground_truth_source": "manual_z3",
        "plate_attributes": {
            "ground_truth_text": "OLD999",
            "ground_truth_source": "manual_z3",
        },
        "characters": chars,
        "review_state": {"status": "in_progress"},
    }
    host = SimpleNamespace(
        _characters_to_text=lambda records, data=None: "".join(
            str(item.get("character", "")) for item in records
        ),
        _derive_preview_status_from_data=lambda probe, records: (
            "perfect"
            if probe.get("ground_truth_text") == "ABC123"
            and (probe.get("plate_attributes") or {}).get("ground_truth_text") == "ABC123"
            else "needs_fix"
        ),
    )

    assert get_review_quality_status(host, data, chars) == "perfect"


def test_review_quality_preserves_z2_reference():
    chars = _chars("ABC123")
    data = {
        "ground_truth_text": "OLD999",
        "ground_truth_source": "manual_z2",
        "plate_attributes": {
            "ground_truth_text": "OLD999",
            "ground_truth_source": "manual_z2",
        },
        "characters": chars,
        "review_state": {"status": "in_progress"},
    }
    host = SimpleNamespace(
        _characters_to_text=lambda records, data=None: "".join(
            str(item.get("character", "")) for item in records
        ),
        _derive_preview_status_from_data=lambda probe, records: (
            "perfect" if probe.get("ground_truth_text") == "ABC123" else "needs_fix"
        ),
    )

    assert get_review_quality_status(host, data, chars) == "needs_fix"


def test_inline_hud_copy_distinguishes_z2_gt_from_operator_decision():
    source = (
        Path(__file__).resolve().parents[1]
        / "auto_annotation_tool"
        / "gui"
        / "z3_inline_hud.py"
    ).read_text(encoding="utf-8-sig")

    assert "GT z Z2:" in source
    assert 'number_text = "GT z Z2: brak"' in source
    assert "Brak GT z Z2 · O zapisze sprawdzony odczyt jako GT Z3" not in source
    assert "preview_inline_hud_action" not in source
    assert "Decyzja:" in source
