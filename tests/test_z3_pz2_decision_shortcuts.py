from types import SimpleNamespace
from pathlib import Path

from auto_annotation_tool.gui import z3_review_runtime


def test_reopen_review_gold_preserves_annotations_and_gt():
    chars = [
        {"character": "A", "bbox": [0, 0, 10, 20]},
        {"character": "1", "bbox": [12, 0, 22, 20]},
    ]
    data = {
        "status": "perfect",
        "ground_truth_text": "A1",
        "ground_truth_source": "manual_z3",
        "characters": chars,
        "review_state": {
            "status": z3_review_runtime.REVIEW_APPROVED,
            "approved_at": "2026-09-27T10:00:00+02:00",
            "approved_reference": {"x": 1},
        },
        "gold_state": {
            "approved": True,
            "candidate": True,
            "excluded": False,
        },
    }
    host = SimpleNamespace(
        preview_metadata={"p1": data},
        _preview_active_pid="p1",
    )

    before_chars = list(data["characters"])
    result = z3_review_runtime.reopen_review_gold(
        host,
        persist=False,
        quiet=True,
        refresh=False,
    )

    assert result["ok"] is True
    assert data["status"] == "needs_fix"
    assert data["review_state"]["status"] == z3_review_runtime.REVIEW_IN_PROGRESS
    assert data["gold_state"]["approved"] is False
    assert data["gold_state"]["candidate"] is False
    assert data["ground_truth_text"] == "A1"
    assert data["characters"] == before_chars


def test_decision_shortcuts_and_help_surfaces_are_synchronized():
    root = Path(__file__).resolve().parents[1]

    events = (root / "auto_annotation_tool/gui/z3_preview_events.py").read_text(
        encoding="utf-8-sig"
    )
    typing = (root / "auto_annotation_tool/gui/z3_preview_typing_runtime.py").read_text(
        encoding="utf-8-sig"
    )
    compass = (root / "auto_annotation_tool/gui/z3_preview_compass_ui.py").read_text(
        encoding="utf-8-sig"
    )
    hud = (root / "auto_annotation_tool/gui/z3_inline_hud.py").read_text(
        encoding="utf-8-sig"
    )

    assert 'if keysym == "r":' in events
    assert 'if keysym == "t":' in events
    assert 'if keysym == "f":' in events
    assert 'if keysym == "o":' not in events
    assert 'if keysym == "n":' not in events
    assert "_on_preview_fit_shortcut(event)" not in events

    assert "R OK" in typing
    assert "T cofnij OK" in typing
    assert "F wyklucz" in typing

    assert '"title": "Decyzja tablicy"' in compass
    assert '"tokens": ["R"]' in compass
    assert '"tokens": ["T"]' in compass
    assert '"tokens": ["F"]' in compass

    assert "Brak GT z Z2 · O zapisze" not in hud
    assert "preview_inline_hud_action" not in hud
    assert 'number_text = "GT z Z2: brak"' in hud
