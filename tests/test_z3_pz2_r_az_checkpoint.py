from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_annotation_tool.gui import z3_review_runtime


def _review_host(data):
    host = SimpleNamespace(
        preview_metadata={"plate_000001": data},
        _preview_active_pid="plate_000001",
    )
    host._ensure_plate_source_metadata = Mock()
    host._get_plate_source_bucket = Mock(return_value="local_manual")

    host._characters_to_text = lambda chars, data=None: "".join(
        str(item.get("character", "") or "") for item in chars
    )
    return host


def test_confirm_review_gold_persists_az_without_full_refresh():
    data = {
        "review_state": {
            "status": z3_review_runtime.REVIEW_IN_PROGRESS,
        },
        "characters": [
            {
                "character": "A",
                "bbox": [1, 1, 10, 20],
            }
        ],
        "gold_state": {
            "excluded": False,
        },
        "status": "needs_fix",
    }
    host = _review_host(data)

    with patch.object(
        z3_review_runtime,
        "get_review_quality_status",
        return_value="perfect",
    ), patch.object(
        z3_review_runtime,
        "plate_gt",
        return_value="A",
    ), patch.object(
        z3_review_runtime,
        "_refresh_after_change",
    ) as refresh, patch.object(
        z3_review_runtime,
        "_persist_review_az_revision_best_effort",
    ) as save_az:
        result = z3_review_runtime.confirm_review_gold(
            host,
            persist=True,
            quiet=True,
            refresh=False,
        )

    assert result["ok"] is True
    refresh.assert_not_called()
    save_az.assert_called_once_with(
        host,
        "plate_000001",
        data,
        event="review_approved",
    )


def test_r_shortcut_keeps_light_refresh_but_requests_persistent_az_checkpoint():
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "auto_annotation_tool" / "gui" / "z3_preview_events.py"
    ).read_text(encoding="utf-8-sig")

    r_start = source.find('    if keysym == "r":')
    assert r_start >= 0

    t_start = source.find('    if keysym == "t":', r_start)
    assert t_start > r_start

    r_block = source[r_start:t_start]

    assert "self._confirm_review_gold(" in r_block
    assert "quiet=True" in r_block
    assert "persist=True" in r_block
    assert "refresh=False" in r_block
    assert "persist=False" not in r_block
