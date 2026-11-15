from types import SimpleNamespace
from unittest.mock import Mock

from auto_annotation_tool.gui import z3_preview_events as events
from auto_annotation_tool.gui import z3_preview_ui as preview_ui
from auto_annotation_tool.gui import z3_plate_gt_runtime as gt_runtime


def _overlay_host():
    return SimpleNamespace(
        app=SimpleNamespace(
            palette={
                'panel': '#252526',
                'border': '#3c3c3c',
                'muted': '#8b949e',
                'success': '#2ecc71',
                'error': '#e74c3c',
            }
        ),
        _preview_active_pid='plate_000001',
        _get_preview_status_presentation=lambda **kwargs: {
            'status': 'needs_fix',
            'expected_texts': ['WI905PW'],
            'candidate_text': '',
            'ready_for_approval': False,
        },
        _characters_to_display_rows=lambda chars, data=None: [],
        _get_preview_plate_layout_dock_text=lambda data: ('1R', 'success'),
        _preview_layout_separator_conflicts_with_chars=lambda data, chars: False,
    )


def test_gt_badge_is_left_click_action():
    result = preview_ui._build_preview_canvas_status_badge_specs(
        _overlay_host(),
        1000,
        data={'plate_id': 'plate_000001', 'characters': []},
        box_chars=[],
    )
    gt_badges = [
        item for item in result['neutral_badges']
        if str(item.get('text', '')).startswith('GT:')
    ]
    assert len(gt_badges) == 1
    tags = tuple(gt_badges[0].get('tags') or ())
    assert 'preview_overlay_action' in tags
    assert 'preview_action::edit_plate_gt' in tags


def test_left_click_gt_hud_dispatches_existing_gt_editor(monkeypatch):
    host = SimpleNamespace(
        _preview_review_batch_running=False,
        _focus_preview_canvas=Mock(),
        _extract_preview_action_from_current_item=lambda: 'edit_plate_gt',
    )
    editor = Mock()
    monkeypatch.setattr(gt_runtime, 'edit_active_plate_ground_truth', editor)
    event = SimpleNamespace(x=10, y=10)
    assert events.on_preview_canvas_press(host, event) == 'break'
    editor.assert_called_once_with(host)
    host._focus_preview_canvas.assert_called_once()
