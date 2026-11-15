"""AKT80: no extra MT/raw outlines while GT frame is actively edited."""
from test_z4_canvas_akt70_selection_drag import setup, key_event


def _tags(canvas):
    return [tuple(op[2].get('tags', ())) for op in canvas.ops]


def test_mt_overlay_hidden_while_selected_gt_is_edited():
    _mod, editor, canvas, _host, _scene, _messages, _ = setup()
    editor.overlay()
    assert any('mt_original' in tags for tags in _tags(canvas))
    editor.select_mode(key_event(canvas))
    tags = _tags(canvas)
    assert not any('mt_original' in t for t in tags)
    assert any('operator_quad' in t for t in tags)
    assert any('corner_handle' in t for t in tags)
