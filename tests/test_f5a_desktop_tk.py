"""Existing Canvas acceptance on private legacy-format F5A synthetic data."""
from pathlib import Path
import json
import time
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import test_mobile_p2b2_canvas_gui as native
from test_f5a_desktop_metadata import legacy_f5a_session
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking.mobile_session_metrics import current, result_key


def open_private_window(tmp_path, monkeypatch, theme='light', size=(1649, 896)):
    from auto_annotation_tool.gui import mobile_canvas_editor as editor
    session = legacy_f5a_session(tmp_path/'source')
    original = editor.palette
    monkeypatch.setattr(editor, 'palette', lambda w, b: original(w, '#FFFFFF' if theme == 'light' else '#17232D'))
    monkeypatch.setattr(native, 'source_session', lambda _: session)
    generator = native.gui.__wrapped__(tmp_path, monkeypatch)
    root, host, ed, session = next(generator)
    host.window.title('F5A/12B — TEST SYNTHETYCZNY')
    host.window.geometry(f'{size[0]}x{size[1]}+20+20')
    host.window.update()
    native.pump(root, lambda: not host._saving and not ed.in_refresh and current(
        host.stats.get('live_session_metrics_v2'), result_key(session)), seconds=30)
    return generator, root, host, ed, session


@pytest.mark.parametrize('theme,size', [('light', (1649, 896)), ('dark', (1649, 896)),
                                       ('light', (1180, 760)), ('dark', (1180, 760))])
def test_native_labels_and_existing_space_text_undo(tmp_path, monkeypatch, theme, size):
    generator, root, h, ed, session = open_private_window(tmp_path, monkeypatch, theme, size)
    try:
        assert ed.available()
        assert 'MT/ROI' in h.progress_label.cget('text')
        assert h.stats['live_session_metrics_v2']['experiment_family'] == 'F5A'
        assert ed.contexts.live_metrics.status.cget('text').startswith('MT/ROI · ')
        native.select(h, ed, (225, 175))
        native.key(h, 'space')
        native.pump(root, lambda: not h._saving and not ed.in_refresh, seconds=30)
        assert core.has_scene(session, 'mt-one')
        # The same inline editor and native bindings are retained.
        ed.start_entry('GT:a0')
        assert ed.entry is not None
        ed.entry.delete(0, 'end'); ed.entry.insert(0, 'SYN1234')
        ed.entry_enter()
        native.pump(root, lambda: not h._saving, seconds=30)
        assert core.scene(session, 'mt-one')['objects'][0]['gt_text'] == 'SYN1234'
        revision = session.review.review_revision
        ed.undo()
        native.pump(root, lambda: not h._saving, seconds=30)
        assert session.review.review_revision > revision
        ed.undo(redo=True)
        native.pump(root, lambda: not h._saving, seconds=30)
        assert core.scene(session, 'mt-one')['objects'][0]['gt_text'] == 'SYN1234'
        assert h.window.winfo_exists()
    finally:
        try: next(generator)
        except StopIteration: pass


def test_legacy_canvas_keeps_existing_metric_caption(tmp_path, monkeypatch):
    generator = native.gui.__wrapped__(tmp_path, monkeypatch)
    root, host, ed, session = next(generator)
    try:
        native.pump(root, lambda: not host._saving and not ed.in_refresh and current(
            host.stats.get('live_session_metrics_v2'), result_key(session)), seconds=30)
        assert 'experiment_family' not in host.stats['live_session_metrics_v2']
        assert ed.contexts.live_metrics.status.cget('text').startswith('MT v2 · ASSISTED · ')
    finally:
        try: next(generator)
        except StopIteration: pass


def preview(directory):
    """Keep the existing private test window available for Computer Use QA."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch = pytest.MonkeyPatch()
    generator, root, h, ed, session = open_private_window(directory, monkeypatch)
    (directory/'ready.json').write_text(json.dumps({'title': h.window.title(), 'session_id': session.session_id}), encoding='utf-8')
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline and h.window.winfo_exists() and not (directory/'finish').exists():
            root.update(); time.sleep(.03)
    finally:
        try: next(generator)
        except StopIteration: pass
        monkeypatch.undo()


if __name__ == '__main__':
    try:
        preview(sys.argv[1])
    except Exception:
        import traceback
        Path(sys.argv[1], 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
