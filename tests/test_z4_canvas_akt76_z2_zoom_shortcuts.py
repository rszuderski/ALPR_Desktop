"""AKT76 — real Tk shortcut dispatch + actual MobileSampleReviewWindow renderer.

No report, GT or .alprsession mutation. Uses a 1000x1000 synthetic image and
checks the Canvas operator workflow, including fallback from a focus widget.
"""
from __future__ import annotations

from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk

import pytest
from PIL import Image

from auto_annotation_tool.gui.mobile_canvas_editor import CanvasReviewEditor
from auto_annotation_tool.gui.z4_mobile_sample_review import MobileSampleReviewWindow


def _window():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('Cannot create Tk window in this environment')
    root.geometry('1100x760+90+90')
    tree = ttk.Treeview(root, height=2)
    tree.insert('', 'end', iid='scene', text='Obraz 19')
    tree.pack(fill='x')
    canvas = tk.Canvas(root, bg='#101820', height=600, width=900, takefocus=1)
    canvas.pack(fill='both', expand=True)
    root.update()
    scene = {
        'image_entry': 'source.png', 'image_size': (1000, 1000),
        'roi': (0, 0, 1000, 1000),
        'parents': [{
            'parent_id': 'p1',
            'quad': [[320., 430.], [480., 430.], [480., 470.], [320., 470.]],
            'mz_prediction': 'EZG83823', 'detection_index': 0,
        }],
        'objects': [{
            'object_id': 'o1', 'parent_id': 'p1',
            'quad': [[320., 430.], [480., 430.], [480., 470.], [320., 470.]],
            'post_nms_suppressed': False, 'gt_text': 'EZG83823',
            'error': False, 'text_exclusion_reason': '',
        }],
    }
    logs = []
    host = SimpleNamespace(
        window=root, canvas=canvas, _image=Image.new('RGB', (1000, 1000), '#1a4554'),
        _image_entry='source.png', _image_source_size=(1000,1000),
        _view_zoom=1.0, _view_center=(500., 500.),
        _view_fit_scale=1.0, _view_pixel_scale=1.0,
        _view_drag_at=None, current_invocation='G', _closed=False, _saving=False,
        session=SimpleNamespace(), _photo=None,
        status_var=SimpleNamespace(set=logs.append),
        zoom_value=SimpleNamespace(configure=lambda **kwargs: None),
        image_help=SimpleNamespace(configure=lambda **kwargs: None),
    )
    # Bind the real image painter rather than a placeholder zoom counter.
    host._paint_image = MobileSampleReviewWindow._paint_image.__get__(host)
    host.fit_image = lambda: _fit_full(host)
    editor = CanvasReviewEditor.__new__(CanvasReviewEditor)
    values = dict(
        h=host,c=canvas,mode='select',selected='',drag=None,_click_candidate=None,
        entry=None,entry_target=None,pointer=None,pointer_canvas=None,points=[],hover='',
        badges={},card_positions={},paint_scheduled=False,paint_after=None,
        show_roi_overlay=True,plate_fit_active=False,plate_fit_gid='',plate_fit_previous=None,
        shortcut_tag=None,shortcut_original_tags={},selection_key_down=False,
        selection_release_after=None,fit_key_down=False,fit_release_after=None,
        refresh_after=None,entry_after=None,tree_branch='',skey='G',read_error='',
    )
    for key,value in values.items():
        setattr(editor, key, value)
    editor._read = lambda:scene
    editor.sync = lambda:None
    editor.clear_tree_selection = lambda:None
    editor.finish_entry = lambda:True
    editor.message = logs.append
    editor.request_overlay = editor.overlay
    host._canvas_editor = editor
    host._paint_image()
    editor.bindings()
    root.update()
    return root,tree,canvas,host,editor,scene,logs


def _fit_full(host):
    host._view_zoom=1.0
    host._view_center=(host._image.width/2,host._image.height/2)
    host._paint_image()


def _hover(root, canvas, editor):
    x,y = editor.to_canvas((400.,450.))
    canvas.event_generate('<Motion>',x=int(x),y=int(y),warp=True)
    root.update()


def test_hover_s_then_a_actual_native_renderer():
    root,tree,canvas,host,editor,scene,logs = _window()
    try:
        _hover(root,canvas,editor)
        tree.focus_force();root.update()
        tree.event_generate('<KeyPress-s>');root.update()
        assert editor.selected == 'o1' and editor.mode == 'edit', logs
        assert len(canvas.find_withtag('corner_handle')) == 4
        tree.event_generate('<KeyRelease-s>');root.update()
        start=(host._view_zoom,host._view_center)
        tree.event_generate('<KeyPress-a>');root.update()
        assert editor.plate_fit_active and host._view_zoom > start[0], logs
        projected=[editor.to_canvas(p) for p in scene['objects'][0]['quad']]
        assert 0 < min(p[0] for p in projected) < max(p[0] for p in projected) < canvas.winfo_width()
        assert 0 < min(p[1] for p in projected) < max(p[1] for p in projected) < canvas.winfo_height()
        tree.event_generate('<KeyRelease-a>');root.update()
        tree.event_generate('<KeyPress-a>');root.update()
        assert editor.plate_fit_active and host._view_zoom > start[0]
        editor.fit();root.update()
        assert not editor.plate_fit_active and host._view_zoom==1
    finally:
        editor.dispose();root.destroy()


def test_global_fallback_from_new_focus_widget_and_toolbar_action():
    root,tree,canvas,host,editor,scene,logs = _window()
    try:
        _hover(root,canvas,editor)
        # Simulate a focusable widget created after the editor was initialized,
        # with bindtags that do not include its early shortcut class.
        extra=ttk.Button(root,text='Inny przycisk')
        extra.pack();root.update()
        extra.focus_force();root.update()
        extra.bindtags(tuple(tag for tag in extra.bindtags()
                             if tag != editor.shortcut_tag and tag != str(root)))
        extra.event_generate('<KeyPress-s>');root.update()
        assert editor.selected == 'o1',logs  # guarded `all` fallback
        extra.event_generate('<KeyRelease-s>');root.update()
        toggle = tk.Button(root, command=editor.fit_plate_toggle)
        start=(host._view_zoom,host._view_center)
        toggle.invoke();root.update()
        assert editor.plate_fit_active and host._view_zoom > start[0]
        toggle.invoke();root.update()
        assert editor.plate_fit_active and host._view_zoom > start[0]
        editor.fit();root.update()
        assert not editor.plate_fit_active and host._view_zoom==1
    finally:
        editor.dispose();root.destroy()


def test_shortcuts_do_not_capture_other_windows_or_gt_typing():
    root,tree,canvas,host,editor,scene,logs = _window()
    try:
        other=tk.Toplevel(root)
        external=tk.Canvas(other,width=200,height=150)
        external.pack();other.update()
        external.focus_force();root.update()
        start=(editor.selected,host._view_zoom)
        external.event_generate('<KeyPress-a>');root.update()
        external.event_generate('<KeyPress-s>');root.update()
        assert (editor.selected,host._view_zoom)==start
        field=tk.Entry(root)
        field.pack();root.update()
        _hover(root,canvas,editor)
        field.focus_force();root.update()
        field.event_generate('<KeyPress-a>');root.update()
        field.event_generate('<KeyPress-s>');root.update()
        assert (editor.selected,host._view_zoom)==start
        other.destroy()
    finally:
        editor.dispose();root.destroy()
