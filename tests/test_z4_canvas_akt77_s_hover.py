"""AKT77 — real Canvas hover target + guarded Windows physical S fallback.

Read-only synthetic scenes; never writes GT, inference, models or sessions.
The physical-key path is exercised with an injected Windows state so this
is deterministic on Linux and Windows without reading the real keyboard.
"""
from __future__ import annotations

from types import SimpleNamespace
from test_z4_canvas_akt70_selection_drag import setup, event, key_event


def test_hover_motion_keeps_exact_plate_when_windows_pointer_drifts():
    mod,e,c,h,scene,messages,_=setup()
    # Mouse actually hovers the tablicy contour, at its right edge.
    e.hover_motion(event(48,35))
    # OS reports the pointer 29 px to the right (outside the contour).
    c.pointer=(77,35)
    assert e.select_mode(key_event(c))=='break'
    assert e.selected=='o1' and e.mode=='edit', messages
    assert len([op for op in c.ops if 'corner_handle' in op[2].get('tags',())]) == 4


def test_one_key_press_selects_once_even_if_tk_does_not_dispatch_s():
    mod,e,c,h,scene,messages,_=setup()
    h.window.focus_get=lambda:None
    c.pointer=(35,35)
    e.hover_motion(event(35,35))
    pressed=False
    e._windows_s_state=lambda:(True,pressed)
    e._windows_s_watch_tick(schedule=False)
    pressed=True
    e._windows_s_watch_tick(schedule=False)
    assert e.selected=='o1', messages
    e._windows_s_watch_tick(schedule=False)
    assert e.selected=='o1', 'Held S must not toggle repeatedly.'
    pressed=False
    e._windows_s_watch_tick(schedule=False)
    pressed=True
    e._windows_s_watch_tick(schedule=False)
    assert e.selected==''


def test_s_from_other_window_or_text_field_is_ignored():
    mod,e,c,h,scene,messages,_=setup()
    c.pointer=(35,35)
    e.hover_motion(event(35,35))
    h.window.focus_get=lambda:None
    e._windows_s_state=lambda:(False,True)
    e._windows_s_watch_tick(schedule=False)
    assert e.selected==''
    field=object()
    h.window.focus_get=lambda:field
    e._is_typing_shortcut=lambda w:w is field
    e._windows_s_state=lambda:(True,True)
    e._win_s_down=False
    e._windows_s_watch_tick(schedule=False)
    assert e.selected==''


def test_cannot_select_raw_suppressed_candidate():
    mod,e,c,h,scene,messages,_=setup()
    scene['objects'][0]['post_nms_suppressed']=True
    c.pointer=(35,35)
    e.hover_motion(event(35,35))
    assert e._s_hover_hit is None
    e.select_mode(key_event(c))
    assert e.selected==''


def test_hover_cache_clears_on_leave():
    mod,e,c,h,scene,messages,_=setup()
    e.hover_motion(event(35,35))
    assert e._s_hover_hit is not None
    e.leave_canvas()
    assert e._s_hover_hit is None
