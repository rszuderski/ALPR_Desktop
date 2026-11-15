"""Native Tk gestures for per-object Space decisions and derived image status."""
import copy
from tkinter import font as tkfont

import pytest
import test_mobile_p2b2_canvas_gui as native
from test_mobile_p2b2_canvas_gui import gui
from test_mobile_p2b2_canvas_review import source_session, MANUAL
from test_z4_frame_stacks import ready, stack_gui, stacks
from auto_annotation_tool.ranking import mobile_canvas_review as core
from auto_annotation_tool.ranking import mobile_canvas_partial_review as partial
from auto_annotation_tool.ranking import mobile_mt_detection_evaluation as detection
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession

G = 'mt-one'


def frame(s):
    return partial.progress(core.scene(s, G))


def accept(root, h, ed, index):
    native.click(h, ed, (225 + 220*index, 175))
    ready(root, h, ed)
    assert ed.selected == f'GT:a{index}', ed.h.status_var.get()
    native.key(h, 'space')
    ready(root, h, ed)
    assert frame(h.session)['object_decisions'][ed.selected] == partial.PLATE, ed.h.status_var.get()


@pytest.fixture
def empty_gui(tmp_path, monkeypatch, request):
    monkeypatch.setattr(native, 'source_session', lambda p: source_session(p, n=0, mz=False))
    return request.getfixturevalue('gui')


def test_space_without_selection_or_hover_cannot_confirm_all_model_frames(gui, monkeypatch):
    root, h, ed, s = gui
    monkeypatch.setattr(ed, '_pointer_for_selection', lambda event=None: None)
    before = s.sidecar_path.read_bytes()
    native.key(h, 'space')
    ready(root, h, ed)
    assert s.sidecar_path.read_bytes() == before
    assert frame(s)['decisions'] == 2 and not ed.checked_var.get()
    assert 'Wybierz ramkę MT' in ed.h.status_var.get()


def test_selected_frame_space_toggles_only_that_decision_and_has_visible_badge(gui):
    root, h, ed, s = gui
    parents = copy.deepcopy(core.scene(s, G)['parents'])
    accept(root, h, ed, 0)
    assert frame(s)['object_decisions'] == {'GT:a0': partial.PLATE, 'GT:a1': partial.UNRESOLVED}
    assert ed.contexts.counters['decisions'].cget('text') == '1'
    assert any('OK' in h.canvas.itemcget(i, 'text') for i in h.canvas.find_withtag('clean_label_text'))
    native.key(h, 'space')
    ready(root, h, ed)
    assert frame(s)['decisions'] == 2 and not ed.checked_var.get()
    assert core.scene(s, G)['parents'] == parents
    assert any(all(word in h.canvas.itemcget(i,'text') for word in ('POSE','decyzja oczekuje','GTR 1'))
               for i in h.canvas.find_withtag('mt_relation_text'))


def test_last_frame_decision_verifies_image_without_extra_image_checkbox(gui):
    root, h, ed, s = gui
    accept(root, h, ed, 0)
    accept(root, h, ed, 1)
    assert ed.checked_var.get() and frame(s)['complete']
    # Mouse verification targets this stack; completion of the whole image is
    # still derived from all frame decisions, without an extra confirmation.
    assert ed.approve.cget('text') == 'Ramki wybranej tablicy zweryfikowane'
    assert ed.approve.cget('state') == 'normal' and ed.frame_checked_var.get()
    assert ed.contexts.counters['inspected'].cget('text') == '1/1'
    assert ed.contexts.counters['remaining'].cget('text') == '0'
    assert ed.contexts.counters['decisions'].cget('text') == '0'
    assert 'Obraz zweryfikowany' in ed.contexts.fields['image'].cget('text')
    assert 'GT/MZ' in ed.contexts.fields['image'].cget('text')
    assert core.group_issues(s, G)  # No GT text was fabricated by Space.


def test_right_click_decisions_also_finish_image_without_space(gui):
    root, h, ed, s = gui
    for point in ((225, 175), (445, 175)):
        native.click(h, ed, point, 3)
        ready(root, h, ed)
    assert frame(s)['complete'] and ed.checked_var.get()
    assert frame(s)['false_positive'] == 2 and frame(s)['plate_count'] == 0
    # This older presence-only fixture lacks source BBOX/confidence. Decisions
    # are complete, but formal detector counts must remain unavailable.
    assert detection.evaluate_unit(s,G)['tp'] is None
    assert ed.contexts.counters['fp'].cget('text') == ed.contexts.counters['gt'].cget('text') == '—'
    assert ed.contexts.counters['decisions'].cget('text') == '0'


def test_new_manual_plate_is_valid_immediately_and_changes_counters_on_verified_image(gui):
    root, h, ed, s = gui
    accept(root, h, ed, 0)
    accept(root, h, ed, 1)
    native.key(h, 'd')
    for point in MANUAL:
        native.click(h, ed, point)
    ready(root, h, ed)
    assert frame(s)['manual_added'] == 1 and frame(s)['complete'] and ed.checked_var.get()
    assert ed.contexts.counter_labels['fn'].cget('text') == 'Pominięte (FN)'
    assert frame(s)['plate_count'] == 3
    assert ed.contexts.counters['fn'].cget('text') == ed.contexts.counters['gt'].cget('text') == '—'
    before = s.sidecar_path.read_bytes()
    native.key(h, 'space')
    ready(root, h, ed)
    assert s.sidecar_path.read_bytes() == before and frame(s)['complete']
    # R deliberately requires a physical pointer on Canvas. Exercise the same
    # action callback without moving the user's real desktop pointer in a test.
    ed.remove_selected()
    ready(root, h, ed)
    assert ed.contexts.counters['fn'].cget('text') == '—'
    assert frame(s)['plate_count'] == 2 and frame(s)['complete']
    ed.undo()
    ready(root, h, ed)
    assert frame(s)['manual_added'] == 1 and frame(s)['complete']


def test_empty_image_has_an_explicit_reversible_no_plates_action(empty_gui):
    root, h, ed, s = empty_gui
    assert not ed.checked_var.get() and ed.clean_empty_button.winfo_ismapped(), (ed.h.status_var.get(), ed.selected, ed.mode)
    assert ed.clean_empty_button.cget('text') == 'Na obrazie nie ma tablic'
    ed.clean_empty_button.invoke()
    ready(root, h, ed)
    assert ed.checked_var.get() and frame(s)['complete']
    assert core.summary(s)['true_negative_frames'] == 1
    ed.clean_empty_button.invoke()
    ready(root, h, ed)
    assert not ed.checked_var.get() and not frame(s)['complete']
    assert core.summary(s)['true_negative_frames'] == 0


def test_manual_after_explicit_empty_image_updates_status_and_removal_requires_new_negative(empty_gui):
    root, h, ed, s = empty_gui
    native.key(h, 'space')
    ready(root, h, ed)
    native.key(h, 'd')
    for point in MANUAL:
        native.click(h, ed, point)
    ready(root, h, ed)
    assert ed.checked_var.get() and frame(s)['manual_added'] == 1
    assert not ed.clean_empty_button.winfo_ismapped()
    ed.remove_selected()
    ready(root, h, ed)
    assert not ed.checked_var.get() and ed.clean_empty_button.winfo_ismapped()


def test_space_autorepeat_is_one_decision_per_press_release(gui):
    root, h, ed, s = gui
    native.click(h, ed, (225, 175))
    ready(root, h, ed)
    native.focus_canvas(h)
    revision = s.review.review_revision
    for _ in range(3):
        h.canvas.event_generate('<KeyPress-space>')
    ready(root, h, ed)
    assert s.review.review_revision == revision+1
    assert frame(s)['object_decisions']['GT:a0'] == partial.PLATE
    h.canvas.event_generate('<KeyRelease-space>')
    native.key(h, 'space')
    ready(root, h, ed)
    assert frame(s)['object_decisions']['GT:a0'] == partial.UNRESOLVED


def test_space_in_gt_entry_is_text_not_a_presence_decision(gui):
    root, h, ed, s = gui
    ed.start_entry('GT:a0')
    native.pump(root)
    assert ed.entry is not None
    ed.entry_var.set('ABC')
    ed.entry.focus_force()
    native.pump(root)
    ed.entry.event_generate('<KeyPress-space>')
    ed.entry.event_generate('<KeyRelease-space>')
    native.pump(root)
    assert ' ' in ed.entry_var.get()
    assert frame(s)['object_decisions']['GT:a0'] == partial.UNRESOLVED


def test_reopening_restores_partial_badges_and_image_counters(gui):
    root, h, ed, s = gui
    accept(root, h, ed, 0)
    native.click(h, ed, (445, 175), 3)
    ready(root, h, ed)
    h.session = MobileReviewSession.open(s.path, sidecar_path=s.sidecar_path)
    ready(root, h, ed)
    assert ed.checked_var.get() and ed.contexts.counters['decisions'].cget('text') == '0'
    assert frame(s)['false_positive'] == frame(s)['plate_count'] == 1
    assert detection.evaluate_unit(s,G)['tp'] is None
    assert ed.contexts.counters['fp'].cget('text') == ed.contexts.counters['gt'].cget('text') == '—'
    assert h.canvas.find_withtag('fp_diagonal')


def test_formal_fn_caption_and_derived_status_fit_both_native_sizes(gui):
    from z4_sidebar_assertions import assert_sidebar_reaches
    root, h, ed, s = gui
    for size in ('1649x896', '1180x760'):
        h.window.geometry(size+'+20+20')
        ready(root, h, ed)
        assert_sidebar_reaches(root,h,[*ed.contexts.counters.values(), *ed.contexts.counter_labels.values(),
                                      *ed.contexts.fields.values(), ed.approve])
        assert tkfont.Font(root=root, font=ed.contexts.counters['fn'].cget('font')).actual('size') == 20
        assert ed.contexts.counter_labels['fn'].cget('text') == 'Pominięte (FN)'


def test_hover_space_resolves_each_frame_without_creating_sticky_selection(gui, monkeypatch):
    root, h, ed, s = gui
    for index in (0, 1):
        native.focus_canvas(h)
        x, y = ed.to_canvas((225+220*index, 175))
        monkeypatch.setattr(ed, '_pointer_for_selection', lambda event=None, xy=(x, y): xy)
        h.canvas.event_generate('<Motion>', x=round(x), y=round(y))
        native.key(h, 'space')
        ready(root, h, ed)
        assert ed.selected == ''
        assert frame(s)['object_decisions'][f'GT:a{index}'] == partial.PLATE
    assert frame(s)['complete']
    native.key(h, 'space')
    ready(root, h, ed)
    assert frame(s)['object_decisions']['GT:a0'] == partial.PLATE
    assert frame(s)['object_decisions']['GT:a1'] == partial.UNRESOLVED


def test_selected_space_works_with_tree_focus_and_pointer_outside_canvas(gui, monkeypatch):
    root, h, ed, s = gui
    native.click(h, ed, (225, 175))
    ready(root, h, ed)
    monkeypatch.setattr(ed, '_pointer_for_selection', lambda event=None: None)
    h.subject_tree.focus_force()
    native.pump(root)
    for expected in (partial.PLATE, partial.UNRESOLVED):
        h.subject_tree.event_generate('<KeyPress-space>')
        h.subject_tree.event_generate('<KeyRelease-space>')
        ready(root, h, ed)
        assert frame(s)['object_decisions']['GT:a0'] == expected
        assert frame(s)['object_decisions']['GT:a1'] == partial.UNRESOLVED


def test_badge_keeps_green_ok_and_red_bledna_separate_from_gt_editing(gui):
    root, h, ed, s = gui
    accept(root, h, ed, 0)
    def header(oid):
        return [i for i in h.canvas.find_withtag('stack_badge_text') if oid in h.canvas.gettags(i)]
    assert any('OK' in h.canvas.itemcget(i, 'text') and h.canvas.itemcget(i, 'fill') == '#61E6A2'
               for i in header('GT:a0'))
    ed.start_entry('GT:a0')
    native.pump(root)
    assert ed.entry is not None
    l, t, r, b = ed.clean_counter_boxes['GT:a0']
    assert ed.entry.winfo_y() >= b
    assert header('GT:a0')
    ed.escape()
    native.click(h, ed, (445, 175), 3)
    ready(root, h, ed)
    assert any('Błędna' in h.canvas.itemcget(i, 'text') and h.canvas.itemcget(i, 'fill') == '#F17987'
               for i in header('GT:a1'))
    ed.undo()
    ready(root, h, ed)
    assert any(all(word in h.canvas.itemcget(i,'text') for word in ('POSE','decyzja oczekuje','GTR 0')) for i in header('GT:a1'))


def test_stack_counter_badge_above_frame_has_short_thin_tether_and_updates_with_zx(stack_gui):
    root, h, ed, s = stack_gui
    ed._choose_plate('GT:a0')
    ready(root, h, ed)
    before = s.sidecar_path.read_bytes()
    for key, expected in (('z', '2/3'), ('z', '3/3'), ('x', '2/3'), ('X', '1/3')):
        native.key(h, key)
        ready(root, h, ed)
        labels = h.canvas.find_withtag('stack_badge_text')
        assert len(labels) == 1 and expected in h.canvas.itemcget(labels[0], 'text')
        l, t, r, b = ed.stack_badges[ed.selected][:4]
        points = [ed.to_canvas(p) for p in core._object(core.scene(s, G), ed.selected)['quad']]
        assert b < min(p[1] for p in points)
        tether = h.canvas.find_withtag('badge_tether')
        assert len(tether) == 1 and float(h.canvas.itemcget(tether[0], 'width')) == 1
        x1, y1, x2, y2 = h.canvas.coords(tether[0])
        assert abs(y1-b) <= 1 and abs(y2-min(p[1] for p in points)) <= 1
        assert ((x2-x1)**2+(y2-y1)**2)**.5 <= 40
    assert s.sidecar_path.read_bytes() == before


def test_tab_and_shift_tab_keep_native_focus_and_do_not_cycle_stack(stack_gui):
    root, h, ed, s = stack_gui
    ed._choose_plate('GT:a0')
    ready(root, h, ed)
    before = s.sidecar_path.read_bytes()
    for state in (0, 1):
        native.focus_canvas(h)
        target = h.canvas.tk_focusPrev() if state else h.canvas.tk_focusNext()
        h.canvas.event_generate('<KeyPress-Tab>', state=state)
        h.canvas.event_generate('<KeyRelease-Tab>', state=state)
        native.pump(root)
        assert ed.selected == 'GT:a0' and h.window.focus_get() is target
    assert s.sidecar_path.read_bytes() == before


def test_zx_typing_and_ctrl_z_do_not_cycle_the_stack(stack_gui):
    root, h, ed, s = stack_gui
    ed._choose_plate('GT:a0')
    ready(root, h, ed)
    ed.start_entry('GT:a0')
    native.pump(root)
    ed.entry.focus_force()
    native.pump(root)
    ed.entry_var.set('')
    for key in ('z', 'x', 'Z', 'X'):
        ed.entry.event_generate('<KeyPress-'+key+'>')
        ed.entry.event_generate('<KeyRelease-'+key+'>')
        native.pump(root)
    assert ed.entry_var.get() == 'ZXZX' and ed.selected == 'GT:a0'  # Existing GT normalization.
    ed.escape()
    ed._choose_plate('GT:a0')
    ready(root, h, ed)
    from auto_annotation_tool.gui.mobile_canvas_stacks import write_reference
    ed._commit(lambda: write_reference(ed, 'GT:a0', text='CHANGED'), 'test GT')
    ready(root, h, ed)
    native.focus_canvas(h)
    h.canvas.event_generate('<KeyPress-z>', state=4)
    h.canvas.event_generate('<KeyRelease-z>', state=4)
    ready(root, h, ed)
    assert ed.selected == 'GT:a0'
    assert core._object(core.scene(s, G), 'GT:a0')['gt_text'] == 'WA01234'


def test_compact_badge_remains_visible_when_zoom_leaves_no_free_label_space(gui):
    root, h, ed, s = gui
    accept(root, h, ed, 0)
    q = core._object(core.scene(s, G), 'GT:a0')['quad']
    h._view_center = tuple(sum(p[i] for p in q)/4 for i in (0, 1))
    h._view_zoom = 100
    h._paint_image()
    native.pump(root)
    assert 'GT:a0' in ed.clean_counter_boxes
    l, t, r, b = ed.clean_counter_boxes['GT:a0']
    assert 0 <= l < r <= h.canvas.winfo_width() and 0 <= t < b <= h.canvas.winfo_height()
    assert 'GT:a0' not in ed.badges  # The GT field still uses its safe toolbar fallback.
    assert any('OK' in h.canvas.itemcget(i, 'text') for i in h.canvas.find_withtag('stack_badge_text'))
