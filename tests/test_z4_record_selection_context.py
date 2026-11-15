"""Delayed record-tree events cannot replace the selected full-scene source."""
import copy
from types import SimpleNamespace as NS

import pytest

from auto_annotation_tool.gui.z4_mobile_sample_review import MobileSampleReviewWindow


@pytest.mark.parametrize('subject_key', ['', 'removed-subject'])
def test_record_event_without_current_subject_is_ignored_before_navigation(subject_key):
    def forbidden():raise AssertionError('Stale event must not finish an edit')
    h=NS(session=NS(subjects={'active':{'attempts':['a'],'samples':[]}}),subject_key=subject_key,
        record_tree=NS(selection=lambda:('old',)),_record_ids={'old':('attempt','a')},
        current_record=('scene','scene-1'),_canvas_editor=NS(prepare_navigation=forbidden))
    MobileSampleReviewWindow._select_record(h)
    assert h.current_record==('scene','scene-1')


def test_record_from_previous_subject_is_ignored_before_navigation():
    def forbidden():raise AssertionError('Foreign record must not finish an edit')
    h=NS(session=NS(subjects={'active':{'attempts':['current'],'samples':[]}}),subject_key='active',
        record_tree=NS(selection=lambda:('old',)),_record_ids={'old':('attempt','previous')},
        current_record=None,_canvas_editor=NS(prepare_navigation=forbidden))
    MobileSampleReviewWindow._select_record(h)
    assert h.current_record is None


def test_native_queued_mt_event_does_not_leave_scene_and_mt_can_reopen(tmp_path,monkeypatch):
    import test_mobile_p2b2_canvas_gui as native
    from test_f5a12c_scene_evidence import private_session
    from auto_annotation_tool.gui.mobile_canvas_scene_frames import open_scene
    s=private_session(tmp_path,'two-roi')
    monkeypatch.setattr(native,'source_session',lambda _:s)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch)
    root,h,ed,s=next(gen)
    try:
        sid=next(iter(s.scene_frames));aid=next(iter(s.attempts))
        before=copy.deepcopy(s.review.to_dict());sidecar=s.sidecar_path.read_bytes()
        # Populate the real MT/sample table first.
        h._multi_focus_attempt(aid);h._select_record();root.update()
        assert h.current_record==('attempt',aid) and h.subject_key in s.subjects
        assert h.record_tree.selection() and h._record_ids
        # Deliver the old selection only AFTER the user chooses the full scene.
        h.record_tree.event_generate('<<TreeviewSelect>>',when='tail')
        assert open_scene(ed,sid)
        root.update()
        assert h.current_record==('scene',sid) and h.subject_key==''
        native.pump(root,lambda:h._image is not None and h._image_entry==s.scene_frames[sid]['source_frame_entry'])
        h._view_zoom=2.;h._view_center=(40.,40.);h._paint_image()
        image=h._image;entry=h._image_entry;viewport=(h._view_zoom,h._view_center)
        h.record_tree.event_generate('<<TreeviewSelect>>',when='tail')
        root.update()
        assert h.current_record==('scene',sid) and h.current_invocation==sid and h.subject_key==''
        assert h._image is image and h._image_entry==entry and (h._view_zoom,h._view_center)==viewport
        assert h._record_ids=={} and not h.record_tree.get_children()
        assert s.review.to_dict()==before and s.sidecar_path.read_bytes()==sidecar
        # Genuine MT navigation recreates its own valid record context.
        h._multi_focus_attempt(aid);h._select_record();root.update()
        assert h.current_record==('attempt',aid) and h.subject_key==s.attempts[aid]['subject_key']
        assert h.current_invocation==s.invocation_by_attempt[aid]
        assert s.review.to_dict()==before and s.sidecar_path.read_bytes()==sidecar
    finally:
        try:next(gen)
        except StopIteration:pass
