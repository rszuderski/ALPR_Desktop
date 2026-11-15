"""Z4MOBREV021 — transparent cancellation state, continuous steps, accessible observation list."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

from auto_annotation_tool.gui import z4_mobile_sample_review as gui


def _group(*, reason='scene_superseded', pose=True, cancelled=True):
    return SimpleNamespace(
        cancelled=cancelled, executed=True, execution_failed=False,
        detection_count=1, execution_errors=(),
        records=[{'mt_status':'VALID_QUAD', 'mt_pose_status': 'VALID_QUAD' if pose else 'MISSING',
                  'cancel_reason':reason}])


def test_real_scene_superseded_is_lifecycle_not_model_negative():
    message=gui._cancelled_mt_explanation(_group())
    assert 'nowszą' in message
    assert 'wykrycie' in message
    assert 'Pose' in message
    assert 'nie wliczamy' in message
    assert 'MT' in message and 'OCR' in message


def test_other_cancel_reason_does_not_invent_scene_superseded_or_pose():
    message=gui._cancelled_mt_explanation(_group(reason='operator_cancelled', pose=False))
    assert 'nowszą' not in message
    assert 'Pose' not in message
    assert 'próbę' in message


def test_accessible_single_column_markers_are_stable_and_meaningful():
    cases=[
        ('MT ocenione',False,False,'✓','MT ocenione','review_done'),
        ('Ocenione MT i MZ',False,False,'✓','MT i MZ ocenione','review_done'),
        ('MT do oceny',False,False,'○','Do oceny MT','review_pending'),
        ('Poza pomiarem MT',True,False,'—','Próba przerwana','review_excluded'),
        ('Poza pomiarem MT',False,True,'—','Błąd wykonania','review_excluded'),
        ('Poza pomiarem MT',False,False,'—','Poza oceną MT','review_excluded'),
    ]
    for state,cancelled,failed,marker,meaning,tag in cases:
        result, actual_tag=gui._observation_list_label(2,state,cancelled=cancelled,execution_failed=failed)
        assert result.startswith(marker+'  02')
        assert meaning in result
        assert tag==actual_tag


def test_no_stage_geometry_is_hidden_when_supposedly_unavailable():
    # Static API guard: no disappearing whole 02 panel, and no mutable GT.
    source=inspect.getsource(gui.MobileSampleReviewWindow._sync_pose_gt_controls)
    assert 'self.pose_gt_title.configure(text="02  ·  OBRYS TABLICY")' in source
    assert 'self.pose_gt_host.grid()' in source
    assert 'self.pose_answer_group' in source
    assert 'self.pose_gt_host.grid_remove()' not in source
    assert 'nie są wymagane' in source
    assert 'self._leave_pose_gt_edit_on_navigation()' in source


def test_does_not_change_scientific_counters_or_export():
    source=inspect.getsource(gui.MobileSampleReviewWindow._refresh_subject_rows)
    assert '_observation_list_label' in source
    assert 'self.subject_tree.item(iid, values=(label,), tags=(tag,))' in source
    assert 'self.session.subject_draft(key)' not in source
    # Production model metrics are in a separate module unchanged by 021.
    assert gui.make_scientific_summary.__module__.endswith('mobile_scientific_review')


def test_internal_record_status_is_not_incorrectly_called_cancelled_model():
    source=inspect.getsource(gui.MobileSampleReviewWindow._sync_review_controls)
    assert 'Status próby: przerwana' in source
    assert 'Wynik modelu: {state}' not in source
    assert 'self._mt_scored_controls' not in source or '"_mt_scored_controls"' in source
