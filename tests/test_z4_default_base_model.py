from unittest.mock import Mock

import pytest

from auto_annotation_tool.gui.tab_training import TrainingTab, AVAILABLE_POSE_MODELS, AVAILABLE_DETECT_MODELS


class Value:
    def __init__(self,value=''): self.value=value
    def get(self): return self.value
    def set(self,value): self.value=value


def owner(target):
    host=TrainingTab.__new__(TrainingTab)
    host._get_selected_training_target=lambda:target
    host.get_campaign_training_target=lambda:target
    host.base_model_var=Value()
    host.base_custom_var=Value()
    host.base_combo=Mock()
    host._on_base_model_change=Mock()
    host._refresh_training_start_state=Mock()
    host._save_step4_training_ui_state=Mock(side_effect=AssertionError('no persistence during refresh'))
    return host


@pytest.mark.parametrize('target,preferred',[('plate','yolo26n-pose'),('char','yolo26n'),('vehicle','yolo26n')])
@pytest.mark.parametrize('explicit_mode',[True,False])
def test_default_explicitly_prefers_YOLO26_without_reordering_catalog(target,preferred,explicit_mode):
    host=owner(target)
    before=(tuple(AVAILABLE_POSE_MODELS),tuple(AVAILABLE_DETECT_MODELS))
    assert host._get_default_base_model_for_mode(target if explicit_mode else None)==preferred
    assert before==(tuple(AVAILABLE_POSE_MODELS),tuple(AVAILABLE_DETECT_MODELS))


@pytest.mark.parametrize('target,preferred',[('tablice','yolo26n-pose'),('znaki','yolo26n'),('pojazdy','yolo26n')])
def test_default_normalizes_target_aliases(target,preferred):
    assert owner('char')._get_default_base_model_for_mode(target)==preferred


@pytest.mark.parametrize('target',['plate','char','vehicle'])
def test_missing_preferred_falls_back_to_first_non_custom_in_existing_order(target):
    host=owner(target)
    host._get_base_model_choices_for_mode=Mock(return_value=['Custom','legacy_first','other_model',host._get_custom_base_model_label()])
    assert host._get_default_base_model_for_mode(target)=='legacy_first'
    host._get_base_model_choices_for_mode.assert_called_once_with(target)


@pytest.mark.parametrize('choices',[['Custom'],['Własny plik .pt'],[]])
def test_no_non_custom_models_falls_back_to_custom_label(choices):
    host=owner('plate')
    host._get_base_model_choices_for_mode=lambda mode:choices
    assert host._get_default_base_model_for_mode('plate')==host._get_custom_base_model_label()


@pytest.mark.parametrize('target,preferred',[('plate','yolo26n-pose'),('char','yolo26n')])
def test_normal_refresh_uses_default_only_without_valid_selection(target,preferred):
    host=owner(target)
    host._refresh_base_model_choices()
    assert host.base_model_var.get()==preferred
    assert preferred in host.base_combo.configure.call_args.kwargs['values']


@pytest.mark.parametrize('target,selected',[('plate','yolo11s-pose'),('char','yolo26s')])
def test_saved_explicit_selection_is_restored_and_survives_repeated_normal_refresh(target,selected):
    host=owner(target)
    payload={'models':{target:{'base_model_key':selected,'base_custom_path':''}}}
    host._load_step4_training_ui_state=lambda:payload
    assert host._resolve_saved_step4_training_model_selection(target)==payload['models'][target]
    assert host._apply_saved_step4_training_model_selection(target)
    host._refresh_base_model_choices()
    host._refresh_base_model_choices()
    assert host.base_model_var.get()==selected
    assert host.base_custom_var.get()==''
    host._save_step4_training_ui_state.assert_not_called()


@pytest.mark.parametrize('target',['plate','char'])
def test_saved_explicit_custom_file_survives_normal_refresh(target,tmp_path):
    host=owner(target)
    path=tmp_path/'selected.pt'
    path.write_bytes(b'unit-test checkpoint')
    host._load_step4_training_ui_state=lambda:{'models':{target:{'base_model_key':'Custom','base_custom_path':str(path)}}}
    host._resolve_training_run_from_model_path=lambda path:None
    assert host._apply_saved_step4_training_model_selection(target)
    host._refresh_base_model_choices()
    assert host.base_model_var.get()==host._get_custom_base_model_label()
    assert host.base_custom_var.get()==str(path)
    host._save_step4_training_ui_state.assert_not_called()
