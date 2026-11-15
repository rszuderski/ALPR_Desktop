import json
from unittest.mock import Mock, patch

import pytest

import run_mz_experiment as experiment


def test_cli_is_read_only_by_default_and_never_constructs_trainer(tmp_path):
    protocol=tmp_path/"protocol.json"
    protocol.write_text(json.dumps({"schema":"test"}))
    with patch.object(experiment,"check_protocol",return_value={"ok":True}), \
        patch('auto_annotation_tool.training.trainer.YOLOPoseTrainer') as trainer:
        assert experiment.main(['--protocol',str(protocol)])==0
    trainer.assert_not_called()


def test_cli_requires_protocol_and_explicit_train_flag():
    with pytest.raises(SystemExit):
        experiment.main([])


def test_unknown_protocol_does_not_enter_training():
    with pytest.raises(ValueError,match="Unsupported"):
        experiment.check_protocol({"schema":"unknown"})


def test_modified_recipe_is_rejected_before_training_or_dataset_access():
    from auto_annotation_tool.dataset_split_assignment import canonical_sha256
    protocol={"schema":"alpr.experiment.mz_ns.v1","experiment_id":"E-MZ-NS-01","training":{"seed":42}}
    protocol["protocol_sha256"]=canonical_sha256(protocol)
    protocol["training"]["seed"]=43
    with pytest.raises(ValueError,match="fingerprint mismatch"):
        experiment.check_protocol(protocol)


def test_cli_delegates_to_the_shared_application_checker():
    protocol={'schema':'fixture'}
    with patch.object(experiment,'check_mz_experiment_protocol',return_value={'ok':True}) as checker:
        assert experiment.check_protocol(protocol,repo_root='fixture-root')=={'ok':True}
    checker.assert_called_once_with(protocol,repo_root='fixture-root')
