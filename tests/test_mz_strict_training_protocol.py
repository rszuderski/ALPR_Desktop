from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from auto_annotation_tool.training.trainer import YOLOPoseTrainer
from auto_annotation_tool.training.training_history import TrainingRun, TrainingHistory
from auto_annotation_tool.training.experiment_protocol import protocol_differences, validate_requested_protocol


def recipe():
    return dict(seed=42,optimizer="SGD",lr0=.01,epochs=100,patience=50,imgsz=320,batch=4,
        amp=False,device=0,cache=False,workers=0,mosaic=0.0,close_mosaic=0,deterministic=True,plots=True,
        hsv_h=0.0,hsv_s=0.0,hsv_v=0.0,fliplr=0.0,flipud=0.0,translate=0.0,scale=0.0,
        degrees=0.0,shear=0.0,perspective=0.0,bgr=0.0,mixup=0.0,cutmix=0.0,copy_paste=0.0,
        erasing=0.0,augmentation_policy="source_only")


def run():
    return TrainingRun(id="run",name="test",created_at="now",output_dir="runs",
        strict_experiment=True,training_protocol_requested=recipe())


def test_both_model_scales_receive_identical_explicit_training_arguments():
    trainer = YOLOPoseTrainer.__new__(YOLOPoseTrainer)
    nano, small = run(), run()
    nano.base_model="yolo26n.pt"
    small.base_model="yolo26s.pt"
    first = trainer._build_train_args(nano,"dataset",100,4,320,0,.01)
    second = trainer._build_train_args(small,"dataset",100,4,320,0,.01)
    assert first==second
    assert not protocol_differences(recipe(),first)
    assert first["optimizer"]=="SGD" and first["seed"]==42
    assert first["amp"] is False and first["fliplr"]==0 and first["mosaic"]==0


def test_strict_recipe_rejects_silently_lowered_batch_or_imgsz():
    trainer = YOLOPoseTrainer.__new__(YOLOPoseTrainer)
    with pytest.raises(RuntimeError,match="drift"):
        trainer._build_train_args(run(),"dataset",100,2,320,0,.01)
    with pytest.raises(RuntimeError,match="drift"):
        trainer._build_train_args(run(),"dataset",100,4,256,0,.01)


@pytest.mark.parametrize("field,value", [("optimizer","auto"),("device","auto"),("batch",-1)])
def test_strict_recipe_rejects_automatic_parameters(field,value):
    protocol=recipe()
    protocol[field]=value
    with pytest.raises(ValueError,match="automatic"):
        validate_requested_protocol(protocol)


def test_actual_runtime_protocol_is_saved_and_drift_fails_run():
    trainer = YOLOPoseTrainer.__new__(YOLOPoseTrainer)
    trainer.history=Mock()
    trial=run()
    trainer._capture_training_protocol(trial,recipe(),source="ultralytics_pretrain_runtime")
    assert trial.training_protocol_snapshot["actual"]["amp"] is False
    assert trial.training_protocol_snapshot["comparable"]
    changed=recipe()|{"optimizer":"AdamW"}
    with pytest.raises(RuntimeError,match="runtime parameter drift"):
        trainer._capture_training_protocol(trial,changed,source="ultralytics_pretrain_runtime")
    assert trial.training_protocol_snapshot["actual"]["optimizer"]=="AdamW"
    assert not trial.training_protocol_snapshot["comparable"]


def test_ram_or_oom_failure_is_explicitly_non_comparable():
    trainer=YOLOPoseTrainer.__new__(YOLOPoseTrainer)
    trainer.history=Mock()
    trial=run()
    with pytest.raises(RuntimeError,match="cannot change protocol"):
        trainer._reject_experiment_memory_fallback(trial,"OOM")
    assert trial.training_protocol_snapshot["memory_fallback_required"]
    assert not trial.training_protocol_snapshot["comparable"]


def test_normal_application_keeps_its_existing_training_defaults():
    trainer=YOLOPoseTrainer.__new__(YOLOPoseTrainer)
    legacy=SimpleNamespace(output_dir="runs")
    args=trainer._build_train_args(legacy,"dataset",100,16,640,"cpu",.01,amp=True,mosaic=1.0)
    assert args["optimizer"]=="auto" and args["seed"]==0
    assert args["batch"]==16 and args["imgsz"]==640 and args["amp"] and args["mosaic"]==1


def test_strict_source_only_removes_hidden_albumentations_before_loader_creation():
    runtime=SimpleNamespace(args=SimpleNamespace())
    YOLOPoseTrainer._apply_experiment_data_policy(run(),runtime)
    assert runtime.args.augmentations==[]
    legacy=SimpleNamespace(args=SimpleNamespace())
    YOLOPoseTrainer._apply_experiment_data_policy(SimpleNamespace(strict_experiment=False),legacy)
    assert not hasattr(legacy.args,"augmentations")


def test_assignment_is_part_of_existing_training_dataset_snapshot(tmp_path):
    import json
    from auto_annotation_tool.training.model_provenance import build_training_dataset_snapshot, training_dataset_snapshots_match
    (tmp_path/"images/train").mkdir(parents=True)
    (tmp_path/"labels/train").mkdir(parents=True)
    (tmp_path/"images/train/a.jpg").write_bytes(b"image")
    (tmp_path/"labels/train/a.txt").write_text("0 .5 .5 .2 .2")
    (tmp_path/"data.yaml").write_text("train: images/train\nval: images/train\nnames: ['A']\n")
    path=tmp_path/"split_assignment_manifest.json"
    path.write_text(json.dumps({"assignment_sha256":"before"}))
    before=build_training_dataset_snapshot(tmp_path,target="char")
    path.write_text(json.dumps({"assignment_sha256":"after"}))
    after=build_training_dataset_snapshot(tmp_path,target="char")
    assert before["manifest_sha256"]!=after["manifest_sha256"]
    assert not training_dataset_snapshots_match(before,after)[0]


def test_history_roundtrip_preserves_requested_and_actual_protocol(tmp_path):
    history=TrainingHistory(tmp_path)
    created=history.create_run("E-MZ-NS-01_n",strict_experiment=True,
        experiment_id="E-MZ-NS-01",training_protocol_requested=recipe())
    history.update_run(created.id,training_protocol_snapshot={"actual":recipe(),"comparable":True})
    reloaded=TrainingHistory(tmp_path).get_run(created.id)
    assert reloaded.strict_experiment and reloaded.experiment_id=="E-MZ-NS-01"
    assert reloaded.training_protocol_requested==recipe()
    assert reloaded.training_protocol_snapshot["actual"]==recipe()


def test_strict_loop_refuses_ram_safe_profile_before_loading_model(tmp_path):
    history=TrainingHistory(tmp_path)
    trial=history.create_run("strict",strict_experiment=True,training_protocol_requested=recipe())
    trainer=YOLOPoseTrainer(history)
    trainer.current_run=trial
    trainer._get_dataset_runtime_profile=Mock(return_value={"is_pose":False,"train_images":909})
    trainer._apply_ultralytics_runtime_safety_overrides=Mock()
    trainer._reset_runtime_state=Mock()
    trainer.on_training_end=Mock()
    model=Mock()
    with patch('auto_annotation_tool.training.trainer.get_yolo_class',return_value=model), \
        patch('auto_annotation_tool.training.trainer._patch_ultralytics_save_model_closed_file_bug'), \
        patch('auto_annotation_tool.training.trainer.sample_system_memory',return_value={"ram_percent":99}):
        trainer._training_loop("yolo26n.pt","dataset",100,4,320,0,.01,None)
    model.assert_not_called()
    assert history.get_run(trial.id).status=="failed"
    assert not history.get_run(trial.id).training_protocol_snapshot["comparable"]


def test_strict_loop_does_not_retry_oom_with_lighter_parameters(tmp_path):
    history=TrainingHistory(tmp_path)
    trial=history.create_run("strict",strict_experiment=True,training_protocol_requested=recipe())
    trainer=YOLOPoseTrainer(history)
    trainer.current_run=trial
    trainer._get_dataset_runtime_profile=Mock(return_value={"is_pose":False,"train_images":909})
    trainer._apply_ultralytics_runtime_safety_overrides=Mock()
    trainer._reset_runtime_state=Mock()
    trainer.on_training_end=Mock()
    model=Mock()
    model.return_value.train.side_effect=RuntimeError("CUDA out of memory")
    with patch('auto_annotation_tool.training.trainer.get_yolo_class',return_value=model), \
        patch('auto_annotation_tool.training.trainer._patch_ultralytics_save_model_closed_file_bug'), \
        patch('auto_annotation_tool.training.trainer.sample_system_memory',return_value={"ram_percent":40,"ram_available_mib":6000}):
        trainer._training_loop("yolo26n.pt","dataset",100,4,320,0,.01,None)
    model.assert_called_once()
    model.return_value.train.assert_called_once()
    assert model.return_value.train.call_args.kwargs["batch"]==4
    assert history.get_run(trial.id).status=="failed"
    assert history.get_run(trial.id).training_protocol_snapshot["memory_fallback_required"]
