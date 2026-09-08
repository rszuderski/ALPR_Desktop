import copy

import pytest
from PIL import Image

from auto_annotation_tool.training.trainer import YOLOPoseTrainer
from auto_annotation_tool.training.training_history import TrainingHistory
from auto_annotation_tool.training.model_provenance import (
    build_training_dataset_snapshot, build_model_training_provenance, training_dataset_snapshots_match,
)


def make_dataset(root, target):
    dataset = root / "dataset"
    for split in ("train", "val", "test"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (32, 32), (32, 64, 128)).save(dataset / "images" / split / "one.jpg", "JPEG")
        (dataset / "labels" / split / "one.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (dataset / "data.yaml").write_text("train: images/train\nval: images/val\ntest: images/test\nnames: [sample]\n", encoding="utf-8")
    return dataset


def new_run(root, dataset, target, model):
    history = TrainingHistory(root / model)
    initial = build_training_dataset_snapshot(dataset, target=target)
    run = history.create_run(model, epochs=150, training_target=target, dataset_path=str(dataset),
                             base_model=model, training_dataset_snapshot=initial,
                             training_dataset_input_snapshot=copy.deepcopy(initial))
    trainer = YOLOPoseTrainer(history)
    return history, trainer, run


@pytest.mark.parametrize("target", ["plate", "char"])
def test_first_and_next_model_share_the_prepared_dataset_id(tmp_path, target):
    from ultralytics.data.utils import verify_image
    dataset = make_dataset(tmp_path, target)
    jpeg = dataset / "images/train/one.jpg"
    jpeg.write_bytes(jpeg.read_bytes() + b"trailing bytes after JPEG end marker")
    history_n, trainer_n, run_n = new_run(tmp_path, dataset, target, "n")
    original = copy.deepcopy(run_n.training_dataset_snapshot)
    # This is the same repair Ultralytics performs while constructing loaders.
    _, found, corrupt, message = verify_image(((str(jpeg), 0), "test: "))
    assert found == 1 and corrupt == 0 and "restored and saved" in message
    trainer_n._capture_prepared_dataset_snapshot(run_n, dataset)
    _, trainer_s, run_s = new_run(tmp_path, dataset, target, "s")
    trainer_s._capture_prepared_dataset_snapshot(run_s, dataset)
    assert run_n.training_dataset_snapshot["dataset_id"] == run_s.training_dataset_snapshot["dataset_id"]
    assert run_n.training_dataset_snapshot["split_sha256"] == run_s.training_dataset_snapshot["split_sha256"]
    assert run_n.training_dataset_snapshot["dataset_id"] != original["dataset_id"]
    assert run_n.training_dataset_input_snapshot == original
    persisted = TrainingHistory(history_n.history_dir).get_run(run_n.id)
    assert persisted.training_dataset_input_snapshot == original
    assert persisted.dataset_preparation["content_changed_during_preparation"] is True
    for run in (run_n, run_s):
        payload = build_model_training_provenance(run, include_dataset_fingerprint=False)
        assert payload["dataset_id"] == run_s.training_dataset_snapshot["dataset_id"]
        assert payload["dataset"]["dataset_id"] == run_s.training_dataset_snapshot["dataset_id"]


def test_resume_uses_prepared_identity_and_rejects_later_content_change(tmp_path):
    dataset = make_dataset(tmp_path, "char")
    _, trainer, run = new_run(tmp_path, dataset, "char", "n")
    trainer._capture_prepared_dataset_snapshot(run, dataset)
    saved = copy.deepcopy(run.training_dataset_snapshot)
    trainer._capture_prepared_dataset_snapshot(run, dataset, is_resuming=True)
    assert run.training_dataset_snapshot == saved
    (dataset / "labels/train/one.txt").write_text("0 0.4 0.4 0.4 0.4\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="zmienił się"):
        trainer._capture_prepared_dataset_snapshot(run, dataset, is_resuming=True)
    assert run.training_dataset_snapshot == saved


def test_identically_named_but_different_datasets_remain_distinct(tmp_path):
    dataset = make_dataset(tmp_path, "plate")
    _, trainer, run = new_run(tmp_path, dataset, "plate", "n")
    trainer._capture_prepared_dataset_snapshot(run, dataset)
    first = copy.deepcopy(run.training_dataset_snapshot)
    (dataset / "labels/train/one.txt").write_text("0 0.25 0.25 0.25 0.25\n", encoding="utf-8")
    _, next_trainer, next_run = new_run(tmp_path, dataset, "plate", "s")
    next_trainer._capture_prepared_dataset_snapshot(next_run, dataset)
    assert first["dataset_id"] != next_run.training_dataset_snapshot["dataset_id"]
    assert training_dataset_snapshots_match(first, next_run.training_dataset_snapshot)[0] is False


def test_verified_historical_correction_keeps_its_reconstruction_source(tmp_path):
    dataset = make_dataset(tmp_path, "plate")
    _, trainer, run = new_run(tmp_path, dataset, "plate", "n")
    trainer._capture_prepared_dataset_snapshot(run, dataset)
    run.training_dataset_snapshot["snapshot_source"] = "reconstructed_from_training_artifacts"
    payload = build_model_training_provenance(run, include_dataset_fingerprint=False)
    assert payload["provenance_capture"] == "reconstructed_from_training_artifacts"
    assert payload["input_dataset_snapshot"] == run.training_dataset_input_snapshot
