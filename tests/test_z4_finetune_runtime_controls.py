from pathlib import Path

from auto_annotation_tool.gui import z4_fine_tune_protocol as ft
from auto_annotation_tool.training.training_history import TrainingRun
from auto_annotation_tool.training.trainer import YOLOPoseTrainer


def test_ram_safe_can_preserve_learning_rate_for_finetune():
    trainer = YOLOPoseTrainer.__new__(YOLOPoseTrainer)
    result = trainer._build_ram_safe_training_attempt(
        batch_size=10,
        img_size=320,
        lr0=0.001,
        dataset_profile={"is_pose": False},
        label="test",
        preserve_lr=True,
    )
    assert result["batch_size"] == 2
    assert result["img_size"] == 256
    assert result["lr0"] == 0.001
    assert result["workers"] == 0


def test_legacy_ram_safe_keeps_legacy_lr_policy():
    trainer = YOLOPoseTrainer.__new__(YOLOPoseTrainer)
    result = trainer._build_ram_safe_training_attempt(
        batch_size=10,
        img_size=320,
        lr0=0.001,
        dataset_profile={"is_pose": False},
        label="legacy",
    )
    assert result["lr0"] == 0.0025


def test_training_run_has_protocol_fields():
    run = TrainingRun(id="x", name="x", created_at="x")
    assert isinstance(run.training_protocol_requested, dict)
    assert isinstance(run.training_protocol_snapshot, dict)


def test_gui_module_declares_both_memory_policies():
    assert ft.POLICY_FIXED == "Stałe parametry"
    assert ft.POLICY_ADAPTIVE == "Automatyczny RAM-safe"


def test_runtime_and_trainer_are_wired():
    root = Path(__file__).resolve().parents[1]
    runtime = (root / "auto_annotation_tool/gui/z4_training_runtime.py").read_text(encoding="utf-8-sig")
    trainer = (root / "auto_annotation_tool/training/trainer.py").read_text(encoding="utf-8-sig")
    assert "Z4FTGUI001-FIX02 runtime protocol" in runtime
    assert "request[\"training_protocol\"]" in runtime
    assert "Z4FTGUI001-FIX02 trainer protocol" in trainer
    assert "ram_safe_enabled" in trainer
    assert "preserve_lr=explicit_ft_protocol" in trainer
    assert "requested_workers" in trainer
