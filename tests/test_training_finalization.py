import csv
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_annotation_tool.config import get_torch_module
from auto_annotation_tool.training.trainer import YOLOPoseTrainer
from auto_annotation_tool.training.training_history import TrainingHistory


class TrainingFinalizationTests(unittest.TestCase):
    def _run_training(self, root, *, target="char", completed=150, fault=""):
        torch = get_torch_module()
        if torch is None:
            self.skipTest("PyTorch unavailable")
        history = TrainingHistory(root / "history")
        dataset = root / "dataset"
        for split in ("train", "val"):
            (dataset / "images" / split).mkdir(parents=True)
            (dataset / "images" / split / "sample.jpg").write_bytes(b"test image")
        (dataset / "data.yaml").write_text("train: images/train\nval: images/val\nnames: [sample]\n", encoding="utf-8")
        # Deliberately misleading names: an explicit char target must win.
        run = history.create_run("plate_project_training", epochs=150, training_target=target,
                                 dataset_path=str(root / "dataset"),
                                 base_model=str(root / "pose_project" / "custom.pt"))
        trainer = YOLOPoseTrainer(history)
        trainer.current_run = run
        trainer.is_training = True
        trainer.on_training_end = Mock(side_effect=RuntimeError("UI unavailable") if fault == "callback" else None)
        model_dir = root / "models"
        campaign = Mock()
        campaign.get_dir.return_value = model_dir
        campaign.get_active_project_name.return_value = "project"
        campaign.get_current_step.return_value = 4

        class FakeYOLO:
            task = "pose" if target == "plate" else "detect"

            def __init__(self, path):
                self.callbacks = {}

            def add_callback(self, name, callback):
                self.callbacks[name] = callback

            def train(self, **kwargs):
                self.callbacks["on_pretrain_routine_end"](self)
                output = Path(kwargs["project"]) / "train"
                weights = output / "weights"
                weights.mkdir(parents=True)
                rows = []
                for epoch in range(1, completed + 1):
                    score = 0.8 if epoch == max(1, completed - 1) else 0.7
                    metrics = {"metrics/mAP50(B)": 0.9, "metrics/mAP50-95(B)": score}
                    if target == "plate":
                        metrics.update({"metrics/mAP50(P)": 0.85, "metrics/mAP50-95(P)": score})
                    rows.append({"epoch": epoch, **metrics})
                    self.callbacks["on_fit_epoch_end"](SimpleNamespace(epoch=epoch - 1, metrics=metrics, loss=1.0))
                with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                results = {key: [row[key] for row in rows] for key in rows[0]}
                for name, row in (("best.pt", rows[max(0, completed - 2)]), ("last.pt", rows[-1])):
                    torch.save({"epoch": -1, "optimizer": None,
                                "train_metrics": {k: v for k, v in row.items() if k != "epoch"},
                                "train_results": results}, weights / name)
                # Ultralytics emits another fit callback for best.pt validation.
                self.callbacks["on_fit_epoch_end"](SimpleNamespace(
                    epoch=completed - 1, metrics=rows[max(0, completed - 2)], loss=1.0))
                if fault == "validation":
                    raise RuntimeError("final validation failed")
                if fault == "missing_weights":
                    for path in weights.iterdir():
                        path.unlink()

        def plots(train_dir, run_dir):
            out = Path(run_dir) / "plots"
            out.mkdir(exist_ok=True)
            return out

        with ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", {
                "auto_annotation_tool.campaign_manager": SimpleNamespace(CAMPAIGN=campaign)}))
            stack.enter_context(patch("auto_annotation_tool.training.trainer.YOLO_AVAILABLE", True))
            stack.enter_context(patch("auto_annotation_tool.training.trainer.get_yolo_class", return_value=FakeYOLO))
            stack.enter_context(patch("auto_annotation_tool.training.trainer._patch_ultralytics_save_model_closed_file_bug"))
            stack.enter_context(patch.object(trainer, "_apply_ultralytics_runtime_safety_overrides"))
            stack.enter_context(patch.object(trainer, "_get_dataset_runtime_profile", return_value={"is_pose": target == "plate"}))
            stack.enter_context(patch.object(trainer, "_is_ram_pressure_high", return_value=False))
            stack.enter_context(patch.object(trainer, "_reset_runtime_state"))
            stack.enter_context(patch("auto_annotation_tool.training.trainer.TrainingReportGenerator.export_plots", side_effect=plots))
            if fault == "snapshot":
                stack.enter_context(patch("auto_annotation_tool.training.trainer.build_output_checkpoint_training_snapshot",
                                          side_effect=OSError("snapshot unavailable")))
            elif fault == "epoch_metadata":
                stack.enter_context(patch.object(trainer, "_resolve_best_epoch_for_snapshot",
                                                side_effect=AttributeError("metadata helper unavailable")))
            elif fault == "report":
                stack.enter_context(patch("auto_annotation_tool.training.trainer.TrainingReportGenerator.generate_html",
                                          side_effect=OSError("report unavailable")))
            elif fault == "copy":
                stack.enter_context(patch("shutil.copy2", side_effect=PermissionError("model directory unavailable")))
            trainer._training_loop(run.base_model, run.dataset_path, 150, 2, 64, "cpu", 0.01, None)

        persisted = TrainingHistory(root / "history").get_run(run.id)
        return trainer, persisted, campaign, model_dir

    def test_detect_chars_and_pose_complete_with_stripped_checkpoints(self):
        for target in ("char", "plate"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                trainer, run, campaign, models = self._run_training(Path(tmp), target=target)
                self.assertEqual(run.status, "completed", run.error_message)
                self.assertEqual(run.current_epoch, 150)
                self.assertEqual(run.output_checkpoint_snapshot["best_epoch"], 149)
                self.assertTrue(Path(run.report_html).is_file())
                self.assertTrue(Path(run.best_weights).is_file())
                self.assertTrue(Path(run.last_weights).is_file())
                copied = list((models / "trained" / ("chars" if target == "char" else "plates")).glob("*.pt"))
                self.assertEqual(len(copied), 1)
                self.assertEqual(copied[0].read_bytes(), Path(run.best_weights).read_bytes())
                campaign.set_global_model.assert_called_once_with(target, str(copied[0]))
                trainer.on_training_end.assert_called_once_with(True, "Trening zakończony")
                self.assertFalse(trainer.is_training)

    def test_early_stopping_preserves_executed_epoch_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, run, _, _ = self._run_training(Path(tmp), completed=3)
            self.assertEqual(run.status, "completed", run.error_message)
            self.assertEqual(run.current_epoch, 3)
            self.assertEqual(run.epochs, 150)
            self.assertEqual(run.output_checkpoint_snapshot["best_epoch"], 2)

    def test_optional_finalization_errors_do_not_invalidate_detect_weights(self):
        for fault in ("snapshot", "epoch_metadata", "report", "copy", "callback"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                trainer, run, _, _ = self._run_training(Path(tmp), completed=3, fault=fault)
                self.assertEqual(run.status, "completed", run.error_message)
                self.assertEqual(run.current_epoch, 3)
                self.assertEqual(run.error_message, "")
                self.assertTrue(Path(run.best_weights).is_file())
                self.assertTrue(Path(run.last_weights).is_file())
                self.assertEqual(trainer.on_training_end.call_count, 1)
                self.assertTrue(trainer.on_training_end.call_args.args[0])

    def test_actual_validation_failure_is_not_promoted_to_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, run, campaign, _ = self._run_training(Path(tmp), completed=3, fault="validation")
            self.assertEqual(run.status, "failed")
            self.assertIn("final validation failed", run.error_message)
            self.assertEqual(run.best_weights, "")
            self.assertTrue(Path(run.last_weights).is_file())
            self.assertFalse(trainer.on_training_end.call_args.args[0])
            campaign.set_global_model.assert_not_called()

    def test_no_saved_weights_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, run, campaign, _ = self._run_training(Path(tmp), completed=3, fault="missing_weights")
            self.assertEqual(run.status, "failed")
            self.assertFalse(trainer.on_training_end.call_args.args[0])
            campaign.set_global_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
