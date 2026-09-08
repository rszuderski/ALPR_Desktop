import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from auto_annotation_tool.config import CONFIG, get_torch_module
from auto_annotation_tool.training.model_provenance import (
    TOTAL_EPOCHS_SCOPE,
    build_checkpoint_training_snapshot,
    build_model_training_provenance,
    build_output_checkpoint_training_snapshot,
    build_training_dataset_snapshot,
    training_dataset_snapshots_match,
)
from auto_annotation_tool.training.trainer import YOLOPoseTrainer
from auto_annotation_tool.training.training_history import TrainingHistory, TrainingRun


def _write_file(path: Path, data: bytes | str = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")


def _make_dataset(root: Path, name: str = "dataset") -> Path:
    dataset = root / name
    for split in ("train", "val", "test"):
        _write_file(dataset / "images" / split / f"{split}_001.jpg", b"image")
        _write_file(dataset / "labels" / split / f"{split}_001.txt", "0 0.5 0.5 0.2 0.2\n")
    _write_file(
        dataset / "data.yaml",
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: plate\n",
    )
    _write_file(
        dataset / "training_variant_manifest.json",
        json.dumps(
            {
                "source_images": 3,
                "source_objects": 3,
                "offline_augmentation_train_added": 0,
                "split_seed": 42,
            },
            ensure_ascii=False,
        ),
    )
    return dataset


def _set_train_image_count(dataset: Path, count: int) -> None:
    image_dir = dataset / "images" / "train"
    label_dir = dataset / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for path in list(image_dir.glob("*")):
        path.unlink()
    for path in list(label_dir.glob("*")):
        path.unlink()
    for index in range(1, count + 1):
        _write_file(image_dir / f"train_{index:03d}.jpg", b"image")
        _write_file(label_dir / f"train_{index:03d}.txt", "0 0.5 0.5 0.2 0.2\n")


def _run(
    run_id: str,
    dataset: Path,
    *,
    epochs: int,
    current_epoch: int | None = None,
    lineage_mode: str = "new",
    parent_run_id: str = "",
    base_model: str = "yolo26n.pt",
    target: str = "plate",
    with_snapshots: bool = True,
) -> dict:
    best_weights = dataset.parent / f"{run_id}_best.pt"
    _write_file(best_weights, b"best checkpoint")
    input_checkpoint = Path(base_model)
    if not input_checkpoint.is_absolute():
        input_checkpoint = dataset.parent / input_checkpoint.name
    _write_file(input_checkpoint, b"input checkpoint")
    payload = {
        "id": run_id,
        "name": f"Run {run_id}",
        "status": "completed",
        "dataset_path": str(dataset),
        "base_model": str(input_checkpoint),
        "epochs": epochs,
        "current_epoch": epochs if current_epoch is None else current_epoch,
        "lineage_mode": lineage_mode,
        "parent_run_id": parent_run_id,
        "parent_model_target": target,
        "training_target": target,
        "best_weights": str(best_weights),
    }
    if with_snapshots:
        payload["training_dataset_snapshot"] = build_training_dataset_snapshot(dataset, target=target)
        payload["input_checkpoint_snapshot"] = build_checkpoint_training_snapshot(
            input_checkpoint,
            name=input_checkpoint.name,
            kind="custom_parent" if lineage_mode == "fine_tune" else "pretrained_base",
        )
        payload["output_checkpoint_snapshot"] = build_output_checkpoint_training_snapshot(
            best_checkpoint=best_weights,
            best_epoch=epochs if current_epoch is None else current_epoch,
        )
    return payload


class ModelTrainingProvenanceTests(unittest.TestCase):
    def test_legacy_new_run_resolves_parent_by_exact_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            parent = _run("20260722_151706", dataset, epochs=130, with_snapshots=False)
            child = _run("20260802_160305", dataset, epochs=1,
                         base_model=parent["best_weights"], with_snapshots=False)
            before = json.dumps(child, sort_keys=True)
            provenance = build_model_training_provenance(child, history_index={parent["id"]: parent})
            self.assertEqual(provenance["total_epochs"], 131)
            self.assertEqual(provenance["run_epochs_completed"], 1)
            self.assertEqual(provenance["parent_run_id"], parent["id"])
            self.assertEqual(provenance["parent_resolution"], "history_checkpoint_path:best")
            self.assertEqual(provenance["mode"], "fine_tune")
            self.assertEqual(provenance["lineage_stage_count"], 2)
            self.assertTrue(provenance["total_epochs_known"])
            self.assertEqual(json.dumps(child, sort_keys=True), before)

    def test_legacy_parent_is_not_guessed_from_timestamp_in_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _make_dataset(root)
            parent = _run("20260722_151706", dataset, epochs=130, with_snapshots=False)
            child = _run("20260802_160305", dataset, epochs=1,
                         base_model=str(root / "elsewhere" / Path(parent["best_weights"]).name), with_snapshots=False)
            provenance = build_model_training_provenance(child, history_index={parent["id"]: parent})
            self.assertIsNone(provenance["total_epochs"])
            self.assertEqual(provenance["known_epochs_minimum"], 1)

    def test_ambiguous_legacy_checkpoint_reference_is_not_summed(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            parent = _run("20260722_151706", dataset, epochs=130, with_snapshots=False)
            other = {**parent, "id": "other"}
            child = _run("20260802_160305", dataset, epochs=1,
                         base_model=parent["best_weights"], with_snapshots=False)
            provenance = build_model_training_provenance(child, history_index={parent["id"]: parent, "other": other})
            self.assertIsNone(provenance["total_epochs"])

    def test_legacy_parent_with_conflicting_frozen_hash_is_not_summed(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            parent = _run("20260722_151706", dataset, epochs=130)
            child = _run("20260802_160305", dataset, epochs=1, base_model=parent["best_weights"])
            # _run writes different input bytes: same path, different checkpoint.
            provenance = build_model_training_provenance(child, history_index={parent["id"]: parent})
            self.assertIsNone(provenance["total_epochs"])

    def test_legacy_chain_uses_completed_epochs_not_early_stop_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _make_dataset(root)
            parent = _run("20260714_223004", dataset, epochs=50, with_snapshots=False)
            child = _run("20260721_194742", dataset, epochs=100,
                         base_model=parent["best_weights"], with_snapshots=False)
            child["output_dir"] = str(root / "run")
            _write_file(root / "run/train/results.csv", "epoch,loss\n" + "".join(f"{i},1\n" for i in range(1, 57)))
            provenance = build_model_training_provenance(child, history_index={parent["id"]: parent})
            self.assertEqual(provenance["run_epochs_planned"], 100)
            self.assertEqual(provenance["run_epochs_completed"], 56)
            self.assertEqual(provenance["total_epochs"], 106)

    def test_legacy_completed_rows_override_stale_counter_in_both_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            for current in (1, 100):
                run = _run("20260901_090000", dataset, epochs=100, current_epoch=current, with_snapshots=False)
                run["metrics_history"] = [{"epoch": i} for i in range(1, 57)]
                provenance = build_model_training_provenance(run)
                self.assertEqual(provenance["run_epochs_completed"], 56)

    def test_legacy_multigeneration_parent_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            first = _run("20260901_090000", dataset, epochs=30, with_snapshots=False)
            second = _run("20260902_090000", dataset, epochs=10, base_model=first["best_weights"], with_snapshots=False)
            third = _run("20260903_090000", dataset, epochs=1, base_model=second["best_weights"], with_snapshots=False)
            provenance = build_model_training_provenance(third, history_index={first["id"]: first, second["id"]: second})
            self.assertEqual(provenance["total_epochs"], 41)
            self.assertEqual(provenance["lineage_stage_count"], 3)

    def test_new_pretrained_run_counts_project_epochs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            run = _run("20260901_090000", dataset, epochs=30)

            provenance = build_model_training_provenance(run, history_index={}, target="plate")

            self.assertEqual(provenance["run_epochs_completed"], 30)
            self.assertEqual(provenance["total_epochs"], 30)
            self.assertTrue(provenance["total_epochs_known"])
            self.assertEqual(provenance["total_epochs_scope"], TOTAL_EPOCHS_SCOPE)
            self.assertEqual(provenance["provenance_status"], "complete")
            self.assertEqual(provenance["provenance_capture"], "frozen_at_training_start")
            self.assertEqual(provenance["lineage_stage_count"], 1)
            self.assertEqual(provenance["run_train_images"], 1)
            self.assertEqual(provenance["run_nominal_sample_presentations"], 30)
            self.assertEqual(provenance["lineage_nominal_sample_presentations"], 30)
            self.assertTrue(provenance["sample_presentations_known"])

    def test_fine_tune_sums_parent_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            parent = _run("20260901_090000", dataset, epochs=30)
            child = _run(
                "20260902_100000",
                dataset,
                epochs=10,
                lineage_mode="fine_tune",
                parent_run_id=parent["id"],
                base_model=str(dataset.parent / "parent_best.pt"),
            )

            provenance = build_model_training_provenance(
                child,
                history_index={parent["id"]: parent},
                target="plate",
            )

            self.assertEqual(provenance["run_epochs_completed"], 10)
            self.assertEqual(provenance["total_epochs"], 40)
            self.assertTrue(provenance["total_epochs_known"])
            self.assertEqual(provenance["lineage_depth"], 2)
            self.assertEqual(provenance["lineage_stage_count"], 2)
            self.assertEqual(provenance["lineage_nominal_sample_presentations"], 40)

    def test_three_generation_fine_tune_sums_recursively(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            run_a = _run("20260901_090000", dataset, epochs=30)
            run_b = _run("20260902_100000", dataset, epochs=10, lineage_mode="fine_tune", parent_run_id=run_a["id"])
            run_c = _run("20260903_110000", dataset, epochs=5, lineage_mode="fine_tune", parent_run_id=run_b["id"])

            provenance = build_model_training_provenance(
                run_c,
                history_index={run_a["id"]: run_a, run_b["id"]: run_b},
                target="plate",
            )

            self.assertEqual(provenance["total_epochs"], 45)
            self.assertEqual(provenance["known_epochs_minimum"], 45)
            self.assertEqual([row["epochs_completed"] for row in provenance["lineage"]], [30, 10, 5])
            self.assertEqual(provenance["lineage_stage_count"], 3)
            self.assertEqual(provenance["lineage_nominal_sample_presentations"], 45)

    def test_resume_is_not_summed_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            run = _run("20260904_120000", dataset, epochs=100, current_epoch=100, lineage_mode="resume")

            provenance = build_model_training_provenance(run, history_index={}, target="char")

            self.assertEqual(provenance["run_epochs_completed"], 100)
            self.assertEqual(provenance["total_epochs"], 100)
            self.assertEqual(provenance["lineage_depth"], 1)
            self.assertEqual(provenance["lineage_stage_count"], 1)
            self.assertEqual(provenance["lineage_nominal_sample_presentations"], 100)

    def test_missing_legacy_parent_keeps_only_known_minimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            run = _run(
                "20260905_130000",
                dataset,
                epochs=10,
                lineage_mode="fine_tune",
                parent_run_id="20260801_090000",
                base_model=str(Path(tmp) / "external_best.pt"),
            )

            provenance = build_model_training_provenance(run, history_index={}, target="char")

            self.assertIsNone(provenance["total_epochs"])
            self.assertFalse(provenance["total_epochs_known"])
            self.assertEqual(provenance["known_epochs_minimum"], 10)
            self.assertIsNone(provenance["lineage_nominal_sample_presentations"])
            self.assertFalse(provenance["sample_presentations_known"])
            self.assertEqual(provenance["known_sample_presentations_minimum"], 10)
            self.assertEqual(provenance["provenance_status"], "partial")

    def test_cycle_in_lineage_is_partial_not_infinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            run_a = _run("20260906_140000", dataset, epochs=7, lineage_mode="fine_tune", parent_run_id="20260906_150000")
            run_b = _run("20260906_150000", dataset, epochs=8, lineage_mode="fine_tune", parent_run_id=run_a["id"])

            provenance = build_model_training_provenance(
                run_a,
                history_index={run_a["id"]: run_a, run_b["id"]: run_b},
                target="plate",
            )

            self.assertIsNone(provenance["total_epochs"])
            self.assertFalse(provenance["total_epochs_known"])
            self.assertEqual(provenance["provenance_status"], "partial")
            self.assertTrue(any("cykl" in warning.lower() for warning in provenance["warnings"]))

    def test_snapshot_does_not_change_after_dataset_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            run = _run("20260907_090000", dataset, epochs=10)

            _write_file(dataset / "images" / "train" / "train_002.jpg", b"new image")
            _write_file(dataset / "labels" / "train" / "train_002.txt", "0 0.5 0.5 0.2 0.2\n")

            provenance = build_model_training_provenance(run, history_index={}, target="plate")

            self.assertEqual(provenance["run_train_images"], 1)
            self.assertEqual(provenance["run_nominal_sample_presentations"], 10)
            self.assertEqual(provenance["dataset"]["train_images"], 1)
            self.assertEqual(provenance["provenance_capture"], "frozen_at_training_start")

    def test_snapshot_survives_deleted_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            run = _run("20260907_100000", dataset, epochs=12)
            expected_dataset_id = run["training_dataset_snapshot"]["dataset_id"]
            expected_split_sha = run["training_dataset_snapshot"]["split_sha256"]

            shutil.rmtree(dataset)
            provenance = build_model_training_provenance(run, history_index={}, target="plate")

            self.assertEqual(provenance["dataset"]["dataset_id"], expected_dataset_id)
            self.assertEqual(provenance["dataset"]["split_sha256"], expected_split_sha)
            self.assertEqual(provenance["run_train_images"], 1)
            self.assertEqual(provenance["run_nominal_sample_presentations"], 12)

    def test_three_generation_sample_presentations_use_each_stage_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_a = _make_dataset(root, "dataset_a")
            dataset_b = _make_dataset(root, "dataset_b")
            dataset_c = _make_dataset(root, "dataset_c")
            _set_train_image_count(dataset_a, 2)
            _set_train_image_count(dataset_b, 3)
            _set_train_image_count(dataset_c, 4)
            run_a = _run("20260908_090000", dataset_a, epochs=30)
            run_b = _run("20260908_100000", dataset_b, epochs=20, lineage_mode="fine_tune", parent_run_id=run_a["id"])
            run_c = _run("20260908_110000", dataset_c, epochs=10, lineage_mode="fine_tune", parent_run_id=run_b["id"])

            provenance = build_model_training_provenance(
                run_c,
                history_index={run_a["id"]: run_a, run_b["id"]: run_b},
                target="plate",
            )

            self.assertEqual(provenance["lineage_stage_count"], 3)
            self.assertEqual(provenance["total_epochs"], 60)
            self.assertEqual(
                [row["nominal_sample_presentations"] for row in provenance["lineage"]],
                [60, 60, 40],
            )
            self.assertEqual(provenance["lineage_nominal_sample_presentations"], 160)

    def test_best_epoch_does_not_replace_completed_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            run = _run("20260909_090000", dataset, epochs=100, current_epoch=100)
            run["output_checkpoint_snapshot"] = build_output_checkpoint_training_snapshot(
                best_checkpoint=run["best_weights"],
                best_epoch=73,
            )

            provenance = build_model_training_provenance(run, history_index={}, target="plate")

            self.assertEqual(provenance["best_epoch"], 73)
            self.assertEqual(provenance["run_epochs_completed"], 100)
            self.assertEqual(provenance["total_epochs"], 100)

    def test_checkpoint_swap_is_reported_without_rewriting_frozen_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            run = _run("20260910_090000", dataset, epochs=5)
            frozen_sha = run["output_checkpoint_snapshot"]["best_checkpoint_sha256"]

            Path(run["best_weights"]).write_bytes(b"changed checkpoint")
            provenance = build_model_training_provenance(run, history_index={}, target="plate")

            self.assertEqual(provenance["best_checkpoint_sha256"], frozen_sha)
            self.assertNotEqual(frozen_sha, "")
            self.assertTrue(any("best.pt" in warning for warning in provenance["warnings"]))
            self.assertEqual(provenance["provenance_status"], "partial")

    def test_resume_dataset_snapshot_allows_same_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            stored = build_training_dataset_snapshot(dataset, target="plate")
            current = build_training_dataset_snapshot(dataset, target="plate")

            matches, reason = training_dataset_snapshots_match(stored, current)

            self.assertTrue(matches, reason)

    def test_resume_dataset_snapshot_blocks_changed_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            stored = build_training_dataset_snapshot(dataset, target="plate")
            _write_file(dataset / "images" / "train" / "train_999.jpg", b"changed")
            _write_file(dataset / "labels" / "train" / "train_999.txt", "0 0.5 0.5 0.2 0.2\n")
            current = build_training_dataset_snapshot(dataset, target="plate")

            matches, reason = training_dataset_snapshots_match(stored, current)

            self.assertFalse(matches)
            self.assertIn("splitu", reason)

    def test_trainer_resume_guard_blocks_changed_dataset_for_controlled_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            run = TrainingRun(
                id="20260911_080000",
                name="Run resume guard",
                created_at="2026-09-11T08:00:00",
                dataset_path=str(dataset),
                training_target="plate",
            )
            run.training_dataset_snapshot = build_training_dataset_snapshot(dataset, target="plate")
            _write_file(dataset / "images" / "train" / "train_999.jpg", b"changed")
            _write_file(dataset / "labels" / "train" / "train_999.txt", "0 0.5 0.5 0.2 0.2\n")
            trainer = YOLOPoseTrainer.__new__(YOLOPoseTrainer)

            allowed = trainer._validate_resume_dataset_contract(
                run,
                str(dataset),
                training_target="plate",
                require_snapshot=True,
            )

            self.assertFalse(allowed)

    def test_early_stop_uses_completed_epochs_not_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            run = _run("20260911_090000", dataset, epochs=200, current_epoch=143)

            provenance = build_model_training_provenance(run, history_index={}, target="plate")

            self.assertEqual(provenance["run_epochs_planned"], 200)
            self.assertEqual(provenance["run_epochs_completed"], 143)
            self.assertEqual(provenance["run_nominal_sample_presentations"], 143)
            self.assertEqual(provenance["total_epochs"], 143)

    def test_zero_current_epoch_falls_back_to_results_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            output_dir = Path(tmp) / "training_runs" / "run_csv"
            rows = ["epoch,metrics/mAP50(B)\n"]
            rows.extend(f"{index},0.{index % 10}\n" for index in range(17))
            _write_file(output_dir / "train" / "results.csv", "".join(rows))
            run = _run("20260911_100000", dataset, epochs=200, current_epoch=0)
            run["output_dir"] = str(output_dir)

            provenance = build_model_training_provenance(run, history_index={}, target="plate")

            self.assertEqual(provenance["run_epochs_planned"], 200)
            self.assertEqual(provenance["run_epochs_completed"], 17)
            self.assertEqual(provenance["total_epochs"], 17)

    def test_zero_current_epoch_falls_back_to_last_checkpoint(self):
        torch = get_torch_module()
        if torch is None:
            self.skipTest("PyTorch niedostępny")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            checkpoint = Path(tmp) / "last.pt"
            torch.save({"epoch": 16}, checkpoint)
            run = _run("20260911_110000", dataset, epochs=200, current_epoch=0)
            run["last_weights"] = str(checkpoint)

            provenance = build_model_training_provenance(run, history_index={}, target="plate")

            self.assertEqual(provenance["run_epochs_planned"], 200)
            self.assertEqual(provenance["run_epochs_completed"], 17)
            self.assertEqual(provenance["total_epochs"], 17)

    def test_materialize_official_pretrained_downloads_before_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "base_models" / "plates"

            def fake_download(target):
                path = Path(target)
                _write_file(path, b"downloaded checkpoint")
                return str(path)

            trainer = YOLOPoseTrainer.__new__(YOLOPoseTrainer)
            with patch.object(CONFIG, "get_base_models_dir", return_value=str(base_dir)):
                with patch("ultralytics.utils.downloads.attempt_download_asset", side_effect=fake_download):
                    materialized = Path(
                        trainer._materialize_official_pretrained_checkpoint(
                            "yolo11n-pose",
                            "yolo11n-pose.pt",
                            training_target="plate",
                        )
                    )

            snapshot = build_checkpoint_training_snapshot(
                materialized,
                name=materialized.name,
                kind="pretrained_base",
            )

            self.assertTrue(materialized.exists())
            self.assertTrue(snapshot["exists_at_capture"])
            self.assertNotEqual(snapshot["sha256"], "")
            self.assertEqual(snapshot["kind"], "pretrained_base")

    def test_pretrained_materialize_failure_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(Path(tmp))
            history = TrainingHistory(Path(tmp) / "history")
            trainer = YOLOPoseTrainer(history=history)
            trainer.validate_dataset = lambda _path: (True, "Dataset OK", {})
            trainer._reset_runtime_state = lambda: None
            trainer._reset_worker_ipc_state = lambda: None

            with patch("auto_annotation_tool.training.trainer.YOLO_AVAILABLE", True):
                with patch("auto_annotation_tool.training.trainer.get_yolo_class", return_value=object):
                    with patch.object(CONFIG, "get_base_models_dir", return_value=str(Path(tmp) / "base_models")):
                        with patch(
                            "ultralytics.utils.downloads.attempt_download_asset",
                            side_effect=RuntimeError("offline"),
                        ):
                            run_id = trainer.start_training(
                                name="Pretrained failure",
                                dataset_path=str(dataset),
                                base_model="yolo11n-pose",
                                epochs=1,
                                batch_size=1,
                                img_size=256,
                                device="cpu",
                                training_target="plate",
                            )

            self.assertIsNone(run_id)
            self.assertEqual(history.runs, {})

    def test_external_parent_known_epochs_does_not_mean_known_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _make_dataset(root)
            parent_package = root / "external_parent.alprmodel"
            with zipfile.ZipFile(parent_package, "w") as archive:
                archive.writestr(
                    "manifest.json",
                    json.dumps(
                        {
                            "training": {
                                "total_epochs": 70,
                                "total_epochs_known": True,
                            }
                        }
                    ),
                )
            run = _run(
                "20260912_090000",
                dataset,
                epochs=10,
                lineage_mode="fine_tune",
                parent_run_id="",
                base_model=str(root / "child_start.pt"),
            )
            run["parent_model_path"] = str(parent_package)
            run["base_model"] = str(parent_package)

            provenance = build_model_training_provenance(run, history_index={}, target="plate")

            self.assertEqual(provenance["total_epochs"], 80)
            self.assertTrue(provenance["total_epochs_known"])
            self.assertEqual(provenance["lineage_stage_count"], 2)
            self.assertFalse(provenance["lineage_stage_count_known"])
            self.assertEqual(provenance["known_stage_count_minimum"], 2)
            self.assertEqual(provenance["provenance_status"], "partial")
            self.assertIsNone(provenance["lineage_nominal_sample_presentations"])
            self.assertFalse(provenance["sample_presentations_known"])

    def test_stripped_checkpoint_falls_back_instead_of_reporting_epoch_zero(self):
        torch = get_torch_module()
        if torch is None:
            self.skipTest("PyTorch niedostępny")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best.pt"
            torch.save({"epoch": -1, "optimizer": None}, checkpoint)
            snapshot = build_output_checkpoint_training_snapshot(
                best_checkpoint=checkpoint, best_epoch=135, best_epoch_source="metrics_history")
            self.assertEqual(snapshot["best_epoch"], 135)
            self.assertEqual(snapshot["best_epoch_source"], "metrics_history")

    def test_stripped_checkpoint_matches_its_metrics_not_the_last_epoch(self):
        torch = get_torch_module()
        if torch is None:
            self.skipTest("PyTorch niedostępny")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best.pt"
            torch.save({"epoch": -1, "train_metrics": {"metrics/mAP50-95(B)": 0.78,
                                                      "metrics/mAP50-95(P)": 0.86},
                        "train_results": {"epoch": [135, 149, 150],
                                          "metrics/mAP50-95(B)": [0.78, 0.79, 0.77],
                                          "metrics/mAP50-95(P)": [0.86, 0.84, 0.84]}}, checkpoint)
            snapshot = build_output_checkpoint_training_snapshot(best_checkpoint=checkpoint, best_epoch=149)
            self.assertEqual(snapshot["best_epoch"], 135)
            self.assertEqual(snapshot["best_epoch_source"], "checkpoint")

    def test_ambiguous_stripped_metrics_do_not_override_fallback_epoch(self):
        torch = get_torch_module()
        if torch is None:
            self.skipTest("PyTorch niedostępny")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best.pt"
            torch.save({"epoch": -1, "train_metrics": {"metrics/mAP50-95(B)": 0.78},
                        "train_results": {"epoch": [135, 150], "metrics/mAP50-95(B)": [0.78, 0.78]}}, checkpoint)
            snapshot = build_output_checkpoint_training_snapshot(
                best_checkpoint=checkpoint, best_epoch=135, best_epoch_source="metrics_history")
            self.assertEqual(snapshot["best_epoch"], 135)
            self.assertEqual(snapshot["best_epoch_source"], "metrics_history")

    def test_best_epoch_prefers_checkpoint_epoch_when_available(self):
        torch = get_torch_module()
        if torch is None:
            self.skipTest("PyTorch niedostępny")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best.pt"
            torch.save({"epoch": 6}, checkpoint)

            snapshot = build_output_checkpoint_training_snapshot(
                best_checkpoint=checkpoint,
                best_epoch=3,
                best_epoch_source="metrics_history",
            )

            self.assertEqual(snapshot["best_epoch"], 7)
            self.assertEqual(snapshot["best_epoch_source"], "checkpoint")


if __name__ == "__main__":
    unittest.main()
