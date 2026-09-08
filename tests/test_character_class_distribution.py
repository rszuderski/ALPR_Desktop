from pathlib import Path
import json
import os
import random
import tempfile
import unittest

from auto_annotation_tool.gui import z4_dataset_builder
from auto_annotation_tool.gui.z4_dataset_readiness import get_training_dataset_readiness
from auto_annotation_tool.config import CV2_AVAILABLE, cv2, np
from auto_annotation_tool.training import AugmentationProfile, DatasetSplitter
from auto_annotation_tool.training.character_class_distribution import (
    CHARACTER_BALANCE_ALPHABET,
    CHARACTER_REPRESENTATION_THRESHOLD_POLICY,
    CHARACTER_TRAINING_VARIANT_SCHEMA,
    CharacterClassMapValidationError,
    analyze_character_class_distribution,
    build_character_training_variant_manifest,
    build_character_dataset_file_fingerprint,
    collect_character_train_augmentation_source_pool,
    compare_character_val_test_unchanged,
    find_character_real_source_candidates,
    plan_character_train_augmentation,
    preview_character_train_synthetic_counts_from_pool,
    save_character_distribution_artifacts,
    save_character_class_distribution_csv,
    save_character_class_distribution_json,
)
from auto_annotation_tool.training.dataset_augmentation import (
    _build_balance_augmented_source_state,
    _finalize_train_augmentation_result,
    _apply_traffic_headlight_effect,
    _select_augmentation_sample_pool,
    is_albumentations_available,
)


class _ValueVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Step4BuilderHost:
    def __init__(self):
        self.train_pct = _ValueVar(80.0)
        self.val_pct = _ValueVar(10.0)
        self._step4_dataset_mode = "char"
        self.split_aug_enabled_var = _ValueVar(True)
        self.split_aug_extra_var = _ValueVar(1)
        self.split_aug_sample_var = _ValueVar(1)
        self.split_mz_representation_status_var = _ValueVar("")
        self.split_progress_var = _ValueVar(0)
        self.split_status = _ValueVar("")
        self.captured_result_modals = []
        self.opened_pz2 = False

    def _get_dataset_split_image_counts(self, dataset_dir):
        root = Path(dataset_dir)
        counts = {}
        for split_name in ("train", "val", "test"):
            total = 0
            for image_dir in (root / "images" / split_name, root / split_name / "images"):
                if image_dir.exists():
                    total += sum(1 for path in image_dir.iterdir() if path.is_file())
            counts[split_name] = total
        counts["total"] = sum(counts.values())
        return counts

    def _build_step4_augmented_dataset_dir(self, source_dir, profile, target="char"):
        source = Path(source_dir)
        return source.parent / f"{source.name}_Aug_test"

    def _ui(self, callback):
        return callback()

    def _set_training_widget_text(self, widget, text):
        if hasattr(widget, "set"):
            widget.set(text)

    def _style_training_success_label(self, widget):
        return None

    def _apply_step4_dataset_postprocessing(self, *args, **kwargs):
        return z4_dataset_builder._apply_step4_dataset_postprocessing(self, *args, **kwargs)

    def _format_training_target_label(self, target):
        return "Znaki" if str(target) == "char" else "Tablice"

    def _can_open_pz2_after_dataset_result(self):
        return True

    def _show_step4_dataset_result_modal(self, **kwargs):
        self.captured_result_modals.append(dict(kwargs))
        return bool(kwargs.get("allow_pz2"))

    def _open_pz2_from_dataset_result(self, *args, **kwargs):
        self.opened_pz2 = True


def _write_balanced_char_dataset(root: Path, repeats: int = 2) -> None:
    (root / "labels" / "train").mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "test").mkdir(parents=True)
    (root / "images" / "test").mkdir(parents=True)
    (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
    for class_id, symbol in enumerate(CHARACTER_BALANCE_ALPHABET):
        stem = f"{class_id:02d}_{symbol}"
        (root / "labels" / "train" / f"{stem}.txt").write_text(
            (f"{class_id} 0.5 0.5 0.1 0.1\n" * max(1, repeats)),
            encoding="utf-8",
        )
        (root / "images" / "train" / f"{stem}.jpg").write_bytes(b"fake")


def _write_split_source_dataset(root: Path) -> None:
    (root / "labels").mkdir(parents=True)
    (root / "images").mkdir(parents=True)
    (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
    for index in range(18):
        class_id = index % len(CHARACTER_BALANCE_ALPHABET)
        stem = f"plate_{index:03d}"
        (root / "labels" / f"{stem}.txt").write_text(f"{class_id} 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (root / "images" / f"{stem}.jpg").write_bytes(b"fake")


def _write_valid_jpeg(path: Path, color: tuple[int, int, int] = (80, 90, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not CV2_AVAILABLE or cv2 is None or np is None:
        path.write_bytes(b"fake")
        return
    image = np.full((32, 64, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_char_split_source_with_one_q_deficit(root: Path) -> None:
    (root / "labels").mkdir(parents=True)
    (root / "images").mkdir(parents=True)
    (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
    q_id = CHARACTER_BALANCE_ALPHABET.index("Q")
    for class_id, symbol in enumerate(CHARACTER_BALANCE_ALPHABET):
        safe_symbol = "Q_target" if class_id == q_id else f"{class_id:02d}_{symbol}"
        repeat = 1 if class_id == q_id else 4
        stem = f"plate_{safe_symbol}"
        (root / "labels" / f"{stem}.txt").write_text(
            (f"{class_id} 0.5 0.5 0.2 0.2\n" * repeat),
            encoding="utf-8",
        )
        _write_valid_jpeg(root / "images" / f"{stem}.jpg", color=(40 + class_id % 80, 80, 130))


def _names_list_yaml() -> str:
    lines = ["path: .", "names:"]
    lines.extend(f"  - \"{symbol}\"" for symbol in CHARACTER_BALANCE_ALPHABET)
    return "\n".join(lines) + "\n"


def _names_dict_yaml() -> str:
    lines = ["path: .", "names:"]
    lines.extend(f"  {index}: \"{symbol}\"" for index, symbol in enumerate(CHARACTER_BALANCE_ALPHABET))
    return "\n".join(lines) + "\n"


class DatasetAugmentationLightingTests(unittest.TestCase):
    @unittest.skipUnless(CV2_AVAILABLE and np is not None, "requires OpenCV and numpy")
    def test_scene_headlights_keep_independent_color_maps(self):
        image = np.full((80, 240, 3), 60, dtype=np.uint8)
        common = {
            "night_light_strength": 0.0,
            "light_normal_strength": 0.0,
            "dark_relief_strength": 0.0,
            "traffic_headlight_1_cone": 0.12,
            "traffic_headlight_2_cone": 0.12,
            "traffic_headlight_1_source_radius": 0.0,
            "traffic_headlight_2_source_radius": 0.0,
            "traffic_headlight_source_world_x": -0.34,
            "traffic_headlight_source_world_y": 0.0,
            "traffic_headlight_source_world_z": 0.62,
            "traffic_headlight_target_world_x": -0.34,
            "traffic_headlight_target_world_y": 0.0,
            "traffic_headlight_2_source_world_x": 0.34,
            "traffic_headlight_2_source_world_y": 0.0,
            "traffic_headlight_2_source_world_z": 0.62,
            "traffic_headlight_2_target_world_x": 0.34,
            "traffic_headlight_2_target_world_y": 0.0,
            "traffic_headlight_1_r": 1.0,
            "traffic_headlight_1_g": 1.0,
            "traffic_headlight_1_b": 0.0,
            "traffic_headlight_2_r": 1.0,
            "traffic_headlight_2_g": 0.0,
            "traffic_headlight_2_b": 0.0,
        }
        red_only = AugmentationProfile(
            **common,
            traffic_headlight_strength=0.0,
            traffic_headlight_2_strength=0.85,
        )
        red_with_yellow_left = AugmentationProfile(
            **common,
            traffic_headlight_strength=0.85,
            traffic_headlight_2_strength=0.85,
        )

        red_only_image = _apply_traffic_headlight_effect(image, red_only, np.random.default_rng(1))
        combined_image = _apply_traffic_headlight_effect(image, red_with_yellow_left, np.random.default_rng(1))
        sample_x, sample_y = 201, 40
        red_only_patch = red_only_image[sample_y - 2 : sample_y + 3, sample_x - 2 : sample_x + 3].mean(axis=(0, 1))
        combined_patch = combined_image[sample_y - 2 : sample_y + 3, sample_x - 2 : sample_x + 3].mean(axis=(0, 1))

        self.assertLessEqual(abs(float(combined_patch[1]) - float(red_only_patch[1])), 4.0)
        self.assertGreater(float(combined_patch[2]) - float(combined_patch[1]), 70.0)


class CharacterClassDistributionTests(unittest.TestCase):
    def test_split_counts_invalid_lines_and_unique_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "labels" / "val").mkdir(parents=True)
            (root / "labels" / "test").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            (root / "labels" / "train" / "plate_a.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n"
                "0 0.6 0.5 0.1 0.1\n"
                "10 0.1 0.1 0.1 0.1\n",
                encoding="utf-8",
            )
            (root / "labels" / "train" / "plate_b.txt").write_text(
                "1 0.2 0.2 0.1 0.1\n36 0 0 0 0\nbad line\n",
                encoding="utf-8",
            )
            (root / "labels" / "val" / "plate_c.txt").write_text(
                "10 0.2 0.2 0.1 0.1\n",
                encoding="utf-8",
            )
            (root / "labels" / "test" / "plate_d.txt").write_text(
                "35 0.2 0.2 0.1 0.1\n",
                encoding="utf-8",
            )

            dist = analyze_character_class_distribution(root)
            rows = {row.symbol: row for row in dist.classes}

            self.assertEqual(dist.layout, "split")
            self.assertEqual(rows["0"].train_count, 2)
            self.assertEqual(rows["0"].unique_train_plate_count, 1)
            self.assertEqual(rows["A"].train_count, 1)
            self.assertEqual(rows["A"].val_count, 1)
            self.assertEqual(rows["Z"].test_count, 1)
            self.assertEqual(dist.summary["invalid_class_ids"], 1)
            self.assertEqual(dist.summary["invalid_label_lines"], 1)

            csv_path = save_character_class_distribution_csv(dist, root / "balance.csv")
            json_path = save_character_class_distribution_json(dist, root / "balance.json")
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

    def test_flat_layout_uses_total_without_faking_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels").mkdir()
            (root / "labels" / "flat.txt").write_text(
                "2 0.5 0.5 0.1 0.1\n2 0.6 0.5 0.1 0.1\n",
                encoding="utf-8",
            )

            dist = analyze_character_class_distribution(root)
            rows = {row.symbol: row for row in dist.classes}

            self.assertEqual(dist.layout, "flat")
            self.assertEqual(dist.diagnostic_split, "total")
            self.assertEqual(rows["2"].train_count, 0)
            self.assertEqual(rows["2"].total_count, 2)
            self.assertEqual(rows["2"].unique_total_plate_count, 1)

    def test_augmentation_manifest_preserves_source_diversity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_dict_yaml(), encoding="utf-8")
            (root / "labels" / "train" / "plate_001.txt").write_text(
                "4 0.5 0.5 0.1 0.1\n",
                encoding="utf-8",
            )
            (root / "labels" / "train" / "plate_002.txt").write_text(
                "4 0.5 0.5 0.1 0.1\n",
                encoding="utf-8",
            )
            (root / "augmentation_manifest.json").write_text(
                '{"generated_files":[{"label":"labels/train/plate_002.txt",'
                '"source_label":"labels/train/plate_001.txt"}]}',
                encoding="utf-8",
            )

            dist = analyze_character_class_distribution(root)
            rows = {row.symbol: row for row in dist.classes}

            self.assertEqual(rows["4"].train_count, 2)
            self.assertEqual(rows["4"].unique_train_plate_count, 1)

    def test_data_yaml_dict_names_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_dict_yaml(), encoding="utf-8")
            (root / "labels" / "train" / "plate.txt").write_text("10 0 0 0 0\n", encoding="utf-8")

            dist = analyze_character_class_distribution(root)

            self.assertEqual(dist.summary["class_map"]["status"], "OK")

    def test_target_count_and_deficit_use_half_median_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            (root / "labels" / "train" / "zeros.txt").write_text(
                "0 0 0 0 0\n"
                "0 0 0 0 0\n"
                "0 0 0 0 0\n"
                "0 0 0 0 0\n"
                "0 0 0 0 0\n"
                "0 0 0 0 0\n",
                encoding="utf-8",
            )
            (root / "labels" / "train" / "ones.txt").write_text(
                "1 0 0 0 0\n"
                "1 0 0 0 0\n"
                "1 0 0 0 0\n"
                "1 0 0 0 0\n",
                encoding="utf-8",
            )
            (root / "labels" / "train" / "twos.txt").write_text("2 0 0 0 0\n", encoding="utf-8")

            dist = analyze_character_class_distribution(root)
            rows = {row.symbol: row for row in dist.classes}

            self.assertEqual(dist.summary["target_class_count"], 2)
            self.assertEqual(rows["0"].deficit_count, 0)
            self.assertEqual(rows["1"].deficit_count, 0)
            self.assertEqual(rows["2"].deficit_count, 1)
            self.assertIn("2", dist.summary["deficit_classes"])

    def test_target_ratio_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            (root / "labels" / "train" / "zeros.txt").write_text("0 0 0 0 0\n0 0 0 0 0\n", encoding="utf-8")
            (root / "labels" / "train" / "ones.txt").write_text("1 0 0 0 0\n", encoding="utf-8")

            dist = analyze_character_class_distribution(root, target_ratio=1.0)
            rows = {row.symbol: row for row in dist.classes}

            self.assertEqual(dist.summary["target_class_count"], 2)
            self.assertEqual(rows["1"].deficit_count, 1)

    def test_train_augmentation_plan_uses_deficit_priority_and_ignores_val_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "labels" / "val").mkdir(parents=True)
            (root / "images" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            (root / "labels" / "train" / "zeros.txt").write_text(
                "0 0 0 0 0\n"
                "0 0 0 0 0\n"
                "0 0 0 0 0\n"
                "0 0 0 0 0\n"
                "0 0 0 0 0\n"
                "0 0 0 0 0\n",
                encoding="utf-8",
            )
            (root / "labels" / "train" / "ones.txt").write_text(
                "1 0 0 0 0\n"
                "1 0 0 0 0\n"
                "1 0 0 0 0\n"
                "1 0 0 0 0\n",
                encoding="utf-8",
            )
            (root / "labels" / "train" / "twos.txt").write_text("2 0 0 0 0\n", encoding="utf-8")
            (root / "images" / "train" / "twos.jpg").write_bytes(b"fake")
            (root / "labels" / "val" / "threes_val.txt").write_text("3 0 0 0 0\n", encoding="utf-8")

            plan = plan_character_train_augmentation(root, max_augmented_variants_per_source=2)

            self.assertEqual(plan.target_count, 2)
            self.assertEqual(plan.deficit_by_symbol["2"], 1)
            self.assertTrue(plan.candidates)
            self.assertEqual(plan.candidates[0].source_key, "twos")
            self.assertEqual(plan.candidates[0].priority, 1.0)
            self.assertEqual(plan.candidates[0].max_augmented_variants, 2)
            self.assertTrue(plan.candidates[0].image_path.endswith("twos.jpg"))
            self.assertFalse(any("threes_val" in candidate.label_path for candidate in plan.candidates))

    def test_training_variant_manifest_carries_before_after_and_source_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            (root / "labels" / "train" / "plate.txt").write_text("0 0 0 0 0\n", encoding="utf-8")
            before = analyze_character_class_distribution(root)
            plan = plan_character_train_augmentation(root, max_augmented_variants_per_source=4)
            refs = save_character_distribution_artifacts(before, root / "analysis", prefix="character_class_distribution_before")
            after_path = root / "analysis" / "character_class_distribution_after.json"
            after_path.write_text('{"after": true}', encoding="utf-8")
            base_fingerprint = build_character_dataset_file_fingerprint(root)
            val_test_guard = compare_character_val_test_unchanged(
                build_character_dataset_file_fingerprint(root, splits=("val", "test")),
                build_character_dataset_file_fingerprint(root, splits=("val", "test")),
            )

            manifest = build_character_training_variant_manifest(
                base_dataset=root,
                before_distribution=refs["json"]["path"],
                after_distribution=after_path,
                plan=plan,
                sources={"real": 7, "added_real": 2, "augmented_real": 3},
                base_dataset_sha_or_fingerprint=base_fingerprint,
                val_test_unchanged=val_test_guard,
            )

            self.assertEqual(manifest["schema"], CHARACTER_TRAINING_VARIANT_SCHEMA)
            self.assertEqual(manifest["alphabet"], CHARACTER_BALANCE_ALPHABET)
            self.assertEqual(manifest["selection_policy"], "deficit_progressive_reuse_v1")
            self.assertEqual(manifest["threshold_policy"], CHARACTER_REPRESENTATION_THRESHOLD_POLICY)
            self.assertEqual(manifest["planned_images"], plan.planned_images)
            self.assertEqual(manifest["predicted_deficit_after"], plan.predicted_deficit_after)
            self.assertEqual(manifest["unique_real_sources_used"], plan.unique_real_sources_used)
            self.assertEqual(manifest["reuse_rounds_used"], plan.reuse_rounds_used)
            self.assertEqual(manifest["sources"]["real"], 7)
            self.assertEqual(manifest["sources"]["added_real"], 2)
            self.assertEqual(manifest["sources"]["augmented_real"], 3)
            self.assertEqual(manifest["sources"]["synthetic"], 0)
            self.assertEqual(manifest["max_augmented_variants_per_source"], 4)
            self.assertEqual(manifest["source_reuse_safety_guard"]["max_augmented_variants_per_source"], 4)
            self.assertTrue(manifest["source_reuse_safety_guard"]["enabled"])
            self.assertEqual(manifest["augmentation"]["max_variants_per_source"], 4)
            self.assertTrue(manifest["before_distribution"]["path"].endswith("character_class_distribution_before.json"))
            self.assertRegex(manifest["before_distribution"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(manifest["after_distribution"]["path"].endswith("character_class_distribution_after.json"))
            self.assertRegex(manifest["after_distribution"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(manifest["val_test_unchanged"])
            self.assertRegex(manifest["base_dataset_sha_or_fingerprint"]["sha256"], r"^[0-9a-f]{64}$")

    def test_plan_deduplicates_augmented_source_key_and_prefers_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "images" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            (root / "labels" / "train" / "heavy.txt").write_text("0 0 0 0 0\n" * 8, encoding="utf-8")
            for stem in ("plate_001", "plate_001_aug1", "plate_001_aug2"):
                (root / "labels" / "train" / f"{stem}.txt").write_text("16 0 0 0 0\n", encoding="utf-8")
                (root / "images" / "train" / f"{stem}.jpg").write_bytes(b"fake")
            (root / "augmentation_manifest.json").write_text(
                json.dumps(
                    {
                        "generated_files": [
                            {"label": "labels/train/plate_001_aug1.txt", "source_label": "labels/train/plate_001.txt"},
                            {"label": "labels/train/plate_001_aug2.txt", "source_label": "labels/train/plate_001.txt"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            plan = plan_character_train_augmentation(root, target_ratio=1.0)
            source_candidates = [candidate for candidate in plan.candidates if candidate.source_key == "plate_001"]

            self.assertEqual(len(source_candidates), 1)
            self.assertTrue(source_candidates[0].label_path.endswith("plate_001.txt"))

    def test_plan_warns_when_only_augmented_representatives_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            (root / "labels" / "train" / "heavy.txt").write_text("0 0 0 0 0\n" * 8, encoding="utf-8")
            (root / "labels" / "train" / "plate_001_aug1.txt").write_text("16 0 0 0 0\n", encoding="utf-8")
            (root / "labels" / "train" / "plate_001_aug2.txt").write_text("16 0 0 0 0\n", encoding="utf-8")

            plan = plan_character_train_augmentation(root, target_ratio=1.0)

            self.assertEqual(sum(1 for candidate in plan.candidates if candidate.source_key == "plate_001"), 1)
            self.assertTrue(any("tylko kopie augmentowane" in warning for warning in plan.warnings))

    def test_plan_uses_symbol_multiplicity_for_planned_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "images" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            zero_id = CHARACTER_BALANCE_ALPHABET.index("0")
            one_id = CHARACTER_BALANCE_ALPHABET.index("1")
            q_id = CHARACTER_BALANCE_ALPHABET.index("Q")
            (root / "labels" / "train" / "zeros.txt").write_text(f"{zero_id} 0 0 0 0\n" * 10, encoding="utf-8")
            (root / "labels" / "train" / "ones.txt").write_text(f"{one_id} 0 0 0 0\n" * 10, encoding="utf-8")
            (root / "labels" / "train" / "qq1.txt").write_text(
                f"{q_id} 0 0 0 0\n{q_id} 0 0 0 0\n{one_id} 0 0 0 0\n",
                encoding="utf-8",
            )
            (root / "images" / "train" / "qq1.jpg").write_bytes(b"fake")

            plan = plan_character_train_augmentation(root, target_ratio=1.0)

            self.assertEqual(plan.target_count, 10)
            self.assertEqual(plan.deficit_by_symbol["Q"], 8)
            self.assertEqual(plan.planned_images, 4)
            self.assertEqual(plan.predicted_deficit_after.get("Q", 0), 0)
            self.assertEqual(plan.candidates[0].source_key, "qq1")
            self.assertEqual(plan.candidates[0].planned_variants, 4)

    def test_plan_accepts_manual_target_for_single_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_balanced_char_dataset(root, repeats=2)

            plan = plan_character_train_augmentation(
                root,
                target_ratio=1.0,
                target_count_by_symbol={"Q": 5},
            )

            self.assertEqual(plan.target_count, 2)
            self.assertEqual(plan.target_count_by_symbol, {"Q": 5})
            self.assertEqual(plan.deficit_by_symbol, {"Q": 3})
            self.assertEqual(plan.train_sources_by_symbol, {"Q": 1})
            self.assertEqual(plan.planned_images, 2)
            self.assertEqual(plan.predicted_deficit_after, {})
            self.assertEqual([candidate.source_key for candidate in plan.candidates], ["26_Q"])
            self.assertEqual(plan.candidates[0].planned_variants, 2)

    def test_plan_accepts_common_manual_target_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_balanced_char_dataset(root, repeats=2)

            plan = plan_character_train_augmentation(
                root,
                target_ratio=1.0,
                target_count=5,
            )

            self.assertEqual(plan.target_count, 5)
            self.assertEqual(plan.deficit_by_symbol["Q"], 3)
            self.assertEqual(plan.planned_images, 72)
            self.assertEqual(plan.predicted_deficit_after, {})

    def test_plan_counts_incidental_symbols_from_selected_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "images" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            a_id = CHARACTER_BALANCE_ALPHABET.index("A")
            k_id = CHARACTER_BALANCE_ALPHABET.index("K")

            (root / "labels" / "train" / "ak_source.txt").write_text(
                f"{a_id} 0.35 0.5 0.1 0.1\n{k_id} 0.65 0.5 0.1 0.1\n",
                encoding="utf-8",
            )
            (root / "images" / "train" / "ak_source.jpg").write_bytes(b"fake")

            plan = plan_character_train_augmentation(
                root,
                target_count=0,
                extra_count_by_symbol={"A": 3},
                max_augmented_variants_per_source=10,
            )

            self.assertEqual(plan.deficit_by_symbol, {"A": 3})
            self.assertEqual(plan.requested_extra_by_symbol, {"A": 3})
            self.assertEqual(plan.synthetic_count_by_symbol, {"A": 3, "K": 3})
            self.assertEqual(plan.planned_images, 3)
            self.assertEqual(len(plan.candidates), 1)
            self.assertEqual(plan.candidates[0].symbol_counts.get("A"), 1)
            self.assertEqual(plan.candidates[0].symbol_counts.get("K"), 1)
            self.assertEqual(plan.candidates[0].planned_variants, 3)

    def test_preview_counts_incidental_symbols_from_source_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "images" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            a_id = CHARACTER_BALANCE_ALPHABET.index("A")
            k_id = CHARACTER_BALANCE_ALPHABET.index("K")

            (root / "labels" / "train" / "ak_source.txt").write_text(
                f"{a_id} 0.35 0.5 0.1 0.1\n{k_id} 0.65 0.5 0.1 0.1\n",
                encoding="utf-8",
            )
            (root / "images" / "train" / "ak_source.jpg").write_bytes(b"fake")

            candidates, existing = collect_character_train_augmentation_source_pool(
                root,
                max_augmented_variants_per_source=10,
            )
            preview = preview_character_train_synthetic_counts_from_pool(
                {"A": 3},
                candidates,
                existing,
                max_augmented_variants_per_source=10,
            )

            self.assertEqual(preview, {"A": 3, "K": 3})

    def test_flat_pz1_preview_counts_incidental_symbols_before_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels").mkdir(parents=True)
            (root / "images").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            a_id = CHARACTER_BALANCE_ALPHABET.index("A")
            k_id = CHARACTER_BALANCE_ALPHABET.index("K")

            (root / "labels" / "ak_source.txt").write_text(
                f"{a_id} 0.35 0.5 0.1 0.1\n{k_id} 0.65 0.5 0.1 0.1\n",
                encoding="utf-8",
            )
            (root / "images" / "ak_source.jpg").write_bytes(b"fake")

            candidates, existing = collect_character_train_augmentation_source_pool(
                root,
                max_augmented_variants_per_source=10,
            )
            preview = preview_character_train_synthetic_counts_from_pool(
                {"A": 3},
                candidates,
                existing,
                max_augmented_variants_per_source=10,
            )
            plan = plan_character_train_augmentation(
                root,
                target_count=0,
                extra_count_by_symbol={"A": 3},
                max_augmented_variants_per_source=10,
            )

            self.assertEqual(preview, {"A": 3, "K": 3})
            self.assertEqual(plan.synthetic_count_by_symbol, {"A": 3, "K": 3})
            self.assertEqual(plan.planned_images, 3)
            self.assertEqual(plan.train_sources_by_symbol, {"A": 1})

    def test_manual_pz1_preview_can_ignore_existing_augmented_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "images" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            a_id = CHARACTER_BALANCE_ALPHABET.index("A")
            k_id = CHARACTER_BALANCE_ALPHABET.index("K")

            (root / "labels" / "train" / "ak_source.txt").write_text(
                f"{a_id} 0.35 0.5 0.1 0.1\n{k_id} 0.65 0.5 0.1 0.1\n",
                encoding="utf-8",
            )
            (root / "labels" / "train" / "ak_source__aug_001.txt").write_text(
                f"{a_id} 0.35 0.5 0.1 0.1\n{k_id} 0.65 0.5 0.1 0.1\n",
                encoding="utf-8",
            )
            (root / "images" / "train" / "ak_source.jpg").write_bytes(b"fake")
            (root / "augmentation_manifest.json").write_text(
                json.dumps(
                    {
                        "generated_files": [
                            {
                                "source_label": "labels/train/ak_source.txt",
                                "label": "labels/train/ak_source__aug_001.txt",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            blocked = plan_character_train_augmentation(
                root,
                target_count=0,
                extra_count_by_symbol={"A": 1},
                max_augmented_variants_per_source=1,
            )
            self.assertEqual(blocked.planned_images, 0)

            plan = plan_character_train_augmentation(
                root,
                target_count=0,
                extra_count_by_symbol={"A": 1},
                max_augmented_variants_per_source=1,
                respect_existing_augmented_variants=False,
            )
            candidates, existing = collect_character_train_augmentation_source_pool(
                root,
                max_augmented_variants_per_source=1,
                respect_existing_augmented_variants=False,
            )
            preview = preview_character_train_synthetic_counts_from_pool(
                {"A": 1},
                candidates,
                existing,
                max_augmented_variants_per_source=1,
                respect_existing_augmented_variants=False,
            )

            self.assertEqual(plan.planned_images, 1)
            self.assertEqual(plan.synthetic_count_by_symbol, {"A": 1, "K": 1})
            self.assertEqual(preview, {"A": 1, "K": 1})

    def test_plan_reports_not_feasible_when_deficit_has_no_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            zero_id = CHARACTER_BALANCE_ALPHABET.index("0")
            one_id = CHARACTER_BALANCE_ALPHABET.index("1")
            (root / "labels" / "train" / "zeros.txt").write_text(f"{zero_id} 0 0 0 0\n" * 10, encoding="utf-8")
            (root / "labels" / "train" / "ones.txt").write_text(f"{one_id} 0 0 0 0\n" * 10, encoding="utf-8")

            plan = plan_character_train_augmentation(root, target_ratio=1.0)

            self.assertFalse(plan.feasible)
            self.assertEqual(plan.completion_status, "PLAN_NOT_FEASIBLE")
            self.assertEqual(plan.planned_images, 0)
            self.assertGreater(plan.predicted_deficit_after.get("Q", 0), 0)

    def test_plan_keeps_available_synthetics_when_another_symbol_has_no_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "images" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            a_id = CHARACTER_BALANCE_ALPHABET.index("A")

            (root / "labels" / "train" / "a_source.txt").write_text(
                f"{a_id} 0.5 0.5 0.1 0.1\n",
                encoding="utf-8",
            )
            (root / "images" / "train" / "a_source.jpg").write_bytes(b"fake")

            plan = plan_character_train_augmentation(
                root,
                target_count=0,
                extra_count_by_symbol={"A": 4, "K": 5},
                max_augmented_variants_per_source=10,
            )

            self.assertFalse(plan.feasible)
            self.assertEqual(plan.deficit_by_symbol["A"], 4)
            self.assertEqual(plan.deficit_by_symbol["K"], 5)
            self.assertEqual(plan.requested_extra_by_symbol, {"A": 4, "K": 5})
            self.assertEqual(plan.synthetic_count_by_symbol, {"A": 4})
            self.assertEqual(plan.train_sources_by_symbol, {"A": 1, "K": 0})
            self.assertEqual(plan.planned_images, 4)
            self.assertEqual(plan.predicted_deficit_after, {"K": 5})

    def test_manual_augmentation_state_has_no_balance_limit_without_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, limits, _initial, stats = _build_balance_augmented_source_state(
                root,
                [(root / "images" / "train" / "a.jpg", root / "labels" / "train" / "a.txt")],
                balance_plan=None,
            )

            self.assertEqual(len(records), 1)
            self.assertFalse(stats["balance_plan_enabled"])
            self.assertIsNone(limits["a"])
            self.assertIsNone(stats["max_augmented_variants_per_source"])

    def test_executor_balance_state_respects_max_variants_per_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "images" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            for stem in ("plate_001", "plate_001__aug_0001", "plate_001__aug_0002"):
                label = root / "labels" / "train" / f"{stem}.txt"
                image = root / "images" / "train" / f"{stem}.jpg"
                label.write_text("16 0 0 0 0\n", encoding="utf-8")
                image.write_bytes(b"fake")
            (root / "augmentation_manifest.json").write_text(
                json.dumps(
                    {
                        "generated_files": [
                            {"label": "labels/train/plate_001__aug_0001.txt", "source_label": "labels/train/plate_001.txt"},
                            {"label": "labels/train/plate_001__aug_0002.txt", "source_label": "labels/train/plate_001.txt"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            plan = {
                "base_dataset": str(root),
                "max_augmented_variants_per_source": 3,
                "candidates": [
                    {
                        "source_key": "plate_001",
                        "label_path": str(root / "labels" / "train" / "plate_001.txt"),
                        "priority": 10,
                        "max_augmented_variants": 3,
                    }
                ],
            }

            records, limits, initial, stats = _build_balance_augmented_source_state(root, [
                (root / "images" / "train" / "plate_001.jpg", root / "labels" / "train" / "plate_001.txt"),
                (root / "images" / "train" / "plate_001__aug_0001.jpg", root / "labels" / "train" / "plate_001__aug_0001.txt"),
                (root / "images" / "train" / "plate_001__aug_0002.jpg", root / "labels" / "train" / "plate_001__aug_0002.txt"),
            ], balance_plan=plan)

            self.assertEqual(len(records), 1)
            self.assertEqual(limits["plate_001"], 3)
            self.assertEqual(initial["plate_001"], 2)
            self.assertEqual(stats["max_augmented_variants_per_source"], 3)

    def test_val_test_fingerprint_detects_untouched_and_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "val").mkdir(parents=True)
            (root / "images" / "val").mkdir(parents=True)
            (root / "labels" / "test").mkdir(parents=True)
            (root / "images" / "test").mkdir(parents=True)
            (root / "labels" / "val" / "a.txt").write_text("0 0 0 0 0\n", encoding="utf-8")
            (root / "images" / "val" / "a.jpg").write_bytes(b"a")

            before = build_character_dataset_file_fingerprint(root, splits=("val", "test"))
            after = build_character_dataset_file_fingerprint(root, splits=("val", "test"))
            self.assertTrue(compare_character_val_test_unchanged(before, after)["unchanged"])

            (root / "labels" / "test" / "b.txt").write_text("1 0 0 0 0\n", encoding="utf-8")
            changed = build_character_dataset_file_fingerprint(root, splits=("val", "test"))
            self.assertFalse(compare_character_val_test_unchanged(before, changed)["unchanged"])

    def test_val_test_fingerprint_excludes_data_yaml_but_full_dataset_includes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "val").mkdir(parents=True)
            (root / "images" / "val").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            (root / "labels" / "val" / "a.txt").write_text("0 0 0 0 0\n", encoding="utf-8")
            (root / "images" / "val" / "a.jpg").write_bytes(b"image")

            val_before = build_character_dataset_file_fingerprint(root, splits=("val", "test"))
            full_before = build_character_dataset_file_fingerprint(root)
            (root / "data.yaml").write_text("path: .\nnames:\n  0: changed\n", encoding="utf-8")
            val_after = build_character_dataset_file_fingerprint(root, splits=("val", "test"))
            full_after = build_character_dataset_file_fingerprint(root)

            self.assertFalse(val_before["includes_config"])
            self.assertTrue(full_before["includes_config"])
            self.assertTrue(compare_character_val_test_unchanged(val_before, val_after)["unchanged"])
            self.assertNotEqual(full_before["sha256"], full_after["sha256"])

    def test_val_test_fingerprint_supports_split_first_yolo_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "val" / "labels").mkdir(parents=True)
            (root / "val" / "images").mkdir(parents=True)
            (root / "val" / "labels" / "a.txt").write_text("0 0 0 0 0\n", encoding="utf-8")
            (root / "val" / "images" / "a.jpg").write_bytes(b"image")

            fingerprint = build_character_dataset_file_fingerprint(root, splits=("val", "test"))

            self.assertEqual(fingerprint["mode"], "content_sha256")
            self.assertEqual(fingerprint["file_count"], 2)
            self.assertTrue(any(entry["relative_path"] == "val/labels/a.txt" for entry in fingerprint["entries"]))

    def test_val_test_fingerprint_detects_same_size_same_mtime_content_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label_dir = root / "labels" / "val"
            label_dir.mkdir(parents=True)
            label_path = label_dir / "a.txt"
            label_path.write_text("0 0 0 0 0\n", encoding="utf-8")
            stamp = 1_700_000_000_000_000_000
            os.utime(label_path, ns=(stamp, stamp))

            before = build_character_dataset_file_fingerprint(root, splits=("val", "test"))
            label_path.write_text("1 0 0 0 0\n", encoding="utf-8")
            os.utime(label_path, ns=(stamp, stamp))
            after = build_character_dataset_file_fingerprint(root, splits=("val", "test"))

            self.assertEqual(before["entries"][0]["size"], after["entries"][0]["size"])
            self.assertNotEqual(before["sha256"], after["sha256"])

    def test_executor_empty_balance_plan_does_not_fall_back_to_random_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "base_dataset": str(root),
                "max_augmented_variants_per_source": 3,
                "candidates": [],
            }

            records, _limits, _initial, stats = _build_balance_augmented_source_state(
                root,
                [(root / "images" / "train" / "a.jpg", root / "labels" / "train" / "a.txt")],
                balance_plan=plan,
            )

            self.assertEqual(records, [])
            self.assertTrue(stats["balance_plan_enabled"])
            self.assertEqual(stats["balance_plan_candidates"], 0)
            self.assertEqual(stats["balance_pool"], 0)

    def test_balance_sample_size_preserves_target_priorities(self):
        pool = [
            {"source_key": "low", "label_key": "labels/train/low.txt", "priority": 1},
            {"source_key": "high", "label_key": "labels/train/high.txt", "priority": 100},
            {"source_key": "mid", "label_key": "labels/train/mid.txt", "priority": 10},
        ]

        selected = _select_augmentation_sample_pool(
            random.Random(42),
            list(pool),
            2,
            balance_plan_enabled=True,
        )

        self.assertEqual([row["source_key"] for row in selected], ["high", "mid"])

    def test_partial_train_augmentation_is_not_success(self):
        ok, message = _finalize_train_augmentation_result(
            {
                "generated": 357,
                "skipped": 4,
                "stop_reason": "wyczerpano limit kandydatów",
            },
            1000,
        )

        self.assertFalse(ok)
        self.assertIn("357 z 1000", message)
        self.assertIn("nie zostanie oznaczony jako gotowy", message)

    def test_real_source_search_counts_unused_sources_for_deficit_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            pool = Path(tmp) / "pool"
            (root / "labels" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            (root / "labels" / "train" / "many.txt").write_text("0 0 0 0 0\n" * 8, encoding="utf-8")
            (root / "labels" / "train" / "used_q.txt").write_text("26 0 0 0 0\n26 0 0 0 0\n", encoding="utf-8")
            pool.mkdir()
            records = [{"source_key": "used_q", "source_expected_text": "Q0"}]
            records.extend({"source_key": f"unused_q_{index}", "source_expected_text": f"AQ{index}"} for index in range(5))
            (pool / "metadata.json").write_text(json.dumps({"plates": records}), encoding="utf-8")

            result = find_character_real_source_candidates(root, (pool,), target_ratio=1.0)

            self.assertEqual(result["Q"]["current_unique_train"], 1)
            self.assertEqual(result["Q"]["available_unused_real_sources"], 5)

    def test_real_source_search_includes_low_diversity_without_deficit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            pool = Path(tmp) / "pool"
            (root / "labels" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            for class_id in range(10):
                for index in range(5):
                    (root / "labels" / "train" / f"c{class_id}_{index}.txt").write_text(
                        f"{class_id} 0 0 0 0\n",
                        encoding="utf-8",
                    )
            (root / "labels" / "train" / "used_q.txt").write_text("26 0 0 0 0\n" * 6, encoding="utf-8")
            pool.mkdir()
            (pool / "metadata.json").write_text(
                json.dumps({"plates": [{"source_key": "unused_q", "source_expected_text": "AQ1"}]}),
                encoding="utf-8",
            )

            dist = analyze_character_class_distribution(root, target_ratio=0.5)
            q_row = {row.symbol: row for row in dist.classes}["Q"]
            result = find_character_real_source_candidates(root, (pool,), target_ratio=0.5)

            self.assertEqual(q_row.deficit_count, 0)
            self.assertEqual(q_row.diversity_status, "LOW_DIVERSITY")
            self.assertIn("Q", result)
            self.assertEqual(result["Q"]["available_unused_real_sources"], 1)

    def test_swapped_class_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            names = list(CHARACTER_BALANCE_ALPHABET)
            names[10], names[11] = names[11], names[10]
            yaml_text = "path: .\nnames:\n" + "".join(f"  - \"{symbol}\"\n" for symbol in names)
            (root / "data.yaml").write_text(yaml_text, encoding="utf-8")

            with self.assertRaises(CharacterClassMapValidationError) as ctx:
                analyze_character_class_distribution(root)

            self.assertEqual(ctx.exception.validation.first_mismatch_index, 10)

    def test_missing_class_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            names = list(CHARACTER_BALANCE_ALPHABET[:-1])
            yaml_text = "path: .\nnames:\n" + "".join(f"  - \"{symbol}\"\n" for symbol in names)
            (root / "data.yaml").write_text(yaml_text, encoding="utf-8")

            with self.assertRaises(CharacterClassMapValidationError) as ctx:
                analyze_character_class_distribution(root)

            self.assertEqual(ctx.exception.validation.first_mismatch_index, 35)

    def test_extra_class_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)
            names = list(CHARACTER_BALANCE_ALPHABET) + ["EXTRA"]
            yaml_text = "path: .\nnames:\n" + "".join(f"  - \"{symbol}\"\n" for symbol in names)
            (root / "data.yaml").write_text(yaml_text, encoding="utf-8")

            with self.assertRaises(CharacterClassMapValidationError) as ctx:
                analyze_character_class_distribution(root)

            self.assertEqual(ctx.exception.validation.first_mismatch_index, 36)

    def test_missing_data_yaml_is_allowed_only_for_flat_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels").mkdir()
            (root / "labels" / "flat.txt").write_text("0 0 0 0 0\n", encoding="utf-8")

            dist = analyze_character_class_distribution(root)

            self.assertEqual(dist.layout, "flat")
            self.assertEqual(dist.summary["class_map"]["status"], "WARNING")

    def test_missing_data_yaml_is_rejected_for_split_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels" / "train").mkdir(parents=True)

            with self.assertRaises(CharacterClassMapValidationError):
                analyze_character_class_distribution(root)

    def test_plan_is_calculated_for_exact_split_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            output = Path(tmp) / "split_output"
            _write_split_source_dataset(source)

            ok, message, _stats = DatasetSplitter(random_seed=7).split_dataset(
                source,
                output,
                {"train": 0.80, "val": 0.10, "test": 0.10},
            )

            self.assertTrue(ok, message)
            plan = plan_character_train_augmentation(output)
            self.assertEqual(Path(plan.base_dataset).resolve(), output.resolve())
            self.assertTrue(z4_dataset_builder._step4_balance_plan_matches_dataset(plan, output))
            self.assertTrue(z4_dataset_builder._step4_pending_plan_fingerprint_matches(plan, output))

    def test_auto_representation_fails_without_plan_instead_of_random_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            _write_balanced_char_dataset(root)
            host = _Step4BuilderHost()

            result = z4_dataset_builder._create_step4_augmented_dataset_variant(
                host,
                source_dataset_path=root,
                target="char",
                profile=AugmentationProfile(enabled=True, extra_count=1),
                progress_var_name="split_progress_var",
                status_attr_name="split_status",
                augmentation_mode="mz_auto_representation",
            )

            self.assertFalse(result["ok"])
            self.assertIn("PLAN_MISSING", result["message"])
            self.assertEqual(len(list(root.parent.glob("*Aug*"))), 0)

    def test_auto_representation_rejects_plan_from_different_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_a = Path(tmp) / "source_a"
            source_b = Path(tmp) / "source_b"
            _write_balanced_char_dataset(source_a)
            _write_balanced_char_dataset(source_b)
            plan = plan_character_train_augmentation(source_a)
            host = _Step4BuilderHost()

            result = z4_dataset_builder._create_step4_augmented_dataset_variant(
                host,
                source_dataset_path=source_b,
                target="char",
                profile=AugmentationProfile(enabled=True, extra_count=1),
                progress_var_name="split_progress_var",
                status_attr_name="split_status",
                balance_plan_override=plan,
                augmentation_mode="mz_auto_representation",
            )

            self.assertFalse(result["ok"])
            self.assertIn("PLAN_DATASET_MISMATCH", result["message"])

    def test_auto_representation_zero_plan_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "balanced"
            _write_balanced_char_dataset(root, repeats=3)
            plan = plan_character_train_augmentation(root)
            host = _Step4BuilderHost()

            self.assertTrue(plan.feasible)
            self.assertEqual(plan.planned_images, 0)
            result = z4_dataset_builder._create_step4_augmented_dataset_variant(
                host,
                source_dataset_path=root,
                target="char",
                profile=AugmentationProfile(enabled=False, extra_count=0),
                progress_var_name="split_progress_var",
                status_attr_name="split_status",
                balance_plan_override=plan,
                augmentation_mode="mz_auto_representation",
            )

            manifest_path = root / "mz_training_variant_manifest.json"
            self.assertTrue(result["ok"], result["message"])
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["augmentation_mode"], "mz_auto_representation")
            self.assertEqual(manifest["completion_status"], "REPRESENTATION_OK")
            self.assertEqual(manifest["planned_images"], 0)
            self.assertEqual(manifest["generated_images"], 0)
            self.assertEqual(manifest["execution_status"], "COMPLETED")
            self.assertEqual(manifest["representation_status"], "REPRESENTATION_OK")
            self.assertEqual(manifest["freeze_status"], "UNCHANGED")
            self.assertTrue(manifest["ready_for_training"])

    def test_failed_mz_variant_is_not_ready_for_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "not_feasible"
            (root / "labels" / "train").mkdir(parents=True)
            (root / "images" / "train").mkdir(parents=True)
            (root / "data.yaml").write_text(_names_list_yaml(), encoding="utf-8")
            zero_id = CHARACTER_BALANCE_ALPHABET.index("0")
            one_id = CHARACTER_BALANCE_ALPHABET.index("1")
            (root / "labels" / "train" / "heavy_zero.txt").write_text(
                f"{zero_id} 0.5 0.5 0.2 0.2\n" * 12,
                encoding="utf-8",
            )
            (root / "labels" / "train" / "heavy_one.txt").write_text(
                f"{one_id} 0.5 0.5 0.2 0.2\n" * 12,
                encoding="utf-8",
            )
            plan = plan_character_train_augmentation(root, target_ratio=1.0)
            host = _Step4BuilderHost()

            result = z4_dataset_builder._create_step4_augmented_dataset_variant(
                host,
                source_dataset_path=root,
                target="char",
                profile=AugmentationProfile(enabled=False, extra_count=0),
                progress_var_name="split_progress_var",
                status_attr_name="split_status",
                balance_plan_override=plan,
                augmentation_mode="mz_auto_representation",
            )

            self.assertFalse(result["ok"])
            manifest = json.loads((root / "mz_training_variant_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["completion_status"], "PLAN_NOT_FEASIBLE")
            self.assertEqual(manifest["execution_status"], "NOT_RUN")
            self.assertEqual(manifest["representation_status"], "TARGET_NOT_REACHED")
            self.assertFalse(manifest["ready_for_training"])
            readiness = get_training_dataset_readiness(root, target="char")
            self.assertFalse(readiness["ok"])

    def test_legacy_manual_char_manifest_remains_selectable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "manual_augmented"
            root.mkdir()
            (root / "mz_training_variant_manifest.json").write_text(
                json.dumps(
                    {
                        "augmentation_mode": "manual_train_augmentation",
                        "completion_status": "COMPLETED",
                        "deficit_after": {"Q": 2},
                    }
                ),
                encoding="utf-8",
            )

            readiness = get_training_dataset_readiness(root, target="char")

            self.assertTrue(readiness["ok"])

    def test_failure_result_can_block_pz2_transition(self):
        host = _Step4BuilderHost()

        z4_dataset_builder._handle_step4_dataset_failure_result(
            host,
            message="PLAN_NOT_FEASIBLE",
            target="char",
            critical=False,
            allow_pz2_override=False,
        )

        self.assertFalse(host.captured_result_modals[-1]["allow_pz2"])
        self.assertFalse(host.opened_pz2)

    def test_full_split_plan_augment_marks_mz_variant_ready(self):
        if not CV2_AVAILABLE or cv2 is None or np is None or not is_albumentations_available():
            self.skipTest("OpenCV/Albumentations unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            split_out = Path(tmp) / "split"
            _write_char_split_source_with_one_q_deficit(source)
            ok, msg, _stats = DatasetSplitter(random_seed=7).split_dataset(
                source,
                split_out,
                {"train": 1.0, "val": 0.0, "test": 0.0},
            )
            self.assertTrue(ok, msg)
            plan = plan_character_train_augmentation(split_out)
            self.assertTrue(plan.feasible)
            self.assertGreater(plan.planned_images, 0)
            host = _Step4BuilderHost()

            result = z4_dataset_builder._create_step4_augmented_dataset_variant(
                host,
                source_dataset_path=split_out,
                target="char",
                profile=AugmentationProfile(enabled=True, extra_count=999, sample_size=1, seed=123),
                progress_var_name="split_progress_var",
                status_attr_name="split_status",
                balance_plan_override=plan,
                augmentation_mode="mz_auto_representation",
            )

            self.assertTrue(result["ok"], result["message"])
            manifest = dict(result["mz_training_variant_manifest"])
            self.assertEqual(manifest["completion_status"], "REPRESENTATION_OK")
            self.assertEqual(manifest["execution_status"], "COMPLETED")
            self.assertEqual(manifest["representation_status"], "REPRESENTATION_OK")
            self.assertEqual(manifest["freeze_status"], "UNCHANGED")
            self.assertTrue(manifest["ready_for_training"])
            self.assertEqual(manifest["generated_images"], plan.planned_images)
            self.assertTrue(result["augmentation_stats"]["balance_plan_enabled"])
            self.assertEqual(result["generated"], plan.planned_images)
            self.assertEqual(get_training_dataset_readiness(result["dataset_path"], target="char")["ok"], True)

    def test_plan_invalidation_clears_source_state_and_ui_flags(self):
        host = _Step4BuilderHost()
        host._pending_character_balance_plan = object()
        host._pending_character_balance_real_sources = {"Q": {}}
        host._pending_character_balance_ratio_key = (80.0, 10.0, 10.0)

        z4_dataset_builder._invalidate_pending_character_balance_plan(host, "Źródło zmieniło się.")

        self.assertIsNone(host._pending_character_balance_plan)
        self.assertIsNone(host._pending_character_balance_real_sources)
        self.assertIsNone(host._pending_character_balance_ratio_key)
        self.assertFalse(host.split_aug_enabled_var.get())
        self.assertEqual(host.split_aug_extra_var.get(), 0)
        self.assertIn("Źródło", host.split_mz_representation_status_var.get())

    def test_ratio_change_invalidates_pending_character_plan(self):
        host = _Step4BuilderHost()
        host.train_pct.set(70.0)
        host.val_pct.set(20.0)
        host._pending_character_balance_plan = object()
        host._pending_character_balance_real_sources = {"Q": {}}
        host._pending_character_balance_ratio_key = (80.0, 10.0, 10.0)

        z4_dataset_builder._update_ratio_labels(host)

        self.assertIsNone(host._pending_character_balance_plan)
        self.assertFalse(host.split_aug_enabled_var.get())
        self.assertIn("Proporcje", host.split_mz_representation_status_var.get())


if __name__ == "__main__":
    unittest.main()
