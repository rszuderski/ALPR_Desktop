from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def test_free_plate_pz1_exposes_ready_split_choice():
    source = read("auto_annotation_tool/gui/z4_shared_ui.py")
    assert 'ready_mode = host._get_creator_source_mode() == "ready"' in source
    assert "_set_pack_visible(ready_row" in source
    assert "_set_pack_visible(source_mode_frame" in source

    start = source.index(
        'else:\n                _set_pack_visible(getattr(host, "creator_campaign_summary_frame", None), False)'
    )
    end = source.index('if bool(vm.in_campaign) and vm.mode == "char"', start)
    free_plate_chunk = source[start:end]
    assert 'host._set_creator_source_mode("xml")' not in free_plate_chunk


def test_plate_panel_has_explicit_ready_dataset_picker():
    source = read("auto_annotation_tool/gui/z4_dataset_panels.py")
    assert "creator_ready_dataset_var" in source
    assert "Gotowy split tablic:" in source
    assert "btn_pick_creator_ready_dataset" in source


def test_ready_plate_source_requires_pose_and_frozen_splits():
    source = read("auto_annotation_tool/gui/z4_dataset_builder.py")
    assert "def _resolve_ready_plate_dataset_source" in source
    assert '"kpt_shape" not in cfg' in source
    assert "train_count <= 0 or val_count <= 0 or test_count <= 0" in source


def test_ready_plate_build_reuses_existing_split_and_augments_only_child():
    source = read("auto_annotation_tool/gui/z4_dataset_builder.py")
    assert "def _create_ready_plate_dataset_variant_thread" in source
    assert "source_dataset_path=source_dir" in source
    assert 'target="plate"' in source
    assert "Val i test pozostają oryginalne" in source


def test_augmentation_preview_uses_ready_train_only():
    source = read("auto_annotation_tool/gui/z4_dataset_builder.py")
    assert 'train_root = ready_root / "images" / "train"' in source
    assert "return ready_images[:80]" in source
    assert "Dla gotowego splitu source_count jest już licznością train." in source
