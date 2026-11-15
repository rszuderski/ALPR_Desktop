from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def test_ready_plate_picker_uses_application_dataset_browser():
    source = read("auto_annotation_tool/gui/z4_dataset_panels.py")
    assert 'text="Przeglądaj datasety"' in source
    assert "z4_dataset_builder._open_ready_plate_dataset_browser(self)" in source
    assert 'state="readonly"' in source


def test_browser_lists_pose_datasets_and_history_identity():
    source = read("auto_annotation_tool/gui/z4_dataset_builder.py")
    assert "def _collect_ready_plate_dataset_browser_rows" in source
    assert '"kpt_shape" not in cfg' in source
    assert 'CONFIG.get_training_runs_dir("plate")' in source
    assert 'payload.get("dataset_id")' in source
    # Sprawdzamy treść etykiety, a nie konkretny rodzaj cudzysłowu w źródle.
    assert "ID historyczne:" in source


def test_browser_exposes_split_counts_preview_and_selection():
    source = read("auto_annotation_tool/gui/z4_dataset_builder.py")
    assert '"Train"' in source
    assert '"Val"' in source
    assert '"Test"' in source
    assert 'open_yolo_dataset_preview(self, row.get("path"))' in source
    assert "self.creator_ready_dataset_var.set(path)" in source


def test_browser_has_filter_and_no_raw_folder_picker_as_primary_action():
    source = read("auto_annotation_tool/gui/z4_dataset_builder.py")
    assert 'text="Filtr:"' in source
    assert 'text="Wybierz dataset"' in source
    panel = read("auto_annotation_tool/gui/z4_dataset_panels.py")
    start = panel.index("self.btn_pick_creator_ready_dataset = ttk.Button")
    end = panel.index("self.btn_pick_creator_ready_dataset.pack", start)
    chunk = panel[start:end]
    assert "_pick_dir(" not in chunk
