from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def test_pz2_exposes_one_append_entry_above_plate_list():
    source = _read("auto_annotation_tool/gui/z3_detection_tab_ui.py")
    assert 'text="Dodaj tablice lub zdjęcia…"' in source
    assert "open_pz2_add_material" in source
    assert "preview_add_material_btn" in source
    assert "preview_import_focus_frame" in source


def test_pz2_append_ui_uses_same_active_preview_not_replacement():
    source = _read("auto_annotation_tool/gui/z3_pz2_material_append.py")
    assert "append_az_package(" in source
    assert "materialize_az_append_to_preview(" in source
    assert "target_artifact_dir=images_dir" in source
    assert "preview_dir=preview_dir" in source
    assert "recalculate_statuses=False" in source
    assert "_flush_scheduled_preview_metadata_save" in source
    # AZ009G1 ustawia focus importu bezpośrednio przed jedyną
    # przebudową listy; nie wolno wracać do _set_preview_import_focus(),
    # bo ta ścieżka wykonywała drugą pełną przebudowę UI.
    assert "_preview_import_focus_plate_ids" in source
    assert "_preview_import_focus_batch_id" in source
    assert "_preview_import_focus_active = True" in source
    assert "preserve_selection=False" in source
    assert "Rozszerzono TEN SAM aktywny zbiór PZ2." in source


def test_pz2_append_ui_has_recovery_after_registry_commit():
    source = _read("auto_annotation_tool/gui/z3_pz2_material_append.py")
    assert "append_result=None" in source
    assert "Dokończ append materiału" in source
    assert "poprzedni append wymaga dokończenia" in source


def test_import_focus_copy_is_transport_neutral_not_cvat_only():
    source = _read("auto_annotation_tool/gui/z3_preview_list_ui.py")
    assert "ostatniego importu CVAT" not in source
    assert "Ostatni import CVAT" not in source
    assert "ostatnio dodanego materiału" in source
