from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_annotation_tool.gui import z3_pz2_material_export as export_ui
from auto_annotation_tool.gui.z3_preview_list_ui import refresh_preview_import_focus_ui
from auto_annotation_tool.registry.az_registry import AZRegistry, project_id_from_folder_name

ROOT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def test_pz2_has_mirrored_add_and_export_buttons():
    source = _read("auto_annotation_tool/gui/z3_detection_tab_ui.py")
    assert 'text="Dodaj tablice lub zdjęcia…"' in source
    assert 'text="Eksportuj wycięcia tablic…"' in source
    assert 'text="Eksportuj cropy + AZ…"' not in source
    assert 'text="Eksportuj cropy + AZ…"' not in source
    assert "open_pz2_add_material" in source
    assert "open_pz2_export_material" in source


def test_export_ui_calls_same_portable_package_backend():
    source = _read("auto_annotation_tool/gui/z3_pz2_material_export.py")
    assert "export_pz2_az_package(" in source
    assert "Dodaj tablice lub zdjęcia…" in source
    assert "_flush_preview_metadata(host)" in source
    assert "Aktywny PZ2 nie został zmodyfikowany." in source


def test_export_ui_has_clear_scopes():
    source = _read("auto_annotation_tool/gui/z3_pz2_material_export.py")
    assert "Wszystkie tablice" in source
    assert "Tylko zatwierdzone [OK]" in source
    assert "Zaznaczone tablice" in source
    assert "Ostatnio dodany materiał" in source


def test_export_button_is_kept_in_project_material_row():
    source = _read("auto_annotation_tool/gui/z3_preview_list_ui.py")
    assert "preview_add_material_btn" in source
    assert "preview_export_material_btn" in source
    assert "project_append_available" in source


def _host(tmp_path):
    preview = tmp_path / "preview_run"
    (preview / "images").mkdir(parents=True)
    (preview / "metadata.json").write_text('{}', encoding="utf-8")
    AZRegistry.for_workspace(tmp_path).initialize()
    return SimpleNamespace(
        preview_dir_var=SimpleNamespace(get=lambda: str(preview)),
        _loaded_meta_path=preview / "metadata.json",
        preview_metadata={"plate": {"crop_id": "CROP-1"}},
        app=SimpleNamespace(campaign_free_mode=True),
        _step3_linear_mode=False,
        _flush_scheduled_preview_metadata_save=Mock(),
    )


def test_free_context_and_export_do_not_create_project_or_flush_metadata(tmp_path):
    host = _host(tmp_path)
    result = SimpleNamespace(exported=1, with_bound_revision=0, package_id="PKG", output_dir="out")
    with patch.object(export_ui.CONFIG, "WORKSPACE_DIR", str(tmp_path)), \
         patch.object(export_ui.CAMPAIGN, "get_active_project_name", return_value=""), \
         patch.object(export_ui, "_prompt_export_scope", return_value="all"), \
         patch.object(export_ui.filedialog, "askdirectory", return_value=str(tmp_path)) as choose, \
         patch.object(export_ui, "export_pz2_az_package", return_value=result) as backend, \
         patch.object(export_ui, "_show_info"), \
         patch.object(export_ui, "_show_error") as error:
        assert export_ui.open_pz2_export_material(host)
    error.assert_not_called()
    host._flush_scheduled_preview_metadata_save.assert_not_called()
    args = backend.call_args.kwargs
    assert args["project_id"] is None
    assert args["iteration_num"] is None
    assert args["metadata"] is host.preview_metadata
    assert args["output_dir"].name.startswith("AZ_FREE_preview_run_all_")
    assert choose.call_args.kwargs["initialdir"] == str(tmp_path)
    with AZRegistry.for_workspace(tmp_path).database.read_connection() as con:
        assert con.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0


def test_free_export_rejects_stale_preview_before_dialog(tmp_path):
    host = _host(tmp_path)
    host._loaded_meta_path = tmp_path / "other" / "metadata.json"
    with patch.object(export_ui, "_show_error") as error, \
         patch.object(export_ui, "_prompt_export_scope") as prompt:
        assert not export_ui.open_pz2_export_material(host)
    error.assert_called_once()
    prompt.assert_not_called()


def test_project_context_resolves_existing_id_without_registry_write(tmp_path):
    host = _host(tmp_path)
    host.app.campaign_free_mode = False
    host._step3_linear_mode = True
    project_root = tmp_path / "Project_F00"
    registry = AZRegistry.for_workspace(tmp_path)
    with registry.database.read_connection() as con:
        before = tuple(con.iterdump())
    with patch.object(export_ui.CONFIG, "WORKSPACE_DIR", str(tmp_path)), \
         patch.object(export_ui.CAMPAIGN, "get_active_project_name", return_value="Project"), \
         patch.object(export_ui.CAMPAIGN, "get_active_project_root_dir", return_value=project_root), \
         patch.object(export_ui.CAMPAIGN, "get_current_iteration_num", return_value=2):
        context = export_ui._resolve_export_context(host)
    assert context["project_id"] == project_id_from_folder_name(project_root.name)
    assert context["iteration_num"] == 2
    assert context["output_initial_dir"] == project_root
    with registry.database.read_connection() as con:
        assert tuple(con.iterdump()) == before


def test_export_controls_visible_in_free_mode_without_project_append():
    add = Mock()
    export = Mock()
    frame = Mock()
    frame.winfo_manager.return_value = ""
    host = SimpleNamespace(preview_add_material_btn=add, preview_export_material_btn=export,
                           preview_import_focus_frame=frame, _step3_linear_mode=False,
                           preview_metadata={"plate": {"crop_id": "CROP-1"}})
    refresh_preview_import_focus_ui(host)
    add.grid_remove.assert_called_once()
    export.grid.assert_called_once()
    frame.grid.assert_called_once()
    export.grid_remove.assert_not_called()


def test_empty_free_preview_hides_export_controls():
    export = Mock()
    host = SimpleNamespace(preview_export_material_btn=export, preview_metadata={})
    refresh_preview_import_focus_ui(host)
    export.grid_remove.assert_called_once()
    export.grid.assert_not_called()


def test_project_output_name_keeps_iteration(tmp_path):
    path = export_ui._build_unique_output_dir(tmp_path, project_name="Project", iteration_num=2, scope="all")
    assert path.name.startswith("AZ_Project_IT002_all_")
