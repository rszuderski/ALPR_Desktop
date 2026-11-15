"""Static UX contract regression: selection center versus export modal."""
from pathlib import Path


def test_mobile_export_ui_owns_preflight_and_calibration():
    path = Path(__file__).resolve().parents[1] / "auto_annotation_tool/gui/z4_model_export.py"
    source = path.read_text(encoding="utf-8-sig")
    modal = source.split("    def open_selected_mobile_export_executor() -> None:", 1)[1].split(
        "\n    def export_package() -> None:", 1
    )[0]
    footer = source.split('    export_button = ttk.Button(\n        actions,\n        text="Eksportuj zaznaczone",', 1)[1]
    footer = footer.split("    # One action row at every supported width/DPI", 1)[0]
    assert "Kalibracja INT8 — wskazany model" not in footer
    assert "Wybierz trening ręcznie" not in footer
    assert "Sprawdź precyzję TFLite" not in footer
    assert "Raporty z telefonu" not in footer
    assert "Sprawdź precyzję TFLite" in modal
    # Z4LIB029_FIX01: sprawdzamy rzeczywiste kontrolki i ich działania,
    # nie historyczne napisy, których nie było już w bazie repozytorium.
    assert 'ttk.Button(calibration, text="Znajdź zgodny manifest", command=lambda: _search_manifest())' in modal
    assert 'ttk.Button(calibration, text="Przygotuj nowy manifest", command=prepare_slot_calibration)' in modal
    assert 'ttk.Button(calibration, text="Wybierz plik ręcznie", command=choose_model_calibration)' in modal
    assert 'def _search_manifest(*, manual_path: str = "", current_state=state) -> None:' in modal
    assert "def prepare_slot_calibration(current_state=state):" in modal
    assert "return _open_candidate_int8_calibration(" in modal
    assert "def choose_model_calibration(current_state=state) -> None:" in modal
    assert 'current_state["calibration"].set(selected)' in modal
    assert 'state["calibration"].trace_add("write", _on_manifest_path_change)' in modal
    assert "_sync_calibration_visibility" in modal
    assert 'if bool(current_state["int8"].get()) or bool(current_state["onnx_int8"].get()):' in modal
    # Z4LIB029_FIX02: eksport ALPR idzie przez przypięcie SHA i ochronę F4,
    # a nie przez historyczne bezpośrednie package_exporter.export(request).
    assert 'if not run_executor_preflight(show_dialog=False):' in modal
    assert 'require_f4_no_conversion_request(request, require_fresh_destination=True)' in modal
    assert 'def export_with_pinned_mz(active_request):' in modal
    assert 'return package_exporter.export(active_request, progress=progress_cb)' in modal
    assert 'replace(active_request, character_package=snapshot), progress=progress_cb,' in modal
    assert 'def export_without_f4_overwrite(active_request):' in modal
    assert 'return export_f4_atomic_no_overwrite(' in modal
    assert 'validate_final=lambda candidate: package_exporter.validate_package(' in modal
    assert 'package_path = export_without_f4_overwrite(pinned_request)' in modal
    assert 'package_path = export_without_f4_overwrite(request)' in modal
    assert 'package_exporter.export(request, progress=progress_cb)' not in modal
    assert 'last_exported_package["path"] = str(package_path)' in modal
    assert 'open_mobile_report_browser(self, parent=executor_modal)' in modal


def test_int8_export_guards_remain_in_place():
    exporter = Path(__file__).resolve().parents[1] / "auto_annotation_tool/exporters/mobile_model_exporter.py"
    code = exporter.read_text(encoding="utf-8-sig")
    assert 'def _calibration_for_request(' in code
    assert 'FULL_INT8_OUTPUT_NOT_PROVEN' in code
    assert 'def _requested_variant_specs(' in code
    assert 'self.validate_package(temp_package, require_precision=True)' in code
