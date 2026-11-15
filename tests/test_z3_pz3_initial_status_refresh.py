from pathlib import Path

from auto_annotation_tool.gui import z3_cvat_tab_ui


def test_pz3_builder_refreshes_status_after_status_widgets_are_created():
    source = Path(z3_cvat_tab_ui.__file__).read_text(encoding="utf-8-sig")

    dataset_refresh_pos = source.find("_refresh_pz3_dataset_mode_ui()")
    run_widget_pos = source.find("self.run_console = tk.Label")
    import_widget_pos = source.find("self.import_console = tk.Label")
    initial_status_refresh_pos = source.find(
        'refresh_status = getattr(self, "_refresh_pz3_status_panel_ui", None)'
    )

    assert dataset_refresh_pos >= 0
    assert run_widget_pos >= 0
    assert import_widget_pos >= 0
    assert initial_status_refresh_pos >= 0

    # Pierwszy refresh lewej części zachodzi podczas budowy jeszcze przed
    # powstaniem tabeli statusu; dlatego wymagany jest drugi refresh po niej.
    assert dataset_refresh_pos < run_widget_pos
    assert import_widget_pos < initial_status_refresh_pos
