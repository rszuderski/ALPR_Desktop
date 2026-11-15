from pathlib import Path

from auto_annotation_tool.gui import tab_campaign


def test_campaign_tab_exposes_structured_annotation_import_modal_wrapper():
    source = Path(tab_campaign.__file__).read_text(encoding="utf-8-sig")

    assert (
        "def _show_project_start_annotation_import_modal(self, *args, **kwargs):"
        in source
    )
    assert (
        "return campaign_step1_assets._show_project_start_annotation_import_modal("
        "self, *args, **kwargs)"
        in source
    )


def test_annotation_import_choice_calls_structured_modal_before_text_fallback():
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "auto_annotation_tool"
        / "gui"
        / "campaign_step1_assets.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def _choose_project_start_annotation_import_mode")
    end = source.index("def _check_project_start_run_compatibility", start)
    segment = source[start:end]

    assert "self._show_project_start_annotation_import_modal(" in segment
    assert "message = (" in segment
    assert (
        segment.index("self._show_project_start_annotation_import_modal(")
        < segment.index("message = (")
    )


def test_structured_modal_contains_colored_summary_table():
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "auto_annotation_tool"
        / "gui"
        / "campaign_step1_assets.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def _show_project_start_annotation_import_modal")
    end = source.index("def _choose_project_start_annotation_import_mode", start)
    segment = source[start:end]

    assert '("Stan", "Co sprawdzam", "Wynik", "Znaczenie")' in segment
    assert '"Podsumowanie importu AT"' in segment
    assert '"PASUJE" if package_matched > 0 else "BRAK"' in segment
    assert '"Importuj pasujące AT do kontroli w Z2"' in segment
