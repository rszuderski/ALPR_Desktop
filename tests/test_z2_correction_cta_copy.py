from pathlib import Path


def test_correction_ctas_are_directional_and_consistent():
    root = Path(__file__).resolve().parents[1]
    panel = (root / "auto_annotation_tool/gui/z2_panel_workflow.py").read_text(
        encoding="utf-8-sig"
    )
    preview = (root / "auto_annotation_tool/gui/z2_preview_editor.py").read_text(
        encoding="utf-8-sig"
    )
    shared = (root / "auto_annotation_tool/gui/z2_shared_ui.py").read_text(
        encoding="utf-8-sig"
    )
    flow = (root / "auto_annotation_tool/gui/z2_free_mode_flow.py").read_text(
        encoding="utf-8-sig"
    )

    assert 'text="Przejdź do Z3"' in panel
    assert 'text="Przejdź do eksportu"' in panel
    assert 'text="Przejdź do Z3"' in preview
    assert 'text="Przejdź do eksportu"' in preview
    assert 'text="Przejdź do Z3"' in shared
    assert 'next_text = "Przejdź do eksportu"' in flow

    assert 'text="Wyodrębnij tablice"' not in panel
    assert 'text="Otwórz eksport"' not in panel
