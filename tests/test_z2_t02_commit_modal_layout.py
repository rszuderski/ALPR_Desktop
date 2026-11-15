from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "auto_annotation_tool" / "gui" / "z2_miniflow_runtime.py"


def _modal_source() -> str:
    source = PATH.read_text(encoding="utf-8-sig")
    start = source.index("def _prompt_t02_at_review_commit_choice")
    end = source.index("def _return_to_campaign_t02_at_review", start)
    return source[start:end]


def test_t02_commit_modal_is_compact_structured_and_theme_safe():
    source = _modal_source()

    assert 'title="Zapis kontroli AT"' in source
    assert 'geometry="760x360"' in source
    assert 'surface_bg = str(body.cget("bg") or panel_bg)' in source
    assert 'summary_bg = blend_hex_colors(surface_bg, warning, 0.09)' in source

    assert '"Do zapisania"' in source
    assert '"Po zapisie"' in source
    assert '"Zmiana toru"' in source
    assert "summary.grid_columnconfigure(0" in source
    assert "summary.grid_columnconfigure(1" in source

    assert 'text="Zapisz kontrolę T02"' in source
    assert 'text="Zapisz kontrolę i wybierz T02"' not in source
    assert 'text="Wróć do kontroli"' in source
    assert "_fit_dialog_to_content" in source


def test_t02_commit_modal_no_longer_uses_dark_palette_card_as_whole_summary():
    source = _modal_source()

    assert 'palette.get("card"' not in source
    assert 'shell.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)' in source
    assert "To jest pierwszy realny wkład T02 w tej iteracji." not in source
    assert "Zapis kontroli AT zamknie wybór toru tej iteracji" not in source
