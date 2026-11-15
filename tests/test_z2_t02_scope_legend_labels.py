from types import SimpleNamespace

from auto_annotation_tool.gui import z2_legend_ui


def _theme():
    return {
        "badge_plate_fill": "a",
        "badge_plate_outline": "b",
        "badge_plate_multi_fill": "c",
        "badge_plate_multi_outline": "d",
        "badge_image_fill": "e",
        "badge_image_outline": "f",
        "entry_fill": "g",
        "panel_outline": "h",
    }


def test_t02_legend_uses_control_and_pool_o_labels():
    owner = SimpleNamespace(
        _get_preview_legend_context=lambda: {
            "image_label": "Kontrola",
            "image_text": "Kontrola: 1/10",
            "plate_text": "Tablica: 1/1",
            "vehicle_text": "Pojazd: 0/0",
            "plate_total": 1,
            "pool_label": "Pula O",
            "pool_text": "Pula O: 299",
        },
        _get_preview_legend_theme=_theme,
    )

    rows = z2_legend_ui.build_preview_controls_context_rows(owner)

    assert rows[0][0] == "Kontrola"
    assert rows[0][1] == "1/10"
    assert rows[-1][0] == "Pula O"
    assert rows[-1][1] == "299"


def test_non_t02_legend_keeps_old_labels():
    owner = SimpleNamespace(
        _get_preview_legend_context=lambda: {
            "image_text": "Zdjęcie: 2/15",
            "plate_text": "Tablica: 1/1",
            "vehicle_text": "Pojazd: 0/0",
            "plate_total": 1,
            "pool_text": "Pula: 100",
        },
        _get_preview_legend_theme=_theme,
    )

    rows = z2_legend_ui.build_preview_controls_context_rows(owner)

    assert rows[0][0] == "Zdjęcie"
    assert rows[0][1] == "2/15"
    assert rows[-1][0] == "Pula"
    assert rows[-1][1] == "100"
