from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / "auto_annotation_tool" / "gui" / name).read_text(
        encoding="utf-8-sig"
    )


def test_variant_options_cta_describes_whole_pz1_editor():
    panels = _read("z4_dataset_panels.py")
    flow = _read("z4_campaign_flow.py")
    shared = _read("z4_shared_ui.py")
    view_models = _read("z4_view_models.py")

    assert 'text="Dostosuj wariant"' in panels
    assert 'split_toggle_label: str = "Dostosuj wariant"' in view_models
    assert 'or "Dostosuj wariant"' in shared

    assert 'split_toggle_label = "Dostosuj wariant"' in flow
    assert (
        'split_toggle_label = "Ukryj ustawienia wariantu" '
        'if show_split_details else "Dostosuj wariant"'
    ) in flow


def test_ready_variant_copy_names_all_editable_areas_not_only_split():
    flow = _read("z4_campaign_flow.py")

    assert "podział train / val / test" in flow
    assert "syntetyczne uzupełnienie train" in flow
    assert "ustawienia pokrycia znaków" in flow

    assert 'split_toggle_label = "Popraw split"' not in flow
    assert '"Ukryj opcje splitu"' not in flow
