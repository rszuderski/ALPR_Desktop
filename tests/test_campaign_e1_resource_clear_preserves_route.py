from pathlib import Path

from auto_annotation_tool.gui import campaign_step1_assets


def _function_segment() -> str:
    path = Path(campaign_step1_assets.__file__)
    source = path.read_text(encoding="utf-8-sig")
    start = source.index("def _clear_project_start_asset")
    end = source.index("def _short_project_start_asset_validation", start)
    return source[start:end]


def test_clearing_e1_resource_does_not_clear_selected_route():
    segment = _function_segment()

    assert "CAMPAIGN.clear_iteration_target()" not in segment
    assert "Wybrana bramka pozostaje aktywna" in segment


def test_route_change_remains_separate_from_resource_clear():
    root = Path(__file__).resolve().parents[1]
    navigation = (
        root
        / "auto_annotation_tool"
        / "gui"
        / "campaign_navigation.py"
    ).read_text(encoding="utf-8-sig")

    assert 'self._return_to_step1_for_char_source_rework(clear_target=True)' in navigation
    assert 'CAMPAIGN.set_iteration_target("")' in navigation
