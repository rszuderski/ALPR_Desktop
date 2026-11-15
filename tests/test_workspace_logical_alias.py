from pathlib import Path

from auto_annotation_tool.config import CONFIG


ROOT = Path(__file__).resolve().parents[1]


def test_workspace_root_keeps_logical_alias_instead_of_resolving_junction():
    expected = Path("Workspace").absolute()

    assert CONFIG.WORKSPACE_DIR == expected
    assert CONFIG.WORKSPACE_DIR.name == "Workspace"
    assert CONFIG.DIR_1_RAW == expected / "1_raw_images"

    source = (ROOT / "auto_annotation_tool" / "config.py").read_text(
        encoding="utf-8-sig"
    )
    assert 'WORKSPACE_DIR: Path = Path("Workspace").absolute()' in source
    assert 'WORKSPACE_DIR: Path = Path("Workspace").resolve()' not in source


def test_physical_resolution_remains_available_only_when_explicitly_requested():
    logical = CONFIG.WORKSPACE_DIR
    physical = logical.resolve()

    # Na zwykłym katalogu obie ścieżki mogą być identyczne.
    # Na junctionie Windows physical może wskazać target, ale logiczny alias
    # aplikacji ma pozostać niezmieniony.
    assert CONFIG.WORKSPACE_DIR == logical
    assert logical.name == "Workspace"
    assert physical.exists() or logical.exists()


def test_image_picker_fallback_uses_logical_dir_1_raw():
    source = (ROOT / "auto_annotation_tool" / "gui" / "tab_campaign.py").read_text(
        encoding="utf-8-sig"
    )

    assert "initial = Path(CONFIG.DIR_1_RAW)" in source
    assert "initial = Path(CONFIG.DIR_1_RAW).resolve()" not in source
