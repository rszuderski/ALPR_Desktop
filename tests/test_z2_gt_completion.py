from pathlib import Path

from PIL import Image

from auto_annotation_tool.gt_pack import ALPRGTPack
from auto_annotation_tool.gui.z2_gt_completion import (
    build_gt_completion_state,
    export_gt_snapshot_pack,
)


def make_complete_pack(tmp_path: Path) -> Path:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (80, 40), "white").save(image_path)

    pack_path = tmp_path / "current_work.alprgt"
    pack = ALPRGTPack.create(pack_path)
    image = pack.add_image(image_path)
    plate = pack.ensure_plate(
        image_id=image["image_id"],
        polygon=[
            [10, 10],
            [70, 10],
            [70, 30],
            [10, 30],
        ],
    )
    plate_id = plate["plate_id"]
    pack.set_ground_truth(plate_id, "WI1234A")
    pack.set_plate_layout_gt(plate_id, "single_row")
    return pack_path


def test_completion_state_requires_gt_layout_and_geometry(tmp_path):
    pack_path = make_complete_pack(tmp_path)

    state = build_gt_completion_state(pack_path)

    assert state["valid"] is True
    assert state["plates"] == 1
    assert state["gt_set"] == 1
    assert state["layout_set"] == 1
    assert state["geometry_set"] == 1
    assert state["missing_gt"] == 0
    assert state["missing_layout"] == 0
    assert state["missing_geometry"] == 0
    assert state["conflicts"] == 0
    assert state["complete"] is True
    assert state["status"] == "GOTOWE"


def test_completion_state_marks_missing_gt_as_incomplete(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (80, 40), "white").save(image_path)
    pack_path = tmp_path / "current_work.alprgt"
    pack = ALPRGTPack.create(pack_path)
    image = pack.add_image(image_path)
    pack.ensure_plate(
        image_id=image["image_id"],
        polygon=[
            [10, 10],
            [70, 10],
            [70, 30],
            [10, 30],
        ],
    )

    state = build_gt_completion_state(pack_path)

    assert state["complete"] is False
    assert state["missing_gt"] == 1
    assert state["status"] == "WYMAGA UZUPEŁNIENIA"


def test_snapshot_is_valid_portable_copy(tmp_path):
    pack_path = make_complete_pack(tmp_path)
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    result = export_gt_snapshot_pack(
        pack_path,
        export_dir,
        timestamp="20260921_160000",
    )

    snapshot = result["path"]
    assert snapshot.name == "gt_copy_20260921_160000.alprgt"
    assert snapshot.is_dir()
    assert (snapshot / "manifest.json").is_file()
    assert not (snapshot / ".pack.lock").exists()
    assert result["validation"]["ok"] is True

    copied = build_gt_completion_state(snapshot)
    assert copied["complete"] is True


def test_gt_completion_is_nested_in_correction_cards():
    root = Path(__file__).resolve().parents[1]
    widgets = (root / "auto_annotation_tool/gui/z2_main_widgets.py").read_text(
        encoding="utf-8-sig"
    )
    shared = (root / "auto_annotation_tool/gui/z2_shared_ui.py").read_text(
        encoding="utf-8-sig"
    )
    flow = (root / "auto_annotation_tool/gui/z2_free_mode_flow.py").read_text(
        encoding="utf-8-sig"
    )

    assert "gt_completion_section" not in widgets
    assert 'key="followup"' in widgets
    assert 'key="manual"' in widgets
    assert "show_auto_followup" in shared
    assert "show_manual_review_followup" in shared
    assert 'AUTO_REVIEW_FOLLOWUP_TITLE = "Korekta i decyzja"' in flow
    assert 'MANUAL_REVIEW_FOLLOWUP_TITLE = "Korekta i decyzja"' in flow


def test_gt_copy_filename_uses_copy_terminology(tmp_path):
    pack_path = make_complete_pack(tmp_path)
    export_dir = tmp_path / "exports_copy"
    export_dir.mkdir()

    result = export_gt_snapshot_pack(
        pack_path,
        export_dir,
        timestamp="20260921_180000",
    )

    assert result["path"].name == "gt_copy_20260921_180000.alprgt"


def test_gt_copy_ui_requires_first_real_gt():
    root = Path(__file__).resolve().parents[1]
    module = (
        root / "auto_annotation_tool/gui/z2_gt_completion.py"
    ).read_text(encoding="utf-8-sig")

    assert 'text="Zapisz kopię GT…"' in module
    assert 'int(state.get("gt_set", 0) or 0) >= 1' in module
    assert "Kopia GT będzie dostępna po zapisaniu pierwszego GT." in module
    assert 'base_name = f"gt_copy_{stamp}.alprgt"' in module
    assert 'text="Snapshot GT…"' not in module
