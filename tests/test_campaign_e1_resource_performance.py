from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auto_annotation_tool.gui import campaign_step1_assets as assets


def _write_xml(path: Path):
    path.write_text(
        "<annotations>\n"
        "  <image id=\"0\" name=\"AA11111_001.jpg\">\n"
        "    <polygon label=\"plate\" points=\"0,0;10,0;10,5;0,5\"/>\n"
        "  </image>\n"
        "  <image id=\"1\" name=\"BB22222_001.jpg\">\n"
        "    <box label=\"plate\" xtl=\"0\" ytl=\"0\" xbr=\"10\" ybr=\"5\"/>\n"
        "    <polygon label=\"plate\" points=\"20,0;30,0;30,5;20,5\"/>\n"
        "  </image>\n"
        "</annotations>\n",
        encoding="utf-8",
    )


def test_lightweight_xml_summary_counts_and_caches(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_xml(run / "annotations.xml")
    host = SimpleNamespace()

    with patch.object(assets.ET, "iterparse", wraps=assets.ET.iterparse) as wrapped:
        first = assets._scan_project_start_plate_xml_summary(host, run)
        second = assets._scan_project_start_plate_xml_summary(host, run)

    assert first["plate_images"] == 2
    assert first["plate_count"] == 3
    assert second["plate_count"] == 3
    assert wrapped.call_count == 1


def test_compatibility_does_not_use_heavy_z2_parser(tmp_path):
    run = tmp_path / "run"
    images = tmp_path / "images"
    run.mkdir()
    images.mkdir()
    _write_xml(run / "annotations.xml")
    for name in ("AA11111_001.jpg", "BB22222_001.jpg"):
        (images / name).write_bytes(b"x")

    annotation = Mock()
    annotation._resolve_safe_annotation_run_dir.return_value = run
    annotation._parse_cvat_preview_annotations.side_effect = AssertionError(
        "pełny parser Z2 nie powinien być używany"
    )

    class Host:
        def __init__(self):
            self.app = SimpleNamespace(tabs={"annotation": annotation})

        def _scan_project_start_normalized_image_names(self, directory):
            return {
                assets.CAMPAIGN._normalize_image_set_name(path.name)
                for path in Path(directory).iterdir()
                if path.is_file()
            }

        def _get_project_start_normalized_image_names(self, directory):
            return self._scan_project_start_normalized_image_names(directory)

        def _get_project_start_approved_normalized_image_names(self):
            return set()

        def _project_start_annotation_covers_filename_plates(self, filename, plate_count):
            return True, 1

    result = assets._check_project_start_run_compatibility(
        Host(), run, images, adoptable_only=True
    )

    assert result["checked"]
    assert result["ok"]
    assert result["matched"] == 2
    assert result["matched_plate_count"] == 3
    assert result["plate_count"] == 3
    annotation._parse_cvat_preview_annotations.assert_not_called()
