from pathlib import Path
from types import SimpleNamespace

from auto_annotation_tool.gui import z2_layout_ui_runtime
from auto_annotation_tool import campaign_manager


ROOT = Path(__file__).resolve().parents[1]


def test_t02_scope_prefers_only_plate_bearing_images_from_project_start_xml(tmp_path, monkeypatch):
    xml = tmp_path / "annotations.xml"
    xml.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="A.jpg" width="100" height="50">
    <polygon label="plate" points="1,1;10,1;10,5;1,5"/>
  </image>
  <image id="1" name="empty.jpg" width="100" height="50"/>
  <image id="2" name="vehicle_only.jpg" width="100" height="50">
    <polygon label="vehicle" points="1,1;20,1;20,10;1,10"/>
  </image>
  <image id="3" name="nested/B.jpg" width="100" height="50">
    <box label="tablica" xtl="1" ytl="1" xbr="10" ybr="5"/>
  </image>
</annotations>
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        campaign_manager.CAMPAIGN,
        "get_project_start_plate_source",
        lambda: {"source_xml_path": str(xml)},
    )

    host = SimpleNamespace(
        _campaign_graph_entry_context={"z2_work_mode": "t02_at_review"},
        current_annotation_run_dir=Path("correction_run"),
        last_staging_run_dir=None,
        _load_annotation_run_manifest=lambda _run: {
            "input_scope_filenames": [f"img_{idx}.jpg" for idx in range(299)]
        },
    )

    assert z2_layout_ui_runtime._get_t02_review_scope_filenames(host) == {
        "a.jpg",
        "b.jpg",
    }


def test_t02_right_panel_subtracts_project_pool_from_session_ok():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z2_annotation_process.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("    t02_project_images = 0")
    end = source.index("    missing_plates =", start)
    segment = source[start:end]

    assert "session_only_names = session_approved_names - project_approved_names" in segment
    assert "filename not in session_only_names" in segment
    assert "t02_session_images = int(session_images)" in segment
    assert "t02_session_plates = int(session_plates)" in segment


def test_t02_review_filter_still_uses_scope_helper():
    source = (
        ROOT
        / "auto_annotation_tool"
        / "gui"
        / "z2_layout_ui_runtime.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def _filter_preview_list_entries")
    end = source.index("def _reset_preview_metric_filters", start)
    segment = source[start:end]

    assert "t02_scope_filenames = _get_t02_review_scope_filenames(self)" in segment
    assert "in t02_scope_filenames" in segment
