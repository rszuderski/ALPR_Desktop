from types import SimpleNamespace
from unittest.mock import Mock
import json
from auto_annotation_tool.campaign_ingest_planner import CampaignIngestPlanner
from auto_annotation_tool.campaign_manager import CampaignManager
from auto_annotation_tool.gui import campaign_step1_ingest as ingest


def test_explicit_empty_ground_truth_does_not_fallback_to_filename():
    fields = CampaignIngestPlanner().normalize_text_metadata("WI1234A_001.jpg", {"ground_truth_texts": []})
    assert fields["ground_truth_texts"] == []
    assert fields["char_histogram"] == {}
    assert fields["filename_text_hints"] == ["WI1234A"]
    assert fields["planning_char_histogram"]
    assert fields["histogram_source"] == "filename_hint"


def test_explicit_gt_overrides_filename_hint_without_deleting_it():
    fields = CampaignIngestPlanner().normalize_text_metadata("WI123AA.jpg", {"ground_truth_texts": ["wi 1234a"]})
    assert fields["ground_truth_texts"] == ["WI1234A"]
    assert fields["filename_text_hints"] == ["WI123AA"]
    assert fields["char_histogram"] == fields["planning_char_histogram"]
    assert fields["histogram_source"] == "explicit_gt"


def test_planner_accepts_unlabelled_images_and_selects_a_mixed_batch(tmp_path):
    for name in ("known_A.jpg", "known_B.jpg", "x.jpg"):
        (tmp_path / name).write_bytes(name.encode())
    planner = CampaignIngestPlanner()
    kwargs = {"batch_size": 2, "source_metadata": {
        "known_a.jpg": {"ground_truth_texts": ["ABCD"]},
        "known_b.jpg": {"ground_truth_texts": ["EFGH"]},
    }}
    result = planner.plan_from_master_pool(tmp_path, **kwargs)
    assert result["candidates_total"] == 3
    assert result["skipped_invalid_ground_truth"] == 0
    assert result["explicit_gt_count"] == 2
    assert result["missing_explicit_gt_count"] == 1
    assert "x.jpg" in [item["name"] for item in result["selected"]]
    again = planner.plan_from_master_pool(tmp_path, **kwargs)
    assert result["selected"] == again["selected"]


def test_filename_hints_do_not_count_as_explicit_gt():
    planner = CampaignIngestPlanner()
    items = [planner.normalize_text_metadata("WI1234A.jpg"),
             planner.normalize_text_metadata("x.jpg", {"ground_truth_texts": ["ABCD"]})]
    assert planner.gt_statistics(items) == {"explicit_gt_count": 1, "missing_explicit_gt_count": 1,
                                          "filename_hint_count": 1, "candidates_with_gt": 1, "candidates_without_gt": 1}


def test_manifest_display_preserves_missing_gt_and_empty_histograms(monkeypatch):
    manager = SimpleNamespace(get_master_pool_dir=lambda: None, get_iteration_raw_dir=lambda: None,
                              get_active_project_name=lambda: "test", get_current_iteration_num=lambda: 1)
    monkeypatch.setattr(ingest, "CAMPAIGN", manager)
    result = ingest._build_ingest_plan_from_manifest_for_display(SimpleNamespace(), {
        "selected_images": [{"name": "WI1234A.jpg", "ground_truth_texts": []}, {"name": "x.jpg"}],
        "selected_count": 2})
    assert result["selected_total"] == 2
    assert result["explicit_gt_count"] == 0
    assert all(not item["ground_truth_texts"] and not item["char_histogram"] for item in result["selected"])


def test_all_manifest_writers_normalize_explicit_gt_and_hints(tmp_path):
    source = tmp_path / "WI1234A.jpg"
    source.write_bytes(b"image")
    manager = SimpleNamespace(get_ingest_manifest_path=lambda *args: tmp_path / "manifest.json")
    manifest = {"selected_images": [{"name": "WI1234A.jpg", "source_path": str(source), "ground_truth_texts": []}]}
    assert CampaignManager.save_ingest_manifest(manager, manifest)
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    row = saved["selected_images"][0]
    assert row["ground_truth_texts"] == [] and row["filename_text_hints"] == ["WI1234A"]
    assert saved["missing_explicit_gt_count"] == 1
