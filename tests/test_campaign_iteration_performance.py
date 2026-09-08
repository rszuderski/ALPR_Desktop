from collections import Counter
import copy
import os
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_annotation_tool.image_directory_index import ImageDirectoryIndex
from auto_annotation_tool.campaign_manager import CampaignManager
from auto_annotation_tool.campaign_ingest_planner import CampaignIngestPlanner, IngestCandidate, CHAR_ALPHABET
from auto_annotation_tool.gui import campaign_iteration_flow as flow, z2_session_runtime, z4_campaign_flow


def test_directory_token_matches_explicit_names_and_keeps_duplicate_file_count(tmp_path):
    (tmp_path / "nested").mkdir()
    for name in ("ABCD.jpg", "nested/abcd.JPG", "nested/OTHER.png", "notes.txt"):
        (tmp_path / name).touch()
    index = ImageDirectoryIndex()
    result = index.snapshot(tmp_path, {".jpg", ".png"}, recursive=True)
    manager = CampaignManager.__new__(CampaignManager)
    assert result.count == 3
    assert result.token == manager.build_image_name_set_token(["ABCD.jpg", "abcd.JPG", "OTHER.png"])
    assert index.snapshot(tmp_path, {".jpg", ".png"}).count == 1


def test_cached_directory_does_not_stat_images_or_rescan_entries(tmp_path, monkeypatch):
    (tmp_path / "ABCD.jpg").touch()
    index = ImageDirectoryIndex()
    result = index.snapshot(tmp_path, {".jpg"}, recursive=True)
    original_stat = os.stat
    def stat(path, *args, **kwargs):
        assert Path(path) == tmp_path
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(os, "stat", stat)
    monkeypatch.setattr(os, "scandir", Mock(side_effect=AssertionError("cached directory was rescanned")))
    assert index.snapshot(tmp_path, {".jpg"}, recursive=True) == result


def test_nested_external_add_remove_rename_invalidates_token(tmp_path):
    child = tmp_path / "nested"
    child.mkdir()
    (child / "ABCD.jpg").touch()
    index = ImageDirectoryIndex()
    first = index.snapshot(tmp_path, {".jpg"}, recursive=True)
    root_stat = tmp_path.stat()
    (child / "EFGH.jpg").touch()
    # A parent directory timestamp alone cannot detect changes inside a child.
    os.utime(tmp_path, ns=(root_stat.st_atime_ns, root_stat.st_mtime_ns))
    second = index.snapshot(tmp_path, {".jpg"}, recursive=True)
    assert second.count == 2 and second.token != first.token
    (child / "EFGH.jpg").rename(child / "JKLM.jpg")
    third = index.snapshot(tmp_path, {".jpg"}, recursive=True)
    assert third.count == 2 and third.token != second.token
    (child / "JKLM.jpg").unlink()
    assert index.snapshot(tmp_path, {".jpg"}, recursive=True).token == first.token


def test_new_child_and_separate_roots_are_not_mixed(tmp_path):
    index = ImageDirectoryIndex(capacity=1)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    assert index.snapshot(a, {".jpg"}, recursive=True).count == 0
    (a / "sub").mkdir()
    (a / "sub/ABCD.jpg").touch()
    assert index.snapshot(a, {".jpg"}, recursive=True).count == 1
    assert index.snapshot(b, {".jpg"}, recursive=True).count == 0
    assert len(index._cache) == 1
    assert index.snapshot(a, {".jpg"}, recursive=True).count == 1


def legacy_selection(planner, candidates, balance, limit):
    remaining = copy.deepcopy(candidates)
    selected = []
    balance = Counter(balance)
    while remaining and len(selected) < limit:
        best_index, best_score, best_details = -1, -1.0, {}
        for index, candidate in enumerate(remaining):
            score, details = planner.score_candidate(candidate.char_histogram, balance)
            if score > best_score:
                best_index, best_score, best_details = index, score, details
        chosen = remaining.pop(best_index)
        chosen.score, chosen.score_details = round(best_score, 6), best_details
        selected.append(chosen)
        balance.update(chosen.char_histogram)
    return selected, balance


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("limit", [0, 20, 150])
def test_vector_planner_preserves_greedy_order_scores_and_details(seed, limit):
    randomizer = random.Random(seed)
    planner = CampaignIngestPlanner()
    candidates = []
    for index in range(100):
        text = "AB1234" if index % 3 == 0 else "".join(randomizer.choices(CHAR_ALPHABET, k=8))
        hist = planner.build_char_histogram([text])
        candidates.append(IngestCandidate(str(index), str(index), str(index), [text], hist))
    balance = Counter({ch: randomizer.randrange(10000) if seed % 2 else 0 for ch in CHAR_ALPHABET})
    expected, expected_balance = legacy_selection(planner, candidates, balance, limit)
    actual, actual_balance = planner._select_candidates(copy.deepcopy(candidates), balance, limit)
    assert [item.to_dict() for item in actual] == [item.to_dict() for item in expected]
    assert actual_balance == expected_balance
    assert len({item.source_key for item in actual}) == len(actual)


def test_pool_scan_preserves_order_keys_exclusions_and_counts(tmp_path):
    (tmp_path / "nested").mkdir()
    for name in ("AB1234_001.jpg", "AB1234_002.jpg", "notes.txt", "nested/WX1234_001.PNG", "bad.jpg"):
        (tmp_path / name).touch()
    planner = CampaignIngestPlanner()
    expected = [(p, planner.make_source_key(p, tmp_path)) for p in tmp_path.rglob("*")
                if p.is_file() and p.suffix.lower() in planner.image_extensions]
    assert list(planner._pool_images(tmp_path)) == expected
    result = planner.plan_from_master_pool(tmp_path, used_filenames=["AB1234_001.jpg"], batch_size=0)
    assert result["selected_total"] == 2 and result["skipped_used"] == 1
    assert result["skipped_invalid_ground_truth"] == 1


def test_iteration_finish_resets_hidden_z2_and_renders_new_graph_once(monkeypatch):
    manager = Mock()
    manager.get_iteration_image_source_dir.return_value = None
    manager.get_current_iteration_num.return_value = 3
    monkeypatch.setattr(flow, "CAMPAIGN", manager)
    annotation = Mock()
    host = Mock()
    host.app.tabs = {"annotation": annotation}
    host._project_open_lightweight_refresh = False
    host._refresh_dashboard.side_effect = lambda: assert_lightweight(host)
    flow._finish_iteration_advance(host, "new_input", {"ok": True, "next_iteration": 3})
    annotation.prepare_campaign_iteration_transition.assert_called_once_with(input_dir=None, refresh_ui=False)
    host._refresh_dashboard.assert_called_once()
    host._rebuild_wizard_stage_ui.assert_not_called()
    assert host._project_open_lightweight_refresh is False


def assert_lightweight(host):
    assert host._project_open_lightweight_refresh is True


def test_hidden_z2_reset_does_not_search_export_runs(monkeypatch, tmp_path):
    host = Mock()
    host._free_mode_session_save_after_id = None
    host._get_campaign_annotation_state_path.return_value = tmp_path / "absent.json"
    z2_session_runtime.prepare_campaign_iteration_transition(host, input_dir=tmp_path, refresh_ui=False)
    host._reset_campaign_runtime_state.assert_called_once_with(input_dir=tmp_path)
    host._refresh_step2_action_states.assert_not_called()
    assert host._campaign_context_project_name == ""


@pytest.mark.parametrize("trained", [False, True])
def test_t06_completion_starts_worker_without_intermediate_graph(trained, monkeypatch):
    manager = Mock()
    manager.get_active_project_name.return_value = "demo"
    manager.get_current_iteration_num.return_value = 2
    manager.get_iteration_target.return_value = "char"
    manager.get_step4_without_training_decision.return_value = {"ready": True, "iteration": 2, "target": "char"}
    monkeypatch.setattr(flow, "CAMPAIGN", manager)
    monkeypatch.setattr(z4_campaign_flow, "CAMPAIGN", manager)
    owner = Mock()
    owner._ask_iteration_advance_mode.return_value = "new_input"
    if trained:
        host = Mock()
        host.app.tabs = {"campaign": owner}
        host._step4_campaign_finish_ready = True
        assert z4_campaign_flow.finish_campaign_step4(host)
        host.main_nb.select.assert_not_called()
        host.app.open_controlled_tab.assert_called_once_with("campaign")
    else:
        owner.app.tabs = {}
        flow._finish_step4_without_training(owner)
    owner._start_iteration_advance.assert_called_once_with("new_input")
    owner._refresh_dashboard.assert_not_called()
    owner._refresh_active_project_wizard_only.assert_not_called()


@pytest.mark.parametrize("saved", [False, True])
def test_stage_transfer_preserves_manifest_links_and_only_consumes_stage_after_save(tmp_path, saved):
    manager = CampaignManager.__new__(CampaignManager)
    stage = tmp_path / "stage/Iteracja_001"
    images = stage / "images"
    images.mkdir(parents=True)
    names = [f"AB1234_{i:03d}.jpg" for i in range(70)]
    for name in names:
        (images / name).touch()
    marker = stage / "stage_manifest.json"
    marker.write_text('{"pending_images": 70}', encoding="utf-8")
    target = tmp_path / "raw/Iteracja_002"
    manager.get_iteration_raw_dir = lambda *args: target
    manager.get_staging_dir = lambda *args: stage.parent
    manager.save_ingest_manifest = Mock(return_value=saved)
    result = manager._seed_iteration_from_stage(source_iteration=1, target_iteration=2, project_name="test")
    assert result["ok"] == saved
    manifest = manager.save_ingest_manifest.call_args.args[0]
    assert manifest["selected_count"] == 70 and manifest["manifest_only"]
    assert {row["name"] for row in manifest["selected_images"]} == set(names)
    for row in manifest["selected_images"]:
        assert row["source_path"] == row["target_path"] == str(images / row["name"])
        assert row["iteration_target_path"] == str(target / row["name"])
    assert not target.exists()
    import json
    assert json.loads(marker.read_text(encoding="utf-8"))["pending_images"] == (0 if saved else 70)
