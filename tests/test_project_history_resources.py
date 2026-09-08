import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from auto_annotation_tool.campaign_history_resources import HISTORY_RESOURCE_SCHEMA, HistoryResourceReader
from auto_annotation_tool.gui import app_project_history as history_ui
from auto_annotation_tool.gui import campaign_graph_actions as graph


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ProjectHistoryResourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / "_campaign_state"
        self.source = self.root / "source"
        self.run1 = self.root / "2_auto_annotations" / "run1"
        self.run2 = self.root / "2_auto_annotations" / "run2"
        self.t1 = "2026-09-02T19:54:30"
        self.t2 = "2026-09-04T17:00:29"
        self.entries = {}
        for i in range(13):
            name = f"image_{i}.jpg"
            count = 2 if i == 11 else 1
            iteration = 1 if i < 10 else 2
            self.entries[str(i)] = {
                "image_name": name, "source_image_path": str(self.source / name),
                "first_approved_iteration": iteration, "approved_iteration": iteration,
                "first_approved_at": self.t1 if iteration == 1 else self.t2,
                "approved_at": "2026-09-02T21:20:59" if iteration == 1 else self.t2,
                "plate_count": count,
            }
        write_json(self.state / "plate_approved_set.json", {"project": "demo", "entries": self.entries})
        self._make_run(self.run1, range(10), self.t1)
        self._make_run(self.run2, range(10, 13), self.t2)
        self.summary_path = self.root / "crops" / "export_summary.json"
        summary = {
            "created_iteration": 1, "created_at": "2026-09-02T22:02:50",
            "exportable_plate_count": 10, "exportable_char_count": 69,
            "gold_dataset_valid": True, "gold_dataset_path": str(self.root / "gold"),
        }
        write_json(self.summary_path, summary)
        self.registry = {
            "iteration_index": {"1": "p1", "2": "p2"},
            "packages": {
                "p1": {"iteration_last_seen": 1, "plate_source": {"run_dir": str(self.run1), "updated_at": self.t1}},
                "p2": {"iteration_last_seen": 2, "plate_source": {"run_dir": str(self.run2), "updated_at": self.t2}},
            },
            "iteration_state": {
                "1": {
                    "t06_contracts": {"pz3_char_dataset": {
                        "summary_path": str(self.summary_path), "created_iteration": 1,
                        "fulfilled_at": "2026-09-02T22:25:00", "exportable_plate_count": 10, "exportable_char_count": 69,
                    }},
                    "step4_dataset": {
                        "dataset_path": str(self.root / "augmented"), "target": "char", "dataset_iteration": 1,
                        "train_images": 152, "val_images": 1, "test_images": 1, "total_images": 154,
                        "updated_at": "2026-09-03T11:52:50",
                    },
                },
            },
        }
        self._save_registry()
        write_json(self.state / "ingest" / "iter_001_manifest.json", {"selected_count": 1000, "created_at": self.t1})
        self.event = {
            "project": "demo", "iteration": 1, "created_at": self.t1,
            "action": "approve_step2", "status": "ok", "step": 2, "transition_id": "e2_to_e3",
            "artifacts": {"approved_images": 0, "approved_plates": 0},
        }
        patcher = patch.object(history_ui.CAMPAIGN, "get_project_root_dir", return_value=self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _save_registry(self):
        write_json(self.state / "artifact_registry.json", self.registry)

    def _reader(self):
        return HistoryResourceReader(self.root, "demo")

    def _make_run(self, root, indices, when):
        write_json(root / "run_manifest.json", {
            "approved_filenames": [f"image_{i}.jpg" for i in indices],
            "source_input_dir": str(self.source), "last_manual_edit_at": when,
            "resume_preview_saved_at": when, "result_total_plates": 0,
        })
        images = []
        for i in list(indices) + [99]:
            polygons = '<polygon label="plate" points="0,0;1,0;1,1;0,1"/>' * (2 if i == 11 else 1)
            images.append(f'<image name="image_{i}.jpg">{polygons}</image>')
        (root / "annotations.xml").write_text("<annotations>" + "".join(images) + "</annotations>", encoding="utf-8")

    def test_demo_legacy_zero_is_recovered_without_using_iteration_two(self):
        reader = self._reader()
        counts = reader.for_event(self.event)["AT"]
        self.assertEqual((counts["images"], counts["plates"]), (10, 10))
        self.assertEqual((counts["iteration_images"], counts["iteration_plates"]), (10, 10))
        artifact = history_ui._format_plate_annotation_artifact(self.event, "demo", reader=reader)
        increment = history_ui._format_plate_annotation_increment(self.event, "demo", reader=reader)
        self.assertIn("10", artifact)
        self.assertIn("+10", increment)
        self.assertNotIn("13", artifact)
        self.assertEqual(self.event["artifacts"]["approved_images"], 0)

    def test_second_iteration_has_separate_total_and_increment_and_no_run_duplicates(self):
        event = dict(self.event, iteration=2, created_at=self.t2)
        counts = self._reader().for_event(event)["AT"]
        self.assertEqual((counts["images"], counts["plates"]), (13, 14))
        self.assertEqual((counts["iteration_images"], counts["iteration_plates"]), (3, 4))
        self.assertEqual((counts["previous_images"], counts["previous_plates"]), (10, 10))

    def test_pending_approved_run_is_counted_before_it_reaches_project_pool(self):
        write_json(self.state / "plate_approved_set.json", {"project": "demo", "entries": {}})
        counts = self._reader().for_event(self.event)["AT"]
        self.assertEqual((counts["images"], counts["plates"]), (10, 10))

    def test_zero_increment_in_versioned_snapshot_is_authoritative(self):
        snapshot = {"schema": HISTORY_RESOURCE_SCHEMA, "project": "demo", "iteration": 2,
                    "AT": {"images": 10, "plates": 10, "iteration_images": 0, "iteration_plates": 0}}
        event = dict(self.event, iteration=2, artifacts={"resource_snapshot": snapshot})
        self.assertEqual(self._reader().for_event(event), snapshot)
        self.assertIn("Bez nowych AT", history_ui._format_plate_annotation_increment(event, "demo"))

    def test_snapshot_of_another_project_cannot_override_counts(self):
        snapshot = {"schema": HISTORY_RESOURCE_SCHEMA, "project": "other", "iteration": 1, "AT": {"images": 999}}
        event = dict(self.event, artifacts={"resource_snapshot": snapshot})
        self.assertEqual(self._reader().for_event(event)["AT"]["images"], 10)

    def test_missing_evidence_is_unknown_not_zero_or_current_project_pool(self):
        write_json(self.state / "plate_approved_set.json", {})
        write_json(self.state / "artifact_registry.json", {})
        with patch.object(history_ui.CAMPAIGN, "get_plate_approved_set_stats", side_effect=AssertionError("current state read")):
            text = history_ui._format_plate_annotation_artifact(self.event, "demo")
        self.assertIn("Brak wiarygodnych", text)
        self.assertNotIn("0 zdjęć", text)

    def test_legacy_positive_totals_are_not_presented_as_increment(self):
        write_json(self.state / "plate_approved_set.json", {})
        write_json(self.state / "artifact_registry.json", {})
        event = dict(self.event, artifacts={"approved_images": 50, "approved_plates": 60})
        self.assertIn("50", history_ui._format_plate_annotation_artifact(event, "demo"))
        self.assertEqual(history_ui._format_plate_annotation_increment(event, "demo"), "Przyrost nie został zapisany")

    def test_later_reused_package_does_not_rewrite_earlier_iteration(self):
        self.registry["packages"]["p1"] = {"iteration_last_seen": 2, "plate_source": {
            "run_dir": str(self.run2), "updated_at": self.t2,
        }}
        self._save_registry()
        counts = self._reader().for_event(self.event)["AT"]
        self.assertEqual(counts["images"], 10)
        self.assertIsNone(counts["plates"])

    def test_changed_old_geometry_is_not_claimed_as_zero_plate_increment(self):
        self.entries["0"].update(approved_iteration=2, approved_at=self.t2, plate_count=2)
        write_json(self.state / "plate_approved_set.json", {"entries": self.entries})
        counts = self._reader().snapshot(2, cutoff=self.t2)["AT"]
        self.assertEqual(counts["plates"], 15)
        self.assertIsNone(counts["iteration_plates"])

    def test_other_resources_are_read_for_their_own_iteration_and_time(self):
        reader = self._reader()
        t05 = reader.snapshot(1, cutoff="2026-09-02T22:04:06")
        self.assertEqual(t05["O"]["images"], 1000)
        self.assertEqual((t05["AZ"]["plates"], t05["AZ"]["characters"]), (10, 69))
        self.assertEqual((t05["DS"]["images"], t05["DS"]["objects"]), (10, 69))
        self.assertNotIn("dataset_variant", t05)
        t06 = reader.snapshot(1, cutoff="2026-09-03T13:12:37")
        self.assertEqual(t06["dataset_variant"]["total_images"], 154)
        self.assertNotIn("AZ", reader.snapshot(2))

    def test_az_reuse_is_not_marked_as_created_in_current_iteration(self):
        self.registry["iteration_state"]["2"] = self.registry["iteration_state"]["1"]
        self._save_registry()
        event = dict(self.event, iteration=2, created_at=self.t2, action="approve_step3", transition_id="e3_to_e4")
        with patch.object(history_ui, "_load_project_training_runs", return_value=[]):
            row = history_ui._build_project_product_rows("demo", [event])[0]
        self.assertIn("źródło IT1", row["artifact"])
        self.assertIn("Bez nowych AZ", row["increment"])

    def test_snapshot_does_not_enumerate_image_directories(self):
        with patch.object(Path, "iterdir", side_effect=AssertionError("directory scan")), \
             patch.object(Path, "rglob", side_effect=AssertionError("recursive scan")):
            self.assertEqual(self._reader().snapshot(1, cutoff=self.t1)["AT"]["plates"], 10)

    def test_count_parser_preserves_unknown_and_key_priority(self):
        entry = {"images": 999, "artifacts": {"approved_images": None}, "resources": {"approved_images": 10}}
        self.assertEqual(history_ui._first_int_value(entry, ("approved_images", "images")), 10)
        self.assertIsNone(history_ui._first_int_value({"approved_images": None}, ("approved_images",)))

    def test_recording_does_not_write_zero_for_pending_approved_run(self):
        write_json(self.state / "plate_approved_set.json", {"entries": {}})
        with patch.object(graph.CAMPAIGN, "get_active_project_name", return_value="demo"), \
             patch.object(graph.CAMPAIGN, "get_current_iteration_num", return_value=1), \
             patch.object(graph.CAMPAIGN, "append_project_history_event") as append:
            graph._record_graph_action_history("approve_step2", {}, graph.CampaignGraphActionResult(True, "approve_step2", "OK"))
        saved = append.call_args.kwargs
        self.assertEqual(saved["artifacts"]["approved_images"], 10)
        self.assertEqual(saved["artifacts"]["resource_snapshot"]["AT"]["iteration_images"], 10)

    def test_approval_record_keeps_iteration_before_handler_advances(self):
        with patch.object(graph.CAMPAIGN, "get_active_project_name", return_value="demo"), \
             patch.object(graph.CAMPAIGN, "get_current_iteration_num", side_effect=[1, 2]), \
             patch.object(graph.CAMPAIGN, "get_current_step", side_effect=[4, 1]), \
             patch.object(graph, "_execute_approve_step4", return_value=graph.CampaignGraphActionResult(True, "approve_step4", "OK")), \
             patch.object(graph.CAMPAIGN, "append_project_history_event") as append:
            graph.execute_campaign_graph_action(object(), "approve_step4")
        self.assertEqual(append.call_args.kwargs["iteration_num"], 1)
        self.assertEqual(append.call_args.kwargs["status"], "ok")

    def test_model_rows_are_not_limited_to_last_twenty_runs(self):
        model = self.root / "best.pt"
        model.write_bytes(b"fixture")
        runs = {str(i): {"id": str(i), "iteration": 1, "training_target": "char", "status": "completed",
                         "best_weights": str(model), "finished_at": f"2026-09-03T12:{i:02d}:00"} for i in range(25)}
        runs["pending"] = {"id": "pending", "iteration": 1, "training_target": "plate", "status": "pending"}
        write_json(self.root / "5_training_runs" / "training_history.json", {"runs": runs})
        rows = history_ui._build_project_product_rows("demo", [])
        self.assertEqual(len(rows), 25)
        self.assertTrue(all(row["code"] == "MZ" for row in rows))

    def test_dataset_metadata_takes_priority_over_misleading_run_name(self):
        dataset = self.root / "plate_project" / "dataset"
        dataset.mkdir(parents=True)
        (dataset / "data.yaml").write_text("names: ['A', 'B', '0']\n", encoding="utf-8")
        self.assertEqual(history_ui._training_run_target({"name": "pose_rescue", "dataset_path": str(dataset)}), "char")

    def test_training_iteration_uses_start_not_finish_date(self):
        entries = [dict(self.event, action="approve_step4", iteration=i, created_at=t, details={"path": "char_from_images"})
                   for i, t in ((1, "2026-09-03T12:00:00"), (2, "2026-09-05T12:00:00"))]
        run = {"created_at": "2026-09-03T11:00:00", "finished_at": "2026-09-04T12:00:00"}
        iteration = history_ui._infer_training_run_iteration_from_history(run, target="char", history_entries=entries, target_by_iteration={1: "char", 2: "char"})
        self.assertEqual(iteration, 1)

    def test_demo_char_route_description_does_not_turn_it_into_plate_training(self):
        selected = dict(self.event, action="set_iteration_path", transition_id="", title="Wybrano ścieżkę E1: Model znaków: brakujące anotacje tablic.", artifacts={})
        approved = dict(self.event, action="approve_step1", details={"path": "char_from_images"})
        entries = [selected, approved]
        self.assertEqual(history_ui._event_iteration_target(selected), "char")
        self.assertEqual(history_ui._iteration_target_map(entries), {1: "char"})
        self.assertEqual(history_ui._iteration_path_map(entries), {1: "char_from_images"})

    def test_explicit_route_beats_conflicting_legacy_description(self):
        selected = dict(self.event, action="set_iteration_path", transition_id="", title="Model tablic")
        approved = dict(self.event, action="approve_step1", details={"path": "char_from_images"})
        self.assertEqual(history_ui._iteration_target_map([selected, approved]), {1: "char"})

    def test_path_selection_records_structured_path_not_only_description(self):
        with patch.object(graph.CAMPAIGN, "get_active_project_name", return_value="demo"), \
             patch.object(graph.CAMPAIGN, "get_current_iteration_num", return_value=1), \
             patch.object(graph.CAMPAIGN, "append_project_history_event") as append:
            graph._record_graph_action_history("set_iteration_path", {"path": "char_from_images"}, graph.CampaignGraphActionResult(True, "set_iteration_path", "OK"))
        self.assertEqual(append.call_args.kwargs["details"]["path"], "char_from_images")

    def test_prepared_variant_is_separate_from_dataset_actually_used_by_model(self):
        self.registry["iteration_state"]["1"]["step4_training"] = {
            "run_id": "run1", "target": "char", "trained_iteration": 1,
        }
        self._save_registry()
        write_json(self.root / "5_training_runs" / "training_history.json", {"runs": {"run1": {
            "status": "completed", "dataset_path": "base_split", "best_weights": "best.pt",
            "created_at": "2026-09-03T11:53:00", "finished_at": "2026-09-03T11:55:00",
        }}})
        before = self._reader().snapshot(1, cutoff=self.t1)
        self.assertNotIn("training_run", before)
        self.assertNotIn("MZ", before)
        after = self._reader().snapshot(1, cutoff="2026-09-03T13:12:37")
        self.assertEqual(after["dataset_variant"]["total_images"], 154)
        self.assertEqual(after["training_run"]["dataset_path"], "base_split")
        self.assertEqual(after["MZ"]["source_iteration"], 1)
        self.assertNotIn("MT", after)

    def test_incomplete_legacy_registry_does_not_crash_history(self):
        write_json(self.state / "plate_approved_set.json", {"entries": []})
        write_json(self.state / "artifact_registry.json", {"packages": [], "iteration_index": {}, "iteration_state": {"1": []}})
        result = self._reader().for_event(self.event)
        self.assertNotIn("AT", result)
        self.assertEqual(result["O"]["images"], 1000)


if __name__ == "__main__":
    unittest.main()
