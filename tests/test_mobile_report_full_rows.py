import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile

from auto_annotation_tool.gui.z4_mobile_report_browser import MobileReportBrowser
from auto_annotation_tool.ranking import (
    ExperimentSessionRecord,
    iter_full_event_rows,
    iter_full_frame_flow_rows,
    iter_full_sample_rows,
    iter_full_thermal_rows,
    iter_full_trace_rows,
    MobileBenchmarkReport,
    MobilePackageExperimentStore,
    read_mobile_report_bundle,
    read_mobile_report_bundles,
)
from auto_annotation_tool.ranking.mobile_package_experiments import MOBILE_BENCHMARK_REPORT_SCHEMA


def _report_payload(report_id: str = "report-001") -> dict:
    return {
        "schema": MOBILE_BENCHMARK_REPORT_SCHEMA,
        "report_id": report_id,
        "package_id": "pkg-mt-mz",
        "variant_id": "tflite-fp32",
        "measured_at": "2026-08-27T20:00:00Z",
        "device": {"name": "test-device", "android_version": "15"},
        "runtime": "tflite",
        "delegate": "cpu",
        "latency": {"pipeline_p95_ms": 123.4},
        "memory": {"ram_peak_mb": 321.0},
        "quality": {"exact_match": 0.91, "cer": 0.03},
    }


def _csv_rows(columns: tuple[str, ...], total: int) -> str:
    lines = [",".join(columns)]
    for index in range(total):
        values = []
        for column in columns:
            if column.endswith("_id"):
                values.append(f"{column}-{index}")
            elif column.endswith("_ms"):
                values.append(str(index * 10))
            else:
                values.append(str(index))
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


def _write_report_zip(path: Path, *, total: int = 12000) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.json", json.dumps(_report_payload(), ensure_ascii=False))
        archive.writestr("traces.csv", _csv_rows(("frame_id", "timestamp_ms", "status", "text"), total))
        archive.writestr("thermal.csv", _csv_rows(("timestamp_ms", "cpu_c", "battery_c"), total))
        archive.writestr("frame_flow.csv", _csv_rows(("frame_id", "stage", "elapsed_ms"), total))
        archive.writestr(
            "events.jsonl",
            "".join(
                json.dumps({"event_id": index, "kind": "frame", "timestamp_ms": index * 10}) + "\n"
                for index in range(total)
            ),
        )
        archive.writestr("samples/index.csv", _csv_rows(("sample_id", "frame_id", "gt"), total))


class MobileReportFullRowsTests(unittest.TestCase):
    def test_full_zip_sources_are_not_limited_by_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "android-report.alprsession"
            _write_report_zip(path)

            bundle = read_mobile_report_bundle(path, max_trace_rows=5000)

            self.assertEqual(bundle.trace_total, 12000)
            self.assertEqual(len(bundle.trace_rows), 5000)
            self.assertEqual(bundle.thermal_total, 12000)
            self.assertEqual(len(bundle.thermal_rows), 5000)
            self.assertEqual(bundle.frame_flow_total, 12000)
            self.assertEqual(len(bundle.frame_flow_rows), 5000)
            self.assertEqual(bundle.event_total, 12000)
            self.assertEqual(len(bundle.event_rows), 5000)
            self.assertEqual(bundle.sample_total, 12000)
            self.assertEqual(len(bundle.sample_rows), 1000)
            self.assertEqual(sum(1 for _ in iter_full_trace_rows(bundle)), 12000)
            self.assertEqual(sum(1 for _ in iter_full_thermal_rows(bundle)), 12000)
            self.assertEqual(sum(1 for _ in iter_full_frame_flow_rows(bundle)), 12000)
            self.assertEqual(sum(1 for _ in iter_full_event_rows(bundle)), 12000)
            self.assertEqual(sum(1 for _ in iter_full_sample_rows(bundle)), 12000)

    def test_pipeline_manifests_build_model_provenance_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "android-report.alprsession"
            plate_manifest = {
                "schema": "alpr.model.v1",
                "model_id": "mt-test",
                "role": "plate",
                "task": "pose",
                "model": {"architecture_label": "YOLO26n Pose"},
                "source": {"checkpoint_sha256": "checkpoint-sha", "parameter_count": 2446959},
                "training": {
                    "provenance_version": 2,
                    "run_id": "20260904_120000",
                    "run_epochs_completed": 10,
                    "total_epochs": 40,
                    "total_epochs_known": True,
                    "known_epochs_minimum": 40,
                    "total_epochs_scope": "project_training_after_pretrained_base",
                    "lineage_total_epochs": 40,
                    "lineage_total_epochs_known": True,
                    "lineage_stage_count": 2,
                    "lineage_stage_count_known": True,
                    "known_stage_count_minimum": 2,
                    "run_train_images": 900,
                    "run_nominal_sample_presentations": 9000,
                    "lineage_nominal_sample_presentations": 32000,
                    "sample_presentations_known": True,
                    "known_sample_presentations_minimum": 32000,
                    "best_epoch_source": "checkpoint",
                    "provenance_capture": "frozen_at_training_start",
                    "provenance_status": "complete",
                    "dataset": {
                        "dataset_id": "DS-MT-ABC",
                        "manifest_sha256": "manifest-sha",
                        "split_sha256": "split-sha",
                    },
                },
                "metrics": {"best_map50": 0.9},
                "variants": [
                    {
                        "id": "tflite-int8",
                        "runtime": "tflite",
                        "precision": "int8",
                        "sha256": {"model.tflite": "variant-sha"},
                    }
                ],
            }
            model_refs = {
                "plate": {
                    "model_id": "mt-test",
                    "installed_model_fingerprint": "installed-sha",
                    "checkpoint_sha256": "checkpoint-sha",
                    "package_sha256": "package-sha",
                    "variant_id": "tflite-int8",
                    "variant_artifact_sha256": ["variant-sha"],
                }
            }
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("report.json", json.dumps(_report_payload(), ensure_ascii=False))
                archive.writestr("pipeline/plate_manifest.json", json.dumps(plate_manifest, ensure_ascii=False))
                archive.writestr("pipeline/model_refs.json", json.dumps(model_refs, ensure_ascii=False))

            bundle = read_mobile_report_bundle(path, max_trace_rows=10)
            provenance = bundle.report_payload["model_provenance"]["plate"]

            self.assertEqual(bundle.pipeline_manifests["plate"]["model_id"], "mt-test")
            self.assertEqual(bundle.model_refs["plate"]["package_sha256"], "package-sha")
            self.assertEqual(provenance["training"]["total_epochs"], 40)
            self.assertEqual(provenance["training"]["lineage_stage_count"], 2)
            self.assertTrue(provenance["training"]["lineage_stage_count_known"])
            self.assertEqual(provenance["training"]["known_stage_count_minimum"], 2)
            self.assertEqual(provenance["training"]["run_train_images"], 900)
            self.assertEqual(provenance["training"]["run_nominal_sample_presentations"], 9000)
            self.assertEqual(provenance["training"]["lineage_nominal_sample_presentations"], 32000)
            self.assertTrue(provenance["training"]["sample_presentations_known"])
            self.assertEqual(provenance["training"]["known_sample_presentations_minimum"], 32000)
            self.assertEqual(provenance["training"]["best_epoch_source"], "checkpoint")
            self.assertEqual(provenance["training"]["provenance_capture"], "frozen_at_training_start")
            self.assertEqual(provenance["training"]["dataset"]["dataset_id"], "DS-MT-ABC")
            self.assertEqual(provenance["variant_id"], "tflite-int8")
            self.assertEqual(provenance["variant_artifact_sha256"], ["variant-sha"])

    def test_json_file_can_hold_many_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reports.json"
            payload = {
                "reports": [
                    dict(_report_payload("report-a"), traces=[{"frame_id": "a-1"}]),
                    dict(_report_payload("report-b"), traces=[{"frame_id": "b-1"}]),
                    dict(_report_payload("report-c"), traces=[{"frame_id": "c-1"}]),
                ]
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            bundles = read_mobile_report_bundles(path)

            self.assertEqual(len(bundles), 3)
            self.assertEqual([bundle.report.report_id for bundle in bundles], ["report-a", "report-b", "report-c"])
            self.assertEqual(sum(1 for _ in iter_full_trace_rows(path)), 3)

    def test_report_store_keeps_replicas_but_deduplicates_same_source_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MobilePackageExperimentStore(Path(tmp))
            payload_a = _report_payload("replica-a")
            payload_a["source_archive_sha256"] = "sha-a"
            payload_a["experiment_index"] = {
                "series_id": "series-1",
                "scenario_id": "scenario-1",
                "experiment_variant": "variant-a",
                "replicate_index": 1,
            }
            payload_b = _report_payload("replica-b")
            payload_b["source_archive_sha256"] = "sha-b"
            payload_b["experiment_index"] = {
                "series_id": "series-1",
                "scenario_id": "scenario-1",
                "experiment_variant": "variant-a",
                "replicate_index": 2,
            }

            report_a = MobileBenchmarkReport.from_dict(payload_a)
            report_b = MobileBenchmarkReport.from_dict(payload_b)
            store.add_report(report_a, save=False)
            store.add_report(report_b, save=False)
            store.add_report(report_a, save=False)

            self.assertEqual(len(store.reports), 2)
            self.assertEqual({report.report_id for report in store.reports}, {"replica-a", "replica-b"})

    def test_comparison_guard_reports_android_version_and_model_fingerprints(self):
        payload_a = _report_payload("guard-a")
        payload_a["device"] = {"name": "phone-a", "android_version": "12"}
        payload_a["experiment_index"] = {
            "series_id": "series-1",
            "scenario_id": "scenario-1",
            "model_fingerprints": {
                "mt": {"sha256": "mt-sha"},
                "mz": {"sha256": "mz-sha-a"},
                "mp": {"sha256": "mp-sha"},
            },
        }
        payload_b = _report_payload("guard-b")
        payload_b["device"] = {"name": "phone-a", "android_version": "13"}
        payload_b["experiment_index"] = {
            "series_id": "series-1",
            "scenario_id": "scenario-1",
            "model_fingerprints": {
                "mt": {"sha256": "mt-sha"},
                "mz": {"sha256": "mz-sha-b"},
                "mp": {"sha256": "mp-sha"},
            },
        }
        reports = [MobileBenchmarkReport.from_dict(payload_a), MobileBenchmarkReport.from_dict(payload_b)]

        class FakeTable:
            def __init__(self):
                self.rows = []

            def set_rows(self, rows):
                self.rows = list(rows)

        browser = object.__new__(MobileReportBrowser)
        browser.store = SimpleNamespace(reports=reports)
        browser.comparison_table = FakeTable()
        browser._populate_comparison_guard(reports[0], None)

        status_by_label = {row[0]: row[3] for row in browser.comparison_table.rows}
        self.assertEqual(status_by_label["Android version"], "Różne")
        self.assertEqual(status_by_label["MT fingerprint"], "Spójne")
        self.assertEqual(status_by_label["MZ fingerprint"], "Różne")

    def test_android_contract_roles_are_normalized_to_mp_mt_mz(self):
        payload = _report_payload("android-contract")
        payload["execution"] = {
            "vehicle": {"sha256": "mp-sha-1234567890"},
            "plate": {"sha256": "mt-sha-1234567890"},
            "character": {"sha256": "mz-sha-1234567890"},
        }
        payload["runtime_composition"] = {
            "models": {
                "vehicle": {"sha256": "mp-sha-should-not-override"},
                "plate": {"sha256": "mt-sha-should-not-override"},
                "character": {"sha256": "mz-sha-should-not-override"},
            }
        }
        report = MobileBenchmarkReport.from_dict(payload)
        record = ExperimentSessionRecord.from_report(report)

        self.assertEqual(record.model_fingerprints["mp"]["sha256"], "mp-sha-1234567890")
        self.assertEqual(record.model_fingerprints["mt"]["sha256"], "mt-sha-1234567890")
        self.assertEqual(record.model_fingerprints["mz"]["sha256"], "mz-sha-1234567890")

        browser = object.__new__(MobileReportBrowser)
        self.assertEqual(browser._guard_model_fingerprint_value(report, {}, "mp"), "mp-sha-1234567890")
        self.assertEqual(browser._guard_model_fingerprint_value(report, {}, "mt"), "mt-sha-1234567890")
        self.assertEqual(browser._guard_model_fingerprint_value(report, {}, "mz"), "mz-sha-1234567890")

    def test_missing_experiment_ids_are_not_auto_generated_for_guard(self):
        report = MobileBenchmarkReport.from_dict(_report_payload("no-index"))
        record = ExperimentSessionRecord.from_report(report)

        self.assertEqual(record.series_id, "")
        self.assertEqual(record.scenario_id, "")
        self.assertEqual(record.replicate_index, 0)

        class FakeTable:
            def __init__(self):
                self.rows = []

            def set_rows(self, rows):
                self.rows = list(rows)

        browser = object.__new__(MobileReportBrowser)
        browser.store = SimpleNamespace(reports=[report])
        browser.comparison_table = FakeTable()
        browser._populate_comparison_guard(report, None)

        self.assertEqual(browser.comparison_table.rows[0][0], "Indeks eksperymentu")
        self.assertEqual(browser.comparison_table.rows[0][3], "Brak serii")

    def test_app_git_sha_and_app_version_are_separate_guard_rows(self):
        payload = _report_payload("app-build")
        payload["app_version"] = "1.2.3"
        payload["experiment_index"] = {
            "series_id": "series-1",
            "scenario_id": "scenario-1",
        }
        report = MobileBenchmarkReport.from_dict(payload)

        class FakeTable:
            def __init__(self):
                self.rows = []

            def set_rows(self, rows):
                self.rows = list(rows)

        browser = object.__new__(MobileReportBrowser)
        browser.store = SimpleNamespace(reports=[report])
        browser.comparison_table = FakeTable()
        browser._populate_comparison_guard(report, None)
        by_label = {row[0]: row for row in browser.comparison_table.rows}

        self.assertEqual(by_label["Build aplikacji"][1], "-")
        self.assertEqual(by_label["Build aplikacji"][3], "Brak danych")
        self.assertEqual(by_label["Wersja aplikacji"][1], "1.2.3")


if __name__ == "__main__":
    unittest.main()
