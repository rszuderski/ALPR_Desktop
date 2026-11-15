"""Persistence/service tests on synthetic PNGs and byte-only fake checkpoints.

No YOLO class is imported, no checkpoint is deserialized, and no real inference
or user approval is performed. Production metrics, image decoder, file ledger,
source hashes, loader and report exporter remain active.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from auto_annotation_tool.image_orientation import read_oriented_bgr
from auto_annotation_tool.ranking import mt_eval_runner as runner
from auto_annotation_tool.ranking.mt_eval_report import export_mt_report, load_mt_report
from auto_annotation_tool.ranking.eval396 import digest


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _quad(x):
    return [[x, 10.0], [x + 30, 10.0], [x + 30, 25.0], [x, 25.0]]


def build_runner_env(tmp_path, monkeypatch):
    """Reusable fixture for crash-recovery tests; only external gates are mocked."""
    experiment = tmp_path / "synthetic_experiment"
    selection = experiment / "evaluation_selections" / "synthetic_selection"
    selection.mkdir(parents=True)
    images = tmp_path / "synthetic_images"
    images.mkdir()
    models = []
    receipt = {}
    for label in runner.MODEL_LABELS:
        path = tmp_path / (label + ".pt")
        path.write_bytes(("SYNTHETIC TEST BYTES - NEVER DESERIALIZE " + label).encode())
        receipt[str(path)] = _sha(path)
        models.append({"label": label, "role": "plate", "run_id": "fixture-" + label,
                       "checkpoint_file": str(path), "checkpoint_sha256": receipt[str(path)]})
    scenes = []
    paths = []
    for index, domain in enumerate(("DAY", "DAY", "NIGHT")):
        path = images / f"scene_{index}.png"
        Image.new("RGB", (160, 96), (20 * (index + 1),) * 3).save(path)
        receipt[str(path)] = _sha(path)
        pixels = read_oriented_bgr(path)
        plates = []
        if index < 2:
            plates.append({"plate_id": "regular_" + str(index), "source_annotation_id": "fixture-regular-" + str(index),
                           "bbox": [10, 10, 40, 25], "quad": _quad(10), "bbox_eligible": True, "pose_eligible": True,
                           "ocr_eligible": True, "pose_exclusion": None, "text_decision": "approved", "fixture_gt_secret": "MUST_NOT_REACH_BACKEND"})
        if index == 1:
            plates.append({"plate_id": "plate_000421", "source_annotation_id": "fixture-excluded", "bbox": [60, 10, 90, 25],
                           "quad": list(reversed(_quad(60))), "bbox_eligible": True, "pose_eligible": False,
                           "ocr_eligible": False, "pose_exclusion": {"reason_code": "TEST_ONLY"},
                           "pose_evaluation_status": "EXCLUDED_BY_OPERATOR", "text_decision": "excluded_unreadable"})
        scenes.append({"sha256": receipt[str(path)], "file": path.name, "domain": domain,
                       "image": {"oriented_size": [160, 96], "decoded_pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()},
                       "plates": plates})
        paths.append(path)
    source_marker = tmp_path / "synthetic_source_metadata.json"
    source_marker.write_text('{"fixture":true}')
    receipt[str(source_marker)] = _sha(source_marker)
    code_marker = tmp_path / "synthetic_code_marker.py"
    code_marker.write_text("# test-only code receipt\n")
    counts = {"scenes": 3, "bbox": 3, "pose": 2, "ocr": 2, "pose_exclusions": 1,
              "per_domain": {"ALL": {"scenes": 3, "bbox": 3, "pose": 2, "ocr": 2},
                             "DAY": {"scenes": 2, "bbox": 3, "pose": 2, "ocr": 2},
                             "NIGHT": {"scenes": 1, "bbox": 0, "pose": 0, "ocr": 0}}}
    reference = {"schema": "TEST_ONLY_SYNTHETIC_PREFLIGHT", "selection_dir": str(selection),
                 "freeze_dir": str(tmp_path / "synthetic_freeze"), "source_scene_root": str(images),
                 "selection_sha": "fixture-selection", "gt_freeze_sha": "fixture-freeze",
                 "metric_policy": {"id": "TEST_ONLY_SYNTHETIC_POLICY"}, "metric_policy_sha256": "fixture-policy",
                 "metric_counts": counts, "models": models, "scenes": scenes, "source_files": receipt,
                 "review_sha256": "TEST_ONLY_REVIEW_NOT_AN_OPERATOR_APPROVAL"}
    decision = {"schema": "TEST_ONLY_SYNTHETIC_APPROVAL", "decision": "APPROVED",
                "operator_signature": "TEST FIXTURE - NO HUMAN APPROVAL", "review_sha256": reference["review_sha256"]}
    decision["contract_sha256"] = digest(decision)
    approval = tmp_path / "synthetic_approval"
    approval.mkdir()
    for filename, payload in (("geometry_policy.json", decision), ("preflight.json", reference),
                              ("metric_eligibility_policy.json", reference["metric_policy"]), ("status.json", {"status": "TEST_ONLY"})):
        (approval / filename).write_text(json.dumps(payload), encoding="utf-8")
    env = SimpleNamespace(reference=reference, decision=decision, source_paths=paths, source_marker=source_marker,
                          code_marker=code_marker, backend_calls=[], event_log=[], seed_calls=[], inspection_calls=[],
                          active_backends=0, max_active_backends=0, after_predict=None, fail_on_call=None,
                          effective_override={}, runtime={"device": "cpu", "ultralytics": "8.4.19", "fixture_only": True})

    def approval_gate(report, signed):
        if signed.get("decision") != "APPROVED" or signed.get("operator_signature") != "TEST FIXTURE - NO HUMAN APPROVAL":
            raise ValueError("EXIF_OPERATOR_APPROVAL_REQUIRED")
        if signed.get("contract_sha256") != digest({key: value for key, value in signed.items() if key != "contract_sha256"}):
            raise ValueError("OPERATOR_CONTRACT_CHANGED")
        if report["review_sha256"] != signed["review_sha256"]:
            raise ValueError("STALE_GEOMETRY_APPROVAL")
        return True

    def fixture_geometry_binding(protocol, report, signed, source_receipt, reader):
        approval_gate(report, signed)
        assert report["schema"] == "TEST_ONLY_SYNTHETIC_PREFLIGHT"
        assert protocol["geometry_review_sha256"] == report["review_sha256"]
        assert protocol["geometry_approval_contract_sha256"] == signed["contract_sha256"]
        assert protocol["metric_counts"] == report["metric_counts"]
        assert protocol["source_receipt_sha256"] == digest(source_receipt)
        runner._validate_protocol_parameters(protocol)

    def inspect_fake(model):
        assert Path(model["checkpoint_file"]).read_bytes().startswith(b"SYNTHETIC TEST BYTES")
        env.inspection_calls.append(model["label"])
        return {**deepcopy(model), "task": "pose", "kpt_shape": [4, 2], "names": {"0": "plate"},
                "native_end2end": True, "stored_parameter_dtype": "torch.float32"}

    @contextmanager
    def deterministic_fake(_protocol):
        yield

    class FakeBackend:
        def __init__(self, model, protocol):
            self.model, self.protocol = model, protocol
            env.active_backends += 1
            env.max_active_backends = max(env.max_active_backends, env.active_backends)
            env.event_log.append(("load", model["label"]))

        def predict(self, pixels):
            # The signature itself rejects file paths, metadata and extra GT args.
            assert isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8 and pixels.shape == (96, 160, 3)
            scene_index = int(pixels[0, 0, 0]) // 20 - 1
            env.backend_calls.append((self.model["label"], scene_index))
            env.event_log.append(("predict", self.model["label"], scene_index))
            if env.fail_on_call == len(env.backend_calls):
                raise RuntimeError("SYNTHETIC_BACKEND_FAILURE")
            if scene_index == 0:
                predictions = []
            elif scene_index == 1:
                predictions = [
                    {"index": 7, "bbox": [10., 10., 40., 25.], "confidence": .923456789, "class_id": 0,
                     "keypoints": list(reversed(_quad(10)))},
                    {"index": 2, "bbox": [60., 10., 90., 25.], "confidence": .812345678, "class_id": 0, "keypoints": _quad(60)},
                    {"index": 9, "bbox": [10., 10., 40., 25.], "confidence": .333333333, "class_id": 0, "keypoints": _quad(10)},
                ]
            else:
                predictions = [{"index": 4, "bbox": [20., 20., 30., 30.], "confidence": .7654321,
                                "class_id": 0, "keypoints": None}]
            actual = {**deepcopy(self.protocol["inference"]), "device": "cpu", "checked_imgsz": [512, 512],
                      "backend_fp16": False, "backend_end2end": True, "backend_kpt_shape": [4, 2],
                      "backend_names": {"0": "plate"}, "fused_backend": True,
                      "determinism": {"deterministic_algorithms": True, "warn_only": True, "cudnn_benchmark": False,
                                      "cudnn_deterministic": True, "matmul_tf32": False, "cudnn_tf32": False,
                                      "cublas_workspace_config": ":4096:8"}}
            actual.update(deepcopy(env.effective_override))
            if env.after_predict:
                env.after_predict(self.model["label"], scene_index)
            return {"predictions": deepcopy(predictions), "effective_inference": actual,
                    "speed_ms": {"preprocess": .1, "inference": .2, "postprocess": .3}}

        def close(self):
            env.active_backends -= 1
            env.event_log.append(("close", self.model["label"]))

    monkeypatch.setattr(runner, "preflight_mt_geometry", lambda *_args, **_kwargs: deepcopy(reference))
    monkeypatch.setattr(runner, "require_current_approval", approval_gate)
    monkeypatch.setattr(runner, "_verify_campaign_geometry", fixture_geometry_binding, raising=False)
    monkeypatch.setattr(runner, "inspect_checkpoint", inspect_fake)
    monkeypatch.setattr(runner, "_runtime", lambda _device: deepcopy(env.runtime))
    monkeypatch.setattr(runner, "_code_receipt", lambda: {str(code_marker): _sha(code_marker)})
    monkeypatch.setattr(runner, "_deterministic_runtime", deterministic_fake)
    monkeypatch.setattr(runner, "_seed_scene", lambda index: env.seed_calls.append(42 + index))
    monkeypatch.setattr(runner, "UltralyticsBackend", FakeBackend)
    env.prepared = runner.prepare_campaign(selection, approval_dir=approval, device="cpu")
    return env


@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    return build_runner_env(tmp_path, monkeypatch)


def test_prepare_inspects_fake_contract_without_predicting(runner_env):
    assert runner_env.inspection_calls == list(runner.MODEL_LABELS)
    assert runner_env.backend_calls == []
    assert not Path(runner_env.prepared["output_root"]).exists()
    assert runner_env.prepared["protocol"]["inference"]["data"] is None


def test_three_models_are_sequential_raw_indices_and_empty_scenes_survive(runner_env):
    result = runner.run_campaign(runner_env.prepared)
    assert result["status"] == "COMPLETE"
    assert runner_env.backend_calls == [(label, index) for label in runner.MODEL_LABELS for index in range(3)]
    assert runner_env.seed_calls == [42, 43, 44] * 3
    assert runner_env.max_active_backends == 1 and runner_env.active_backends == 0
    for model in result["models"]:
        assert len(model["rows"]) == 3 and model["rows"][0]["predictions"] == []
        assert [p["index"] for p in model["rows"][1]["predictions"]] == [7, 2, 9]
        assert model["rows"][1]["predictions"][0]["confidence"] == .923456789
        assert model["rows"][1]["predictions"][0]["keypoints"] == list(reversed(_quad(10)))
        all_metrics = model["metrics"]["domains"]["ALL"]
        assert (all_metrics["box_tp"], all_metrics["box_fp"], all_metrics["box_fn"]) == (2, 2, 1)
        assert all_metrics["pose_gt"] == 2 and all_metrics["pose_excluded_matched"] == 1
        assert all_metrics["corner_error_px"]["n"] == 4 and all_metrics["corner_error_px"]["mean"] > 0


def test_cancel_resume_skips_recorded_scenes_and_complete_run(runner_env):
    partial = runner.run_campaign(runner_env.prepared, cancelled=lambda: len(runner_env.backend_calls) >= 2)
    assert partial["status"] == "PARTIAL"
    assert len(partial["models"][0]["rows"]) == 2
    assert partial["models"][0]["metrics"]["domains"]["ALL"]["pending_scenes"] == 1
    result = runner.run_campaign(runner_env.prepared, resume_dir=partial["run_dir"])
    assert result["status"] == "COMPLETE"
    assert runner_env.backend_calls == [(label, index) for label in runner.MODEL_LABELS for index in range(3)]
    before = list(runner_env.backend_calls)
    again = runner.run_campaign(runner_env.prepared, resume_dir=result["run_dir"])
    assert again["status"] == "COMPLETE" and runner_env.backend_calls == before


def test_cancelled_before_first_scene_is_not_no_detection(runner_env):
    result = runner.run_campaign(runner_env.prepared, cancelled=lambda: True)
    assert result["status"] == "PARTIAL" and runner_env.backend_calls == []
    assert all(model["rows"] == [] and model["metrics"] is None for model in result["models"])


def test_backend_error_keeps_valid_prefix_and_resume_retries_only_failed_scene(runner_env):
    runner_env.fail_on_call = 2
    partial = runner.run_campaign(runner_env.prepared)
    assert partial["status"] == "PARTIAL"
    assert "SYNTHETIC_BACKEND_FAILURE" in partial["status_detail"]["error"]
    assert len(partial["models"][0]["rows"]) == 1
    assert runner_env.active_backends == 0
    runner_env.fail_on_call = None
    complete = runner.run_campaign(runner_env.prepared, resume_dir=partial["run_dir"])
    assert complete["status"] == "COMPLETE"
    assert runner_env.backend_calls.count(("MT-n", 0)) == 1
    assert runner_env.backend_calls.count(("MT-n", 1)) == 2
    assert len(runner_env.backend_calls) == 10


@pytest.mark.parametrize("kind", ["image", "metadata", "checkpoint", "code", "runtime"])
def test_source_or_runtime_changed_before_start_never_predicts(runner_env, kind):
    if kind == "runtime":
        runner_env.runtime["ultralytics"] = "changed"
    else:
        path = {"image": runner_env.source_paths[0], "metadata": runner_env.source_marker,
                "checkpoint": Path(runner_env.prepared["protocol"]["models"][0]["checkpoint_file"]),
                "code": runner_env.code_marker}[kind]
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="CHANGED"):
        runner.run_campaign(runner_env.prepared)
    assert runner_env.backend_calls == []


def test_image_changed_during_prediction_is_not_published(runner_env):
    def mutate(_label, index):
        path = runner_env.source_paths[index]
        path.write_bytes(path.read_bytes() + b"changed after decode")
    runner_env.after_predict = mutate
    result = runner.run_campaign(runner_env.prepared)
    assert result["status"] == "PARTIAL"
    assert len(runner_env.backend_calls) == 1
    assert all(model["rows"] == [] for model in result["models"])
    assert "INPUT_CHANGED" in result["status_detail"]["error"]


def test_metadata_changed_during_run_cannot_be_complete(runner_env):
    def mutate(_label, _index):
        runner_env.source_marker.write_text("changed metadata")
    runner_env.after_predict = mutate
    result = runner.run_campaign(runner_env.prepared)
    assert result["status"] == "PARTIAL"
    assert "SOURCE_OR_CODE_CHANGED" in result["status_detail"]["error"]


def test_unsigned_approval_blocks_before_backend(runner_env):
    prepared = deepcopy(runner_env.prepared)
    prepared["decision"]["decision"] = "REJECTED"
    with pytest.raises(ValueError, match="APPROVAL_REQUIRED"):
        runner.run_campaign(prepared)
    assert runner_env.backend_calls == []


@pytest.mark.parametrize("name,value", [("conf", .5), ("rect", True), ("half", True), ("imgsz", 640), ("data", "unexpected.yaml")])
def test_rehashed_parameter_changes_are_still_rejected(runner_env, name, value):
    prepared = deepcopy(runner_env.prepared)
    prepared["protocol"]["inference"][name] = value
    prepared["protocol"]["protocol_sha256"] = digest(runner._without(prepared["protocol"], "protocol_sha256"))
    with pytest.raises(ValueError, match="FIXED_PARAMETERS_CHANGED"):
        runner.run_campaign(prepared)
    assert runner_env.backend_calls == []


def test_tampered_saved_row_is_rejected_by_loader_and_export(runner_env):
    result = runner.run_campaign(runner_env.prepared)
    raw = next((Path(result["run_dir"]) / "MT-n/raw").glob("*.json"))
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA"):
        runner.load_campaign(result["run_dir"])
    with pytest.raises(ValueError, match="SHA"):
        load_mt_report(result["run_dir"])


def test_effective_parameter_change_is_detected_even_with_rehashed_row(runner_env):
    partial = runner.run_campaign(runner_env.prepared, cancelled=lambda: len(runner_env.backend_calls) >= 1)
    path = Path(partial["run_dir"])
    raw = next((path / "MT-n/raw").glob("*.json"))
    payload = json.loads(raw.read_text())
    payload["effective_inference"]["rect"] = True
    payload["record_sha256"] = digest(runner._without(payload, "record_sha256"))
    raw_sha = runner._write_json(raw, payload)
    ledger_path = path / "MT-n/progress.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["records"][payload["scene_sha256"]] = raw_sha
    runner._write_json(ledger_path, ledger)
    with pytest.raises(ValueError, match="EFFECTIVE_PARAMETER_CHANGED"):
        runner.load_campaign(path)


def test_real_loader_to_report_export_complete_and_partial(runner_env):
    partial = runner.run_campaign(runner_env.prepared, cancelled=lambda: len(runner_env.backend_calls) >= 1)
    partial_report = export_mt_report(partial["run_dir"], Path(partial["run_dir"]) / "reports")
    assert json.loads((partial_report / "status.json").read_text())["status"] == "PARTIAL"
    complete = runner.run_campaign(runner_env.prepared, resume_dir=partial["run_dir"])
    complete_report = export_mt_report(complete["run_dir"], Path(complete["run_dir"]) / "reports")
    assert json.loads((complete_report / "status.json").read_text())["status"] == "COMPLETE"
    provenance = json.loads((complete_report / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["geometry_approval"] == complete["geometry_approval"]
    assert provenance["measurement_source_receipt"] == complete["source_receipt"]
    assert len(runner_env.backend_calls) == 9
    assert runner.load_campaign(complete["run_dir"])["status"] == "COMPLETE"
