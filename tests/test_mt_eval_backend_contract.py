"""Exercise the native result adapter without model forward or real scene inference."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results

from auto_annotation_tool.ranking import mt_eval_runner as runner


@pytest.fixture
def backend(tmp_path, monkeypatch):
    import ultralytics
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint = tmp_path / "synthetic_not_a_checkpoint.pt"
    checkpoint.write_bytes(b"UNIT_TEST_ONLY")
    model_row = {"checkpoint_file": str(checkpoint), "checkpoint_sha256": runner._sha(checkpoint),
                 "kpt_shape": [4, 2], "names": {"0": "plate"}}
    protocol = {"inference": runner._inference_parameters("cpu", True), "runtime": {"device": "cpu"},
                "rng": deepcopy(runner.RNG_PROTOCOL)}
    calls = []
    pixels = np.zeros((80, 160, 3), dtype=np.uint8)
    # Deliberately reversed and cyclically shifted. An adapter must preserve it exactly.
    points = torch.tensor([[[90., 40.], [90., 20.], [20., 20.], [20., 40.]]])
    result = Results(pixels, "test", {0: "plate"}, boxes=torch.tensor([[20., 20., 90., 40., .8, 0.]]), keypoints=points)
    fake = SimpleNamespace(task="pose", predictor=SimpleNamespace(
        args=SimpleNamespace(**protocol["inference"]), imgsz=[512, 512],
        model=SimpleNamespace(device=torch.device("cpu"), fp16=False, end2end=True,
                              kpt_shape=[4, 2], names={0: "plate"})))
    def predict(*, source, stream, **kwargs):
        calls.append({"source": source, "stream": stream, "kwargs": kwargs,
                      "grad_enabled": torch.is_grad_enabled()})
        return [result]
    fake.predict = predict
    monkeypatch.setattr(ultralytics, "YOLO", lambda *args, **kwargs: fake)
    adapter = runner.UltralyticsBackend(model_row, protocol)
    return SimpleNamespace(adapter=adapter, fake=fake, result=result, pixels=pixels, points=points,
                           protocol=protocol, model=model_row, calls=calls)


def test_adapter_retains_native_keypoint_indices_and_actual_parameters(backend):
    with runner._deterministic_runtime(backend.protocol):
        output = backend.adapter.predict(backend.pixels)
    prediction = output["predictions"][0]
    assert prediction["keypoints"] == backend.points[0].tolist()
    assert prediction["confidence"] == backend.result.boxes.conf.item()
    assert prediction["index"] == 0 and prediction["class_id"] == 0
    call = backend.calls[0]
    assert call["source"] is backend.pixels
    assert call["stream"] is False and call["grad_enabled"] is False
    assert call["kwargs"] == backend.protocol["inference"]
    assert output["effective_inference"]["backend_names"] == {"0": "plate"}
    assert output["effective_inference"]["determinism"]["warn_only"] is True


def test_adapter_zero_detections_is_valid_scene(backend):
    backend.result.boxes.data = torch.empty((0, 6))
    backend.result.keypoints.data = torch.empty((0, 4, 2))
    with runner._deterministic_runtime(backend.protocol):
        assert backend.adapter.predict(backend.pixels)["predictions"] == []


@pytest.mark.parametrize("field,value", [("half", True), ("conf", .01), ("rect", True), ("augment", True)])
def test_adapter_rejects_changed_effective_parameter(backend, field, value):
    setattr(backend.fake.predictor.args, field, value)
    with runner._deterministic_runtime(backend.protocol), pytest.raises(ValueError, match="MT_EFFECTIVE_PARAMETER_CHANGED"):
        backend.adapter.predict(backend.pixels)


def test_adapter_rejects_wrong_keypoint_shape(backend):
    backend.fake.predictor.model.kpt_shape = [17, 3]
    with runner._deterministic_runtime(backend.protocol), pytest.raises(ValueError, match="MT_EFFECTIVE_BACKEND_CHANGED"):
        backend.adapter.predict(backend.pixels)


def test_deterministic_context_restores_process_flags(backend, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "BEFORE_TEST")
    before = runner._actual_determinism()
    state = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="test cancellation"):
        with runner._deterministic_runtime(backend.protocol):
            runner._seed_scene(1)
            assert runner._actual_determinism()["cublas_workspace_config"] == ":4096:8"
            raise RuntimeError("test cancellation")
    assert runner._actual_determinism() == before
    assert torch.equal(state, torch.random.get_rng_state())
