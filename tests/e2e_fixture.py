"""Synthetic E2E fixture only. No checkpoint is loaded, no real model executes."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from auto_annotation_tool.ranking import e2e_eval_runner as runner
from auto_annotation_tool.ranking.eval396 import digest
from auto_annotation_tool.ranking.e2e_eval_metrics import validate_reference
from auto_annotation_tool.image_orientation import read_oriented_bgr


def quad(x):
    return [[x, 10.], [x+30., 10.], [x+30., 30.], [x, 30.]]


def build_env(tmp_path, monkeypatch):
    root = tmp_path / "SYNTHETIC_TEST_ONLY"
    selection = root / "evaluation_selections" / "fixture"
    selection.mkdir(parents=True)
    cache = root / "evaluation_runs" / "EVAL396_MT_run_SYNTHETIC"
    cache.mkdir(parents=True)
    images = root / "images"
    images.mkdir()
    models, receipt = [], {}
    for label in (*runner.MT_LABELS, *runner.MZ_LABELS):
        path = root / (label + ".pt")
        path.write_bytes(("SYNTHETIC BYTES - NEVER DESERIALIZE " + label).encode())
        receipt[str(path)] = runner.mt._sha(path)
        models.append({"label": label, "checkpoint_file": str(path), "checkpoint_sha256": receipt[str(path)], "run_id": "TEST_"+label})
    reference, base = [], []
    for index, domain in enumerate(("DAY", "DAY", "NIGHT")):
        path = images / f"scene_{index}.png"
        Image.new("RGB", (128, 64), (40 + 20*index,) * 3).save(path)
        receipt[str(path)] = runner.mt._sha(path)
        pixels = read_oriented_bgr(path)
        pixel_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
        plates = []
        if index < 2:
            plates.append({"plate_id": "GT_"+str(index), "bbox": [10, 10, 40, 30], "quad": quad(10),
                "bbox_eligible": True, "pose_eligible": True, "ocr_eligible": True, "text_decision": "approved", "ground_truth_text": "AB"})
        if index == 1:
            plates.append({"plate_id": "plate_000421", "bbox": [65, 10, 95, 30], "quad": list(reversed(quad(65))),
                "bbox_eligible": True, "pose_eligible": False, "ocr_eligible": False, "pose_exclusion": {"reason": "TEST_ONLY"},
                "pose_evaluation_status": "EXCLUDED_BY_OPERATOR", "text_decision": "excluded_unreadable", "ground_truth_text": None})
        predictions = []
        if index == 1:
            predictions = [{"index": i, "bbox": [x, 10, x+30, 30], "keypoints": quad(x),
                "confidence": c, "class_id": 0} for i, x, c in [(0, 10., .9), (1, 65., .8), (2, 10., .7)]]
        elif index == 2:
            predictions = [{"index": 0, "bbox": [10., 10., 40., 30.], "keypoints": None, "confidence": .6, "class_id": 0}]
        reference.append({"sha256": receipt[str(path)], "file": path.name, "domain": domain,
                          "image": {"oriented_size": [128, 64], "decoded_pixel_sha256": pixel_sha}, "plates": plates})
        base.append({"scene_sha256": receipt[str(path)], "file": path.name, "width": 128, "height": 64,
            "decoded_pixel_sha256": pixel_sha, "mt_raw_sha256": digest(predictions), "predictions": predictions})
    marker = root / "source_metadata.json"
    marker.write_text('{"kind":"SYNTHETIC_TEST_ONLY"}')
    receipt[str(marker)] = runner.mt._sha(marker)
    code = root / "synthetic_code.py"
    code.write_text("# TEST_ONLY\n")
    context = {"selection_dir": str(selection), "cache_dir": str(cache), "output_root": str(cache.parent),
        "source_root": str(images), "models": models, "inputs": {label: deepcopy(base) for label in runner.MT_LABELS},
        "reference": reference, "counts": validate_reference(reference), "source_files": receipt,
        "selection_sha": "TEST_SELECTION", "gt_freeze_sha": "TEST_FREEZE", "geometry_approval_sha": "TEST_GEOMETRY",
        "mt_protocol_sha": "TEST_MT_PROTOCOL", "mt_completion_sha": "TEST_MT_COMPLETION",
        "evidence_kind": "SYNTHETIC_TEST_ONLY_NOT_AN_EXPERIMENT", "protected_roots": [str(selection), str(images), str(cache)]}
    env = SimpleNamespace(context=context, marker=marker, code=code, calls=[], inspections=[], after_predict=None,
        active=0, max_active=0, fail_on=None, chars="AB", effective_override={}, events=[])

    def sources(*args, **kwargs):
        runner._check_receipt(context["source_files"])
        return deepcopy(context)

    def inspect_fake(row):
        assert Path(row["checkpoint_file"]).read_bytes().startswith(b"SYNTHETIC BYTES")
        env.inspections.append(row["label"])
        return {**deepcopy(row), "task": "detect", "names": deepcopy(runner.ALPHABET), "native_end2end": True, "loaded_parameter_dtype": "torch.float32"}

    class Backend:
        def __init__(self, model, protocol):
            assert set(protocol) == {"inference", "runtime"}  # No GT/domain/cohort/protocol reference.
            self.model, self.protocol = model, protocol
            env.active += 1
            env.max_active = max(env.max_active, env.active)

        def predict(self, pixels):
            assert isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8 and pixels.shape[2] == 3
            env.calls.append(self.model["label"])
            env.events.append("predict")
            if env.fail_on == len(env.calls):
                raise RuntimeError("TEST_CUDA_ERROR")
            chars = [{"index": i, "class_id": ord(ch)-ord('A')+10, "character": ch,
                "bbox": [20.+i*20, 10., 35.+i*20, 40.], "confidence": .9} for i,ch in enumerate(env.chars)]
            effective = {**deepcopy(self.protocol["inference"]), "device": "cpu", "checked_imgsz": [320,320],
                "backend_fp16": False, "backend_end2end": True, "backend_names": deepcopy(runner.ALPHABET),
                "determinism": {"deterministic_algorithms": True, "warn_only": True, "cudnn_benchmark": False,
                    "cudnn_deterministic": True, "matmul_tf32": False, "cudnn_tf32": False, "cublas_workspace_config": ":4096:8"}}
            effective.update(env.effective_override)
            if env.after_predict:
                env.after_predict()
            return {"characters": chars, "effective_inference": effective, "speed_ms": {"inference": .01}}

        def close(self):
            env.active -= 1

    @contextmanager
    def deterministic(_protocol):
        yield

    monkeypatch.setattr(runner, "_source_context", sources)
    monkeypatch.setattr(runner, "inspect_mz_checkpoint", inspect_fake)
    monkeypatch.setattr(runner, "_code_receipt", lambda: {str(code): runner.mt._sha(code)})
    monkeypatch.setattr(runner, "_git_state", lambda: {"head": "SYNTHETIC_TEST_ONLY", "status": []})
    monkeypatch.setattr(runner.mt, "_runtime", lambda device: {"device": "cpu", "test_only": True})
    monkeypatch.setattr(runner.mt, "_deterministic_runtime", deterministic)
    monkeypatch.setattr(runner.mt, "_seed_scene", lambda i: None)
    monkeypatch.setattr(runner, "MZBackend", Backend)
    env.prepared = runner.prepare_campaign(selection, cache, pairs=[("MT-n", "MZ-s")], device="cpu")
    env.approval = runner.save_approval(env.prepared, operator="TEST FIXTURE ONLY - NOT OPERATOR", acknowledged=True)["approval"]
    return env
