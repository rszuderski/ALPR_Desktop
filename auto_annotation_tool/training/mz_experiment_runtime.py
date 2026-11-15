"""Read-only frozen MZ preflight and requests shared by GUI and CLI."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from ..dataset_split_assignment import canonical_sha256, validate_group_assignment
from .experiment_protocol import validate_requested_protocol
from .model_provenance import build_training_dataset_snapshot, training_dataset_snapshots_match

MZ_VARIANTS = ("MZ-n", "MZ-s")


def file_sha256(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_mz_experiment_protocol(path) -> dict:
    protocol = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(protocol, dict):
        raise ValueError("Unsupported experiment protocol")
    return protocol


def _check_environment(protocol: dict) -> None:
    import torch
    import ultralytics
    import albumentations
    expected = protocol["environment"]
    for name, version in (("torch", torch.__version__), ("ultralytics", ultralytics.__version__),
                          ("albumentations", albumentations.__version__)):
        if str(version) != expected[name]:
            raise ValueError("Training library version differs from the common protocol: " + name)
    if "python" in expected:
        import platform
        if platform.python_version() != expected["python"]:
            raise ValueError("Python version differs from the common protocol")
    device = protocol["training"]["device"]
    if device == "cpu":
        return
    if not torch.cuda.is_available():
        raise ValueError("The declared CUDA device is unavailable")
    index = int(str(device).removeprefix("cuda:"))
    if torch.cuda.get_device_name(index) != expected["gpu"]:
        raise ValueError("The declared GPU differs from the common protocol")
    if "cuda_runtime" in expected and torch.version.cuda != expected["cuda_runtime"]:
        raise ValueError("CUDA runtime differs from the common protocol")


def check_mz_experiment_protocol(protocol: dict, *, repo_root=None) -> dict:
    """Validate existing files only; never repair YAML, create runs or load YOLO."""
    repo = Path(repo_root or Path(__file__).absolute().parents[2])
    if not isinstance(protocol, dict) or protocol.get("schema") != "alpr.experiment.mz_ns.v1" or protocol.get("experiment_id") != "E-MZ-NS-01":
        raise ValueError("Unsupported experiment protocol")
    if protocol.get("protocol_sha256") != canonical_sha256({k:v for k,v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("Experiment protocol fingerprint mismatch")
    validate_requested_protocol(protocol["training"])
    dataset = protocol["dataset"]
    root = Path(dataset["path"])
    snapshot = build_training_dataset_snapshot(root, target="char")
    ok, message = training_dataset_snapshots_match(dataset["training_dataset_snapshot"], snapshot)
    if not ok or snapshot["dataset_id"] != dataset["dataset_id"]:
        raise ValueError("Experiment input changed: " + message)
    for name in ("split_sha256", "data_yaml_sha256"):
        if snapshot[name] != dataset[name]:
            raise ValueError("Experiment input fingerprint changed: " + name)
    assignment = read_mz_experiment_protocol(root / "split_assignment_manifest.json")
    manifest = read_mz_experiment_protocol(root / "metadata_manifest.json")
    validated = validate_group_assignment(assignment, source_manifest=manifest)
    if not validated["ok"] or assignment["assignment_sha256"] != dataset["assignment_sha256"]:
        raise ValueError("Assignment changed or source group leakage was found")
    for name in ("train", "val", "test"):
        if dataset[name + "_count"] != validated["counts"][name + "_images"]:
            raise ValueError("Experiment image counts changed: " + name)
    source = Path(dataset.get("source_path") or root.parent / dataset["source_dataset_id"])
    if file_sha256(source / "metadata_manifest.json") != dataset["source_manifest_sha256"]:
        raise ValueError("Source manifest changed after protocol preparation")
    source_manifest = read_mz_experiment_protocol(source / "metadata_manifest.json")
    if str(source_manifest.get("gold_source_contract_sha256") or "") != dataset["source_contract_sha256"]:
        raise ValueError("Source contract changed after protocol preparation")
    if assignment["source_manifest_sha256"] != dataset["source_manifest_sha256"]:
        raise ValueError("Assignment source manifest differs from the protocol")
    code_hashes = protocol["code"]["file_sha256"]
    if not code_hashes or canonical_sha256(code_hashes) != protocol["code"]["working_code_sha256"]:
        raise ValueError("Frozen code fingerprint mismatch")
    for name, expected in code_hashes.items():
        path = repo / name
        if not path.resolve().is_relative_to(repo.resolve()) or file_sha256(path) != expected:
            raise ValueError("Code changed after protocol preparation: " + name)
    for variant in MZ_VARIANTS:
        model = protocol["models"][variant]
        if model["architecture"] != ("yolo26n" if variant == "MZ-n" else "yolo26s"):
            raise ValueError("Unsupported model architecture: " + variant)
        if file_sha256(model["checkpoint"]) != model["checkpoint_sha256"]:
            raise ValueError("Base checkpoint changed: " + variant)
    _check_environment(protocol)
    return {"ok":True, "dataset_id":snapshot["dataset_id"],
        "assignment_sha256":assignment["assignment_sha256"], "split_sha256":snapshot["split_sha256"],
        "counts":validated["counts"], "protocol_sha256":protocol["protocol_sha256"]}


def build_mz_training_request(protocol: dict, variant: str) -> dict:
    """Build only from a verified JSON object, independent of editable GUI fields."""
    if variant not in MZ_VARIANTS:
        raise ValueError("Unknown experiment variant: " + str(variant))
    training = protocol["training"]
    return {"name":protocol["experiment_id"] + "_" + variant,
        "dataset_path":protocol["dataset"]["path"],
        "base_model":protocol["models"][variant]["checkpoint"],
        "epochs":training["epochs"], "batch_size":training["batch"], "img_size":training["imgsz"],
        "device":training["device"], "lr0":training["lr0"], "training_target":"char",
        "strict_experiment":True, "experiment_id":protocol["experiment_id"],
        "training_protocol":deepcopy(training)}


@dataclass(frozen=True)
class MZExperimentSelection:
    protocol: dict
    variant: str
    preflight: dict

    @property
    def request(self) -> dict:
        return build_mz_training_request(self.protocol, self.variant)


def prepare_mz_experiment(protocol: dict, variant: str, *, repo_root=None) -> MZExperimentSelection:
    if variant not in MZ_VARIANTS:
        raise ValueError("Unknown experiment variant: " + str(variant))
    frozen = deepcopy(protocol)
    checked = check_mz_experiment_protocol(frozen, repo_root=repo_root)
    return MZExperimentSelection(frozen, variant, checked)
