"""Explicit controlled training arguments and runtime comparison for MZ trials."""
from __future__ import annotations

CONTROLLED_TRAIN_FIELDS = (
    "seed", "optimizer", "lr0", "epochs", "patience", "imgsz", "batch", "amp", "device",
    "cache", "workers", "mosaic", "close_mosaic", "deterministic", "plots",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "bgr", "mixup", "cutmix", "copy_paste", "erasing",
    "momentum", "weight_decay", "lrf", "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
    "nbs", "cos_lr", "rect", "fraction", "freeze", "single_cls", "multi_scale",
    "end2end",
)
REQUIRED_TRAIN_FIELDS = {
    "seed", "optimizer", "lr0", "epochs", "patience", "imgsz", "batch", "amp", "device",
    "cache", "workers", "mosaic", "close_mosaic", "deterministic", "plots",
    "augmentation_policy",
}


def validate_requested_protocol(protocol: dict) -> None:
    if not isinstance(protocol, dict) or REQUIRED_TRAIN_FIELDS-set(protocol):
        raise ValueError("Strict experiment requires all controlled training parameters")
    if protocol["optimizer"] == "auto" or protocol["device"] == "auto" or int(protocol["batch"]) <= 0:
        raise ValueError("Strict experiment forbids automatic optimizer/device/batch selection")
    if protocol["augmentation_policy"] != "source_only":
        raise ValueError("This strict MZ protocol supports the declared source_only policy")
    zeros = ["mosaic","hsv_h","hsv_s","hsv_v","degrees","translate","scale","shear","perspective",
        "flipud","fliplr","bgr","mixup","cutmix","copy_paste","erasing"]
    if any(protocol.get(name)!=0 for name in zeros) or protocol["close_mosaic"]!=0:
        raise ValueError("Strict source_only requires every online augmentation value to be explicitly zero")


def protocol_differences(expected: dict, actual: dict) -> dict:
    changes = {}
    for name,value in expected.items():
        if name not in CONTROLLED_TRAIN_FIELDS:
            continue
        observed = actual.get(name)
        if name == "device":
            same = str(value).removeprefix("cuda:") == str(observed).removeprefix("cuda:")
        else:
            same = value == observed
        if not same:
            changes[name] = {"requested":value,"actual":observed}
    return changes


def training_protocol_snapshot(requested: dict, actual: dict, *, strict: bool, source: str) -> dict:
    differences = protocol_differences(requested,actual) if requested else {}
    return {"schema":"alpr.training_protocol_snapshot.v1","requested":dict(requested),
        "actual":{name:actual[name] for name in CONTROLLED_TRAIN_FIELDS if name in actual},
        "capture_source":source,"strict_experiment":bool(strict),"differences":differences,
        "comparable":not differences,"memory_fallback_required":False}
