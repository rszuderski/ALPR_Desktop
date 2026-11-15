"""Research-only calibration authority; never a strict F4-B2 READY manifest.

The unchanged strict audit is retained as BLOCKED evidence, not relabelled. A
separate validator rebuilds this contract against live inputs before conversion.
JSON fingerprints detect drift, not an adversarially forged operator signature.
"""
from pathlib import Path
import importlib.metadata

from . import int8_calibration as strict

SCHEMA_V1 = "alpr.int8_calibration.exploratory.v1"
SCHEMA = "alpr.int8_calibration.exploratory.v2"
SUPPORTED_SCHEMAS = {SCHEMA_V1, SCHEMA}
MODE = "exploratory"
LABEL = "Badawcza kalibracja INT8 — bez final_test"
CONFIRMATION = "Używam wyłącznie do eksperymentu eksploracyjnego; to nie jest final_test ani ostatecznie zatwierdzony model"
LIMITATIONS = ["FINAL_TEST_NOT_VERIFIED", "ANDROID_RUNTIME_UNVERIFIED",
               "EXACT_SHA_EXCLUSION_ONLY_NOT_CAPTURE_GROUP_INDEPENDENCE"]


def runtime_versions():
    names = ("ultralytics", "onnx2tf", "tensorflow", "tensorflow-cpu", "onnxruntime", "torch")
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "NOT_INSTALLED"
    return {name: version(name) for name in names}


def _build_manifest(*, schema, history_path, run_id, role, eval_inventory, image_size, limit,
                   approved_by="", operator_confirmation=False, confidence_threshold=.25,
                   iou_threshold=.45, runtime="litert_cpu", checkpoint_sha256=""):
    strict._check(runtime == "litert_cpu", "Tryb badawczy wymaga LiteRT CPU.")
    strict._check(type(operator_confirmation) is bool, "Nieprawidłowe potwierdzenie operatora.")
    strict._check(all(type(v) in (float, int) and 0 <= v <= 1 for v in (confidence_threshold, iou_threshold)),
                  "Nieprawidłowe progi conf/IoU.")
    # Do not inject a fake final inventory or change any strict result/status.
    audit = strict.build_manifest(history_path=history_path, run_id=run_id, role=role,
                                  eval_inventory=eval_inventory, final_inventory="", image_size=image_size,
                                  limit=limit, approved_by=approved_by)
    config = {k: v for k, v in audit["config"].items() if k != "final_inventory"}
    config.update(confidence_threshold=float(confidence_threshold),
                  iou_threshold=float(iou_threshold), runtime=runtime, checkpoint_sha256=checkpoint_sha256)
    if schema == SCHEMA_V1:
        config["operator_confirmation"] = operator_confirmation
    else:
        config.pop("approved_by")
    # Only the missing final holdout is an allowed research limitation.
    blockers = [b for b in audit["blockers"] if not b.startswith("FINAL_TEST_INVENTORY_NOT_VERIFIED:")]
    source_blockers = [b for b in blockers if b != "Brak jawnego zatwierdzenia źródła przez operatora."]
    if schema == SCHEMA:
        blockers = list(source_blockers)
    if checkpoint_sha256 and checkpoint_sha256 != audit["checkpoint"]["sha256"]:
        source_blockers.append("CHECKPOINT_SHA_MISMATCH")
        blockers.append("CHECKPOINT_SHA_MISMATCH")
    if not strict._sha(checkpoint_sha256):
        blockers.append("CHECKPOINT_SHA_CONFIRMATION_REQUIRED")
    if schema == SCHEMA_V1 and not operator_confirmation:
        blockers.append("EXPLORATORY_OPERATOR_CONFIRMATION_REQUIRED")
    run = strict._read(history_path)["runs"][run_id]
    run_root = Path(run["dataset_path"]).resolve()
    base = Path(audit["source_dataset"])
    scope = run_root / "augmentation_scope.json"
    if scope.exists():
        a = strict._read(scope)
        expected_role = "plate" if role == "plate" else "char"
        strict._check(a.get("target") == expected_role and a.get("split_scope") == "train_only"
                      and Path(a.get("augmented_dataset_dir", "")).resolve() == run_root
                      and a.get("validation_test_original") is True
                      and base != run_root and not (base / "augmentation_scope.json").exists(),
                      "Niesprawdzona relacja augmentacja → bazowy train.")
    else:
        strict._check(not (run_root / "augmentation_manifest.json").exists(),
                      "Brak relacji bazowego źródła augmentacji.")
    samples = []
    maps = []
    if role == "character":
        for name in ("metadata_manifest.json", "split_assignment_manifest.json"):
            maps.append({str(r.get("image_path", "")).replace("\\", "/"): r
                         for r in strict._read(run_root / name)["items"]})
    for row in audit["samples"]:
        groups = sorted({str(m[row["relative_path"]].get("group_id") or "") for m in maps} - {""})
        samples.append({**row, "source_group_ids": groups,
                        "source_identity": "img-sha256-" + row["source_scene_sha256"]})
    strict._check(strict.fingerprint(run) == audit["run_fingerprint"], "Run zmienił się podczas kontroli grup.")
    for ref in audit["source_files"] + [audit["checkpoint"]] + [f for e in audit["exclusions"] for f in e["files"]]:
        strict._check(strict.file_sha256(ref["path"]) == ref["sha256"], "Źródło zmieniło się podczas kontroli grup.")
    result = dict(schema=schema, purpose="research_calibration",
                  status="BLOCKED" if blockers else "EXPLORATORY_ONLY", blockers=blockers,
                  source_status="BLOCKED" if source_blockers else "CALIBRATION_SOURCE_READY_EXPLORATORY",
                  config=config, role=role, split="train", checkpoint=audit["checkpoint"],
                  source_files=audit["source_files"], exclusions=audit["exclusions"],
                  source_audit=audit, training_source_fingerprint=audit["training_inventory_fingerprint"],
                  eval396_fingerprint=next((e["hashes_fingerprint"] for e in audit["exclusions"]
                                           if e["purpose"] == "EVAL396"), ""),
                  final_test_status="NOT_VERIFIED", strict_gate="FINAL_TEST_INVENTORY_NOT_VERIFIED",
                  warnings=list(LIMITATIONS), limitations=list(LIMITATIONS),
                  counts=audit["counts"], samples=samples, preprocessing=audit["preprocessing"],
                  runtime_versions=runtime_versions(), execution_precision="UNVERIFIED")
    if schema == SCHEMA_V1:
        result["operator_confirmation"] = {"accepted": operator_confirmation, "text": CONFIRMATION,
                                           "operator": config["approved_by"]}
        result["confirmation_fingerprint"] = strict.fingerprint(result)
    result["input_fingerprint"] = strict.fingerprint(result)
    return result


def build_manifest(*, history_path, run_id, role, eval_inventory, image_size, limit,
                   confidence_threshold=.25, iou_threshold=.45, runtime="litert_cpu", checkpoint_sha256=""):
    """New v2 research evidence; choosing this API does not invent an approval."""
    return _build_manifest(schema=SCHEMA, history_path=history_path, run_id=run_id, role=role,
        eval_inventory=eval_inventory, image_size=image_size, limit=limit,
        confidence_threshold=confidence_threshold, iou_threshold=iou_threshold,
        runtime=runtime, checkpoint_sha256=checkpoint_sha256)


def build_manifest_v1(**config):
    """Rebuild historical v1 with its original confirmation semantics, never migrate."""
    return _build_manifest(schema=SCHEMA_V1, **config)


def rebuild_manifest(payload):
    strict._check(payload.get("schema") in SUPPORTED_SCHEMAS, "Nieznany schemat badawczy.")
    builder = build_manifest_v1 if payload["schema"] == SCHEMA_V1 else build_manifest
    return builder(**payload["config"])


def validate_embedded_manifest(payload, *, role=None):
    strict._check(payload.get("schema") in SUPPORTED_SCHEMAS and payload.get("purpose") == "research_calibration",
                  "Wymagany osobny manifest badawczy; ścisłe BLOCKED nie jest zgodą.")
    strict._check(payload.get("status") == "EXPLORATORY_ONLY" and not payload.get("blockers"),
                  "BLOCKED: niekompletny dowód kalibracji badawczej.")
    strict._check(payload.get("input_fingerprint") == strict.fingerprint(
        {k: v for k, v in payload.items() if k != "input_fingerprint"}), "Zmieniono fingerprint badawczy.")
    c = payload["config"]
    legacy = payload["schema"] == SCHEMA_V1
    if legacy:
        strict._check(payload.get("confirmation_fingerprint") == strict.fingerprint(
            {k: v for k, v in payload.items() if k not in ("input_fingerprint", "confirmation_fingerprint")}),
            "Zmieniono dane objęte historyczną zgodą badawczą.")
        strict._check(c["operator_confirmation"] is True and str(c["approved_by"]).strip()
            and payload["operator_confirmation"] == dict(accepted=True, text=CONFIRMATION, operator=c["approved_by"]),
            "Brak historycznego potwierdzenia v1.")
    else:
        forbidden = {"approved_by", "operator_confirmation", "confirmation_fingerprint"}
        strict._check(not forbidden.intersection(payload) and not forbidden.intersection(c),
                      "Pola potwierdzenia nie należą do schematu badawczego v2.")
    strict._check(payload["role"] in {"plate", "character"}, "Nieobsługiwana rola badawcza.")
    strict._check(all(type(c[k]) in (float, int) and 0 <= c[k] <= 1
                      for k in ("confidence_threshold", "iou_threshold")), "Nieprawidłowe progi conf/IoU.")
    a = payload["source_audit"]
    strict._check(a["schema"] == strict.SCHEMA and a["status"] == "BLOCKED"
                  and a["input_fingerprint"] == strict.fingerprint({k: v for k, v in a.items() if k != "input_fingerprint"})
                  and len([b for b in a["blockers"] if b.startswith("FINAL_TEST_INVENTORY_NOT_VERIFIED:")]) == 1
                  and [b for b in a["blockers"] if not b.startswith("FINAL_TEST_INVENTORY_NOT_VERIFIED:")]
                      == ([] if legacy else ["Brak jawnego zatwierdzenia źródła przez operatora."])
                  and a["gates"] == dict(final_test_inventory="FINAL_TEST_INVENTORY_NOT_VERIFIED", eval396_inventory="VERIFIED"),
                  "Nieprawidłowy audyt pochodzenia źródeł badawczych.")
    strict._check(payload["source_status"] == "CALIBRATION_SOURCE_READY_EXPLORATORY"
                  and payload["execution_precision"] == "UNVERIFIED"
                  and payload["final_test_status"] == "NOT_VERIFIED"
                  and payload["strict_gate"] == "FINAL_TEST_INVENTORY_NOT_VERIFIED"
                  and payload["limitations"] == LIMITATIONS and payload["warnings"] == LIMITATIONS,
                  "Usunięto ograniczenia trybu badawczego.")
    strict._check(c["runtime"] == "litert_cpu" and c["checkpoint_sha256"] == a["checkpoint"]["sha256"]
                  and strict._sha(c["checkpoint_sha256"]), "Niezgodny runtime/checkpoint.")
    excluded = {"final_inventory"} if legacy else {"final_inventory", "approved_by"}
    strict._check({k:v for k,v in c.items() if k in a["config"]} ==
                  {k:v for k,v in a["config"].items() if k not in excluded}
                  and a["config"]["final_inventory"] == ""
                  and (legacy or a["config"]["approved_by"] == ""), "Niezgodna konfiguracja źródeł.")
    if role is not None:
        strict._check(payload["role"] == role, "Niezgodna rola kalibracji.")
    for key in ("role", "split", "checkpoint", "counts", "source_files", "exclusions", "preprocessing"):
        strict._check(payload[key] == a[key], "Niezgodność dowodu badawczego: " + key)
    strict._check(payload["split"] == "train" and payload["samples"]
                  and payload["counts"]["selected"] == len(payload["samples"]), "Nieprawidłowe próbki train.")
    actual = [{k: v for k,v in r.items() if k not in ("source_group_ids", "source_identity")} for r in payload["samples"]]
    strict._check(all(r["source_identity"] == "img-sha256-" + r["source_scene_sha256"]
                      and isinstance(r["source_group_ids"], list)
                      and all(isinstance(g, str) and g for g in r["source_group_ids"])
                      for r in payload["samples"]), "Nieprawidłowa tożsamość scen lub grup.")
    strict._check(actual == a["samples"] and len({r["sha256"] for r in actual}) == len(actual)
                  and all(r["split"] == "train" and strict._sha(r["sha256"]) and strict._sha(r["source_scene_sha256"])
                          for r in actual), "Niezgodne SHA/próbki train.")
    strict._check(len(payload["exclusions"]) == 1 and payload["exclusions"][0]["purpose"] == "EVAL396"
                  and payload["exclusions"][0]["scene_count"] == 396
                  and payload["eval396_fingerprint"] == payload["exclusions"][0]["hashes_fingerprint"]
                  and payload["training_source_fingerprint"] == a["training_inventory_fingerprint"],
                  "Brak dowodu EVAL396 lub źródeł treningu.")
    return payload


def validate_manifest(path, *, role=None, checkpoint=None, image_size=None,
                      confidence_threshold=None, iou_threshold=None):
    payload = validate_embedded_manifest(strict._read(path), role=role)
    if checkpoint is not None:
        strict._check(strict.file_sha256(checkpoint) == payload["checkpoint"]["sha256"], "Zmieniono checkpoint SHA.")
    if image_size is not None:
        strict._check(strict.preprocessing(image_size) == payload["preprocessing"], "Niezgodny preprocessing/imgsz.")
    for key, value in (("confidence_threshold", confidence_threshold), ("iou_threshold", iou_threshold)):
        if value is not None:
            strict._check(float(value) == payload["config"][key], "Zmieniono parametr " + key)
    strict._check(rebuild_manifest(payload) == payload, "Zmieniono wejścia/wersje kalibracji badawczej.")
    return payload


def usage_evidence(manifest):
    """Call only after the guarded loader completed; a preflight is not usage."""
    validate_embedded_manifest(manifest)
    return dict(schema="alpr.calibration_usage.v1", status="CONSUMED_BY_VERIFIED_LOADER",
                purpose="research_calibration", manifest_fingerprint=manifest["input_fingerprint"],
                sample_sha256=[r["sha256"] for r in manifest["samples"]],
                source_scene_sha256=sorted({r["source_scene_sha256"] for r in manifest["samples"]}),
                source_group_ids=sorted({g for r in manifest["samples"] for g in r["source_group_ids"]}),
                source_identities=sorted({r["source_identity"] for r in manifest["samples"]}),
                group_scope="Recorded metadata groups and exact scene identities, not verified capture groups")
