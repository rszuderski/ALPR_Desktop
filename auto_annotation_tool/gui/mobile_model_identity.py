"""Recorded phone identity only; never consult the active Desktop model library."""
from __future__ import annotations


def _dict(value):
    return value if isinstance(value, dict) else {}


def recorded_model_identity(provenance, *, verified=False, recorded_package_id=""):
    p = _dict(provenance) if verified else {}
    refs = _dict(p.get("model_refs"))
    manifests = _dict(p.get("pipeline_manifests"))
    package = _dict(manifests.get("package"))
    experiment = _dict(p.get("experiment_session"))
    package_ids = {str(v) for v in (recorded_package_id if verified else "",
        experiment.get("package_id"), package.get("package_id")) if v}
    conflict = len(package_ids) > 1
    result = dict(verified=bool(verified), package_id=next(iter(package_ids), "") if not conflict else "",
                  package_name=str(package.get("name") or ""), conflicts=[], models={})
    if conflict:
        result["conflicts"].append("Sprzeczne identyfikatory pakietu w metadanych źródła")
    for role, title in (("plate", "MT"), ("character", "MZ")):
        ref, manifest = _dict(refs.get(role)), _dict(manifests.get(role))
        different = bool(ref.get("model_id") and manifest.get("model_id")
                         and ref["model_id"] != manifest["model_id"])
        if different:
            result["conflicts"].append(f"{title}: sprzeczne model_id w metadanych źródła")
        model = {k:str(ref.get(k) or "") for k in
                 ("model_display_id", "model_id", "variant_id", "checkpoint_sha256", "package_sha256")}
        model["verified"] = bool(verified and not conflict and not different and ref)
        variant = model["variant_id"].lower()
        precision = str(ref.get("precision") or "").lower()
        if not precision:
            precision = next((part for part in ("fp32", "fp16", "int8")
                              if variant.endswith("-" + part)), "")
        model["precision"] = precision
        model["caption"] = ((model["model_display_id"] or model["model_id"])
                            if model["verified"] else "") or f"{title}: brak danych źródłowych"
        result["models"][role] = model
    result["caption"] = " + ".join(result["models"][r]["caption"] for r in ("plate", "character"))
    a, b = (result["models"][r] for r in ("plate", "character"))
    if a["verified"] and b["verified"] and a["variant_id"] == b["variant_id"] and a["variant_id"]:
        result["variant_caption"] = a["variant_id"].replace("tflite", "TFLite").replace("-", " ").replace("fp", "FP").replace("int8", "INT8")
    else:
        result["variant_caption"] = " / ".join(
            f"{title} {m['precision'].upper() if m['verified'] and m['precision'] else 'brak danych'}"
            for title, m in (("MT", a), ("MZ", b)))
    return result


def identity_from_bundle(bundle):
    verified = bool(bundle is not None and getattr(getattr(bundle, "validation", None), "ok", False))
    result = recorded_model_identity({k:getattr(bundle, k, {}) for k in
        ("model_refs", "pipeline_manifests", "experiment_session")}, verified=verified)
    from pathlib import Path
    import re
    repeat = ""
    if verified:
        parents = list(Path(str(getattr(bundle, "path", ""))).parents)
        if any(p.name.lower() == "02_sesje" for p in parents):
            repeat = next((p.name.lower() for p in parents if re.fullmatch(r"r\d{2}", p.name.lower())), "")
        if not repeat:
            raw = _dict(getattr(bundle, "collection_session", {})).get("replicate_index")
            try:
                repeat = f"r{int(raw):02d}" if int(raw) > 0 else ""
            except (ValueError, TypeError):
                pass
    result["repeat_label"] = repeat or "powtórzenie: brak danych"
    return result


def chart_session_label(identity, *, scenario, repeat, session_id, precision="unknown"):
    identity = identity or recorded_model_identity({})
    variant = identity["variant_caption"]
    # Short separate lines also fit the existing comparison window on laptops.
    models = identity["models"]
    return (f"{models['plate']['caption']}\n{models['character']['caption']}\n"
            f"{variant} · {repeat} · {session_id[-7:]}\n{scenario}")


def export_identity(record):
    return dict(package_and_models=record.model_identity,
                source_archive_sha256=record.source_sha256, session_id=record.session_id,
                canvas_review_revision=record.review_revision,
                review_export_manifest_sha256=record.manifest_sha256)


def chart_identity_caption(records):
    captions = list(dict.fromkeys(
        (r.model_identity or recorded_model_identity({}))["caption"] + " · " +
        (r.model_identity or recorded_model_identity({}))["variant_caption"] for r in records))
    return "\n".join(captions) if len(captions) <= 2 else "Pakiety MT+MZ — identyfikatory przy każdej sesji"
