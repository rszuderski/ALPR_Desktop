"""Read-only evidence for ranking catalogs. Never load checkpoints or reconcile history."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

from .model_display import build_model_display_ref
from .run_display import build_run_display_ref

UNKNOWN = "nieustalone"


def search_text(value):
    if isinstance(value, dict):
        return " ".join(search_text(k)+" "+search_text(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return " ".join(search_text(v) for v in value)
    return str(value if value is not None else "").casefold().replace("\\", "/")


def path_key(value):
    return str(Path(value).resolve()).casefold() if str(value or "").strip() else ""


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig")), ""
    except (OSError, ValueError) as exc:
        return {}, f"{path}: {exc}"


def ranking_catalog(workspace, extra_files=()):
    """Keep every raw historical row, including fields lost by legacy dataclass loading."""
    workspace = Path(workspace)
    files = [*workspace.glob("7_rankings/**/model_ranking.json"),
             *workspace.glob("9_projects/*/7_rankings/**/model_ranking.json"), *map(Path, extra_files)]
    rows, errors, seen = [], [], set()
    for path in files:
        key = path_key(path)
        if key in seen:
            continue
        seen.add(key)
        payload, error = read_json(path)
        if error or not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
            errors.append(error or f"{path}: brak listy entries")
            continue
        for index, row in enumerate(payload["entries"]):
            if not isinstance(row, dict):
                errors.append(f"{path}: nieprawidłowy rekord {index}")
                continue
            rows.append({**row, "_ranking_file": str(path.resolve()), "_record_index": index})
    return {"rows": rows, "errors": errors, "files": sorted(seen)}


def history_catalog(workspace):
    workspace = Path(workspace)
    paths = [*workspace.glob("5_training_runs/*/training_history.json"),
             *workspace.glob("9_projects/*/5_training_runs/training_history.json"),
             *workspace.glob("9_projects/*/5_training_runs/*/training_history.json")]
    rows = []
    for path in paths:
        data, _ = read_json(path)
        runs = data.get("runs", []) if isinstance(data, dict) else []
        runs = list(runs.values()) if isinstance(runs, dict) else runs
        for row in runs:
            if isinstance(row, dict):
                rows.append({**row, "_history_file": str(path.resolve())})
    return rows


def location_scope(path, workspace):
    try:
        parts = Path(path).resolve().relative_to(Path(workspace).resolve()).parts
    except ValueError:
        return "zewnętrzny / " + UNKNOWN
    if len(parts) > 1 and parts[0] == "9_projects":
        return "Projekt: " + parts[1]
    return "Globalny"


def _first(*values):
    return next((value for value in values if value is not None and value != ""), None)


def model_evidence(path, histories, workspace, target):
    from ..validators import _metadata_payload_matches_model, _model_metadata_sidecar_candidates
    path = Path(path).resolve()
    metadata_path = next((p for p in _model_metadata_sidecar_candidates(path) if p.is_file()),
                         path.with_suffix(path.suffix + ".metadata.json"))
    payload, error = read_json(metadata_path)
    if not isinstance(payload, dict):
        payload, error = {}, "Nieprawidłowy format metadanych"
    if payload and not _metadata_payload_matches_model(path, payload):
        payload, error = {}, "Metadane nie odpowiadają obecnemu plikowi modelu"
    model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    info = model.get("info", payload.get("info", {}))
    training = payload.get("training", {})
    snapshot = payload.get("run_snapshot", {})
    info = info if isinstance(info, dict) else {}
    training = training if isinstance(training, dict) else {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    dataset = training.get("dataset") if isinstance(training.get("dataset"), dict) else {}
    explicit_run = _first(training.get("run_id"), snapshot.get("id"))
    # An exact recorded output path or an explicit sidecar ID is evidence; filename fragments are not.
    matched = [r for r in histories if any(path_key(r.get(k)) == path_key(path)
               for k in ("best_weights", "last_weights", "model_path"))]
    if not matched and explicit_run:
        matched = [r for r in histories if r.get("id") == explicit_run]
    run = matched[0] if len(matched) == 1 else {}
    merged = {**run, **snapshot, **training}
    run_id = _first(explicit_run, run.get("id"))
    role = _first(info.get("target"), training.get("target"),
                  dataset.get("target"), run.get("model_type"))
    if role in {"plate", "plates"} or (info.get("task") == "pose" and info.get("classes") == ["plate"]):
        role = "MT"
    elif role in {"char", "chars", "character"}:
        role = "MZ"
    else:
        role = UNKNOWN
    architecture = _first(info.get("architecture_label"), info.get("yolo_variant")) or UNKNOWN
    size = _first(info.get("model_scale"), info.get("yolo_size")) or UNKNOWN
    completed = _first(training.get("run_epochs_completed"), merged.get("current_epoch"))
    planned = _first(merged.get("epochs"), merged.get("run_epochs_planned"))
    date = _first(merged.get("started_at"), merged.get("created_at")) or UNKNOWN
    participant = build_model_display_ref(path, run={**merged, "id": run_id or ""}, target_hint=target).id
    model_map = _first(merged.get("best_map50_95"), merged.get("map50_95"))
    row = {"path": str(path), "key": path_key(path), "participant": participant, "role": role,
           "architecture": architecture, "size": size, "run_id": run_id or UNKNOWN,
           "checkpoint": path.name, "date": date, "completed": completed, "planned": planned,
           "epochs": f"{completed if completed is not None else UNKNOWN} / {planned if planned is not None else UNKNOWN}",
           "scope": location_scope(path, workspace), "map": model_map,
           "train_images": _first(training.get("run_train_images"), dataset.get("train_images")),
           "metadata_file": str(metadata_path) if metadata_path.exists() else "brak",
           "metadata_error": error, "history_file": run.get("_history_file", "brak jednoznacznego powiązania"),
           "base_model": merged.get("base_model") or UNKNOWN,
           "metadata_sha": _first(training.get("best_checkpoint_sha256"), model.get("sha256")),
           "provenance_warning": "Niejednoznaczne ID treningu" if len(matched) > 1 else ""}
    row["search"] = search_text(row)
    return row


def actual_sha256(path):
    path = Path(path)
    before = path.stat()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Plik modelu zmienił się podczas obliczania SHA-256")
    return sha


def _metadata_summary(value, depth=0):
    if depth > 3:
        return "[zagnieżdżone dane]"
    if isinstance(value, list):
        return f"lista: {len(value)} elementów"
    if isinstance(value, dict):
        return {str(k): _metadata_summary(v, depth+1) for k, v in value.items() if len(str(k)) < 55}
    return value


def track_evidence(candidate, ranking_rows, workspace):
    row = dict(candidate)
    path = Path(row["path"]).resolve()
    if path.name.lower() in {"annotations.xml", "data.yaml", "selection_manifest.json"}:
        path = path.parent
    key = path_key(path)
    records = [r for r in ranking_rows if path_key(r.get("reference_path")) == key]
    metadata, errors = {}, []
    for name in ("run_manifest.json", "metadata.json", "track_manifest.json", "selection_manifest.json", "sample_selection.json"):
        file = path / name
        if file.is_file():
            payload, error = read_json(file)
            if error:
                errors.append(error)
            elif isinstance(payload, dict):
                metadata[name] = payload
    controlled = metadata.get("track_manifest.json", {})
    run = metadata.get("run_manifest.json", {})
    created = _first(controlled.get("created_at"), run.get("generated_at"), run.get("created_at"),
                     metadata.get("metadata.json", {}).get("created_at"))
    date_origin = "metadane"
    if not created and path.exists():
        created = datetime.fromtimestamp(path.stat().st_ctime).isoformat(timespec="seconds")
        date_origin = "system plików (kopia może mieć inną datę)"
    xml = Path(row.get("xml_path") or path / "annotations.xml")
    relative_xml = controlled.get("ground_truth", {}).get("relative_path")
    if relative_xml and not Path(relative_xml).is_absolute() and ".." not in Path(relative_xml).parts:
        xml = path / relative_xml
    count = row.get("counts", {}).get("total")
    count = _first(count, row.get("split_count"), controlled.get("member_count"))
    if count is None and (path / "images").is_dir():
        from ..utils import get_image_files
        count = len(get_image_files(path / "images"))
    z2 = build_run_display_ref(path, kind_hint="annotation").id if (path / "annotations.xml").is_file() else UNKNOWN
    if controlled:
        z2 = run.get("source_run_id") or UNKNOWN  # A TRK is not a Z2 run.
    summary = {name: _metadata_summary(data) for name, data in metadata.items()}
    row.update(path=str(path), key=key, directory=path.name, z2_id=z2,
               track_id=controlled.get("track_id") or row.get("id") or UNKNOWN,
               source=location_scope(path, workspace), source_count=count,
               evaluated_counts=sorted({int(r["total_images"]) for r in records if r.get("total_images") is not None}),
               evaluations=records, created=created or UNKNOWN, date_origin=date_origin,
               xml_path=str(xml), xml_exists=xml.is_file(), metadata=summary, metadata_errors=errors,
               history_status=f"{len(records)} zapisanych ocen" if records else "brak rekordu",
               pool_count=metadata.get("sample_selection.json", {}).get("candidate_count_before"))
    row["search"] = search_text(row)
    return row


def historical_tracks(candidates, catalog, workspace, target):
    """Discover reference paths lost from the Z2 list; never join on an image count."""
    result = [track_evidence(c, catalog["rows"], workspace) for c in candidates]
    seen = {r["key"] for r in result}
    for record in catalog["rows"]:
        path = str(record.get("reference_path") or "").strip()
        task = str(record.get("task_type") or "").casefold()
        relevant = ("znak" in task or "char" in task) if target == "char" else ("tablic" in task or "pose" in task or "plate" in task)
        if not path or path_key(path) in seen or not relevant:
            continue
        seen.add(path_key(path))
        result.append(track_evidence({"path": path, "id": record.get("reference_name") or UNKNOWN,
             "type": "historyczny", "split": record.get("split_name") or "—", "ready": False,
             "status": "tylko historia" if Path(path).exists() else "brak katalogu"}, catalog["rows"], workspace))
    return result


def evaluation_lines(records):
    if not records:
        return "Brak powiązanych historycznych ocen (dopasowanie po pełnej ścieżce, nie po liczbie obrazów)."
    return "\n".join(f"{r.get('date_evaluated') or UNKNOWN} | {r.get('model_name') or UNKNOWN} | "
        f"oceniono: {r.get('total_images', UNKNOWN)} obrazów | {r.get('reference_name') or UNKNOWN}\n"
        f"  Zapisane miary: Precision={r.get('precision', UNKNOWN)}; Recall={r.get('recall', UNKNOWN)}; accuracy={r.get('accuracy', UNKNOWN)}\n"
        f"  Eksperyment: {r.get('experiment_id') or UNKNOWN}; tor: {r.get('track_id') or UNKNOWN}\n"
        f"  Model: {r.get('model_path') or UNKNOWN}\n  Zapis: {r['_ranking_file']} [rekord {r['_record_index']}]"
        for r in records)


def track_details(row):
    return (f"Katalog: {row['directory']}\nZ2: {row['z2_id']} | Tor: {row['track_id']}\n"
        f"annotations.xml: {row['xml_path']} — {'istnieje' if row['xml_exists'] else 'brak / nie dotyczy'}\n"
        f"Obrazy toru źródłowego: {row['source_count'] if row['source_count'] is not None else UNKNOWN}\n"
        f"Próbki historycznie ocenione: {row['evaluated_counts'] or 'brak rekordu'}\n"
        f"Pula przed selekcją (jeśli zapisana): {row['pool_count'] if row['pool_count'] is not None else UNKNOWN}\n"
        f"Utworzono: {row['created']} — {row['date_origin']}\nZakres: {row['source']}\n\n"
        f"HISTORYCZNE OCENY\n{evaluation_lines(row['evaluations'])}\n\n"
        f"METADANE\n{json.dumps(row['metadata'], ensure_ascii=False, indent=2) if row['metadata'] else 'Brak metadanych'}\n"
        + "\n".join(row['metadata_errors']))


def model_details(row, sha="obliczanie…"):
    return (f"Uczestnik: {row['participant']}\nRola: {row['role']} | Architektura: {row['architecture']} | Rozmiar: {row['size']}\n"
            f"Trening: {row['run_id']} | Data treningu: {row['date']}\nEpoki ukończone / planowane: {row['epochs']}\n"
            f"Checkpoint: {row['checkpoint']}\nZakres: {row['scope']}\nSHA-256 pliku: {sha}\n"
            f"SHA zapisane w metadanych: {row['metadata_sha'] or UNKNOWN}\n"
            f"Metadane: {row['metadata_file']}\nHistoria: {row['history_file']}\nBaza wg metadanych: {row['base_model']}\n"
            f"{row['provenance_warning']}\n{row['metadata_error']}")
