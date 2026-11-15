"""Canonical per-plate GT authored in PZ2, using the shared writable GT pack."""
from __future__ import annotations

from pathlib import Path
import copy
from tkinter import messagebox, simpledialog

from ..gt_pack import ALPRGTPack
from ..gt_resource_companions import DEFAULT_WORKING_PACK_NAME
from ..plate_ground_truth import normalize_plate_ground_truth_text

PRODUCER = "auto_annotation_tool.desktop.z3"


def working_pack_path(host) -> Path:
    # Respect the project/resource working mount used by Z2. Imported companions
    # are never chosen as automatic write targets.
    if getattr(host, "_campaign_controlled", False) or getattr(host, "_campaign_step3_context_active", False):
        from .z2_gt_pack_runtime import get_working_gt_pack_path
        path = get_working_gt_pack_path(host)
        if path is not None and path.name.lower() == DEFAULT_WORKING_PACK_NAME:
            return path
        raise ValueError("Projekt nie ma dostępnego roboczego zbioru numerów tablic.")
    directory = getattr(host, "preview_dir_var", None)
    raw = str(directory.get() if directory is not None else "").strip()
    if not raw or not Path(raw).is_dir():
        raise ValueError("Najpierw wczytaj katalog tablic.")
    return Path(raw) / DEFAULT_WORKING_PACK_NAME


def _plate_id(data):
    attrs = data.get("plate_attributes") or {}
    return str(data.get("source_annotation_id") or data.get("plate_annotation_id")
               or attrs.get("plate_annotation_id") or "").strip()


def _inherit_plate_history(host, target, data, source, plate_id):
    """Copy just this plate's immutable ancestry; never write an imported pack."""
    from ..gt_resource_companions import collect_image_resource_gt_pack_paths
    from ..campaign_manager import CAMPAIGN
    paths = collect_image_resource_gt_pack_paths(
        source.parent, campaign=CAMPAIGN if getattr(host, "_campaign_controlled", False) else None
    )
    prior = str(data.get("working_gt_pack_path") or "").strip()
    if prior:
        paths.append(Path(prior))
    required = set(data.get("source_gt_revision_ids") or [])
    if data.get("source_gt_revision_id"):
        required.add(data["source_gt_revision_id"])
    for path in dict.fromkeys(map(Path, paths)):
        if path.resolve() == target.root.resolve() or not (path / "manifest.json").is_file():
            continue
        original = ALPRGTPack.open(path)
        record = original.get_plate(plate_id)
        if not record or (required and not required.intersection(record.get("revision_ids") or [])):
            continue
        current = target.get_plate(plate_id)
        if current and current.get("image_id") != record.get("image_id"):
            raise ValueError("Historia tablicy wskazuje inny obraz źródłowy.")
        image = original.get_image(record["image_id"])
        if image:
            target._merge_image_record(image)
        for value in original._geometry_record_map(record).values():
            target._merge_geometry_record(value)
        for value in original._revision_record_map(record).values():
            target._merge_revision_record(value)
        for value in original._layout_revision_record_map(record).values():
            target._merge_layout_revision_record(value)
        target._merge_plate_record(record)


def _apply_reference(host, data, *, plate_id, text, revision_ids, pack_path, source="manual_z3"):
    from .z3_extraction_sources import build_plate_source_gt_hash
    attrs = dict(data.get("plate_attributes") or {})
    attrs.update(plate_annotation_id=plate_id, ground_truth_text=text, ground_truth_source=source)
    gt_hash = build_plate_source_gt_hash(str(data.get("source_image_name") or Path(str(data.get("source_image") or "")).name), attrs)
    attrs["source_gt_hash"] = gt_hash
    data.update(plate_attributes=attrs, source_annotation_id=plate_id,
                ground_truth_text=text, ground_truth_source=source, source_gt_hash=gt_hash,
                source_gt_revision_ids=list(revision_ids),
                source_gt_revision_id=revision_ids[0] if len(revision_ids) == 1 else None,
                working_gt_pack_path=str(pack_path), gt_revision_conflict=False)
    raw = data.get("raw_detection")
    if isinstance(raw, dict):
        data["raw_validation"] = host._build_raw_detection_validation(data, raw.get("characters") or [])


def save_plate_ground_truth(host, data, text, *, prepare=True):
    """Write the canonical revision before changing metadata or allowing approval."""
    text = normalize_plate_ground_truth_text(text)
    if not text:
        raise ValueError("Podaj numer tablicy.")
    path = working_pack_path(host)
    source = Path(str(data.get("source_image") or ""))
    plate_id = _plate_id(data)
    polygon = data.get("source_polygon") or []
    if not source.is_file() or not plate_id or len(polygon) < 4:
        raise ValueError("Brakuje obrazu źródłowego, identyfikatora lub geometrii tablicy. Nie zapisano numeru.")
    path.parent.mkdir(parents=True, exist_ok=True)
    pack = ALPRGTPack.open(path) if (path / "manifest.json").is_file() else ALPRGTPack.create(path, producer=PRODUCER)
    with pack.write_lock():
        _inherit_plate_history(host, pack, data, source, plate_id)
        plate = pack.get_plate(plate_id)
        if plate:
            image_id = str(data.get("source_image_id") or "")
            if image_id and plate.get("image_id") != image_id:
                raise ValueError("Identyfikator tablicy wskazuje inny obraz. Sprawdź źródło danych.")
        else:
            image = pack.add_image(source, producer=PRODUCER)
            pack.ensure_plate(image_id=image["image_id"], polygon=polygon, plate_id=plate_id)
        previous_text = normalize_plate_ground_truth_text(data.get("ground_truth_text"))
        if previous_text and not pack.resolve_ground_truth(plate_id).get("resolved"):
            pack.set_ground_truth(plate_id, previous_text,
                                  source=str(data.get("ground_truth_source") or "legacy_explicit_gt"), producer=PRODUCER)
        revision = pack.set_ground_truth(plate_id, text, source="manual_z3", producer=PRODUCER)
    if data.get("ground_truth_text") == text and data.get("source_gt_revision_ids") == [revision["revision_id"]]:
        return revision
    from .z3_review_runtime import mark_review_edit_started, prepare_working_annotation_from_raw
    from .z3_review_runtime import build_review_reference_snapshot
    data.setdefault("gt_origin_reference", copy.deepcopy(build_review_reference_snapshot(data)))
    had_work = bool(data.get("review_state") or data.get("characters") or (data.get("gold_state") or {}).get("approved"))
    mark_review_edit_started(host, data)
    _apply_reference(host, data, plate_id=plate_id, text=text,
                     revision_ids=[revision["revision_id"]], pack_path=path)
    if prepare:
        if not had_work:
            prepare_working_annotation_from_raw(host, data, plate_id=plate_id, overwrite=True)
        else:
            host._apply_live_gt_assist(data, prepare=True)
    data["status"] = "needs_fix"
    return revision


def refresh_working_ground_truth(host, metadata):
    """Read current revision heads on load; never modify geometry while selecting."""
    try:
        path = working_pack_path(host)
        if not (path / "manifest.json").is_file():
            return False
        pack = ALPRGTPack.open(path)
    except (ValueError, OSError):
        return False
    changed = False
    for data in metadata.values():
        if not isinstance(data, dict):
            continue
        plate_id = _plate_id(data)
        state = pack.resolve_ground_truth(plate_id) if plate_id else {}
        if state.get("conflict"):
            data["gt_revision_conflict"] = True
            host._mark_review_edit_started(data)
            data["status"] = "needs_fix"
            changed = True
            continue
        ids = state.get("revision_ids") or []
        if not state.get("resolved") or not ids:
            continue
        revision = pack.get_revision(ids[0]) or {}
        if revision.get("source") != "manual_z3" and data.get("ground_truth_source") != "manual_z3":
            continue
        text = state.get("text") or ""
        if ids == data.get("source_gt_revision_ids") and text == data.get("ground_truth_text"):
            continue
        _apply_reference(host, data, plate_id=plate_id, text=text, revision_ids=ids, pack_path=path,
                         source=revision.get("source") or "manual_z3")
        host._mark_review_edit_started(data)
        data["status"] = "needs_fix"
        changed = True
    return changed


def edit_active_plate_ground_truth(host):
    if any(getattr(host, flag, False) for flag in ("is_processing", "fast_test_running", "_preview_review_batch_running")):
        return
    data = host._get_preview_active_data(create=False)
    if not isinstance(data, dict):
        return
    current_text = str(host._get_preview_ground_truth_text(data) or "").strip()
    attrs = data.get("plate_attributes")
    attrs = attrs if isinstance(attrs, dict) else {}
    number_source = str(
        data.get("ground_truth_source")
        or attrs.get("ground_truth_source")
        or ""
    ).strip().lower()
    inherited_from_z2 = bool(current_text and number_source == "manual_z2")

    prompt = "Wpisz prawidłowy numer tablicy:"
    if inherited_from_z2:
        prompt = (
            f"Numer odziedziczony z Z2: {current_text}\n\n"
            "Zmiana tutaj utworzy nową wersję numeru w PZ2. "
            "Boxy i wpisane znaki nie zmieniają numeru z Z2 automatycznie.\n\n"
            "Wpisz nowy prawidłowy numer:"
        )

    text = simpledialog.askstring(
        "Numer tablicy",
        prompt,
        initialvalue=current_text,
        parent=host.frame.winfo_toplevel(),
    )
    if text is None:
        return

    normalized_new = normalize_plate_ground_truth_text(text)
    if (
        inherited_from_z2
        and normalized_new
        and normalized_new != normalize_plate_ground_truth_text(current_text)
    ):
        if not messagebox.askyesno(
            "Zmienić numer odziedziczony z Z2?",
            (
                f"Numer z Z2: {current_text}\n"
                f"Nowy numer: {normalized_new}\n\n"
                "Ta operacja zapisze nową wersję numeru w PZ2. Kontynuować?"
            ),
            parent=host.frame.winfo_toplevel(),
        ):
            return

    host._push_preview_history_snapshot()
    try:
        # Materialization is an explicit consequence of setting the number.
        from .z3_review_runtime import _refresh_after_change
        save_plate_ground_truth(host, data, text)
        _refresh_after_change(host, host._preview_active_pid, persist=True, message="Zapisano numer. Sprawdź ramki i zatwierdź tablicę.")
    except Exception as exc:
        messagebox.showerror("Nie zapisano numeru", str(exc), parent=host.frame.winfo_toplevel())


def refresh_plate_ground_truth_ui(host):
    widget = getattr(host, "preview_plate_gt_label", None)
    if widget is None:
        return
    data = host._get_preview_active_data(create=False)
    text = host._get_preview_ground_truth_text(data) if data else ""
    attrs = data.get("plate_attributes") if isinstance(data, dict) else {}
    attrs = attrs if isinstance(attrs, dict) else {}
    number_source = str(
        (data or {}).get("ground_truth_source")
        or attrs.get("ground_truth_source")
        or ""
    ).strip().lower()

    if text and number_source == "manual_z2":
        label_text = f"Numer z Z2: {text}"
    elif text:
        label_text = f"Zapisany numer: {text}"
    else:
        label_text = "Zapisany numer: brak"

    widget.configure(text=label_text)
    host.preview_plate_gt_button.configure(
        text="Zmień numer" if text else "Ustaw numer ręcznie",
        state="normal" if data else "disabled",
    )
