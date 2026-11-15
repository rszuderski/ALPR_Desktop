"""UI/runtime glue for GT packs that accompany an image resource O."""

from __future__ import annotations

from pathlib import Path
from tkinter import messagebox

from ..campaign_manager import CAMPAIGN
from ..config import SESSION, logger
from ..gt_resource_companions import (
    DEFAULT_WORKING_PACK_NAME,
    discover_gt_pack_companions,
    get_campaign_gt_companion_paths,
    import_and_bind_campaign_gt_companions,
    bind_campaign_gt_companions,
)
from . import z2_gt_pack_runtime


FREE_BINDINGS_SCHEMA = "alpr.gt.free-image-resource-bindings.v1"
FREE_BINDINGS_SESSION_KEY = "gt_image_resource_bindings"


def _resource_key(path: Path | str) -> str:
    try:
        return str(Path(path).resolve()).replace("\\", "/").lower()
    except Exception:
        return str(path or "").replace("\\", "/").strip().lower()


def _format_prompt(discovery: dict, *, project_import: bool) -> str:
    packs = list(discovery.get("ground_truth", []) or [])
    total_images = sum(int(item.get("images", 0) or 0) for item in packs)
    total_plates = sum(int(item.get("plates", 0) or 0) for item in packs)
    total_gt = sum(int(item.get("gt_set", 0) or 0) for item in packs)
    total_layout = sum(int(item.get("layout_set", 0) or 0) for item in packs)

    if len(packs) == 1:
        header = f"W wybranym katalogu znaleziono Ground Truth:\n{packs[0].get('name') or 'GT Pack'}"
    else:
        header = f"W wybranym katalogu znaleziono {len(packs)} pakiety Ground Truth."

    stats = (
        f"\n\nPacki opisują łącznie:\n"
        f"• obrazy: {total_images}\n"
        f"• anotacje tablic: {total_plates}\n"
        f"• tablice z tekstem GT: {total_gt}"
    )
    if total_layout:
        stats += f"\n• tablice z GT układu 1R/2R: {total_layout}"

    if project_import:
        tail = (
            "\n\nZaimportować GT do projektu i powiązać je z tym zasobem O? "
            "Źródła zostaną skopiowane do _campaign_state/ground_truth/imported i używane read-only."
        )
    else:
        tail = (
            "\n\nPodłączyć znalezione GT jako źródło dla tego katalogu obrazów? "
            f"Niezależnie od wyboru nowe poprawki Z2 będą zapisywane do {DEFAULT_WORKING_PACK_NAME}."
        )
    return header + stats + tail


def _confirm(host, title: str, message: str, *, parent=None) -> bool:
    app = getattr(host, "app", None)
    themed = getattr(app, "themed_confirm", None)
    if callable(themed):
        try:
            return bool(themed(title, message, parent=parent))
        except TypeError:
            try:
                return bool(themed(title, message))
            except Exception:
                pass
        except Exception:
            pass
    return bool(messagebox.askyesno(title, message, parent=parent))


def offer_campaign_gt_import(host, image_dir: Path | str, *, parent=None) -> dict:
    """Discover/import GT after campaign O has been accepted."""
    image_dir = Path(image_dir)
    discovery = discover_gt_pack_companions(image_dir)
    packs = list(discovery.get("ground_truth", []) or [])
    if not packs:
        return {"found": False, "imported": False, "discovery": discovery}

    accepted = _confirm(
        host,
        "Wykryto Ground Truth",
        _format_prompt(discovery, project_import=True),
        parent=parent or getattr(host, "frame", None),
    )
    try:
        iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        iteration_num = 1

    if not accepted:
        # An explicit decline belongs to this O/package, not to global Z2 mounts.
        try:
            bind_campaign_gt_companions(
                CAMPAIGN,
                image_dir=image_dir,
                imported_packs=[],
                iteration_num=iteration_num,
            )
        except Exception as exc:
            logger.debug("Nie udało się zapisać pominięcia companion GT: %s", exc)
        return {"found": True, "imported": False, "discovery": discovery}

    result = import_and_bind_campaign_gt_companions(
        CAMPAIGN,
        image_dir=image_dir,
        discoveries=packs,
        iteration_num=iteration_num,
    )
    paths = list(result.get("paths", []) or [])

    # If Z2 is already instantiated, make the new O binding visible immediately.
    try:
        annotation_tab = getattr(host.app, "tabs", {}).get("annotation")
    except Exception:
        annotation_tab = None
    if annotation_tab is not None:
        try:
            z2_gt_pack_runtime.set_gt_resource_binding(
                annotation_tab,
                source_paths=paths,
                resource_key=f"campaign:{_resource_key(image_dir)}",
            )
        except Exception:
            pass

    return {
        "found": True,
        "imported": True,
        "paths": paths,
        "discovery": discovery,
        "bundle": result.get("bundle", {}),
    }


def activate_campaign_gt_binding(host, image_dir: Path | str) -> list[Path]:
    """Activate only companions that belong to the current campaign O."""
    try:
        iteration_num = int(CAMPAIGN.get_current_iteration_num() or 1)
    except Exception:
        iteration_num = 1
    try:
        paths = get_campaign_gt_companion_paths(
            CAMPAIGN,
            image_dir=Path(image_dir),
            iteration_num=iteration_num,
        )
    except Exception as exc:
        logger.debug("Nie udało się rozwiązać companionów GT dla O: %s", exc)
        paths = []
    z2_gt_pack_runtime.set_gt_resource_binding(
        host,
        source_paths=paths,
        resource_key=f"campaign:{_resource_key(image_dir)}",
    )
    return paths


def _load_free_bindings() -> dict:
    try:
        payload = SESSION.get("annotation", FREE_BINDINGS_SESSION_KEY, {}) if SESSION else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict) or payload.get("schema") not in (None, "", FREE_BINDINGS_SCHEMA):
        payload = {}
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        bindings = {}
    return {"schema": FREE_BINDINGS_SCHEMA, "bindings": dict(bindings)}


def _save_free_bindings(payload: dict) -> None:
    bindings = dict(payload.get("bindings", {}) or {})
    if len(bindings) > 64:
        for key in list(bindings)[:-64]:
            bindings.pop(key, None)
    safe = {"schema": FREE_BINDINGS_SCHEMA, "bindings": bindings}
    try:
        if SESSION:
            SESSION.set("annotation", FREE_BINDINGS_SESSION_KEY, safe)
            SESSION.save_session()
    except Exception as exc:
        logger.debug("Nie udało się zapisać powiązań GT free mode: %s", exc)


def prepare_free_mode_gt_binding(host, image_dir: Path | str, *, parent=None) -> dict:
    """Automatically bind valid GT companions for one free-mode image resource.

    Free mode no longer exposes read-only/writable mount decisions to the
    operator. Every valid direct-child source ``*.alprgt`` is mounted read-only,
    while ``current_work.alprgt`` is the canonical write target. Discovery
    already excludes that canonical working pack from source candidates.

    Older saved ``accepted=False`` decisions are intentionally ignored and
    migrated to the automatic companion policy.
    """
    image_dir = Path(image_dir)
    key = _resource_key(image_dir)
    payload = _load_free_bindings()
    discovery = discover_gt_pack_companions(image_dir)
    packs = list(discovery.get("ground_truth", []) or [])
    signature = str(discovery.get("signature") or "")
    working_path = image_dir / DEFAULT_WORKING_PACK_NAME

    source_paths = []
    seen = set()
    for item in packs:
        raw = str((item or {}).get("path") or "").strip()
        if not raw:
            continue
        path = Path(raw)
        try:
            path_key = str(path.resolve()).replace("\\", "/").lower()
        except Exception:
            path_key = str(path).replace("\\", "/").lower()
        if not path_key or path_key in seen:
            continue
        if not path.is_dir() or not (path / "manifest.json").is_file():
            continue
        seen.add(path_key)
        source_paths.append(path)

    previous = dict(payload["bindings"].get(key) or {})
    binding = {
        "signature": signature,
        "policy": "auto_companion_v1",
        "accepted": bool(source_paths),
        "source_paths": [str(path) for path in source_paths],
        "working_path": str(working_path),
    }
    payload["bindings"][key] = binding

    if previous != binding:
        _save_free_bindings(payload)

    z2_gt_pack_runtime.set_gt_resource_binding(
        host,
        source_paths=source_paths,
        working_path=working_path,
        resource_key=f"free:{key}",
    )

    return {
        "found": bool(packs),
        "accepted": bool(source_paths),
        "auto_bound": True,
        "paths": source_paths,
        "working_path": str(working_path),
        "discovery": discovery,
        "restored_binding": bool(
            previous
            and previous.get("policy") == "auto_companion_v1"
            and previous == binding
        ),
        "invalid_count": len(list(discovery.get("invalid", []) or [])),
    }
