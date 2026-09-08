"""Resolve the AT source used by T02, including the project's approved pool."""

from pathlib import Path


def resolve_t02_review_source(campaign, context=None, *, build_approved_source=None) -> dict:
    context = dict(context or {})
    stored = dict(campaign.get_project_start_plate_source() or {})
    explicit_source = bool(context.get("restore_run_dir") or context.get("xml_path"))
    if explicit_source:
        stored = {}
    elif not any(stored.get(key) for key in ("source_run_path", "source_xml_path")):
        saved = dict((campaign.get_iteration_state() or {}).get("t02_review_source") or {})
        for key in ("restore_run_dir", "xml_path", "input_dir", "input_source"):
            if saved.get(key):
                context.setdefault(key, saved[key])
    run = str(context.get("restore_run_dir") or stored.get("source_run_path") or "").strip()
    xml = str(context.get("xml_path") or stored.get("source_xml_path") or "").strip()
    images = str(context.get("input_dir") or stored.get("source_input_path") or "").strip()
    source = str(context.get("input_source") or "t02_at_review")
    if not run and not xml:
        stats = dict(campaign.get_plate_approved_set_stats() or {})
        if int(stats.get("plates", 0) or 0) <= 0:
            return {"ok": False, "reason": "missing_t02_source", "message":
                    "T02 nie ma anotacji tablic do kontroli. Wskaż import AT w zasobach bramki albo przygotuj tablice przez T01."}
        if build_approved_source is None:
            return {"ok": True, "needs_approved_source": True}
        approved = dict(build_approved_source() or {})
        run = str(approved.get("run_dir") or "")
        xml = str(approved.get("xml_path") or "")
        images = str(approved.get("images_dir") or "")
        source = "t02_project_approved"
        if not run or not xml or not images:
            return {"ok": False, "reason": "unavailable_approved_source", "message":
                    "Nie udało się odtworzyć tablic z puli projektu. Sprawdź dostępność obrazów źródłowych."}
    xml_path = Path(xml) if xml else Path(run) / "annotations.xml"
    if not xml_path.is_file():
        return {"ok": False, "reason": "missing_t02_xml", "message":
                "Nie znaleziono pliku anotacji AT. Wskaż istniejące źródło w zasobach T02."}
    run_path = Path(run) if run else xml_path.parent
    if not images and (run_path / "images").is_dir():
        images = str(run_path / "images")
    if images and not Path(images).is_dir():
        return {"ok": False, "reason": "missing_t02_images", "message":
                "Nie znaleziono obrazów dla anotacji AT. Popraw źródło obrazów w zasobach T02."}
    resolved = {"restore_run_dir": str(run_path), "xml_path": str(xml_path), "input_source": source}
    if images:
        resolved["input_dir"] = images
    if source == "t02_project_approved" and build_approved_source is not None:
        campaign.upsert_iteration_state(updates={"t02_review_source": resolved})
    return {"ok": True, "context": resolved}
