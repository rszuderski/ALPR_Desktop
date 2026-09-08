#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Path resolution helpers for Z3 character workflow."""

import json
from pathlib import Path

from ..campaign_manager import CAMPAIGN
from ..config import CONFIG, logger
from ..project_cache import PROJECT_CACHE
from .z3_metadata_cache import read_preview_metadata


def _has_campaign_preview_context(host) -> bool:
    if bool(getattr(getattr(host, "app", None), "campaign_free_mode", False)):
        return False
    try:
        return bool(CAMPAIGN.get_active_project_name())
    except Exception:
        return False


def _in_campaign_step3_context(host) -> bool:
    return bool(getattr(host, "_step3_linear_mode", False) and _has_campaign_preview_context(host))


def get_step3_chars_root_dir(host, ensure_exists: bool = False) -> Path:
    in_campaign_context = _in_campaign_step3_context(host)

    chars_root = getattr(host, "_campaign_chars_dir", None) if in_campaign_context else None
    if not chars_root and in_campaign_context:
        try:
            campaign_chars_root = CAMPAIGN.get_dir("chars")
            if campaign_chars_root is not None:
                chars_root = campaign_chars_root
        except Exception:
            chars_root = None
    chars_root = chars_root or CONFIG.DIR_3_CHARS
    root = Path(chars_root).absolute()
    if ensure_exists:
        root.mkdir(parents=True, exist_ok=True)
    return root


def get_step3_datasets_root_dir(host, ensure_exists: bool = False) -> Path:
    in_campaign_context = _in_campaign_step3_context(host)

    datasets_root = getattr(host, "_campaign_datasets_dir", None) if in_campaign_context else None
    if not datasets_root and in_campaign_context:
        try:
            campaign_datasets_root = CAMPAIGN.get_dir("datasets")
            if campaign_datasets_root is not None:
                datasets_root = campaign_datasets_root
        except Exception:
            datasets_root = None
    datasets_root = datasets_root or CONFIG.get_datasets_dir("char")
    root = Path(datasets_root).absolute()
    if ensure_exists:
        root.mkdir(parents=True, exist_ok=True)
    return root


def get_step3_char_classification_datasets_root_dir(host, ensure_exists: bool = False) -> Path:
    in_campaign_context = _in_campaign_step3_context(host)

    datasets_root = getattr(host, "_campaign_datasets_dir", None) if in_campaign_context else None
    if not datasets_root and in_campaign_context:
        try:
            campaign_datasets_root = CAMPAIGN.get_dir("datasets")
            if campaign_datasets_root is not None:
                datasets_root = campaign_datasets_root
        except Exception:
            datasets_root = None

    if datasets_root:
        root = Path(datasets_root).absolute() / CONFIG.get_char_classification_datasets_dir().name
    else:
        root = CONFIG.get_char_classification_datasets_dir().absolute()

    if ensure_exists:
        root.mkdir(parents=True, exist_ok=True)
    return root


def get_step3_summary_dir(host) -> Path:
    try:
        preview_raw = str(host.preview_dir_var.get() or "").strip()
    except Exception:
        preview_raw = ""
    preview_dir = Path(preview_raw) if preview_raw else None
    if preview_dir and preview_dir.exists():
        return preview_dir

    campaign_chars_dir = getattr(host, "_campaign_chars_dir", None)
    if campaign_chars_dir:
        path = Path(campaign_chars_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    fallback = Path(CONFIG.DIR_3_CHARS).absolute()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def get_preview_dir_plate_count(host, preview_dir=None) -> int:
    try:
        preview_dir_raw = str(preview_dir if preview_dir is not None else (host.preview_dir_var.get() or "")).strip()
    except Exception:
        preview_dir_raw = ""
    if not preview_dir_raw:
        return 0

    try:
        preview_path = Path(preview_dir_raw)
        meta_path = preview_path / "metadata.json"
        images_dir = preview_path / "images"
        if not meta_path.exists() or not images_dir.exists() or not images_dir.is_dir():
            return 0

        manifest_path = preview_path / "extract_manifest.json"
        try:
            signature = (
                meta_path.stat().st_mtime_ns,
                meta_path.stat().st_size,
                manifest_path.stat().st_mtime_ns if manifest_path.exists() else 0,
                manifest_path.stat().st_size if manifest_path.exists() else 0,
            )
            cache_key = str(preview_path.resolve())
            cached = dict(getattr(host, "_preview_dir_plate_count_cache", {}).get(cache_key, {}) or {})
            if cached.get("signature") == signature:
                return int(cached.get("count", 0) or 0)
        except Exception:
            signature = None
            cache_key = str(preview_path)

        if manifest_path.exists():
            try:
                manifest = PROJECT_CACHE.load_json(manifest_path, default={})
                if isinstance(manifest, dict):
                    manifest_count = int(manifest.get("plate_count", 0) or 0)
                    if manifest_count > 0:
                        try:
                            host._preview_dir_plate_count_cache[cache_key] = {
                                "signature": signature,
                                "count": manifest_count,
                            }
                        except Exception:
                            pass
                        return manifest_count
            except Exception:
                pass

        loaded = read_preview_metadata(host, meta_path)
        if not isinstance(loaded, dict):
            return 0

        count = sum(1 for pid in loaded.keys() if str(pid or "").strip())
        try:
            host._preview_dir_plate_count_cache[cache_key] = {
                "signature": signature,
                "count": count,
            }
        except Exception:
            pass
        return count
    except Exception:
        return 0


def preview_dir_has_plate_entries(host, preview_dir=None) -> bool:
    return get_preview_dir_plate_count(host, preview_dir) > 0


def campaign_preview_meets_min_extracted_plate_count(host, preview_dir=None) -> bool:
    if not host._is_campaign_char_step3_context():
        return True
    return bool(
        get_preview_dir_plate_count(host, preview_dir)
        >= host._get_campaign_step3_min_extracted_plate_count()
    )


def campaign_preview_is_below_min_extracted_plate_count(host, preview_dir=None) -> bool:
    if not host._is_campaign_char_step3_context():
        return False
    count = get_preview_dir_plate_count(host, preview_dir)
    return bool(0 <= int(count) < host._get_campaign_step3_min_extracted_plate_count())


def get_campaign_expected_step3_preview_plate_count(host) -> int:
    try:
        in_campaign = host._is_campaign_char_step3_context()
    except Exception:
        in_campaign = False
    if not in_campaign:
        return 0

    try:
        stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
        return max(0, int(stats.get("plates", 0) or 0))
    except Exception:
        return 0


def preview_dir_is_campaign_inflated(host, preview_dir=None) -> bool:
    expected = get_campaign_expected_step3_preview_plate_count(host)
    if expected <= 0:
        return False

    count = get_preview_dir_plate_count(host, preview_dir)
    if count <= 0:
        return False

    # Mały margines chroni legalne różnice robocze, ale odcina mnożenie 2x/3x.
    max_reasonable = max(int(expected) + 2, int(round(float(expected) * 1.25)))
    return bool(int(count) > int(max_reasonable))


def is_usable_step3_preview_dir(
    host,
    preview_dir=None,
    *,
    require_plates: bool = False,
    check_campaign_inflated: bool = True,
) -> bool:
    preview_dir_raw = str(preview_dir or "").strip()
    if not preview_dir_raw:
        return False
    try:
        candidate = Path(preview_dir_raw)
        if not (
            candidate.exists()
            and candidate.is_dir()
            and (candidate / "metadata.json").exists()
            and (candidate / "images").exists()
            and (not require_plates or preview_dir_has_plate_entries(host, candidate))
        ):
            return False
        if bool(check_campaign_inflated) and preview_dir_is_campaign_inflated(host, candidate):
            logger.debug(
                "Pominięto napompowany preview run E3/PZ2: %s (count=%s, expected=%s)",
                candidate,
                get_preview_dir_plate_count(host, candidate),
                get_campaign_expected_step3_preview_plate_count(host),
            )
            return False
        return True
    except Exception:
        return False


def get_saved_step3_preview_dir(host, require_plates: bool = False) -> str:
    in_campaign_context = _has_campaign_preview_context(host)
    if not in_campaign_context:
        return ""

    try:
        saved_preview_dir = str(CAMPAIGN.get_step3_preview_dir() or "").strip()
    except Exception:
        saved_preview_dir = ""
    if not saved_preview_dir:
        return ""
    if is_usable_step3_preview_dir(
        host,
        saved_preview_dir,
        require_plates=require_plates,
        check_campaign_inflated=False,
    ):
        return str(Path(saved_preview_dir))
    return ""


def get_preferred_step3_preview_dir(host, require_plates: bool = False, allow_fallback: bool = True) -> str:
    in_campaign_context = _has_campaign_preview_context(host)

    if in_campaign_context:
        saved_preview_dir = get_saved_step3_preview_dir(host, require_plates=require_plates)
        if saved_preview_dir:
            return saved_preview_dir

    try:
        current_preview_dir = str(host.preview_dir_var.get() or "").strip()
    except Exception:
        current_preview_dir = ""
    if current_preview_dir and is_usable_step3_preview_dir(host, current_preview_dir, require_plates=require_plates):
        return str(Path(current_preview_dir))

    # In free mode a status refresh is not a request to select a historical run.
    # PZ1 explicitly supplies the result or searches by the selected XML/images.
    if allow_fallback and in_campaign_context:
        return find_latest_preview_run_dir(host, require_plates=require_plates)
    return ""


def find_latest_preview_run_dir(host, require_plates: bool = False) -> str:
    """
    Szuka najnowszego poprawnego preview runu w katalogu chars projektu.
    Poprawny preview run to katalog zawierający:
    - metadata.json
    - images/
    """
    try:
        chars_root = host._get_step3_chars_root_dir(ensure_exists=False)
    except Exception:
        chars_root = None
    if not chars_root:
        return ""

    root = Path(chars_root)
    if not root.exists() or not root.is_dir():
        return ""

    candidates = []
    try:
        # Inspect run metadata, not every crop file in the images/ trees.
        for meta_file in root.rglob("metadata.json"):
            path = meta_file.parent
            if path == root or not meta_file.is_file():
                continue

            images_dir = path / "images"

            if meta_file.exists() and images_dir.exists() and images_dir.is_dir():
                if require_plates and not preview_dir_has_plate_entries(host, path):
                    continue
                if preview_dir_is_campaign_inflated(host, path):
                    continue
                candidates.append(path)
    except Exception:
        return ""

    if not candidates:
        return ""

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0])
