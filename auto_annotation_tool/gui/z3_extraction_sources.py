#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z3/PZ1 source matching helpers for XML/image inputs."""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from ..campaign_manager import CAMPAIGN
from ..config import CONFIG, logger
from ..project_cache import PROJECT_CACHE


def normalize_xml_image_relpath(raw_name: str) -> str:
    raw = str(raw_name or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]

    parts = [part for part in PurePosixPath(raw).parts if part not in ("", ".")]
    return "/".join(parts)


def xml_relpath_to_path(rel_path: str) -> Path:
    parts = [part for part in PurePosixPath(rel_path).parts if part not in ("", ".")]
    return Path(*parts) if parts else Path()


def read_xml_image_names(xml_path: Path) -> list[str]:
    tree = ET.parse(xml_path)
    names = []

    for image_el in tree.getroot().findall(".//image"):
        normalized = normalize_xml_image_relpath(image_el.get("name") or "")
        if normalized:
            names.append(normalized)

    return list(dict.fromkeys(names))


def count_xml_plate_cut_targets(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    image_elements = {
        str(image_el.get("name") or ""): image_el
        for image_el in tree.getroot().findall(".//image")
        if str(image_el.get("name") or "").strip()
    }
    plate_count = 0
    images_with_plates = 0
    geometry_parts: list[str] = []

    for raw_image_name, image_el in image_elements.items():
        image_plate_count = 0
        image_geometry_parts: list[str] = []
        for poly in image_el.findall(".//polygon[@label='plate']"):
            try:
                points = [
                    tuple(map(float, raw_point.split(",")))
                    for raw_point in str(poly.get("points") or "").split(";")
                    if raw_point.strip()
                ]
            except Exception:
                continue
            if len(points) >= 4:
                image_plate_count += 1
                image_geometry_parts.append(
                    ";".join(f"{float(x):.3f},{float(y):.3f}" for x, y in points[:4])
                )

        if image_plate_count > 0:
            images_with_plates += 1
            plate_count += image_plate_count
            geometry_parts.append(
                "|".join(
                    (
                        normalize_xml_image_relpath(raw_image_name),
                        str(image_el.get("width") or ""),
                        str(image_el.get("height") or ""),
                        "#".join(image_geometry_parts),
                    )
                )
            )

    return {
        "plate_count": int(plate_count),
        "images_with_plates": int(images_with_plates),
        "xml_images_total": int(len(image_elements)),
        "xml_plate_geometry_hash": hashlib.sha1("\n".join(geometry_parts).encode("utf-8")).hexdigest()
        if geometry_parts
        else "",
    }


def extract_source_manifest_path(host: "CharacterAnnotationTab", preview_dir: Path | str | None = None) -> Path | None:
    preview_dir_raw = str(
        preview_dir
        if preview_dir is not None
        else (host.preview_dir_var.get() if hasattr(host, "preview_dir_var") else "")
        or ""
    ).strip()
    if not preview_dir_raw:
        return None
    try:
        return Path(preview_dir_raw) / "extract_manifest.json"
    except Exception:
        return None


def current_extract_source_payload(host: "CharacterAnnotationTab") -> dict:
    def _path_text(var_name: str) -> str:
        try:
            raw = str(getattr(host, var_name).get() or "").strip()
        except Exception:
            raw = ""
        if not raw:
            return ""
        try:
            return str(Path(raw).resolve())
        except Exception:
            try:
                return str(Path(raw))
            except Exception:
                return raw

    return {
        "annotation_run_dir": _path_text("annotation_run_dir_var"),
        "xml_path": _path_text("xml_path_var"),
        "images_dir": _path_text("images_dir_var"),
    }


def current_extract_source_signature(host: "CharacterAnnotationTab") -> dict:
    source = current_extract_source_payload(host)
    xml_path_raw = str(source.get("xml_path") or "").strip()
    signature = {
        "xml_mtime_ns": 0,
        "xml_size": 0,
        "xml_plate_count": 0,
        "xml_images_with_plates": 0,
        "xml_images_total": 0,
        "xml_plate_geometry_hash": "",
    }
    if not xml_path_raw:
        return signature
    try:
        xml_path = Path(xml_path_raw)
        stat = xml_path.stat()
        signature["xml_mtime_ns"] = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)) or 0)
        signature["xml_size"] = int(getattr(stat, "st_size", 0) or 0)
        cache_key = (
            str(xml_path.resolve()),
            int(signature["xml_mtime_ns"]),
            int(signature["xml_size"]),
        )
    except Exception:
        return signature

    try:
        cache = getattr(host, "_extract_source_signature_cache", None)
        if not isinstance(cache, dict):
            cache = {}
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            signature.update(cached)
            return signature
    except Exception:
        cache = {}

    try:
        counts = count_xml_plate_cut_targets(xml_path)
        counted = {
            "xml_plate_count": int(counts.get("plate_count", 0) or 0),
            "xml_images_with_plates": int(counts.get("images_with_plates", 0) or 0),
            "xml_images_total": int(counts.get("xml_images_total", 0) or 0),
            "xml_plate_geometry_hash": str(counts.get("xml_plate_geometry_hash") or "").strip(),
        }
        signature.update(counted)
        try:
            cache[cache_key] = counted
            if len(cache) > 8:
                for old_key in list(cache.keys())[:-8]:
                    cache.pop(old_key, None)
            host._extract_source_signature_cache = cache
        except Exception:
            pass
    except Exception:
        pass
    return signature


def current_extract_source_payload_with_signature(host: "CharacterAnnotationTab") -> dict:
    payload = current_extract_source_payload(host)
    payload.update(current_extract_source_signature(host))
    return payload


def extract_manifest_matches_current_source(
    host: "CharacterAnnotationTab",
    manifest: dict,
) -> bool:
    if not isinstance(manifest, dict):
        return True
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    current = current_extract_source_payload(host)
    current_signature = current_extract_source_signature(host)
    saved_hash = str(source.get("xml_plate_geometry_hash") or "").strip()
    current_hash = str(current_signature.get("xml_plate_geometry_hash") or "").strip()
    geometry_hash_matches = bool(saved_hash and current_hash and saved_hash == current_hash)

    def _saved_current_int_match(key: str) -> bool:
        saved_value = int(source.get(key, 0) or 0)
        current_value = int(current_signature.get(key, 0) or 0)
        return bool(saved_value > 0 and current_value > 0 and saved_value == current_value)

    legacy_signature_matches = bool(
        not (saved_hash and current_hash)
        and _saved_current_int_match("xml_size")
        and _saved_current_int_match("xml_plate_count")
        and _saved_current_int_match("xml_images_with_plates")
        and _saved_current_int_match("xml_images_total")
    )
    source_semantics_match = bool(geometry_hash_matches or legacy_signature_matches)

    for key in ("images_dir", "xml_path", "annotation_run_dir"):
        saved_value = str(source.get(key) or "").strip()
        current_value = str(current.get(key) or "").strip()
        if not saved_value or not current_value:
            continue
        if not host._paths_equivalent(saved_value, current_value):
            if source_semantics_match:
                continue
            return False

    if saved_hash and current_hash:
        if saved_hash != current_hash:
            return False
    else:
        # Starsze manifesty nie mają hasha geometrii. W takim przypadku nie ufamy
        # samemu mtime, bo XML bywa przepisywany bez zmiany anotacji.
        for key in ("xml_size",):
            saved_value = int(source.get(key, 0) or 0)
            current_value = int(current_signature.get(key, 0) or 0)
            if saved_value > 0 and current_value > 0 and saved_value != current_value:
                return False

    for key in ("xml_images_with_plates", "xml_images_total"):
        saved_value = int(source.get(key, 0) or 0)
        current_value = int(current_signature.get(key, 0) or 0)
        if saved_value > 0 and current_value > 0 and saved_value != current_value:
            return False

    current_plate_count = int(current_signature.get("xml_plate_count", 0) or 0)
    saved_plate_count = int(source.get("xml_plate_count", 0) or 0)
    if saved_plate_count <= 0:
        saved_plate_count = int(manifest.get("plate_count", 0) or 0)
    if current_plate_count > 0 and saved_plate_count > 0 and current_plate_count != saved_plate_count:
        return False

    return True


def get_extract_preview_manifest_state(host: "CharacterAnnotationTab", preview_dir=None) -> dict:
    preview_dir_raw = str(
        preview_dir
        if preview_dir is not None
        else (host.preview_dir_var.get() if hasattr(host, "preview_dir_var") else "")
        or ""
    ).strip()
    state = {
        "ready": False,
        "manifest_exists": False,
        "plate_count": 0,
        "source_matches": False,
        "meets_minimum": False,
    }
    if not preview_dir_raw:
        return state

    try:
        preview_path = Path(preview_dir_raw)
        manifest_path = preview_path / "extract_manifest.json"
        if not manifest_path.exists():
            return state
        state["manifest_exists"] = True
        manifest = PROJECT_CACHE.load_json(manifest_path, default={})
        if not isinstance(manifest, dict):
            return state
        plate_count = int(manifest.get("plate_count", 0) or 0)
        state["plate_count"] = max(0, plate_count)
        if plate_count <= 0:
            return state

        source_matches = extract_manifest_matches_current_source(host, manifest)

        meets_minimum = True
        try:
            if host._is_campaign_char_step3_context():
                meets_minimum = plate_count >= host._get_campaign_step3_min_extracted_plate_count()
        except Exception:
            meets_minimum = True

        state["source_matches"] = bool(source_matches)
        state["meets_minimum"] = bool(meets_minimum)
        state["ready"] = bool(source_matches and meets_minimum)
        return state
    except Exception:
        return state


def preview_matches_current_extract_source(host: "CharacterAnnotationTab", preview_dir=None) -> bool:
    manifest_path = extract_source_manifest_path(host, preview_dir)
    current = current_extract_source_payload(host)
    try:
        binding_result = dict(getattr(host, "_extract_last_source_binding_result", {}) or {})
        expected_plate_count = int(binding_result.get("plate_count") or 0)
    except Exception:
        binding_result = {}
        expected_plate_count = 0

    binding_matches_current_source = False
    if expected_plate_count > 0 and bool(binding_result.get("ok")):
        binding_matches_current_source = True
        for binding_key, current_key in (
            ("source_xml", "xml_path"),
            ("source_images_dir", "images_dir"),
        ):
            binding_value = str(binding_result.get(binding_key) or "").strip()
            current_value = str(current.get(current_key) or "").strip()
            if not binding_value or not current_value:
                binding_matches_current_source = False
                break
            try:
                if not host._paths_equivalent(binding_value, current_value):
                    binding_matches_current_source = False
                    break
            except Exception:
                binding_matches_current_source = False
                break

    if binding_matches_current_source:
        actual_plate_count = int(host._get_preview_dir_plate_count(preview_dir) or 0)
        if actual_plate_count != expected_plate_count:
            return False

    if manifest_path is None or not manifest_path.exists():
        try:
            current_plate_count = int(current_extract_source_signature(host).get("xml_plate_count", 0) or 0)
            preview_plate_count = int(host._get_preview_dir_plate_count(preview_dir) or 0)
            if current_plate_count > 0 and preview_plate_count > 0 and current_plate_count != preview_plate_count:
                return False
        except Exception:
            pass
        return True
    try:
        manifest = PROJECT_CACHE.load_json(manifest_path, default={})
    except Exception:
        manifest = {}
    if not isinstance(manifest, dict):
        return True

    return extract_manifest_matches_current_source(host, manifest)


def find_latest_extract_preview_run_dir(host: "CharacterAnnotationTab", require_plates: bool = False) -> str:
    """
    Szuka najnowszego preview zgodnego z aktualnym źródłem PZ1.
    To chroni PZ2 przed podpięciem starego runu, gdy PZ1 przejął już nowy XML.
    """
    from .z3_paths import _has_campaign_preview_context

    if not _has_campaign_preview_context(host):
        # Without an explicit source there is nothing to match. In particular,
        # clearing a session must not silently reattach an unrelated old run.
        xml_var = getattr(host, "xml_path_var", None)
        images_var = getattr(host, "images_dir_var", None)
        if not (xml_var and str(xml_var.get() or "").strip()
                and images_var and str(images_var.get() or "").strip()):
            return ""
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
        for meta_path in root.rglob("metadata.json"):
            p = meta_path.parent
            if p == root or not meta_path.is_file():
                continue
            try:
                usable = host._is_usable_step3_preview_dir(
                    p,
                    require_plates=require_plates,
                    check_campaign_inflated=False,
                )
            except TypeError:
                usable = host._is_usable_step3_preview_dir(p, require_plates=require_plates)
            if not usable:
                continue
            if not preview_matches_current_extract_source(host, p):
                continue
            candidates.append(p)
    except Exception:
        return ""

    if not candidates:
        return ""

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def write_extract_source_manifest(host: "CharacterAnnotationTab", run_dir: Path, *, plate_count: int) -> None:
    try:
        payload = {
            "kind": "pz1_plate_extraction",
            "status": "completed",
            "plate_count": int(plate_count or 0),
            "source": current_extract_source_payload_with_signature(host),
        }
        host._atomic_write_json(Path(run_dir) / "extract_manifest.json", payload)
    except Exception as exc:
        logger.debug(f"Nie udało się zapisać manifestu wyodrębniania PZ1: {exc}")


def is_extract_preview_ready(host: "CharacterAnnotationTab", preview_dir=None) -> bool:
    preview_dir_raw = str(
        preview_dir
        if preview_dir is not None
        else (host.preview_dir_var.get() if hasattr(host, "preview_dir_var") else "")
        or ""
    ).strip()
    if not preview_dir_raw:
        return False
    try:
        return bool(
            host._is_usable_step3_preview_dir(preview_dir_raw, require_plates=True)
            and preview_matches_current_extract_source(host, preview_dir_raw)
            and host._campaign_preview_meets_min_extracted_plate_count(preview_dir_raw)
        )
    except Exception:
        return False


def is_extract_preview_ready_fast(host: "CharacterAnnotationTab", preview_dir=None) -> bool:
    return bool(get_extract_preview_manifest_state(host, preview_dir).get("ready"))


def get_extract_preview_ready_count(host: "CharacterAnnotationTab", preview_dir=None) -> int:
    preview_dir_raw = str(
        preview_dir
        if preview_dir is not None
        else (host.preview_dir_var.get() if hasattr(host, "preview_dir_var") else "")
        or ""
    ).strip()
    if not preview_dir_raw:
        return 0
    try:
        manifest_state = get_extract_preview_manifest_state(host, preview_dir_raw)
        manifest_count = int(manifest_state.get("plate_count", 0) or 0)
        if manifest_count > 0:
            return manifest_count
        return int(host._get_preview_dir_plate_count(preview_dir_raw) or 0)
    except Exception:
        return 0


def extract_source_plate_tokens_from_filename(filename: str) -> list[str]:
    stem = Path(str(filename or "")).stem.upper()
    if not stem:
        return []

    ignore_tokens = {
        "PLATE",
        "PLATES",
        "TABLICA",
        "TABLICE",
        "IMG",
        "IMAGE",
        "PHOTO",
        "RAW",
        "SOURCE",
        "RUN",
        "SAMPLE",
        "SAMPLES",
        "FRAME",
        "CAPTURE",
        "PREVIEW",
        "FILE",
        "PLIK",
        "CROP",
    }
    parts = [
        str(match.group(0) or "").strip().upper()
        for match in re.finditer(r"[A-Z0-9]+", stem)
        if str(match.group(0) or "").strip()
    ]
    parts = [part for part in parts if part not in ignore_tokens]
    if len(parts) > 1 and parts[-1].isdigit():
        previous_plate_like = any(
            3 <= len(part) <= 12
            and (any(ch.isalpha() for ch in part) or part.isdigit())
            for part in parts[:-1]
        )
        if previous_plate_like:
            parts = parts[:-1]

    tokens: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part in seen or len(part) < 3 or len(part) > 12:
            continue
        if not any(ch.isalpha() for ch in part) and not part.isdigit():
            continue
        seen.add(part)
        tokens.append(part)
    return tokens


def plate_cut_reading_order_key(index_and_detection):
    original_index, detection = index_and_detection
    try:
        x1, y1, x2, y2 = detection.bbox
        height = max(1.0, float(y2) - float(y1))
        row_bucket = int(round(float(y1) / max(24.0, height * 0.75)))
        return (row_bucket, float(y1), float(x1), int(original_index))
    except Exception:
        return (0, 0.0, 0.0, int(original_index))


def evaluate_images_dir_for_xml(images_dir: Path, xml_image_names: list[str]) -> dict:
    matched = 0
    missing = []

    for rel_name in xml_image_names:
        candidate = images_dir / xml_relpath_to_path(rel_name)
        if candidate.exists() and candidate.is_file():
            matched += 1
        else:
            missing.append(rel_name)

    return {
        "images_dir": images_dir,
        "total": len(xml_image_names),
        "matched": matched,
        "missing_count": len(missing),
        "missing": missing,
    }


def summarize_missing_xml_images(missing: list[str], limit: int = 3) -> str:
    if not missing:
        return ""

    preview = ", ".join(missing[:limit])
    if len(missing) > limit:
        preview += ", ..."
    return f"Brakuje {len(missing)} plików z XML, np. {preview}."


def is_path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def get_preferred_source_roots() -> list[Path]:
    roots = []

    try:
        campaign_raw = CAMPAIGN.get_dir("raw")
        if campaign_raw:
            roots.append(Path(campaign_raw))
    except Exception:
        pass

    try:
        roots.append(Path(CONFIG.DIR_1_RAW))
    except Exception:
        pass

    unique = []
    seen = set()
    for root in roots:
        try:
            resolved = str(root.resolve())
        except Exception:
            resolved = str(root)
        if resolved in seen:
            continue
        seen.add(resolved)
        if root.exists() and root.is_dir():
            unique.append(root)

    return unique


def is_recommended_images_dir(images_dir: Path) -> bool:
    for root in get_preferred_source_roots():
        if is_path_within(images_dir, root):
            return True
    return False


def derive_candidate_root_for_match(found_file: Path, xml_rel_name: str) -> Path | None:
    parts = [part for part in PurePosixPath(xml_rel_name).parts if part not in ("", ".")]
    if not parts:
        return None

    ascend_levels = len(parts) - 1
    parents = found_file.parents
    if ascend_levels >= len(parents):
        return None

    candidate_root = parents[ascend_levels]
    try:
        candidate_target = (candidate_root / xml_relpath_to_path(xml_rel_name)).resolve()
        if candidate_target != found_file.resolve():
            return None
    except Exception:
        return None

    return candidate_root


def find_matching_images_dir_for_xml(xml_image_names: list[str]) -> dict | None:
    if not xml_image_names:
        return None

    search_roots = get_preferred_source_roots()
    if not search_roots:
        return None

    sample_names = xml_image_names[: min(24, len(xml_image_names))]
    sample_by_basename = {}
    for rel_name in sample_names:
        sample_by_basename.setdefault(PurePosixPath(rel_name).name, []).append(rel_name)

    candidate_hits = {}

    for search_root in search_roots:
        try:
            for file_path in search_root.rglob("*"):
                if not file_path.is_file():
                    continue

                candidate_rel_names = sample_by_basename.get(file_path.name)
                if not candidate_rel_names:
                    continue

                for rel_name in candidate_rel_names:
                    candidate_root = derive_candidate_root_for_match(file_path, rel_name)
                    if candidate_root is None:
                        continue

                    try:
                        key = str(candidate_root.resolve())
                    except Exception:
                        key = str(candidate_root)

                    entry = candidate_hits.setdefault(
                        key,
                        {
                            "images_dir": candidate_root,
                            "sample_hits": set(),
                        },
                    )
                    entry["sample_hits"].add(rel_name)
        except Exception as e:
            logger.debug(f"Nie udało się przeskanować {search_root} podczas auto-odnajdywania zestawu obrazów: {e}")

    if not candidate_hits:
        return None

    ranked_candidates = sorted(
        candidate_hits.values(),
        key=lambda item: (len(item["sample_hits"]), -len(str(item["images_dir"]))),
        reverse=True,
    )

    best_match = None
    best_score = None

    for candidate in ranked_candidates[:8]:
        stats = evaluate_images_dir_for_xml(candidate["images_dir"], xml_image_names)
        stats["sample_hits"] = len(candidate["sample_hits"])
        score = (stats["matched"], -stats["missing_count"], stats["sample_hits"])

        if best_match is None or score > best_score:
            best_match = stats
            best_score = score

    return best_match


def refresh_source_binding_status(host, allow_autofind: bool = True) -> dict:
    host._source_binding_after_id = None

    def _finish(payload: dict) -> dict:
        host._extract_last_source_binding_result = dict(payload or {})
        try:
            host._refresh_extract_workflow_ui()
        except Exception:
            pass
        return payload

    xml_raw = (host.xml_path_var.get() or "").strip()
    images_raw = (host.images_dir_var.get() or "").strip()
    base_note = "Plik annotations.xml i katalog obrazów są nierozerwalnie powiązane."

    result = {
        "ok": False,
        "tone": "warning",
        "message": "",
        "source_xml": xml_raw,
        "source_images_dir": images_raw,
        "matched": 0,
        "total": 0,
        "missing_count": 0,
        "plate_count": 0,
        "images_with_plates": 0,
        "xml_images_total": 0,
    }

    if not xml_raw:
        result["message"] = (
            f"{base_note} Wskaż annotations.xml, a system spróbuje odnaleźć właściwy katalog w "
            "Workspace/1_raw_images/."
        )
        host._set_source_binding_status(result["message"], result["tone"])
        return _finish(result)

    xml_path = Path(xml_raw)
    if not xml_path.exists():
        result["tone"] = "error"
        result["message"] = (
            f"Nie znaleziono pliku annotations.xml. {base_note} Wskaż poprawny XML wygenerowany dla tego samego katalogu obrazów."
        )
        host._set_source_binding_status(result["message"], result["tone"])
        return _finish(result)

    try:
        xml_image_names = host._read_xml_image_names(xml_path)
    except Exception as e:
        result["tone"] = "error"
        result["message"] = f"Nie mogę odczytać annotations.xml: {e}"
        host._set_source_binding_status(result["message"], result["tone"])
        return _finish(result)

    if not xml_image_names:
        result["tone"] = "error"
        result["message"] = (
            "Ten plik XML nie zawiera listy obrazów, więc nie da się powiązać go z katalogiem źródłowym."
        )
        host._set_source_binding_status(result["message"], result["tone"])
        return _finish(result)

    try:
        result.update(host._count_xml_plate_cut_targets(xml_path))
    except Exception as e:
        result["tone"] = "error"
        result["message"] = f"Nie mogę policzyć tablic do wyodrębnienia w annotations.xml: {e}"
        host._set_source_binding_status(result["message"], result["tone"])
        return _finish(result)

    if int(result.get("plate_count") or 0) <= 0:
        result["tone"] = "error"
        result["message"] = host._append_extract_cut_plan(
            "XML został odczytany, ale PZ1 nie znalazł w nim poprawnych polygonów tablic. "
            "Wycinanie utworzyłoby pusty zestaw tablic, więc najpierw popraw lub wybierz właściwy annotations.xml.",
            result,
        )
        host._set_source_binding_status(result["message"], result["tone"])
        return _finish(result)

    current_images_dir = Path(images_raw) if images_raw else None
    current_stats = None

    if current_images_dir and current_images_dir.exists() and current_images_dir.is_dir():
        current_stats = host._evaluate_images_dir_for_xml(current_images_dir, xml_image_names)
    elif current_images_dir:
        result["tone"] = "error"
        result["message"] = (
            f"Nie znaleziono wskazanego katalogu obrazów: {current_images_dir}. {base_note}"
        )

    best_candidate = None
    need_autofind = allow_autofind and (
        current_stats is None or current_stats["matched"] < current_stats["total"]
    )

    if need_autofind:
        best_candidate = host._find_matching_images_dir_for_xml(xml_image_names)
        if best_candidate and best_candidate["matched"] == best_candidate["total"]:
            candidate_dir = Path(best_candidate["images_dir"])
            same_as_current = False
            if current_images_dir:
                try:
                    same_as_current = candidate_dir.resolve() == current_images_dir.resolve()
                except Exception:
                    same_as_current = candidate_dir == current_images_dir

            if not same_as_current:
                host._source_binding_sync_in_progress = True
                try:
                    host.images_dir_var.set(str(candidate_dir))
                finally:
                    host._source_binding_sync_in_progress = False
                try:
                    host._force_save_all()
                except Exception:
                    pass

            current_images_dir = candidate_dir
            result["source_images_dir"] = str(candidate_dir)
            current_stats = best_candidate
            result["auto_found"] = True

    if current_stats and current_stats["matched"] == current_stats["total"]:
        recommended = host._is_recommended_images_dir(Path(current_stats["images_dir"]))
        result.update(
            {
                "ok": True,
                "tone": "success",
                "matched": current_stats["matched"],
                "total": current_stats["total"],
                "missing_count": 0,
            }
        )

        if result.get("auto_found"):
            result["message"] = (
                f"Powiązanie potwierdzone. Auto-odnaleziono katalog obrazów: "
                f"{current_stats['matched']}/{current_stats['total']} plików z XML w "
                f"{current_stats['images_dir']}."
            )
        elif recommended:
            result["message"] = (
                f"Powiązanie potwierdzone: {current_stats['matched']}/{current_stats['total']} plików z XML "
                f"znaleziono w tym zestawie obrazów."
            )
        else:
            result["message"] = (
                f"Powiązanie XML-katalog jest poprawne ({current_stats['matched']}/{current_stats['total']}), "
                f"ale źródła są poza Workspace/1_raw_images/. To działa w trybie swobodnym, lecz nie jest zalecane."
            )

        result["message"] = host._append_extract_cut_plan(result["message"], result)
        host._set_source_binding_status(result["message"], result["tone"])
        return _finish(result)

    if current_stats:
        missing_hint = host._summarize_missing_xml_images(current_stats["missing"])
        result.update(
            {
                "matched": current_stats["matched"],
                "total": current_stats["total"],
                "missing_count": current_stats["missing_count"],
                "tone": "error",
            }
        )
        result["message"] = (
            f"Wybrany katalog obrazów nie pasuje do tego XML: znaleziono "
            f"{current_stats['matched']}/{current_stats['total']} wymaganych plików. {missing_hint}"
        ).strip()

        if best_candidate and best_candidate["matched"] > current_stats["matched"]:
            result["message"] += (
                f" Najlepszy kandydat w Workspace/1_raw_images/ daje "
                f"{best_candidate['matched']}/{best_candidate['total']} dopasowań."
            )

        result["message"] = host._append_extract_cut_plan(result["message"], result)
        host._set_source_binding_status(result["message"], result["tone"])
        return _finish(result)

    if best_candidate and best_candidate["matched"] > 0:
        result.update(
            {
                "matched": best_candidate["matched"],
                "total": best_candidate["total"],
                "missing_count": best_candidate["missing_count"],
                "tone": "warning",
                "message": (
                    f"Nie udało się jednoznacznie potwierdzić katalogu obrazów. Najlepszy kandydat w "
                    f"Workspace/1_raw_images/ zawiera {best_candidate['matched']}/{best_candidate['total']} plików z XML."
                ),
            }
        )
        result["message"] = host._append_extract_cut_plan(result["message"], result)
    elif not images_raw:
        result["message"] = host._append_extract_cut_plan(
            (
                f"{base_note} Wskaż katalog obrazów, na których wykonano anotacje. "
                "Jeśli leży w Workspace/1_raw_images/, system odnajdzie ją automatycznie."
            ),
            result,
        )

    result["message"] = host._append_extract_cut_plan(result["message"], result)
    host._set_source_binding_status(result["message"], result["tone"])
    return _finish(result)


def prepare_plate_cut_detections_for_source(host, image_name: str, plates: list) -> list:
    ordered_pairs = sorted(
        list(enumerate(list(plates or []))),
        key=host._plate_cut_reading_order_key,
    )
    ordered_plates = [detection for _, detection in ordered_pairs]
    expected_tokens = host._extract_source_plate_tokens_from_filename(image_name)
    has_direct_mapping = bool(len(expected_tokens) == 1 and len(ordered_plates) == 1)
    expected_tokens_json = json.dumps(expected_tokens, ensure_ascii=False) if expected_tokens else "[]"

    for sorted_index, detection in enumerate(ordered_plates):
        attributes = dict(getattr(detection, "attributes", {}) or {})
        attributes["source_plate_index"] = str(sorted_index)
        attributes["source_plate_count"] = str(len(ordered_plates))
        attributes["source_expected_texts"] = expected_tokens_json
        if has_direct_mapping:
            attributes["source_expected_text"] = expected_tokens[sorted_index]
            attributes["source_expected_text_source"] = "filename_order"
        elif expected_tokens:
            attributes.pop("source_expected_text", None)
            attributes["source_expected_text_source"] = "ambiguous_filename_tokens"
        detection.attributes = attributes

    return ordered_plates


def backfill_preview_expected_texts_from_sources(host, metadata_map: dict) -> bool:
    if not isinstance(metadata_map, dict):
        return False

    groups: dict[str, list[tuple[str, dict]]] = {}
    changed = False
    for plate_id, data in metadata_map.items():
        if not isinstance(data, dict):
            continue
        source_image = str(
            data.get("source_image")
            or data.get("source_name")
            or data.get("filename")
            or ""
        ).strip()
        if not source_image:
            continue
        groups.setdefault(source_image, []).append((str(plate_id), data))

    for source_image, items in groups.items():
        expected_tokens = host._extract_source_plate_tokens_from_filename(source_image)
        if not expected_tokens:
            continue
        expected_tokens_json = json.dumps(expected_tokens, ensure_ascii=False)

        ordered_items = sorted(
            list(items),
            key=lambda item: host._plate_cut_reading_order_key(
                (
                    int(str(item[0]).rsplit("_", 1)[-1]) if str(item[0]).rsplit("_", 1)[-1].isdigit() else 0,
                    SimpleNamespace(bbox=tuple(item[1].get("source_bbox", (0, 0, 0, 0)) or (0, 0, 0, 0))),
                )
            ),
        )
        direct_mapping = bool(len(expected_tokens) == 1 and len(ordered_items) == 1)
        for sorted_index, (_plate_id, data) in enumerate(ordered_items):
            if data.get("source_plate_index") != sorted_index:
                data["source_plate_index"] = sorted_index
                changed = True
            if data.get("source_plate_count") != len(ordered_items):
                data["source_plate_count"] = len(ordered_items)
                changed = True
            if data.get("source_expected_texts") != expected_tokens:
                data["source_expected_texts"] = list(expected_tokens)
                changed = True

            if direct_mapping and not str(data.get("source_expected_text") or "").strip():
                data["source_expected_text"] = expected_tokens[sorted_index]
                data["source_expected_text_source"] = "filename_order_backfill"
                changed = True
            elif not direct_mapping:
                if str(data.get("source_expected_text_source") or "").strip() != "ambiguous_filename_tokens":
                    data["source_expected_text_source"] = "ambiguous_filename_tokens"
                    changed = True
                if str(data.get("source_expected_text") or "").strip():
                    data["source_expected_text"] = None
                    changed = True

            attrs = data.get("plate_attributes")
            if isinstance(attrs, dict):
                if attrs.get("source_expected_texts") != expected_tokens_json:
                    attrs["source_expected_texts"] = expected_tokens_json
                    changed = True
                if direct_mapping and not str(attrs.get("source_expected_text") or "").strip():
                    attrs["source_expected_text"] = expected_tokens[sorted_index]
                    attrs["source_expected_text_source"] = "filename_order_backfill"
                    changed = True
                elif not direct_mapping:
                    if str(attrs.get("source_expected_text_source") or "").strip() != "ambiguous_filename_tokens":
                        attrs["source_expected_text_source"] = "ambiguous_filename_tokens"
                        changed = True
                    if str(attrs.get("source_expected_text") or "").strip():
                        attrs.pop("source_expected_text", None)
                        changed = True

    return changed
