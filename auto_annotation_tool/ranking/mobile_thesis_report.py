# Z4MOBREV009_THESIS_EXPORT_20261017
"""Auditable, read-only thesis package from one completed/partial mobile review.

Does not evaluate models anew: the MobileReviewSession statistics() and
Z4MOBREV008 _pose_geometry_rows are the only numerical sources. GUI passes the
rows already exported as mt_pose_geometry.csv. No archive or sidecar writes.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from PIL import Image, ImageDraw, ImageFont

SCHEMA = "z4_mobile_thesis_export_v1"
METRIC_FIELDS = (
    ("subject_count", "Obserwacje", "szt.", "subject"),
    ("reviewed_subjects", "Obserwacje ocenione", "szt.", "subject"),
    ("not_reviewed_subjects", "Obserwacje nieocenione", "szt.", "subject"),
    ("evaluable_subjects", "Tablice kwalifikujące się do oceny", "szt.", "subject"),
    ("subjects_with_exact_read", "Tablice z co najmniej jednym poprawnym odczytem", "szt.", "subject"),
    ("evaluable_reads", "Ocenialne odczyty MZ", "szt.", "sample"),
    ("exact_reads", "Odczyty Exact Match", "szt.", "sample"),
    ("incorrect_reads", "Błędne odczyty MZ", "szt.", "sample"),
    ("no_reads", "Braki odczytu MZ", "szt.", "sample"),
    ("gt_characters", "Znaki GT uwzględnione w CER", "znaki", "character"),
    ("mt_invocation_count", "Wywołania MT", "szt.", "mt_invocation"),
    ("evaluable_mt_invocations", "Ocenialne wywołania MT", "szt.", "mt_invocation"),
    ("mt_successful_invocations", "Poprawna lokalizacja MT", "szt.", "mt_invocation"),
    ("mt_false_detections", "Fałszywe detekcje MT", "szt.", "attempt"),
    ("crop_count", "Cropy MZ", "szt.", "sample"),
    ("median_time_to_first_exact_ms", "Mediana czasu do pierwszego poprawnego odczytu", "ms", "subject"),
    ("p90_time_to_first_exact_ms", "P90 czasu do pierwszego poprawnego odczytu", "ms", "subject"),
)
RATE_FIELDS = (
    ("subject_success_rate", "Poprawny odczyt tablicy (co najmniej raz)", "evaluable_subjects", "subjects_with_exact_read", "subject"),
    ("exact_read_rate", "Exact Match MZ", "evaluable_reads", "exact_reads", "sample"),
    ("no_read_rate", "Brak odczytu MZ", "evaluable_reads", "no_reads", "sample"),
    ("mt_localization_success_rate", "Skuteczność lokalizacji MT", "evaluable_mt_invocations", "mt_successful_invocations", "mt_invocation"),
    ("cer", "Micro-CER (znaki MZ)", "gt_characters", None, "character"),
)


def _safe_attr(obj: Any, name: str, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _safe_object(obj: Any):
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _safe_object(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_safe_object(v) for v in obj]
    return str(obj)


def _numeric(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _nonnegative_int(value):
    num = _numeric(value)
    return int(num) if num is not None and num >= 0 and num.is_integer() else None


def _sha_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(text: str):
    return re.sub(r"[^a-zA-Z0-9._-]", "-", str(text))[:42].strip(".-") or "session"


def _csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _tex_escape(value):
    substitutions = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                     "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(substitutions.get(char, char) for char in str(value))


def _pl_value(number, *, percent=False):
    if number is None:
        return "brak danych"
    if percent:
        return f"{number * 100:.2f}".replace(".", ",") + "\\%"
    return (f"{number:.2f}" if not float(number).is_integer() else f"{int(number)}").replace(".", ",")


def _font(size, bold=False):
    candidates = (
        (r"C:\Windows\Fonts\arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold else
        (r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )
    for item in candidates:
        try:
            return ImageFont.truetype(item, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _bars_png(path: Path, title: str, subtitle: str, labels: list[str], values: list[float], *,
              value_unit: str = "liczba przypadków", max_value: float | None = None, percent=False):
    """Render deterministic raster bar chart using Pillow; no GUI/display backend."""
    if not labels or len(labels) != len(values):
        raise ValueError("Brak spójnych danych do wykresu")
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Wykres wymaga skończonych, nieujemnych wartości")
    w, h = 1680, 920
    im = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(im)
    font_title, font_sub, font_ticks, font_labels = _font(45, True), _font(25), _font(23), _font(24)
    d.text((100, 64), title, font=font_title, fill="#15263e")
    d.text((100, 130), subtitle, font=font_sub, fill="#52647a")
    left, right, bottom, top = 152, 1555, 745, 220
    peak = max(values) if values else 0.0
    ylim = max_value if max_value is not None else max(1.0, peak * 1.18)
    if ylim <= 0:
        ylim = 1.0
    for i in range(6):
        y = round(bottom - (bottom - top) * i / 5)
        v = ylim * i / 5
        d.line((left, y, right, y), fill="#dce4ec", width=2)
        label = f"{v:.0f}%" if percent else (str(round(v, 1)).replace(".0", ""))
        d.text((35, y - 13), label, font=font_ticks, fill="#46566c")
    bar_space = (right - left) / len(labels)
    bar_width = min(145, max(38, bar_space * 0.54))
    for i, (label, value) in enumerate(zip(labels, values)):
        center = left + (i + 0.5) * bar_space
        peak_y = round(bottom - (bottom - top) * min(value, ylim) / ylim)
        x0, x1 = round(center - bar_width / 2), round(center + bar_width / 2)
        if value:
            d.rounded_rectangle((x0, peak_y, x1, bottom), radius=9, fill="#2876a5")
        legend = f"{value:.2f}%" if percent else f"{value:g}"
        value_box = d.textbbox((0, 0), legend, font=font_labels)
        d.text((center - (value_box[2] - value_box[0]) / 2, max(top, peak_y - 42)), legend, font=font_labels, fill="#173f5f")
        # Two lines, if a longer label contains an explicit delimiter.
        parts = label.split("|")
        for k, part in enumerate(parts[:2]):
            bbox = d.textbbox((0, 0), part, font=font_labels)
            d.text((center - (bbox[2] - bbox[0]) / 2, bottom + 26 + 33 * k), part, font=font_labels, fill="#263c50")
    d.line((left, bottom, right, bottom), fill="#607588", width=3)
    d.text((left, 854), f"Oś Y: {value_unit}.  Dla braku GT/oceny nie tworzono wartości zerowych.", font=font_sub, fill="#52647a")
    im.save(path, format="PNG", optimize=True)


def _histogram(values, edges):
    """Bins are [lower, upper) with the last bin including the right endpoint."""
    bins = [0] * (len(edges) - 1)
    for value in values:
        for i in range(len(bins)):
            if edges[i] <= value < edges[i + 1] or (i == len(bins) - 1 and value == edges[-1]):
                bins[i] += 1
                break
        else:
            raise ValueError(f"Wartość poza zakresem histogramu: {value}")
    return bins


def _metric_rows(summary, pose_rows):
    """Never turn unknown, invalid, or denominator=0 into an observed 0%."""
    result, warnings = [], []
    for key, label, unit, level in METRIC_FIELDS:
        value = _numeric(summary.get(key))
        if value is not None and value < 0:
            value = None
        result.append(dict(metric=key, label=label, level=level, value=value, unit=unit,
                           numerator="", denominator="", status="measured" if value is not None else "unavailable",
                           source="MobileReviewSession.statistics().summary"))
    for key, label, denom_key, numer_key, level in RATE_FIELDS:
        rate = _numeric(summary.get(key))
        den = _nonnegative_int(summary.get(denom_key))
        num = _nonnegative_int(summary.get(numer_key)) if numer_key else None
        if den is None or den == 0 or rate is None or rate < 0 or (key != "cer" and rate > 1):
            rate = None
        if rate is not None and num is not None and num > den:
            warnings.append(f"{key}: numerator > denominator; wynik pominięty")
            rate = None
        if rate is not None and num is not None and abs(rate - num / den) > 0.0001:
            warnings.append(f"{key}: wartość niezgodna z licznikami; wynik pominięty")
            rate = None
        result.append(dict(metric=key, label=label, level=level, value=rate, unit="fraction",
                           numerator=num if num is not None else "", denominator=den if den is not None else "",
                           status="measured" if rate is not None else "unavailable",
                           source="MobileReviewSession.statistics().summary"))
    measured = [row for row in pose_rows if row.get("metric_status") == "measured"
                and _numeric(row.get("nmce_pct")) is not None and _numeric(row.get("polygon_iou_pct")) is not None
                and 0 <= _numeric(row.get("polygon_iou_pct")) <= 100 and _numeric(row.get("nmce_pct")) >= 0]
    eligible = sum(row.get("eligible") is True for row in pose_rows)
    def add(name, label, value, unit, denominator="", population="attempt"):
        result.append(dict(metric=name, label=label, level=population, value=value, unit=unit, numerator="",
                           denominator=denominator, status="measured" if value is not None else "unavailable",
                           source="Z4MOBREV008 _pose_geometry_rows(session)"))
    add("pose_eligible", "Prawdziwe tablice Pose kwalifikujące się do pomiaru", eligible, "szt.")
    add("pose_measured", "Pomiary Pose z ręcznym GT", len(measured), "szt.")
    add("pose_pending", "Próby Pose z nieuzupełnionym GT", max(0, eligible - len(measured)), "szt.")
    add("pose_nmce_mean_pct", "Średni NMCE Pose", (sum(float(r["nmce_pct"]) for r in measured) / len(measured)) if measured else None,
        "percent", len(measured) if measured else "")
    add("pose_iou_mean_pct", "Średnie Polygon IoU Pose", (sum(float(r["polygon_iou_pct"]) for r in measured) / len(measured)) if measured else None,
        "percent", len(measured) if measured else "")
    return result, warnings, measured


def _relative_sources(export_files, output_root):
    entries = []
    for file in export_files:
        path = Path(file).resolve()
        if not path.is_file() or not path.is_relative_to(output_root.resolve()):
            raise ValueError(f"Eksport źródłowy nie istnieje lub leży poza katalogiem docelowym: {path}")
        entries.append({"file": path.relative_to(output_root.resolve()).as_posix(),
                        "sha256": _sha_file(path), "bytes": path.stat().st_size})
    return entries


def _write_report_md(out, manifest, metrics, figure_specs, warnings, pose_rows):
    meta = manifest["session"]
    lines = [
        "# Raport weryfikacji mobilnego ALPR — dane do pracy dyplomowej", "",
        f"Sesja: `{meta['session_id']}`  ",
        f"Status oceny: **{manifest['review']['quality_status']}**  ",
        f"Seria / wariant zadeklarowany przez Androida / scenariusz: `{meta['series_id'] or 'brak'}` / `{meta['variant'] or 'brak'}` / `{meta['scenario_id'] or 'brak'}`  ",
        f"Wariant F4 (rejestr SHA): `{meta.get('f4_variant',{}).get('label','NIEPOTWIERDZONE')}`; provenance=`{meta.get('f4_variant',{}).get('provenance') or 'NIEPOTWIERDZONE'}`  ",
        f"Źródłowe archiwum SHA-256: `{meta['source_archive_sha256']}`  ",
        f"Rewizja oceny operatora: `{manifest['review']['review_revision']}`  ", "",
        "## Wyniki i mianowniki", "",
        "| Wskaźnik | Wartość | Poziom | Mianownik | Status |", "|---|---:|---|---:|---|",
    ]
    for m in metrics:
        value = m["value"]
        if value is None:
            shown = "brak danych"
        elif m["unit"] == "fraction":
            shown = f"{value * 100:.2f}%"
        elif m["unit"] == "percent":
            shown = f"{value:.2f}%"
        else:
            shown = f"{value:g}" + (f" {m['unit']}" if m['unit'] != "szt." else "")
        lines.append(f"| {m['label']} | {shown} | {m['level']} | {m['denominator'] or '—'} | {m['status']} |")
    lines += ["", "## Geometria i brakujące dane", "",
              "Wyniki NMCE / Polygon IoU obejmują wyłącznie próby z prawdziwą tablicą, poprawnym czteropunktowym polygonem MT, "
              "zgodnym obrazem wejściowym MT oraz ręcznie narysowanymi i zweryfikowanymi czterema narożnikami GT.",
              "Brak GT, błędne powiązanie dowodu i geometria nieprawidłowa pozostają osobnymi statusami w `mt_pose_geometry.csv`. "
              "Nie są traktowane jako NMCE=0 ani IoU=0.", "",
              f"Statusy geometrii wszystkich prób: `{json.dumps(manifest['pose_status_counts'], ensure_ascii=False)}`.", "",
              "## Wykresy", ""]
    if figure_specs:
        for spec in figure_specs:
            lines.append(f"- [{spec['caption']}]({spec['file']}) — n={spec['sample_size']}" if isinstance(spec['sample_size'], int) else f"- [{spec['caption']}]({spec['file']}) — odrębne mianowniki (zob. podpis osi).")
    else:
        lines.append("Nie wygenerowano wykresów bez odpowiedniego mianownika / ręcznego GT.")
    lines += ["", "## Metodyka i ograniczenia", "",
              "- Jeden eksport dotyczy jednej sesji i jednego zestawu modeli. Nie agreguje wariantów ani scenariuszy o różnym pochodzeniu.",
              "- GT znaków jest weryfikowany w trybie asystowanym, jeżeli predykcja pozostaje widoczna; nie opisuj go jako oceny zaślepionej.",
              "- GT narożników jest manualne; pomiary NMCE i IoU nie zastępują niezależnego mAP detektora ani pomiaru Recall na pełnej bazie.",
              "- Exact Match i CER dotyczą próbek / znaków z GT; są czym innym niż liczba wywołań MT i liczba cropów.",
              "- Obrazy TV nagrywane kamerą telefonu (camera-in-the-loop) nie są pomiarami w ruchu drogowym ani reprezentatywnym zbiorem nocnym.",
              "- Rewizja weryfikacji i SHA archiwum pozwalają odróżnić wyniki od kolejnych eksportów po zmianach GT.",
              "- Tożsamość modeli i scenariusza należy weryfikować w `thesis_manifest.json`; nie porównywać par FP32/INT8 bez wspólnych checkpointów.",
              "- Każde zestawienie z pojedynczej, małej sesji oznaczać jako pilotażowe; pełne wnioskowanie wymaga zdefiniowanego protokołu.",
              "", "## Dane odtwarzalne", "",
              "W podfolderze `sources/` znajdują się kopie plików bieżącego eksportu z sumami kontrolnymi SHA-256. "
              "W `thesis_manifest.json` zapisano także ich ścieżki zewnętrzne oraz oryginalny SHA źródła. "
              "Kolejny eksport do folderu nadrzędnego nie zmieni kopii przypisanych do tej rewizji. "
              "Ten pakiet nie modyfikuje żadnego `.alprsession`, GT ani sidecara.", ""]
    if warnings:
        lines += ["## Kontrole spójności / ostrzeżenia", ""] + ["- " + x for x in warnings] + [""]
    (out / "thesis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tex(out, directory_name, metrics, figures, manifest):
    sha8 = manifest["session"]["source_archive_sha256"][:8]
    table_metrics = [m for m in metrics if m["metric"] in (
        "evaluable_subjects", "subject_success_rate", "evaluable_reads", "exact_read_rate", "cer",
        "evaluable_mt_invocations", "mt_localization_success_rate", "pose_measured", "pose_nmce_mean_pct", "pose_iou_mean_pct",
    )]
    tex = ["% Z4MOBREV009: fragments to include manually in unchanged EE-dyplom template.",
           "% Place this folder under the LaTeX project root or adjust paths below.",
           r"\begin{table}[htbp]", r"\centering", r"\small", r"\begin{tabular}{lrr}",
           r"\hline", "Wskaźnik & Wartość & Mianownik " + r"\\", r"\hline"]
    for metric in table_metrics:
        val = metric["value"]
        shown = _pl_value(val, percent=metric["unit"] == "fraction")
        if val is not None and metric["unit"] == "percent":
            shown += r"\%"
        tex.append(f"{_tex_escape(metric['label'])} & {shown} & {_tex_escape(metric['denominator'] or '—')} " + r"\\")
    tex += [r"\hline", r"\end{tabular}",
            r"\caption{Wyniki weryfikacji mobilnego ALPR dla jednej sesji; brak danych nie oznacza wyniku zerowego.}",
            rf"\label{{tab:alpr_mobile_review_{sha8}}}", r"\end{table}", ""]
    (out / "thesis_tables.tex").write_text("\n".join(tex), encoding="utf-8")
    latex_figures = ["% Z4MOBREV009: requires graphicx (usually present in thesis template).",
                     "% Source .png files should be copied with this entire thesis_report folder."]
    for item in figures:
        latex_figures.extend([r"\begin{figure}[htbp]", r"\centering",
                              rf"\includegraphics[width=0.92\linewidth]{{{directory_name}/{item['file']}}}",
                              rf"\caption{{{_tex_escape(item['caption'])}" + (f" (n={item['sample_size']})" if isinstance(item['sample_size'], int) else " (odrębne mianowniki)") + ".}",
                              rf"\label{{fig:alpr_mobile_{item['code']}_{sha8}}}", r"\end{figure}", ""])
    if not figures:
        latex_figures.append("% No plots: no evaluated denominator; no synthetic values were generated.")
    (out / "thesis_figures.tex").write_text("\n".join(latex_figures) + "\n", encoding="utf-8")


def export_thesis_bundle(session, statistics, pose_rows, export_files, output_root: Path):
    """Write one immutable-versioned research subfolder next to existing exports.

    Precondition: caller has performed session.export() and mt_pose_geometry.csv.
    Never mutates session, its review, models, archive or original export files.
    Returns paths created in the new subfolder only.
    """
    output_root = Path(output_root).resolve()
    if not output_root.is_dir():
        raise ValueError("Najpierw wykonaj standardowy eksport do istniejącego katalogu")
    summary = (statistics or {}).get("summary", {})
    if not isinstance(summary, dict):
        raise TypeError("statistics.summary must be a dict")
    if not isinstance(pose_rows, list):
        raise TypeError("pose_rows must be a list")
    source_sha = str(_safe_attr(session.review, "source_archive_sha256", "") or "")
    if not re.fullmatch(r"[a-fA-F0-9]{64}", source_sha):
        raise ValueError("Brak kompletnego SHA-256 archiwum źródłowego; eksport naukowy zablokowany")
    if any(str(row.get("source_archive_sha256")) != source_sha for row in pose_rows):
        raise ValueError("Mieszane źródła SHA w pose_rows; eksport zablokowany")
    source_path = Path(_safe_attr(session, "path", "")) if _safe_attr(session, "path") else None
    source_verified = False
    if source_path and source_path.is_file():
        if _sha_file(source_path) != source_sha.lower():
            raise ValueError("SHA-256 archiwum na dysku różni się od weryfikowanego źródła")
        source_verified = True
    sources = _relative_sources(export_files, output_root)
    if not any(x["file"].endswith("mt_pose_geometry.csv") for x in sources):
        raise ValueError("Nie znaleziono wyeksportowanego mt_pose_geometry.csv")
    revision = _nonnegative_int(_safe_attr(session.review, "review_revision"))
    dt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder_name = f"thesis_{_slug(session.session_id)}_{source_sha[:10]}_r{revision if revision is not None else 'unknown'}_{dt}"
    final = output_root / folder_name
    if final.exists():
        raise FileExistsError(f"Folder raportu już istnieje: {final}")
    metrics, consistency_warnings, measured_pose = _metric_rows(summary, pose_rows)
    review_status = str(_safe_attr(session.review, "review_status", "UNKNOWN"))
    missing = _nonnegative_int(summary.get("not_reviewed_subjects"))
    pose_pending = max(0, sum(row.get("eligible") is True for row in pose_rows) - len(measured_pose))
    complete = review_status == "COMPLETED" and missing == 0 and pose_pending == 0
    quality_status = "REVIEW_COMPLETED" if complete else "PRELIMINARY_INCOMPLETE_REVIEW"
    exp = _safe_attr(session.bundle, "experiment_session", {}) or {}
    metadata = _safe_attr(session.bundle, "metadata", {}) or {}
    identity = {}
    for key in ("series_id", "variant", "scenario_id", "replicate_index"):
        identity[key] = _safe_attr(exp, key) or _safe_attr(metadata, key) or ""
    from .mobile_f4_variant import for_session
    identity['f4_variant']=for_session(session)
    status_counts = {}
    for row in pose_rows:
        status = str(row.get("metric_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    warnings = list(consistency_warnings)
    if not source_verified:
        warnings.append("Archiwum źródłowe nie było dostępne do ponownego wyliczenia SHA podczas eksportu; identyfikator pochodzi z weryfikowanej wcześniej sesji")
    if not complete:
        warnings.append("Weryfikacja nie jest zakończona lub nie wszystkie GT są uzupełnione; wyniki wstępne")
    if _nonnegative_int(summary.get("mt_invocation_count")) is not None and _nonnegative_int(summary.get("mt_invocation_count")) < 30:
        warnings.append("Mała liczba wywołań MT (pilotaż); wyniki nie są estymatą skuteczności populacyjnej")
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "session": {"session_id": str(session.session_id), "source_archive_sha256": source_sha.lower(),
                    "source_archive_path": str(source_path) if source_path else None,
                    "source_verified_at_export": source_verified, **_safe_object(identity)},
        "review": {"review_revision": revision, "review_status": review_status,
                   "review_mode": str(_safe_attr(session.review, "review_mode", "unknown")),
                   "review_mode_origin": _safe_attr(session.review, "review_mode_origin"),
                   "quality_status": quality_status},
        "experiment": _safe_object(exp),
        "model_provenance": _safe_object(_safe_attr(session.review, "provenance", {})),
        "inputs": sources,
        "pose_status_counts": status_counts,
        "metric_count": len(metrics),
        "method": {
            "pose_gt": "manual_gt_quad_on_mt_input", "pose_metrics": "Z4MOBREV008 NMCE/convex polygon IoU",
            "mz_gt": "reviewed_in_GUI; assisted if prediction shown", "aggregation": "single session; no cross-variant merging",
            "missing_gt": "unavailable_not_zero", "levels": ["subject", "mt_invocation", "attempt", "sample", "character"],
        },
        "warnings": warnings,
    }
    staging = Path(tempfile.mkdtemp(dir=output_root, prefix=".thesis_pending_"))
    try:
        # Each version retains its own immutable input snapshot; the standard
        # export in the parent directory may be overwritten next time.
        for item in sources:
            relative = Path(item["file"])
            snapshot = staging / "sources" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output_root / relative, snapshot)
            if _sha_file(snapshot) != item["sha256"]:
                raise ValueError(f"Źródłowy CSV zmienił się podczas kopiowania: {item['file']}")
            item["snapshot"] = (Path("sources") / relative).as_posix()
        _csv(staging / "thesis_metrics.csv", metrics,
             ("metric", "label", "level", "value", "unit", "numerator", "denominator", "status", "source"))
        figures = []
        if measured_pose:
            iou = [float(row["polygon_iou_pct"]) for row in measured_pose]
            nmce = [float(row["nmce_pct"]) for row in measured_pose]
            iou_edges = [0, 20, 40, 60, 80, 100]
            iou_freq = _histogram(iou, iou_edges)
            _bars_png(staging / "fig_pose_iou.png", "Dokładność geometrii tablic — Polygon IoU",
                      f"Ocenione tablice z czterema narożnikami MT i GT, n = {len(iou)}",
                      ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"], iou_freq)
            figures.append(dict(code="pose_iou", file="fig_pose_iou.png", sample_size=len(iou),
                                caption="Rozkład Polygon IoU pomiędzy punktami MT Pose i ręcznym GT"))
            nmce_freq = _histogram(nmce, [0, 2, 5, 10, 20, float("inf")])
            _bars_png(staging / "fig_pose_nmce.png", "Dokładność narożników — NMCE",
                      f"Błąd znormalizowany przekątną obwiedni GT, n = {len(nmce)}",
                      ["0–2%", "2–5%", "5–10%", "10–20%", "≥20%"], nmce_freq)
            figures.append(dict(code="pose_nmce", file="fig_pose_nmce.png", sample_size=len(nmce),
                                caption="Rozkład znormalizowanego błędu narożników NMCE"))
        summary_metrics = {m["metric"]: m for m in metrics}
        quality_keys = [("subject_success_rate", "ALPR|tablica"), ("exact_read_rate", "Exact Match|MZ"),
                        ("mt_localization_success_rate", "Lokalizacja|MT")]
        quality = [(k, label) for k, label in quality_keys if summary_metrics[k]["value"] is not None]
        if quality:
            bars = [summary_metrics[k]["value"] * 100 for k, label in quality]
            ns = [int(summary_metrics[k]["denominator"]) for k, label in quality]
            _bars_png(staging / "fig_quality_rates.png", "Wyniki jakościowe — odrębne populacje",
                      "Mianowniki: " + ", ".join(f"{key}: n={n}" for (key, _), n in zip(quality, ns)),
                      [label for _, label in quality], bars, value_unit="udział [%]", max_value=100, percent=True)
            figures.append(dict(code="quality_rates", file="fig_quality_rates.png", sample_size="różne mianowniki",
                                caption="Udziały poprawnych wyników MT, MZ i całego ALPR — z odrębnymi mianownikami"))
        successful_subjects = _nonnegative_int(summary.get("subjects_with_exact_read"))
        time_keys = ("median_time_to_first_exact_ms", "p90_time_to_first_exact_ms")
        times = [_numeric(summary.get(key)) for key in time_keys]
        if successful_subjects and all(value is not None and value >= 0 for value in times):
            _bars_png(staging / "fig_time_to_first_exact.png", "Czas do pierwszego poprawnego odczytu",
                      f"Tylko tablice z poprawnym odczytem, n = {successful_subjects}",
                      ["Mediana", "P90"], times, value_unit="ms", max_value=None)
            figures.append(dict(code="time_to_first_exact", file="fig_time_to_first_exact.png", sample_size=successful_subjects,
                                caption="Czas do pierwszego poprawnego odczytu tablicy (mediana i P90), wyłącznie dla tablic z Exact Match"))
        manifest["figures"] = figures
        _write_report_md(staging, manifest, metrics, figures, warnings, pose_rows)
        _write_tex(staging, folder_name, metrics, figures, manifest)
        (staging / "README.txt").write_text(
            "Z4MOBREV009 — raport naukowy do pracy dyplomowej\n"
            "1) Pakiet zawiera kopie CSV w sources/; SHA w thesis_manifest.json. Folder nadrzędny może być nadpisany.\n"
            "2) Skopiuj cały folder raportu do katalogu głównego projektu LaTeX, by ścieżki do PNG działały.\n"
            "3) Fragmenty thesis_tables.tex i thesis_figures.tex wstaw ręcznie, zgodnie z niezmienionym szablonem PW.\n"
            "4) Brak danych oznacza brak pomiaru, nie zero. Nie scalać scenariuszy bez formalnego protokołu.\n",
            encoding="utf-8")
        manifest["outputs"] = [dict(file=p.relative_to(staging).as_posix(), sha256=_sha_file(p), bytes=p.stat().st_size)
                               for p in sorted(staging.rglob("*")) if p.is_file()]
        (staging / "thesis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return sorted(final.iterdir())
