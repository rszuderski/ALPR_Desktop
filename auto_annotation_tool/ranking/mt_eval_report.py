"""Immutable native Z4 exports of verified MT runs; no inference or rescoring.

The runner owns validation and metric calculation. This module only projects its
verified records into tables, SVG figures and a human-readable report. A partial
measurement remains partial even when its report has been exported successfully.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
from pathlib import Path
import uuid


DOMAINS = ("ALL", "DAY", "NIGHT")
MODEL_DESCRIPTIONS = (
    ("MT-n", "Bazowy model lokalizacji tablic YOLO Pose w wariancie nano."),
    ("MT-DAY", "Model MT-n dostrojony z wykorzystaniem materiału i augmentacji dziennych."),
    ("MT-NIGHT", "Model MT-n dostrojony z wykorzystaniem materiału i augmentacji nocnych."),
)
COUNT_FIELDS = (
    "processed_scenes", "cohort_scenes", "pending_scenes", "cohort_box_gt", "cohort_pose_gt", "no_detection_scenes",
    "box_gt", "box_tp", "box_fp", "box_fn", "pose_gt", "pose_matched_gt", "pose_unmatched_gt",
    "pose_valid_matched", "pose_missing_keypoints", "pose_invalid_quad", "pose_bbox_fallback",
    "pose_reversed_winding", "pose_excluded_gt", "pose_excluded_matched",
    "unindexed_geometry_gt", "unindexed_geometry_matched", "unindexed_geometry_valid_matched",
)
RATE_FIELDS = ("box_precision", "box_recall", "box_f1", "pose_coverage", "unindexed_geometry_coverage")
DISTRIBUTIONS = ("box_iou", "corner_error_px", "corner_error_normalized")
SUMMARY_COLUMNS = ("model", "model_status", "domain", "scenes_recorded", *COUNT_FIELDS, *RATE_FIELDS,
                   *(f"{name}_{stat}" for name in DISTRIBUTIONS for stat in ("n", "mean", "median", "p90", "p95")),
                   "raw_quad_status_counts", "box_ap", "pose_ap")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_sources(report, progress=None):
    receipt = report.get("source_files")
    if not isinstance(receipt, dict) or not receipt:
        raise ValueError("MT_REPORT_SOURCE_RECEIPT_REQUIRED")
    if progress:
        progress("Sprawdzam integralność zapisanych wyników MT…")
    for raw_path, expected in receipt.items():
        path = Path(raw_path)
        if not path.is_file() or _sha(path) != expected:
            raise ValueError("MT_REPORT_SOURCE_CHANGED: " + str(path))


def load_mt_report(run_dir):
    """Load the runner's verified immutable snapshot, including partial runs."""
    from .mt_eval_runner import load_campaign

    report = load_campaign(run_dir)
    _verify_sources(report)
    return report


def summary_rows(report, domain=None):
    rows = []
    for model in report["models"]:
        metrics = model.get("metrics") or {}
        recorded = model.get("rows") or []
        for current_domain in ((domain,) if domain else DOMAINS):
            values = metrics.get("domains", {}).get(current_domain, {})
            row = {"model": model["label"], "model_status": model["status"], "domain": current_domain,
                   "scenes_recorded": sum(current_domain == "ALL" or scene["domain"] == current_domain for scene in recorded)}
            row.update({key: values.get(key) for key in (*COUNT_FIELDS, *RATE_FIELDS)})
            unindexed = values.get("unindexed_geometry_diagnostics") or {}
            row.update({"unindexed_geometry_" + key: unindexed.get(key) for key in ("gt", "matched", "valid_matched", "coverage")})
            row["raw_quad_status_counts"] = (values.get("prediction_diagnostics") or {}).get("raw_quad_status_counts")
            for name in DISTRIBUTIONS:
                row.update({name + "_" + stat: (values.get(name) or {}).get(stat)
                            for stat in ("n", "mean", "median", "p90", "p95")})
            official = metrics.get("official_metrics") or {}
            if any(official.get(key, "NOT_COMPUTED") != "NOT_COMPUTED" for key in ("box_ap", "pose_ap")):
                raise ValueError("MT_REPORT_UNSUPPORTED_AP_PROTOCOL")
            row.update(box_ap=official.get("box_ap", "NOT_COMPUTED"), pose_ap=official.get("pose_ap", "NOT_COMPUTED"))
            rows.append(row)
    return rows


def _text(value, *, percent=False, digits=3):
    if value is None:
        return "NIEPOLICZONE"
    if isinstance(value, str):
        return value
    if not math.isfinite(float(value)):
        raise ValueError("MT_REPORT_NONFINITE_METRIC")
    return (f"{100 * value:.2f}%" if percent else f"{value:.{digits}f}").replace(".", ",")


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _campaign_status(report):
    if (report["status"] == "COMPLETE" and report["models"]
            and all(model["status"] == "COMPLETE" and (model.get("metrics") or {}).get("status") == "COMPLETE"
                    for model in report["models"])):
        return "COMPLETE"
    return "PARTIAL"


def report_markdown(report):
    protocol = report["protocol"]
    preflight = report["preflight"]
    status = _campaign_status(report)
    lines = ["# EVAL396 — MT: lokalizacja i geometria", "", f"Status pomiaru: **{status}**.",
             "Pełne sceny; wynik operacyjny przy jednym progu confidence. AP Box i AP Pose: **NIEPOLICZONE / NOT_COMPUTED**.",
             "396 scen (199 DAY / 197 NIGHT), 479 bbox GT (235 / 244), 478 GT indeksowanego Pose (234 / 244).",
             "30 tablic nieczytelnych dla MZ pozostaje w ocenie MT; 449 tekstów MZ nie jest tutaj przeliczanych.", ""]
    if status != "COMPLETE":
        lines += ["**WYNIK CZĘŚCIOWY:** tabela dotyczy zapisanych scen. Nie jest pełnym porównaniem trzech modeli na EVAL396.",
                  "Nieprzetworzone sceny nie są scenami bez detekcji. Brak wyniku oznacza NIEPOLICZONE, nie zero.", ""]
    lines += ["## Modele", ""]
    lines += [f"- **{label}** — {description}" for label, description in MODEL_DESCRIPTIONS]
    lines += ["", "DAY/NIGHT oznaczają oświetlenie, a „n” rozmiar nano; „n” nie oznacza NIGHT.", "",
              "## Wynik operacyjny", "",
              "| Model | Status | Domena | Zapisane sceny | GT bbox | TP | FP | FN | Precision | Recall | F1 | GT Pose | Poprawne dopasowane Pose | Coverage Pose | Mediana błędu px | P90 błędu px | P95 błędu px | N narożników |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows(report):
        cells = [row["model"], row["model_status"], row["domain"], row["scenes_recorded"],
                 *[row[key] if row[key] is not None else "NIEPOLICZONE" for key in ("box_gt", "box_tp", "box_fp", "box_fn")],
                 *[_text(row[key], percent=True) for key in ("box_precision", "box_recall", "box_f1")],
                 row["pose_gt"] if row["pose_gt"] is not None else "NIEPOLICZONE",
                 row["pose_valid_matched"] if row["pose_valid_matched"] is not None else "NIEPOLICZONE",
                 _text(row["pose_coverage"], percent=True), _text(row["corner_error_px_median"]),
                 _text(row["corner_error_px_p90"]), _text(row["corner_error_px_p95"]),
                 row["corner_error_px_n"] if row["corner_error_px_n"] is not None else "NIEPOLICZONE"]
        lines.append("| " + " | ".join(_cell(value) for value in cells) + " |")
    lines += ["", "P/R/F1 i coverage: więcej = lepiej. Błąd narożników: mniej = lepiej.",
              "Błąd znormalizowany = odległość Euclidean / przekątna bbox GT. Statystyki dotyczą ocenionych narożników; n nie oznacza liczby wszystkich tablic.",
              "CSV zachowuje surowe ułamki, współrzędne i pełną precyzję. Wykres ECDF pokazuje procent ocenionych narożników poniżej danego błędu.", "",
              "## Diagnostyka geometrii", "",
              "| Model | Domena | Dopasowane Pose: brak kpt | Dopasowane Pose: invalid | Bbox fallback | Odwrotny obieg raw | GT wyłączone z indeksów | GT geometryczne bez indeksów | Coverage geometrii bez indeksów |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows(report):
        cells = [row["model"], row["domain"],
                 *[row[key] if row[key] is not None else "NIEPOLICZONE" for key in (
                     "pose_missing_keypoints", "pose_invalid_quad", "pose_bbox_fallback", "pose_reversed_winding",
                     "pose_excluded_gt", "unindexed_geometry_gt")], _text(row["unindexed_geometry_coverage"], percent=True)]
        lines.append("| " + " | ".join(_cell(value) for value in cells) + " |")
    lines += ["", "Geometria bez identyfikacji indeksów obejmuje wszystkie 479 GT, w tym plate_000421; indeksowane błędy Pose obejmują 478 GT.",
              "Pełne statusy również niedopasowanych predykcji są zachowane w CSV oraz summary.json; dopasowane poprawne geometrycznie punkty nie są certyfikatem fizycznej tożsamości ich indeksów.", "",
              "## Polityka i wykluczenie", "",
              "`plate_000421` pozostaje w inferencji sceny, dopasowaniu bbox i diagnostyce geometrii. Jest wyłączona wyłącznie z metryk wymagających tożsamości indeksów Pose.",
              "Oryginalne punkty GT oraz indeksy/confidence predykcji nie są zamieniane w celu poprawienia wyniku. Bbox fallback nie jest poprawnym Pose.",
              "Dopasowanie bbox jest one-to-one, zgodnie z protokołem. Wykluczenie jednego GT Pose nie pochłania dodatkowych, niedopasowanych predykcji.", "",
              "```json", json.dumps(preflight.get("metric_policy", {}), ensure_ascii=False, indent=2), "```", "",
              "## Protokół wykonania", "", "```json", json.dumps(protocol, ensure_ascii=False, indent=2), "```", "",
              "## Pochodzenie", "",
              f"- Katalog pomiaru: `{report['run_dir']}`",
              f"- Protokół SHA: `{protocol.get('protocol_sha256', 'UNKNOWN')}`",
              f"- Selekcja SHA: `{preflight.get('selection_sha', 'UNKNOWN')}`",
              f"- Freeze GT SHA: `{preflight.get('gt_freeze_sha', 'UNKNOWN')}`",
              f"- Decyzja operatora SHA: `{protocol.get('geometry_approval_contract_sha256', 'UNKNOWN')}`",
              f"- Polityka metryk SHA: `{preflight.get('metric_policy_sha256', 'UNKNOWN')}`", ""]
    for model in protocol.get("models", []):
        lines.append(f"- {_cell(model.get('label', '?'))}: checkpoint `{model.get('checkpoint_file', 'UNKNOWN')}`, SHA `{model.get('checkpoint_sha256', 'UNKNOWN')}`")
    lines += ["", "## Ograniczenia", "",
              "- Historyczne etykiety MT zawierają samoprzecięcia i odwrócone obiegi. To ograniczenie interpretacji indeksów Pose; ta kampania nie wykonuje retreningu.",
              "- Poprawny obrys nie certyfikuje fizycznej tożsamości indeksów surowego modelu. Nie szukamy permutacji najlepiej pasującej do GT.",
              "- raw_keypoints oznacza natywne Results.keypoints.data po odjęciu paddingu, przeskalowaniu do pikseli sceny i ograniczeniu XY do granic obrazu przez Ultralytics; nie są to tensory sprzed postprocessingu. Aplikacja nie sortuje tych indeksów ani nie zastępuje ich bboxem.",
              "- Ultralytics 8.4.19 nie zeruje keypointów na podstawie niskiego confidence w obiekcie Keypoints. Zero XY może powstać wskutek ograniczenia współrzędnej do granicy obrazu. Konwersja XY zachowuje ewentualny trzeci składnik confidence; modele [4,2] nie udostępniają confidence narożnika.",
              "- AP nie obliczono: stały próg confidence nie stanowi krzywej AP. Osobny protokół ignorowania GT jest potrzebny dla oficjalnego Pose AP478.",
              "- Dokładne źródłowe SHA nie certyfikują braku near-duplicates ani znajomości korpusu pretrainingu.",
              "- Czas Desktop CUDA służy diagnostyce; nie jest czasem Androida. Wynik MT nie jest wynikiem E2E ani izolowanego MZ.",
              "- elapsed_ms obejmuje wywołanie backendu i odbiór wyników; pierwsza scena nowej instancji obejmuje inicjalizację predictora/backendu oraz warmup. Nie jest to ustabilizowany benchmark wydajności. Osobne speed_ms pochodzą z profilera Ultralytics, uruchamianego po warmup.",
              "- Nie wykonano automatycznej promocji modelu, treningu ani final_test.",
              "- W trybie end2end=True backend jest NMS-free: zapisane iou inferencji nie steruje tłumieniem NMS. Próg IoU dopasowania GT jest odrębny.", "",
              "Pełny kontrakt, decyzje, źródła i SHA: `provenance.json`. To eksport istniejącego pomiaru; eksport nie wykonuje inferencji."]
    return "\n".join(lines) + "\n"


def _svg_start(title, subtitle, *, height=600):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="{height}" viewBox="0 0 1180 {height}">',
            '<rect width="100%" height="100%" fill="#101820"/>',
            f'<text x="24" y="35" fill="#f4f7f6" font-size="23" font-family="Segoe UI, Arial">{escape(title)}</text>',
            f'<text x="24" y="61" fill="#b6c6c1" font-size="13" font-family="Segoe UI, Arial">{escape(subtitle)}</text>']


def _bar_svg(report, metric, title):
    rows = summary_rows(report)
    height = max(230, 120 + 42 * len(rows))
    parts = _svg_start(title, f"{_campaign_status(report)}; stały próg operacyjny; więcej = lepiej. Źródło: "
                       + str(report["protocol"].get("protocol_sha256", "UNKNOWN"))[:16], height=height)
    left, width, axis_y = 325, 720, height - 35
    for i in range(5):
        x = left + width * i / 4
        parts += [f'<line x1="{x}" y1="78" x2="{x}" y2="{axis_y}" stroke="#34484e"/>',
                  f'<text x="{x}" y="{axis_y + 22}" text-anchor="middle" fill="#b6c6c1" font-size="13">{25 * i}%</text>']
    colors = {"MT-n": "#4aa3ff", "MT-DAY": "#2ecc71", "MT-NIGHT": "#f0b44c"}
    for i, row in enumerate(rows):
        y = 82 + i * 42
        label = f"{row['model']} / {row['domain']} ({row['scenes_recorded']} scen)"
        parts.append(f'<text x="24" y="{y + 18}" fill="#f4f7f6" font-size="14" font-family="Segoe UI, Arial">{escape(label)}</text>')
        value = row.get(metric)
        if value is not None:
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("MT_REPORT_INVALID_RATE")
            parts.append(f'<rect x="{left}" y="{y}" width="{value * width:.3f}" height="23" fill="{colors.get(row["model"], "#4aa3ff")}"/>')
        parts.append(f'<text x="{left + width + 10}" y="{y + 18}" fill="#f4f7f6" font-size="12">{escape(_text(value, percent=True))}</text>')
    return "\n".join(parts + ["</svg>"])


def _corner_values(model, domain):
    values = []
    for scene in (model.get("metrics") or {}).get("scenes", []):
        if domain != "ALL" and scene.get("domain") != domain:
            continue
        for match in scene.get("matched", []):
            for value in match.get("corner_errors_normalized") or []:
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("MT_REPORT_INVALID_CORNER_ERROR")
                values.append(float(value))
    return sorted(values)


def corner_distribution_svg(report, domain="ALL"):
    series = [(model["label"], _corner_values(model, domain)) for model in report["models"]]
    maximum = max((100 * value for _, values in series for value in values), default=1.0)
    maximum = max(maximum, 1.0)
    parts = _svg_start(f"Błąd narożników — ECDF — {domain}",
                       f"{_campaign_status(report)}; ocenione narożniki bez reorder/fallback; n = narożniki, nie tablice. Mniej błędu = lepiej.")
    left, top, width, height = 95, 100, 990, 350
    for i in range(5):
        x = left + width * i / 4
        y = top + height - height * i / 4
        parts += [f'<line x1="{left}" y1="{y}" x2="{left + width}" y2="{y}" stroke="#34484e"/>',
                  f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" fill="#b6c6c1" font-size="13">{i * 25}%</text>',
                  f'<text x="{x}" y="{top + height + 24}" text-anchor="middle" fill="#b6c6c1" font-size="13">{_text(maximum * i / 4, digits=2)}%</text>']
    colors = ("#4aa3ff", "#2ecc71", "#f0b44c")
    for index, (label, values) in enumerate(series):
        color = colors[index % len(colors)]
        if values:
            segments = [f"M {left} {top + height}"]
            for rank, value in enumerate(values, 1):
                x = left + 100 * value / maximum * width
                y = top + height - rank / len(values) * height
                segments.append(f"H {x:.3f} V {y:.3f}")
            parts.append(f'<path d="{" ".join(segments)}" fill="none" stroke="{color}" stroke-width="2"/>')
        label_text = f"{label}: n={len(values)}" if values else f"{label}: brak ocenionych narożników"
        parts.append(f'<text x="{left + index * 340}" y="540" fill="{color}" font-size="14" font-family="Segoe UI, Arial">{escape(label_text)}</text>')
    parts += ['<text x="590" y="501" text-anchor="middle" fill="#f4f7f6" font-size="15" font-family="Segoe UI, Arial">Błąd / przekątna bbox GT (%)</text>',
              '<text x="95" y="82" fill="#f4f7f6" font-size="14">Odsetek ocenionych narożników</text>']
    return "\n".join(parts + ["</svg>"])


def _write_csv(path, columns, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json(value) if isinstance(value, (list, tuple, dict)) else value for key, value in row.items()})


def _raw_tables(report):
    scenes, predictions, matches, corners = [], [], [], []
    for model in report["models"]:
        details = {scene["scene_sha256"]: scene for scene in (model.get("metrics") or {}).get("scenes", [])}
        for scene in model.get("rows") or []:
            sha = scene["scene_sha256"]
            common = {"model": model["label"], "scene_sha256": sha, "domain": scene["domain"], "file": scene["file"]}
            detected = scene["predictions"]
            scenes.append(dict(common, width=scene["width"], height=scene["height"], elapsed_ms=scene.get("elapsed_ms"),
                               detections=len(detected), no_detection=not detected, status=scene.get("status", "RECORDED")))
            detail = details.get(sha, {})
            by_prediction = {row["prediction_index"]: row for row in detail.get("matched", [])}
            evaluated = {row["index"]: row for row in detail.get("predictions", [])}
            for match in detail.get("matched", []):
                matches.append(dict(common, **match, outcome="MATCHED"))
            for plate_id in detail.get("unmatched_gt_ids", []):
                matches.append(dict(common, plate_id=plate_id, outcome="UNMATCHED_GT"))
            for prediction in detected:
                index = prediction["index"]
                match = by_prediction.get(index, {})
                diagnosed = evaluated.get(index, {})
                filtered = str(diagnosed.get("box_status", "")).startswith("FILTERED")
                outcome = diagnosed["box_status"] if filtered else ("MATCHED" if match else "UNMATCHED_PREDICTION")
                predictions.append(dict(common, prediction_index=index, bbox=prediction["bbox"],
                    confidence=prediction["confidence"], class_id=prediction["class_id"], raw_keypoints=prediction.get("keypoints"),
                    matched_plate_id=match.get("plate_id"), iou=match.get("iou"),
                    pose_status=match.get("pose_status", diagnosed.get("pose_status")), raw_quad_status=diagnosed.get("raw_quad_status"), outcome=outcome))
                if not match and not filtered:
                    matches.append(dict(common, prediction_index=index, outcome=outcome))
                points = prediction.get("keypoints") or []
                for corner_index, point in enumerate(points):
                    errors = match.get("corner_errors_px") or []
                    normalized = match.get("corner_errors_normalized") or []
                    gt = match.get("gt_quad") or []
                    pair = point if isinstance(point, (list, tuple)) else []
                    corners.append(dict(common, prediction_index=index, corner_index=corner_index,
                        raw_point=point, raw_x=pair[0] if pair else None, raw_y=pair[1] if len(pair) > 1 else None,
                        raw_confidence=pair[2] if len(pair) > 2 else None,
                        matched_plate_id=match.get("plate_id"), pose_status=match.get("pose_status", diagnosed.get("pose_status")),
                        gt_x=gt[corner_index][0] if corner_index < len(gt) else None,
                        gt_y=gt[corner_index][1] if corner_index < len(gt) else None,
                        error_px=errors[corner_index] if corner_index < len(errors) else None,
                        error_normalized=normalized[corner_index] if corner_index < len(normalized) else None))
    return scenes, predictions, matches, corners


def _safe_output(report, output_root):
    output = Path(output_root).resolve()
    preflight = report["preflight"]
    for name in ("selection_dir", "freeze_dir", "source_scene_root"):
        value = preflight.get(name)
        if value and output.is_relative_to(Path(value).resolve()):
            raise ValueError("MT_REPORT_PROTECTED_OUTPUT_DIRECTORY")
    selection = preflight.get("selection_dir")
    if selection:
        mz = Path(selection).resolve().parent.parent / "evaluation_runs" / "E-MZ-DN-01_EVAL396_MZ_v1"
        if output.is_relative_to(mz):
            raise ValueError("MT_REPORT_PROTECTED_OUTPUT_DIRECTORY")
    if "final_test" in {part.lower() for part in output.parts}:
        raise ValueError("MT_REPORT_PROTECTED_OUTPUT_DIRECTORY")
    run = Path(report["run_dir"]).resolve()
    if output.is_relative_to(run) and not output.is_relative_to(run / "reports"):
        raise ValueError("MT_REPORT_USE_RUN_REPORTS_SUBDIRECTORY")
    return output


def export_mt_report(run_dir, output_root, *, progress=None):
    """Export a new report with receipt verification before and after writing."""
    report = load_mt_report(run_dir)
    output = _safe_output(report, output_root)
    _verify_sources(report, progress)
    if progress:
        progress("Tworzę raport MT, tabele danych i wykresy…")
    output.mkdir(parents=True, exist_ok=True)
    name = "eval396_mt_report_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    stage = output / (name + ".partial")
    stage.mkdir()
    (stage / "status.json").write_text(_json({"status": "PARTIAL", "kind": "mt_report_export"}), encoding="utf-8")
    rows = summary_rows(report)
    _write_csv(stage / "summary.csv", SUMMARY_COLUMNS, rows)
    scenes, predictions, matches, corners = _raw_tables(report)
    common = ("model", "scene_sha256", "domain", "file")
    _write_csv(stage / "scenes.csv", (*common, "width", "height", "elapsed_ms", "detections", "no_detection", "status"), scenes)
    _write_csv(stage / "predictions.csv", (*common, "prediction_index", "confidence", "class_id", "bbox", "raw_keypoints",
                                         "matched_plate_id", "iou", "pose_status", "raw_quad_status", "outcome"), predictions)
    _write_csv(stage / "matches.csv", (*common, "plate_id", "prediction_index", "outcome", "iou", "pose_status", "gt_bbox",
                                      "gt_quad", "predicted_bbox", "raw_keypoints", "corner_errors_px", "corner_errors_normalized"), matches)
    _write_csv(stage / "corners.csv", (*common, "prediction_index", "corner_index", "raw_point", "raw_x", "raw_y", "raw_confidence",
                                      "matched_plate_id", "pose_status", "gt_x", "gt_y", "error_px", "error_normalized"), corners)
    (stage / "report.md").write_text(report_markdown(report), encoding="utf-8")
    for metric, title in (("box_f1", "Box F1"), ("box_recall", "Box Recall"), ("pose_coverage", "Coverage indeksowanego Pose")):
        (stage / (metric + ".svg")).write_text(_bar_svg(report, metric, "EVAL396 — " + title), encoding="utf-8")
    for domain in DOMAINS:
        (stage / ("corner_ecdf_" + domain + ".svg")).write_text(corner_distribution_svg(report, domain), encoding="utf-8")
    (stage / "summary.json").write_text(_json({"status": _campaign_status(report), "rows": rows,
        "summaries": {model["label"]: model.get("metrics") for model in report["models"]}}) + "\n", encoding="utf-8")
    provenance = {"schema": "alpr.mt_eval_report.v1", "created_at": datetime.now(timezone.utc).isoformat(),
                  "source_run_dir": report["run_dir"], "measurement_status": report["status"],
                  "protocol": report["protocol"], "preflight": report["preflight"],
                  "geometry_approval": report.get("geometry_approval"), "source_files": report["source_files"],
                  "measurement_source_receipt": report.get("source_receipt", {}),
                  "report_code_sha256": _sha(__file__), "inference_performed_by_export": False}
    (stage / "provenance.json").write_text(_json(provenance) + "\n", encoding="utf-8")
    _verify_sources(report, progress)
    manifest = {path.name: _sha(path) for path in sorted(stage.iterdir()) if path.name != "status.json"}
    (stage / "status.json").write_text(_json({"schema": "alpr.mt_eval_report_status.v1", "kind": "mt_report_export",
        "status": _campaign_status(report), "export_integrity": "VERIFIED", "source_status": report["status"],
        "protocol_sha256": report["protocol"].get("protocol_sha256"), "files_sha256": manifest}), encoding="utf-8")
    destination = output / name
    stage.rename(destination)
    return destination
