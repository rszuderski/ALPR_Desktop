"""Immutable E2E exports: CSV/JSONL, SVG and PNG, Markdown, provenance."""
from datetime import datetime
import csv
import json
from pathlib import Path
import uuid

from .eval396 import canonical, require
from .mt_eval_runner import _sha, _write_json
from .e2e_eval_runner import load_campaign


def summary_rows(report, domain=None):
    rows = []
    for pair in report["protocol"]["pairs"]:
        score = (report.get("scoring") or {}).get(pair["id"])
        if score is None:
            continue
        for group, metrics in score["domains"].items():
            if domain and group != domain:
                continue
            rows.append({"pair": pair["id"], "domain": group, "status": report["status"],
                **{key: metrics[key] for key in ("processed_scenes", "plates", "ground_truth_characters", "exact_matches",
                    "exact_match_rate", "cer", "no_read", "box_tp", "box_fp", "box_fn", "box_precision", "box_recall", "box_f1",
                    "conditional_plates", "conditional_exact_matches", "conditional_exact_rate", "fp_ocr")},
                "box_iou_mean": metrics["box_iou"]["mean"], "box_iou_median": metrics["box_iou"]["median"],
                "box_iou_p90": metrics["box_iou"]["p90"], "box_iou_p95": metrics["box_iou"]["p95"]})
    return rows


def _percent(value):
    return "—" if value is None else f"{100*value:.2f}%".replace(".", ",")


def report_markdown(report):
    p = report["protocol"]
    lines = ["# EVAL396 — jakość pełnego ALPR E2E", "", f"Status pomiaru: **{report['status']}**. Rodzaj danych: **{p['evidence_kind']}**.",
        "Pełna scena po EXIF → zapisane predykcje MT → przewidziany quad → rektyfikacja → MZ → tekst.",
        "GT uczestniczy dopiero w oddzielnym scoringu. Nie jest to oficjalne Pose mAP ani pomiar pełnej latencji.", "",
        f"Sceny ukończone: {report['completed_scenes']}/{report['total_scenes']} (suma po parach).",
        f"Protokół: `{p['protocol_sha256']}`; odbiór operatora: `{report['approval']['contract_sha256']}`.",
        f"Selekcja: `{p['selection_sha256']}`; freeze: `{p['gt_freeze_sha256']}`.",
        f"MT cache: `{p['cache_dir']}`; MT protocol: `{p['mt_protocol_sha256']}`.",
        f"Rektyfikacja: `{p['rectification']['id']}`; reading order: `{p['reading_order']}`; normalizacja: `{p['normalization']}`.", "",
        "| Para | Domena | Exact | Exact % | Micro-CER | NO_READ | bbox P / R / F1 | Średnie IoU TP | Conditional Exact | FP OCR |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for row in summary_rows(report):
        lines.append(f"| {row['pair']} | {row['domain']} | {row['exact_matches']}/{row['plates']} | {_percent(row['exact_match_rate'])} | "
            f"{_percent(row['cer'])} | {row['no_read']} | {_percent(row['box_precision'])} / {_percent(row['box_recall'])} / {_percent(row['box_f1'])} | "
            f"{_percent(row['box_iou_mean'])} | {row['conditional_exact_matches']}/{row['conditional_plates']} | {row['fp_ocr']} |")
    if not report.get("scoring"):
        lines += ["", "PARTIAL — brak zamkniętego kompletu raw. Końcowe metryki nie są dostępne; scen oczekujących nie liczono jako FN/NO_READ."]
    if report.get("scoring"):
        lines += ["", "## Przyczyny niepowodzeń", "", "GT readable i wszystkie GT mają odrębne liczniki; FP OCR jest poza mianownikiem tekstowym.", "",
                  "| Para | Domena | Przyczyna | Readable GT | Wszystkie GT |", "|---|---|---|---:|---:|"]
        for pair, scoring in report["scoring"].items():
            for domain, values in scoring["domains"].items():
                for reason in sorted(set(values["failures_readable"]) | set(values["failures_all_gt"])):
                    lines.append(f"| {pair} | {domain} | {reason} | {values['failures_readable'].get(reason,0)} | {values['failures_all_gt'].get(reason,0)} |")
    lines += ["", "## Kontrakt i ograniczenia", "",
        "- Każda para ocenia wszystkie 396 scen, 479 bbox i 449 zatwierdzonych tekstów; przekroje DAY/NIGHT nie przełączają modeli.",
        "- Mianownik micro-CER jest sumą długości zatwierdzonych tekstów (ALL 3052, DAY 1522, NIGHT 1530); CSV zachowuje ułamek, nie zaokrąglony procent.",
        "- NO_READ na readable GT oznacza pustą predykcję i deletions. 30 unreadable pozostaje w bbox; FP OCR raportuje się osobno.",
        "- Conditional Exact jest uzupełniający: readable GT matched do wyprostowanego cropu, także z pustym MZ.",
        "- Wyłączenie plate_000421 dotyczy tylko indeksowanego Pose. Readability pochodzi z odrębnej zamrożonej decyzji tekstowej.",
        "- Raw k0–k3 zachowano; osobna permutacja homografii nie certyfikuje fizycznej tożsamości narożników.",
        "- Historyczne wady etykiet MT pozostają ograniczeniem eksperymentalnym. Nie wykonano ponownego treningu.",
        "- Brak bbox fallback, MP, ROI, AutoZoom, EasyOCR, reguł numerów i agregacji czasowej.",
        "- Cache MT służy jakości przy tym samym punkcie pracy. Nie używać czasu MZ + historycznego MT jako zmierzonej latencji E2E.",
        "- Nie wybrano zwycięzcy ani nie wykonano final_test.", "", "## Rzeczywiste parametry i modele", "",
        "Parametr YOLO end2end dotyczy głowicy sieci. Gdy jest aktywny, iou=0.45 MZ nie steruje NMS; próg matching GT 0.50 jest osobny.", "",
        "```json", json.dumps({"inference": p["inference"], "runtime": p["runtime"], "rectification": p["rectification"],
            "pairs": p["pairs"], "mt_models": p["mt_models"], "mz_models": p["mz_models"]}, ensure_ascii=False, indent=2), "```", ""]
    if report.get("io_compatibility"):
        addition = report["io_compatibility"]
        writer = addition["document"]["writer_sha256"]
        lines += ["## Zgodność poprawki zapisu I/O", "",
            "Kontynuacja tego przebiegu używa zatwierdzonego retry atomowej publikacji Windows. "
            "Oryginalny protokół, podpis i historyczne SHA pozostają bez zmian.",
            f"Dokument zgodności: `{addition['document_sha256']}`.",
            f"SHA modułu zapisu: `{writer['original']}` → `{writer['corrected']}`.",
            "Rzeczywisty kod kontynuacji i zakres poprawki: `io_compatibility_applied.json`.", ""]
    return "\n".join(lines)


def _csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def _charts(report, output):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    pairs = [p["id"] for p in report["protocol"]["pairs"]]
    for metric, filename, title in (("exact_match_rate", "e2e_exact", "Exact Match — wszystkie readable GT"),
                                    ("cer", "e2e_cer", "Micro-CER — suma edycji / znaki GT"),
                                    ("box_f1", "e2e_bbox_f1", "Detekcja bbox — F1 przy IoU ≥ 0,50")):
        fig = Figure(figsize=(max(8, len(pairs)*1.6), 4.8), dpi=150)
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        rows = summary_rows(report)
        if rows:
            for offset, domain, color in ((-.24, "DAY", "#dfad3a"), (0, "NIGHT", "#5264a5"), (.24, "ALL", "#28876d")):
                by_pair = {r["pair"]: r for r in rows if r["domain"] == domain}
                values = [100 * (by_pair[key][metric] or 0) for key in pairs]
                bars = ax.bar([i+offset for i in range(len(pairs))], values, width=.24, label=domain, color=color)
                ax.bar_label(bars, labels=[_percent(by_pair[key][metric]) for key in pairs], fontsize=7, padding=2)
            ax.set_xticks(range(len(pairs)), [p.replace("__", " +\n") for p in pairs])
            upper = max(100, max(100*(r[metric] or 0) for r in rows)*1.15)
            ax.set_ylim(0, upper)
            if upper == 100:
                ax.set_yticks([0, 25, 50, 75, 100])
            ax.legend(loc="upper right")
            ax.set_ylabel("%")
            ax.grid(axis="y", alpha=.2)
        else:
            ax.text(.5, .5, "PARTIAL — metryki oczekują na komplet raw", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        fig.suptitle(report["protocol"]["evidence_kind"], fontsize=8)
        fig.tight_layout()
        for extension in ("svg", "png"):
            fig.savefig(output / (filename + "." + extension), format=extension)


def export_report(run_dir, output_root, *, progress=None):
    report = load_campaign(run_dir)
    require(report["status"] != "RUNNING", "E2E_EXPORT_RUNNING_FORBIDDEN")
    p, root = report["protocol"], Path(output_root).resolve()
    protected = p["protected_roots"] + [report["run_dir"]] + [str(Path(m["checkpoint_file"]).parent) for m in p["mt_models"] + p["mz_models"]]
    forbidden_components = {"final_test", "4_training_datasets", "5_training_runs", "6_models", "9_projects"}
    require(not any(root.is_relative_to(Path(value).resolve()) for value in protected)
            and not forbidden_components.intersection(part.lower() for part in root.parts), "E2E_REPORT_PROTECTED_DIRECTORY")
    # Do not create subdirectories inside any finished report/experiment.
    require(not any((parent / "completion_manifest.json").exists() or (parent / "provenance.json").exists()
                    or parent.name.startswith(("ranking_report_", "eval396_mt_report_", "eval396_e2e_report_", "eval396_import_"))
                    for parent in (root, *root.parents)), "E2E_REPORT_INSIDE_EXISTING_ARTIFACT")
    inputs = {str(file.resolve()): _sha(file) for file in Path(run_dir).rglob("*") if file.is_file()}
    name = "eval396_e2e_report_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    stage, destination = root / (name + ".partial"), root / name
    stage.mkdir(parents=True, exist_ok=False)
    _write_json(stage / "status.json", {"status": "PARTIAL", "source_status": report["status"]})
    summaries = summary_rows(report)
    _csv(stage / "summary.csv", summaries, list(summaries[0]) if summaries else ["pair", "domain", "status", "exact_match_rate", "cer"])
    scenes, detections, characters, gt, fp = [], [], [], [], []
    for pair, rows in report["pairs"].items():
        for scene in rows:
            scenes.append({"pair": pair, "scene_sha256": scene["scene_sha256"], "status": scene["status"], "detections": len(scene["detections"])})
    for name, row in report["ledger"]["rows"].items():
        if row.get("schema") != "alpr.e2e.prediction.v1":
            continue
        common = {"pair": row["pair_id"], "scene_sha256": row["scene_sha256"], "prediction_index": row["mt"]["index"]}
        detections.append({**common, "status": row["status"], "failure": row["failure"], "text": row["reading"]["prediction"],
            "raw_mt": row["mt"], "rectification": row["rectification"], "reading_order": row["reading"]["ordered_predictions"],
            "effective_inference": row["effective_inference"], "mz_elapsed_ms": row["mz_elapsed_ms"]})
        characters.extend({**common, **char} for char in row["characters"])
    for pair, score in (report.get("scoring") or {}).items():
        gt.extend({"pair": pair, **r} for r in score["gt_rows"])
        fp.extend({"pair": pair, **r} for r in score["fp_rows"])
    for filename, rows, fields in (
        ("scenes.csv", scenes, ["pair", "scene_sha256", "status", "detections"]),
        ("detections.csv", detections, ["pair", "scene_sha256", "prediction_index", "status", "failure", "text", "raw_mt", "rectification", "reading_order", "effective_inference", "mz_elapsed_ms"]),
        ("characters.csv", characters, ["pair", "scene_sha256", "prediction_index", "index", "class_id", "character", "bbox", "confidence"]),
        ("gt_scoring.csv", gt, ["pair", "scene_sha256", "plate_id", "domain", "prediction_index", "iou", "readable", "conditional_eligible", "prediction", "failure", "scoring"]),
        ("fp_ocr.csv", fp, ["pair", "scene_sha256", "prediction_index", "domain", "prediction", "failure", "fp_ocr"])):
        _csv(stage / filename, rows, fields)
    (stage / "rows.jsonl").write_bytes(b"".join(canonical(row)+b"\n" for row in gt))
    (stage / "report.md").write_text(report_markdown(report), encoding="utf-8")
    _write_json(stage / "summary.json", summaries, exclusive=True)
    _write_json(stage / "scoring.json", report.get("scoring") or {}, exclusive=True)
    if report.get("io_compatibility"):
        _write_json(stage / "io_compatibility_applied.json", report["io_compatibility"], exclusive=True)
    if progress:
        progress("Tworzę SVG i PNG z zapisanych wyników…")
    _charts(report, stage)
    _write_json(stage / "provenance.json", {"schema": "alpr.e2e.report_provenance.v1", "source_run": report["run_dir"],
        "protocol": p, "approval": report["approval"], "source_receipt": report["prepared"]["source_receipt"],
        "source_files": inputs, "io_compatibility": report.get("io_compatibility"),
        "export_code_sha256": _sha(__file__), "inference_performed_by_export": False}, exclusive=True)
    for file, sha in inputs.items():
        require(_sha(file) == sha, "E2E_SOURCE_CHANGED_DURING_EXPORT")
    files = {file.name: _sha(file) for file in stage.iterdir() if file.is_file() and file.name != "status.json"}
    _write_json(stage / "status.json", {"schema": "alpr.e2e.report_status.v1", "status": report["status"],
        "source_status": report["status"], "export_integrity": "VERIFIED", "files_sha256": files})
    require(not destination.exists(), "E2E_REPORT_DESTINATION_EXISTS")
    stage.rename(destination)
    return destination
