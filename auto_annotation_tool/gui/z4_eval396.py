"""EVAL396 views reached from the existing Z4 track/results/report controls."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import uuid

from ..ranking.eval396 import (DOMAINS, VARIANTS, EvidenceReader, load_eval396,
                              is_selection, default_selection, selection_reference)

MODEL_DESCRIPTIONS = (
    ("MZ-s", "Bazowy model rozpoznawania znaków YOLO Detect w wariancie small. "
     "Punkt odniesienia, bez dodatkowego dostrajania DAY/NIGHT."),
    ("MZ-DAY", "Model dostrojony z wykorzystaniem materiału i augmentacji dziennych."),
    ("MZ-NIGHT", "Model dostrojony z wykorzystaniem materiału i augmentacji nocnych."),
)
MODEL_NAMING_NOTE = (
    "DAY i NIGHT oznaczają warunki oświetleniowe, natomiast „s” oznacza rozmiar modelu (small), nie domenę."
)


def show_model_info(parent):
    existing = getattr(parent, "_model_info_dialog", None)
    if existing is not None and existing.winfo_exists():
        existing.lift()
        return existing
    dialog = tk.Toplevel(parent)
    parent._model_info_dialog = dialog
    dialog.title("O porównywanych modelach")
    dialog.transient(parent)
    dialog.resizable(False, False)
    shell = ttk.Frame(dialog, padding=20)
    shell.pack(fill="both", expand=True)
    ttk.Label(shell, text="O porównywanych modelach", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 14))
    for name, description in MODEL_DESCRIPTIONS:
        ttk.Label(shell, text=name, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(shell, text=description, wraplength=540, justify="left").pack(anchor="w", pady=(3, 12))
    ttk.Separator(shell).pack(fill="x", pady=(0, 12))
    ttk.Label(shell, text=MODEL_NAMING_NOTE, wraplength=540, justify="left").pack(anchor="w")
    close = ttk.Button(shell, text="Zamknij", command=dialog.destroy)
    close.pack(anchor="e", pady=(16, 0))
    dialog.bind("<Escape>", lambda _event: dialog.destroy())
    dialog.update_idletasks()
    dialog.lift()
    close.focus_set()
    return dialog


def selected_eval396(tab):
    variable = getattr(tab, "rank_data_dir", None)
    return is_selection(variable.get()) if variable is not None else False


def participant_models(selection_dir):
    """Read the pinned participant list, without history reconciliation or model loading."""
    from ..ranking import eval396 as contract
    selection_dir = Path(selection_dir).resolve()
    if selection_dir.name == "selection_manifest.json":
        selection_dir = selection_dir.parent
    reader = EvidenceReader()
    manifest = reader.json(selection_dir / "selection_manifest.json", contract.SELECTION_MANIFEST_SHA)
    contract.require(manifest.get("schema") == "emzdn01.eval_scene_selection.v1"
                     and manifest.get("selection_fingerprint_sha256") == contract.SELECTION_SHA,
                     "SELECTION_FINGERPRINT_MISMATCH")
    models = reader.json(selection_dir / "model_lineage.json", manifest["files"]["model_lineage.json"])
    contract.require(len(models) == len(contract.MODELS)
                     and {m["label"] for m in models} == set(contract.MODELS), "SIX_MODELS_REQUIRED")
    for model in models:
        role = "plate" if model["label"].startswith("MT-") else "character"
        contract.require(model.get("role") == role and bool(model.get("checkpoint_file"))
                         and len(model.get("checkpoint_sha256", "")) == 64
                         and bool(model.get("run_id")), "MODEL_LINEAGE_INVALID")
    reader.finish()
    return sorted(models, key=lambda m: contract.MODELS.index(m["label"]))


def open_participants(tab):
    """The Horses CTA shows the fixed EVAL396 models, separately from result windows."""
    selection_dir = Path(tab.rank_data_dir.get()).resolve()
    if selection_dir.name == "selection_manifest.json":
        selection_dir = selection_dir.parent
    existing = getattr(tab, "_eval396_participants_dialog", None)
    if existing is not None and existing.winfo_exists():
        if existing.selection_dir == selection_dir:
            existing.deiconify()
            existing.lift()
            return existing
        existing.destroy()
    dialog = tk.Toplevel(tab.frame)
    tab._eval396_participants_dialog = dialog
    dialog.selection_dir = selection_dir
    dialog.title("Z4 — EVAL396 — konie / modele")
    dialog.geometry("1080x520")
    dialog.minsize(780, 440)
    dialog.resizable(True, True)
    dialog.wm_transient("")
    dialog._aat_skip_window_recovery = True
    shell = ttk.Frame(dialog, padding=16)
    shell.pack(fill="both", expand=True)
    shell.columnconfigure(0, weight=1)
    shell.rowconfigure(3, weight=1)
    ttk.Label(shell, text="Konie — modele EVAL396", font=("Segoe UI", 16, "bold")).grid(row=0, sticky="w")
    intro = ttk.Label(shell, text="EVAL396 ma ustalony skład: 3 modele MT i 3 modele MZ. "
                      "Zaznacz model, aby zobaczyć jego wagi i pochodzenie.", wraplength=1000)
    intro.grid(row=1, sticky="ew", pady=(5, 12))
    controls = ttk.Frame(shell)
    controls.grid(row=2, sticky="ew", pady=(0, 8))
    ttk.Label(controls, text="Modele:").pack(side="left")
    groups = ("MT — tablice", "MZ — znaki", "Wszystkie")
    target = getattr(tab, "_get_ranking_task_target", lambda: "")()
    group = tk.StringVar(value=groups[0] if target == "plate" else groups[1] if target in {"char", "character"} else groups[2])
    choice = ttk.Combobox(controls, textvariable=group, values=groups, state="readonly", width=22)
    choice.pack(side="left", padx=(6, 16))
    status = tk.StringVar(value="Odczytuję listę modeli…")
    ttk.Label(controls, textvariable=status).pack(side="left")
    table_frame = ttk.Frame(shell)
    table_frame.grid(row=3, sticky="nsew")
    table_frame.columnconfigure(0, weight=1)
    table_frame.rowconfigure(0, weight=1)
    tree = ttk.Treeview(table_frame, columns=("model", "task", "variant", "run", "weights"),
                        show="headings", selectmode="browse", height=6)
    for column, caption, width in (("model", "Model", 120), ("task", "Zadanie", 170),
                                   ("variant", "Wariant", 120), ("run", "Trening", 170), ("weights", "Wagi", 190)):
        tree.heading(column, text=caption)
        tree.column(column, width=width, minwidth=90, stretch=column in ("task", "weights"))
    tree.grid(row=0, column=0, sticky="nsew")
    ys = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    xs = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
    ys.grid(row=0, column=1, sticky="ns")
    xs.grid(row=1, column=0, sticky="ew")
    details = ttk.LabelFrame(shell, text="Wybrany model", padding=10)
    details.grid(row=4, sticky="ew", pady=(10, 8))
    details.columnconfigure(1, weight=1)
    description, weights, sha, parent = (tk.StringVar() for _ in range(4))
    description_label = ttk.Label(details, textvariable=description, wraplength=960)
    description_label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 7))
    for row, title, variable in ((1, "Plik wag:", weights), (2, "SHA w protokole:", sha),
                                  (3, "Trening bazowy:", parent)):
        ttk.Label(details, text=title).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=2)
        ttk.Entry(details, textvariable=variable, state="readonly").grid(row=row, column=1, sticky="ew")
    bottom = ttk.Frame(shell)
    bottom.grid(row=5, sticky="ew")
    ttk.Label(bottom, text="DAY / NIGHT: oświetlenie • n / s: rozmiar modelu. Weryfikacja wag odbywa się przed pomiarem.",
              wraplength=650).pack(side="left", fill="x", expand=True)

    def close():
        if getattr(tab, "_eval396_participants_dialog", None) is dialog:
            tab._eval396_participants_dialog = None
        dialog.destroy()

    ttk.Button(bottom, text="Zamknij", command=close).pack(side="right", padx=(12, 0))
    dialog.protocol("WM_DELETE_WINDOW", close)
    dialog.bind("<Escape>", lambda _event: close())
    models = {}
    descriptions = dict(MODEL_DESCRIPTIONS)
    descriptions.update({
        "MT-n": "Bazowy model lokalizacji tablic YOLO Pose w wariancie nano.",
        "MT-DAY": "Model MT-n dostrojony z wykorzystaniem materiału i augmentacji dziennych.",
        "MT-NIGHT": "Model MT-n dostrojony z wykorzystaniem materiału i augmentacji nocnych."})

    def selected(_event=None):
        selection = tree.selection()
        model = models.get(selection[0]) if selection else None
        description.set(descriptions[model["label"]] if model else "")
        weights.set(model["checkpoint_file"] if model else "")
        sha.set(model["checkpoint_sha256"] if model else "")
        parent.set((model.get("parent_run_id") or "Model bazowy") if model else "")

    def refresh(_event=None):
        tree.delete(*tree.get_children())
        for label, model in models.items():
            if group.get() == groups[0] and model["role"] != "plate":
                continue
            if group.get() == groups[1] and model["role"] != "character":
                continue
            tree.insert("", "end", iid=label, values=(label,
                "Lokalizacja tablic" if model["role"] == "plate" else "Rozpoznawanie znaków",
                "Bazowy" if not model.get("parent_run_id") else label.split("-", 1)[1],
                model["run_id"], Path(model["checkpoint_file"]).name))
        status.set(f"Pokazano {len(tree.get_children())} z {len(models)} modeli protokołu")
        if tree.get_children():
            tree.selection_set(tree.get_children()[0])
        selected()

    def resized(event):
        if event.widget is dialog:
            intro.configure(wraplength=max(500, event.width - 40))
            description_label.configure(wraplength=max(500, event.width - 70))

    tree.bind("<<TreeviewSelect>>", selected)
    choice.bind("<<ComboboxSelected>>", refresh)
    dialog.bind("<Configure>", resized, add="+")
    dialog.participant_table, dialog.participant_group = tree, choice
    dialog.participant_status = status
    dialog.participant_weights, dialog.participant_sha = weights, sha
    try:
        models.update((m["label"], m) for m in participant_models(selection_dir))
        refresh()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        status.set("Nie można odczytać modeli EVAL396")
        description.set(str(exc))
        choice.configure(state="disabled")
    return dialog


def track_candidate(repo_root):
    path = default_selection(repo_root)
    if not path.is_dir():
        return None
    try:
        info = selection_reference(path)
        return {"path": str(path), "id": info["reference_name"], "type": "MT / MZ",
                "split": "EVAL396", "split_count": 396, "counts": {}, "created": "2026-10-09",
                "source": "Zamrożony", "status": "do weryfikacji", "ready": True,
                "details": "396 scen • 479 tablic MT • 449 tekstów MZ"}
    except (OSError, ValueError) as exc:
        return {"path": str(path), "id": "EVAL396 — zamrożona selekcja", "ready": False,
                "status": "FAIL", "details": str(exc)}


def report_rows(report, domain=None, criterion="Exact Match"):
    rows = []
    for selected in ((domain,) if domain else DOMAINS):
        group = [{"model": label, "domain": selected, **report["summaries"][label]["per_domain"][selected]}
                 for label in VARIANTS]
        group.sort(key=lambda row: (-row["exact_match_rate"], row["model"]) if criterion == "Exact Match"
                   else (row["cer"], row["model"]))
        rows.extend(group)
    return rows


def percent_text(ratio):
    return f"{ratio * 100:.2f}%".replace(".", ",")


def report_markdown(report, criterion="Exact Match"):
    lines = ["# EVAL396 — ukończone pomiary MZ", "", "Status: PASS (zweryfikowany import istniejących wyników).",
             "Izolowane MZ na zatwierdzonych cropach GT. To nie jest mAP, E2E ani pomiar Androida.",
             "396 scen; 479 tablic lokalizacji; 449 zatwierdzonych tekstów; 30 unreadable poza mianownikiem OCR.",
             f"Porządek: {criterion}; bez automatycznego wyboru modelu wdrożeniowego.", "",
             "## Porównywane modele", ""]
    lines += [f"- **{name}** — {description}" for name, description in MODEL_DESCRIPTIONS]
    lines += ["", MODEL_NAMING_NOTE, "", "## Wyniki", "",
              "| Model | Domena | Tablice | Exact | Exact % | CER % | CER (wartość) | no-read | Poprawne | Zamiany | Braki | Nadmiary |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report_rows(report, criterion=criterion):
        lines.append(f"| {row['model']} | {row['domain']} | {row['plates']} | {row['exact_matches']} | "
                     f"{row['exact_match_rate'] * 100:.2f} | {percent_text(row['cer'])} | {row['cer']:.4f} | {row['no_read']} | "
                     f"{row['correct_characters']} | {row['incorrect_characters']} | {row['missing_characters']} | {row['extra_characters']} |")
    lines += ["", "CER = suma odległości edycyjnych / liczba znaków GT (3052 ALL, 1522 DAY, 1530 NIGHT).", "",
              "## Pochodzenie", ""]
    for key in ("experiment_id", "selection_sha", "gt_freeze_sha", "comparison_sha", "selection_dir", "runs_dir"):
        lines.append(f"- {key}: `{report[key]}`")
    for model in report["models"]:
        lines.append(f"- {model['label']}: run `{model['run_id']}`, checkpoint SHA `{model['checkpoint_sha256']}`")
    lines += ["", "Parametry inferencji zapisanych pomiarów:", "", "```json",
              json.dumps(report["inference"], ensure_ascii=False, indent=2), "```", "", "## Ograniczenia", ""]
    lines += ["- " + text for text in report["limitations"]]
    lines += ["", "Pełne źródła i SHA plików: `provenance.json`. Import nie wykonuje inferencji ani treningu."]
    return "\n".join(lines) + "\n"


def export_report(report, output_root, criterion="Exact Match", *, progress=None):
    """Versioned report, never a new measurement. Verify sources again before publishing."""
    from .z4_analysis_ranking import _ranking_report_bar_svg
    output_root = Path(output_root).resolve()
    for name in ("selection_dir", "runs_dir"):
        if output_root.is_relative_to(Path(report[name]).resolve().parent.parent):
            raise ValueError("Raport musi być poza chronionym katalogiem eksperymentu.")
    reader = EvidenceReader(progress)
    reader.receipt = dict(report["source_files"])
    reader.finish()
    reader.progress("Tworzę tabelę, raport i wykresy…")
    output_root.mkdir(parents=True, exist_ok=True)
    name = "eval396_import_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    partial = output_root / (name + ".partial")
    partial.mkdir()
    (partial / "status.json").write_text('{"status":"PARTIAL","kind":"report_export"}', encoding="utf-8")
    rows = report_rows(report, criterion=criterion)
    columns = ("model", "domain", "plates", "exact_matches", "exact_match_rate", "cer", "no_read",
               "ground_truth_characters", "edit_distance", "correct_characters", "incorrect_characters",
               "missing_characters", "extra_characters")
    with (partial / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    (partial / "report.md").write_text(report_markdown(report, criterion), encoding="utf-8")
    for metric, label in (("exact_match_rate", "Exact Match"), ("cer", "CER")):
        maximum = 100 if metric == "exact_match_rate" else max(20, max(row[metric] * 100 for row in rows))
        bars = [{"label": row["model"] + " / " + row["domain"],
                 "score": row[metric] * 100, "score_label": percent_text(row[metric])} for row in rows]
        subtitle = "449 zatwierdzonych cropów GT; DAY 230 / NIGHT 219. " + (
            "Więcej = lepiej." if metric == "exact_match_rate" else "Mniej = lepiej; micro CER = edit distance / znaki GT.")
        subtitle += f" Skala wykresu: 0–{percent_text(maximum / 100)}."
        svg = _ranking_report_bar_svg(bars, title="EVAL396 — MZ — " + label + " (%)",
                                     maximum=maximum, unit="%", precision=2, subtitle=subtitle, axis_ticks=True)
        (partial / (metric + ".svg")).write_text(svg, encoding="utf-8")
    (partial / "provenance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reader.finish()
    (partial / "status.json").write_text('{"status":"PASS","kind":"report_export"}', encoding="utf-8")
    target = output_root / name
    partial.rename(target)
    return target


def open_results(tab, *, export_after_load=False):
    existing = getattr(tab, "_eval396_dialog", None)
    if existing is not None and existing.winfo_exists():
        if existing.state() == "iconic":
            existing.deiconify()
        existing.lift()
        if export_after_load:
            existing.export_eval_report()
        return existing
    dialog = tk.Toplevel(tab.frame)
    tab._eval396_dialog = dialog
    dialog.title("Z4 — EVAL396 — wyniki MZ i geometria MT")
    dialog.geometry("1160x790")
    dialog.minsize(860, 640)
    # Windows removes the native minimize/maximize controls for transient dialogs.
    dialog.wm_transient("")
    dialog.resizable(True, True)
    dialog._aat_skip_window_recovery = True
    notebook = ttk.Notebook(dialog)
    notebook.pack(fill="both", expand=True)
    shell = ttk.Frame(notebook, padding=16)
    notebook.add(shell, text="MZ — rozpoznawanie znaków")
    from .z4_mt_geometry import GeometryReviewPanel
    geometry = GeometryReviewPanel(notebook, tab.rank_data_dir.get())
    notebook.add(geometry, text="MT — lokalizacja i geometria")
    dialog.evaluation_notebook, dialog.mt_geometry_panel = notebook, geometry
    from .z4_e2e_results import install_e2e_panel
    dialog.after_idle(lambda: install_e2e_panel(dialog, notebook, tab.rank_data_dir.get()))
    shell.columnconfigure(0, weight=1)
    shell.rowconfigure(6, weight=1)
    ttk.Label(shell, text="EVAL396 — zamrożona selekcja", font=("Segoe UI", 17, "bold")).grid(row=0, sticky="w")
    ttk.Label(shell, text="396 scen  •  479 tablic MT  •  449 tekstów MZ  •  30 nieczytelnych",
              font=("Segoe UI", 11)).grid(row=1, sticky="w", pady=(5, 3))
    ttk.Label(shell, text="Odczyt ukończonych MZ na cropach GT. MT: oczekuje na decyzję EXIF. Final_test: nieużyty.",
              wraplength=1040).grid(row=2, sticky="w", pady=(0, 10))
    status = tk.StringVar(value="INCOMPLETE — trwa weryfikacja SHA, rodowodu i zapisanych wyników…")
    progress_area = ttk.Frame(shell)
    progress_area.grid(row=3, sticky="ew", pady=(0, 10))
    progress_area.columnconfigure(0, weight=1)
    ttk.Label(progress_area, textvariable=status, wraplength=1040).grid(row=0, sticky="ew")
    progress_bar = ttk.Progressbar(progress_area, mode="indeterminate", maximum=100)
    progress_bar.grid(row=1, sticky="ew", pady=(6, 0))
    controls = ttk.Frame(shell)
    controls.grid(row=4, sticky="ew", pady=(0, 10))
    domain, criterion = tk.StringVar(value="ALL"), tk.StringVar(value="Exact Match")
    for text, variable, values in (("Domena:", domain, DOMAINS), ("Porządek:", criterion, ("Exact Match", "CER"))):
        ttk.Label(controls, text=text).pack(side="left", padx=(0, 6))
        combo = ttk.Combobox(controls, textvariable=variable, values=values, state="readonly", width=14)
        combo.pack(side="left", padx=(0, 16))
        combo.bind("<<ComboboxSelected>>", lambda _event: refresh())
    ttk.Label(controls, text="Exact: więcej = lepiej  •  CER: mniej = lepiej", wraplength=180).pack(side="left")
    ttk.Button(controls, text="[ ? ] O porównywanych modelach",
               command=lambda: show_model_info(dialog)).pack(side="right", padx=(12, 0))
    columns = ("model", "plates", "exact", "cer", "no_read", "correct", "substitutions", "missing", "extra")
    table = ttk.Treeview(shell, columns=columns, show="headings", height=len(VARIANTS))
    for name, label, width in zip(columns,
            ("Model", "Tablice", "Exact Match", "CER (%)", "no-read", "Poprawne", "Zamiany", "Braki", "Nadmiary"),
            (125, 70, 155, 92, 72, 85, 75, 70, 75)):
        table.heading(name, text=label)
        table.column(name, width=width, minwidth=65, anchor="w" if name == "model" else "center")
    table.grid(row=5, sticky="ew")
    canvas = tk.Canvas(shell, height=200, background="#101820", highlightthickness=0)
    canvas.grid(row=6, sticky="nsew", pady=12)
    details = tk.Text(shell, height=7, wrap="word", font=("Consolas", 9))
    details.grid(row=7, sticky="ew")
    bottom = ttk.Frame(shell)
    bottom.grid(row=8, sticky="ew", pady=(12, 0))
    export = ttk.Button(bottom, text="[ RAPORT ] Eksportuj SVG / CSV / Markdown", state="disabled")
    export.pack(side="left")
    ttk.Button(bottom, text="Zamknij", command=dialog.destroy).pack(side="right")
    messages, report = queue.Queue(), None
    selection_path = Path(tab.rank_data_dir.get())
    busy = True
    pending_export = export_after_load

    def refresh():
        table.delete(*table.get_children())
        canvas.delete("all")
        if report is None:
            return
        rows = report_rows(report, domain.get(), criterion.get())
        for row in rows:
            table.insert("", "end", values=(row["model"], row["plates"],
                f"{percent_text(row['exact_match_rate'])} ({row['exact_matches']}/{row['plates']})",
                percent_text(row["cer"]), row["no_read"], row["correct_characters"], row["incorrect_characters"],
                row["missing_characters"], row["extra_characters"]))
        exact = criterion.get() == "Exact Match"
        maximum = 100 if exact else max(20, max(row["cer"] * 100 for row in rows))
        canvas.create_text(16, 19, anchor="w", fill="#f4f7f6", font=("Segoe UI", 11, "bold"),
                           text=f"MZ / {domain.get()} — {criterion.get()} (%) — GT crops")
        available = max(200, canvas.winfo_width() - 260)
        axis_y = canvas.winfo_height() - 30
        canvas.create_line(150, axis_y, 150 + available, axis_y, fill="#9fb0aa")
        for index in range(5):
            x = 150 + available * index / 4
            value = maximum * index / 4
            canvas.create_line(x, 38, x, axis_y, fill="#26343a", dash=(3, 4))
            canvas.create_line(x, axis_y, x, axis_y + 5, fill="#9fb0aa")
            canvas.create_text(x, axis_y + 16, text=f"{value:g}%".replace(".", ","),
                               fill="#c1cdc8", tags=("axis_tick",))
        step = max(1, (canvas.winfo_height() - 80) / max(1, len(rows)))
        bar_height = min(48, step * .55)
        for index, row in enumerate(rows):
            ratio = row["exact_match_rate"] if exact else row["cer"]
            value = ratio * 100
            y = 44 + index * step + (step - bar_height) / 2
            canvas.create_text(16, y + bar_height / 2, text=row["model"], anchor="w", fill="#f4f7f6", font=("Segoe UI", 11))
            canvas.create_rectangle(150, y, 150 + available * value / maximum, y + bar_height,
                                    fill="#4aa3ff", outline="")
            canvas.create_text(165 + available * value / maximum, y + bar_height / 2,
                               text=percent_text(ratio), anchor="w", fill="#f4f7f6", font=("Segoe UI", 11, "bold"))

    def finish_progress(success):
        progress_bar.stop()
        progress_bar.configure(mode="determinate", value=100 if success else 0)

    def request_export():
        nonlocal busy, pending_export
        notebook.select(shell)
        if report is None:
            pending_export = True
            start_selected_view()
            return
        if busy:
            return
        pending_export = False
        busy = True
        export.configure(state="disabled")
        progress_bar.configure(mode="indeterminate", value=0)
        progress_bar.start(40)
        chosen = criterion.get()
        def worker():
            from ..config import CONFIG
            try:
                destination = export_report(report, Path(CONFIG.get_ranking_dir("char")) / "reports", chosen,
                    progress=lambda text: messages.put(("progress", text)))
                messages.put(("exported", destination))
            except Exception as exc:
                messages.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()
        status.set("Ponownie sprawdzam źródła przed eksportem raportu…")

    def load_worker():
        try:
            data = load_eval396(selection_path, progress=lambda text: messages.put(("progress", text)))
            messages.put(("loaded", data))
        except Exception as exc:
            messages.put(("error", str(exc)))

    def poll():
        nonlocal report, busy
        if not dialog.winfo_exists():
            return
        try:
            while True:
                kind, value = messages.get_nowait()
                if kind == "progress":
                    status.set("INCOMPLETE — " + value)
                elif kind == "loaded":
                    report, busy = value, False
                    finish_progress(True)
                    dialog.verified_report = value
                    status.set("PASS — SHA, 6 modeli, rodowód, 449 odczytów/model i agregacje zgodne. Źródła niezmienione.")
                    details.configure(state="normal")
                    details.delete("1.0", "end")
                    details.insert("1.0", "\n".join(f"{key}: {value[key]}" for key in
                        ("experiment_id", "selection_sha", "gt_freeze_sha", "comparison_sha", "runs_dir")) +
                        "\nOgraniczenia: near-duplicates i korpus pretrainingu nie są certyfikowane; import nie wybiera zwycięzcy.")
                    details.configure(state="disabled")
                    export.configure(state="normal")
                    refresh()
                    if pending_export:
                        request_export()
                elif kind == "exported":
                    busy = False
                    finish_progress(True)
                    export.configure(state="normal")
                    status.set("PASS — raport zapisany: " + str(value))
                elif kind == "error":
                    report, busy = None, False
                    finish_progress(False)
                    dialog.verified_report = None
                    status.set("FAIL — " + str(value))
                    export.configure(state="disabled")
                    details.configure(state="normal")
                    details.delete("1.0", "end")
                    details.configure(state="disabled")
                    refresh()
        except queue.Empty:
            pass
        dialog._eval396_poll_job = dialog.after(80, poll)

    def stop_timers(event):
        if event.widget is dialog:
            if progress_bar.winfo_exists():
                progress_bar.stop()
            job = getattr(dialog, "_eval396_poll_job", None)
            if job is not None:
                dialog.after_cancel(job)

    dialog.export_eval_report = request_export
    dialog.result_table, dialog.status_variable = table, status
    dialog.progress_bar, dialog.result_chart = progress_bar, canvas
    dialog.bind("<Destroy>", stop_timers, add="+")
    canvas.bind("<Configure>", lambda _event: refresh())
    export.configure(command=request_export)
    dialog.update_idletasks()
    dialog.lift()
    mz_started = False
    def start_selected_view(_event=None):
        nonlocal mz_started
        if notebook.select() == str(shell) and not mz_started:
            mz_started = True
            progress_bar.start(40)
            threading.Thread(target=load_worker, daemon=True).start()
    notebook.bind("<<NotebookTabChanged>>", start_selected_view, add="+")
    target = getattr(tab, "_get_ranking_task_target", lambda: "char")()
    notebook.select(geometry if target == "plate" and not export_after_load else shell)
    start_selected_view()
    dialog._eval396_poll_job = dialog.after(80, poll)
    return dialog
