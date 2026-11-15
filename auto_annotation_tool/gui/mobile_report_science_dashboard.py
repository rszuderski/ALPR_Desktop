"""Z4MOBREP016: a readable, source-grounded scientific dashboard.

The calculation of MT/Pose/MZ/ALPR is delegated to the existing canonical
mobile_scientific_review module. This module is presentation-only: it never
changes reviews, models, session archives, methodology or raw statistics.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import tkinter as tk
from tkinter import ttk


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _count(value: Any) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _text_fraction(n: Any, d: Any) -> str:
    n, d = _count(n), _count(d)
    if n is None or d is None or d <= 0 or n > d:
        return "Brak oceny"
    return f"{n} z {d}"


def _format_pct(value: Any, *, places: int = 1) -> str:
    num = _number(value)
    return (f"{num:.{places}f}%".replace(".", ",") if num is not None else "—")


def _latency_value_ms(report: Any, bundle: Any, name: str, field: str) -> tuple[float | None, int | None]:
    """Read native aggregate values; never infer from missing/partial trace previews."""
    payload = _dict(getattr(bundle, "report_payload", None)) if bundle is not None else {}
    latency = _dict(payload.get("latency")) or _dict(getattr(report, "latency", None))
    stages = _dict(latency.get(name))
    n = _count(stages.get("count"))
    if n is None or n == 0:
        return None, n
    aliases = {"p50": ("median_ms", "p50_ms"), "p95": ("p95_ms",)}
    for key in aliases.get(field, (field,)):
        value = _number(stages.get(key))
        if value is not None:
            return value, n
    return None, n


def _time_ms(value: float | None) -> str:
    return f"{value / 1000:.2f} s".replace(".", ",") if value is not None else "—"


def science_from_verified_bundle(bundle: Any) -> dict:
    """Read-only, canonical metrics; must be called off the Tk thread."""
    if bundle is None or not bool(getattr(getattr(bundle, "validation", None), "ok", False)):
        raise ValueError("Brak zweryfikowanego archiwum sesji")
    from ..ranking.mobile_human_review import MobileReviewSession
    from ..ranking.mobile_scientific_review import make_scientific_summary
    from .z4_mobile_sample_review import _pose_geometry_rows

    session = MobileReviewSession(bundle)  # Source + SHA-pinned operator sidecar.
    from ..ranking.mobile_verification_policy import verification_progress
    result = make_scientific_summary(session, session.statistics(), _pose_geometry_rows(session))
    # Distinguish mandatory form completeness from MT scientific denominators.
    result["form_verification"] = verification_progress(session)
    from ..ranking.mobile_canvas_presentation import build_inventory
    inventory = build_inventory(session)
    result["_ui_canvas_progress"] = inventory['operator_review_progress']
    return result


def build_overview(report: Any, bundle: Any = None, science: Any = None, *, loading: bool = False,
                   error: str = "") -> dict:
    """Pure readable view model. Unknown is never converted to zero.

    An unverified source or an unrelated/stale review MUST NOT produce model
    quality percentages. Performance aggregates from the saved report remain
    visible but are explicitly distinguished from verified source measurements.
    """
    source_hash = str(getattr(report, "source_archive_sha256", "") or "").lower()
    from .mobile_report_canvas_view import canvas_overview
    from .mobile_model_identity import identity_from_bundle
    from .z4_mobile_report_browser_ux import report_identity
    info = report_identity(report, bundle)
    canvas = canvas_overview(_dict(science).get("_ui_canvas_snapshot"),
        session_id=info["session_id"], source_sha=source_hash,
        progress=_dict(science).get("_ui_canvas_progress"), loading=loading)
    if error:
        canvas.update(status="Nie można potwierdzić weryfikacji", progress_ratio=None,
                      progress="Nie ustalono liczby obrazów do oceny",
                      progress_subtitle="Nie udało się odczytać oceny operatora: " + error)
    validated = bool(bundle is not None and getattr(getattr(bundle, "validation", None), "ok", False)
                     and str(getattr(bundle, "source_archive_sha256", "") or "").lower() == source_hash
                     and str(getattr(getattr(bundle, "report", None), "report_id", "")) == str(getattr(report, "report_id", "")))
    identity = identity_from_bundle(bundle if validated else None)
    if not validated or str(_dict(science).get("source_archive_sha256") or "").lower() != source_hash:
        science = None
    blocks = [
        {"title": "MT (telefon) · lokalizacja tablicy", "value": "Brak wyniku", "note": "Model MT wskazuje obszary tablic. Potrzebna weryfikacja operatora."},
        {"title": "Pose MT (telefon) · dokładność obrysu", "value": "Brak pomiaru", "note": "IoU i NMCE wymagają niezależnych, ręcznie wskazanych narożników GT."},
        {"title": "MZ (telefon) · odczyt z wycięcia", "value": "Brak wyniku", "note": "Poprawność numeru z pojedynczego cropa względem ręcznego GT."},
        {"title": "ALPR (telefon) · rozpoznanie numeru", "value": "Brak wyniku", "note": "Poprawny numer w obserwacji po przejściu całego potoku."},
    ]
    progress = "Formularz weryfikacji: brak danych"
    progress_subtitle = "Podłącz zweryfikowane archiwum i weryfikację operatora."
    progress_ratio = None
    if science is not None:
        state = _dict(science)
        p = _dict(state.get("progress"))
        form = _dict(state.get("form_verification"))
        total = _count(form.get("total"))
        ready = _count(form.get("verified"))
        issues = form.get("issues", ())
        if total is not None and ready is not None and total >= ready:
            progress = f"Kompletny formularz: {ready} z {total} obserwacji"
            progress_ratio = ready / total if total else None
            state_name = str(form.get("status") or "")
            if state_name == "READY_FOR_EVALUATION":
                progress = f"Weryfikacja zatwierdzona: {ready} z {total} obserwacji"
            elif state_name == "READY_TO_APPROVE":
                progress = f"Formularz kompletny: {ready} z {total} · czeka na zatwierdzenie"
            active = _count(p.get("mt_executed_active"))
            rated = _count(p.get("mt_invocations_cardinality_reviewed"))
            check_count = (f"Liczba tablic zweryfikowana w {rated}/{active} wywołaniach MT"
                           if active is not None and rated is not None else
                           "Liczba tablic MT: brak pełnego zestawienia")
            remaining = len(issues) if isinstance(issues, (list, tuple)) else None
            progress_subtitle = (check_count +
                                 (f" · decyzje do uzupełnienia: {remaining}" if remaining is not None else ""))
        else:
            progress = "Weryfikacja formularza: brak potwierdzonego statusu"
            progress_subtitle = "Pozostaje dołączenie aktualnego formularza operatora."
        mt = _dict(state.get("mt"))
        mt_d = _count(mt.get("single_plate_n"))
        mt_n = _count(mt.get("localized_n"))
        blocks[0]["value"] = (f"{mt_n}/{mt_d} trafnych wskazań" if mt_d and mt_n is not None and mt_n <= mt_d
                              else "Brak wyniku")
        if mt_d and mt_n is not None and mt_n <= mt_d:
            blocks[0]["note"] = ("Model MT uruchomiony na telefonie: trafienia / zweryfikowane "
                                 f"wywołania z jedną rzeczywistą tablicą ({mt_d}). "
                                 f"Brak prawidłowego wskazania: {mt_d-mt_n}. "
                                 "To nie jest pełny recall ani mAP.")
        pose = _dict(state.get("pose"))
        pn = _count(pose.get("measured_n"))
        if pn:
            iou = _number(pose.get("polygon_iou_mean_pct"))
            nmce = _number(pose.get("nmce_mean_pct"))
            if iou is not None and nmce is not None:
                blocks[1]["value"] = f"IoU {_format_pct(iou)}"
                blocks[1]["note"] = ("Średnie pokrycie obrysu Pose MT i ręcznego GT: "
                                      f"{pn} sparowane obrysy (n={pn}). "
                                      f"Średni błąd narożników NMCE: {_format_pct(nmce)}. "
                                      "Nie obejmuje tablic bez GT obrysu.")
        missed = _count(pose.get("gt_missed_plate_n"))
        if not pn and missed:
            blocks[1]["note"] = f"Ręczne obrysy tablic pominiętych przez MT: {missed}. Bez predykcji nie liczymy IoU/NMCE."
        mz = _dict(state.get("mz"))
        mz_n = _count(mz.get("evaluable_reads"))
        mz_exact = _count(mz.get("exact_n"))
        blocks[2]["value"] = (f"{mz_exact}/{mz_n} poprawnych" if mz_n and mz_exact is not None else "Brak wyniku")
        if mz_n and mz_exact is not None:
            cer = _number(mz.get("cer"))
            blocks[2]["note"] = (f"Model MZ na telefonie: poprawne całe numery / {mz_n} zweryfikowanych "
                                 "wycięć (cropów) z ręcznym GT. " +
                                 (f"CER znaków: {_format_pct(cer * 100)}." if cer is not None
                                  else "CER znaków: brak danych."))
        alpr = _dict(state.get("alpr"))
        an = _count(alpr.get("evaluable_subjects"))
        ae = _count(alpr.get("exact_n"))
        blocks[3]["value"] = (f"{ae}/{an} poprawnych" if an and ae is not None else "Brak wyniku")
        if an and ae is not None:
            blocks[3]["note"] = ("Cały potok ALPR na telefonie: obserwacje z przynajmniej "
                                 f"jednym poprawnym numerem / {an} kwalifikujących się obserwacji "
                                 "z ręcznym GT. Inny mianownik niż MZ.")

    speed = {}
    for stage, label in (("mt", "Model MT"), ("mz", "Model MZ"), ("pipeline", "Potok ALPR")):
        med, n = _latency_value_ms(report, bundle if validated else None, stage, "p50")
        p95, _ = _latency_value_ms(report, bundle if validated else None, stage, "p95")
        speed[stage] = {"name": label, "median": _time_ms(med), "p95": _time_ms(p95), "n": n}
    from .z4_mobile_report_cards_ux import frame_counts
    flow = frame_counts(report, bundle if validated else None)
    received, processed, gate = flow.get("received"), flow.get("processed"), flow.get("gate")
    if received is not None and processed is not None:
        frames = f"Przetworzono {processed} z {received} odebranych klatek"
        if gate is not None:
            frames += f" · {gate} pominięto przez filtr klatek (nie przez model MT)"
    elif processed is not None:
        frames = f"Przetworzono {processed} klatek; liczba odebranych nieznana"
    else:
        frames = "Brak pełnych liczników klatek"

    if not validated:
        status = "Brak potwierdzenia archiwum źródłowego"
        warnings = ["Czasy pochodzą z zapisanego raportu. Weryfikacja jakości wymaga oryginalnego archiwum i GT."]
    elif loading and science is None:
        status = "Wczytuję zapis weryfikacji operatora…"
        warnings = ["Weryfikacja zostanie dołączona tylko po sprawdzeniu zgodności archiwum."]
    elif error:
        status = "Nie można odczytać weryfikacji"
        warnings = ["Archiwum jest poprawne, ale zapis GT wymaga sprawdzenia: " + str(error)[:170]]
    elif science is None:
        status = "Brak dostępnego formularza weryfikacji"
        warnings = ["Bez zweryfikowanego GT nie podajemy miar jakości modeli."]
    else:
        state = str(science.get("scientific_state") or "")
        form_state = str(_dict(science.get("form_verification")).get("status") or "")
        status = ("Formularz zatwierdzony — dane gotowe do ewaluacji" if form_state == "READY_FOR_EVALUATION" else
                  "Formularz kompletny — wymaga zatwierdzenia" if form_state == "READY_TO_APPROVE" else
                  "Wyniki wstępne — weryfikacja trwa")
        warnings = []
        if str(science.get("review_mode")) == "assisted":
            warnings.append("Weryfikacja z widocznymi predykcjami (niezaślepiona); opisz ten tryb w metodyce.")
        if (_count(_dict(science.get("mt")).get("single_plate_n")) or 0) < 30:
            warnings.append("Mało zweryfikowanych wywołań z jedną tablicą; wynik MT ma charakter opisowy.")
        if not warnings and form_state != "READY_FOR_EVALUATION":
            warnings.append("Nie zatwierdzono kompletnej weryfikacji; wyniki jakości są wstępne.")
    perf_n = speed["pipeline"]["n"]
    if perf_n is not None and perf_n < 30 and "—" not in speed["pipeline"]["p95"]:
        warnings.append(f"Czas P95 opiera się na {perf_n} pomiarach; przy małej próbie jest niestabilny.")
    source_text = "Integralność źródła potwierdzona (ZIP/SHA-256) · " + source_hash[:12] + "…" if validated else "Raport zapisany · integralność źródła jeszcze niepotwierdzona"
    from ..ranking.mobile_operator_progress import evidence_caption
    evidence = evidence_caption(_dict(science).get('_ui_canvas_progress')) if validated else ''
    if evidence:
        source_text += "\n" + evidence
    if _dict(science).get('_ui_canvas_progress', {}).get('schema') == 'alpr.desktop.operator_progress.v1' and validated:
        status, progress = canvas['status'], canvas['progress']
        progress_subtitle, progress_ratio = canvas['progress_subtitle'], canvas['progress_ratio']
    return {"status": status, "progress": progress, "progress_subtitle": progress_subtitle,
            "progress_ratio": progress_ratio, "cards": tuple(blocks), "performance": speed,
            "frames": frames, "warnings": tuple(warnings[:2]), "source": source_text,
            "quality_verified": science is not None, "source_verified": validated,
            "canvas": canvas, "model_identity": identity,
            "package_caption": "Pakiet: " + identity["caption"],
            "model_context": f"{info['scenario'] or 'Scenariusz: brak danych'} · {identity['repeat_label']} · {identity['variant_caption']}"}


class ScientificDashboard(tk.Frame):
    """Accessible, compact Tk panel; all views are populated from build_overview."""
    def __init__(self, master, *, palette: dict, on_review: Callable, on_refresh: Callable,
                 on_details: Callable, show_identity: bool = True):
        bg = palette.get("panel", "#252526")
        fg = palette.get("fg", "#eeeeee")
        muted = palette.get("muted", "#bbbbbb")
        border = palette.get("border", "#555555")
        accent = palette.get("accent", "#4f8de3")
        super().__init__(master, bg=bg)
        self._bg, self._accent, self._fg = bg, accent, fg
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scroll.set)
        self.content = tk.Frame(canvas, bg=bg, padx=14, pady=11)
        window_id = canvas.create_window((0, 0), window=self.content, anchor="nw")
        def on_canvas_width(event):
            canvas.itemconfigure(window_id, width=event.width)
            if hasattr(self, "caveat_label"):
                self.caveat_label.configure(wraplength=max(230, event.width - 48))
            for name in ("package_label", "model_context_label", "history_progress_label", "progress_info_label",
                         "time_title_label", "time_description_label", "frame_label"):
                if hasattr(self, name):
                    getattr(self, name).configure(wraplength=max(230, event.width - 52))
            if hasattr(self, "history"):
                self.history.header.configure(wraplength=max(200, event.width-76))
        canvas.bind("<Configure>", on_canvas_width)
        self.content.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        self.canvas = canvas
        self.content.grid_columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Wybierz sesję z listy")
        self.progress_var = tk.StringVar(value="")
        self.progress_info_var = tk.StringVar(value="")
        self.frame_var = tk.StringVar(value="")
        self.source_var = tk.StringVar(value="")
        self.caveat_var = tk.StringVar(value="")
        self.package_var = tk.StringVar(value="Pakiet/MT/MZ: brak danych źródłowych")
        self.model_context_var = tk.StringVar(value="")
        header = tk.Frame(self.content, bg=bg)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(header, text="Weryfikacja operatora", bg=bg, fg=fg,
                 font=("Segoe UI", 12, "bold"), anchor="w").grid(row=0, column=0, sticky="w")
        tk.Label(header, textvariable=self.status_var, bg=bg, fg=muted, font=("Segoe UI", 9),
                 anchor="w", justify="left").grid(row=1, column=0, sticky="ew", pady=(3, 0))
        # One unmistakable main action: verification is not an auxiliary detail.
        # Keep source refresh as a small text link, never as a competing CTA.
        controls = tk.Frame(header, bg=bg)
        controls.grid(row=0, column=1, rowspan=2, sticky="e", padx=(10, 0))
        tk.Button(controls, text="Odśwież wyniki", command=on_refresh,
                  bg=bg, fg=accent, activebackground=bg, activeforeground=fg,
                  relief=tk.FLAT, bd=0, highlightthickness=0,
                  font=("Segoe UI", 9, "underline"), cursor="hand2").pack(side="left")
        self.package_label = tk.Label(header, textvariable=self.package_var, bg=bg, fg=fg,
            font=("Segoe UI", 10, "bold"), anchor="w", justify="left", wraplength=600)
        self.package_label.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.model_context_label = tk.Label(header, textvariable=self.model_context_var, bg=bg, fg=muted,
            font=("Segoe UI", 9), anchor="w", justify="left", wraplength=600)
        self.model_context_label.grid(row=3, column=0, columnspan=2, sticky="ew")
        # Browser already presents this identity in its persistent report header.
        if not show_identity:
            self.package_label.grid_remove()
            self.model_context_label.grid_remove()
        action = tk.Frame(self.content, bg=bg)
        action.grid(row=1, column=0, sticky="ew", pady=(2, 11))
        action.grid_columnconfigure(0, weight=1)
        action_button = tk.Button(action, text="Przejdź do weryfikacji próbek  →",
                                  command=on_review, bg=accent, fg="#ffffff",
                                  activebackground=accent, activeforeground="#ffffff",
                                  font=("Segoe UI", 9, "bold"), relief=tk.FLAT,
                                  bd=0, padx=12, pady=5, cursor="hand2")
        action_button.grid(row=0, column=0, sticky="ew")
        self.review_action_button = action_button
        progress = tk.Frame(self.content, bg=bg)
        progress.grid(row=2, column=0, sticky="ew", pady=(1, 10))
        progress.grid_columnconfigure(0, weight=1)
        tk.Label(progress, textvariable=self.progress_var, bg=bg, fg=fg, font=("Segoe UI", 10, "bold"),
                 anchor="w").grid(row=0, column=0, sticky="ew")
        self.progress_info_label = tk.Label(progress, textvariable=self.progress_info_var, bg=bg,
            fg=muted, font=("Segoe UI", 9), anchor="w", justify="left", width=1)
        self.progress_info_label.grid(row=1, column=0, sticky="ew", pady=(2, 5))
        self.progress_bar = ttk.Progressbar(progress, mode="determinate", maximum=100, value=0)
        self.progress_bar.grid(row=2, column=0, sticky="ew")
        self.card_vars = []
        self._card_frames = []
        self._card_notes = []
        self._card_titles = []
        self._card_columns = 2
        cards = tk.Frame(self.content, bg=bg)
        self.cards_host = cards
        cards.grid(row=3, column=0, sticky="ew")
        cards.grid_columnconfigure(0, weight=1, uniform="science_cards")
        cards.grid_columnconfigure(1, weight=1, uniform="science_cards")
        cards.bind("<Configure>", lambda event: self._reflow_cards(event.width), add="+")
        titles = ("MT — wykrycie tablic", "MT — dopasowanie obrysu",
                  "MZ — pełne zgodne odczyty", "MZ — błędy znaków")
        for i, name in enumerate(titles):
            box = tk.Frame(cards, bg=bg, highlightbackground=border, highlightthickness=1, padx=10, pady=7)
            box.grid(row=i // 2, column=i % 2, sticky="nsew", padx=(0 if i % 2 == 0 else 5, 5 if i % 2 == 0 else 0), pady=(0, 7))
            box.grid_columnconfigure(0, weight=1)
            title_label = tk.Label(box, text=name, bg=bg, fg=muted,
                                   font=("Segoe UI", 9, "bold"), anchor="w",
                                   justify="left", wraplength=280)
            title_label.grid(row=0, column=0, sticky="ew")
            v = tk.StringVar(value="Brak wyniku")
            note = tk.StringVar(value="")
            tk.Label(box, textvariable=v, bg=bg, fg=fg, font=("Segoe UI", 16, "bold"), anchor="w").grid(row=1, column=0, sticky="ew", pady=(3, 1))
            note_label = tk.Label(box, textvariable=note, bg=bg, fg=muted, font=("Segoe UI", 9),
                     anchor="w", justify="left", wraplength=280)
            note_label.grid(row=2, column=0, sticky="ew")
            self.card_vars.append((v, note))
            self._card_frames.append(box)
            self._card_notes.append(note_label)
            self._card_titles.append(title_label)
        timebox = tk.Frame(self.content, bg=bg)
        timebox.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        timebox.grid_columnconfigure(0, weight=1)
        self.time_title_label = tk.Label(timebox, text="Czas wykonywania na telefonie (bez weryfikacji operatora)",
                 bg=bg, fg=fg, font=("Segoe UI", 11, "bold"),
                 anchor="w", justify="left", width=1)
        self.time_title_label.grid(row=0, column=0, sticky="ew")
        self.time_description_label = tk.Label(timebox, text="Czas w sekundach · mediana i P95 z pomiarów Androida; n = liczba wywołań etapu",
                 bg=bg, fg=muted, font=("Segoe UI", 8), anchor="w", justify="left", width=1)
        self.time_description_label.grid(row=1, column=0, sticky="ew", pady=(2, 4))
        self.speed_vars = {}
        speedgrid = tk.Frame(timebox, bg=bg)
        speedgrid.grid(row=2, column=0, sticky="ew")
        for idx, key in enumerate(("mt", "mz", "pipeline")):
            speedgrid.grid_columnconfigure(idx, weight=1, uniform="science_times")
            item = tk.Frame(speedgrid, bg=bg, highlightthickness=1, highlightbackground=border, padx=7, pady=5)
            item.grid(row=0, column=idx, sticky="ew", padx=(0 if idx == 0 else 5, 0))
            title = {"mt": "Model MT · lokalizacja", "mz": "Model MZ · odczyt", "pipeline": "Potok ALPR · przetwarzanie"}[key]
            label = tk.Label(item, text=title, bg=bg, fg=muted, font=("Segoe UI", 9), anchor="w", justify="left", width=1)
            label.pack(fill="x")
            label.bind("<Configure>", lambda e: e.widget.configure(wraplength=max(80,e.width-4)))
            text = tk.StringVar(value="—")
            self.speed_vars[key] = text
            tk.Label(item, textvariable=text, bg=bg, fg=fg, font=("Segoe UI", 9), justify="left", anchor="w").pack(fill="x", pady=(2, 0))
        self.frame_label = tk.Label(timebox, textvariable=self.frame_var, bg=bg, fg=muted, font=("Segoe UI", 8),
                 anchor="w", justify="left", width=1)
        self.frame_label.grid(row=3, column=0, sticky="ew", pady=(5, 0))
        self.caveat_label = tk.Label(self.content, textvariable=self.caveat_var,
                                    bg=bg, fg=muted, font=("Segoe UI", 9),
                                    anchor="w", justify="left", wraplength=900)
        self.caveat_label.grid(row=5, column=0, sticky="ew", pady=(9, 2))
        bottom = tk.Frame(self.content, bg=bg)
        bottom.grid(row=6, column=0, sticky="ew", pady=(4, 0))
        bottom.grid_columnconfigure(0, weight=1)
        tk.Label(bottom, textvariable=self.source_var, bg=bg, fg=muted, font=("Segoe UI", 8),
                 anchor="w").grid(row=0, column=0, sticky="ew")
        from .mobile_canvas_ui import Disclosure, palette as local_palette
        self.history = Disclosure(self.content,
            "Oddzielny protokół historyczny / niezależna ocena nieprzeprowadzona",
            p=local_palette(self, bg))
        self.history.grid(row=7, column=0, sticky="ew", pady=(10, 2))
        self.history_progress_var = tk.StringVar(value="")
        self.history_progress_label = tk.Label(self.history.body, textvariable=self.history_progress_var,
            bg=self.history.body.cget("bg"), fg=fg, justify="left", anchor="w", wraplength=600)
        self.history_progress_label.pack(fill="x", pady=(0, 8))
        self.history_vars = []
        for _ in range(4):
            value = tk.StringVar(value="")
            box = tk.Label(self.history.body, textvariable=value, justify="left", anchor="w",
                bg=bg, fg=muted, padx=8, pady=8, wraplength=550,
                highlightthickness=1, highlightbackground=border)
            box.pack(fill="x", pady=3)
            box.bind("<Configure>", lambda e: e.widget.configure(wraplength=max(100, e.width-20)))
            self.history_vars.append(value)
        # All source data and methods are accessible via the already visible
        # 'Szczegóły' tab; a second navigation button would only duplicate it.

    def _reflow_cards(self, width: int) -> None:
        """Keep the four cards in two rows in the laptop report pane."""
        columns = 2 if width >= 450 else 1
        width_each = max(145, int(width / columns) - 34)
        for idx, frame in enumerate(self._card_frames):
            col = idx % columns
            frame.grid_configure(row=idx // columns, column=col, sticky="nsew",
                                 padx=(0 if col == 0 else 5, 5 if col < columns - 1 else 0),
                                 pady=(0, 7))
            self._card_notes[idx].configure(wraplength=width_each)
            self._card_titles[idx].configure(wraplength=width_each)
        for col in range(4):
            self._card_frames[0].master.grid_columnconfigure(col,
                    weight=1 if col < columns else 0,
                    uniform="science_cards" if col < columns else "")
        self._card_columns = columns

    def render(self, data: dict):
        primary = data.get("canvas") or {}
        self.status_var.set(str(primary.get("status", "BRAK · Do weryfikacji")))
        self.progress_var.set(str(primary.get("progress", "")))
        self.progress_info_var.set(str(primary.get("progress_subtitle", "")))
        self.package_var.set(data.get("package_caption", "Pakiet/MT/MZ: brak danych źródłowych"))
        self.model_context_var.set(data.get("model_context", ""))
        self.history_progress_var.set(str(data.get("status", ""))+"\n"+
            str(data.get("progress", ""))+"\n"+str(data.get("progress_subtitle", "")))
        for variable, row in zip(self.history_vars, data.get("cards", ())):
            variable.set(f"{row['title']}\n{row['value']}\n{row['note']}")
        if data.get("quality_verified") and any(row.get("value") not in ("Brak wyniku", "Brak pomiaru", "—") for row in data.get("cards", ())):
            self.history.title = "Oddzielny protokół historyczny · ocena i własne mianowniki"
        else:
            self.history.title = "Oddzielny protokół historyczny / niezależna ocena nieprzeprowadzona"
        self.history._sync()
        self.cards_host.grid() if primary.get("ready") else self.cards_host.grid_remove()
        ratio = primary.get("progress_ratio")
        if isinstance(ratio, (int, float)) and 0 <= ratio <= 1:
            self.progress_bar.configure(value=round(100 * ratio, 1))
            self.progress_bar.grid()
        else:
            self.progress_bar.grid_remove()
        cards = primary.get("cards", ())
        for i, variables in enumerate(self.card_vars):
            values = cards[i] if i < len(cards) else {}
            variables[0].set(str(values.get("value", "Brak wyniku")))
            variables[1].set(str(values.get("note", "")))
        for key, value in self.speed_vars.items():
            item = _dict(data.get("performance")).get(key, {})
            n = _count(item.get("n"))
            value.set(f"Mediana: {item.get('median', '—')}\nP95: {item.get('p95', '—')} · n={n if n is not None else '—'}")
        self.frame_var.set(str(data.get("frames", "")))
        self.caveat_var.set(primary.get("methodology", ""))
        self.source_var.set(str(data.get("source", "")))
