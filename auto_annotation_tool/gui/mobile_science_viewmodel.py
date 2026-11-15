"""Z4MOBREV031: lossless, read-only scientific summary presentation.

This module does not calculate study metrics, modify the session, infer GT, or
interpret human-assigned multi-plate links as formal model TP/FP/FN.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class MetricRow:
    metric_id: str
    group: str
    label: str
    formatted_value: str
    numerator: int | None
    denominator: int | None
    unit: str
    scientific_scope: str
    status: str
    reason: str
    source_path: str
    review_provenance: str
    is_formal: bool
    is_partial: bool


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return int(number) if isfinite(number) and number >= 0 and number.is_integer() else None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) else None


def _pct(value: Any, *, fraction: bool = False) -> str:
    value = _number(value)
    if value is None:
        return "—"
    return (f"{value * (100 if fraction else 1):.2f}%").replace(".", ",")


def _summary_count(numerator: Any, denominator: Any, fraction: Any) -> str:
    n, d = _count(numerator), _count(denominator)
    if n is None or not d or n > d:
        return "—"
    suffix = f" ({_pct(fraction, fraction=True)})" if _number(fraction) is not None else ""
    return f"{n} z {d}{suffix}"


def _status(*, available: bool, source_ok: bool, preliminary: bool,
            assisted: bool = False, diagnostic: bool = False,
            formal_unavailable: bool = False) -> str:
    if not source_ok:
        return "SOURCE_UNVERIFIED"
    if formal_unavailable:
        return "NOT_EVALUABLE"
    if not available:
        return "MISSING"
    if diagnostic or assisted:
        return "DIAGNOSTIC"
    return "PRELIMINARY" if preliminary else "AVAILABLE"


STATUS_DISPLAY = {
    "SOURCE_UNVERIFIED": "⚠ Źródło niesprawdzone",
    "NOT_EVALUABLE": "— Nie można ocenić",
    "MISSING": "○ Brak danych",
    "NOT_RUN": "— Nie wykonano",
    "DIAGNOSTIC": "◒ Diagnostyczny",
    "PRELIMINARY": "◒ Wstępny",
    "AVAILABLE": "✓ Obliczony",
    "INVALID": "⚠ Niespójne dane",
}


def make_science_dashboard(report: dict, form: dict | None = None, *, source_verified: bool = True) -> dict:
    """Format canonical report fields, never derive or replace scientific metrics.

    ``form`` is the existing ``verification_progress(session)`` output.
    ``source_verified`` must be determined by the calling validated session.
    """
    report = _dict(report)
    form = _dict(form)
    mt, pose, mz, alpr = (_dict(report.get(key)) for key in ("mt", "pose", "mz", "alpr"))
    performance = _dict(report.get("performance"))
    progress = _dict(report.get("progress"))
    multi = _dict(report.get("multi_plate"))
    condition = _dict(report.get("conditions"))
    f5a = condition.get('experiment_family') == 'F5A'
    status = str(form.get("status") or "NOT_STARTED")
    mode = str(report.get("review_mode") or "")
    assisted = mode == "assisted"
    formal_multi_blocked = _count(multi.get("total_multi_invocations")) not in (None, 0) and str(
        multi.get("formal_status") or "NOT_EVALUABLE") != "MEASURED"
    preliminary = (str(report.get("scientific_state") or "") != "FINAL_FULL_REVIEW"
                   or str(report.get("review_status") or "") != "COMPLETED")
    if formal_multi_blocked:
        preliminary = True

    rows: list[MetricRow] = []
    scope = "asystowana" if assisted else "niezależna, o ile udokumentowana"

    def add(metric_id: str, group: str, label: str, value: str, numerator: Any, denominator: Any,
            *, unit: str, reason: str, source_path: str,
            diagnostic: bool = False, independent: bool = False,
            formal_unavailable: bool = False) -> None:
        nonlocal rows
        available = value != "—"
        row_status = _status(available=available, source_ok=source_verified,
                             preliminary=preliminary,
                             assisted=assisted and not independent,
                             diagnostic=diagnostic,
                             formal_unavailable=formal_unavailable)
        if not source_verified:
            value = "—"
            reason = "Brak potwierdzonej zgodności archiwum i weryfikacji."
        rows.append(MetricRow(
            metric_id, group, label, value,
            _count(numerator) if source_verified else None,
            _count(denominator) if source_verified else None,
            unit, "niezależne GT" if independent else scope, row_status, reason,
            source_path, "independent" if independent else mode or "unknown",
            independent and not formal_unavailable,
            preliminary,
        ))

    n_mt, good_mt = _count(mt.get("single_plate_n")), _count(mt.get("localized_n"))
    mt_value = _summary_count(good_mt, n_mt, mt.get("success_rate"))
    add("mt_localization", "MT", "Potwierdzona lokalizacja", mt_value, good_mt, n_mt,
        unit="wywołania MT z jedną tablicą",
        reason=("Tylko zweryfikowane wywołania z jedną tablicą; to nie jest recall ani precision."
                if mt_value != "—" else "Brak ocenionych wywołań MT z jedną tablicą."),
        source_path="mt.localized_n / mt.single_plate_n", diagnostic=True)
    no_det = _count(mt.get("no_detection_n"))
    add("mt_no_detection", "MT", "Brak jakiejkolwiek detekcji",
        str(no_det) if n_mt and no_det is not None else "—", no_det, n_mt,
        unit="ocenione wywołania MT",
        reason="Nie jest to liczba pominiętych tablic poza obszarem ROI.",
        source_path="mt.no_detection_n", diagnostic=True)

    n_pose = _count(pose.get("measured_n"))
    iou = _number(pose.get("polygon_iou_mean_pct"))
    nmce = _number(pose.get("nmce_mean_pct"))
    pose_reason = ("Im większy wynik, tym zgodniejsze obrysy; tylko niezależne GT."
                   if n_pose and iou is not None else
                   "Brak niezależnych, porównywalnych obrysów GT.")
    add("pose_iou", "Pose", "Zgodność obrysu (IoU)",
        _pct(iou) if n_pose else "—", None, n_pose,
        unit="niezależne pary MT–GT", reason=pose_reason,
        source_path="pose.polygon_iou_mean_pct / pose.measured_n", independent=True)
    add("pose_nmce", "Pose", "Błąd narożników (NMCE)",
        _pct(nmce) if n_pose else "—", None, n_pose,
        unit="niezależne pary MT–GT", reason=("Im mniejszy wynik, tym lepiej."
        if n_pose and nmce is not None else "Brak niezależnych, porównywalnych obrysów GT."),
        source_path="pose.nmce_mean_pct / pose.measured_n", independent=True)

    n_mz, good_mz = _count(mz.get("evaluable_reads")), _count(mz.get("exact_n"))
    mz_value = _summary_count(good_mz, n_mz, mz.get("exact_rate"))
    add("mz_exact", "MZ", "Pełny poprawny numer", mz_value, good_mz, n_mz,
        unit="zweryfikowane odczyty MZ z GT",
        reason=("Wynik pojedynczych wycięć tablic, nie całych obserwacji."
                if mz_value != "—" else "Brak porównań odczytu MZ z numerem GT."),
        source_path="mz.exact_n / mz.evaluable_reads")
    gt_chars = _count(mz.get("gt_characters"))
    errors = _count(mz.get("error_characters"))
    cer = _number(mz.get("cer"))
    add("mz_cer", "MZ", "Błąd znaków (CER)",
        _pct(cer, fraction=True) if gt_chars and cer is not None else "—",
        errors, gt_chars, unit="znaki GT / błędy edycji",
        reason=("Im niżej, tym lepiej; wstawienia mogą dać CER powyżej 100%."
                if gt_chars and cer is not None else "Brak porównywalnych znaków GT i odczytów MZ."),
        source_path="mz.cer / mz.gt_characters")

    n_alpr, good_alpr = _count(alpr.get("evaluable_subjects")), _count(alpr.get("exact_n"))
    alpr_value = _summary_count(good_alpr, n_alpr, alpr.get("exact_rate"))
    add("alpr_exact", "ALPR", "Poprawny numer końcowy", alpr_value, good_alpr, n_alpr,
        unit="kwalifikujące się obserwacje z GT",
        reason=("Inny mianownik niż MZ; jeden wynik na obserwację."
                if alpr_value != "—" else "Brak zakończonych par wynik ALPR–numer GT."),
        source_path="alpr.exact_n / alpr.evaluable_subjects")

    if (_count(multi.get("total_multi_invocations")) or 0) > 0:
        formally_measured = str(multi.get("formal_status") or "") == "MEASURED"
        add("multi_formal", "MT", "Formalne TP / FP / FN (wiele tablic)", "—", None,
            _count(multi.get("total_multi_invocations")), unit="wywołania wielotablicowe",
            reason=("Kanoniczne wartości formalnych TP/FP/FN nie są w tym podsumowaniu."
                    if formally_measured else
                    "Wymagane niezależne GT, zweryfikowany zakres ROI i protokół dopasowania IoU."),
            source_path="multi_plate.formal_status", formal_unavailable=not formally_measured)
        diag = _dict(multi.get("operator_diagnostic"))
        annotated = _count(diag.get("plate_count"))
        linked = _count(diag.get("linked_predictions"))
        add("multi_links", "MT", "Ręczne powiązania (diagnostyka)",
            str(linked) if linked is not None else "—", linked, annotated,
            unit="ręczne powiązania MT–GT",
            reason="Decyzje operatora, nie formalne trafienia detektora.",
            source_path="multi_plate.operator_diagnostic", diagnostic=True)

    by_id = {row.metric_id: row for row in rows}
    card_specs = (
        ("mt_localization", "MT · Wykrywanie tablic", "Brak ocenionych wywołań MT"),
        ("pose_iou", "Pose · Położenie obrysu", "Brak niezależnego GT obrysu"),
        ("mz_exact", "MZ · Odczyt znaków", "Brak zweryfikowanych numerów GT"),
        ("alpr_exact", "ALPR · Cały system", "Brak par wynik–GT"),
    )
    cards = []
    for key, heading, empty_reason in card_specs:
        row = by_id[key]
        missing = "Brak porównania" if key == "pose_iou" and source_verified else "Brak wyniku"
        cards.append({"id": key, "heading": heading,
                      "value": row.formatted_value if row.formatted_value != "—" else missing,
                      "reason": row.reason if row.formatted_value != "—" and row.status not in {"SOURCE_UNVERIFIED"}
                      else empty_reason if source_verified else "Archiwum wymaga weryfikacji",
                      "status": row.status})

    active = _count(progress.get("mt_executed_active"))
    rated = _count(progress.get("mt_invocations_cardinality_reviewed"))
    total = _count(form.get("total"))
    verified = _count(form.get("verified"))
    def progress_item(label: str, done: int | None, count: int | None, *, kind: str):
        valid = source_verified and count is not None and done is not None and done <= count
        return {"label": label, "done": done if valid else None,
                "total": count if valid else None,
                "text": (f"{done} z {count}" if count else "Brak wywołań / obserwacji") if valid else "Brak danych",
                "kind": kind}
    checks = [progress_item("Liczba tablic sprawdzona w MT", rated, active, kind="mt"),
              progress_item("Obserwacje z kompletnymi decyzjami", verified, total, kind="form")]

    labels = {
        "NOT_STARTED": "Weryfikacja nierozpoczęta",
        "IN_PROGRESS": "Weryfikacja w toku",
        "READY_TO_APPROVE": "Formularz kompletny — do zatwierdzenia",
        "READY_FOR_EVALUATION": "Formularz zatwierdzony",
        "APPROVED_PENDING_SNAPSHOT": "Decyzje zatwierdzone — pakiet do kontroli",
    }
    form_label = labels.get(status, "Stan weryfikacji nieustalony")
    assessment = _dict(report.get('canvas_assessment')) if f5a else {}
    if assessment:
        q, count = _count(assessment.get('n_qualified')), _count(assessment.get('n_total'))
        if q is not None and count is not None:
            form_label = f"Zapisane oceny MT/ROI: {q}/{count} · {assessment.get('scope', 'N/A')}"
            checks = [progress_item('Ocenione wywołania MT/ROI', q, count, kind='mt'),
                      dict(label='Pełne sceny TV', done=None, total=None,
                           text='N/A — brak potwierdzonego pokrycia scen', kind='form')]
    science_label = ("Ocena naukowa nieukończona" if formal_multi_blocked else
                     "Raport wstępny" if preliminary else "Dane ocenione w swoim zakresie")
    if not source_verified:
        science_label = "Niezweryfikowane źródło"
    caveats = [str(v).strip() for v in report.get("caveats", ()) if str(v).strip()] if isinstance(report.get("caveats"), (list, tuple)) else []
    methods = [str(v).strip() for v in report.get("methods", ()) if str(v).strip()] if isinstance(report.get("methods"), (list, tuple)) else []
    misses = [f"{row.label}: {row.reason}" for row in rows
              if row.status in {"MISSING", "NOT_EVALUABLE", "SOURCE_UNVERIFIED"}]
    if not misses:
        misses = ["Brak zgłoszonych braków mianowników w prezentowanych miarach."]
    why = list(dict.fromkeys(misses + [str(x) for x in form.get("issues", ()) if isinstance(x, str)]))
    latencies = []
    latency = _dict(performance.get("latency"))
    for key, label in (("mt", "MT · Inferencja modelu tablic"),
                       ("mz", "MZ · Inferencja modelu znaków"),
                       ("pipeline", "ALPR · Przetwarzanie potoku")):
        value = _dict(latency.get(key))
        if f5a:
            label = {'mt': 'MT · suma inferencji w śladzie',
                     'mz': 'MZ · suma inferencji w śladzie',
                     'pipeline': 'Potok · wszystkie zarejestrowane przebiegi'}[key]
        n = _count(value.get("n"))
        med, p95 = _number(value.get("p50_ms")), _number(value.get("p95_ms"))
        if not source_verified or not n or med is None or p95 is None:
            result = "Brak pomiaru"
        else:
            result = (f"p50 {med:.1f} ms · P95 {p95:.1f} ms · n={n}").replace(".", ",")
        latencies.append({"label": label, "value": result})

    source = [
        ("Sesja", str(report.get("session_id") or "Brak danych")),
        ("Scenariusz", str(condition.get("scenario") or "Brak danych")),
        ("Wariant F4 (rejestr SHA)", str(condition.get("variant_display") or "NIEPOTWIERDZONE")),
        ("provenance", str(condition.get("variant_provenance") or "NIEPOTWIERDZONE")),
        ("Wariant zadeklarowany przez Androida", str(condition.get("variant") or "Brak danych")),
        ("Telefon", str(condition.get("device") or "Brak danych")),
        ("Tryb GT", "Asystowany — predykcje były widoczne" if assisted else
         "Zaślepiony" if mode == "blinded_gt_v1" else "Nieustalony"),
        ("Rewizja weryfikacji", str(report.get("review_revision") if report.get("review_revision") is not None else "—")),
        ("Archiwum SHA-256", str(report.get("source_archive_sha256") or "Brak danych")),
    ]
    variant_title = 'Wariant F4 (rejestr SHA)'
    if f5a:
        variant_title = 'Wariant F5A (Android)'
        source[2:5] = [
            (variant_title, str(condition.get('variant_display') or 'NIEPOTWIERDZONE')),
            ('Seria', str(condition.get('series_id') or 'NIEPOTWIERDZONE')),
            ('Pochodzenie', str(condition.get('identity_status') or 'NIEPOTWIERDZONE')),
        ]
    def condition_value(key, title):
        value = str(condition.get(key) or "").strip()
        if key=='variant_provenance':return 'provenance='+str(value or 'NIEPOTWIERDZONE')
        if key=='variant_display':return title+': '+str(value or 'NIEPOTWIERDZONE')
        return f"{title}: nie podano w raporcie" if value.lower() in ("", "brak danych", "—") else value

    return {
        "title": "Podsumowanie wyników sesji",
        "subtitle": "Wyniki, stan weryfikacji i ograniczenia pomiaru",
        "form_status": form_label,
        "science_status": science_label,
        "review_mode": "GT asystowany" if assisted else "GT zaślepiony" if mode == "blinded_gt_v1" else "GT: tryb nieustalony",
        "conditions": " · ".join(condition_value(key, title) for key, title in
                                   (("scenario", "Scenariusz"), ("variant_display", variant_title),
                                     ("variant_provenance", "provenance"), ("variant", "Wariant zadeklarowany przez Androida"), ("device", "Telefon")))
                      + (f" · Seria: {condition.get('series_id') or 'NIEPOTWIERDZONE'}" if f5a else ''),
        "cards": cards,
        "rows": tuple(rows),
        "progress": checks,
        "pose_pair_count": n_pose if source_verified else None,
        "mz_reviewed_count": n_mz if source_verified else None,
        "latencies": latencies,
        "why": why,
        "caveats": caveats,
        "methods": methods,
        "source": source,
        "source_verified": source_verified,
        "formal_multi_blocked": formal_multi_blocked,
    }
