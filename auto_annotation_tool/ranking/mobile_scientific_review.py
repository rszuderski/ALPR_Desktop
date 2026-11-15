"""Scientific reading layer over verified mobile ALPR review measurements.

No modifications to source sessions, annotations, GT or model outputs. This module
never imputes unavailable measurements or treats MT invocation yield as plate recall.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from statistics import median

from .mobile_verification_policy import subject_verification_issues

SCIENCE_SCHEMA = "alpr.mobile.scientific_summary.v1"


def _numeric(value):
    try:
        if value is None or isinstance(value, bool):
            return None
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _integer(value):
    n = _numeric(value)
    return int(n) if n is not None and n >= 0 and n.is_integer() else None


def _fraction(numerator, denominator):
    n, d = _integer(numerator), _integer(denominator)
    return n / d if n is not None and d else None


def _rate_text(n, d):
    n, d = _integer(n), _integer(d)
    return "nieobliczalne (brak mianownika)" if not d or n is None else f"{n}/{d} ({100*n/d:.2f}%)"


def _mean(values):
    vals = [n for v in values if (n := _numeric(v)) is not None]
    return sum(vals) / len(vals) if vals else None


def _id_row(session, key):
    return session.review.subjects.get(key, {})



def subject_stage(session, subject_key, geometry_rows):
    """Independent MT, MZ and optional Pose-GT workflow states.

    Quantitative Pose metrics stay separate from mandatory human verification;
    absence of manual four-point GT does not undo a visual MT decision.
    """
    subject = session.subjects.get(subject_key, {})
    ids = [i for i in subject.get("attempts", ()) if i in session.attempts]
    groups = {session.invocation_by_attempt[i] for i in ids if i in session.invocation_by_attempt}
    active = [gid for gid in groups if session.mt_invocations[gid].executed and
              not session.mt_invocations[gid].cancelled and
              not session.mt_invocations[gid].execution_failed and
              session.invocation_annotation(gid).get("evaluable") is not False and
              session.invocation_annotation(gid).get("visible_plate_count") != "uncertain"]
    all_issues = subject_verification_issues(session, subject_key)
    mt_issues = [reason for reason in all_issues if not (reason.startswith("Brak GT numeru")
                 or reason.startswith("Brak uzasadnienia wyłączenia GT")
                 or reason.startswith("GT numeru wymaga")
                 or "brak niezależnego GT numeru lub przyczyny wykluczenia OCR" in reason
                 or "nie można jednocześnie wykluczyć i oceniać tego samego numeru" in reason)]
    mt_complete = bool(active) and not mt_issues
    if not groups or not active:
        mt_status = "Poza weryfikacją MT"
    elif mt_issues:
        mt_status = "MT: weryfikacja w toku"
    else:
        mt_status = "MT zweryfikowane"

    mz_runs = []
    for sid in subject.get("samples", ()):
        sample = session.samples.get(sid, {})
        attempt = session.attempts.get(sample.get("attempt_id"), {})
        if session.cancelled(sample):
            continue
        state = str(attempt.get("mz_status") or sample.get("mz_status") or "").upper()
        if state not in {"", "NOT_RUN", "UNKNOWN"}:
            mz_runs.append(sample)
    gt = _id_row(session, subject_key)
    from .mobile_multi_plate_verification import is_multi,group_issues
    exclusively_multi = bool(groups) and all(is_multi(session,gid) for gid in groups)
    if exclusively_multi and mz_runs:
        pending = any(group_issues(session,gid) for gid in groups)
        mz_status, mz_pending = ("MZ: weryfikacja w toku",True) if pending else ("MZ zweryfikowane",False)
    elif not mz_runs:
        mz_status, mz_pending = "MZ nie uruchomiono", False
    elif gt.get("evaluable") is False:
        mz_status, mz_pending = "MZ wyłączone z weryfikacji", not bool(str(gt.get("note") or "").strip())
    elif not str(gt.get("ground_truth") or "").strip() or gt.get("evaluable") is not True:
        mz_status, mz_pending = "MZ: brak numeru GT", True
    else:
        mz_status, mz_pending = "MZ zweryfikowane", False
    if mt_complete and mz_status == "MZ zweryfikowane":
        label = "MT i MZ zweryfikowane"
    elif mt_complete:
        label = "MT zweryfikowane" if not mz_pending else "MT zweryfikowane · MZ w toku"
    elif mt_status == "Poza weryfikacją MT":
        label = mt_status
    else:
        label = mt_status
    return {"label": label, "mt_status": mt_status, "mt_complete": mt_complete,
            "mt_pending": bool(mt_issues), "mt_excluded_groups": len(groups) - len(active),
            "mz_status": mz_status, "mz_pending": mz_pending,
            "pose_quantitative_optional": True}


def make_scientific_summary(session, statistics, pose_rows):
    """Grounded ratios only; preserve stage denominators, outcome provenance."""
    summary = statistics["summary"]
    pose_map = {row["attempt_id"]: row for row in pose_rows}
    subjects = {key: subject_stage(session, key, pose_map) for key in session.subjects}
    mt_rows = statistics.get("mt_invocations", ())
    active = [r for r in mt_rows if r.get("executed") and not r.get("cancelled")
              and not r.get("execution_failed")]
    rated = [r for r in active if session.invocation_annotation(r["mt_invocation_id"]).get(
                 "visible_plate_count") in {"none", "one", "multiple", "uncertain"}]
    sample_one = [r for r in rated if r.get("included")]
    n_singles = _integer(summary.get("evaluable_mt_invocations"))
    n_success = _integer(summary.get("mt_successful_invocations"))
    n_missed = _integer(summary.get("mt_no_detection_invocations"))
    all_pose_measured = [r for r in pose_rows if r.get("metric_status") == "measured"
                         and _numeric(r.get("nmce_pct")) is not None
                         and _numeric(r.get("polygon_iou_pct")) is not None]
    # Never mix independent, blinded GT with polygons drawn while looking
    # at MT predictions or with historical GT whose provenance is unknown.
    pose_measured = [r for r in all_pose_measured if str(r.get("pose_gt_review_mode") or "")
                     in {"independent_manual_v1", "blinded_manual_v1", "manual_blinded_v1"}]
    pose_older_assisted = [r for r in all_pose_measured if str(r.get("pose_gt_review_mode") or "")
                           not in {"independent_manual_v1", "blinded_manual_v1", "manual_blinded_v1"}]
    missed_gt = [r for r in pose_rows if r.get("metric_status") == "missed_detection_gt_saved"]
    eligible_pose = [r for r in pose_rows if r.get("eligible")]
    n_reads = _integer(summary.get("evaluable_reads"))
    n_exact = _integer(summary.get("exact_reads"))
    gt_char = _integer(summary.get("gt_characters"))
    errors = sum(_integer(summary.get(name)) or 0 for name in
                 ("incorrect_characters", "missing_characters", "extra_characters"))
    n_alpr = _integer(summary.get("evaluable_subjects"))
    n_alpr_exact = _integer(summary.get("subjects_with_exact_read"))
    done_mt = sum(1 for s in subjects.values() if s["mt_complete"])
    mz_not_run = sum(1 for s in subjects.values() if s["mz_status"] == "MZ nie uruchomiono")
    full = str(session.review.review_status).upper() == "COMPLETED"
    complete_sample = bool(active) and all(s["mt_complete"] or s["mt_status"] == "Poza weryfikacją MT"
                                       for s in subjects.values())
    complete_for_mt = done_mt > 0 and complete_sample and len(rated) == len(active)
    mode = str(session.review.review_mode)
    caveats = []
    if len(rated) < len(active):
        caveats.append(f"Niezweryfikowane wywołania MT: {len(active)-len(rated)} z {len(active)} aktywnych.")
    if any(not s["mt_complete"] and s["mt_status"] != "Poza weryfikacją MT" for s in subjects.values()):
        caveats.append("Część GT i decyzji MT/Pose pozostaje nieukończona.")
    if mode == "assisted":
        caveats.append("Weryfikacja asystowana: operator widział predykcje; nie jest to ślepa anotacja GT.")
    if n_singles is not None and n_singles < 30:
        caveats.append("Mała liczba zweryfikowanych wywołań z jedną tablicą: wynik MT jest opisowy, bez uogólnienia.")
    if n_singles is None or not n_singles:
        caveats.append("Brak mianownika dla skuteczności lokalizacji MT.")
    if len(pose_measured) == 0:
        caveats.append("Brak niezależnych/parowanych GT Pose: główny IoU i NMCE są nieobliczalne.")
    if pose_older_assisted:
        caveats.append("Historyczny GT Pose był rysowany przy widocznej predykcji lub bez potwierdzonej proweniencji; raportujemy go oddzielnie.")
    if not n_reads:
        caveats.append("Brak zweryfikowanych odczytów MZ: nie podawaj Exact/CER jako 0%.")
    if n_alpr is None or not n_alpr:
        caveats.append("Brak pełnych par wynik ALPR–numer GT: nie obliczaj skuteczności całego ALPR.")
    if mz_not_run:
        caveats.append(f"W {mz_not_run} obserwacjach MZ nie został uruchomiony; nie jest to błąd OCR.")
    if not full:
        caveats.append("Weryfikacja formularza trwa; agregaty są wstępne i nie stanowią zatwierdzonego wyniku naukowego.")
    payload = getattr(session.bundle, "report_payload", {}) or {}
    if not isinstance(payload, dict):
        payload = {}
    latency = payload.get("latency") or {}
    if not isinstance(latency, dict):
        latency = {}
    native_latency = {}
    for key in ("mt", "mz", "pipeline"):
        block = latency.get(key) or {}
        if not isinstance(block, dict):
            block = {}
        count = _integer(block.get("count"))
        native_latency[key] = {
            "n": count,
            "p50_ms": _numeric(block.get("median_ms", block.get("p50_ms"))) if count else None,
            "p95_ms": _numeric(block.get("p95_ms")) if count else None,
        }
    frame_flow = payload.get("frame_flow") or {}
    if not isinstance(frame_flow, dict):
        frame_flow = {}
    frames = {"received": _integer(frame_flow.get("frames_received")),
              "processed": _integer(frame_flow.get("frames_processed")),
              "skipped_frame_gate": _integer(frame_flow.get("frames_skipped_frame_gate",
                                                        frame_flow.get("frames_skipped_gate"))),
              "estimated_upstream_gaps": _integer(frame_flow.get("estimated_upstream_gaps"))}
    build_info = (session.bundle.experiment_session or {}).get("app_build") or {}
    if build_info.get("git_dirty") is True:
        caveats.append("APK zbudowano z lokalnymi zmianami (git_dirty=true); zachowaj kod/diff dla odtwarzalności.")
    from .mobile_multi_plate_verification import multi_scientific_summary
    multi_results = multi_scientific_summary(session)
    from .mobile_pose_correction import correction_report
    assisted = correction_report(session)
    if assisted["assisted_correction_n"]:
        caveats.append("Pomiary odchyleń od ręcznych korekt Pose są asystowane predykcją; nie są niezależnym GT.")
    if assisted["assisted_correction_invalid_n"]:
        caveats.append("Niektóre zapisane korekty Pose mają niespójną proweniencję i nie są liczone.")
    if multi_results["total_multi_invocations"]:
        caveats.append("Sceny wielotablicowe są liczone osobno per tablica GT; starsze liczniki MT/ALPR z jedną tablicą nie są ich sumą.")
        caveats.append("Weryfikacja wielu tablic: ukończone przypisania operatora nie są formalnymi TP/FP/FN. "
                       "Brak niezależnego pełnoklatkowego GT, udowodnionego pokrycia ROI i zamrożonego dopasowania IoU.")
        if multi_results["verified_multi_invocations"] < multi_results["total_multi_invocations"]:
            caveats.append("Nieukończona weryfikacja części scen wielotablicowych: nie obliczamy dla nich formalnych TP/FP/FN.")
    result = {
        "multi_plate": multi_results,
        "schema": SCIENCE_SCHEMA,
        "session_id": session.session_id,
        "source_archive_sha256": session.review.source_archive_sha256,
        "review_revision": session.review.review_revision,
        "review_status": session.review.review_status,
        "review_mode": mode,
        "conditions": session_conditions(session),
        "scientific_state": ("FORM_COMPLETE_SCIENCE_PENDING" if full and complete_for_mt
                             and multi_results["total_multi_invocations"] and
                             multi_results["formal_status"] != "MEASURED" else
                             "FINAL_FULL_REVIEW" if full and complete_for_mt else
                             "MT_STAGE_COMPLETE_REVIEW_OPEN" if complete_for_mt else "PRELIMINARY"),
        "subjects": subjects,
        "progress": {"subjects_total": len(session.subjects), "subjects_mt_done": done_mt,
                     "mt_executed_active": len(active), "mt_invocations_cardinality_reviewed": len(rated),
                     "mz_not_run_subjects": mz_not_run},
        "mt": {"unit": "jedno wykonane wywołanie MT z dokładnie jedną tablicą GT",
               "single_plate_n": n_singles, "localized_n": n_success,
               "failed_localization_n": (n_singles - n_success if n_singles is not None
                                         and n_success is not None and 0 <= n_success <= n_singles else None),
               "no_detection_n": n_missed, "success_rate": _fraction(n_success, n_singles),
               "false_detections_confirmed": _integer(summary.get("mt_false_detections")),
               "multi_plate_invocations_excluded": _integer(summary.get("mt_multi_plate_invocations")),
               "execution_error_invocations": _integer(summary.get("mt_execution_error_invocations"))},
        "pose": {"unit": "sparowana detekcja MT z ręcznym GT na tym samym wejściu MT",
                 "eligible_n": len(eligible_pose), "measured_n": len(pose_measured),
                 "gt_missed_plate_n": len(missed_gt),
                 "polygon_iou_mean_pct": _mean(r["polygon_iou_pct"] for r in pose_measured),
                 "nmce_mean_pct": _mean(r["nmce_pct"] for r in pose_measured),
                 "polygon_iou_median_pct": median(float(r["polygon_iou_pct"]) for r in pose_measured) if pose_measured else None,
                 "nmce_median_pct": median(float(r["nmce_pct"]) for r in pose_measured) if pose_measured else None,
                 "manual_assisted_gt_n": len(pose_older_assisted),
                 "manual_assisted_iou_mean_pct": _mean(r["polygon_iou_pct"] for r in pose_older_assisted),
                 "manual_assisted_nmce_mean_pct": _mean(r["nmce_pct"] for r in pose_older_assisted),
                 "independent_gt_n": len(pose_measured),
                 "assisted_corrections": assisted},
        "mz": {"unit": "zweryfikowany obraz tablicy z GT i uruchomionym MZ",
               "evaluable_reads": n_reads, "exact_n": n_exact,
               "exact_rate": _fraction(n_exact, n_reads),
               "gt_characters": gt_char, "error_characters": errors,
               "cer": _fraction(errors, gt_char)},
        "alpr": {"unit": "kwalifikująca się obserwacja z potwierdzonym numerem GT",
                 "evaluable_subjects": n_alpr, "exact_n": n_alpr_exact,
                 "exact_rate": _fraction(n_alpr_exact, n_alpr)},
        "performance": {"median_time_to_first_exact_ms": _numeric(summary.get("median_time_to_first_exact_ms")),
                        "p90_time_to_first_exact_ms": _numeric(summary.get("p90_time_to_first_exact_ms")),
                        "latency": native_latency, "frame_flow": frames,
                        "note": "Latency MT/MZ/pipeline to natywne miary czasu z telefonu. Czas do trafnego odczytu to osobna miara. Pipeline ≠ kamera→ekran."},
        "caveats": caveats,
        "methods": [
            "Lokalizacja MT = poprawne wywołania / zweryfikowane wywołania z dokładnie 1 tablicą; nie jest to pełny recall tablic ani precision.",
            "Nieudane lokalizacje to mianownik 1-tablicowych wywołań minus prawidłowo zlokalizowane; samo NO_DETECTION jest węższą kategorią.",
            "Fałszywe detekcje MT: tylko jawnie odrzucone przez operatora; bez kompletnego mianownika nie podajemy precision.",
            "Pose IoU = pole przecięcia polygonów / pole ich sumy; średnia po sparowanych GT.",
            "NMCE = średnia odległość odpowiadających rogów / przekątna obwiedni GT × 100%; niżej = lepiej.",
            "Pominięta przez MT tablica z własnym GT należy do weryfikacji wykrywania, nigdy do mianownika IoU/NMCE.",
            "Exact = identyczny odczyt i numer GT po regule normalizacji zapisanej w review; CER = S+D+I / znaki GT.",
            "MZ NOT_RUN nie jest mylone z błędnym odczytem ani z wartością CER=0.",
            "Wszystkie liczebności i mianowniki są z zatwierdzonego indeksu sesji i bieżącego sidecara; jednostka wywołanie ≠ klatka ≠ crop ≠ obserwacja.",
            "Powtarzające się klatki jednej sceny nie są niezależnymi próbkami; wywołań nie należy mylić z liczbą scen.",
            "MT lokalizacja: decyzja operatora i status VALID_QUAD, nie próg Polygon IoU; raport Pose podaje geometrię osobno.",
            "Wynik ALPR na obserwację może uwzględniać zgromadzony konsensus; odczyt pojedynczego cropa jest odrębnie rozliczany.",
            "Status etapu MT nie zmienia natywnego review_status ani warunków pełnego zakończenia sesji.",
        ],
    }
    if result['conditions'].get('experiment_family') == 'F5A':
        mt_scope = statistics.get('mt_object_evaluation_v2') or {}
        qualified, total = mt_scope.get('n_qualified', 0), mt_scope.get('n_total', 0)
        result['canvas_assessment'] = dict(n_qualified=qualified, n_total=total,
            n_unscored=max(0, total-qualified),
            scope='ASSISTED_SUBSET' if 0 < qualified < total else 'ASSISTED' if qualified else 'N/A',
            unit='MT_INVOCATION_ROI', tv_scene_coverage=None)
    from .mobile_mz_input_evidence import is_diagnostic_purpose
    if is_diagnostic_purpose(result['conditions'].get('measurement_purpose')):
        result['measurement_purpose'] = result['conditions']['measurement_purpose']
        result['scientific_state'] = result['conditions']['measurement_purpose']
    return result


def session_conditions(session):
    """Short user-facing configuration; retain full raw provenance separately."""
    bundle = session.bundle
    from .mobile_experiment_identity import from_bundle
    identity = from_bundle(bundle)
    from .mobile_tv_bbox_evidence import decorate_conditions
    if identity.get('family') == 'DIAGNOSTIC':
        result = _f5a_session_conditions(session, identity)
        result.update(experiment_family=identity.get('measurement_purpose'),
                      measurement_purpose=identity.get('measurement_purpose'))
        return decorate_conditions(session, result)
    if identity.get('family') == 'F5A':
        return decorate_conditions(session, _f5a_session_conditions(session, identity))
    experiment = (bundle.experiment_session or {}) if isinstance(bundle.experiment_session, dict) else {}
    execution = experiment.get("execution") or {}
    stages = execution.get("stages") or {}
    refs = bundle.model_refs or {}
    device = getattr(bundle.report, "device", {}) or {}
    if not isinstance(device, dict):
        device = {"name": str(device)}
    model = {}
    for key, role in (("MT", "plate"), ("MZ", "character")):
        info = refs.get(role) or {}
        stage = stages.get("mt" if role == "plate" else "mz") or {}
        model[key] = {
            "name": info.get("model_display_id") or info.get("model_id") or "brak danych",
            "variant": info.get("variant_id") or stage.get("variant_id") or "brak danych",
            "precision": info.get("precision") or stage.get("precision") or "brak danych",
            "runtime": info.get("runtime") or stage.get("runtime") or "brak danych",
            "delegate": stage.get("delegate") or "brak danych",
            "model_sha256": info.get("package_sha256") or "brak danych",
        }
    from .mobile_f4_variant import for_session
    f4_variant=for_session(session)
    return decorate_conditions(session, {
        "session_id": session.session_id,
        "scenario": experiment.get("scenario_id") or "brak danych",
        "variant": experiment.get("variant") or "brak danych",
        "f4_variant": f4_variant,
        "variant_display": f4_variant['label'],
        "variant_provenance": f4_variant['provenance'],
        "device": device.get("name") or device.get("model") or "brak danych",
        "android": device.get("android_version") or "brak danych",
        "build": (experiment.get("app_build") or {}).get("build_type") or "brak danych",
        "models": model,
        "source_archive_sha256": session.review.source_archive_sha256,
        "review_revision": session.review.review_revision,
        "review_mode": session.review.review_mode,
        "review_status": session.review.review_status,
    })


def _f5a_session_conditions(session, identity):
    """Use recorded ROI metadata without consulting or modifying F4's registry."""
    bundle = session.bundle
    raw = bundle.report_payload or {}
    collection = bundle.collection_session or {}
    build = collection.get('app_build') or raw.get('app_build') or {}
    stages = (identity.get('execution') or {}).get('stages') or {}
    refs = bundle.model_refs or {}
    models = {}
    for key, role in (('MP', 'vehicle'), ('MT', 'plate'), ('MZ', 'character')):
        info, stage = refs.get(role) or {}, stages.get(key.lower()) or {}
        artifacts = info.get('variant_artifact_sha256') or []
        if not isinstance(artifacts, list):
            artifacts = []
        models[key] = dict(name=info.get('model_display_id') or info.get('model_id') or stage.get('model_id') or 'brak danych',
            variant=info.get('variant_id') or stage.get('variant_id') or 'brak danych',
            precision=info.get('precision') or stage.get('precision') or 'brak danych',
            runtime=info.get('runtime') or stage.get('runtime') or 'brak danych',
            delegate=stage.get('delegate') or 'brak danych', cpu_threads=stage.get('cpu_threads'),
            model_sha256=artifacts[0] if len(artifacts) == 1 else 'brak pojedynczego artefaktu',
            artifact_sha256=list(artifacts), package_sha256=info.get('package_sha256'))
    device = getattr(bundle.report, 'device', {}) or {}
    if not isinstance(device, dict):
        device = {'name': str(device)}
    return dict(session_id=session.session_id, experiment_family='F5A',
        experiment_type=identity.get('experiment_type'), identity_status=identity['status'],
        identity_issues=list(identity['issues']), series_id=identity.get('series_id'),
        is_pilot=identity['is_pilot'], scenario=identity.get('scenario_id') or 'brak danych',
        variant=identity.get('variant') or 'brak danych', variant_display=identity['variant_display'],
        variant_provenance=identity.get('provenance'), roi_budget_policy=identity.get('roi_budget_policy'),
        device=device.get('name') or device.get('model') or 'brak danych',
        android=device.get('android_version') or 'brak danych',
        build=build.get('build_type') or 'brak danych',
        apk_sha256=(build.get('apk_identity') or {}).get('base_apk_sha256') or build.get('apk_sha256'),
        models=models, source_archive_sha256=session.review.source_archive_sha256,
        review_revision=session.review.review_revision, review_mode=session.review.review_mode,
        review_status=session.review.review_status)


def session_conditions_text(info):
    from .mobile_f4_variant import text as variant_text
    f5a = info.get('experiment_family') == 'F5A'
    variant=(f"F5A · {info['variant_display']} · seria {info.get('series_id') or 'niepotwierdzona'}\n"
             f"Scenariusz: {info['scenario']} · tożsamość: {info.get('identity_status')}" if f5a else
             variant_text(info['f4_variant']) if info.get('f4_variant') else
             f"Scenariusz: {info['scenario']}  •  Wariant zadeklarowany przez Androida: {info['variant']}")
    from .mobile_mz_input_evidence import is_diagnostic_purpose
    if is_diagnostic_purpose(info.get('measurement_purpose')):
        variant = (f"{info['measurement_purpose']} · poza oficjalnymi kohortami F4/F5A\n"
                   f"Seria: {info.get('series_id')} · scenariusz: {info['scenario']}")
    mp=(f"MP: {info['models']['MP']['name']} · {info['models']['MP']['precision']} · {info['models']['MP']['delegate']}\n" if f5a else '')
    input_note = ''
    if info.get('input_mode') == 'TV_BBOX_CROP':
        ctx = info['input_context']
        input_note = (f"TV_BBOX_CROP · {ctx['status']} · konfiguracja {ctx.get('variant')} · H_TV=NOT_APPLIED\n"
                      f"Pełna kamera dla GT · BBOX {ctx['tv_bbox_px']} · offset {ctx['offset_to_camera_full']}\n")
    return (variant+'\n'+input_note +
            f"Telefon: {info['device']}  •  Android: {info['android']}  •  APK: {info['build']}\n"
            + mp +
            f"MT: {info['models']['MT']['name']}  •  {info['models']['MT']['precision']}  •  {info['models']['MT']['delegate']}\n"
            f"MZ: {info['models']['MZ']['name']}  •  {info['models']['MZ']['precision']}  •  {info['models']['MZ']['delegate']}\n"
            f"Archiwum SHA-256: {info['source_archive_sha256']}\n"
            f"Weryfikacja: rewizja {info['review_revision']}; status {info.get('review_status','nieustalony')}; tryb {info['review_mode']}\n"
            "Pełne SHA modeli, ścieżki, historia treningu i pola techniczne znajdują się poniżej.")


def display_scientific_summary(report):
    """Short, human-readable paragraphs for a scrollable GUI read-only panel."""
    pr, mt, pose, mz, alpr = (report[k] for k in ("progress", "mt", "pose", "mz", "alpr"))
    mode = "asystowana (predykcje widoczne)" if report["review_mode"] == "assisted" else "zaślepiona"
    lines = [
        ("title", "WYNIKI NAUKOWE SESJI"),
        ("warning" if report["scientific_state"] != "FINAL_FULL_REVIEW" else "subtitle",
         "RAPORT WSTĘPNY • nie cytuj jako wyniku końcowego" if report["scientific_state"] != "FINAL_FULL_REVIEW"
         else "Pełna weryfikacja zakończona"),
        ("caption", f"Wywołania MT zweryfikowane co do liczby tablic: {pr['mt_invocations_cardinality_reviewed']}/{pr['mt_executed_active']}  •  "
         f"obserwacje ze zweryfikowanym MT: {pr['subjects_mt_done']}/{pr['subjects_total']}  •  weryfikacja {mode}"),
        ("heading", "1   Wykrywanie tablic — MT"),
        ("metric", f"Trafienia MT potwierdzone przez operatora: {_rate_text(mt['localized_n'], mt['single_plate_n'])}"),
        ("normal", f"Nieudane lokalizacje: {mt['failed_localization_n'] if mt['failed_localization_n'] is not None else '—'}  •  "
         f"bez jakiejkolwiek detekcji: {mt['no_detection_n'] if mt['no_detection_n'] is not None else '—'}"),
        ("normal", f"Fałszywe wskazania potwierdzone: {mt['false_detections_confirmed'] if mt['false_detections_confirmed'] is not None else '—'}"),
        ("caption", "Jednostka: wykonane wywołanie MT z jedną widoczną tablicą. To nie jest pełne recall/precision."),
        ("heading", "2   Położenie narożników — Pose"),
        ("metric", f"Polygon IoU średnio: {pose['polygon_iou_mean_pct']:.2f}%" if pose['polygon_iou_mean_pct'] is not None
         else "Polygon IoU: — (brak porównywalnych par)"),
        ("metric", f"NMCE średnio: {pose['nmce_mean_pct']:.2f}%" if pose['nmce_mean_pct'] is not None
         else "NMCE: — (brak porównywalnych par)"),
        ("normal", f"Medianowe Polygon IoU: {pose['polygon_iou_median_pct']:.2f}%  •  NMCE: {pose['nmce_median_pct']:.2f}%"
         if pose['measured_n'] else "Medianowe IoU/NMCE: —"),
        ("caption", f"Sparowane predykcje i GT: {pose['measured_n']}  •  "
         f"GT tablic pominiętych przez MT: {pose['gt_missed_plate_n']} (bez IoU/NMCE)."),
        ("caption", f"Korekty wspomagane predykcją MT: {pose['assisted_corrections']['assisted_correction_n']}; "
         f"historyczny ręczny GT asystowany/bez udokumentowanego trybu: {pose['manual_assisted_gt_n']}. "
         "Dwie osobne populacje — żadna nie jest niezależnym GT."),
        ("heading", "3   Rozpoznawanie znaków — MZ"),
        ("metric", f"Dokładne numery: {_rate_text(mz['exact_n'], mz['evaluable_reads'])}"),
        ("metric", f"CER: {mz['cer']*100:.2f}% ({mz['error_characters']}/{mz['gt_characters']} znaków)"
         if mz['cer'] is not None else "CER: — (brak GT i zweryfikowanego odczytu)"),
        ("caption", "Próby bez uruchomionego MZ nie są błędnymi odczytami."),
        ("heading", "4   Cały proces ALPR"),
        ("metric", f"Poprawnie odczytane tablice GT: {_rate_text(alpr['exact_n'], alpr['evaluable_subjects'])}"),
        ("heading", "5   Wydajność telefonu — pomiary niezależne od GT"),
        ("normal", "  •  ".join(
            (f"{name.upper()}: mediana {report['performance']['latency'][name]['p50_ms']:.1f} ms, "
             f"P95 {report['performance']['latency'][name]['p95_ms']:.1f} ms "
             f"(n={report['performance']['latency'][name]['n']})"
             if report['performance']['latency'][name]['p50_ms'] is not None and
                report['performance']['latency'][name]['p95_ms'] is not None
             else f"{name.upper()}: —") for name in ("mt", "mz", "pipeline"))),
        ("normal", f"Klatki: otrzymano {report['performance']['frame_flow']['received']}  •  "
         f"przetworzono {report['performance']['frame_flow']['processed']}  •  "
         f"pominięto przez frame gate {report['performance']['frame_flow']['skipped_frame_gate']}"
         if report['performance']['frame_flow']['received'] is not None
         else "Przepływ klatek: brak liczników w raporcie sesji"),
        ("normal", f"Mediana do pierwszego trafnego odczytu: {report['performance']['median_time_to_first_exact_ms']:.1f} ms"
         if report["performance"]["median_time_to_first_exact_ms"] is not None else "Czas do pierwszego poprawnego odczytu: —"),
        ("caption", "Pipeline = metryka etapu pipeline w raporcie telefonu, nie całkowity czas kamera→ekran. "
         "P95 wyliczony na próbkach zapisanych przez Androida."),
        ("heading", "OGRANICZENIA I WARUNKI ODCZYTU"),
    ]
    multi = report.get("multi_plate") or {}
    if multi.get("total_multi_invocations"):
        lines.extend([
            ("heading", "6   Weryfikacja scen z wieloma tablicami"),
            ("metric", f"Przypisania operatora: {multi['verified_multi_invocations']}/{multi['total_multi_invocations']} wywołań"),
            ("warning", "Formalne TP / FP / FN: NIEOBLICZALNE — brak wymaganego dowodu i niezależnego GT."),
            ("normal", f"Diagnostyka (asystowana): {multi['operator_diagnostic']['plate_count']} oznaczonych tablic; "
                       f"{multi['operator_diagnostic']['linked_predictions']} powiązanych wskazań MT; "
                       f"{multi['operator_diagnostic']['unmatched_plate_annotations']} tablic bez powiązania."),
            ("caption", "To decyzje operatora, nie modelowe TP/FP/FN. MZ poniżej również jest weryfikacją asystowaną."),
        ])
        if multi.get("text_gt_reads"):
            lines.append(("normal", f"MZ na sparowanych GT: {multi['text_exact']}/{multi['text_gt_reads']} poprawnych numerów; "
                          f"CER: {100*multi['text_errors']/multi['text_gt_characters']:.2f}%" if multi.get('text_gt_characters') else
                          "MZ — brak mianownika dla CER"))
    lines.extend(("warning", "• " + v) for v in report["caveats"])
    lines.extend([("caption", "Pełna proweniencja, surowe dowody i eksport metryk pozostają dostępne."),
                  ("caption", "W raporcie do pracy: licznik, mianownik, jednostka, przypadki wyłączone i tryb GT.")])
    return lines


def scientific_markdown(report):
    """Methodological companion; readable as-is and keeps an audit trail."""
    p, mt, pose, mz, alpr = (report[k] for k in ("progress", "mt", "pose", "mz", "alpr"))
    f = lambda n: "—" if n is None else f"{n:.2f}%"
    lines = [
        "# Analiza naukowa sesji ALPR — raport roboczy" if report["scientific_state"] != "FINAL_FULL_REVIEW" else
        "# Analiza naukowa sesji ALPR",
        "", f"- Session ID: `{report['session_id']}`",
        f"- SHA-256 źródłowego archiwum: `{report['source_archive_sha256']}`",
        f"- Rewizja weryfikacji: `{report['review_revision']}`",
        f"- Tryb GT: `{report['review_mode']}` (asystowana ≠ zaślepiona)",
        f"- Status naukowy: `{report['scientific_state']}`; stan pełnej weryfikacji: `{report['review_status']}`",
        f"- Scenariusz: {report['conditions']['scenario']}; wariant: {report['conditions']['variant']}",
        f"- Telefon: {report['conditions']['device']}; Android: {report['conditions']['android']}",
        f"- Model MT: {report['conditions']['models']['MT']['name']} ({report['conditions']['models']['MT']['precision']})",
        f"- Model MZ: {report['conditions']['models']['MZ']['name']} ({report['conditions']['models']['MZ']['precision']})",
        f"- Zweryfikowane wywołania MT: {p['mt_invocations_cardinality_reviewed']}/{p['mt_executed_active']}",
        f"- Obserwacje ze zweryfikowanym MT: {p['subjects_mt_done']}/{p['subjects_total']}",
        "", "## Wyniki z jawnymi mianownikami", "",
        "| Moduł | Miara | Wynik | Jednostka i zakres |", "|---|---|---|---|",
        f"| MT | Potwierdzone trafienia modelu | {_rate_text(mt['localized_n'], mt['single_plate_n'])} | Wywołania z jedną tablicą GT |",
        f"| MT | Nieudana lokalizacja | {mt['failed_localization_n']} | Wywołania z 1 tablicą bez poprawnej lokalizacji |",
        f"| MT | Brak jakiejkolwiek detekcji | {mt['no_detection_n']} | Podzbiór nieudanych lokalizacji |",
        f"| MT | Fałszywe detekcje | {mt['false_detections_confirmed']} | Potwierdzone przez operatora, nie pełne FP |",
        f"| Pose | Średnie Polygon IoU | {f(pose['polygon_iou_mean_pct'])} | N={pose['measured_n']} sparowanych GT |",
        f"| Pose | Średni NMCE | {f(pose['nmce_mean_pct'])} | N={pose['measured_n']} sparowanych GT |",
        f"| Pose | Mediana IoU | {f(pose['polygon_iou_median_pct'])} | N={pose['measured_n']} sparowanych GT |",
        f"| Pose | Mediana NMCE | {f(pose['nmce_median_pct'])} | N={pose['measured_n']} sparowanych GT |",
        f"| Pose | Pominięte tablice z GT | {pose['gt_missed_plate_n']} | Bez porównania IoU/NMCE |",
        f"| Pose (asystowane) | Ręczne GT przy widocznej predykcji lub bez udokumentowanej niezależności | {pose['manual_assisted_gt_n']} | NIE jest niezależnym GT |",
        f"| Pose (asystowane) | IoU z historycznym GT ręcznym | {f(pose['manual_assisted_iou_mean_pct'])} | N={pose['manual_assisted_gt_n']} osobno od niezależnego GT |",
        f"| Pose (asystowane) | NMCE z historycznym GT ręcznym | {f(pose['manual_assisted_nmce_mean_pct'])} | N={pose['manual_assisted_gt_n']} osobno od niezależnego GT |",
        f"| Pose (asystowane) | Ręczne korekty MT | {pose['assisted_corrections']['assisted_correction_n']} | Korekta po obejrzeniu predykcji, N osobne od GT niezależnego |",
        f"| Pose (asystowane) | Odchylenie od korekty, IoU | {f(pose['assisted_corrections']['assisted_delta_iou_mean_pct'])} | Nie jest niezależnym Polygon IoU GT |",
        f"| Pose (asystowane) | Odchylenie od korekty, NMCE | {f(pose['assisted_corrections']['assisted_delta_nmce_mean_pct'])} | Nie jest niezależnym NMCE GT |",
        f"| MZ | Pełny poprawny numer | {_rate_text(mz['exact_n'], mz['evaluable_reads'])} | Zweryfikowane odczyty MZ z GT |",
        f"| MZ | CER | {f(mz['cer']*100 if mz['cer'] is not None else None)} | Znaki GT: {mz['gt_characters']}; błędy S+D+I: {mz['error_characters']} |",
        f"| ALPR | Pełny poprawny odczyt | {_rate_text(alpr['exact_n'], alpr['evaluable_subjects'])} | Kwalifikujące się obserwacje z numerem GT |",
        "", "## Wydajność na urządzeniu (natywny raport telefonu)", "",
        "| Etap | n | Mediana [ms] | P95 [ms] |", "|---|---:|---:|---:|",
    ]
    for key in ("mt", "mz", "pipeline"):
        item = report["performance"]["latency"][key]
        shown = lambda v: f"{v:.2f}" if v is not None else "—"
        lines.append(f"| {key.upper()} | {item['n']} | {shown(item['p50_ms'])} | {shown(item['p95_ms'])} |")
    frames = report["performance"]["frame_flow"]
    lines.extend([
        "", f"- Klatki otrzymane: {frames['received']}; przetworzone: {frames['processed']}; "
            f"odrzucone przez frame gate: {frames['skipped_frame_gate']}.",
        "- Pipeline jest etapem raportowanym przez Androida, nie pomiarem całkowitego czasu kamera→ekran.",
        "", "## Interpretacja i definicje", "",
    ])
    lines.extend(f"- {method}" for method in report["methods"])
    lines.extend(["", "## Ograniczenia i warunki ważności", ""])
    lines.extend(f"- {issue}" for issue in report["caveats"])
    multi = report.get("multi_plate") or {}
    if multi.get("total_multi_invocations"):
        lines.extend(["", "## Sceny z wieloma tablicami — osobny protokół MT", "",
                      f"- Przypisania operatora: {multi['verified_multi_invocations']}/{multi['total_multi_invocations']} wywołań.",
                      "- Formalne TP / FP / FN: NIEOBLICZALNE (`NOT_EVALUABLE`), bez niezależnego GT i zamrożonej walidacji ROI/IoU.",
                      f"- Diagnostyka: {multi['operator_diagnostic']['plate_count']} tablic, "
                      f"{multi['operator_diagnostic']['linked_predictions']} powiązań MT, "
                      f"{multi['operator_diagnostic']['unmatched_plate_annotations']} bez powiązania MT.",
                      "- Te liczniki to ręczne przypisania operatora, nie pomiary trafień i pominięć detektora.",
                      f"- Odczyty MZ na sparowanym GT: {multi['text_exact']}/{multi['text_gt_reads']} poprawnych numerów." if multi['text_gt_reads'] else "- MZ: brak zakończonych porównań z GT."])
    lines.extend(["", "## Zasada analizy", "",
                  "Dokument jest wyłącznie pochodną zapisanych pomiarów i decyzji operatora. "
                  "Nie modyfikuje pliku `.alprsession`, sidecara GT, wartości predykcji ani danych modeli. "
                  "Pełne tabele szczegółowe i proweniencja są dostępne w równoległych plikach eksportu.", ""])
    return "\n".join(lines)


def write_scientific_companion(report, directory):
    """Atomic new companion files, without touching existing native exports."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    generated = []
    session_key = re.sub(r"[^A-Za-z0-9_-]+", "_", str(report["session_id"]))[:60] or "sesja"
    revision = int(report["review_revision"])
    basename = f"ocena_naukowa_{session_key}_rew{revision:04d}"
    for name, content in (
        (basename + ".md", scientific_markdown(report)),
        (basename + ".json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
    ):
        path = directory / name
        tmp = directory / (name + ".tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        generated.append(path)
    return generated
