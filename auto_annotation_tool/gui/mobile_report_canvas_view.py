"""Read-only bridge from the AKT89E export gate to the main results panel."""
from dataclasses import asdict, replace
from pathlib import Path

from .mobile_compare_review_gate import imported_verification, inspect_canvas, _live_review_state


def results_root_for_source(source):
    for parent in Path(str(source or "")).parents:
        if parent.name.lower() == "02_sesje":
            candidate = parent.parent / "03_wyniki"
            return candidate if candidate.is_dir() else None
    return None


def export_stamp(results_root, session_id):
    if results_root is None:
        return ()
    try:
        return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size)
                     for p in sorted((Path(results_root) / session_id).glob("*/export_manifest.json")))
    except OSError as error:
        return (type(error).__name__, str(error))


def selected_canvas_export(session_id, source_sha, results_root, review_root):
    """Same revision selection and fail-closed rules as the import checkboxes."""
    from .mobile_comparison_core import load_session_export
    gate = imported_verification(session_id, source_sha, results_root, review_root)
    if not gate.approved:
        return gate, None
    # Retain manifest verification when fetching the selected payload. Never
    # read a guessed export, or render a previous READY after a concurrent edit.
    try:
        item = load_session_export(gate.export_dir, verify=True)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return replace(gate, state="UNREADABLE" if isinstance(error, OSError) else "INVALID",
                       label="Brak odczytu" if isinstance(error, OSError) else "Wymaga kontroli",
                       reason=str(error)), None
    live = _live_review_state(review_root, source_sha, session_id)
    canvas_sid = (item.canvas or {}).get("session_id", session_id)
    if (item.session_id != session_id or item.source_sha256 != source_sha or
            item.review_revision != gate.revision or canvas_sid != session_id or
            (not item.canvas_ready if gate.ready else not item.operator_approved)
            or live not in (None, (item.review_revision, "COMPLETED"))):
        return replace(gate, state="STALE", label="Eksport nieaktualny",
                       reason="Tożsamość lub rewizja zmieniła się podczas odczytu."), None
    return gate, item


def load_canvas_snapshot(session_id, source_sha, results_root, review_root):
    gate, item = selected_canvas_export(session_id, source_sha, results_root, review_root)
    result = dict(verification=asdict(gate), session_id=session_id, source_sha256=source_sha,
                  summary=None, identity=None)
    if item is not None:
        result.update(summary=item.canvas, identity=item.model_identity,
                      manifest_sha256=item.manifest_sha256, review_revision=item.review_revision,mt_evaluation=item.mt_evaluation,
                      mt_object_evaluation=item.mt_object_evaluation)
    return result


def canvas_overview(snapshot, *, session_id="", source_sha="", progress=None, loading=False):
    snapshot = snapshot or {}
    gate = snapshot.get("verification") or {}
    state = gate.get("state", "UNKNOWN")
    c = snapshot.get("summary") or {}
    bound = (state == "READY" and snapshot.get("session_id") == session_id
             and snapshot.get("source_sha256") == source_sha
             and c.get("source_archive_sha256") == source_sha
             and c.get("review_revision") == gate.get("revision")
             and c.get("session_id", session_id) == session_id
             and inspect_canvas(c, "COMPLETED")[0])
    titles = ("MT — wykrycie tablic", "MT — dopasowanie obrysu",
              "MZ — pełne zgodne odczyty", "MZ — błędy znaków")
    cards = [dict(title=t, value="—", note="Brak zatwierdzonej oceny Canvas.") for t in titles]
    status = {"READY":"WYMAGA KONTROLI", "NONE":"Nie rozpoczęto",
        "IN_PROGRESS":"W TOKU", "STALE":"EKSPORT NIEAKTUALNY", "NEEDS_EXPORT":"EKSPORT NIEAKTUALNY",
        "UNREADABLE":"BRAK ODCZYTU", "INVALID":"WYMAGA KONTROLI", "INCOMPLETE":"W TOKU"}.get(state, "Nie potwierdzono stanu weryfikacji")
    counter, detail, ratio = "Nie ustalono liczby obrazów do oceny", gate.get("reason", ""), None
    p = progress or {}
    def pct(value):
        return f"{value:.2f}%".replace(".", ",") if isinstance(value, (int, float)) else "—"
    if bound:
        status = "ZATWIERDZONA"
        counter = f"Archiwalne oceny MT/ROI: {c['checked_scenes']}/{c['scene_count']} · rewizja {gate['revision']}"
        ratio = (c['checked_scenes'] + c['excluded_scenes'])/c['scene_count']
        detail = f"Ocena asystowana · poza oceną: {c['excluded_scenes']} · SHA {source_sha[:12]}…"
        cards[0].update(value=f"{c['linked_plates']}/{c['expected_plates']}",
                        note=f"Powiązane detekcje / tablice referencyjne · FN: {c['missed_plates']} · FP: {c['false_positive_mt']}")
        cards[1].update(value=f"IoU {pct(c.get('polygon_iou_mean_pct'))}",
                        note=f"NMCE {pct(c.get('nmce_mean_pct'))} · n={c['geometry_n']} par obrysów. FN poza geometrią.")
        cards[2].update(value=f"{c['exact_reads']}/{c['evaluated_reads']}",
                        note="Pełne zgodne numery / ocenione odczyty MZ. MZ NOT_RUN poza mianownikiem.")
        chars, errors = c['gt_characters'], c['text_errors']
        cards[3].update(value=pct(errors/chars*100) if chars else "Nie oceniano",
                        note=f"Micro-CER: {errors}/{chars} · błędy edycji / znaki GT.")
    elif (state=='APPROVED_LIMITED' and snapshot.get('session_id')==session_id
          and snapshot.get('source_sha256')==source_sha and c.get('source_archive_sha256')==source_sha
          and c.get('review_revision')==gate.get('revision') and c.get('session_id')==session_id):
        status='ZATWIERDZONA · OGRANICZENIA POMIARÓW'
        checked=(c.get('operator_progress') or {}).get('inspected',c.get('inspected_scenes',0))
        total=c.get('scene_count',0)
        counter, ratio = 'Nie ustalono liczby obrazów do oceny', None
        from .mobile_measurement_availability import limits_text
        detail='Weryfikacja zakończona. Nie każdy obraz ma dane do wszystkich pomiarów.\n'+limits_text(c,snapshot.get('mt_evaluation'))
        detail+=f'\nArchiwalny stan operacyjny MT/ROI: {checked}/{total} · rewizja {gate["revision"]}; nie jest to postęp ocen obrazów przez operatora.'
    elif loading:
        status = "Wczytywanie oceny Canvas…"
    objects=snapshot.get('mt_object_evaluation') or {}
    from ..ranking.mobile_mt_object_evaluation import POLICY_ID,POLICY_SHA256,format_rate
    object_bound=(state in ('READY','APPROVED_LIMITED') and objects.get('matching_policy_id')==POLICY_ID
        and objects.get('matching_policy_sha256')==POLICY_SHA256 and objects.get('session_id')==session_id
        and objects.get('source_archive_sha256')==source_sha and objects.get('review_revision')==gate.get('revision'))
    if object_bound:
        geo,ocr=objects['geometry_summary'],objects['ocr_summary']
        def number(key):return str(objects[key]) if objects.get(key) is not None else '—'
        cards[0].update(title='MT v2 — wykrycie tablic',value=number('tp')+'/'+number('n_gt'),
            note=f"TP/GT · FP {number('fp')} · FN {number('fn')} · {objects['n_qualified']}/{objects['n_total']} wywołań. "
                 f"P {format_rate(objects['precision'])} · R {format_rate(objects['recall'])} · F1 {format_rate(objects['f1'])}.")
        cards[1].update(value='IoU '+pct(geo['polygon_iou_mean_pct']),
            note=f"NMCE {pct(geo['nmce_mean_pct'])} · n={geo['n_geometry']}/{geo['n_pairs']} par v2; geometria pomocnicza.")
        cards[2].update(value=f"{ocr['n_exact']}/{ocr['n_reads']}" if ocr['n_reads'] else '—',
            note=f"Warunkowe MZ · {ocr['status']} · potwierdzony TP/crop/GTZ; FP i NOT_RUN poza mianownikiem.")
        cards[3].update(value=pct(100*ocr['micro_cer']) if ocr['micro_cer'] is not None else '—',
            note=f"Micro-CER {ocr['n_text_errors']}/{ocr['n_gt_characters']} znaków GTZ; również błędne i puste wykonane odczyty.")
        detail+='\nF4-EVAL-2.0 · ASSISTED · retrospektywnie 2026-11-01 · odrębne zakresy MT/geometrii/OCR.'
    elif bound or state=='APPROVED_LIMITED':
        cards[0]['title']='MT — archiwalna ocena Canvas'
        detail+='\nArchiwalny wynik Canvas/v1; brak eksportu F4-EVAL-2.0. Bieżące v2 jest dostępne w weryfikacji i analizie.'
    from ..ranking.mobile_operator_progress import SCHEMA, presentation
    if (p.get('schema') == SCHEMA and p.get('session_id') == session_id
            and p.get('source_sha256') == source_sha
            and not (p.get('sidecar_state') == 'ABSENT' and (bound or state == 'APPROVED_LIMITED'))):
        operator = presentation(p)
        status, counter, ratio = operator['status'], operator['progress'], operator['progress_ratio']
        detail = operator['progress_subtitle'] + ('\nOcena GT/MT/MZ: ' + detail if detail else '')
    elif p and not (bound or state == 'APPROVED_LIMITED'):
        counter, ratio = 'Nie ustalono liczby obrazów do oceny', None
    if bound or state == 'APPROVED_LIMITED':
        detail += '\nPochodzenie zatwierdzonej oceny: ' + str(gate.get('export_dir') or 'zweryfikowany eksport źródłowy')
    return dict(status=status, cards=cards, progress=counter, progress_subtitle=detail,
                progress_ratio=ratio, ready=bound,
                methodology="Ocena asystowana: z widoczną predykcją. Pełnej zgodności całego potoku nie oceniano osobno.")
