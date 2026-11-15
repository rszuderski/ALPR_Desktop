#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4MOBREP019: conservative comparison of recorded ALPR measurement conditions.

This is a read-only presentation model, not a statistical equivalence test.
Common series/scenario does not prove identical conditions; lack of evidence
never becomes 'matched'. Variant/model differences may be intentional, but a
planned independent factor must be documented separately. No hidden thresholds
are used to approve scientific comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Iterable, Mapping


def _obj(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    # Stored reports compact large telemetry objects under '_preview'.
    if value.get('_kind') == 'object' and isinstance(value.get('_preview'), dict):
        return value['_preview']
    return value


def _nested(data: Any, *path: str) -> Any:
    value: Any = data
    for key in path:
        value = _obj(value).get(key)
    return value


def _value(*values: Any) -> str | None:
    for item in values:
        if item is not None and not isinstance(item, (dict, list)):
            item = str(item).strip()
            if item and item not in ('-', 'None', 'null'):
                return item
    return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _count(value: Any) -> int | None:
    n = _number(value)
    return int(n) if n is not None and n.is_integer() else None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    t = str(value or '').strip().lower()
    return True if t in ('true','1','yes') else False if t in ('false','0','no') else None


def _stable_json(value: Any) -> str | None:
    if not isinstance(value, (dict, list)) or not value:
        return None
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    except (ValueError, TypeError):
        return None


def _sha(value: Any) -> str | None:
    text = _value(value)
    return text.lower() if text and len(text) == 64 and all(c in '0123456789abcdefABCDEF' for c in text) else None


def _format_temp(value: Any) -> str | None:
    n = _number(value)
    return f'{n:.1f}°C'.replace('.', ',') if n is not None else None


def _fingerprint(stage: dict) -> str | None:
    stage = _obj(stage)
    if _bool(stage.get('enabled')) is False:
        return 'Wyłączony'
    # Never create an artificial hash of a role/disabled descriptor.
    return _value(stage.get('model_fingerprint'), stage.get('fingerprint'),
                  stage.get('model_sha256'), stage.get('sha256'), stage.get('model_id'))


@dataclass(frozen=True)
class Conditions:
    report_id: str
    source_sha: str | None
    session_id: str | None
    series: str | None
    scenario: str | None
    variant: str | None
    device_model: str | None
    android: str | None
    apk: str | None
    runtime: str | None
    delegate: str | None
    camera: str | None
    settings: str | None
    mp: str | None
    mt: str | None
    mz: str | None
    model_config: str | None
    independent_variable: str | None
    charging: str | None
    thermal_status: str | None
    temp_start: str | None
    temp_end: str | None
    frames_received: int | None
    frames_processed: int | None
    frame_gate: int | None
    measured_runs: int | None
    source_gt_text_count: int | None
    source_reviewed_count: int | None


def extract_conditions(report: Any, *, bundle: Any = None) -> Conditions:
    """Use verified original payload for selected report; otherwise stored metadata.

    Missing fields remain unknown. Never read arbitrary disk paths or assume
    that a compact report is equivalent to a verified original archive.
    """
    base_raw = _obj(getattr(report, 'raw', None))
    source_sha = _sha(getattr(report, 'source_archive_sha256', None))
    verified = bool(bundle is not None and getattr(getattr(bundle,'validation',None),'ok',False)
                    and _sha(getattr(bundle, 'source_archive_sha256', None)) == source_sha
                    and str(getattr(getattr(bundle,'report',None), 'report_id','')) == str(getattr(report,'report_id','')))
    raw = _obj(getattr(bundle,'report_payload',None)) if verified else base_raw
    if not raw:
        raw = base_raw
    index = _obj(getattr(bundle,'experiment_session',None)) if verified else _obj(getattr(report,'experiment_index',None))
    if not index:
        index = _obj(raw.get('experiment_session'))
    configuration = _obj(raw.get('research_execution_config')) or _obj(_nested(index,'execution')) or _obj(raw.get('execution'))
    stages = _obj(configuration.get('stages'))
    execution = _obj(raw.get('execution'))
    mp = _obj(stages.get('mp')) or _obj(execution.get('vehicle'))
    mt = _obj(stages.get('mt')) or _obj(execution.get('plate'))
    mz = _obj(stages.get('mz')) or _obj(execution.get('character'))
    device = _obj(raw.get('device')) or _obj(getattr(report,'device',None))
    app_build = _obj(raw.get('app_build')) or _obj(index.get('app_build'))
    apk_identity = _obj(app_build.get('apk_identity'))
    camera = _obj(raw.get('capture'))
    width = _count(camera.get('actual_width'))
    height = _count(camera.get('actual_height'))
    camera_size = f'{width}×{height}' if width and height else None
    thermal = _obj(raw.get('thermal'))
    thermal_start = _obj(thermal.get('start'))
    thermal_end = _obj(thermal.get('end'))
    flow = _obj(raw.get('frame_flow')) or _obj(raw.get('_desktop_frame_flow_totals_v1'))
    quality = _obj(raw.get('quality')) or _obj(getattr(report,'quality',None))
    profile = _value(configuration.get('recognition_profile'), raw.get('recognition_profile'))
    measurement_settings = {
        'analysis_mode': _value(configuration.get('analysis_mode'),raw.get('analysis_mode')),
        'scene_handling_mode': _value(configuration.get('scene_handling_mode'),raw.get('scene_handling_mode')),
        'roi_budget_policy': _value(configuration.get('roi_budget_policy'),raw.get('roi_budget_policy')),
        'recognition_profile': profile,
        'feature_flags': configuration.get('feature_flags'),
        'dynamic_mt': configuration.get('dynamic_mt'),
    }
    if not any(v is not None for v in measurement_settings.values()):
        settings = None
    elif any(measurement_settings[k] is None for k in ('analysis_mode','scene_handling_mode','roi_budget_policy','recognition_profile')):
        # Incomplete configs cannot be certified equal.
        settings = None
    else:
        settings = _stable_json(measurement_settings)
    model_parameters = {}
    for role, entry in (('MT',mt),('MZ',mz)):
        if entry:
            model_parameters[role] = {
                'runtime':entry.get('runtime'),'precision':entry.get('precision'),
                'delegate':entry.get('delegate'),'threads':entry.get('cpu_threads'),
                'input':entry.get('input'),
            }
    source_intent = (_nested(index,'design','independent_variable')
                    or index.get('independent_variable')
                    or _nested(raw,'experiment','design','independent_variable'))
    thermal_code = thermal_start.get('thermal_status')
    if thermal_code is None:
        thermal_code = device.get('thermal_status')
    ch = thermal_start.get('charging')
    if ch is None:
        ch = device.get('charging')
    charge = _bool(ch)
    return Conditions(
        report_id=str(getattr(report,'report_id','')),
        source_sha=source_sha,
        session_id=_value(index.get('session_id'),index.get('experiment_session_id'),_nested(raw,'experiment_session','session_id')),
        series=_value(index.get('series_id')),
        scenario=_value(index.get('scenario_id')),
        variant=_value(index.get('experiment_variant'),index.get('variant'),raw.get('variant_id'),getattr(report,'variant_id',None)),
        device_model=' '.join(v for v in (_value(device.get('manufacturer')),_value(device.get('model'))) if v) or _value(device.get('name')),
        android=_value(device.get('android_version'),index.get('android_version')),
        apk=_sha(apk_identity.get('base_apk_sha256')) or _sha(app_build.get('apk_sha256')),
        runtime=_value(getattr(report,'runtime',None),raw.get('runtime')),
        delegate=_value(getattr(report,'delegate',None),raw.get('delegate')),
        camera=camera_size,
        settings=settings,
        mp=_fingerprint(mp),mt=_fingerprint(mt),mz=_fingerprint(mz),
        model_config=_stable_json(model_parameters),
        independent_variable=_value(source_intent),
        charging='Ładowanie' if charge is True else 'Bez ładowania' if charge is False else None,
        thermal_status=_value(thermal_code),
        temp_start=_format_temp(thermal_start.get('battery_temperature_c')),
        temp_end=_format_temp(thermal_end.get('battery_temperature_c')),
        frames_received=_count(flow.get('frames_received')),
        frames_processed=(_count(flow['frames_processed']) if 'frames_processed' in flow else _count(_nested(raw,'summary','processed_frames'))),
        frame_gate=_count(flow.get('frames_skipped_frame_gate')),
        measured_runs=(_count(raw['measured_runs']) if 'measured_runs' in raw else _count(_nested(raw,'summary','processed_frames'))),
        source_gt_text_count=_count(quality.get('ground_truth_samples')),
        source_reviewed_count=_count(quality.get('reviewed_samples')),
    )


def _short(value: Any, limit: int = 45) -> str:
    if value is None:
        return 'Nieustalone'
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return 'Nieustalone'
        if len(value) == 64 and all(c in '0123456789abcdef' for c in value.lower()):
            return 'SHA ' + value[:12] + '…'
        if len(value) > limit:
            return value[:limit-1] + '…'
    return str(value)


def _different(values: Iterable[Any]) -> bool:
    values = list(values)
    return len({str(value) for value in values}) > 1


def assess_cohort(selected: Any, others: Iterable[Any], *, bundle: Any = None) -> dict:
    """Conservative conditions audit; no claim of statistical equivalence.

    A source-matched in-memory bundle may enrich *selected* report only; other
    sources are not silently assumed validated merely because cached data exists.
    """
    current = extract_conditions(selected,bundle=bundle)
    seen = {(current.report_id,current.source_sha)}
    candidates=[current]
    if current.series and current.scenario:
        for report in others:
            row=extract_conditions(report)
            key=(row.report_id,row.source_sha)
            if row.series == current.series and row.scenario == current.scenario and key not in seen:
                candidates.append(row)
                seen.add(key)
    row_output=[]
    anomalies=[]
    missing=[]
    notes=[]
    def emit(name: str, value: str, cohort_value: str, status: str, explanation: str, tone: str='muted'):
        row_output.append((name,value,cohort_value,status,explanation,{'tag':tone}))

    count=len(candidates)
    if count < 2:
        status='Brak pary do porównania'
        reason='Wybierz drugą sesję z tą samą serią i scenariuszem. Sam wybór sesji nie oznacza porównania.'
    else:
        status='Weryfikacja warunków'
        reason='Zgodność części konfiguracji nie potwierdza automatycznie porównywalności naukowej.'
    emit('Wniosek',status,f'{count} sesja' if count==1 else f'{count} sesje','WSTĘPNE',reason,'warning')
    emit('Seria i scenariusz',_short(current.series)+' / '+_short(current.scenario),f'{count} raporty' if count > 1 else '1 raport',
         'Wspólny zakres' if count>1 else 'Brak pary',
         'Seria i scenariusz służą wyłącznie do wyszukania kandydatów; nie gwarantują tych samych warunków.',
         'muted' if count>1 else 'warning')

    if count > 1:
        source_hashes = [entry.source_sha for entry in candidates]
        session_ids = [entry.session_id for entry in candidates]
        if any(value is not None and source_hashes.count(value)>1 for value in source_hashes):
            anomalies.append('Powtórzone archiwum')
            emit('Niezależność źródeł', 'Powtórzone archiwum', 'To samo SHA w więcej niż jednym wpisie',
                 'NIEZALEŻNOŚĆ NIEPOTWIERDZONA',
                 'Dwa wpisy tego samego archiwum nie są dwoma niezależnymi pomiarami.', 'error')
        elif any(value is not None and session_ids.count(value)>1 for value in session_ids):
            anomalies.append('Powtórzony identyfikator sesji')
            emit('Niezależność źródeł', 'Powtórzona sesja', 'Ten sam identyfikator sesji',
                 'NIEZALEŻNOŚĆ NIEPOTWIERDZONA',
                 'Do porównania wymagane są oddzielne wykonania, nie kopie tego samego pomiaru.', 'error')
        elif None in source_hashes or None in session_ids:
            missing.append('Tożsamość źródeł')
            emit('Niezależność źródeł', 'Brak kompletnego identyfikatora', 'Weryfikuj SHA i sesję',
                 'BRAK DOWODÓW', 'Nie można potwierdzić niezależności zapisanych pomiarów.', 'warning')
        else:
            emit('Niezależność źródeł', 'Osobne identyfikatory',f'{count} wpisy',
                 'WSTĘPNIE ODRĘBNE',
                 'Różne SHA i identyfikatory nie dowodzą jeszcze, że materiał nie był powtórzony.', 'muted')
        selected_verified = bool(bundle is not None and getattr(getattr(bundle,'validation',None),'ok',False)
                                 and _sha(getattr(bundle,'source_archive_sha256',None)) == current.source_sha)
        emit('Integralność źródeł',
             'Wybrane archiwum sprawdzone' if selected_verified else 'Wybrane archiwum niezweryfikowane',
             'Sprawdź oddzielnie pozostałe', 'BRAK PEŁNEGO POTWIERDZENIA',
             'Zapisany SHA nie zastępuje weryfikacji oryginalnego archiwum każdej sesji.', 'warning')
        emit('Identyczność materiału scen', 'Ten sam scenariusz', 'Nie sprawdzono identyczności klatek',
             'DO POTWIERDZENIA',
             'Zgodna nazwa scenariusza nie dowodzi, że pokazano identyczne obrazy.', 'warning')

    def compare(label: str, attr: str, explanation: str, *, category: str='required', format_value=None):
        values=[getattr(entry,attr) for entry in candidates]
        shown=[_short(v) if format_value is None else format_value(v) for v in values]
        present=[v is not None and v != '' for v in values]
        if not all(present):
            kind='Brak danych' if count>1 else 'Do potwierdzenia'
            tone='warning'
            missing.append(label)
        elif count<2:
            kind='Jedna sesja';tone='muted'
        elif _different(values):
            if category=='variable':
                if _different(entry.variant for entry in candidates):
                    kind='Planowana zmiana?';notes.append(label);tone='warning'
                else:
                    kind='Różne modele';anomalies.append(label);tone='error'
            elif category=='observed':
                kind='Różne wartości';notes.append(label);tone='warning'
            else:
                kind='Różne warunki';anomalies.append(label);tone='error'
        else:
            kind='Zgodne (wstępnie)';tone='muted'
        if attr=='charging':
            pass
        if count>1:
            summary=(shown[0] if len(set(shown))==1 else ' / '.join(dict.fromkeys(shown)))
            if len(summary)>58:summary=summary[:57]+'…'
        else:
            summary=shown[0]
        emit(label,shown[0],summary,kind,explanation,tone)

    if count > 1:
        variants = {entry.variant for entry in candidates}
        declarations = {entry.independent_variable for entry in candidates}
        if len(variants) > 1:
            declared = len(declarations) == 1 and None not in declarations
            emit('Co celowo zmieniono w teście?',
                 _short(current.independent_variable) if declared else 'Nie wskazano w planie',
                 ', '.join(_short(v) for v in sorted(variants, key=lambda x: str(x))[:4]),
                 'Plan podany' if declared else 'WYMAGA PLANU',
                 'Zmiana FP32/INT8 lub modelu musi być zapisana jako badany czynnik, nie domyślana z nazwy.',
                 'muted' if declared else 'warning')
            if not declared:
                notes.append('Brak deklaracji czynnika badanego')
        else:
            emit('Wariant badania',_short(current.variant),f'{count} powtórzenia',
                 'Ten sam wariant', 'To powtórzenia wariantu, nie porównanie FP32 z INT8.','muted')

    compare('Telefon (model)','device_model','Ten sam model urządzenia nie potwierdza tożsamości fizycznego egzemplarza.')
    compare('Android','android','Wersja systemu wpływa na działanie aplikacji i sterowników.')
    compare('Aplikacja (SHA APK)','apk','Porównujemy rzeczywiste bajty APK, nie tylko commit Git. Inne APK wymagają uzasadnienia.')
    compare('Silnik i wykonanie','runtime','Środowisko uruchomieniowe powinno być stałe poza zaplanowanym czynnikiem.')
    compare('Procesor / akcelerator','delegate','CPU, GPU i NNAPI mają różne właściwości wydajnościowe.')
    compare('Rozmiar obrazu kamery','camera','Rzeczywisty rozmiar wejściowej klatki kamery; nie rozmiar wejścia modelu.')
    compare('Ustawienia badania','settings','Tryb sceny, progi, ogranicznik klatek i pozostałe ustawienia muszą być kontrolowane.',
            format_value=lambda value:'Zapisano konfigurację' if value is not None else 'Nieustalone')
    compare('Model pojazdów (MP)','mp','Wyłączony oznacza brak użytego modelu MP. Nie jest to fingerprint.',category='variable')
    compare('Model tablic (MT)','mt','Zmiana modelu może być zamierzonym czynnikiem, lecz wymaga zapisanego planu.',category='variable')
    compare('Model znaków (MZ)','mz','Zmiana modelu może być zamierzonym czynnikiem, lecz wymaga zapisanego planu.',category='variable')
    compare('Wariant i parametry modeli','model_config','Liczba wątków, typ precyzji i rozmiar wejścia modeli.',category='variable',
            format_value=lambda value:'Konfiguracja zapisana' if value is not None else 'Nieustalone')
    compare('Ładowanie telefonu','charging','Inny sposób zasilania może zmieniać zachowanie termiczne i częstotliwości CPU.')
    compare('Stan termiczny na początku','thermal_status','Android raportuje klasy stanu termicznego; zmiana jest warunkiem pomiaru.',category='required')
    compare('Temperatura początkowa','temp_start','Temperatura baterii jest informacją pomocniczą, nie temperaturą procesora.',category='observed')
    compare('Temperatura końcowa','temp_end','Porównanie warunków termicznych, nie bezpośredni pomiar temperatury CPU.',category='observed')
    compare('Odebrane klatki','frames_received','Różne liczby klatek wpływają na obciążenie i sposób próbkowania.',category='observed')
    compare('Przetworzone klatki','frames_processed','Liczba prób ma znaczenie dla wiarygodności mediany i P95.',category='observed')
    compare('Klatki pominięte przed analizą','frame_gate','Pominięcia bramki klatek nie są błędami wykrywania tablicy.',category='observed')
    compare('Dostępność GT znaków w Androidzie','source_gt_text_count','Zero tekstowego GT w Androidzie nie wyklucza ręcznych ocen w Desktopie.',category='observed')
    if count > 1:
        scores = [entry.source_gt_text_count for entry in candidates]
        status_gt = 'Brak GT w archiwach' if all(v == 0 for v in scores) else 'Sprawdź GT każdej sesji'
        emit('Ocena jakości MT/MZ', 'Wymaga osobnej weryfikacji GT',
             f'Tekst GT w archiwum: {" / ".join(_short(v) for v in scores)}',
             'NIE POTWIERDZONO',
             'Tekstowe GT Androida nie obejmuje ręcznych ocen Desktop. Do porównania jakości sprawdź również ich zakończenie i mianowniki.',
             'warning')
    # The presence of a bundle for *selected* report says nothing about peers.
    if count > 1:
        source_note='Sprawdź osobno SHA-256 źródeł każdej sesji, ukończenie ocen GT oraz zgodność materiału testowego.'
        if _different(e.variant for e in candidates) and not (all(e.independent_variable for e in candidates)
                                                            and not _different(e.independent_variable for e in candidates)):
            notes.append('Brak potwierdzenia planowanej zmiennej modelu')
        if anomalies:
            status='Różne warunki — brak zgody na porównanie'
        elif missing:
            status='Niepełne dane — porównanie niepotwierdzone'
        elif notes:
            status='Różnice do wyjaśnienia'
        else:
            status='Warunki wstępnie zgodne — wymagana kontrola'
        row_output[0]=('Wniosek',status,f'{count} sesje','NIE ZATWIERDZONO',source_note,
                       {'tag':'error' if anomalies else 'warning'})
    return {'status':status,'rows':tuple(row_output),'issues':tuple(dict.fromkeys(anomalies)),
            'unknown':tuple(dict.fromkeys(missing)),'notes':tuple(dict.fromkeys(notes)),
            'cohort_size':count,'session_ids':tuple(e.session_id for e in candidates)}
