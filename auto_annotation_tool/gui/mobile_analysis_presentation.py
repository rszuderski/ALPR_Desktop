"""Polish chart vocabulary; rendering only, never metric qualification or data."""
from pathlib import Path
import json,math,re

DICTIONARY_PATH=Path(__file__).with_name('mobile_analysis_labels.json')
COPY=json.loads(DICTIONARY_PATH.read_text(encoding='utf-8'))
LABELS=COPY['metrics']
SCENARIOS=COPY['scenarios']
STYLES={'FP32':dict(color='#226b9a',hatch=''), 'INT8':dict(color='#bd641d',hatch='///'),
        'R0':dict(color='#226b9a',hatch=''), 'R1':dict(color='#c48222',hatch='///'),
        'R2':dict(color='#36826d',hatch='xx')}
BOUNDED={key for key,value in LABELS.items() if value['display_type'] in ('percent','bounded_percent')}


def metric_labels(key, question=None):
    from .mobile_analysis_f5a import NEW_METRICS
    if question != 'F5A' and key in LABELS:
        return LABELS[key]
    from .mobile_analysis import metric_definition
    definition = metric_definition(key)
    title = definition[0]
    if key.startswith('pipeline_'):
        title = 'ALPR_MT_TRACE — zarejestrowany czas przebiegu (' + ('p95' if 'p95' in key else 'p50') + ')'
    count = key.endswith('_calls')
    unit = definition[1]
    return dict(title=title,
        subtitle=('Suma jawnie zapisanych liczników; brak licznika pozostaje nieznany.' if count else
                  'Mediana percentyli sesyjnych; każda sesja ma tę samą wagę.'),
        x_axis='Zapisane wywołania [liczba]' if count else 'Czas [ms]' if unit == 'ms' else definition[0],
        display_type='ms' if unit == 'ms' else 'count', coverage_unit='śladów telefonu',
        footnote=('Suma częściowa przy brakujących licznikach; uwzględnia zapisane wywołania w anulowanych śladach.' if count else
                  'n oznacza ślady z pomiarem; to nie liczba niezależnych ekspozycji. Bez pełnej latencji kamera–ekran.'),
        direction='none' if count else 'lower')


def number(value,places=1,*,trim=False):
    if value is None:return '—'
    if 0<abs(value)<10**(-places):
        places=min(9,max(places,int(-math.floor(math.log10(abs(value))))+1))
    text=f'{value:,.{places}f}'.replace(',','\u202f').replace('.',',')
    return text.rstrip('0').rstrip(',') if trim and places else text


def variant(session):
    raw=str(session.metadata.get('variant') or 'Wariant nieustalony')
    return COPY['variant'].get(raw,raw)


def session_label(session):
    if session.metadata.get('experiment') == 'F5A':
        block = session.metadata.get('block')
        return f"{variant(session)} · {('blok ' + block) if block else session.input.session_id} · {session.metadata.get('repeat', '—')}"
    repeat=str(session.metadata.get('repeat') or '')
    match=re.fullmatch(r'r(\d+)',repeat,re.IGNORECASE)
    repeat=str(int(match[1])) if match else 'nieustalone'
    return f'{variant(session)} · powtórzenie {repeat}'


def ordered_sessions(cohort,scenario):
    def order(s):
        rep=re.fullmatch(r'r(\d+)',str(s.metadata.get('repeat') or ''),re.IGNORECASE)
        return (0 if variant(s)=='FP32' else 1 if variant(s)=='INT8' else 2,variant(s),int(rep[1]) if rep else 999,s.input.session_id)
    return sorted((s for s in cohort.sessions if s.metadata.get('scenario')==scenario),key=order)


def scenario_label(scenario):return SCENARIOS.get(scenario,str(scenario))


def context_title(cohort,scenario,sessions):
    material={'PAIR_DAY_01':'Zdjęcia dzienne na TV','PAIR_NIGHT_01':'Zdjęcia nocne na TV'}.get(scenario,scenario_label(scenario))
    variants=list(dict.fromkeys(variant(s) for s in sessions))
    stage='Diagnostyka' if cohort.question=='DIAGNOSTIC' else cohort.question
    return stage+' · '+material+' · '+' i '.join(variants)


def format_value(metric):
    if metric.value is None:return 'Brak danych do wiarygodnej oceny'
    kind=metric_labels(metric.metric_id)['display_type']
    if kind in ('percent','bounded_percent','percent_unbounded'):return number(metric.value)+'%'
    if kind=='ms':return number(metric.value,1,trim=True)+' ms'
    if kind=='memory':return number(metric.value)+' MiB'
    return number(metric.value,0)


def coverage(metric):
    label=metric_labels(metric.metric_id)['coverage_unit']
    if metric.n_scored is None:return 'liczba pomiarów nieustalona'
    if metric.n_total_eligible is None:return f'{metric.n_scored} {label}; pełny zakres nieustalony'
    return f'{metric.n_scored} z {metric.n_total_eligible} {label}'


def missing_reason(metric):
    reasons=set(metric.missing_reasons)
    if metric.metric_id in ('scene_coverage_pct','exact_e2e_pct','no_read_e2e_pct') and 'SCENE_UNMAPPED' in reasons:
        return 'Nie obliczono — brak mapy scen'
    names={'PER_CALL_TIMING_UNAVAILABLE':'brak czasów pojedynczych wywołań MZ',
        'OS_PSS_NOT_MEASURED':'brak pomiarów pamięci PSS',
        'INCOMPLETE_TRACE_SOURCE':'niekompletny zapis pomiarów telefonu',
        'MT_CALL_TIMING_UNPROVEN':'brak wiarygodnych czasów pojedynczych wywołań MT',
        'ZERO_DENOMINATOR':'brak ocenialnych przypadków w mianowniku',
        'OBJECT_COUNTS_NOT_VERIFIED':'niepotwierdzone dane oceny MT',
        'SCENE_UNMAPPED':'brak mapy scen',
        'SOURCE_MISSING':'niedostępne archiwum źródłowe',
        'SOURCE_MISMATCH':'niezgodna tożsamość źródła',
        'MANIFEST_INVALID':'nieprawidłowy manifest źródła',
        'QUESTION_NOT_COMPARABLE':'brak warunków do tego porównania'}
    return next((names[r] for r in metric.missing_reasons if r in names),
        'brak potwierdzonych pomiarów tej miary' if metric.status!='NOT_COMPARABLE' else 'brak warunków do tego porównania')


def bar_label(metric):
    """Counts mean objects/reads only when the canonical numerator says so."""
    if metric.value is None:return 'Brak danych do wiarygodnej oceny\n'+missing_reason(metric)
    key=metric.metric_id;value=format_value(metric)
    suffix={'exact_mz_pct':'numerów','mt_precision_pct':'wskazań','mt_recall_pct':'tablic',
        'bbox_v1_precision_pct':'wskazań','bbox_v1_recall_pct':'tablic','pose_availability_pct':'par',
        'scene_coverage_pct':'scen','exact_e2e_pct':'tablic','no_read_e2e_pct':'tablic'}.get(key)
    if suffix and metric.numerator is not None and metric.denominator is not None:
        detail=f'{number(metric.numerator,0)} z {number(metric.denominator,0)} {suffix}'
    elif key=='micro_cer_pct' and metric.numerator is not None and metric.denominator is not None:
        detail=f'{number(metric.numerator,0)} operacji / {number(metric.denominator,0)} znaków'
    else:detail=coverage(metric)
    text=value+' · '+detail
    if metric.status=='SUBSET':
        if not suffix and key!='micro_cer_pct':text=value
        text+=' *\n* dane częściowe: '+coverage(metric)
        if metric.n_scored==metric.n_total_eligible:
            text+=' w dostępnym zakresie'
    return text


def status_label(metric):
    return 'Brak danych' if metric.value is None else 'Częściowe' if metric.status=='SUBSET' else 'Bez braków'


def report_sections(cohort):
    """Readable report with raw identifiers retained in a separate source table."""
    from .mobile_analysis import METRICS,PROTOCOL_ID
    if cohort.question == 'F5A':
        from .mobile_analysis_f5a import ALGORITHM_ID, METRIC_IDS, LIMITATIONS, aggregate
        lines=['# F5A — opisowe porównanie wydajności', '', ALGORITHM_ID,
            'Kwalifikacja danych: ' + cohort.performance.get('status', 'N/A'),
            'Układ 36 sesji: ' + str(cohort.performance.get('balanced_36_sessions')), '',
            'Czasy: mediana percentyli sesyjnych; liczniki: suma jawnych wartości. GT poza zakresem.',
            'Plan 120 ekspozycji na wariant nie jest mianownikiem czasu ani liczników.', '']
        for scenario in sorted({s.metadata.get('scenario', 'NIEOKREŚLONY') for s in cohort.sessions}):
            for key in METRIC_IDS:
                lines += ['## ' + metric_labels(key, 'F5A')['title'], '',
                    '| Wariant | Wynik | Sesje | Ślady z wartością / wszystkie | Status |', '|---|---:|---:|---:|---|']
                for row in aggregate(cohort, key, scenario):
                    lines.append(f"| {row['variant']} | {number(row['value']) if row['value'] is not None else 'N/A'} | "
                        f"{row['n_sessions']}/{row['n_sessions_selected']} | {row['n_source_observations']}/{row['n_source_observations_total']} | {row['status']} |")
                lines.append('')
        lines += ['## Źródła', '', '| Sesja | session_id | SHA-256 |', '|---|---|---|']
        lines += [f'| {session_label(s)} | {s.input.session_id} | {s.input.source_sha256} |' for s in cohort.sessions]
        lines += ['', '## Ograniczenia', '', *['- ' + text for text in LIMITATIONS], '',
            'Pełne warunki, różnice, powody N/A i serie pomiarów: analysis.json; sesje: readiness_matrix.csv; agregaty: variant_metrics.csv.']
        return lines
    lines=['# Analiza wybranych raportów ALPR','',
        'Porównanie opisowe. Dla F4 materiałem są zdjęcia wyświetlane na telewizorze; w obu materiałach użyto modeli DAY.',
        'Referencje są asystowane. F4-EVAL-2.0 przyjęto retrospektywnie 2026-11-01; wyniki nie są niezależnym testem F6.','']
    for scenario in sorted({s.metadata.get('scenario','NIEOKREŚLONY') for s in cohort.sessions}):
        sessions=ordered_sessions(cohort,scenario)
        lines+=['## '+scenario_label(scenario),'']
        for key in METRICS:
            copy=LABELS[key]
            lines+=['### '+copy['title'],'',copy['subtitle'],copy['footnote'],'',
                '| Wariant i powtórzenie | Wynik i zakres pomiarów | Dostępność |','|---|---|---|']
            for s in sessions:
                metric=s.metrics[key];label=bar_label(metric).replace('\n','; ').replace('|','/')
                lines.append(f'| {session_label(s)} | {label} | {status_label(metric)} |')
            lines.append('')
    lines+=['## Pochodzenie i dane techniczne','',PROTOCOL_ID+' / F4-EVAL-2.0 · DESCRIPTIVE · ASSISTED.',
        'Surowe jednostki, statusy FULL/SUBSET/N/A, liczniki i mianowniki pozostają w analysis.json i readiness_matrix.csv.',
        'v1_v2_all_units.csv zachowuje osobne wyniki pomocniczego dopasowania BBOX v1 i głównej oceny MT v2.','',
        '| Materiał | Wariant i powtórzenie | ID sesji | SHA-256 archiwum | Rewizja |','|---|---|---|---|---|']
    for s in cohort.sessions:
        lines.append(f"| {scenario_label(s.metadata.get('scenario'))} | {session_label(s)} | {s.input.session_id} | {s.input.source_sha256} | {s.metadata.get('review_revision','—')} |")
    lines+=['','## Ograniczenia','',*['- '+note for note in cohort.payload()['limitations']]]
    return lines
