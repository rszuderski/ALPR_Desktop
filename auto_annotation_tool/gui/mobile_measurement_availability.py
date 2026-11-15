"""Polish presentation of approval versus measurement coverage; no scoring."""
from __future__ import annotations
import re


def measurement_limits(canvas,mt=None):
    canvas=canvas or {};mt=mt or {};scenes=canvas.get('scenes') or []
    labels={s.get('mt_invocation_id'):f'Obraz {i:02d}' for i,s in enumerate(scenes,1)}
    rows={}
    def add(gid,text):
        if not gid:return
        row=rows.setdefault(gid,dict(gid=gid,label=labels.get(gid,'Obraz'),reasons=[]))
        if text not in row['reasons']:row['reasons'].append(text)
    names={'FINAL_SOURCE_BBOX_UNAVAILABLE':'brak wiarygodnego BBOX modelu — bez pomiaru dopasowania MT',
           'GT_GEOMETRY_UNCONFIRMED':'brak geometrii referencyjnej do pomiaru dopasowania MT',
           'FINAL_CONFIDENCE_UNAVAILABLE':'brak zapisanej pewności wskazania MT',
           'GT_PARTIAL_ROI_COVERAGE':'tablica przecina granicę ocenianego obszaru',
           'FINAL_POPULATION_UNPROVEN':'nieustalony zestaw końcowych wskazań MT',
           'SOURCE_FINAL_ROLE_UNPROVEN':'brak dowodu tożsamości FINAL/RAW — techniczne N/A',
           'SOURCE_OR_REVIEW_UNAVAILABLE':'brak zgodnego źródła lub referencji',
           'REVIEW_INCOMPLETE':'brak wymaganej decyzji operatora',
           'SCENE_EXCLUDED':'obraz wyłączony z pomiaru',
           'MT_NOT_COMPLETED':'model MT nie zakończył pracy'}
    for unit in mt.get('units') or []:
        if unit.get('status')=='MEASURED':continue
        for reason in unit.get('unavailable_reasons') or ['SOURCE_OR_REVIEW_UNAVAILABLE']:
            add(unit.get('mt_invocation_id'),names.get(reason,'pomiar niedostępny — sprawdź dane obrazu'))
    for s in scenes:
        if s.get('status') in ('CHECKED','EXCLUDED'):continue
        for issue in s.get('issues') or []:
            if 'Brak czterech rogów' in issue:
                text='brak czterech narożników referencji — bez pomiaru geometrii'
            elif 'Brak GT numeru' in issue:text='brak numeru GT lub przyczyny nieczytelności'
            elif 'Sprawdź cały obszar ROI' in issue:text='obraz czeka na sprawdzenie'
            elif 'Potwierdź obecność tablicy w ramce' in issue:text='wskazanie MT czeka na decyzję operatora'
            elif 'Zatwierdź nową lub zmienioną GTR' in issue:text='obrys operatora czeka na zatwierdzenie'
            elif 'Rozstrzygnij wskazanie na Canvas' in issue:text='wskazanie MT wymaga decyzji'
            else:
                text=issue.split(': ',1)[-1]
                text=re.sub(r'(?:GT:|PGT:|MANUAL:)?exp-[A-Za-z0-9-]+','',text).strip(' :')
                text=re.sub(r'(?:GT|PGT|MANUAL|MT):[A-Za-z0-9_.:-]+','',text).strip(' :')
            add(s.get('mt_invocation_id'),text)
    for row in rows.values():
        geometry='brak czterech narożników referencji — bez pomiaru geometrii'
        if names['GT_GEOMETRY_UNCONFIRMED'] in row['reasons'] and geometry in row['reasons']:
            row['reasons']=[('brak geometrii referencyjnej — bez pomiaru dopasowania MT i geometrii'
                if r==names['GT_GEOMETRY_UNCONFIRMED'] else r) for r in row['reasons'] if r!=geometry]
    return sorted(rows.values(),key=lambda r:list(labels).index(r['gid']) if r['gid'] in labels else len(labels))


def limits_text(canvas,mt=None):
    return '\n'.join(r['label']+': '+'; '.join(r['reasons']) for r in measurement_limits(canvas,mt))


def polish_model_caption(value):
    # Translate the recorded usage note for display, never alter model IDs/data.
    return re.sub(r'research\s+only\s*,?\s*no\s+production\s+import\s+approval',
                  'wyłącznie do badań; import produkcyjny niezatwierdzony',str(value),flags=re.I)
