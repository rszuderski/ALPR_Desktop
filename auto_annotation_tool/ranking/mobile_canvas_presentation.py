"""Read-only canvas inventory for live status/navigation, separate from metrics.

Unopened eligible frames are included in progress. Metric denominators still
come exclusively from mobile_canvas_review.summary and confirmed/evaluable data.
"""
from __future__ import annotations

from . import mobile_canvas_review as core


def build_inventory(session):
    result = dict(total=0, editable=0, inspected=0, assessed=0, excluded=0,
                  remaining=0, incomplete=0, unavailable=0, decisions=0, evidence_gaps=0, groups={})
    if session is None:
        return result
    for gid, group in session.mt_invocations.items():
        result["total"] += 1
        item = dict(gid=gid, scene=None, status="UNAVAILABLE", cells=("○", "—", "—", "—", "○"),
                    issues=[], persisted=core.has_scene(session, gid),
                    subject_keys=tuple(group.subject_keys))
        try:
            scene = core.scene(session, gid, include_history=False)
        except (ValueError, TypeError, KeyError) as exc:
            item["issues"] = [str(exc)]
            from .mobile_frame_review_state import for_session
            frame=for_session(session,gid)
            item.update(status=frame['state'],approval_issues=[],frame_review=frame,progress=review_progress(None,frame))
            result['inspected']+=int(frame['complete'])
            result["unavailable"] += 1
        else:
            result["editable"] += 1
            item["scene"] = scene
            from .mobile_frame_review_state import for_session,READY,NO_REFERENCE,SOURCE_NA
            frame=for_session(session,gid,scene);item['frame_review']=frame
            item['progress'] = review_progress(scene,frame)
            result['decisions'] += item['progress']['decisions']
            result['evidence_gaps'] += item['progress']['evidence_gaps']
            item["cells"] = core.scene_review_cells(scene)
            excluded = scene["checked"] and bool(scene.get("exclusion_reason"))
            issues = [] if excluded else core._content_issues(scene)
            item["issues"] = issues
            item['approval_issues']=core.approval_scene_issues(scene)
            presence_verified = item['progress']['complete']
            if presence_verified or excluded:
                result["inspected"] += 1
            if excluded:
                result["excluded"] += 1
                item["status"] = "EXCLUDED"
            elif frame['state']==READY:
                result["assessed"] += 1
                item["status"] = "READY"
                if issues:result['incomplete']+=1
            elif frame['state'] in {NO_REFERENCE,SOURCE_NA}:
                item['status']=frame['state']
            elif presence_verified:
                result["incomplete"] += 1
                item["status"] = "VERIFIED"
            else:
                item["status"] = "PENDING"
        result["groups"][gid] = item
    if getattr(session.review, 'scene_annotations', {}):
        from .mobile_scene_frame_review import unit_for_mt
        from .mobile_mt_object_evaluation import evaluate_unit
        for gid,item in result['groups'].items():
            if unit_for_mt(session,gid) is None:continue
            measured=evaluate_unit(session,gid)
            if measured['n_qualified']:
                item.update(status='READY',persisted=True,issues=[],approval_issues=[],
                            reference_scope='SCENE_FRAME_SAME_STAMP_TO_MT_ROI',
                            progress=dict(decisions=0,evidence_gaps=0,unresolved=0,complete=True,title='Ocena z pełnej sceny',action='Zweryfikowany',tone='green'))
                item['frame_review']=dict(item['frame_review'],complete=True,metric_complete=True)
        result['inspected']=sum(bool(r.get('frame_review',{}).get('complete')) for r in result['groups'].values())
        result['assessed']=sum(r['status']=='READY' for r in result['groups'].values())
        result['decisions']=sum(r['progress']['decisions'] for r in result['groups'].values())
    result["remaining"] = result["total"] - result["inspected"]
    result['approval_ready']=sum(not r.get('approval_issues',r['issues']) for r in result['groups'].values())
    result['approval_remaining']=result['total']-result['approval_ready']
    result['source_na']=sum(r['status']=='SOURCE_NOT_EVALUABLE' for r in result['groups'].values())
    result['no_reference']=sum(r['status']=='NO_REFERENCE_EVIDENCE' for r in result['groups'].values())
    result['approval_decisions']=result['decisions']+sum(bool(r.get('approval_issues',r['issues']))
        for r in result['groups'].values() if r.get('progress',{}).get('complete'))
    from .mobile_experiment_identity import from_bundle
    identity = from_bundle(getattr(session, 'bundle', None))
    if identity.get('family') == 'F5A':
        result['experiment_family'] = 'F5A'
        result['operator_assessed'] = sum(r['persisted'] and r['status'] == 'READY'
                                           for r in result['groups'].values())
        result['operationally_resolved'] = result['inspected']
    if getattr(session, 'scene_frames', {}):
        from .mobile_scene_frame_review import summary as scene_summary
        result['scene_scope'] = scene_summary(session)
    if getattr(session, 'bundle', None) is not None and hasattr(session, 'sidecar_path'):
        from .mobile_operator_progress import from_session
        result['operator_review_progress'] = from_session(session, result)
    return result


def subject_groups(session, subject_key):
    ids = {session.invocation_by_attempt.get(aid)
           for aid in session.subjects.get(subject_key, {}).get("attempts", ())}
    return [gid for gid in session.mt_invocations if gid in ids]


def percent(value):
    return "—" if value is None else f"{value:.2f}%".replace(".", ",")


def review_progress(scene,frame=None):
    """One meaning for list, panel and progress, independent of metric counts."""
    from .mobile_canvas_partial_review import progress as frame_progress
    frame = frame if frame is not None else frame_progress(scene)
    if scene and scene.get('exclusion_reason'):
        return dict(decisions=0, evidence_gaps=0, unresolved=0, title='Obraz poza oceną',
                    action='Poza oceną', tone='purple', complete=False)
    from .mobile_frame_review_state import NO_REFERENCE,SOURCE_NA
    if frame['state'] in {NO_REFERENCE,SOURCE_NA}:
        text='Brak referencji · N/A' if frame['state']==NO_REFERENCE else 'Źródło nieocenialne · N/A'
        return dict(decisions=0,evidence_gaps=1,unresolved=0,title=text,action=text,tone='blue',complete=True)
    return dict(decisions=frame['decisions'], evidence_gaps=sum(not p.get('quad') for p in (scene or {}).get('parents',[])),
                unresolved=int(frame['source_unknown']), complete=frame['complete'],
                title='Obraz zweryfikowany' if frame['complete'] else f"Decyzje ramek: {frame['decisions']}",
                action='Zweryfikowany' if frame['complete'] else f"Decyzje: {frame['decisions']}",
                tone='green' if frame['complete'] else 'amber')


def _legacy_review_progress(scene):
    """Historical diagnostic copy retained for non-Canvas callers."""
    from .mobile_canvas_resolution import missing_mt_geometry, valid_quad
    missing = missing_mt_geometry(scene)
    unresolved = [o for o in missing if o.get('identity_unresolved')]
    pending = [o for o in missing if not o.get('identity_unresolved')]
    other = [i for i in core._content_issues(scene)
             if not any(o['object_id'] in i for o in missing)]
    decisions = len(pending) + len(other)
    gaps = sum(not valid_quad(scene, p.get('quad')) for p in scene['parents']
               if not any(o.get('post_nms_suppressed') and o['parent_id'] == p['parent_id'] for o in scene['objects']))
    if scene.get('exclusion_reason'):
        title, action, tone = 'Obraz poza oceną', 'Poza oceną', 'purple'
        decisions = 0
    elif decisions:
        action = f'Decyzje: {decisions}'
        title = ('Przejrzany' if scene['checked'] else 'Do przejrzenia') + f' · decyzje: {decisions}'
        tone = 'amber'
    elif unresolved:
        title, action, tone = 'Nie można ustalić · zapisano przyczynę', 'Nie można ustalić', 'amber'
    elif scene['checked']:
        title, action, tone = 'Przejrzany · ocena kompletna', 'Gotowe', 'green'
    else:
        title, action, tone = 'Do przejrzenia', 'Do sprawdzenia', 'blue'
    return dict(decisions=decisions, evidence_gaps=gaps, unresolved=len(unresolved), title=title, action=action, tone=tone)


def progress_caption(inventory):
    if inventory.get('operator_review_progress'):
        from .mobile_operator_progress import presentation
        p = presentation(inventory['operator_review_progress'])
        return f"Weryfikacja operatora: {p['status']} · {p['progress']}"
    return (f"Przejrzane: {inventory['inspected']}/{inventory['total']}  ·  "
            f"Decyzje do podjęcia: {inventory.get('decisions', inventory['incomplete'])}  ·  "
            f"Braki dowodów modelu: {inventory.get('evidence_gaps', 0)}")


def review_contexts(session, scene, model, inventory):
    """Three read-only captions over existing object policy/counts/inventory.

    No matching, approval, new metric denominator or automatic object choice.
    The caller supplies a model resolved from explicit selection/text editing.
    """
    execution = model['execution']
    obj = next((o for o in (scene or {}).get('objects', ())
                if o['object_id'] == model.get('focus_id')), None)
    item = inventory['groups'].get((scene or {}).get('mt_invocation_id'), {})
    complete = item.get('status') == 'READY'
    selected = 'Wybierz ramkę, aby zobaczyć jej szczegóły'
    if obj:
        parent = next((p for p in scene['parents'] if p['parent_id'] == obj.get('parent_id')), None)
        name = (f"Wskazanie MT nr {int(parent['detection_index']) + 1}" if parent else
                'Tablica dodana ręcznie — bez wskazania MT')
        if obj.get('post_nms_suppressed'):
            selected = name + '\nSurowy kandydat odrzucony przez NMS; poza oceną końcową.'
        elif parent and (obj.get('error') or obj.get('duplicate_of')):
            selected = name + (' — błędne wskazanie' if complete else ' — robocza decyzja FP')
            selected += ('\nDodatkowe wskazanie tej samej tablicy.' if obj.get('duplicate_of') else
                         '\nModel wskazał obszar bez tablicy.')
        elif obj.get('error'):
            selected = 'Ramka ręczna — oznaczony błąd operatora'
        else:
            qualified = bool(parent and complete and not model.get('identity_reason'))
            selected = name + (' — trafne' if qualified else '')
            gt = obj.get('gt_text') if obj.get('text_confirmed') else ''
            mz = ('nie wykonano' if not parent or parent.get('mz_status') == 'NOT_RUN' else
                  (parent.get('mz_prediction') or 'model nie odczytał numeru') if parent.get('mz_ran') else
                  'wykonanie niepotwierdzone')
            selected += f"\nPrawdziwy numer: {gt or 'niepodany'}\nModel odczytał: {mz}"
            if model.get('state') == 'MISMATCH':selected += '\nOdczyt różni się od numeru GT.'
            elif model.get('state') == 'EXACT':selected += '\nOdczyt zgodny z numerem GT.'
            elif not complete:selected += '\nOcena robocza — obraz wymaga rozstrzygnięcia.'
            if model.get('geometry_status') == 'NOT_AVAILABLE':
                decision = obj.get('identity_decision') or {}
                selected += ('\nPowiązanie tablicy potwierdzone.' if decision else '')
                selected += '\nNie można zmierzyć dokładności narożników: '
                selected += ('niepoprawny wynik Pose.' if decision.get('pose_status') in ('INVALID', 'INVALID_QUAD') else 'brak danych narożników.')
            elif model.get('frame_state') == 'MT_INVALID_POSE':
                row = session.attempts.get(obj.get('attempt_id'), {}) if session else {}
                invalid = (row.get('mt_status') == 'DETECTION_INVALID_QUAD' or
                           str(row.get('mt_pose_status') or '').upper() == 'INVALID_QUAD')
                selected += ('\nNie można zmierzyć dokładności narożników: ' +
                             ('niepoprawny wynik Pose.' if invalid else 'brak danych narożników.'))
    image = execution.get('title', 'Brak obrazu') + '\n' + execution.get('detail', '')
    if scene and execution.get('state') == 'MT_EXECUTED':
        from .mobile_candidate_stacks import physical_count_view
        tally = core.counts(scene)
        physical_n,confirmed=physical_count_view(scene)
        raw = tally['mt']
        final = raw - (1 if scene.get('_post_nms_pending') else tally['raw_suppressed'])
        final_label = 'nieustalone' if scene.get('_post_nms_unknown') else str(final)
        image = 'GT fizyczne: ' + (str(physical_n) if confirmed else 'do ustalenia')
        image += f'\nWskazania MT: {raw} surowych / {final_label} końcowych'
        image += (f"\nTrafne: {tally['linked']} · Błędne: {tally['false_positive']} · Pominięte: {tally['missed']}"
                  if complete else '\nWyniki robocze — do rozstrzygnięcia' if not scene.get('exclusion_reason') else '')
        image += '\n' + _legacy_review_progress(scene)['title']
        if scene.get('exclusion_reason'):image += ': ' + scene['exclusion_reason']
    total = inventory['total']
    status = 'Zatwierdzona' if session and session.review.review_status == 'COMPLETED' else 'W toku'
    whole = (f"Przejrzane: {inventory['inspected']}/{total} · Oceny kompletne: {inventory['assessed']}/{total}"
             f"\nDecyzje do podjęcia: {inventory['decisions']} · Braki dowodów: {inventory['evidence_gaps']}"
             f'\n{status}' + (' · zakres roboczy' if status == 'W toku' else '')) if session else 'Brak sesji'
    if inventory.get('operator_review_progress'):
        from .mobile_review_status import approval_caption
        whole = approval_caption(inventory, session.review.review_status, session.review.review_revision)
    return dict(object=selected, image=image, session=whole)


def operator_context(session, gid, scene, selected, execution, *, target_id=None):
    """Read-only causal explanation and explicit pair choices for the current view."""
    from .mobile_canvas_resolution import missing_mt_geometry, model_location
    result = dict(focus_id=selected, pairs=[], target=None, location=None, location_reason='',
                  marking='', model='', question='', consequence='', numbers='', technical='')
    if not scene or execution.get('state') != 'MT_EXECUTED':
        return result
    missing = missing_mt_geometry(scene)
    manuals = [o for o in scene['objects'] if not o.get('parent_id') and not o['error']]
    targets = [o for o in scene['objects'] if o.get('parent_id') and not o['error'] and not o.get('post_nms_suppressed')]
    obj = next((o for o in scene['objects'] if o['object_id'] == selected), None)
    if obj is None and not selected and len(manuals) == 1 and missing:
        obj = manuals[0]  # preview only; never mutate selection or the sidecar
        result['focus_id'] = obj['object_id']
    if obj is None and not selected:
        candidates = [o for o in scene['objects'] if not o.get('post_nms_suppressed') and not o['error']]
        if len(candidates) == 1:
            obj = candidates[0]
            result['focus_id'] = obj['object_id']
    if obj:
        number = scene['objects'].index(obj) + 1
        result['marking'] = ('Twoja ramka — zaznaczona ręcznie.' if not obj.get('parent_id') else
                             f'Tablica {number} — ramka operatora powiązana z modelem.' if obj.get('identity_decision') else
                             f'Wskazanie modelu {number}.')
        if obj.get('parent_id'):
            result['pairs'] = [(m,obj) for m in manuals if obj['object_id'] not in m.get('separate_from_mt', ())]
        else:
            result['pairs'] = [(obj,t) for t in targets if t['object_id'] not in obj.get('separate_from_mt', ())]
    if result['pairs'] and len(result['pairs']) == 1:
        result['target'] = result['pairs'][0][1]
    elif obj and obj.get('parent_id'):
        result['target'] = obj
    elif len(missing) == 1:
        result['target'] = missing[0]
    if target_id:
        explicit = [t for m,t in result['pairs'] if t['object_id'] == target_id]
        if len(explicit) == 1:result['target'] = explicit[0]
    target = result['target']
    parent = next((p for p in scene['parents'] if target and p['parent_id'] == target['parent_id']), None)
    if target:
        try:
            result['location'] = model_location(session, gid, scene, target)
        except (ValueError, TypeError, KeyError) as exc:
            result['location_reason'] = str(exc)
        row = session.attempts.get(target['attempt_id'], {})
        invalid = str(row.get('mt_pose_status') or '').upper() == 'INVALID_QUAD' or row.get('mt_status') == 'DETECTION_INVALID_QUAD'
        result['model'] = (f"Wskazanie modelu {int(parent['detection_index'])+1}: " +
            ('brak poprawnych narożników. Obrysu nie można ocenić.' if invalid else
             'Brakuje dowodu narożników. Nie można ocenić obrysu ani przypisać przyczyny modelowi.' if not parent['quad'] else
             'Zapisano oryginalne cztery narożniki.'))
        result['numbers'] = ('Odczytu numeru nie wykonano.' if parent['mz_status'] == 'NOT_RUN' else
            'Odczyt numeru: ' + (parent['mz_prediction'] or 'nie odczytano znaków') if parent['mz_ran'] else 'Wykonanie odczytu niepotwierdzone.')
        result['technical'] = f"attempt_id: {target['attempt_id']}\nPose: {row.get('mt_pose_status') or 'brak dowodu'}\nPozycja: {result['location'] or result['location_reason']}"
    else:
        result['model'] = f"Wykrycia modelu: {execution.get('detections', 'nieustalone')}."
    if missing and target in missing:
        if not result['location']:
            result['model'] += '\nNie mamy wiarygodnego położenia wskazania. Nie można potwierdzić, co wskazał model.'
        if target.get('identity_unresolved'):
            result['question'] = 'Zapisano: nie można ustalić tożsamości wskazania.'
        elif manuals:
            result['question'] = 'Czy model wskazał tę tablicę?'
        else:
            result['question'] = 'Jeśli widzisz prawdziwą tablicę, zaznacz jej cztery rogi klawiszem D.'
        result['consequence'] = 'Tak: trafne wskazanie; obrys nadal bez oceny.'
        result['numbers'] += '\nPrawdziwy numer: najpierw rozstrzygnij powiązanie tablicy.'
    elif obj:
        state = plate_review_state(scene, obj, execution)
        number = obj.get('gt_text') or ('niewymagany — odczytu nie wykonano.' if parent and parent['mz_status'] == 'NOT_RUN' else
            'wpisz numer przy ramce.' if state['gt_editable'] else state['gt_reason'])
        result['numbers'] += '\nPrawdziwy numer: ' + number
    if obj and obj.get('error'):
        result.update(marking='Wskazanie oznaczone jako błędne (nie tablica).',
                      numbers='Numer i dokładność obrysu nie dotyczą błędnego wskazania.')
    return result


def plate_review_state(scene, obj, execution=None):
    """Read-only per-object edit policy and current/audit geometry provenance.

    The editable child is compared to its immutable, identity-bound MT parent.
    A checked image or a historical geometry_state is not proof of correction.
    """
    import math
    if obj and obj.get('duplicate_of'):obj=dict(obj,error=True)

    result = dict(gt_editable=False, gt_reason="Nie można potwierdzić końcowej detekcji MT tej tablicy.",
                  gt_note="", frame_state="UNKNOWN", frame="Brak ramki / nie można ustalić pochodzenia",
                  canvas_frame="Pochodzenie nieustalone", correction_history="Historia korekty: nieustalona",
                  mt_label="", bound_parent_id=None, identity_reason="Brak wybranej ramki.")
    if scene and scene.get('scope') == 'SCENE_FRAME' and obj:
        result.update(gt_editable=True, gt_reason='', identity_reason='', frame_state='MANUAL',
                      frame='GTR · pełna scena · ASSISTED', canvas_frame='Referencja sceny', correction_history='')
        return result
    if not scene or not obj:
        return result
    oid = obj.get("object_id")
    if len([o for o in scene.get("objects", ()) if o.get("object_id") == oid]) != 1:
        result["identity_reason"] = "Identyfikator ramki nie jest jednoznaczny. Wybierz wskazanie ponownie."
        return result
    parents = [p for p in scene.get("parents", ()) if obj.get("parent_id")
               and p.get("parent_id") == obj["parent_id"]]
    siblings = [o for o in scene.get("objects", ()) if obj.get("parent_id")
                and o.get("parent_id") == obj["parent_id"]]
    parent = (parents[0] if len(parents) == len(siblings) == 1
              and obj.get("attempt_id") == parents[0].get("attempt_id") else None)
    raw = (obj.get("post_nms_suppressed", False) or bool(obj.get("attempt_id")
           and obj["attempt_id"] == scene.get("_post_nms_pending")))
    executed = bool(execution and execution.get("state") == "MT_EXECUTED"
                    and execution.get("mt_invocation_id") == scene.get("mt_invocation_id")
                    and execution.get("image_entry") == scene.get("image_entry"))
    manual = obj.get("parent_id") is None and not obj.get("attempt_id") and obj.get("kind") == "manual"
    result["identity_reason"] = ("" if (parent or manual) and executed else
        "Nie potwierdzono wykonania MT dla tej klatki." if not executed else
        "Brak jednoznacznego powiązania object_id → parent_id → attempt_id. Sprawdź źródło ramki.")
    if parent:
        result.update(mt_label=f"MT {int(parent['detection_index']) + 1}", bound_parent_id=parent["parent_id"])
    result["gt_editable"] = bool(executed and parent and not raw and not obj.get("error"))
    if raw:
        result.update(frame_state="RAW_SUPPRESSED", frame="RAW odrzucony przez NMS · poza końcowymi detekcjami",
                      canvas_frame="RAW · poza oceną", gt_reason="RAW odrzucony przez NMS nie jest końcową detekcją MT.")
    elif manual:
        result["gt_reason"] = ("MT nie wykrył tej tablicy. MZ nie uruchomiono — numer GT nie jest tutaj wymagany "
                               "do oceny. Potwierdź obecność tablicy, aby rozliczyć pominięcie MT."
                               if executed else "Brak zakończonego wywołania i powiązanej detekcji MT. "
                               "Historyczny numer GT pozostaje tylko do odczytu; bez FN modelu MT.")
    elif obj.get("error"):
        result["gt_reason"] = "Ramka oznaczona jako błędna. Cofnij oznaczenie błędu przed edycją GT."
    elif not executed:
        result["gt_reason"] = "Nie potwierdzono zakończonego wywołania MT tej klatki. Numer GT pozostaje tylko do odczytu."
    else:
        result["gt_reason"] = "" if result["gt_editable"] else result["gt_reason"]
    if not result["gt_editable"]:
        result["gt_note"] = ("Zapis historyczny · tylko do odczytu" if obj.get("gt_text") else "Edycja GT niedostępna")

    if not raw:
        try:
            quad = core.validate_mt_quad(obj.get("quad"), *scene["image_size"])
            if manual:
                result.update(frame_state="MANUAL", canvas_frame="Ręczna · brak MT",
                    frame="Ramka referencyjna · dodana ręcznie" + (" (MT nie wykrył tablicy)" if executed else " · brak powiązanej detekcji MT"),
                    correction_history="")
            elif parent and executed:
                original = core.validate_mt_quad(parent.get("quad"), *scene["image_size"])
                same = all(math.isclose(a, b, abs_tol=1e-6, rel_tol=0)
                           for p, q in zip(quad, original) for a, b in zip(p, q))
                result.update(frame_state="MT_ORIGINAL" if same else "MT_CORRECTED",
                    frame="Ramka MT · bez korekty operatora" if same else "Ramka MT · skorygowana przez operatora",
                    canvas_frame="MT · oryginalna" if same else "MT · po korekcie")
                if any(e.get("object_id") == oid and e.get("action") in {"correct_child_geometry", "operator_link_manual_to_mt", "operator_edit_shared_gt"}
                       for e in scene.get("events", ()) if isinstance(e, dict)):
                    result["correction_history"] = "Historia: zapisano operację korekty" + (" · obecnie oryginał MT" if same else "")
        except (ValueError, TypeError, KeyError, OverflowError):
            if parent and executed:
                from .mobile_canvas_resolution import valid_quad
                if not valid_quad(scene, parent.get("quad")):
                    result.update(frame_state="MT_INVALID_POSE", frame="Wskazanie MT · brak źródłowych narożników",
                                  canvas_frame="Model · narożniki niedostępne", correction_history="")
    if parent and executed and obj.get("error") and not raw:
        result.update(frame_state="MT_FP", frame=f"{result['mt_label']} — błędne wykrycie (FP)",
                      canvas_frame="MT · błędne wykrycie (FP)", correction_history="")
        if obj.get('duplicate_of'):
            result.update(frame='Dodatkowe końcowe wskazanie tej samej tablicy · FP',
                          canvas_frame='Model · duplikat tej tablicy',gt_reason='Jedna tablica ma jedną referencję; ten wynik jest końcowym duplikatem.')
    elif parent and executed and not parent.get("quad") and obj.get("identity_decision") and obj.get("quad"):
        pose = obj["identity_decision"].get("pose_status")
        result.update(frame_state="MT_MATCHED_NO_POSE", frame=f"{result['mt_label']} · trafne wskazanie; " +
            ("błąd Pose" if pose == "INVALID" else "brak dowodu Pose"),
            canvas_frame="Tablica powiązana · obrys bez oceny", correction_history="")
    elif manual and executed and not obj.get("separate_from_mt"):
        from .mobile_canvas_resolution import missing_mt_geometry
        if missing_mt_geometry(scene):
            result.update(frame="Ramka ręczna · powiązanie do rozstrzygnięcia",
                canvas_frame="Twoja ramka · jeszcze niepowiązana",
                gt_reason="Najpierw określ powiązanie ramki ręcznej z MT. GT znaków nie zastępuje tej decyzji.")
    return result


def image_frame_status(scene, execution):
    """Current visible outlines, separate from quality metric denominators."""
    if not scene:
        return "Obrysy: nieustalone (brak zweryfikowanej sceny)"
    counts = dict(MT_ORIGINAL=0, MT_CORRECTED=0, MT_FP=0, MANUAL=0, MT_INVALID_POSE=0, MT_MATCHED_NO_POSE=0, UNKNOWN=0)
    for obj in (scene or {}).get("objects", ()):
        state = plate_review_state(scene, obj, execution)["frame_state"]
        if state in counts:
            counts[state] += 1
    titles = dict(MT_ORIGINAL="MT bez korekty", MT_CORRECTED="MT po korekcie", MANUAL="ręczna",
                  MT_FP="błędne MT (FP)", MT_INVALID_POSE="MT bez poprawnej Pose",
                  MT_MATCHED_NO_POSE="powiązane MT bez Pose", UNKNOWN="nieustalone")
    return "Obrysy: " + (" · ".join(f"{n} {titles[k]}" for k, n in counts.items() if n) or "brak")


def execution_state(session, gid, *, image_entry=None, image_size=None):
    """Read execution evidence for this invocation AND source frame, never GT."""
    from .mobile_source_frame_roi import source_invocation_context, source_frame_entry
    from .mobile_mt_postdedup import post_nms_proof
    if gid in getattr(session, 'scene_frames', {}):
        from .mobile_scene_frame_review import context
        ctx = context(session, gid, image_size)
        return dict(state='SCENE_FRAME', title='Pełna scena · niezależnie od MT',
                    detail='MT/ROI oceniane osobno; TV E2E N/A', detections=None,
                    mt_invocation_id=None, scene_event_id=gid, image_entry=ctx['entry'] if ctx else None)

    result = dict(state="MT_UNKNOWN", title="MT: nie potwierdzono wykonania",
                  detail="Ocena MT: nieustalona (bez FN)", detections=None,
                  mt_invocation_id=gid, image_entry=None)
    group = session.mt_invocations.get(gid) if session else None
    if not group or group.mt_invocation_id != gid or group.session_id != session.session_id:
        return result
    ctx = source_invocation_context(group.records, gid, session.entry_names, image_size)
    # Strict NOT_RUN records have no invocation id: their attempt id binds the
    # historical image. Do not manufacture an invocation or borrow a neighbour.
    historical = (not group.executed and group.legacy_identity and bool(group.records)
        and all(str(r.get("id")) == gid and not r.get("mt_invocation_id")
                and str(r.get("mt_status")).upper() == "NOT_RUN"
                and str(r.get("mt_executed")).lower() == "false"
                for r in group.records))
    if ctx:
        if (ctx["source_sequence"] != str(group.source_sequence or "")
                or ctx["source_timestamp_nanos"] != str(group.source_timestamp_nanos or "")):
            return result
        entry = ctx["entry"]
    elif historical:
        entries = {source_frame_entry(r, session.entry_names) for r in group.records}
        if len(entries) != 1 or None in entries:
            return result
        entry = next(iter(entries))
    else:
        return result
    if image_entry is not None and image_entry != entry:
        return result
    result["image_entry"] = entry
    if group.cancelled:
        result.update(state="MT_CANCELLED", title="MT: anulowano",
                      detail="Poza oceną skuteczności MT (nie FN)")
    elif group.execution_failed:
        result.update(state="MT_EXECUTION_FAILED", title="MT: awaria wykonania",
                      detail="Awaria wykonania, poza oceną jakości MT (nie FN)")
    elif not group.executed:
        if not all(str(r.get("mt_status")).upper() == "NOT_RUN" for r in group.records):
            return result
        result.update(state="MT_NOT_RUN", title="MT: nie uruchomiono",
                      detail="Obraz historyczny · poza metrykami MT (nie FN)")
    else:
        # Validate completeness of detections, including invalid source quads.
        if any(str(r.get("mt_status") or "").upper() not in core.DETECTION_STATUSES | {"NO_DETECTION"}
               or str(r.get("mt_executed")).lower() == "false" for r in group.records):
            return result
        try:
            parents = core._parents(session, ctx)
        except (ValueError, TypeError, KeyError):
            return result
        from .mobile_mt_postdedup import canvas_post_nms_proof,suppressed_ids
        proof = canvas_post_nms_proof(session, gid)
        suppressed = len(set(suppressed_ids(proof)) & {p['attempt_id'] for p in parents})
        n = len(parents) - suppressed
        result.update(state="MT_EXECUTED", detections=n,
                      title=f"MT: uruchomiono · detekcje: {n}",
                      detail=(f"RAW MT: {len(parents)} · po NMS: {n}" if suppressed else
                              "Zakończone wywołanie tej klatki / ROI"))
    if result["state"] in {"MT_NOT_RUN", "MT_CANCELLED", "MT_EXECUTION_FAILED"}:
        saved = session.invocation_annotation(gid).get(core.KEY, {})
        if (isinstance(saved, dict) and saved.get("source_archive_sha256") == session.review.source_archive_sha256
                and saved.get("mt_invocation_id") == gid and saved.get("image_entry") == entry):
            # Historical GT is evidence only. Never restore this as an editable
            # scene or pass it to any metric when execution is unavailable.
            result["historical_gt"] = [str(o["gt_text"]) for o in saved.get("objects", ())
                if o.get("text_confirmed") and o.get("gt_text") and not o.get("error")
                and not o.get("post_nms_suppressed")]
            canonical=saved.get('candidate_stacks_v1')
            if isinstance(canonical,dict) and canonical.get('source_binding')==saved.get('source_binding'):
                result['historical_gt']=[str(p['gt_text']) for p in canonical.get('plates',{}).values()
                                         if p.get('text_confirmed') and p.get('gt_text')]
        statuses = {str(r.get("mz_status") or "").upper() for r in group.records}
        result["mz_not_run"] = statuses == {"NOT_RUN"}
    return result


def focused_assessment(scene, selected_id, execution, *, text_edit_id=None, live_quad=None, allow_auto=True):
    """Resolve read-only focus anew; never turn an automatic preview into selection."""
    objects = (scene or {}).get("objects", ())
    target_id = text_edit_id or selected_id
    selected = [o for o in objects if o["object_id"] == target_id
                and not o.get("post_nms_suppressed")]
    obj = selected[0] if len(selected) == 1 else None
    automatic = False
    if allow_auto and not obj and not target_id and scene and not scene.get("_post_nms_pending"):
        candidates = [o for o in objects if not o.get("post_nms_suppressed") and not o.get("error")]
        if len(candidates) == 1:
            candidate = candidates[0]
            parents = [p for p in scene["parents"] if p["parent_id"] == candidate["parent_id"]]
            siblings = [o for o in objects if candidate["parent_id"] and o["parent_id"] == candidate["parent_id"]]
            if ((candidate["parent_id"] is None and not candidate.get("attempt_id"))
                    or (len(parents) == 1 and len(siblings) == 1
                        and candidate["attempt_id"] == parents[0]["attempt_id"])):
                obj, automatic = candidate, True
    model = selected_assessment(scene, obj["object_id"] if obj else "", execution, live_quad=live_quad)
    model.update(scene_result_status(scene, execution))
    model["image_frames"] = image_frame_status(scene, execution)
    if not obj:
        model.update(frame="Wskaż tablicę, aby zobaczyć pochodzenie obrysu", correction_history="")
        if execution.get("historical_gt"):
            model["gt_note"] = "Zapis historyczny · tylko do odczytu"
    model.update(focus_id=obj["object_id"] if obj else "", focus_mode="AUTO" if automatic else "TEXT_EDIT" if obj and text_edit_id else "SELECTED" if obj else "NONE",
                 focus="Wskaż tablicę na obrazie")
    if obj:
        index = next(i for i, o in enumerate(objects, 1) if o["object_id"] == obj["object_id"])
        identity = model["mt_label"] or (f"Ręczna {index}" if not obj.get("parent_id") else "Wskazanie niepowiązane")
        model["focus"] = identity + (" · podgląd bez zaznaczenia" if automatic else
                                     " · edycja numeru GT" if text_edit_id else " · zaznaczona do edycji")
        if obj.get("error") and model.get("frame_state") == "MT_FP":
            model["summary_note"] = ("Decyzja FP uwzględniona; geometria i odczyt poza oceną"
                if scene.get("checked") and not core._content_issues(scene) and not scene.get("exclusion_reason") else
                "Wstępna decyzja FP — obraz wymaga sprawdzenia" if not scene.get("exclusion_reason") else model["summary_note"])
    elif allow_auto and scene and scene.get("roi_presence") == "NO_PLATES":
        n = core.counts(scene)["false_positive"]
        outcome = f"Negatywna klatka z alarmem · FP: {n}" if n else "TN · jedna oceniona klatka/ROI"
        if not scene.get("checked") or core._content_issues(scene):
            outcome = "Wstępna decyzja GT=0 · obraz wymaga sprawdzenia"
        model.update(focus="Brak tablic w analizowanym obszarze", frame="Negatywna obserwacja ROI",
            geometry="IoU / NMCE — nie dotyczy", reference="Jawna decyzja operatora: GT=0",
            gt="GT: potwierdzony brak tablic", mz="MZ: poza oceną — brak tablic GT",
            text="CER / Exact Match — nie dotyczy", outcome=outcome)
    return model


def scene_result_status(scene, execution):
    """Operator copy follows existing completeness, exclusion and execution rules."""
    from .mobile_canvas_resolution import missing_mt_geometry

    text, tone = "Poza oceną jakości — brak potwierdzonego wykonania MT", "muted"
    if (scene and execution.get("state") == "MT_EXECUTED"
            and scene.get("mt_invocation_id") == execution.get("mt_invocation_id")
            and scene.get("image_entry") == execution.get("image_entry")):
        issues = core._content_issues(scene)
        if scene.get("checked") and scene.get("exclusion_reason"):
            text, tone = "Obraz wyłączony z oceny jakości", "purple"
        elif not scene.get("checked"):
            text = "Wynik wstępny — jeszcze nie liczony do podsumowania"
        elif issues:
            text, tone = (("Sprawdzone — rozstrzygnij wskazanie MT" if missing_mt_geometry(scene)
                           else "Sprawdzone — " + "; ".join(issues)), "amber")
        else:
            text, tone = "Wynik uwzględniony w podsumowaniu sesji", "green"
        if execution.get("detections") == 0 and not scene.get("parents"):
            text += "\nMT nie wykrył tablicy. MZ nie uruchomiono — CER nie dotyczy"
    return dict(summary_note=text, summary_tone=tone)


def geometry_assessment(scene, obj, execution, *, live_quad=None):
    """One display-only pair/qualification for panel, badge and drag HUD.

    Seeded MT copies are not operator references. Live corners are an explicitly
    unsaved preview; they never change the scene or canonical metric inclusion.
    """
    if obj and obj.get('duplicate_of'):obj=dict(obj,error=True)
    display_obj = dict(obj, quad=live_quad) if obj and live_quad is not None else obj
    result = plate_review_state(scene, display_obj, execution)
    mt = result["mt_label"]
    result.update(geometry_title=f"GEOMETRIA {mt} ↔ REFERENCJA" if mt else "DOPASOWANIE RAMKI MT",
        geometry="IoU / NMCE — wskaż tablicę", reference="", geometry_status="NO_SELECTION",
        geometry_pair=None, polygon_iou_pct=None, nmce_pct=None)
    if not scene or not obj:
        return result
    if result["identity_reason"]:
        result.update(geometry_status="UNBOUND", geometry="IoU / NMCE — brak potwierdzonej pary",
                      reference=result["identity_reason"])
        return result
    parent = next((p for p in scene["parents"] if p["parent_id"] == result["bound_parent_id"]), None)
    if result["frame_state"] == "RAW_SUPPRESSED":
        result.update(geometry_status="RAW_SUPPRESSED", geometry="IoU: nie dotyczy · NMCE: nie dotyczy",
                      reference="RAW odrzucony przez NMS — poza końcowymi detekcjami")
    elif obj.get("error") or scene.get("exclusion_reason"):
        result.update(geometry_status="FP" if parent and obj.get("error") else "EXCLUDED",
                      geometry="IoU: nie dotyczy · NMCE: nie dotyczy",
                      reference=f"{mt}: brak referencji rzeczywistej tablicy (FP)" if parent and obj.get("error") else
                      "Ramka lub cały obraz wyłączone z oceny")
    elif not parent:
        result.update(geometry_status="UNLINKED", geometry="Brak obrysu MT → IoU / NMCE nie dotyczy",
                      reference="Ramka ręczna · bez powiązanej detekcji MT")
    else:
        if not parent.get("quad") and obj.get("identity_decision") and display_obj.get("quad"):
            pose = obj["identity_decision"].get("pose_status")
            result.update(geometry_status="NOT_AVAILABLE", geometry="IoU / NMCE — niedostępne (NOT_AVAILABLE)",
                reference=f"Ramka operatora ↔ {mt} · ASSISTED; " +
                ("źródłowy Pose niepoprawny" if pose == "INVALID" else "brak dowodu źródłowego Pose"))
            return result
        try:
            core.validate_mt_quad(display_obj.get("quad"), *scene["image_size"])
            core.validate_mt_quad(parent.get("quad"), *scene["image_size"])
        except (ValueError, TypeError, KeyError, OverflowError):
            result.update(geometry_status="INVALID_GEOMETRY", geometry=("IoU / NMCE — nieprawidłowy obrys źródłowy MT"
                          if result["frame_state"] == "MT_INVALID_POSE" else "IoU / NMCE — nieocenialny obrys operatora"),
                          reference=f"{mt}: brak poprawnych czterech narożników MT lub referencji")
            return result
        accepted = obj.get("geometry_state") in {"operator_corrected", "operator_confirmed_unchanged"}
        if not accepted and live_quad is None:
            result.update(geometry_status="UNCONFIRMED", geometry="Jeszcze nie oceniono geometrii",
                reference=f"Referencja {mt}: kopia MT niepotwierdzona. Sprawdź obrys i potwierdź obraz lub popraw narożniki.")
            return result
        metrics = core.compare_mt_pose_to_gt(parent["quad"], display_obj["quad"], *scene["image_size"])
        result.update(polygon_iou_pct=metrics["polygon_iou_pct"], nmce_pct=metrics["nmce_pct"],
            geometry_status="LIVE" if live_quad is not None else "ASSESSED",
            geometry_pair=dict(object_id=obj["object_id"], parent_id=parent["parent_id"], attempt_id=parent["attempt_id"],
                               mt_invocation_id=scene["mt_invocation_id"], image_entry=scene["image_entry"]),
            geometry=f"IoU {percent(metrics['polygon_iou_pct'])} · NMCE {percent(metrics['nmce_pct'])}",
            reference=(f"Referencja ↔ {mt}: korekta niezapisana · ASSISTED" if live_quad is not None else
                f"Referencja: oryginał {mt} zaakceptowany · ASSISTED" if result["frame_state"] == "MT_ORIGINAL" else
                f"Ramka operatora · powiązana z {mt} · ASSISTED"))
        if live_quad is not None:
            result.update(frame=f"Ramka operatora · korekta {mt} niezapisana", canvas_frame="MT · korekta niezapisana")
    return result


def selected_assessment(scene, object_id, execution, *, live_quad=None):
    """Small preview of ONE plate using the canonical alignment/pose functions.

    Draft values never contribute to aggregates. This function does not call
    summary(), persist GT, or change inference evidence.
    """
    from .mobile_human_review import align_plate_text
    from .mobile_pose_metrics import validate_mt_quad

    result = dict(state="NO_SELECTION", execution=execution, draft=True,
                  geometry="IoU / NMCE — wskaż tablicę", outcome=execution["detail"],
                  mz="MZ: —", gt="GT: —", text="Wybierz tablicę, aby porównać MZ z GT",
                  cer=None, edit_distance=None, gt_characters=None, exact=None,
                  polygon_iou_pct=None, nmce_pct=None)
    matches = [o for o in (scene or {}).get("objects", ()) if o["object_id"] == object_id]
    obj = matches[0] if len(matches) == 1 else None
    if obj and obj.get('duplicate_of'):obj=dict(obj,error=True)
    result.update(geometry_assessment(scene, obj, execution, live_quad=live_quad))
    if not scene:
        result["geometry"] = "IoU / NMCE — brak potwierdzonej pary obrysów"
        if execution["state"] != "MT_EXECUTED":
            result.update(state=execution["state"], text="CER / Exact Match — poza oceną",
                          mz="MZ: nie uruchomiono" if execution.get("mz_not_run") else "MZ: — brak potwierdzonego odczytu")
            if execution.get("historical_gt"):
                result["gt"] = "GT zapisane (historia): " + ", ".join(execution["historical_gt"])
        return result
    if (scene.get("mt_invocation_id") != execution["mt_invocation_id"]
            or scene.get("image_entry") != execution["image_entry"]):
        result.update(state="UNBOUND", text="Brak zgodnego powiązania tablicy z klatką")
        return result
    result["draft"] = not scene["checked"] or bool(core._content_issues(scene))
    if not obj:
        return result
    if result["identity_reason"] and execution["state"] == "MT_EXECUTED":
        result.update(state="UNBOUND", outcome=result["identity_reason"], text="CER / Exact Match — brak potwierdzonej pary")
        return result
    parent = next((p for p in scene["parents"] if p["parent_id"] == result["bound_parent_id"]), None)
    result["gt"] = "GT: " + (obj["gt_text"] or "—")
    result["mz"] = ("MZ: nie uruchomiono" if not parent or parent["mz_status"] == "NOT_RUN" else
                    "MZ: " + (parent["mz_prediction"] or "brak odczytanych znaków") if parent["mz_ran"] else
                    "MZ: nie potwierdzono wykonania")
    if execution["state"] != "MT_EXECUTED":
        result.update(state=execution["state"], text="CER / Exact Match — poza oceną")
        return result
    reason = ("RAW odrzucony przez NMS" if obj.get("post_nms_suppressed") else
              "Błędne wskazanie MT" if obj["error"] and parent else
              "Błąd operatora" if obj["error"] else scene.get("exclusion_reason"))
    if reason:
        result.update(state="EXCLUDED", outcome=reason, text="Poza oceną — " + reason)
        if obj["error"] and parent:
            result.update(outcome=f"{result['mt_label']} — błędne wykrycie (FP)",
                          mz="MZ: poza oceną (FP)", text="CER / Exact Match — nie dotyczy (FP)",
                          gt="GT (historia): " + obj["gt_text"] if obj["gt_text"] else "GT: nie dotyczy (FP)")
        return result
    try:
        validate_mt_quad(obj["quad"], *scene["image_size"])
    except (TypeError, ValueError):
        result.update(state="INVALID_GEOMETRY", outcome="Nieocenialny obrys operatora",
                      text="CER / Exact Match — poza oceną: nieocenialny obrys")
        return result
    result["outcome"] = ("Powiązana z MT (decyzja operatora)" if parent else
                         "Pominięcie MT (FN) · roboczo; do podsumowania po potwierdzeniu kompletnej oceny obrazu"
                         if result["draft"] else "Pominięcie MT (FN) · obraz potwierdzony")
    if parent and result["geometry_status"] == "UNCONFIRMED":
        result["outcome"] = f"{result['mt_label']}: powiązanie źródłowe; referencja niepotwierdzona"
    elif not parent and result.get("canvas_frame") == "Twoja ramka · jeszcze niepowiązana":
        result["outcome"] = "Powiązanie ręcznej ramki wymaga decyzji."
    if obj["text_exclusion_reason"]:
        result.update(state="TEXT_EXCLUDED", text="Numer wyłączony z oceny — " + obj["text_exclusion_reason"])
    elif not parent or parent["mz_status"] == "NOT_RUN":
        result.update(state="MZ_NOT_RUN", text="CER / Exact Match — nie dotyczy")
    elif not parent["mz_ran"]:
        result.update(state="MZ_UNKNOWN", text="CER / Exact Match — nieustalone wykonanie MZ")
    elif parent["mz_status"] == "AMBIGUOUS_SAMPLES":
        result.update(state="MZ_AMBIGUOUS", text="Sprzeczne odczyty MZ — poza oceną")
    elif not obj["text_confirmed"] or not obj["gt_text"]:
        result.update(state="GT_MISSING", text="Uzupełnij numer GT · CER —")
    else:
        a = align_plate_text(obj["gt_text"], parent["mz_prediction"])
        result.update(state="EXACT" if a.exact_match else "MISMATCH", cer=a.cer,
                      edit_distance=a.edit_distance, gt_characters=len(a.ground_truth), exact=a.exact_match)
        result["text"] = (f"{'Pełna zgodność' if a.exact_match else 'Numer niezgodny'} · CER {percent(a.cer * 100)}\n"
                          f"Operacje edycyjne / znaki GT: {a.edit_distance}/{len(a.ground_truth)}")
    return result


def current_summary(data, session):
    """Reject another source, session or revision, including missing identity."""
    return bool(data and session and data.get("session_id") == session.session_id
                and data.get("source_archive_sha256") == session.review.source_archive_sha256
                and data.get("review_revision") == session.review.review_revision)


def summary_caption(data, inventory):
    from .mobile_session_metrics import canvas_metric_projection
    d = canvas_metric_projection(data,data.get('mt_object_evaluation_v2'))
    cer = d["text_errors"] / d["gt_characters"] if d["gt_characters"] else None
    counter = f"Sprawdzono {inventory['inspected']}/{inventory['total']}"
    if inventory.get('operator_review_progress'):
        from .mobile_operator_progress import presentation
        counter = presentation(inventory['operator_review_progress'])['progress']
    return (f"{counter}  |  "
            f"MZ zgodne {d['exact_reads']}/{d['evaluated_reads']}  |  "
            f"CER {percent(cer * 100 if cer is not None else None)} ({d['text_errors']}/{d['gt_characters']})  |  "
            f"IoU {percent(d['polygon_iou_mean_pct'])} (n={d['geometry_n']})  |  "
            f"NMCE {percent(d['nmce_mean_pct'])} (n={d['geometry_n']})")
