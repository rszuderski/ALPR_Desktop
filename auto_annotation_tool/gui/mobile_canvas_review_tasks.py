"""Read-only operator work units. Scientific readiness/scoring stay canonical."""
from __future__ import annotations
import copy
from ..ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
from ..ranking.mobile_canvas_operator_view import reference_numbers
from .mobile_canvas_spatial_reference import project


def pending_review_units(scene,state,*,issues=()):
    """Pair one pending FINAL with its reference; never merge distinct FINALs.

    `reasons` retain every canonical pending record. A reference reason is
    attached once, whereas each unresolved FINAL retains its own source ID.
    Whole-image inspection, missing text and evidence are separate steps.
    """
    if not scene:return dict(units=[],count=0,steps=[dict(kind='SOURCE',text='Brak pełnego obrazu')],next_action='Brak pełnego obrazu',needs_review=False)
    if scene.get('exclusion_reason'):return dict(units=[],count=0,steps=[],next_action='Obraz poza oceną',needs_review=False)
    progress=partial.progress(scene);numbers=reference_numbers(state)
    finals={c['object_id']:c for c in state['candidates'] if c['role']=='FINAL'}
    pending_gt=set(progress['pending_gtr_ids']);used=set();units=[];steps=[]
    objects={o['object_id']:o for o in scene['objects']}
    def target(c):
        pid=c.get('plate_id');rel=c.get('spatial_relation',{});canonical=c.get('canonical_link',{})
        if not pid and canonical.get('status')=='ASSOCIATED_FOR_GEOMETRY':
            # A removed/moved outline does not erase the explicit physical ID.
            pid=canonical.get('plate_id')
        if pid not in state['plates'] or rel.get('conflict'):return None
        origin=objects.get(state['plates'][pid]['origin_object_id'],{})
        if c['object_id'] in origin.get('separate_from_mt',()):return None
        if canonical.get('plate_id') not in (None,pid):return None
        return pid
    for oid in progress['pending_ids']:
        c=finals.get(oid)
        if not c:continue
        pid=target(c);reasons=['pending_mt:'+oid]
        if pid in pending_gt and pid not in used:reasons.append('pending_gtr:'+pid);used.add(pid)
        text=f"Zweryfikuj tablicę GTR {numbers[pid]}" if pid in numbers else f"Zweryfikuj wskazanie MT {c['mt_detection_index']+1}"
        if pid in state['plates'] and not state['plates'][pid]['gt_quad']:text=f"Narysuj obrys GTR {numbers.get(pid,'—')} (D)"
        units.append(dict(id=f"{scene['mt_invocation_id']}:FINAL:{c['attempt_id']}:GT:{pid or '-'}",kind='PAIR' if pid else 'MT',physical_gt_id=pid,final_mt_id=c['attempt_id'],object_id=oid,reasons=reasons,text=text))
        if pid in state['plates'] and state['plates'][pid]['geometry_state']=='operator_removed' and not state['plates'][pid]['gt_quad']:
            from .mobile_canvas_reference_state import missing_outline_task
            units[-1].update(missing_outline_task(scene,state,pid,numbers.get(pid)))
    for pid in sorted(pending_gt-used):
        p=state['plates'].get(pid)
        c=next((c for c in finals.values() if target(c)==pid),None)
        text=f"Zweryfikuj tablicę GTR {numbers.get(pid,'—')}"
        detail={}
        if not p or not p['gt_quad']:
            from .mobile_canvas_reference_state import missing_outline_task
            detail=missing_outline_task(scene,state,pid,numbers.get(pid))
        units.append(dict(id=f"{scene['mt_invocation_id']}:GT:{pid}",kind='PAIR' if c else 'GTR',physical_gt_id=pid,final_mt_id=c['attempt_id'] if c else None,object_id=c['object_id'] if c else p['origin_object_id'] if p else '',reasons=['pending_gtr:'+pid],text=text))
        units[-1].update(detail)
    if progress['source_unknown']:
        return dict(units=[],count=0,steps=[],next_action='Źródło nieocenialne · N/A',needs_review=False,canonical_progress=progress,diagnostic_issues=list(issues))
    text_issues=[v for v in issues if 'GT numeru' in v or 'GTZ' in v]
    if text_issues:steps.append(dict(kind='GTZ',text='Uzupełnij GTZ przy tablicy',reasons=text_issues))
    # Keep all existing blockers auditable, without counting them as plates.
    other=[v for v in issues if v not in text_issues and not any(x in v for x in ('Rozstrzygnij wskazanie','Potwierdź obecność','Zatwierdź nową','nie został oznaczony jako sprawdzony'))]
    if other and not units and not progress['source_unknown'] and not progress['empty_eligible']:
        steps.append(dict(kind='DATA',text='Sprawdź brakujące dane w szczegółach',reasons=other))
    action=units[0]['text'] if units else steps[0]['text'] if steps else 'Brak referencji · N/A' if progress['state']=='NO_REFERENCE_EVIDENCE' else 'Obraz zweryfikowany'
    return dict(units=units,count=len(units),steps=steps,next_action=action,needs_review=bool(units or steps),canonical_progress=progress,diagnostic_issues=list(issues))


def for_scene(session,gid,scene=None,*,issues=None):
    from ..ranking.mobile_scene_frame_review import unit_for_mt, mt_review_issues
    source=unit_for_mt(session,gid)
    if source is not None:
        reasons=mt_review_issues(session,gid)
        return dict(units=[],count=len(reasons),steps=[dict(kind='DATA',text=v) for v in reasons],
                    next_action=reasons[0] if reasons else 'Ocena z pełnej sceny; miary według dostępnego dowodu',
                    needs_review=bool(reasons),diagnostic_issues=[])
    scene=scene if scene is not None else core.scene(session,gid,include_history=False)
    state=stacks.snapshot(session,gid,scene)
    return pending_review_units(scene,project(session,gid,scene,state),issues=core.approval_scene_issues(scene) if issues is None else issues)


def inventory(session,canonical,*,cache=None):
    """Decorate GUI rows; preserve status, cells, approval gates and old totals."""
    entries={}
    if cache is not None:
        review=session.review
        try:
            stat=session.path.stat();source_stamp=(stat.st_size,stat.st_mtime_ns)
        except (OSError,AttributeError):source_stamp=None
        inputs=(session,review.subjects,review.attempt_annotations,review.sample_annotations)
        context=(source_stamp,review.review_id,review.review_mode,review.source_archive_sha256)
        previous=cache.get('inputs',())
        if (cache.get('context')!=context or len(previous)!=len(inputs)
                or any(a is not b for a,b in zip(previous,inputs))):
            cache.clear();cache.update(context=context,inputs=inputs,scenes={})
        entries=cache['scenes']
    result=dict(canonical,groups={});total=0;steps=0
    for gid,item in canonical['groups'].items():
        annotation=session.review.invocation_annotations.get(gid) if cache is not None else None
        issues=item.get('approval_issues',item['issues'])
        saved=entries.get(gid)
        # Canvas writes replace one invocation annotation, retaining all other
        # annotations. Hold the input itself to prevent Python ID reuse. Cache
        # view tasks only, never scores, decisions or history.
        reusable=bool(annotation and core.KEY in annotation and item['scene'])
        if (reusable and saved and saved[0] is annotation and saved[1]==tuple(issues)
                and saved[3]==item['scene']):
            task=saved[2]
        else:
            input_scene=copy.deepcopy(item['scene']) if reusable else None
            task=for_scene(session,gid,item['scene'],issues=issues) if item['scene'] else pending_review_units(None,None)
            if reusable:entries[gid]=(annotation,tuple(issues),task,input_scene)
            else:entries.pop(gid,None)
        result['groups'][gid]=dict(item,review_tasks=task)
        total+=task['count'];steps+=len(task['steps'])
    result.update(task_decisions=total,task_steps=steps)
    return result


def image_action(rows):
    if not rows:return 'Poza oceną','canvas_excluded'
    if all(r['status']=='EXCLUDED' for r in rows):return 'Poza oceną','canvas_excluded'
    if all(r.get('frame_review',{}).get('complete') or r['status']=='READY' for r in rows):
        if all(r['status'] in {'NO_REFERENCE_EVIDENCE','SOURCE_NOT_EVALUABLE'} for r in rows):return 'N/A','canvas_excluded'
        return 'Zweryfikowany','canvas_ready'
    tasks=[r['review_tasks'] for r in rows];count=sum(t['count'] for t in tasks)
    kinds={s['kind'] for t in tasks for s in t['steps']}
    if 'SOURCE' in kinds:return 'Sprawdź źródło','canvas_pending'
    if count:return f'Decyzje: {count}','canvas_pending'
    if 'GTZ' in kinds:return 'Uzupełnij GTZ','canvas_pending'
    if 'DATA' in kinds:return 'Uzupełnij dane','canvas_pending'
    return 'Zweryfikowany','canvas_ready'
