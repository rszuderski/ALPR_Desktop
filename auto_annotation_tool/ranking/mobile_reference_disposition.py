"""Read compatibility for historical explicit reference withdrawals.

The Canvas no longer offers a withdrawal workflow. Historical events and
records remain source-bound evidence, including records archived by cleanup.
"""
from __future__ import annotations
import hashlib,json

KEY='reference_dispositions_v1'
SCHEMA='alpr.mobile.reference_dispositions.v1'
WITHDRAWN='WITHDRAWN_FALSE_REFERENCE'
ACTION='operator_withdraw_false_physical_reference'
STALE='Weryfikacja zmieniła się od otwarcia zadania. Odczytaj obraz ponownie.'
INACTIVE='Referencja jest wycofana. Przywrócenie wymaga jawnego cofnięcia decyzji w historii.'

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def _state(scene,state=None):
    if state is not None:return state
    from . import mobile_candidate_stacks as stacks
    return scene.get(stacks.KEY) or stacks._legacy(scene)

def withdrawn_ids(scene,state=None,*,session_id=None):
    if state is None:
        from . import mobile_candidate_stacks as stacks
        state=scene.get(stacks.KEY) or {}
    doc=state.get(KEY)
    if doc is None:return set()
    if (not isinstance(doc,dict) or doc.get('schema')!=SCHEMA
            or doc.get('source_binding')!=scene.get('source_binding') or not isinstance(doc.get('records'),dict)):
        raise ValueError('Nieprawidłowy zapis dyspozycji fizycznej referencji.')
    for pid,record in doc['records'].items():
        plate=state['plates'].get(pid)
        if (not isinstance(record,dict) or not plate or record.get('plate_id')!=pid
                or record.get('status')!=WITHDRAWN or record.get('reason')!='FALSE_PHYSICAL_REFERENCE'
                or record.get('source_binding')!=scene.get('source_binding')
                or record.get('source_archive_sha256')!=scene.get('source_archive_sha256')
                or record.get('mt_invocation_id')!=scene.get('mt_invocation_id')
                or not record.get('session_id') or (session_id and record['session_id']!=session_id)
                or not isinstance(record.get('event_id'),str) or len(record['event_id'])!=32
                or type(record.get('review_revision_before')) is not int
                or record.get('previous_plate')!=plate or not isinstance(record.get('previous_links'),dict)):
            raise ValueError('Wycofana GTR nie odpowiada zapisanej tożsamości, historii lub źródłu.')
    return set(doc['records'])

def validate_audit(scene,*,session_id=None):
    """Persisted dispositions must have their original event and history entry."""
    if not scene:return
    from . import mobile_candidate_stacks as stacks
    state=scene.get(stacks.KEY) or {}
    from .mobile_reference_lifecycle import RETIRED,validate_retired
    validate_retired(scene,state)
    ids=withdrawn_ids(scene,state,session_id=session_id)
    records={pid:state[KEY]['records'][pid] for pid in ids}
    for link in state.get('links',{}).values():
        archived=link.get(RETIRED) or {}
        if archived.get('legacy_disposition'):
            r=archived['legacy_disposition']
            if (r.get('plate_id')!=archived['plate_id'] or r.get('previous_plate')!=archived['plate']
                    or r.get('source_binding')!=scene.get('source_binding')
                    or r.get('source_archive_sha256')!=scene.get('source_archive_sha256')
                    or r.get('mt_invocation_id')!=scene.get('mt_invocation_id')
                    or (session_id and r.get('session_id')!=session_id)):
                raise ValueError('Archiwalna dyspozycja GTR nie odpowiada źródłu.')
            records[archived['plate_id']]=r
    entries=scene.get(stacks.HISTORY,{}).get('entries',[])
    for pid,r in records.items():
        events=[e for e in scene.get('events',[]) if e.get('event_id')==r['event_id']]
        matches=[h for h in entries if h.get('action')==ACTION and h.get('review_revision_before')==r['review_revision_before']
                 and h.get('after',{}).get(stacks.KEY,{}).get(KEY,{}).get('records',{}).get(pid)==r]
        if (len(events)!=1 or len(matches)!=1 or events[0].get('action')!=ACTION
                or any(events[0].get(k)!=r[k] for k in ('plate_id','reason','session_id','source_binding','source_archive_sha256','mt_invocation_id','review_revision_before'))
                or matches[0].get('before',{}).get(stacks.KEY,{}).get('plates',{}).get(pid)!=r['previous_plate']):
            raise ValueError('Brak spójnego zdarzenia i historii wycofania fizycznej GTR.')

def require_active(scene,*,plate_id=None,object_id=None):
    state=_state(scene);withdrawn=withdrawn_ids(scene,state)
    if plate_id in withdrawn:raise ValueError(INACTIVE)
    if not withdrawn or plate_id in state['plates']:return
    if object_id:
        link=state['links'].get(object_id,{})
        if link.get('plate_id') in state['plates'] and link['plate_id'] not in withdrawn:return
        ids={link.get('plate_id'),(link.get('previous_link') or {}).get('plate_id')}
        ids.update(pid for pid,p in state['plates'].items() if object_id in (p['origin_object_id'],p.get('primary_object_id'),pid+':reference'))
        if ids & withdrawn:raise ValueError(INACTIVE)

def validate_transition(before,after,*,history=False):
    """Stored evidence cannot be edited/deleted/reactivated as a side effect."""
    from . import mobile_candidate_stacks as stacks
    if KEY not in (after.get(stacks.KEY) or {}) and KEY not in ((before or {}).get(stacks.KEY) or {}):return
    old=_state(before) if before else {};new=_state(after)
    was=withdrawn_ids(before,old) if before else set();now=withdrawn_ids(after,new)
    if history:return
    for pid in was:
        if pid not in now or old['plates'][pid]!=new['plates'].get(pid) or old[KEY]['records'][pid]!=new[KEY]['records'].get(pid):
            from .mobile_reference_lifecycle import archived_reference
            archived=archived_reference(new,pid)
            if not (pid not in new['plates'] and archived and archived.get('plate')==old['plates'][pid]
                    and archived.get('legacy_disposition')==old[KEY]['records'][pid]
                    and archived.get('source_binding')==after['source_binding']):
                raise ValueError(INACTIVE)
    from . import mobile_canvas_partial_review as partial
    decisions=partial._state(after)[1]['decisions']
    if any(link.get('plate_id') in now and decisions.get(oid)==partial.PLATE for oid,link in new['links'].items()):
        raise ValueError('Potwierdzone MT nie może wskazywać wycofanej referencji. Rozstrzygnij relację jawnie.')

def _disk_hash(session):
    from .mobile_human_review import MobileHumanReview
    try:raw=session.sidecar_path.read_bytes();data=json.loads(raw.decode('utf-8'))
    except (OSError,ValueError) as exc:raise ValueError('Brak zgodnego zapisu historii weryfikacji.') from exc
    if set(data)-set(MobileHumanReview.__dataclass_fields__):raise ValueError('Nieznane pola zapisu weryfikacji — zachowano dane bez konwersji.')
    loaded=MobileHumanReview.from_dict(data,source_sha256=session.review.source_archive_sha256,session_id=session.session_id)
    # Compare JSON values: in-memory geometry may use tuples, which JSON
    # legitimately reads back as lists. The raw hash still guards the write.
    if _digest(loaded.to_dict())!=_digest(session.review.to_dict()):raise ValueError(STALE)
    return hashlib.sha256(raw).hexdigest()

def _context(session,gid,expected_revision):
    from . import mobile_canvas_review as core,mobile_candidate_stacks as stacks,mobile_canvas_partial_review as partial
    from .mobile_canvas_resolution import verify_frame
    from .mobile_gt_link_contract import validate
    if expected_revision is not None and expected_revision!=session.review.review_revision:raise ValueError(STALE)
    disk=_disk_hash(session);session.verify_source(force=True);verify_frame(session,gid)
    scene=core._legacy_scene(session,gid);state=stacks.snapshot(session,gid,scene)
    partial.validate(session,gid,scene);validate(scene)
    if partial.progress(scene)['source_unknown'] or any(c['role']=='UNKNOWN' for c in state['candidates']):
        raise ValueError('Niepotwierdzone źródło lub konflikt GT. Nie wycofano referencji.')
    if scene.get('exclusion_reason'):raise ValueError('Najpierw rozstrzygnij wyłączenie całego obrazu.')
    ticket=dict(session_id=session.session_id,source_archive_sha256=session.review.source_archive_sha256,
        mt_invocation_id=gid,source_binding=scene['source_binding'],review_revision=session.review.review_revision,
        review_id=session.review.review_id,sidecar_path=str(session.sidecar_path.resolve()),sidecar_sha256=disk)
    return scene,state,ticket

def _history_proven(scene,state,pid):
    from . import mobile_candidate_stacks as stacks
    history=scene.get(stacks.HISTORY,{})
    entries=history.get('entries',[]);index=history.get('head');seen=set()
    if type(index) is not int or not 0<=index<len(entries):raise ValueError('Brak spójnej historii wskazanej referencji.')
    current=entries[index].get('after',{})
    if (current.get('source_binding')!=scene['source_binding']
            or current.get(stacks.KEY,{}).get('plates',{}).get(pid)!=state['plates'][pid]
            or current.get(stacks.KEY,{}).get('links')!=state['links']):
        raise ValueError('Stan referencji jest sprzeczny z historią. Nie zapisano decyzji.')
    while index is not None:
        if type(index) is not int or index in seen or not 0<=index<len(entries):raise ValueError('Konflikt łańcucha historii GTR.')
        seen.add(index);entry=entries[index]
        if entry.get('after',{}).get('source_binding')!=scene['source_binding']:raise ValueError('Historia dotyczy innego źródła.')
        previous=entry.get('previous')
        if previous is not None and (type(previous) is not int or previous>=index):raise ValueError('Konflikt kolejności historii GTR.')
        index=previous
    return _digest(history)
