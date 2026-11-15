"""Source-dependent working references, retired with their sole rejected FINAL.

No proximity inference and no migration on read. Operator geometry/text and
shared references are independent evidence and always survive model rejection.
"""
from __future__ import annotations
import copy

ORIGIN = 'reference_origin_v1'
RETIRED = 'retired_model_reference_v1'
SCHEMA = 'alpr.mobile.model_reference_origin.v1'
RETIRE_SCHEMA = 'alpr.mobile.retired_model_reference.v1'
ACCEPT_ACTIONS = {'operator_toggle_frame_presence', 'operator_confirm_associated_frames',
                  'operator_accept_shared_gt_reference'}


def mark_model(scene, plate, oid):
    obj = next(o for o in scene['objects'] if o['object_id'] == oid)
    plate[ORIGIN] = dict(schema=SCHEMA, kind='MODEL_DERIVED', source_object_id=oid,
        source_attempt_id=obj['attempt_id'], source_binding=scene['source_binding'])


def mark_independent(plate, reason):
    old = plate.get(ORIGIN, {})
    plate[ORIGIN] = dict(old, schema=SCHEMA, kind='OPERATOR_REFERENCE', evidence=reason)


def dependency_evidence(scene, state, pid):
    """Fail closed: prove the entire current history, not just missing corners."""
    from . import mobile_candidate_stacks as stacks, mobile_canvas_partial_review as partial
    p = state['plates'].get(pid)
    if not p: raise ValueError('REFERENCE_MISSING')
    oid = p['origin_object_id']; objects = partial._objects(scene); obj = objects.get(oid, {})
    parent = next((x for x in scene['parents'] if x['parent_id'] == obj.get('parent_id')), None)
    marker = p.get(ORIGIN)
    if not parent or not obj.get('parent_id'): raise ValueError('INDEPENDENT_MANUAL_REFERENCE')
    if marker is not None and (marker.get('schema') != SCHEMA or marker.get('kind') != 'MODEL_DERIVED'
            or marker.get('source_object_id') != oid or marker.get('source_attempt_id') != parent['attempt_id']
            or marker.get('source_binding') != scene['source_binding']):
        raise ValueError('INDEPENDENT_OR_UNPROVEN_ORIGIN')
    for key, link in state['links'].items():
        if key != oid and (link.get('plate_id') == pid or
                ((link.get('previous_link') or {}).get('plate_id') == pid and
                 partial.progress_decisions(scene).get(key) != partial.NOT_PLATE)):
            raise ValueError('REFERENCE_SHARED_WITH_OTHER_FINAL')
    history = scene.get(stacks.HISTORY, {}); entries = history.get('entries', [])
    index = history.get('head'); seen = set(); chain = []
    if type(index) is not int: raise ValueError('ORIGIN_HISTORY_MISSING')
    head = entries[index] if 0 <= index < len(entries) else {}
    if head.get('after', {}).get(stacks.KEY, {}).get('plates', {}).get(pid) != p:
        raise ValueError('REFERENCE_HISTORY_DIVERGED')
    expected = p
    while index is not None:
        if type(index) is not int or index in seen or not 0 <= index < len(entries):
            raise ValueError('REFERENCE_HISTORY_CHAIN_INVALID')
        seen.add(index); h = entries[index]
        if h.get('after', {}).get('source_binding') != scene['source_binding']:
            raise ValueError('REFERENCE_HISTORY_SOURCE_MISMATCH')
        before = h['before'].get(stacks.KEY, {}).get('plates', {}).get(pid)
        after = h['after'].get(stacks.KEY, {}).get('plates', {}).get(pid)
        if after != expected: raise ValueError('REFERENCE_HISTORY_CHAIN_DIVERGED')
        if after:
            if (after.get('origin_object_id') != oid or after.get('gt_text') or
                    after.get('text_exclusion_reason') or after.get('text_confirmed') or
                    after.get('geometry_state') not in {'unconfirmed', 'operator_confirmed_unchanged', 'operator_removed'} or
                    (after.get(ORIGIN) or {}).get('kind') == 'OPERATOR_REFERENCE'):
                raise ValueError('INDEPENDENT_OPERATOR_GEOMETRY_OR_GTZ')
            # Any saved relation to another source protects the physical GT,
            # including an association later undone in the current chain.
            if any(k != oid and v.get('plate_id') == pid for k,v in h['after'].get(stacks.KEY,{}).get('links',{}).items()):
                raise ValueError('REFERENCE_HAS_INDEPENDENT_ASSOCIATION')
        chain.append(dict(action=h['action'], review_revision_before=h['review_revision_before'], index=index))
        if before is None and after is not None:
            restored=h['before'].get(stacks.KEY,{}).get('links',{}).get(oid,{}).get(RETIRED)
            if (h['action']=='operator_candidate_no_plate' and restored and restored['plate']==after
                    and restored['evidence']['source_object_id']==oid
                    and restored['source_binding']==scene['source_binding']):
                return dict(restored['evidence'],restored_revision_before=h['review_revision_before'],chain=chain+restored['evidence']['chain'])
            if (h['action'] not in ACCEPT_ACTIONS or after.get('gt_quad') != parent.get('quad') or
                    h['after'].get(stacks.KEY, {}).get('links', {}).get(oid, {}).get('plate_id') != pid):
                raise ValueError('REFERENCE_NOT_CREATED_BY_MODEL_ACCEPTANCE')
            return dict(source_object_id=oid, source_attempt_id=parent['attempt_id'],
                source_binding=scene['source_binding'], created_revision_before=h['review_revision_before'], chain=chain)
        expected = before
        previous = h.get('previous')
        if previous is not None and (type(previous) is not int or previous >= index):
            raise ValueError('REFERENCE_HISTORY_ORDER_INVALID')
        index = previous
    raise ValueError('MODEL_ACCEPTANCE_ORIGIN_NOT_PROVEN')


def independent_ids(scene, state):
    """Physical references do not disappear merely because their MT is BAD.

    Unknown legacy provenance is protected. Only proven solely model-dependent
    records follow their source decision. This function never writes a record.
    """
    from .mobile_reference_disposition import withdrawn_ids
    withdrawn = withdrawn_ids(scene, state)
    ids = set()
    for pid,p in state['plates'].items():
        if pid in withdrawn or p['origin_object_id'] in state.get('removed_manual_ids', []): continue
        # An active positive already makes this ID active; avoid replaying its
        # history in the hot statistics/drag path.
        from .mobile_canvas_partial_review import progress_decisions, PLATE
        decisions = progress_decisions(scene)
        if any(link.get('plate_id') == pid and decisions.get(oid) == PLATE for oid,link in state['links'].items()): continue
        try: dependency_evidence(scene, state, pid)
        except ValueError: ids.add(pid)
    return ids


def retire(scene, state, oid):
    """Retire only a proved, solely model-created reference in the same write."""
    link = state['links'].get(oid, {})
    if link.get('status') != 'UNASSOCIATED_NO_PLATE': return None
    previous = link.get('previous_link') or {}; pid = previous.get('plate_id')
    if not pid or pid not in state['plates']: return None
    try: evidence = dependency_evidence(scene, state, pid)
    except ValueError: return None
    from . import mobile_candidate_stacks as stacks, mobile_canvas_partial_review as partial
    from . import mobile_reference_disposition as disposition
    plate = copy.deepcopy(state['plates'][pid])
    record = dict(schema=RETIRE_SCHEMA, reason='SOLE_MODEL_REFERENCE_REJECTED',
        plate_id=pid, plate=plate, evidence=evidence, source_binding=scene['source_binding'])
    doc = state.get(disposition.KEY)
    if doc and pid in doc['records']:
        record['legacy_disposition'] = copy.deepcopy(doc['records'].pop(pid))
        if not doc['records']: state.pop(disposition.KEY)
    link[RETIRED] = record
    state['plates'].pop(pid)
    if partial.KEY in scene:
        scene[partial.KEY]['pending_gtr_ids'] = [x for x in scene[partial.KEY].get('pending_gtr_ids', []) if x != pid]
    return record


def archived_reference(state, pid):
    records = [link[RETIRED] for link in state.get('links', {}).values()
               if isinstance(link.get(RETIRED), dict) and link[RETIRED].get('plate_id') == pid]
    return records[0] if len(records) == 1 else None


def validate_retired(scene,state):
    if not any(isinstance(link,dict) and RETIRED in link for link in state.get('links',{}).values()):return
    from .mobile_canvas_partial_review import _objects
    objects=_objects(scene)
    parents={p['attempt_id']:p for p in scene['parents']}
    seen=set()
    for oid,link in state.get('links',{}).items():
        r=link.get(RETIRED)
        if r is None:continue
        if not isinstance(r,dict):raise ValueError('Nieprawidłowa historia roboczej referencji MT.')
        p=r.get('plate',{});e=r.get('evidence',{});pid=r.get('plate_id')
        if (r.get('schema')!=RETIRE_SCHEMA or r.get('reason')!='SOLE_MODEL_REFERENCE_REJECTED'
                or not pid or pid in seen or pid in state['plates'] or p.get('plate_id')!=pid
                or p.get('origin_object_id')!=oid or p.get('gt_text') or p.get('text_exclusion_reason') or p.get('text_confirmed')
                or p.get('geometry_state') not in {'unconfirmed','operator_confirmed_unchanged','operator_removed'}
                or (p.get(ORIGIN) or {}).get('kind')=='OPERATOR_REFERENCE'
                or e.get('source_object_id')!=oid or e.get('source_attempt_id') not in parents
                or objects.get(oid,{}).get('attempt_id')!=e.get('source_attempt_id')
                or e.get('source_binding')!=scene['source_binding'] or r.get('source_binding')!=scene['source_binding']
                or link.get('status')!='UNASSOCIATED_NO_PLATE' or link.get('plate_id') is not None
                or (link.get('previous_link') or {}).get('plate_id')!=pid):
            raise ValueError('Historia roboczej GTR nie odpowiada odrzuconemu źródłu MT.')
        seen.add(pid)


def restore(state, link):
    record = link.get(RETIRED)
    if not record: return
    pid = record['plate_id']
    if pid in state['plates']: raise ValueError('Konflikt tożsamości przy cofnięciu odrzucenia MT.')
    state['plates'][pid] = copy.deepcopy(record['plate'])


def signature(scene):
    """Only physical content invalidates an existing whole-image inspection."""
    from .mobile_gt_link_contract import state_of
    from .mobile_canvas_operator_view import eligible_physical_plate_ids
    state = state_of(scene)
    # GTZ eligibility is separate from physical scene completeness: correcting
    # a number must not remove an already verified MT unit from its denominator.
    return {pid: {'gt_quad':p.get('gt_quad')}
            for pid in eligible_physical_plate_ids(scene,state) for p in [state['plates'][pid]]}
