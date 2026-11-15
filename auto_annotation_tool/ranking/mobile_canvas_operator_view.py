"""Read-only projections of stored references and explicit dispositions."""


def reference_numbers(state):
    """Stable Canvas numbers, including gaps left by explicit reconciliation."""
    numbers=dict(state.get('reference_numbers',{}))
    next_number=max(numbers.values(),default=0)+1
    for pid,p in state['plates'].items():
        if p.get('gt_quad') and pid not in numbers:
            numbers[pid]=next_number;next_number+=1
    return numbers


def physical_plate_ids(scene, state):
    """Persisted physical identities, also when only GTZ has been supplied."""
    removed=set(state.get('removed_manual_ids',()))
    return {pid for pid,p in state['plates'].items()
            if p['origin_object_id'] not in removed}


def saved_reference_ids(scene, state):
    """Persisted physical GTR, including provisional corner accuracy."""
    return {pid for pid in eligible_physical_plate_ids(scene,state)
            if state['plates'][pid].get('gt_quad')}

def eligible_physical_plate_ids(scene,state):
    """Current physical identities; historical IDs remain in physical_plate_ids."""
    from .mobile_reference_disposition import withdrawn_ids
    return physical_plate_ids(scene,state)-withdrawn_ids(scene,state)


def collision_reference_ids(scene,state):
    # Every displayed, persisted GTR reserves its area, including one whose
    # corner accuracy is still provisional. An unreviewed model is not saved.
    return saved_reference_ids(scene,state)


def display_snapshot(scene, snapshot):
    """Pending decisions hide metric pairs, not the operator's saved GTR/GTZ."""
    saved=eligible_physical_plate_ids(scene,snapshot)
    result=dict(snapshot,active_plate_ids=sorted(set(snapshot['active_plate_ids'])|saved))
    result['candidates']=[]
    for candidate in snapshot['candidates']:
        row=dict(candidate);link=snapshot['links'].get(row['object_id'],{})
        if (row['role']=='FINAL' and link.get('status')=='ASSOCIATED_FOR_GEOMETRY'
                and link.get('plate_id') in saved):
            row.update(plate_id=link['plate_id'],association_status=link['status'])
        result['candidates'].append(row)
    return result


def linked_references(scene, snapshot, candidate):
    """One source FINAL -> distinct, explicitly linked, saved physical GTR IDs.

    v1 stores one physical ID per candidate; several candidates can share it.
    It cannot represent two different IDs on one FINAL. No spatial fallback.
    """
    unknown=dict(count=None,ids=[],reason='Brak wiarygodnej relacji GTR')
    if (not scene or not snapshot or not candidate or candidate.get('role')!='FINAL'
            or snapshot.get('source_binding')!=scene.get('source_binding')):return unknown
    matches=[p for p in scene['parents'] if p['attempt_id']==candidate['attempt_id']
             and int(p['detection_index'])==candidate['mt_detection_index']]
    if len(matches)!=1:return unknown
    link=snapshot['links'].get(candidate['object_id'])
    if not isinstance(link,dict):return unknown
    if link.get('status') not in ('ASSOCIATED_FOR_GEOMETRY','UNRESOLVED','UNASSOCIATED_NO_PLATE'):return unknown
    pid=link.get('plate_id')
    if pid is not None and not isinstance(pid,str):return unknown
    if pid and (pid not in snapshot['plates'] or any(c.get('plate_id')==pid for c in snapshot['conflicts'])):return unknown
    if bool(pid)!=(link['status']=='ASSOCIATED_FOR_GEOMETRY'):return unknown
    ids=sorted({pid} & saved_reference_ids(scene,snapshot)) if pid else []
    return dict(count=len(ids),ids=ids,reason='Jawne ID fizycznej GTR; niezależne od decyzji i TP')
