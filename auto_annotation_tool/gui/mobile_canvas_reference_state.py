"""Read-only presentation of saved outlines and retained, removed references.

Neither a rejected MT nor a removed outline revokes a physical GT identity.
Canonical review tasks and scientific eligibility remain owned by ranking.
"""
from __future__ import annotations

def saved_outline_count(scene,state):
    """Count documented stored outlines independently of FINAL completeness."""
    from ..ranking import mobile_canvas_review as core
    from ..ranking.mobile_canvas_operator_view import saved_reference_ids
    if (not scene or not state or not scene.get('source_binding')
            or state.get('source_binding')!=scene['source_binding']):
        return None,'brak zgodnego źródła referencji'
    objects={o['object_id']:o for o in [*scene.get('objects',[]),*scene.get('_stack_legacy',{}).get('objects',[])]}
    ids=saved_reference_ids(scene,state)
    conflicts={c.get('plate_id') for c in state.get('conflicts',[])}
    for pid in ids:
        p=state['plates'][pid]
        owner=objects.get(p.get('origin_object_id'),{})
        manual_bad=owner.get('error') and (owner.get('kind')=='manual' or not owner.get('parent_id'))
        if (p.get('plate_id')!=pid or p.get('origin_object_id') not in objects
                or not p.get('confirmed') or pid in conflicts or manual_bad):
            return None,'niepotwierdzona tożsamość zapisanej GTR'
        try:core.validate_mt_quad(p['gt_quad'],*scene['image_size'])
        except (ValueError,TypeError,KeyError):return None,'nieprawidłowy zapis obrysu GTR'
    return len(ids),''

def missing_outline_task(scene,state,pid,number):
    """Restore a retained real outline; rejected model seeds are not tasks."""
    from ..ranking import mobile_candidate_stacks as stacks
    label=f'GTR {number}' if number is not None else 'GTR'
    p=state['plates'].get(pid)
    if not p:return dict(kind='REFERENCE_CONFLICT',text=f'{label}: brak zgodnego zapisu referencji',reference_state='MISSING_REFERENCE')
    canonical=scene.get(stacks.KEY,state)
    from ..ranking.mobile_canvas_operator_view import eligible_physical_plate_ids
    active=eligible_physical_plate_ids(scene,canonical)
    if pid in active and p['origin_object_id'] not in canonical.get('removed_manual_ids',[]):
        return dict(kind='GTR',text=f'Przywróć usunięty obrys {label} (D)',reference_state='RETAINED_PHYSICAL_IDENTITY')
    return dict(kind='REFERENCE_CONFLICT',text=f'{label}: brak zgodnej tożsamości referencji',reference_state='REMOVED_IDENTITY_UNRESOLVED')
