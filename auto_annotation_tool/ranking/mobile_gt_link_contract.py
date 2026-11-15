"""Single canonical GTR per source MT. Read-only validation, never migration."""
from __future__ import annotations

OCCUPIED='MT ma już powiązaną GTR. Popraw istniejącą referencję przez D; nie przypisano drugiej GTR.'

def state_of(scene):
    from . import mobile_candidate_stacks as stacks
    return scene.get(stacks.KEY) or stacks._legacy(scene)

def validate(scene):
    from . import mobile_candidate_stacks as stacks
    if not scene:return
    state=state_of(scene)
    if (state.get('schema')!=stacks.SCHEMA or state.get('source_binding')!=scene.get('source_binding')
            or not isinstance(state.get('plates'),dict) or not isinstance(state.get('links'),dict)):
        raise ValueError('Nieprawidłowy kontrakt GTR lub źródło relacji.')
    if state.get('conflicts'):raise ValueError('Konflikt historycznych GTR — wymagane jawne uzgodnienie, bez migracji.')
    objects={o['object_id']:o for o in [*(scene.get('_stack_legacy') or {}).get('objects',[]),*scene['objects']]}
    parents={p['parent_id']:p for p in scene['parents']};claims={}
    for oid,link in state['links'].items():
        if not isinstance(link,dict):raise ValueError('Relacja MT → GTR musi być pojedynczym zapisem.')
        pid=link.get('plate_id')
        if pid is not None and (not isinstance(pid,str) or pid not in state['plates']):
            raise ValueError('MT może wskazywać tylko jedną istniejącą GTR (0..1).')
        if (link.get('status') not in {'ASSOCIATED_FOR_GEOMETRY','UNRESOLVED','UNASSOCIATED_NO_PLATE'}
                or bool(pid)!=(link.get('status')=='ASSOCIATED_FOR_GEOMETRY')):
            raise ValueError('Niespójna pojedyncza relacja MT → GTR.')
        obj=objects.get(oid)
        if obj is None:raise ValueError('Relacja GTR dotyczy nieistniejącego obiektu.')
        if obj.get('parent_id'):
            parent=parents.get(obj['parent_id'])
            if not parent or obj.get('attempt_id')!=parent['attempt_id']:
                raise ValueError('Relacja GTR nie ma jednoznacznego źródłowego MT.')
            key=(parent['attempt_id'],str(parent['detection_index']))
            if key in claims and claims[key]!=oid:raise ValueError('Jedna detekcja MT ma kilka zapisów relacji GTR.')
            claims[key]=oid

def require_unoccupied(scene,oid,target_pid=None):
    validate(scene)
    pid=state_of(scene)['links'].get(oid,{}).get('plate_id')
    if pid and pid!=target_pid:raise ValueError(OCCUPIED)

def validate_transition(before,after,*,history=False,merge_oid=None,cross_identity=False):
    validate(after)
    from .mobile_reference_disposition import validate_transition as validate_disposition
    validate_disposition(before,after,history=history)
    if not before or history:return
    validate(before)
    old=state_of(before);new=state_of(after)
    source={o['object_id'] for o in before['objects'] if o.get('parent_id')}
    for oid in source:
        a=old['links'].get(oid,{}).get('plate_id');b=new['links'].get(oid,{}).get('plate_id')
        if a and b and a!=b:
            # Existing explicit "same physical plate" reconciliation is not
            # assignment of a second reference. It removes the redundant ID,
            # retains the selected reference and records a reversible history.
            explicit_merge=(oid==merge_oid and a not in new['plates'] and b in old['plates']
                and new['links'][oid].get('provenance')=='operator_duplicate_final'
                and before['parents']==after['parents'])
            # Existing cross-call identity reconciliation only renames this
            # physical reference, without replacing its geometry/text or MT.
            fields=lambda p:{k:v for k,v in p.items() if k not in {'plate_id','scene_identity'}}
            explicit_cross=(cross_identity and a not in new['plates'] and b not in old['plates']
                and (new.get('cross_invocation_evidence') or {}).get('gt_id')==b
                and fields(old['plates'][a])==fields(new['plates'][b])
                and before['parents']==after['parents'])
            if not (explicit_merge or explicit_cross):raise ValueError(OCCUPIED)

def validate_review(session,review,*,history=False,merge=None):
    from .mobile_canvas_review import KEY
    for gid,annotation in review.invocation_annotations.items():
        scene=annotation.get(KEY)
        if not scene:continue
        old=session.review.invocation_annotations.get(gid,{}).get(KEY)
        # Historical snapshots inside the scene stay immutable evidence; only
        # the restored/current scene must satisfy the active relation contract.
        selected=merge and merge[0]==gid
        validate_transition(old,scene,history=history,merge_oid=merge[1] if selected else None,
            cross_identity=bool(selected and len(merge)>2 and merge[2]=='cross'))
