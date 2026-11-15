"""Explicit physical plates, separate from immutable detection records.

Never infer identity from overlap, OCR text or NOT_RUN. Legacy checked reviews
are read as-is; this contract is created only by an operator write.
"""
from __future__ import annotations
import copy
import hashlib
import json

KEY = 'physical_gt_v1'
IDENTITY_STATES = {'DISTINCT','DISTINCT_CONFIRMED','SAME_AS_PRIOR','UNRESOLVED_ACROSS_INVOCATIONS'}


def scope(session, gid):
    rows = session.mt_invocations[gid].records
    values = {(str(r.get('session_id')),str(r.get('scene_generation')),str(r.get('visual_epoch'))) for r in rows}
    if len(values) != 1 or any(v in ('','None') for v in next(iter(values))):
        # Legacy inputs can be reviewed per invocation, but cannot be joined
        # across calls without scene provenance.
        return 'invocation:'+session.review.source_archive_sha256+'/'+gid
    return '/'.join(next(iter(values)))


def _gt_id(scene, oid):
    token = scene['source_archive_sha256']+'|'+scene['mt_invocation_id']+'|'+oid
    return 'PGT:'+hashlib.sha256(token.encode()).hexdigest()[:24]


def sync_references(scene):
    registry = scene.get(KEY)
    if not registry:return
    objects = {o['object_id']:o for o in scene['objects']}
    for plate in registry['plates']:
        obj = objects.get(plate['reference_object_id'])
        if obj and obj.get('quad'):
            plate['quad'] = copy.deepcopy(obj['quad'])


def validate(session, gid, scene):
    from . import mobile_canvas_review as core
    reg = scene.get(KEY)
    if reg is None:return
    if reg.get('scope') != scope(session,gid) or type(reg.get('confirmed')) is not bool:
        raise ValueError('Niespójny kontrakt fizycznych tablic.')
    plates = reg.get('plates'); associations = reg.get('associations')
    if not isinstance(plates,list) or not isinstance(associations,dict):
        raise ValueError('Brak wykazu fizycznych tablic lub przypisań.')
    ids=[p.get('gt_id') for p in plates]
    if any(not isinstance(v,str) or not v.startswith('PGT:') for v in ids) or len(set(ids)) != len(ids):
        raise ValueError('Fizyczny identyfikator GT musi wystąpić raz w wywołaniu.')
    objects={o['object_id']:o for o in scene['objects']}
    for p in plates:
        if p.get('scene_identity') not in IDENTITY_STATES:
            raise ValueError('Nieznany stan tożsamości fizycznej tablicy.')
        if p.get('reference_object_id') not in objects:
            raise ValueError('Brak ramki referencyjnej fizycznej tablicy.')
        core.validate_mt_quad(p.get('quad'),*scene['image_size'])
    for oid,gt in associations.items():
        if oid not in objects or gt not in ids or objects[oid].get('post_nms_suppressed'):
            raise ValueError('Niespójne przypisanie wskazania do fizycznej tablicy.')
    for obj in scene['objects']:
        target=obj.get('duplicate_of')
        if target:
            if (target not in objects or target==obj['object_id'] or objects[target].get('duplicate_of')
                    or not obj.get('parent_id') or not objects[target].get('parent_id')
                    or associations.get(obj['object_id']) != associations.get(target)
                    or associations.get(target) not in ids):
                raise ValueError('Niespójne przypisanie końcowego duplikatu do GT.')
    assigned=[associations[o['object_id']] for o in scene['objects'] if o['object_id'] in associations
              and not o['error'] and not o.get('duplicate_of') and o.get('parent_id')]
    if len(assigned) != len(set(assigned)):
        raise ValueError('Jedna fizyczna tablica może mieć najwyżej jedno TP w wywołaniu.')


def tally(scene):
    reg=scene.get(KEY)
    if not reg:return None
    active=[o for o in scene['objects'] if not o.get('post_nms_suppressed')]
    matched={reg['associations'][o['object_id']] for o in active if o.get('parent_id') and
             not o['error'] and not o.get('duplicate_of') and o['object_id'] in reg['associations']}
    return dict(expected=len(reg['plates']),linked=len(matched),missed=len(reg['plates'])-len(matched),
        false_positive=sum(bool(o.get('parent_id')) and bool(o['error'] or o.get('duplicate_of')) for o in active))


def issues(scene):
    reg=scene.get(KEY)
    if not reg:return []
    unresolved=[o for o in scene['objects'] if not o['error'] and not o.get('duplicate_of') and
                not o.get('post_nms_suppressed') and o['object_id'] not in reg['associations']]
    if 'candidate_stacks_v1' in scene:
        result=[]  # Canvas links are validated per candidate; checked is separate.
    else:
        result=['Potwierdź fizyczne tablice po zmianie ramek.'] if unresolved or not reg['confirmed'] else []
    if any(p.get('scene_identity')=='UNRESOLVED_ACROSS_INVOCATIONS' for p in reg['plates']):
        result.append('Ustal, czy tablica jest tą samą GT co w innym wywołaniu tej sceny.')
    return result


def require_final(session,gid,objects):
    """For explicit duplicate decisions, require final IDs from this invocation."""
    from .mobile_mt_postdedup import canvas_post_nms_proof,_trace_index,_source_integer,final_ids
    proof=canvas_post_nms_proof(session,gid)
    if proof:
        final=set(final_ids(proof))
    else:
        group=session.mt_invocations[gid];rows=group.records;stat=session.path.stat()
        key=tuple(_source_integer(rows[0].get(k)) for k in ('source_sequence','source_timestamp_nanos','scene_generation'))
        matches=_trace_index(str(session.path),session.bundle.source_archive_sha256,stat.st_size,stat.st_mtime_ns).get(key,[])
        if len(matches)!=1:raise ValueError('Nie potwierdzono końcowych wyników po NMS.')
        c=matches[0].get('counters',{})
        if (c.get('mt_runs_this_frame')!=1 or c.get('plate_detections_raw')!=len(rows)
                or c.get('plate_detections_after_dedup')!=len(rows) or c.get('plate_detections_suppressed')!=0):
            raise ValueError('Nie ustalono, które wskazania są końcowe po NMS.')
        final={r['id'] for r in rows}
    if any(o.get('attempt_id') not in final or o.get('post_nms_suppressed') for o in objects):
        raise ValueError('Surowy kandydat nie może być końcowym duplikatem FP.')


def mark_duplicate(session,gid,duplicate_id,reference_id):
    """One explicit same-physical-plate decision inside one final invocation."""
    from . import mobile_canvas_review as core
    s=core.scene(session,gid)
    duplicate,reference=core._object(s,duplicate_id),core._object(s,reference_id)
    if duplicate_id==reference_id or reference['error'] or reference.get('duplicate_of') or s.get('_post_nms_pending'):
        raise ValueError('Wybierz odrębne końcowe wskazanie i właściwą tablicę.')
    require_final(session,gid,[duplicate,reference])
    from .mobile_canvas_resolution import model_location,verify_frame
    verify_frame(session,gid)
    for obj in (duplicate,reference):model_location(session,gid,s,obj)
    from . import mobile_candidate_stacks as stacks
    state=stacks._state(s)
    target=state['links'].get(reference_id,{}).get('plate_id')
    old=state['links'].get(duplicate_id,{}).get('plate_id')
    if not target:
        core.validate_mt_quad(reference['quad'],*s['image_size'])
        target=stacks._id(s,reference_id);plate=stacks._plate(s,reference,target)
        plate.update(confirmed=True,geometry_state='operator_confirmed_unchanged')
        state['plates'][target]=plate
        state['links'][reference_id]=dict(plate_id=target,status='ASSOCIATED_FOR_GEOMETRY',provenance='operator_explicit_duplicate_reference')
    if old and old!=target:
        if any(link.get('plate_id')==old for oid,link in state['links'].items() if oid!=duplicate_id):
            raise ValueError('Najpierw rozstrzygnij pozostałe podramki tej GT.')
        state['plates'].pop(old)
    state['links'][duplicate_id]=dict(plate_id=target,status='ASSOCIATED_FOR_GEOMETRY',provenance='operator_duplicate_final')
    return stacks._write(session,gid,core._legacy_scene(session,gid),state,'operator_same_physical_plate_duplicate_final',duplicate_id)


def share_across_invocations(session,gid,object_id,other_gid,other_object_id):
    """Explicit physical identity across calls, without suppressing a detection."""
    from . import mobile_canvas_review as core
    if gid==other_gid:raise ValueError('Dla jednego wywołania użyj dopasowania 1:1.')
    s=core.scene(session,gid);other=core.scene(session,other_gid)
    from . import mobile_candidate_stacks as stacks
    s=stacks.adopt_canvas_write(session,gid,s)
    if scope(session,gid)!=scope(session,other_gid):raise ValueError('To inna fizyczna scena; brak podstaw do wspólnego GT.')
    obj=core._object(s,object_id);reference=core._object(other,other_object_id)
    if any(o['error'] or o.get('duplicate_of') or o.get('post_nms_suppressed') for o in (obj,reference)):
        raise ValueError('Wybierz rzeczywiste tablice, nie FP ani RAW.')
    from .mobile_canvas_partial_review import stack_is_verified
    other_gt=(other.get(KEY) or {}).get('associations',{}).get(other_object_id)
    if (KEY not in other or (not other[KEY]['confirmed']
            and not stack_is_verified(other,plate_id=other_gt))):
        raise ValueError('Najpierw potwierdź tablicę w poprzednim wywołaniu.')
    target=other[KEY]['associations'].get(other_object_id)
    if not target:raise ValueError('Brak fizycznej tablicy odniesienia.')
    if any(p['gt_id']==target and p.get('scene_identity')=='UNRESOLVED_ACROSS_INVOCATIONS' for p in other[KEY]['plates']):
        raise ValueError('Najpierw rozstrzygnij tożsamość tablicy odniesienia.')
    if 'candidate_stacks_v1' in s:
        from . import mobile_candidate_stacks as stacks
        state=stacks._state(s);old=state['links'].get(object_id,{}).get('plate_id')
        if not old:raise ValueError('Najpierw ustanów referencję GT na Canvas.')
        if target in state['plates'] and target!=old:raise ValueError('Ta tablica ma już przypisanie w bieżącym wywołaniu.')
        plate=state['plates'].pop(old);plate.update(plate_id=target,scene_identity='SAME_AS_PRIOR')
        state['plates'][target]=plate
        for link in state['links'].values():
            if link.get('plate_id')==old:link['plate_id']=target
        state['cross_invocation_evidence']=dict(source_invocation=other_gid,source_object=other_object_id,gt_id=target)
        return stacks._write(session,gid,core._legacy_scene(session,gid),state,'operator_same_physical_plate_across_invocations',object_id)


def declare_distinct(session,gid,object_id):
    from . import mobile_canvas_review as core
    from . import mobile_candidate_stacks as stacks
    s=stacks.adopt_canvas_write(session,gid,core.scene(session,gid));reg=s.get(KEY)
    if not reg:raise ValueError('Najpierw oznacz i potwierdź tablice w tym obrazie.')
    gt=reg['associations'].get(object_id)
    plate=next((p for p in reg['plates'] if p['gt_id']==gt),None)
    if not plate or plate.get('scene_identity')!='UNRESOLVED_ACROSS_INVOCATIONS':
        raise ValueError('Ta tablica nie wymaga rozróżnienia między wywołaniami.')
    if 'candidate_stacks_v1' in s:
        from . import mobile_candidate_stacks as stacks
        state=stacks._state(s);state['plates'][gt]['scene_identity']='DISTINCT_CONFIRMED'
        return stacks._write(session,gid,core._legacy_scene(session,gid),state,'operator_distinct_physical_plate_across_invocations',object_id,keep_inspected=True)


def remove_reference(scene,object_id):
    reg=scene.get(KEY)
    if not reg:return
    removed={p['gt_id'] for p in reg['plates'] if p['reference_object_id']==object_id}
    reg['plates']=[p for p in reg['plates'] if p['gt_id'] not in removed]
    reg['associations']={oid:gt for oid,gt in reg['associations'].items() if oid!=object_id and gt not in removed}
    for obj in scene['objects']:
        if obj.get('duplicate_of')==object_id:obj.pop('duplicate_of')
    reg['confirmed']=False


def link_manual(scene,manual_id,target_id):
    reg=scene.get(KEY)
    if not reg:return
    manual_gt=reg['associations'].pop(manual_id,None)
    target_gt=reg['associations'].get(target_id)
    chosen=manual_gt or target_gt
    if not chosen:return
    if manual_gt and target_gt and manual_gt!=target_gt:
        reg['plates']=[p for p in reg['plates'] if p['gt_id']!=target_gt]
        reg['associations']={oid:chosen if gt==target_gt else gt for oid,gt in reg['associations'].items()}
    reg['associations'][target_id]=chosen
    for plate in reg['plates']:
        if plate['gt_id']==chosen:plate['reference_object_id']=target_id


def inspect_export(data):
    """Validate physical IDs and one-to-one scoring without touching legacy exports."""
    if not data.get('physical_gt_contract'):return True,''
    if data['physical_gt_contract']!='alpr.mobile.physical_gt.v1':return False,'Nieznany kontrakt fizycznych GT.'
    inventory={};seen=set()
    for scene in data.get('scenes',()):
        reg=scene.get('physical_gt')
        if not reg:continue
        if not isinstance(reg,dict) or not reg.get('scope') or (reg.get('decision_source')!='canvas_objects' and not reg.get('confirmed')):
            return False,'Nie potwierdzono fizycznych tablic.'
        ids=[p.get('gt_id') for p in reg.get('plates',())]
        if any(p.get('scene_identity') not in IDENTITY_STATES or p.get('scene_identity')=='UNRESOLVED_ACROSS_INVOCATIONS' for p in reg.get('plates',())):
            return False,'Nieustalona tożsamość fizycznej tablicy między wywołaniami.'
        if len(ids)!=len(set(ids)) or any(not isinstance(v,str) or not v.startswith('PGT:') for v in ids):
            return False,'Powielone fizyczne GT.'
        if scene.get('status')=='CHECKED':
            if scene.get('counts',{}).get('expected')!=len(ids):return False,'Liczba tablic nie odpowiada fizycznym GT.'
            for gt in ids:inventory.setdefault((reg['scope'],gt),set()).add(scene['mt_invocation_id'])
        rows=[r for r in data.get('rows',()) if r.get('mt_invocation_id')==scene.get('mt_invocation_id')]
        by_id={r.get('object_id'):r for r in rows}
        for plate in reg['plates']:
            reference=by_id.get(plate.get('reference_object_id'))
            if not reference or reference.get('operator_quad')!=plate.get('quad'):
                return False,'Referencja fizycznej GT nie odpowiada zapisanej ramce.'
        for row in rows:
            gt=row.get('physical_gt_id')
            if row.get('object_kind')!='physical_reference' and gt!=reg.get('associations',{}).get(row.get('object_id')):
                return False,'Wiersz detekcji ma inne przypisanie fizycznej GT.'
            if row.get('post_nms_suppressed_raw'):
                if gt:return False,'Odrzucony RAW ma przypisaną fizyczną GT.'
                continue
            if not row.get('included'):continue
            if row.get('outcome') in {'LINKED_MT','MISSED_MT'}:
                key=(scene['mt_invocation_id'],gt)
                if gt not in ids or key in seen:return False,'Naruszone dopasowanie 1:1 do fizycznej GT.'
                seen.add(key)
            if row.get('false_positive_subtype')=='duplicate_final' and (row.get('outcome')!='FALSE_POSITIVE_MT' or gt not in ids):
                return False,'Niespójny końcowy duplikat.'
    expected=[dict(physical_scene_id=s,gt_id=g,invocations=sorted(inv)) for (s,g),inv in inventory.items()]
    if (data.get('physical_plates')!=expected or data.get('physical_gt_scene_n')!=len(expected)
            or data.get('physical_gt_known_scene_n')!=len(expected)):
        return False,'Niespójny scenowy mianownik fizycznych GT.'
    return True,''
