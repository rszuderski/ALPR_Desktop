"""Read-only F4 FINAL detector scoring, approved by the operator on 2026-10-26.

The adapter retains existing review decisions. The pure scorer reuses the MT
experiment's greedy matcher, with an already-FINAL population and no new filter.
Pose/OCR, historical Canvas summaries and READY gates remain separate contracts.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import threading
from weakref import WeakKeyDictionary

from .mt_eval_metrics import METRIC_PROTOCOL, _bbox, _scene_result, box_iou

SCHEMA = 'alpr.mobile.mt_detection_evaluation.v1'
POLICY_ID = 'f4_mt_final_bbox_iou50_greedy_v1'
POLICY = dict(id=POLICY_ID, geometry='continuous_source_bbox_xyxy', match_iou=0.5,
    matching='confidence_desc_index_asc_iou_desc_gt_id_asc_greedy_1to1',
    population='source_proven_FINAL_no_refilter', unit='mt_invocation_id+source_frame+ROI',
    gt_mode='ASSISTED', partial_roi='NOT_EVALUABLE',
    approval=dict(date='2026-10-26', actor='operator',
        statement='Zatwierdzam opisany kontrakt dla F4',
        document='docs/experiments/F4_MT_FINAL_BBOX_IOU50_GREEDY_V1_20261026.md',
        approved_candidate_sha256='8f0e33be0854efbedf5525d00413bfae93b1f547b075a055f59c1ec6b9e1999a'))
_FROZEN_POLICY_JSON = json.dumps(POLICY, sort_keys=True, ensure_ascii=False,separators=(',', ':'))
POLICY_SHA256 = hashlib.sha256(_FROZEN_POLICY_JSON.encode('utf-8')).hexdigest()
_MATCHER_PROTOCOL_JSON = json.dumps(dict(METRIC_PROTOCOL,operational_conf=-1,match_iou=.5,plate_class_ids=[0]))
_cache = WeakKeyDictionary()
_verified_frames = WeakKeyDictionary()
_cache_lock = threading.RLock()

def _approved(policy):
    try:return json.dumps(policy,sort_keys=True,ensure_ascii=False,separators=(',', ':'))==_FROZEN_POLICY_JSON
    except (TypeError,ValueError):return False


def rates(tp, fp, fn, *, n_gt=None, n_final_mt=None):
    """Ratios from integer counts; F1 does not depend on defined P/R ratios."""
    keys = ('precision', 'recall', 'f1')
    invalid = any(type(v) is not int or v < 0 for v in (tp, fp, fn))
    if not invalid:
        invalid |= n_gt is not None and (type(n_gt) is not int or n_gt != tp+fn)
        invalid |= n_final_mt is not None and (type(n_final_mt) is not int or n_final_mt != tp+fp)
    if invalid:
        return dict.fromkeys(keys, None) | {'rate_reasons':dict.fromkeys(keys, 'INCONSISTENT_OR_UNAVAILABLE_COUNTS')}
    numerators, denominators = (tp, tp, 2*tp), (tp+fp, tp+fn, 2*tp+fp+fn)
    return {k:(n/d if d else None) for k,n,d in zip(keys,numerators,denominators)} | {
        'rate_reasons':{k:'ZERO_DENOMINATOR' for k,d in zip(keys,denominators) if not d}}

def format_rate(number):
    if (isinstance(number,bool) or not isinstance(number,(int,float))
            or not math.isfinite(number) or not 0<=number<=1):return '—'
    return f'{100*number:.1f}%'.replace('.', ',')


def score_unit(unit, *, policy=POLICY):
    """Pure normalized-input contract; absent/unapproved policy cannot yield TP."""
    identity = {k:copy.deepcopy(unit.get(k)) for k in (
        'session_id','source_archive_sha256','review_revision','unit_id','source_frame_id',
        'mt_invocation_id','source_binding','image_entry','image_size','roi')}
    result = identity | dict(schema=SCHEMA, matching_policy_id=None,
        matching_policy_provenance=None, gt_mode='ASSISTED', n_gt=None, n_final_mt=None,
        n_raw_suppressed=unit.get('n_raw_suppressed'), tp=None, fp=None, fn=None,
        matched_pairs=[], final_outcomes=[], excluded_gt_ids=[], n_qualified=0, n_total=1,
        status='NOT_EVALUABLE', unavailable_reasons=list(unit.get('unavailable_reasons', [])))
    def unavailable(reason=None, status='NOT_EVALUABLE'):
        if reason and reason not in result['unavailable_reasons']:result['unavailable_reasons'].append(reason)
        result.update(status=status)
        result.update(rates(None,None,None))
        return result
    if not _approved(policy):
        return unavailable('NO_APPROVED_MATCHING_POLICY', 'BLOCKED_METHODOLOGY')
    result.update(matching_policy_id=POLICY_ID,
                  matching_policy_provenance=dict(sha256=POLICY_SHA256, **json.loads(_FROZEN_POLICY_JSON)))
    if unit.get('execution_status') != 'COMPLETED':return unavailable('MT_NOT_COMPLETED')
    if unit.get('source_verified') is not True:return unavailable('SOURCE_NOT_VERIFIED')
    if unit.get('complete') is not True:return unavailable('REVIEW_INCOMPLETE')
    if result['unavailable_reasons']:return unavailable()
    if type(unit.get('n_raw_suppressed')) is not int or unit['n_raw_suppressed']<0:
        return unavailable('RAW_POPULATION_UNPROVEN')
    try:
        width,height=unit['image_size'];roi=_bbox(unit['roi'])
        if (type(width) not in (int,float) or type(height) not in (int,float)
                or not math.isfinite(width) or not math.isfinite(height) or width<=0 or height<=0
                or roi is None or not (0<=roi[0]<roi[2]<=width and 0<=roi[1]<roi[3]<=height)):
            return unavailable('INVALID_SOURCE_ROI')
        if width!=int(width) or height!=int(height):return unavailable('INVALID_SOURCE_SIZE')
        width,height=int(width),int(height)
        if not all(identity.get(k) for k in ('session_id','source_archive_sha256','unit_id',
                                            'source_frame_id','mt_invocation_id','source_binding')):
            return unavailable('SOURCE_IDENTITY_INCOMPLETE')
        plates=[];gt_ids=set()
        for gt in unit['gt']:
            pid=gt['gt_id'];box=_bbox(gt['bbox'])
            if not isinstance(pid,str) or not pid or pid in gt_ids:return unavailable('GT_ID_CONFLICT')
            gt_ids.add(pid)
            if not gt.get('confirmed') or box is None:return unavailable('GT_GEOMETRY_UNCONFIRMED')
            if not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):return unavailable('GT_SOURCE_GEOMETRY_INVALID')
            if box[2]<=roi[0] or box[0]>=roi[2] or box[3]<=roi[1] or box[1]>=roi[3]:
                result['excluded_gt_ids'].append(pid);continue
            if not (roi[0]<=box[0]<box[2]<=roi[2] and roi[1]<=box[1]<box[3]<=roi[3]):
                return unavailable('GT_PARTIAL_ROI_COVERAGE')
            plates.append(dict(plate_id=pid,bbox=box,quad=gt.get('quad'),pose_eligible=False,pose_exclusion=None))
        predictions=[];indices=set();attempts=set();forced=[];finals=unit['finals']
        for candidate in finals:
            aid,index,confidence=candidate['attempt_id'],candidate['index'],candidate['confidence']
            box=_bbox(candidate['bbox']);decision=candidate['decision']
            if not isinstance(aid,str) or not aid or aid in attempts:return unavailable('ATTEMPT_ID_CONFLICT')
            if type(index) is not int or index<0 or index in indices:return unavailable('FINAL_INDEX_CONFLICT')
            if (isinstance(confidence,bool) or not isinstance(confidence,(int,float))
                    or not math.isfinite(confidence) or not 0<=confidence<=1):return unavailable('FINAL_CONFIDENCE_UNAVAILABLE')
            if box is None or not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):
                return unavailable('FINAL_SOURCE_BBOX_UNAVAILABLE')
            if decision not in ('PLATE','NOT_PLATE'):return unavailable('FINAL_DECISION_UNRESOLVED')
            attempts.add(aid);indices.add(index)
            if decision=='NOT_PLATE':forced.append(dict(attempt_id=aid,index=index,outcome='FP',reason='OPERATOR_NOT_PLATE'))
            else:predictions.append(dict(index=index,confidence=confidence,class_id=0,bbox=box))
        protocol=json.loads(_MATCHER_PROTOCOL_JSON)
        # FINAL membership has already been proved against the phone trace.
        # -1 makes every legitimate [0,1] confidence eligible, including zero.
        protocol['operational_conf']=-1
        scored=_scene_result(dict(domain='F4',width=width,height=height,predictions=predictions),
            dict(domain='F4',image=dict(oriented_size=[width,height]),plates=plates,
                 sha256=unit['source_archive_sha256'],file=unit['image_entry']),protocol)
        by_index={c['index']:c['attempt_id'] for c in finals}
        pairs=[dict(gt_id=p['plate_id'],attempt_id=by_index[p['prediction_index']],
                    mt_detection_index=p['prediction_index'],bbox_iou=p['iou']) for p in scored['matched']]
        tp=len(pairs);fp=len(finals)-tp;fn=len(plates)-tp
        outcomes=[dict(attempt_id=by_index[p['index']],index=p['index'],outcome=p['box_status'],
                       gt_id=p['matched_plate_id'],reason=None if p['box_status']=='TP' else 'UNMATCHED_FINAL')
                  for p in scored['predictions']]+forced
        occupied={p['gt_id'] for p in pairs}
        prediction_boxes={p['index']:p['bbox'] for p in predictions}
        for outcome in outcomes:
            if outcome['reason']=='UNMATCHED_FINAL':
                outcome['reason']=('DUPLICATE_FINAL' if any(p['plate_id'] in occupied and
                    box_iou(prediction_boxes[outcome['index']],p['bbox'])>=protocol['match_iou'] for p in plates)
                    else 'LOCALIZATION_CRITERION_NOT_MET')
        result.update(n_gt=len(plates),n_final_mt=len(finals),tp=tp,fp=fp,fn=fn,
                      matched_pairs=pairs,final_outcomes=sorted(outcomes,key=lambda c:c['index']),
                      unmatched_gt_ids=scored['unmatched_gt_ids'],n_qualified=1,status='MEASURED')
        result.update(rates(tp,fp,fn,n_gt=len(plates),n_final_mt=len(finals)))
        return result
    except (KeyError,TypeError,ValueError,OverflowError) as e:
        return unavailable('INVALID_UNIT_INPUT: '+str(e))


def _input(session, gid):
    from . import mobile_canvas_review as core, mobile_candidate_stacks as stacks
    from . import mobile_canvas_partial_review as partial
    from .mobile_pose_metrics import validate_mt_quad
    sha=session.review.source_archive_sha256
    unit=dict(session_id=session.session_id,source_archive_sha256=sha,
        review_revision=session.review.review_revision,unit_id=sha+':'+gid,mt_invocation_id=gid,
        source_frame_id=None,source_binding=None,image_entry=None,image_size=None,roi=None,
        gt=[],finals=[],n_raw_suppressed=None,source_verified=False,complete=False,
        execution_status='NOT_COMPLETED',unavailable_reasons=[])
    group=session.mt_invocations.get(gid)
    if not group or not group.executed or group.cancelled or group.execution_failed:return unit
    unit['execution_status']='COMPLETED'
    try:
        session.verify_source()
        scene=core.scene(session,gid,include_history=False);v=stacks.snapshot(session,gid,scene)
        stat=session.path.stat()
        frame_key=(sha,stat.st_size,stat.st_mtime_ns,gid,scene['image_entry'],tuple(scene['image_size']))
        with _cache_lock:verified=frame_key in _verified_frames.get(session,set())
        if not verified:
            from .mobile_canvas_resolution import verify_frame
            verify_frame(session,gid)  # bounded JPEG read and actual bitmap dimensions
            with _cache_lock:_verified_frames.setdefault(session,set()).add(frame_key)
        unit.update(source_verified=True,source_frame_id=sha+':'+scene['image_entry'],
            source_binding=scene['source_binding'],image_entry=scene['image_entry'],
            image_size=scene['image_size'],roi=scene['roi'])
        progress=partial.progress(scene)
        unit['complete']=progress['complete']
        if scene.get('exclusion_reason'):unit['unavailable_reasons'].append('SCENE_EXCLUDED')
        if v['conflicts']:unit['unavailable_reasons'].append('GT_IDENTITY_CONFLICT')
        if any(c['role']=='UNKNOWN' for c in v['candidates']):unit['unavailable_reasons'].append('FINAL_POPULATION_UNPROVEN')
        unit['n_raw_suppressed']=(None if any(c['role']=='UNKNOWN' for c in v['candidates'])
                                else sum(c['role']=='SUPPRESSED' for c in v['candidates']))
        for pid in v['active_plate_ids']:
            p=v['plates'][pid];quad=p.get('gt_quad');box=None
            if quad:
                try:
                    quad=validate_mt_quad(quad,*scene['image_size'])
                    box=[min(q[0] for q in quad),min(q[1] for q in quad),max(q[0] for q in quad),max(q[1] for q in quad)]
                except ValueError:pass
            geometry_confirmed=(p['geometry_state']!='unconfirmed'
                or (scene['checked'] and partial.KEY not in scene))
            unit['gt'].append(dict(gt_id=pid,bbox=box,quad=quad,
                                  confirmed=bool(p['confirmed'] and geometry_confirmed)))
        for c in v['candidates']:
            if c['role']!='FINAL':continue
            row=session.attempts[c['attempt_id']]
            confidence=row.get('mt_confidence')
            if confidence is None or str(confidence).strip()=='':confidence=row.get('plate_confidence')
            try:confidence=float(confidence)
            except (TypeError,ValueError,OverflowError):confidence=None
            try:
                box=[float(row[k]) for k in ('plate_left','plate_top','plate_right','plate_bottom')]
                if (row.get('source_frame_coordinate_space')!='mt_source_bitmap_xy'
                        or row.get('source_frame_origin')!='pipeline_bitmap_before_roi'
                        or [float(row[k]) for k in ('source_frame_width_px','source_frame_height_px')]!=list(scene['image_size'])):
                    box=None
            except (KeyError,TypeError,ValueError,OverflowError):box=None
            unit['finals'].append(dict(attempt_id=c['attempt_id'],index=c['mt_detection_index'],
                confidence=confidence,bbox=box,decision=progress['object_decisions'].get(c['object_id'],'UNRESOLVED')))
    except (ValueError,KeyError,TypeError,OSError) as e:
        unit['unavailable_reasons'].append('SOURCE_OR_REVIEW_UNAVAILABLE: '+str(e))
    return unit


def evaluate_unit(session, gid, *, policy=POLICY, cache=True):
    """Source-bound, revision-keyed projection; never persists a derived value."""
    key=None
    try:
        stat=session.path.stat()
        key=(id(session.review),session.review.review_revision,session.review.source_archive_sha256,
             stat.st_size,stat.st_mtime_ns,gid)
        if cache and _approved(policy):
            with _cache_lock:
                if key in _cache.get(session,{}):return copy.deepcopy(_cache[session][key])
    except (AttributeError,OSError,TypeError):pass
    result=score_unit(_input(session,gid),policy=policy)
    if cache and key is not None and _approved(policy):
        try:
            with _cache_lock:
                old=_cache.setdefault(session,{})
                if old and next(iter(old))[:3]!=key[:3]:old.clear()
                old[key]=copy.deepcopy(result)
        except TypeError:pass
    return result


def aggregate_units(units):
    """Deduplicate exact aliases; conflicting aliases cannot choose a winner."""
    unique={};conflicts=set()
    for unit in units:
        key=unit['unit_id']
        if key in unique and unique[key]!=unit:conflicts.add(key)
        else:unique[key]=unit
    for key in conflicts:
        invalid=copy.deepcopy(unique[key])
        invalid.update(status='NOT_EVALUABLE',n_qualified=0,tp=None,fp=None,fn=None,n_gt=None,n_final_mt=None,
                       matched_pairs=[],final_outcomes=[],unavailable_reasons=['CONFLICTING_UNIT_ALIASES'])
        invalid.update(rates(None,None,None));unique[key]=invalid
    measured=[u for key,u in unique.items() if u['status']=='MEASURED' and key not in conflicts]
    cohort={(u.get('session_id'),u.get('source_archive_sha256'),u.get('review_revision'),
             u.get('matching_policy_id'),u.get('gt_mode')) for u in unique.values()}
    mixed=len(cohort)>1
    valid=all(rates(u['tp'],u['fp'],u['fn'],n_gt=u['n_gt'],n_final_mt=u['n_final_mt'])['rate_reasons']
              .get('precision')!='INCONSISTENT_OR_UNAVAILABLE_COUNTS' for u in measured)
    if not valid or mixed:measured=[]
    result=dict(schema=SCHEMA,matching_policy_id=POLICY_ID,matching_policy_sha256=POLICY_SHA256,
        gt_mode='ASSISTED',unit='mt_invocation_id+source_frame+ROI',n_qualified=len(measured),
        n_total=len(unique),units=list(unique.values()),alias_conflicts=sorted(conflicts),
        status='COMPLETE' if measured and len(measured)==len(unique) else 'PARTIAL' if measured else 'NOT_EVALUABLE',
        unavailable_reasons=(['CONFLICTING_UNIT_ALIASES'] if conflicts else [])+
                            ([] if valid else ['INCONSISTENT_UNIT_COUNTS'])+
                            (['MIXED_SESSION_COHORT'] if mixed else []))
    for key in ('tp','fp','fn','n_gt','n_final_mt'):
        result[key]=sum(u[key] for u in measured) if measured else None
    result['n_raw_suppressed']=sum(u['n_raw_suppressed'] or 0 for u in measured) if measured else None
    result.update(rates(result['tp'],result['fp'],result['fn'],n_gt=result['n_gt'],n_final_mt=result['n_final_mt']))
    if not measured:result['unavailable_reasons'].append('NO_QUALIFIED_UNITS')
    return result


def evaluate_session(session):
    identity=dict(session_id=session.session_id,source_archive_sha256=session.review.source_archive_sha256,
                  review_revision=session.review.review_revision)
    epoch=id(session.review)
    result=aggregate_units(evaluate_unit(session,gid) for gid in session.mt_invocations)
    if (epoch!=id(session.review) or any(getattr(session.review,k)!=v for k,v in identity.items() if k!='session_id')
            or any(any(u.get(k)!=v for k,v in identity.items()) for u in result['units'])):
        result.update(status='STALE_REVIEW',n_qualified=0,tp=None,fp=None,fn=None,n_gt=None,n_final_mt=None)
        result['unavailable_reasons'].append('REVIEW_REVISION_CHANGED')
        result.update(rates(None,None,None))
    result.update(**identity,matching_policy=json.loads(_FROZEN_POLICY_JSON))
    return result


def reason_text(result):
    codes=result.get('unavailable_reasons') or []
    first=codes[0].split(':',1)[0] if codes else ''
    return {'GT_GEOMETRY_UNCONFIRMED':'Potwierdź obrys GT',
        'REVIEW_INCOMPLETE':'Rozstrzygnij ramki', 'GT_PARTIAL_ROI_COVERAGE':'Nieustalona granica ROI',
        'MT_NOT_COMPLETED':'MT nie zostało zakończone', 'SCENE_EXCLUDED':'Obraz poza oceną',
        'FINAL_POPULATION_UNPROVEN':'Nieustalone końcowe wskazania MT',
        'SOURCE_OR_REVIEW_UNAVAILABLE':'Brak zgodnego źródła lub referencji',
        'NO_QUALIFIED_UNITS':'Brak ocenialnych wywołań',
        'FINAL_SOURCE_BBOX_UNAVAILABLE':'Brak wiarygodnego BBOX MT',
        'FINAL_CONFIDENCE_UNAVAILABLE':'Brak pewności MT',
        'NO_APPROVED_MATCHING_POLICY':'Brak zatwierdzonej metody'}.get(first,'Brak kompletnych danych MT')


def export(session, directory):
    """Additive export in an exclusively reserved child; existing files untouched."""
    session.verify_source(force=True)
    revision=session.review.review_revision
    data=evaluate_session(session)
    if data['review_revision']!=revision or session.review.review_revision!=revision or data['status']=='STALE_REVIEW':
        raise ValueError('Zmieniono rewizję podczas obliczania MT; uruchom eksport ponownie')
    root=Path(directory)
    root.mkdir(parents=True,exist_ok=True)
    for index in range(1,10000):
        dest=root/f'mt_detection_v1_{index:03d}'
        try:dest.mkdir()
        except FileExistsError:continue
        break
    else:raise ValueError('Brak wolnego katalogu eksportu MT')
    paths=[]
    def write(name,content):
        p=dest/name
        with p.open('x',encoding='utf-8',newline='') as f:f.write(content)
        paths.append(p)
    write('mt_detection_evaluation.json',json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
    write('matching_policy.json',json.dumps(json.loads(_FROZEN_POLICY_JSON),ensure_ascii=False,indent=2))
    import io
    stream=io.StringIO(newline='')
    columns=('session_id','source_archive_sha256','review_revision','unit_id','source_frame_id','mt_invocation_id',
             'matching_policy_id','gt_mode','n_gt','n_final_mt','n_raw_suppressed','tp','fp','fn',
             'precision','recall','f1','n_qualified','n_total','status','unavailable_reasons','rate_reasons')
    writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
    for unit in data['units']:
        row={k:unit.get(k) for k in columns};row['unavailable_reasons']='; '.join(row['unavailable_reasons'])
        row['rate_reasons']=json.dumps(unit['rate_reasons'],ensure_ascii=False,sort_keys=True)
        writer.writerow(row)
    write('mt_detection_units.csv',stream.getvalue())
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=('unit_id','gt_id','attempt_id','mt_detection_index','bbox_iou'))
    writer.writeheader()
    for unit in data['units']:
        for pair in unit['matched_pairs']:writer.writerow(dict(unit_id=unit['unit_id'],**pair))
    write('mt_detection_pairs.csv',stream.getvalue())
    write('report.md',f"# MT — ASSISTED\n\nPolityka: {POLICY_ID}.\n\n"
        f"Stan: {data['status']}; ocenione {data['n_qualified']}/{data['n_total']} wywołań.\n\n"
        f"Precision: {format_rate(data['precision'])}; Recall: {format_rate(data['recall'])}; F1: {format_rate(data['f1'])}.\n\n"
        "Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN).\n\n"
        "Surowe liczniki, wartości ilorazów, pary i przyczyny null zawiera JSON. "
        "RAW poza populacją FINAL; wyniki częściowe dotyczą wyłącznie ocenionych wywołań.\n")
    session.verify_source(force=True)
    if session.review.review_revision!=revision:raise ValueError('Zmieniono rewizję podczas eksportu MT')
    manifest=dict(schema=SCHEMA,session_id=session.session_id,source_archive_sha256=data['source_archive_sha256'],
        review_revision=revision,matching_policy_sha256=POLICY_SHA256,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    write('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    return paths
