"""FINAL adapter over the existing native Android reader and matching engine.

No alternate Android schema, post-hoc OCR identity, NMS replay or best-read
selection. Every planned exposure remains in the result, including missing data.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import csv
import io
import json
import math
import tempfile
import zipfile

from PIL import Image
from . import final180_reference as ref
from . import final180_markers as markers
from . import photo_reference as photo
from .mobile_f5a_assisted import STAMP
from .mt_eval_metrics import METRIC_PROTOCOL, _scene_result
from .mobile_mt_detection_evaluation import rates
from ..registration_text import normalize_registration
from ..gt_pack import _atomic_write_json


def _bbox(points):
    return [min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]


def _inside(box, region):
    return all(math.isfinite(v) for v in box) and region[0]<=box[0]<box[2]<=region[2] and region[1]<=box[1]<box[3]<=region[3]


def configuration(bundle, expected, *, layout_version=None):
    coll=bundle.collection_session;execution=coll.get('execution') or {};stages=execution.get('stages') or {}
    observed=dict(execution=deepcopy(execution),model_refs=deepcopy(coll.get('model_refs') or bundle.model_refs),
        app_build=deepcopy(coll.get('app_build')),device=deepcopy(bundle.report_payload.get('device')),
        runtime_execution_evidence=deepcopy(bundle.report_payload.get('runtime_execution_evidence')))
    errors=[]
    f2_alias=(layout_version in markers.BOUND_LAYOUTS and execution.get('variant')=='R0_FINAL' and coll.get('variant')=='R0_FINAL'
        and execution.get('capture_profile',coll.get('capture_profile'))=='f6_final180_v1' and execution.get('roi_budget_policy')=='r0_full_frame'
        and (stages.get('mp') or {}).get('enabled') is False)
    if (execution.get('variant')!='R0' and not f2_alias) or execution.get('roi_budget_policy') not in ('r0_full_frame','full_frame'):
        errors.append('REQUIRES_RECORDED_R0')
    if (stages.get('mp') or {}).get('enabled') is not False:errors.append('MP_MUST_BE_OFF')
    for role in ('mt','mz'):
        stage=stages.get(role) or {}
        if stage.get('runtime')!='tflite' or stage.get('precision')!='fp32' or stage.get('enabled') is not True:
            errors.append(role.upper()+'_REQUIRES_TFLITE_FP32')
        for key,value in (expected.get(role) or {}).items():
            if stage.get(key)!=value:errors.append(role.upper()+'_PIN_MISMATCH:'+key)
    build=coll.get('app_build') or {};apk=build.get('apk_sha256') or (build.get('apk_identity') or {}).get('base_apk_sha256')
    if expected.get('apk_sha256') and expected['apk_sha256']!=apk:errors.append('APK_PIN_MISMATCH')
    pins_complete=bool(expected.get('apk_sha256') and all((expected.get(role) or {}).get('model_id') for role in ('mt','mz')))
    return dict(status='PASS' if not errors else 'N/A',errors=errors,observed=observed,
        pins_status='FROZEN' if pins_complete else 'NOT_VERIFIED_CONFIGURATION_PINS_MISSING')


def _order(frame):
    elapsed=frame.get('captured_elapsed_nanos');sequence=(frame.get('source_stamp') or {}).get('source_sequence')
    if type(elapsed) is not int or type(sequence) is not int:return None
    return elapsed,sequence,str(frame['frame_key'])


def select_first(exposure_id, frames):
    identified=[f for f in frames if f.get('exposure_id')==exposure_id and f.get('identity_status')=='PASS']
    started=[f for f in identified if f.get('mt_invocation_ids')]
    if any(_order(f) is None for f in started):return None,'SOURCE_ORDER_UNPROVEN'
    if not started:return None,'NOT_RUN_OBSERVED_NO_STARTED_MT' if identified else 'EXPOSURE_UNASSIGNED'
    first=min(started,key=_order)
    if any(f.get('f6_lifecycle_required') for f in started):
        if any(type(f.get('mt_started_elapsed_nanos')) is not int for f in started):return None,'ACTUAL_MT_START_UNPROVEN'
        backend_first=min(started,key=lambda f:(f['mt_started_elapsed_nanos'],f.get('mt_start_event_sequence',0)))
        if backend_first['frame_key']!=first['frame_key']:return None,'FIRST_SOURCE_ORDER_VS_MT_START_CONFLICT'
    return first,''


def first_mt_scope(frames, f6, recording):
    """Persist the native reader's bounds on the START inventory, not completeness of FINAL."""
    if not f6.get('available'):return dict(available=False,status='LEGACY_UNCHANGED')
    summary=f6.get('summary') or {};calls=f6.get('mt_invocations') or {};reasons=[]
    for key in ('unlocated_loss_possible','sequence_gaps','malformed_records','binding_gaps'):
        if f6.get(key):reasons.append(key)
    for key in ('pending_loss_unknown','unlocated_loss_possible'):
        if summary.get(key) is not False:reasons.append(key)
    for key in ('lifecycle_loss_count','projection_error_count'):
        if summary.get(key)!=0:reasons.append(key)
    for key in ('dropped_sample_count','dropped_telemetry_count','integrity_loss_count','scene_evidence_pending_loss_unknown'):
        if recording.get(key):reasons.append(key)
    persisted=f6.get('events_persisted')
    if type(persisted) is not int or any(summary.get(k)!=persisted for k in ('events_expected','events_written','events_persisted')):
        reasons.append('LIFECYCLE_INVENTORY_UNPROVEN')
    if summary.get('time_domain')!='ANDROID_ELAPSED_REALTIME' or summary.get('time_unit')!='ns':reasons.append('LIFECYCLE_CLOCK_UNPROVEN')
    if summary.get('mt_started_count')!=len(calls) or f6.get('event_counts',{}).get('MT_STARTED',0)!=len(calls):reasons.append('MT_START_INVENTORY_UNPROVEN')
    observed=[gid for frame in frames for gid in frame.get('mt_invocation_ids',[])]
    by_key={f['frame_key']:f for f in frames}
    if sorted(observed)!=sorted(calls) or any(c['frame_key'] not in by_key for c in calls.values()):reasons.append('MT_SOURCE_FRAME_MISSING')
    return dict(available=True,status='UNBOUNDED' if reasons else 'BOUNDED',reasons=sorted(set(reasons)),
        mt_invocation_ids=sorted(calls),events_persisted=persisted,
        basis='NATIVE_VALIDATED_LIFECYCLE_AND_SOURCE_BINDINGS; PARTIAL_FINAL_DOES_NOT_ERASE_START')


def first_mt_risk(candidate, frames, scope=None):
    """One predicate for analysis/preflight. Never assign an unidentified frame.

    A real START (including CANCELLED) is harmless to this candidate only when
    BOTH its source and backend start are strictly later in proven domains.
    """
    unknown=[f for f in frames if f.get('mt_invocation_ids') and f.get('identity_status')!='PASS'
        and f.get('geometry_reason')!='TECHNICAL_TRANSITION_FRAME']
    result=dict(status='PASS',reason='NO_UNIDENTIFIED_MT',unknown_mt_ids=[gid for f in unknown for gid in f['mt_invocation_ids']],
        later_mt_ids=[],blocking_mt_ids=[],comparisons=[])
    def blocked(reason):return dict(result,status='N/A',reason=reason)
    if candidate is None:return blocked('NO_SELECTED_MT_INPUT')
    if scope and scope.get('available') is False:return dict(result,reason='LEGACY_UNCHANGED')
    if scope and scope.get('status')!='BOUNDED':return dict(blocked('UNBOUNDED_MT_SOURCE_LOSS'),scope_reasons=scope.get('reasons',[]))
    if scope and sorted(gid for f in frames for gid in f.get('mt_invocation_ids',[]))!=scope.get('mt_invocation_ids'):
        return blocked('MT_START_INVENTORY_MISMATCH')
    if not unknown:return result
    if not scope:return blocked('MT_START_INVENTORY_UNPROVEN')

    def point(frame):
        stamp=frame.get('source_stamp') or {}
        if (set(STAMP)-stamp.keys() or stamp.get('source_timestamp_domain')!='CAMERAX_SENSOR'
                or any(type(stamp.get(k)) is not int or stamp[k]<0 for k in STAMP if k!='source_timestamp_domain')):return None
        if (frame.get('android_clock_domain')!='ANDROID_ELAPSED_REALTIME' or frame.get('android_unit')!='ns'
                or frame.get('capture_clock')!='AS_RECORDED_CAPTURED_ELAPSED_NANOS_NO_CROSS_CLOCK_CONVERSION'
                or type(frame.get('captured_elapsed_nanos')) is not int or frame['captured_elapsed_nanos']<0):return None
        calls=frame.get('mt_lifecycle') or [];starts=[]
        if sorted(c.get('mt_invocation_id','') for c in calls)!=sorted(frame.get('mt_invocation_ids',[])) or not calls:return None
        for call in calls:
            start=(call.get('events') or {}).get('MT_STARTED') or {}
            if (call.get('source_stamp')!=stamp or call.get('frame_key')!=frame.get('frame_key')
                    or start.get('time_domain')!='ANDROID_ELAPSED_REALTIME'
                    or any(type(start.get(k)) is not int or start[k]<0 for k in ('event_sequence','elapsed_nanos'))):return None
            starts.append((start['event_sequence'],start['elapsed_nanos']))
        event,elapsed=min(starts)
        if (elapsed!=min(s[1] for s in starts) or event!=frame.get('mt_start_event_sequence')
                or elapsed!=frame.get('mt_started_elapsed_nanos')):return None
        return (frame['captured_elapsed_nanos'],stamp['source_sequence'],stamp['source_timestamp_nanos'],elapsed,event)

    chosen=point(candidate)
    for frame in unknown:
        other=point(frame)
        same_scope=(all(candidate.get(k) and candidate[k]==frame.get(k) for k in ('session_id','source_archive_sha256'))
            and (candidate.get('source_stamp') or {}).get('camera_transform_generation')==(frame.get('source_stamp') or {}).get('camera_transform_generation'))
        deltas=[b-a for a,b in zip(chosen,other)] if chosen is not None and other is not None and same_scope else None
        later=deltas is not None and all(d>0 for d in deltas)
        result['later_mt_ids' if later else 'blocking_mt_ids'].extend(frame['mt_invocation_ids'])
        result['comparisons'].append(dict(frame_key=frame['frame_key'],mt_invocation_ids=list(frame['mt_invocation_ids']),
            status='STRICTLY_LATER' if later else 'ORDER_UNPROVEN' if deltas is None else 'MAY_BE_EARLIER_OR_ORDER_CONFLICT',
            delta_unknown_minus_candidate=dict(zip(('capture_ns','source_sequence','sensor_ns','mt_start_ns','mt_start_event'),deltas)) if deltas is not None else None))
    if result['blocking_mt_ids']:return blocked('UNKNOWN_MT_MAY_BE_EARLIER')
    return dict(result,reason='UNKNOWN_MT_STRICTLY_LATER')


def score_frame(exposure, state, frame, *, policy=ref.POLICY):
    """Pure projection used by native adapter and explicitly synthetic tests."""
    result=dict(exposure_id=exposure['exposure_id'],position=exposure['position'],photo_sha256=exposure['photo_sha256'],
        source_role=exposure['source_role'],status='N/A',reason='',tp=None,fp=None,fn=None,
        readable_gt=0,exact_reads=0,wrong_reads=0,missing_reads=0,unknown_reads=0,
        system_failure=False,negative_false_positive=None,conditional_mz_n=0,conditional_mz_exact=0,
        negative_text_only=None,negative_correct_silence=None,
        plates=[dict(plate_id=o['object_id'],source_plate_annotation_id=o.get('source_plate_annotation_id'),
            text_state=photo.text_kind(o),gt_text=o['gt_text'] if photo.text_kind(o)=='FULL_TEXT' else None,
            source_gt_quad=o['quad'],detection='N/A',text_outcome='N/A',exact_match=None) for o in state['objects']],
        evidence=frame,selection_policy=policy['id'])
    def unavailable(reason):
        result['reason']=reason
        for plate in result['plates']:plate['missing_reason']=reason
        return result
    if state['presence']=='UNRESOLVED':return unavailable('REFERENCE_UNRESOLVED')
    if frame is None:return unavailable('NO_SELECTED_MT_INPUT')
    if frame.get('geometry_status')!='PASS':return unavailable(frame.get('geometry_reason') or 'GEOMETRY_UNPROVEN')
    gt=frame.get('projected_gt')
    if gt is None or len(gt)!=len(state['objects']):return unavailable('GT_TRANSFORM_UNPROVEN')
    region=frame['input_region'];size=frame['image_size']
    if any(not _inside(g['bbox'],region) for g in gt):return unavailable('GT_OUTSIDE_OR_PARTIAL_MODEL_INPUT')
    objects={o['object_id']:o for o in state['objects']}
    readable=[o for o in state['objects'] if photo.text_kind(o)=='FULL_TEXT' and o['gt_text']]
    result['readable_gt']=len(readable)
    if frame.get('execution_status')!='COMPLETED':
        if frame.get('execution_status') in ('ERROR','CANCELLED'):
            result.update(status='SYSTEM_FAILURE',reason=frame['execution_status'],system_failure=True,missing_reads=len(readable))
            result['plates']=[dict(plate_id=o['object_id'],text_state=photo.text_kind(o),detection='N/A',
                text_outcome='SYSTEM_FAILURE' if o in readable else 'N/A') for o in state['objects']]
            return result
        return unavailable('MT_EXECUTION_UNPROVEN')
    if frame.get('final_population_complete') is not True:return unavailable('FINAL_POPULATION_UNPROVEN')
    finals=frame.get('finals')
    if not isinstance(finals,list):return unavailable('FINAL_RECORDS_MISSING')
    if any(not _inside(c.get('bbox') or [math.nan]*4,[0,0,*size]) for c in finals):return unavailable('FINAL_BBOX_INVALID')
    if any(not isinstance(c.get('confidence'),(int,float)) or not math.isfinite(c['confidence']) or not 0<=c['confidence']<=1 for c in finals):
        return unavailable('FINAL_CONFIDENCE_UNPROVEN')
    if len({c['attempt_id'] for c in finals})!=len(finals):return unavailable('DUPLICATE_FINAL_ID')
    finals=sorted(finals,key=lambda c:(str(c.get('mt_invocation_id') or ''),c.get('index',0),c['attempt_id']))
    result['final_order']=[c['attempt_id'] for c in finals]
    protocol=dict(METRIC_PROTOCOL,operational_conf=-1,match_iou=.5,plate_class_ids=[0])
    scene=dict(sha256=exposure['photo_sha256'],domain='F6',image=dict(oriented_size=size),
        plates=[dict(plate_id=g['plate_id'],bbox=g['bbox'],quad=g['quad'],pose_eligible=False,pose_exclusion=None) for g in gt])
    predictions=[dict(index=i,confidence=c['confidence'],class_id=0,bbox=c['bbox']) for i,c in enumerate(finals)]
    scored=_scene_result(dict(domain='F6',width=size[0],height=size[1],predictions=predictions),scene,protocol)
    result.update(status='MEASURED',tp=len(scored['matched']),fp=len(scored['unmatched_prediction_indices']),fn=len(scored['unmatched_gt_ids']),
        matching=scored,negative_false_positive=bool(finals) if state['presence']=='NO_PLATES' else None)
    if state['presence']=='NO_PLATES':
        result['negative_text_only']=bool(frame.get('text_without_geometry'))
        result['negative_correct_silence']=not finals and not frame.get('text_without_geometry')
    pairs={p['plate_id']:finals[p['prediction_index']] for p in scored['matched']}
    result['plates']=[]
    for g in gt:
        o=objects[g['plate_id']];candidate=pairs.get(o['object_id']);kind=photo.text_kind(o)
        detail=dict(plate_id=o['object_id'],source_plate_annotation_id=o.get('source_plate_annotation_id'),
            text_state=kind,gt_text=o['gt_text'] if kind=='FULL_TEXT' else None,
            detection='TP' if candidate else 'FN',text_outcome='N/A',exact_match=None,
            attempt_id=candidate['attempt_id'] if candidate else None,geometry=g)
        if kind=='FULL_TEXT':
            mz=(candidate or {}).get('mz') or {}
            if candidate is None or mz.get('executed') is False:
                detail.update(text_outcome='NO_READ',exact_match=False);result['missing_reads']+=1
            elif mz.get('status')=='ERROR':
                detail.update(text_outcome='MZ_EXECUTION_ERROR',exact_match=False);result['missing_reads']+=1
            elif mz.get('executed') is True and mz.get('source_bound') is True and mz.get('raw_field_present') is True and mz.get('unambiguous') is True:
                predicted=normalize_registration(mz['raw_prediction']);exact=predicted==normalize_registration(o['gt_text'])
                detail.update(text_outcome='EXACT' if exact else 'WRONG' if predicted else 'NO_READ',exact_match=exact,
                    raw_prediction=mz['raw_prediction'],source_sample_ids=mz.get('sample_ids',[]))
                result['exact_reads']+=int(exact);result['wrong_reads']+=int(not exact and bool(predicted));result['missing_reads']+=int(not predicted)
                result['conditional_mz_n']+=1;result['conditional_mz_exact']+=int(exact)
            else:
                detail['text_outcome']='N/A_MZ_EVIDENCE_MISSING';result['unknown_reads']+=1
        result['plates'].append(detail)
    result.update(rates(result['tp'],result['fp'],result['fn']))
    return result


def _readout(session, candidate):
    from .mobile_canvas_review import _prediction
    aid=candidate['attempt_id'];attempt=session.attempts[aid]
    ids=list(session.sample_ids_by_attempt.get(aid,()))
    samples=[session.samples[k] for k in ids]
    executed={'true':True,'false':False,'1':True,'0':False}.get(str(attempt.get('mz_executed','')).lower())
    value,status,_=_prediction(session,aid)
    if status=='NOT_RUN':executed=False
    lineage=all(str(sample[k])==str(attempt.get(k,'')) for sample in samples
        for k in ('session_id','mt_invocation_id',*STAMP) if sample.get(k) is not None)
    crop=attempt.get('evidence_entry')
    bound=bool(lineage and crop in session.entry_names and attempt.get('evidence_kind')=='plate_crop'
        and all((s.get('image_entry') or s.get('evidence_entry'))==crop for s in samples))
    raw_present=all('raw_prediction' in s or 'prediction' in s for s in samples) if samples else 'raw_prediction' in attempt or 'prediction' in attempt
    f6=getattr(session,'f6_evidence',{})
    lifecycle=f6.get('mz_invocations',{}).get(f6.get('mz_by_attempt',{}).get(aid))
    if f6.get('available'):
        if lifecycle:
            executed=True
            if lifecycle['status']=='ERROR':status='ERROR'
            if lifecycle['status']=='INTERRUPTED':bound=False
        elif executed is True:bound=False
    return dict(executed=executed,status=status,raw_prediction=value,source_bound=bound,lifecycle=lifecycle,
        unambiguous=len(samples)<=1 and status!='AMBIGUOUS_SAMPLES',raw_field_present=raw_present,
        sample_ids=ids,evidence_entry=crop)


def _native_frames(session, manifest, manifest_sha, run_id, *, progress=None, run_context=None):
    from .mobile_f5a_scene_gt import source_candidates
    tv=session.tv_bbox_evidence
    if not tv.get('available'):return [],['TV_BBOX_SOURCE_EXTENSION_MISSING']
    roles={r['mt_invocation_id']:r for r in (session.bundle.scene_evidence.get('roles') or [])}
    f6=getattr(session,'f6_evidence',{});active=f6.get('available')
    frames=[];descriptor=tv['descriptor'];offset=descriptor['offset_to_camera_full']
    with zipfile.ZipFile(session.path) as archive:
        for index,row in enumerate(tv['rows']):
            gids=sorted({link['mt_invocation_id'] for link in row['attempt_links'] if link.get('mt_executed') is True and link.get('mt_invocation_id')})
            if active:gids=list(f6.get('frames',{}).get(row['frame_key'],{}).get('mt_invocation_ids',[]))
            calls=[f6['mt_invocations'][gid] for gid in gids] if active else []
            frame=dict(frame_key=row['frame_key'],source_stamp=deepcopy(row['source_stamp']),
                f6_lifecycle_required=True,
                captured_elapsed_nanos=row['captured_elapsed_nanos'],capture_clock='AS_RECORDED_CAPTURED_ELAPSED_NANOS_NO_CROSS_CLOCK_CONVERSION',
                mt_invocation_ids=gids,scene_event_id=row.get('scene_event_id'),identity_status='N/A',geometry_status='N/A',
                source_image=deepcopy(row['source_image']),crop_image=deepcopy(row['crop_image']),
                original_status=row['pipeline_status'],stale_or_cancelled=row['stale_or_cancelled'],
                source_archive_sha256=session.review.source_archive_sha256,session_id=session.session_id,
                input_region=[descriptor['tv_bbox_px'][k] for k in ('left','top','right','bottom')],
                crop_translation=offset,processing_transform=descriptor['processing_transform'],H_TV=descriptor['H_TV'])
            if active:
                starts=[c['events']['MT_STARTED'] for c in calls]
                start=min(starts,key=lambda e:e['event_sequence']) if starts else {}
                frame.update(f6_lifecycle_required=True,mt_started_elapsed_nanos=start.get('elapsed_nanos'),
                    mt_start_event_sequence=start.get('event_sequence'),android_clock_domain='ANDROID_ELAPSED_REALTIME',android_unit='ns',
                    mt_lifecycle=deepcopy(calls),input_region=[f6['photo_roi']['photo_roi_camera_xy'][k] for k in ('left','top','right','bottom')],
                    actual_mt_inputs=[i for c in calls for i in c.get('inputs',[])],tv_crop_role='CONTEXT_ONLY_MAY_CONTAIN_MARKERS')
            image_record=row['source_image']
            if image_record.get('status')!='WRITTEN':frame['geometry_reason']='SOURCE_IMAGE_MISSING';frames.append(frame);continue
            raw=archive.read(image_record['entry'])
            if __import__('hashlib').sha256(raw).hexdigest()!=image_record['sha256']:raise ValueError('SOURCE_FRAME_SHA_CHANGED')
            with Image.open(io.BytesIO(raw)) as original:image=original.convert('RGB')
            decoded=markers.decode(image,manifest,manifest_sha,run_id,run_context=run_context)
            frame.update(marker_decoding=decoded,image_size=list(image.size),identity_status=decoded['status'],
                geometry_reason=decoded.get('reason',''),exposure_id=decoded.get('exposure_id'))
            if decoded['status']=='PASS':
                rois=[[link['roi_local_xy'][k]+(offset['x'] if k in ('left','right') else offset['y'])
                       for k in ('left','top','right','bottom')] for link in row['attempt_links'] if link.get('mt_executed') is True]
                exclusion=markers.input_exclusion(decoded,manifest['layout'],frame['input_region'],rois)
                if row['crop_image'].get('status')!='WRITTEN':exclusion=dict(status='N/A',reason='RECORDED_CROP_IMAGE_MISSING')
                if active and (not calls or any(not c.get('inputs') or any(i['status']!='WRITTEN' for i in c['inputs']) for c in calls)):
                    exclusion=dict(status='N/A',reason='ACTUAL_MT_INPUT_EVIDENCE_MISSING')
                frame.update(input_exclusion=exclusion,geometry_status=exclusion['status'],geometry_reason=exclusion.get('reason',''))
            native_roles=[roles.get(gid) for gid in gids]
            bound=all(r and r['source_stamp']==row['source_stamp'] for r in native_roles)
            failed=bool(row['stale_or_cancelled'] or any(r and (r.get('stale_or_cancelled') or r.get('execution_error') or r.get('processing_error')) for r in native_roles))
            complete=bool(gids and bound and all(r.get('postprocess_status')=='COMPLETED' and r.get('role_proof_status')=='COMPLETE' for r in native_roles))
            if active:
                failed=any(c['status']=='ERROR' for c in calls)
                complete=bool(calls and bound and all(c.get('final_population_complete') for c in calls))
            frame['execution_status']='CANCELLED' if row['stale_or_cancelled'] else 'ERROR' if failed else 'COMPLETED' if complete else 'UNPROVEN'
            frame['final_population_complete']=complete
            finals=[]
            if complete:
                expected=set(aid for role in native_roles for aid in role['final_attempt_ids'])
                for gid in gids:
                    for c in source_candidates(session,gid):
                        if c['role']=='FINAL':finals.append(dict(c,mt_invocation_id=gid,mz=_readout(session,c)))
                if {c['attempt_id'] for c in finals}!=expected:frame['final_population_complete']=False
            frame['finals']=finals
            final_ids={c['attempt_id'] for c in finals}
            frame['text_without_geometry']=[dict(attempt_id=aid,raw_prediction=str(a.get('raw_prediction',a.get('prediction','')) or ''),
                mz_executed=a.get('mz_executed'),reason='TEXT_WITHOUT_PROVEN_FINAL_GEOMETRY') for aid,a in session.attempts.items()
                if aid not in final_ids and a.get('mt_invocation_id') in gids
                and all(str(a.get(k))==str(row['source_stamp'].get(k)) for k in STAMP)
                and str(a.get('raw_prediction',a.get('prediction','')) or '').strip()
                and not any(aid in (role.get('suppressed_attempt_ids') or []) for role in native_roles if role)]
            frames.append(frame)
            if progress:progress(index+1,len(tv['rows']))
    return frames,[]


def summarize(rows, frames, manifest):
    measured=[r for r in rows if r['status']=='MEASURED']
    system=[r for r in rows if r['status'] in ('MEASURED','SYSTEM_FAILURE')]
    tp=sum(r['tp'] for r in measured);fp=sum(r['fp'] for r in measured);fn=sum(r['fn'] for r in measured)
    denominator=sum(r['readable_gt'] for r in system);exact=sum(r['exact_reads'] for r in system);unknown=sum(r['unknown_reads'] for r in system)
    conditional_n=sum(r['conditional_mz_n'] for r in system);conditional_exact=sum(r['conditional_mz_exact'] for r in system)
    planned_readable=sum(p['text_state']=='FULL_TEXT' for r in rows for p in r['plates'])
    return dict(planned_exposures=len(manifest['exposures']),identified_exposures=len({f['exposure_id'] for f in frames if f.get('exposure_id') and f.get('identity_status')=='PASS'}),
        planned_reference_counts={k:manifest.get('input_audit',{}).get(k) for k in ('plates','full_text','unreadable','partial')},
        geometrically_evaluable_plates=sum(len(r['plates']) for r in system),
        unassessable_plates_by_reason=dict(Counter({reason:sum(len(r['plates']) for r in rows if r['status']=='N/A' and r['reason']==reason)
            for reason in {r['reason'] for r in rows if r['status']=='N/A'}})),
        selected_mt_exposures=sum(r['evidence'] is not None for r in rows),unassigned_frames=sum(f['identity_status']!='PASS' for f in frames),
        exposure_status_counts=dict(Counter(r['status'] for r in rows)),na_reasons=dict(Counter(r['reason'] for r in rows if r['status']=='N/A')),
        planned_positive=sum(e['source_role']=='POSITIVE' for e in manifest['exposures']),planned_negative=sum(e['source_role']=='NEGATIVE' for e in manifest['exposures']),
        detection=dict(tp=tp if measured else None,fp=fp if measured else None,fn=fn if measured else None,
            n_exposures=len(measured),**rates(tp,fp,fn)),
        system_full_read=dict(correct=exact,denominator=planned_readable,covered_denominator=denominator,unknown=unknown,
            uncovered=max(0,planned_readable-denominator),rate=exact/planned_readable if planned_readable and denominator==planned_readable and not unknown else None,
            wrong=sum(r['wrong_reads'] for r in system),missing=sum(r['missing_reads'] for r in system),
            basis=ref.POLICY['system_denominator']),
        evaluated_subset_full_read=dict(correct=exact,denominator=denominator,unknown=unknown,
            rate=exact/denominator if denominator and not unknown else None,scope='EVALUATED_SUBSET_ONLY'),
        conditional_mz=dict(correct=conditional_exact,denominator=conditional_n,rate=conditional_exact/conditional_n if conditional_n else None),
        negatives_with_false_detection=(sum(r['negative_false_positive'] is True for r in rows)
            if any(r['negative_false_positive'] is not None for r in rows) else None),
        evaluable_negatives=sum(r['negative_false_positive'] is not None for r in rows),
        negatives_with_text_without_geometry=sum(r['negative_text_only'] is True for r in rows),
        negative_correct_silences=sum(r['negative_correct_silence'] is True for r in rows),
        execution_failures=sum(r['system_failure'] for r in rows))


def analyze_archive(prepared, archive_path, run_path, *, progress=None,reference_aware=False,reference_path=None):
    from .mobile_human_review import MobileReviewSession
    reference_aware=reference_aware or photo.read_json(Path(prepared)/'manifest.json').get('schema')==ref.SCHEMA_V2 or reference_path is not None
    manifest,material,reference=ref.load(prepared,strict_reference=not reference_aware,reference_path=reference_path)
    reference_aware=reference_aware or manifest.get('schema')==ref.SCHEMA_V2 or reference_path is not None
    preflight=ref.reference_preflight(prepared,reference_path=reference_path) if reference_aware else None
    coverage={r['exposure_id']:r for r in preflight['coverage']} if preflight else {}
    def state_for(exposure):
        if reference is None or (preflight and coverage[exposure['exposure_id']]['status']!='GT_COMPLETE'):return photo.blank()
        return reference.state(exposure['image_id'])
    manifest_sha=photo.digest(Path(prepared)/'manifest.json');run=photo.read_json(run_path)
    if run.get('manifest_sha256')!=manifest_sha or run.get('schema')!='alpr.final.presentation_run.v1':raise ValueError('RUN_MANIFEST_MISMATCH')
    from . import final180_timeline as timeline
    run,events,_=timeline.read_run(run_path,manifest,manifest_sha)
    pc_rows=timeline.project(manifest,events)
    presentation_plan=ref.plan_times(len(manifest['exposures']),manifest['dwell_seconds'],
        countdown_seconds=manifest['countdown_seconds'],transition_seconds=manifest['transition_seconds'])
    for r in pc_rows:r['file']=material.images[manifest['exposures'][r['position']-1]['image_id']]['source_relative_path']
    pc_by_id={r['exposure_id']:r for r in pc_rows}
    requested=[e['exposure_id'] for e in events if e.get('event')=='PHOTO_REQUESTED']
    if len(requested)!=len(set(requested)):raise ValueError('DUPLICATE_EXPOSURE_REQUEST_IN_RUN')
    with tempfile.TemporaryDirectory(prefix='final180-readonly-') as private:
        try:
            session=MobileReviewSession.open(Path(archive_path),sidecar_path=Path(private)/'absent.review.json')
        except (ValueError,OSError,zipfile.BadZipFile) as exc:
            rows=[]
            for exposure in manifest['exposures']:
                row=score_frame(exposure,state_for(exposure),None)
                row['reason']='NATIVE_ARCHIVE_IMPORT_FAILED'
                for plate in row['plates']:plate['missing_reason']=row['reason']
                rows.append(row)
            failed=dict(schema='alpr.final.evaluation.v1',contract=ref.CONTRACT,manifest_sha256=manifest_sha,
                reference_sha256=manifest.get('reference_sha256'),source_archive_path=str(archive_path),
                source_archive_sha256=photo.digest(archive_path) if Path(archive_path).is_file() else None,
                rows=rows,frames=[],presentation_timeline=pc_rows,presentation_plan=presentation_plan,summary=summarize(rows,[],manifest),integration_gaps=[str(exc)],
                go='NO_GO',go_reasons=['NATIVE_ARCHIVE_IMPORT_FAILED'],
                acceptance=dict(desktop_import='FAIL',producer_contract='NOT_VERIFIED',a12_tv='NOT_VERIFIED'),
                policy=manifest['policy'],interpretation='SELECTED_SOURCE_FRAME_PROCESSING_NOT_FINAL_HUD')
            return reference_report(failed,preflight,prepared) if preflight else failed
        if not session.bundle.validation.ok:raise ValueError('NATIVE_ARCHIVE_INVALID')
        config=configuration(session.bundle,manifest['configuration'],layout_version=markers.F2_LAYOUT if reference_aware else manifest['layout'].get('layout_version'))
        probe=[]
        if reference_aware:
            # Reproducible pixel discovery before analysis. Seed is fixed for QR
            # processing, never selected by GT or recognition quality; models
            # are not run here. Legacy entry points keep their original path.
            import cv2
            cv2.setRNGSeed(0)
            context=timeline.marker_context(run,events,manifest,manifest_sha,prepared=prepared)
            with zipfile.ZipFile(session.path) as archive:
                for source in session.tv_bbox_evidence.get('rows',[]):
                    record=source['source_image']
                    if record.get('status')!='WRITTEN':continue
                    raw=archive.read(record['entry'])
                    if __import__('hashlib').sha256(raw).hexdigest()!=record['sha256']:raise ValueError('SOURCE_FRAME_SHA_CHANGED')
                    with Image.open(io.BytesIO(raw)) as image:decoded=markers.decode(image.convert('RGB'),manifest,manifest_sha,run['run_id'],run_context=context)
                    probe.append(dict(frame_key=source['frame_key'],status=decoded['status'],reason=decoded.get('reason'),tokens=decoded.get('decoded_tokens')))
                    if decoded['status']=='PASS':break
        frames,gaps=_native_frames(session,manifest,manifest_sha,run['run_id'],progress=progress,
            run_context=timeline.marker_context(run,events,manifest,manifest_sha,prepared=prepared))
        f6=getattr(session,'f6_evidence',{})
        mt_scope=first_mt_scope(frames,f6,session.bundle.collection_session)
        unlocated=f6.get('available') and (f6.get('unlocated_loss_possible') or any(
            f.get('mt_invocation_ids') and f.get('identity_status')!='PASS' and f.get('geometry_reason')!='TECHNICAL_TRANSITION_FRAME' for f in frames)
            or any(c['frame_key'] not in {f['frame_key'] for f in frames} for c in f6.get('mt_invocations',{}).values()))
        if unlocated:gaps.append('EARLIER_MT_INPUT_MAY_BE_UNLOCATED')
        for frame in frames:
            if frame.get('exposure_id') and frame['exposure_id'] not in requested:
                frame.update(identity_status='N/A',geometry_status='N/A',geometry_reason='EXPOSURE_NOT_IN_PC_REQUEST_LOG')
        recording={k:session.bundle.collection_session.get(k) for k in ('collection_complete','dropped_sample_count',
            'dropped_telemetry_count','integrity_loss_count','scene_evidence_pending_loss_unknown','mt_detection_roles_complete')}
        if recording['collection_complete'] is not True or any(recording[k] for k in (
                'dropped_sample_count','dropped_telemetry_count','integrity_loss_count','scene_evidence_pending_loss_unknown')):
            gaps.append('SOURCE_RECORDING_INCOMPLETE')
        recording['f6']=dict(status=f6.get('status'),completeness=f6.get('completeness'),summary=f6.get('summary'),binding_gaps=f6.get('binding_gaps'))
        if f6.get('available') and f6.get('status')!='VALID':gaps.append('F6_EVIDENCE_PARTIAL')
        rows=[]
        for exposure in manifest['exposures']:
            state=state_for(exposure);frame,reason=select_first(exposure['exposure_id'],frames)
            risk=first_mt_risk(frame,frames,mt_scope)
            pc=pc_by_id[exposure['exposure_id']]
            if pc['status']=='NOT_SHOWN':frame,reason=None,'NOT_SHOWN'
            elif frame is not None and risk['status']!='PASS':frame,reason=None,'EARLIER_MT_INPUT_MAY_BE_UNLOCATED'
            if frame:
                frame=deepcopy(frame)
                if config['status']!='PASS':frame.update(geometry_status='N/A',geometry_reason='CONFIGURATION_MISMATCH')
                if frame['geometry_status']=='PASS':
                    projected=[];size=[material.images[exposure['image_id']][k] for k in ('width','height')]
                    photo_quad,photo_chain=markers.transform_gt([[0,0],[size[0],0],size,[0,size[1]]],size,manifest['layout'],frame['marker_decoding'])
                    frame['photo_camera_quad']=photo_quad
                    if not _inside(_bbox(photo_quad),frame['input_region']):
                        frame.update(geometry_status='N/A',geometry_reason='PHOTOGRAPH_NOT_FULLY_COVERED_BY_MODEL_INPUT')
                    for obj in state['objects']:
                        if obj['quad'] is None:frame.update(geometry_status='N/A',geometry_reason='REFERENCE_GEOMETRY_UNRESOLVED');break
                        q,chain=markers.transform_gt(obj['quad'],size,manifest['layout'],frame['marker_decoding'])
                        projected.append(dict(plate_id=obj['object_id'],quad=q,bbox=_bbox(q),chain=chain))
                    frame['projected_gt']=projected
            result=score_frame(exposure,state,frame)
            result['presentation']=pc
            result['first_mt_risk']=risk
            if frame is None:
                result['reason']=reason
                for plate in result['plates']:plate['missing_reason']=reason
            rows.append(result)
        report=dict(schema='alpr.final.evaluation.v1',contract=ref.CONTRACT,manifest_sha256=manifest_sha,
            reference_sha256=manifest['reference_sha256'],source_archive_sha256=session.review.source_archive_sha256,
            source_archive_path=str(Path(archive_path).resolve()),session_id=session.session_id,run_id=run['run_id'],
            policy=manifest['policy'],geometry_policy=manifest['geometry_policy'],configuration=config,
            rows=rows,frames=frames,presentation_timeline=pc_rows,presentation_plan=presentation_plan,summary=summarize(rows,frames,manifest),integration_gaps=gaps,
            recording_completeness=recording,first_mt_scope=mt_scope,pc_run_status=run.get('status'),pc_requested_exposures=requested,
            acceptance=dict(desktop_import='NATIVE_READER_VALIDATED',producer_contract='NATIVE_F6_FORMAT_VALIDATED' if f6.get('available') else 'NOT_VERIFIED',
                marker_source_chain='NOT_VERIFIED_ON_A12',a12_tv='NOT_VERIFIED',actual_tensor_pixels='JPEG_EVIDENCE_NOT_EXACT_TENSOR'),
            go='NO_GO',go_reasons=['A12_TV_ACCEPTANCE_NOT_VERIFIED','F6_MARKER_SOURCE_CHAIN_NOT_VERIFIED_ON_DEVICE']+
                (['CONFIGURATION_PINS_NOT_FROZEN'] if config['pins_status']!='FROZEN' else [])+manifest['input_errors']+gaps+config['errors'],
            interpretation='SELECTED_SOURCE_FRAME_PROCESSING_NOT_FINAL_HUD',original_sources_readonly=True)
        session.verify_source()
        if manifest['layout'].get('layout_version')==markers.F3_LAYOUT:
            report['readability_preflight']=readability_preflight(report,manifest)
        if preflight:
            report['identity_probe']=probe;report['decoder_execution']=dict(qr_rng_seed=0,scope='PIXEL_DISCOVERY_THEN_ALL_SOURCE_FRAMES',model_parameters_changed=False)
            by_frame={f['frame_key']:f for f in frames}
            report['raw_mt']=[dict(deepcopy(call),exposure_id=by_frame.get(call['frame_key'],{}).get('exposure_id')) for call in f6.get('mt_invocations',{}).values()]
            raw_mz=[]
            for call in f6.get('mz_invocations',{}).values():
                samples=[s for s in session.samples.values() if s.get('attempt_id')==call['attempt_id']]
                parent=f6.get('mt_invocations',{}).get(call['mt_invocation_id'],{})
                raw_mz.append(dict(deepcopy(call),exposure_id=by_frame.get(call['frame_key'],{}).get('exposure_id'),
                    raw_prediction=samples[0].get('raw_prediction') if len(samples)==1 else None,
                    source_bound=call.get('source_stamp')==parent.get('source_stamp'),sample_count=len(samples)))
            report['raw_mz']=raw_mz
            return reference_report(report,preflight,prepared)
        return report


def readability_preflight(report,manifest):
    """Diagnostic projection of the existing native reader; never changes scoring.

    Recorded bitmap dimensions are NOT inferred CameraX buffer dimensions.
    P/T pair proof and first MT ordering are independent of GT availability.
    """
    board=manifest['layout'];rows=[];exposures={e['exposure_id']:e for e in manifest['exposures']}
    for f in report['frames']:
        d=f.get('marker_decoding') or {};codes=d.get('qr_codes') or []
        geometry_ok=len(codes)==2  # decode_codes is called only after the frozen geometry gate.
        pair=bool(d.get('pair_verified')) if board.get('qr_version')==markers.F3_QR else bool(
            geometry_ok and all(c.get('value') for c in codes) and codes[0]['value']==codes[1]['value']
            and (f.get('identity_status')=='PASS' or d.get('reason')=='TECHNICAL_TRANSITION_FRAME'))
        phase='P' if f.get('identity_status')=='PASS' else 'T' if pair and d.get('reason')=='TECHNICAL_TRANSITION_FRAME' else None
        row=dict(frame_key=f['frame_key'],source_stamp=f['source_stamp'],source_image=f['source_image'],
            recorded_bitmap_size=f.get('image_size'),phase=phase,exposure_id=f.get('exposure_id'),
            pair_verified=pair,geometry_status='PASS' if geometry_ok else 'N/A',reason=d.get('reason',f.get('geometry_reason')),
            four_aruco=len(d.get('detected_ids',[]))==4 and sorted(d.get('detected_ids',[]))==[0,1,2,3],
            reprojection_max_px=d.get('reprojection_max_px'),leave_one_marker_out_max_px=d.get('leave_one_marker_out_max_px'),
            left_decoded=bool(codes[0].get('value')) if geometry_ok else None,
            right_decoded=bool(codes[1].get('value')) if geometry_ok else None,
            left_token=codes[0].get('value') if geometry_ok else None,right_token=codes[1].get('value') if geometry_ok else None,
            pixels_per_module=None,markers_inside_full_frame=None,configured_roi_marker_exclusion='N/A',
            actual_mt_marker_exclusion='N/A',photograph_inside_actual_mt=None,mt_invocation_ids=f.get('mt_invocation_ids',[]))
        if geometry_ok:
            row['pixels_per_module']=[min(c['camera_edge_px'])/board['qr']['raster_modules'] for c in codes]
            width,height=f['image_size'];matrix=d['matrix_tv_to_camera']
            rects=board['qr_rects']+[[a['rect'][0]-a['quiet'],a['rect'][1]-a['quiet'],a['rect'][2]+a['quiet'],a['rect'][3]+a['quiet']] for a in board['aruco']]
            row['markers_inside_full_frame']=all(_inside(_bbox(markers.project(markers.rect_points(r),matrix)),[0,0,width,height]) for r in rects)
            valid_geometry=dict(d,status='PASS')
            row['configured_roi_marker_exclusion']=markers.input_exclusion(valid_geometry,board,f['input_region'])['status']
            inputs=f.get('actual_mt_inputs',[])
            if inputs and all(i.get('status')=='WRITTEN' and len(i.get('roi_camera_xy',[]))==4 for i in inputs):
                regions=[i['roi_camera_xy'] for i in inputs]
                row['actual_mt_marker_exclusion']=markers.input_exclusion(valid_geometry,board,regions[0],regions[1:])['status']
                exposure=exposures.get(f.get('exposure_id'))
                if exposure:
                    rect=exposure.get('actual_image_rect') or markers.photo_placement(exposure['photo_placement']['source_size'],board)['rect']
                    bbox=_bbox(markers.project(markers.rect_points(rect),matrix))
                    row['photograph_inside_actual_mt']=all(_inside(bbox,r) for r in regions)
        rows.append(row)
    unlocated=[gid for r in rows if r['phase'] is None for gid in r['mt_invocation_ids']]
    scope=report.get('first_mt_scope')
    if scope is None and 'EARLIER_MT_INPUT_MAY_BE_UNLOCATED' in report.get('integration_gaps',[]):
        scope=dict(status='UNBOUNDED',reasons=['HISTORICAL_REPORT_WITHOUT_START_INVENTORY'])
    pc={r['exposure_id']:r for r in report.get('presentation_timeline',[])};first=[]
    for e in manifest['exposures']:
        frame,reason=select_first(e['exposure_id'],report['frames'])
        risk=first_mt_risk(frame,report['frames'],scope)
        if pc.get(e['exposure_id'],{}).get('status')=='NOT_SHOWN':frame,reason=None,'NOT_SHOWN'
        elif frame is not None and risk['status']!='PASS':frame,reason=None,'EARLIER_MT_INPUT_MAY_BE_UNLOCATED'
        first.append(dict(exposure_id=e['exposure_id'],status='PASS_SOURCE_ORDER' if frame else 'N/A',reason=reason,
            frame_key=frame['frame_key'] if frame else None,mt_invocation_ids=frame['mt_invocation_ids'] if frame else None,first_mt_risk=risk))
    n=len(rows);paired=sum(r['pair_verified'] for r in rows);geometry_n=sum(r['geometry_status']=='PASS' for r in rows)
    return dict(schema='alpr.final.readability_preflight.v1',layout_version=board.get('layout_version','F6v1'),
        qr_version=board.get('qr_version','F6v1'),manifest_sha256=report['manifest_sha256'],
        source_archive_sha256=report['source_archive_sha256'],source_archive_path=report.get('source_archive_path'),
        session_id=report.get('session_id'),run_id=report.get('run_id'),
        source='EXISTING_NATIVE_READER_AND_FULL_SOURCE_IMAGE_SHA',frames=rows,first_mt=first,
        summary=dict(source_frames=n,geometry_qualified=geometry_n,left_decoded=sum(r['left_decoded'] is True for r in rows),
            right_decoded=sum(r['right_decoded'] is True for r in rows),individually_tested=geometry_n,
            verified_pairs=paired,pair_rate_all_source_frames=paired/n if n else None,
            phase_counts=dict(Counter(r['phase'] or 'UNIDENTIFIED' for r in rows)),
            identified_exposures=len({r['exposure_id'] for r in rows if r['phase']=='P'}),
            unlocated_mt_count=len(unlocated),unlocated_mt_ids=unlocated,
            first_mt_order_proven=sum(r['status']=='PASS_SOURCE_ORDER' for r in first),
            recorded_bitmap_sizes=sorted({tuple(r['recorded_bitmap_size']) for r in rows if r['recorded_bitmap_size']})),
        camera_buffer_dimensions=None,preview_dimensions=None,af_acceptance='NOT_VERIFIED',
        resolution_recommendation=None,a12_f3_acceptance='NOT_VERIFIED',go='NO_GO',
        limits=['RECORDED_BITMAP_IS_NOT_CAMERA_BUFFER_OR_PREVIEW_PROOF','PHYSICAL_CONFIGURATION_REQUIRES_OPERATOR_ACCEPTANCE',
            'NO_RESOLUTION_CHOICE_FROM_NOMINAL_PIXELS','NO_GT_OR_SCORING_POLICY_CHANGES'])


def locate_run(archive_path,scopes):
    """Locate, never assign frames: independent full-source QR + PC journal proof."""
    from .mobile_package_experiments import read_mobile_report_bundle
    from . import final180_timeline as timeline
    import cv2
    candidates,issues=timeline.discover_runs(scopes)
    bundle=read_mobile_report_bundle(archive_path)
    if not bundle.validation.ok:raise ValueError('NATIVE_ARCHIVE_INVALID')
    groups={}
    for candidate in candidates:
        manifest=candidate['manifest']
        key=json.dumps([manifest['layout'],manifest['geometry_policy']],sort_keys=True)
        groups.setdefault(key,[]).append(candidate)
    observations=[];matched=set();seen_tokens=set();proven_tokens=set()
    technical=set().union(*(c['technical_tokens'] for c in candidates)) if candidates else set()
    with zipfile.ZipFile(archive_path) as archive:
        for group in groups.values():
            manifest=group[0]['manifest'];cv2.setRNGSeed(0)
            for row in bundle.tv_bbox_evidence.get('rows',[]):
                record=row['source_image']
                if record.get('status')!='WRITTEN':continue
                raw=archive.read(record['entry'])
                if __import__('hashlib').sha256(raw).hexdigest()!=record['sha256']:raise ValueError('SOURCE_FRAME_SHA_CHANGED')
                with Image.open(io.BytesIO(raw)) as original:image=original.convert('RGB')
                geometry=markers.geometry(image,manifest['layout'],manifest['geometry_policy'])
                if geometry['status']!='PASS':continue
                codes=markers.decode_codes(image,manifest['layout'],geometry)['decoded_tokens']
                if len(codes)!=2 or not all(codes):continue
                paired=manifest['layout'].get('layout_version')==markers.F3_LAYOUT
                if paired:
                    # A nearby historical layout may decode its F2/F6 symbols
                    # in these regions. Such a read is not a foreign F3 pair.
                    if not markers.f3_pair_syntax(codes):continue
                    # Exact ordered pair membership requires both independent
                    # pixels. Never search for an identity from either half.
                    token=markers.token_key(codes)
                else:
                    if codes[0]!=codes[1]:continue
                    token=codes[0]
                hits=[c for c in group if token in c['tokens']]
                paths=[c['path'] for c in hits];matched.update(paths)
                seen_tokens.add(token)
                if hits:proven_tokens.add(token)
                observations.append(dict(frame_key=row['frame_key'],source_image=record,token=token,run_paths=paths,
                    technical_only=token in technical,proof='TWO_INDEPENDENT_SOURCE_QR_AND_FOUR_ARUCO'))
    unmatched=seen_tokens-proven_tokens-technical
    reason='UNIQUE_VERIFIED_RUN' if len(matched)==1 and not unmatched else 'AMBIGUOUS_RUNS' if len(matched)>1 else 'FOREIGN_OR_UNOBSERVED_TOKEN' if unmatched else 'NO_VERIFIED_RUN_TOKEN'
    result=dict(status='PASS' if reason=='UNIQUE_VERIFIED_RUN' else 'N/A',reason=reason,
        source_archive_sha256=bundle.source_archive_sha256,observations=observations,candidate_issues=issues,
        unmatched_tokens=sorted(unmatched,key=repr),matched_paths=sorted(matched))
    if result['status']=='PASS':
        candidate=next(c for c in candidates if c['path'] in matched)
        timeline.verify_run_proof(candidate['proof']);result.update(run_path=candidate['path'],run_proof=candidate['proof'])
    return result


def analyze_import(archive_path,run_path,*,output=None,reference_path=None,progress=None,run_match=None):
    """Shared GUI/CLI path. A selected run is a scope, not frame identity proof."""
    run_path=Path(run_path).resolve(strict=True);state=photo.read_json(run_path)
    if run_match is not None:
        from . import final180_timeline as timeline
        if run_match.get('status')!='PASS' or str(run_path)!=run_match['run_path'] or photo.digest(archive_path)!=run_match['source_archive_sha256']:
            raise ValueError('AUTO_RUN_MATCH_CHANGED')
        timeline.verify_run_proof(run_match['run_proof'])
    candidates=[run_path.parents[2],Path(state.get('prepared_path',''))]
    prepared=next((p for p in candidates if (p/'manifest.json').is_file() and photo.digest(p/'manifest.json')==state.get('manifest_sha256')),None)
    if prepared is None:raise ValueError('Brak przygotowania zgodnego z SHA wskazanego run.json.')
    result=analyze_archive(prepared,archive_path,run_path,reference_aware=True,reference_path=reference_path,progress=progress)
    if run_match is not None:
        timeline.verify_run_proof(run_match['run_proof'])
        if result['source_archive_sha256']!=run_match['source_archive_sha256']:raise ValueError('ARCHIVE_CHANGED_AFTER_MATCH')
        result['run_discovery']=deepcopy(run_match)
    result['pc_run_path']=str(run_path)
    return export_report(result,Path(output) if output else prepared.parent/'analysis_results')


def reference_report(report,preflight,prepared):
    """Add GT/evidence dimensions and CER without changing association or first MT."""
    from .mobile_human_review import align_plate_text
    report=deepcopy(report);report.update(schema='alpr.final.evaluation.v2',reference_preflight=deepcopy(preflight),analysis_reference_sha256=preflight['reference_sha256'],prepared_path=str(Path(prepared).resolve()))
    coverage={r['exposure_id']:r for r in preflight['coverage']};frames=report['frames'];chars=errors=0;pairs=0
    for row in report['rows']:
        gt=coverage[row['exposure_id']];related=[f for f in frames if f.get('identity_status')=='PASS' and f.get('exposure_id')==row['exposure_id']]
        selected,selection_reason=select_first(row['exposure_id'],frames)
        row.update(gt_status=gt['status'],gt_reason=gt['reason'],gt_plate_count=gt['plate_count'],gt_known_objects=gt['known_objects'],
            known_mt_count=sum(len(f['mt_invocation_ids']) for f in related),known_mz_count=sum(c.get('exposure_id')==row['exposure_id'] for c in report.get('raw_mz',[])),
            identified_frame_keys=[f['frame_key'] for f in related],diagnostic_frame_key=selected['frame_key'] if selected else related[0]['frame_key'] if related else None,
            diagnostic_is_selected_evidence=False,evidence_reason=row['reason'],evidence_selection_reason=selection_reason)
        evidence=row.get('evidence')
        row['evidence_status']='FIRST_MT_SOURCE_BOUND' if evidence is not None else 'N/A'
        if evidence is not None:
            row['evidence_geometry_status']=evidence.get('geometry_status')
            row['evidence_reason']=evidence.get('geometry_reason') or (evidence.get('execution_status') if evidence.get('execution_status')!='COMPLETED' else '')
        if gt['status']!='GT_COMPLETE':
            row.update(status='N/A',reason={'GT_MISSING':'NOT_AVAILABLE_NO_GT','GT_INCOMPLETE':'GT_INCOMPLETE','GT_CONFLICT':'GT_CONFLICT'}[gt['status']],
                tp=None,fp=None,fn=None,plates=[],system_failure=False,negative_false_positive=None,negative_correct_silence=None,
                exact_reads=None,wrong_reads=None,missing_reads=None,unknown_reads=None)
        elif row['status']=='N/A':
            for k in ('exact_reads','wrong_reads','missing_reads','unknown_reads'):row[k]=None
        for plate in row['plates']:
            plate['cer']=None
            if row['status']=='MEASURED' and plate.get('detection')=='TP' and plate['text_state']=='FULL_TEXT' and plate['text_outcome'] in ('EXACT','WRONG','NO_READ','MZ_EXECUTION_ERROR'):
                aligned=align_plate_text(plate['gt_text'],plate.get('raw_prediction') or '')
                plate.update(cer=aligned.cer,edit_distance=aligned.edit_distance,gt_characters=len(aligned.ground_truth))
                chars+=len(aligned.ground_truth);errors+=aligned.edit_distance;pairs+=1
    manifest=photo.read_json(Path(prepared)/'manifest.json')
    if photo.digest(Path(prepared)/'manifest.json')!=report['manifest_sha256']:raise ValueError('PREPARED_MANIFEST_CHANGED_DURING_ANALYSIS')
    summary=summarize(report['rows'],frames,manifest);report['summary']=summary
    all_evaluable=all(r['status'] in ('MEASURED','SYSTEM_FAILURE') for r in report['rows'])
    mode=preflight['mode'];measured=any(r['status'] in ('MEASURED','SYSTEM_FAILURE') for r in report['rows'])
    summary['reference_counts']=preflight['counts'];summary['reference_mode']=mode
    summary['verified_reference_counts']=dict(plates=preflight['complete_gt_plates'],full_text=preflight['full_text'],unreadable=preflight['unreadable'],negative_images=preflight['negative_images'])
    summary['planned_reference_counts']=dict(plates=preflight['complete_gt_plates'],full_text=preflight['full_text'],unreadable=preflight['unreadable'],
        partial=sum(p['text_state']=='PARTIAL_TEXT' for r in report['rows'] for p in r['plates'])) if mode=='complete' else {k:None for k in ('plates','full_text','unreadable','partial')}
    summary['conditional_mz_cer']=dict(value=errors/chars if chars else None,edit_distance=errors if chars else None,gt_characters=chars,matched_plate_pairs=pairs,
        scope='COMPLETE_GT_MATCHED_FINAL_WITH_KNOWN_MZ_OUTCOME_INCLUDING_EMPTY_READ')
    summary['raw_mt_calls']=len(report.get('raw_mt',[]));summary['raw_mz_calls']=len(report.get('raw_mz',[]))
    summary['identified_p_frames']=sum(f.get('identity_status')=='PASS' for f in frames)
    summary['technical_frames']=sum(f.get('geometry_reason')=='TECHNICAL_TRANSITION_FRAME' for f in frames)
    summary['unidentified_frames']=len(frames)-summary['identified_p_frames']-summary['technical_frames']
    summary['unlocated_mt_calls']=sum(len(f.get('mt_invocation_ids',[])) for f in frames if f.get('identity_status')!='PASS' and f.get('geometry_reason')!='TECHNICAL_TRANSITION_FRAME')
    if mode!='complete' or not all_evaluable:summary['system_full_read']['rate']=None
    if mode!='complete':summary['system_full_read']['denominator']=None;summary['system_full_read']['uncovered']=None
    if mode in ('none','invalid'):
        summary['system_full_read']['denominator']=None
        summary['planned_reference_counts']={k:None for k in ('plates','full_text','unreadable','partial')}
        summary['planned_positive']=None;summary['planned_negative']=None
    if not measured:
        for k in ('correct','wrong','missing','unknown'):summary['system_full_read'][k]=None
        for k in ('negative_correct_silences','negatives_with_text_without_geometry','negatives_with_false_detection'):summary[k]=None
    report['metrics_status']='NOT_AVAILABLE_NO_GT' if mode=='none' else 'NOT_AVAILABLE_GT_CONFLICT' if mode=='invalid' else 'EVALUATED_SUBSET' if measured and (mode!='complete' or not all_evaluable) else 'MEASURED' if measured and all_evaluable else 'NOT_AVAILABLE_EVIDENCE'
    summary['detection']['scope']='FULL_VERIFIED_POPULATION' if mode=='complete' and all(r['status']=='MEASURED' for r in report['rows']) else 'EVALUATED_SUBSET_ONLY'
    return report


def export_report(report, root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    digest=ref.fingerprint(report)
    for number in range(1,100000):
        target=root/f'result_{digest[:12]}_{number:03d}'
        if not target.exists():break
    target.mkdir(exist_ok=False)
    _atomic_write_json(target/'evaluation.json',report)
    from .final180_timeline import export_csv
    export_csv(target/'presentation_timeline.csv',report.get('presentation_timeline',[]))
    for name,rows in (('exposures.csv',report['rows']),('plates.csv',[dict(exposure_id=r['exposure_id'],**p) for r in report['rows'] for p in r['plates']])):
        columns=list(dict.fromkeys(k for row in rows for k in row)) or ['exposure_id','status']
        with (target/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns,delimiter=';');writer.writeheader()
            for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    if report.get('reference_preflight'):
        for name in ('raw_mt','raw_mz'):export_csv(target/(name+'.csv'),report.get(name,[]))
        export_csv(target/'reference_coverage.csv',report['reference_preflight']['coverage'])
        export_csv(target/'frame_evidence.csv',[dict(frame_key=f['frame_key'],exposure_id=f.get('exposure_id'),identity_status=f['identity_status'],
            geometry_status=f.get('geometry_status'),reason=f.get('geometry_reason'),mt_invocation_ids=f.get('mt_invocation_ids'),
            qr_a=((f.get('marker_decoding') or {}).get('qr_codes') or [{}])[0].get('status'),
            qr_b=(((f.get('marker_decoding') or {}).get('qr_codes') or [])+[{},{}])[1].get('status')) for f in report['frames']])
    if report.get('readability_preflight'):
        diagnostic=report['readability_preflight']
        _atomic_write_json(target/'readability_preflight.json',diagnostic)
        export_csv(target/'readability_frames.csv',diagnostic['frames'])
        export_csv(target/'readability_first_mt.csv',diagnostic['first_mt'])
    summary=report['summary'];det=summary['detection'];system=summary['system_full_read'];mz=summary['conditional_mz']
    shown=lambda v:'N/A' if v is None else str(v)
    text=['# FINAL — przetwarzanie wybranej klatki źródłowej','',
        'To nie jest końcowy napis HUD po 30 s. Brak dowodu pozostaje N/A.',
        'GO: '+report['go'],'',
        f"Plan: {summary['planned_exposures']} ekspozycji ({summary['planned_positive']} dodatnich + {summary['planned_negative']} negatywów).",
        f"Zidentyfikowane ekspozycje: {summary['identified_exposures']}; wybrane wejście MT: {summary['selected_mt_exposures']}; nieprzypisane klatki: {summary['unassigned_frames']}.",
        '', '| Miara | Wynik / mianownik |', '|---|---|',
        f"| Detekcja TP / FP / FN | {shown(det['tp'])} / {shown(det['fp'])} / {shown(det['fn'])}; ekspozycje n={det['n_exposures']} |",
        f"| Pełny odczyt systemowy | {system['correct']} / {system['denominator']}; nieznane: {system['unknown']} |",
        f"| Pokrycie mianownika / odsetek całego planu | {system['covered_denominator']} / {system['denominator']}; {shown(system['rate'])} |",
        f"| Błędne / brakujące odczyty | {system['wrong']} / {system['missing']} |",
        f"| MZ po trafieniu MT i wykonaniu MZ | {mz['correct']} / {mz['denominator']} |",
        f"| Negatywy z fałszywą ramką | {shown(summary['negatives_with_false_detection'])}; ocenialne n={summary['evaluable_negatives']} |",
        f"| Negatywy z tekstem bez geometrii | {summary['negatives_with_text_without_geometry']} |",
        f"| Błędy i przerwania wybranego MT | {summary['execution_failures']} |", '',
        'Referencja i pokrycie: '+json.dumps(summary['planned_reference_counts'],ensure_ascii=False),
        'Nieocenialne tablice: '+json.dumps(summary['unassessable_plates_by_reason'],ensure_ascii=False),
        '', 'Ograniczenia integracji: '+', '.join(report['go_reasons']),
        'Pełna droga dowodowa: evaluation.json; wyniki per fotografia i tablica: CSV.']
    if report.get('presentation_plan'):
        timing=report['presentation_plan']
        text.extend(['',f"Czas zadany: {timing['dwell_seconds']:g} s/fotografię. Nominalny plan: {timing['planned_total_seconds']:g} s ({ref.format_seconds(timing['planned_total_seconds'])}). Wykonane przedziały PC: presentation_timeline.csv."])
    if report.get('reference_preflight'):
        gt=report['reference_preflight'];completed=sum(p['status']=='COMPLETED' for p in report.get('presentation_timeline',[]))
        text=['# Wyniki pokazu i ocena względem GT','',
            f"Sesja: {report.get('session_id','N/A')}. Run: {report.get('run_id','N/A')}.",
            f"Pokaz PC: {completed}/{summary['planned_exposures']} COMPLETED; Android: {summary['identified_exposures']} zidentyfikowanych fotografii.",
            f"GT: {gt['mode']}; {gt['counts']}. Metryki: {report['metrics_status']}.",
            'Automatyczna ocena względem istniejącego GT. Bez ręcznych zatwierdzeń i bez utożsamiania wybranego MT z końcowym HUD.',
            '', '| Miara | Wynik |','|---|---|',
            f"| Pełna skuteczność | {shown(system['rate'])} |",
            f"| Pełne GT tekstu / pokryty mianownik | {shown(system['denominator'])} / {system['covered_denominator']} |",
            f"| TP / FP / FN | {shown(det['tp'])} / {shown(det['fp'])} / {shown(det['fn'])}; {det['scope']} |",
            f"| Odczyt evaluated_subset | {shown(summary['evaluated_subset_full_read']['rate'])}; n={summary['evaluated_subset_full_read']['denominator']} |",
            f"| CER dopasowanych tablic | {shown(summary['conditional_mz_cer']['value'])}; GT znaków={summary['conditional_mz_cer']['gt_characters']} |",
            '', 'Brak GT nie oznacza NO_PLATES ani FN. Brak dowodów nie oznacza zera detekcji. Znane liczby wywołań nie są liczbą poprawnych odczytów.',
            'Powody N/A i oddzielne statusy GT/dowodów znajdują się we wszystkich wierszach exposures.csv.',
            'Surowe wywołania: raw_mt.csv / raw_mz.csv. GT: reference_coverage.csv. Powiązania: frame_evidence.csv. Pełne dane: evaluation.json.',
            'Ta analiza nie zmienia oryginalnego runu, referencji ani wcześniejszych raportów.']
    if report.get('readability_preflight'):
        diagnostic=report['readability_preflight']['summary']
        text.extend(['',f"Czytelność: para L/R + geometria {diagnostic['verified_pairs']}/{diagnostic['source_frames']} pełnych klatek; "
            f"L={diagnostic['left_decoded']}, R={diagnostic['right_decoded']} (badane indywidualnie: {diagnostic['individually_tested']}).",
            f"MT bez identyfikacji fazy/fotografii: {diagnostic['unlocated_mt_count']}; udowodnione pierwszeństwo: {diagnostic['first_mt_order_proven']}/{len(report['rows'])}.",
            'Szczegóły: readability_preflight.json, readability_frames.csv i readability_first_mt.csv. Odbiór rozdzielczości/AF/A12: NOT_VERIFIED.'])
    if report.get('first_mt_scope'):
        labels=dict(UNKNOWN_MT_STRICTLY_LATER='Nieznane MT istnieje, ale źródło i START są późniejsze.',
            UNKNOWN_MT_MAY_BE_EARLIER='Nieznane MT może być wcześniejsze lub kolejność jest nieudowodniona.',
            UNBOUNDED_MT_SOURCE_LOSS='Brak zamkniętego, wiarygodnego rejestru START.',NO_UNIDENTIFIED_MT='Brak nieprzypisanego MT.')
        text.extend(['','| Fotografia | Kwalifikacja pierwszego MT | Dowód |','|---|---|---|'])
        for row in report['rows']:
            reason=(row.get('first_mt_risk') or {}).get('reason','')
            qualified='PASS_SOURCE_ORDER' if row.get('evidence') is not None else 'N/A'
            text.append(f"| {row['exposure_id']} | {qualified} | {labels.get(reason,reason or row.get('evidence_reason',''))} |")
    (target/'report.md').write_text('\n'.join(text),'utf-8')
    _atomic_write_json(target/'manifest.json',dict(schema='alpr.final.result_manifest.v1',result_sha256=digest,
        created_utc=ref.utc(),files={p.name:photo.digest(p) for p in target.iterdir() if p.is_file()},
        algorithm_sha256={name:photo.digest(Path(__file__).with_name(name)) for name in (
            'final180_mobile.py','final180_reference.py','final180_markers.py','mobile_package_experiments.py',
            'mobile_tv_bbox_evidence.py','mobile_f5a_scene_gt.py','mt_eval_metrics.py')}))
    return target
