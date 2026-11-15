"""Optional F6 lifecycle extension of the native alprsession reader.

The producer contract is F6_FINAL180_SPLIT_V1. JSONL is hashed and read a
record at a time. Summaries retain identities/times, not SYNTHETIC_LOAD bodies.
Legacy F5A limits and qualification are unchanged when this capability is absent.
"""
import hashlib
import json
import math
from .mobile_scene_evidence import safe_path, integer, stamp, _unique_fields

CAPABILITY='f6_final180_v1'
FILE='samples/f6_execution.jsonl'
SCHEMA='alpr.f6_execution_event.v1'
MAX_RECORD_BYTES=32*1024*1024
# The completed producer delivery contains a 34,608,299-byte scene object.
# Only the F6 scene index may use this bounded 64 MiB row budget. Native ZIP,
# image and legacy F5A limits remain unchanged; peak memory is measured in QA.
MAX_SCENE_RECORD_BYTES=64*1024*1024
EVENTS={'FRAME_OPENED','FRAME_FINISHED','MT_STARTED','MT_RETURNED','MT_DECODED','MT_FAILED',
        'MZ_STARTED','MZ_RETURNED','MZ_FAILED','SCENE_OBSERVATION','SYNTHETIC_LOAD'}


def require(ok,reason):
    if not ok:raise ValueError('F6_CONTRACT: '+reason)


def declared(descriptor,session):
    value=(descriptor.get('capabilities') or {}).get(CAPABILITY,False)
    require(type(value) is bool,'Capability must be boolean')
    require(value==(session.get('capture_profile')==CAPABILITY),'Capture profile/capability conflict')
    return value


def loss_declared(session):
    summary=session.get('f6_evidence') or {}
    return (summary.get('complete') is False and summary.get('pending_loss_unknown') is True
            and (summary.get('projection_error_count',0)>0 or summary.get('lifecycle_loss_count',0)>0
                 or session.get('scene_evidence_integrity_loss_count',0)>0))


def checked_rows(archive,name,manifest,*,allow_malformed=False,diagnostics=None,loader=None):
    name=safe_path(name);info=archive.getinfo(name);hasher=hashlib.sha256();count=0;offset=0
    limit=MAX_SCENE_RECORD_BYTES if name=='samples/scene_evidence.jsonl' else MAX_RECORD_BYTES
    loader=loader or (lambda line:json.loads(line,object_pairs_hook=_unique_fields,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON'))))
    with archive.open(info) as stream:
        while True:
            raw=stream.readline(limit+1)
            if not raw:break
            require(len(raw)<=limit,'JSONL record exceeds defensive limit: '+name)
            hasher.update(raw);count+=len(raw)
            try:
                text=raw.decode('utf-8');require(not text.startswith('\ufeff'),'JSONL BOM')
                if text.strip():
                    value=loader(text);require(isinstance(value,dict),'JSONL object required');yield value
            except (ValueError,UnicodeError) as exc:
                if not allow_malformed:raise
                if diagnostics is not None:diagnostics.append(dict(file=name,byte_offset=offset,bytes=len(raw),reason=str(exc),sha256=hashlib.sha256(raw).hexdigest()))
            offset+=len(raw)
    require(count==info.file_size and hasher.hexdigest()==(manifest.get('entry_sha256') or {}).get(name),'JSONL SHA/size mismatch: '+name)


def checked_hash(archive,name,manifest):
    info=archive.getinfo(safe_path(name));digest=hashlib.sha256();size=0
    with archive.open(info) as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk);size+=len(chunk)
    require(size==info.file_size and digest.hexdigest()==manifest.get('entry_sha256',{}).get(name),'Entry SHA mismatch: '+name)
    return dict(bytes=size,sha256=digest.hexdigest())


def read_extension(archive,descriptor,session,manifest,report):
    from . import mobile_tv_bbox_evidence as tv
    active=declared(descriptor,session)
    if not active:
        require(FILE not in archive.namelist(),'Undeclared F6 lifecycle')
        return dict(available=False,status='ABSENT')
    require(descriptor.get('f6_execution_file')==FILE and descriptor.get('f6_time_domain')=='ANDROID_ELAPSED_REALTIME'
        and descriptor.get('f6_time_unit')=='ns','Lifecycle path/time contract')
    require(report.get('research_collection')==session,'Session/report identity conflict')
    require(session.get('input_mode')=='TV_BBOX_CROP','F6 requires explicit source/crop context')
    execution=session.get('execution') or {}
    # The device's FINAL label is not an ROI policy. Keep it unchanged, and
    # admit it only with the explicitly recorded R0 policy and matching session.
    r0=execution.get('variant')=='R0' or (execution.get('variant')=='R0_FINAL'
        and session.get('variant')=='R0_FINAL' and execution.get('roi_budget_policy')=='r0_full_frame')
    require(r0 and execution.get('stages',{}).get('mp',{}).get('enabled') is False,'F6 is recorded R0/MP OFF')
    # Decimal-preserving read is needed for the producer's canonical seal.
    raw_session=tv.loads(archive.read('session.json').decode('utf-8-sig'))
    roi=raw_session.get('execution',{}).get('f6_photo_roi')
    require(isinstance(roi,dict) and roi.get('schema')=='alpr.f6_photo_roi.v1','Missing photo ROI descriptor')
    frozen=raw_session['tv_bbox_crop'];require(roi.get('digest_sha256')==tv.digest(roi),'Photo ROI seal mismatch')
    require(roi.get('coordinate_space')=='camera_full_bitmap_xy' and roi.get('tv_bbox_crop_digest_sha256')==frozen['digest_sha256'],'Photo ROI space/TV binding')
    w,h=roi['source_width_px'],roi['source_height_px']
    require((w,h)==(frozen['source_width_px'],frozen['source_height_px']),'Photo ROI source dimensions')
    corners=roi.get('corners_norm');require(isinstance(corners,list) and len(corners)==4,'Photo ROI corners')
    xy=[(tv.number(p['x'])*w,tv.number(p['y'])*h) for p in corners]
    require(all(0<=x<=w and 0<=y<=h for x,y in xy),'Photo corners outside source')
    bounds=[math.floor(min(x for x,y in xy)),math.floor(min(y for x,y in xy)),math.ceil(max(x for x,y in xy)),math.ceil(max(y for x,y in xy))]
    actual=[roi['photo_roi_camera_xy'][k] for k in ('left','top','right','bottom')]
    require(actual==bounds and actual[0]<actual[2] and actual[1]<actual[3],'Photo ROI floor/ceil conflict')
    outer=frozen['tv_bbox_px'];require(outer['left']<=actual[0]<actual[2]<=outer['right'] and outer['top']<=actual[1]<actual[3]<=outer['bottom'],'Photo ROI outside TV context')
    summary=session.get('f6_evidence') or {}
    require(summary.get('schema')=='alpr.f6_evidence_summary.v1','F6 summary schema')
    for key in ('complete','pending_loss_unknown','unlocated_loss_possible'):require(type(summary.get(key)) is bool,'Missing completeness flag: '+key)
    for key in ('events_persisted','mt_started_count','projection_error_count'):integer(summary[key])
    frames={};calls={};mz_calls={};malformed=[];last=0;gaps=[];counts={};persisted=0
    for row in checked_rows(archive,FILE,manifest,allow_malformed=loss_declared(session),diagnostics=malformed):
        require(row.get('schema')==SCHEMA and row.get('session_id')==session['session_id'],'Lifecycle schema/session')
        sequence=integer(row['event_sequence']);require(sequence>last,'Duplicate/nonmonotonic lifecycle sequence')
        if sequence!=last+1:gaps.append([last+1,sequence-1])
        last=sequence;persisted+=1;event=row['event'];require(event in EVENTS,'Unknown lifecycle event')
        counts[event]=counts.get(event,0)+1;elapsed=integer(row['elapsed_nanos'])
        require(row.get('time_domain')=='ANDROID_ELAPSED_REALTIME','Lifecycle clock domain')
        st=row.get('source_stamp');key=row.get('frame_key')
        if st is not None:stamp(st)
        if key is not None:
            require(st is not None and key==tv.digest(dict(source_stamp=st,snapshot=frozen['digest_sha256'])),'Lifecycle frame key/stamp binding')
        point=dict(event_sequence=sequence,elapsed_nanos=elapsed,time_domain=row['time_domain'],event=event)
        if event=='SYNTHETIC_LOAD':continue
        if key:
            frame=frames.setdefault(key,dict(source_stamp=st,mt_invocation_ids=[],event_sequences=[]))
            require(frame['source_stamp']==st,'Conflicting frame SourceStamp');frame['event_sequences'].append(sequence)
            if event=='FRAME_OPENED':
                require('opened' not in frame,'Duplicate frame OPEN');frame['opened']=point
            if event=='FRAME_FINISHED':
                require('finished' not in frame,'Duplicate frame FINISH');frame['finished']=dict(point,details=row.get('details') or {})
        gid=row.get('mt_invocation_id');aid=row.get('attempt_id')
        if event.startswith('MT_'):
            require(isinstance(gid,str) and gid and key and isinstance(aid,str) and aid,'MT lifecycle identities')
            call=calls.setdefault(gid,dict(mt_invocation_id=gid,attempt_id=aid,frame_key=key,source_stamp=st,events={}))
            require((call['frame_key'],call['source_stamp'],call['attempt_id'])==(key,st,aid),'MT identity conflict')
            require(event not in call['events'],'Duplicate MT event');call['events'][event]=dict(point,details=row.get('details') or {})
            if event=='MT_STARTED':frames[key]['mt_invocation_ids'].append(gid)
        if event.startswith('MZ_'):
            mid=row.get('mz_invocation_id');require(isinstance(mid,str) and mid and gid and aid and key,'MZ lifecycle identities')
            call=mz_calls.setdefault(mid,dict(mz_invocation_id=mid,mt_invocation_id=gid,attempt_id=aid,frame_key=key,source_stamp=st,events={}))
            require((call['frame_key'],call['source_stamp'],call['attempt_id'],call['mt_invocation_id'])==(key,st,aid,gid),'MZ identity conflict')
            require(event not in call['events'],'Duplicate MZ event');call['events'][event]=dict(point,details=row.get('details') or {})
    incomplete=[]
    for gid,call in calls.items():
        events=call['events'];require('MT_STARTED' in events,'MT outcome without backend START')
        start=events['MT_STARTED'];require(all(e['event_sequence']>=start['event_sequence'] and e['elapsed_nanos']>=start['elapsed_nanos'] for e in events.values()),'MT lifecycle order')
        require(not ('MT_FAILED' in events and ('MT_RETURNED' in events or 'MT_DECODED' in events)),'MT failure/return conflict')
        if 'MT_DECODED' in events:
            require('MT_RETURNED' in events,'MT decoded without backend return');integer(events['MT_DECODED']['details']['raw_detection_count'])
            require(events['MT_DECODED']['event_sequence']>events['MT_RETURNED']['event_sequence'] and
                events['MT_DECODED']['elapsed_nanos']>=events['MT_RETURNED']['elapsed_nanos'],'Decode before MT return')
        call['status']='ERROR' if 'MT_FAILED' in events else 'DECODED' if 'MT_DECODED' in events else 'INTERRUPTED'
        if call['status']=='INTERRUPTED':incomplete.append(gid)
    for mid,call in mz_calls.items():
        ev=call['events'];require('MZ_STARTED' in ev and call['mt_invocation_id'] in calls,'MZ without START/MT parent')
        require(not ('MZ_RETURNED' in ev and 'MZ_FAILED' in ev),'MZ outcome conflict')
        start=ev['MZ_STARTED'];require(all(e['event_sequence']>=start['event_sequence'] and e['elapsed_nanos']>=start['elapsed_nanos'] for e in ev.values()),'MZ lifecycle order')
        parent=calls[call['mt_invocation_id']]['events'].get('MT_DECODED')
        require(parent and start['event_sequence']>parent['event_sequence'] and start['elapsed_nanos']>=parent['elapsed_nanos'],'MZ started before decoded MT')
        call['status']='ERROR' if 'MZ_FAILED' in ev else 'RETURNED' if 'MZ_RETURNED' in ev else 'INTERRUPTED'
        if call['status']=='INTERRUPTED':incomplete.append(mid)
    for key,frame in frames.items():
        if 'opened' not in frame or 'finished' not in frame:incomplete.append(key)
    require(summary.get('events_persisted')==persisted and summary.get('mt_started_count')==counts.get('MT_STARTED',0),'Lifecycle summary counts')
    loss=bool(malformed or gaps or incomplete or summary['pending_loss_unknown'] or summary['unlocated_loss_possible']
        or summary.get('projection_error_count') or summary.get('lifecycle_loss_count'))
    require(not summary['complete'] or not loss,'False complete F6 lifecycle claim')
    return dict(available=True,status='PARTIAL' if loss or not summary['complete'] else 'VALID',
        summary=summary,photo_roi=tv.plain(roi),frames=frames,mt_invocations=calls,mz_invocations=mz_calls,
        malformed_records=malformed,sequence_gaps=gaps,unmatched_lifecycle=incomplete,
        unlocated_loss_possible=bool(summary['unlocated_loss_possible'] or malformed or gaps),
        event_counts=counts,events_persisted=persisted,source_file=FILE,source_sha256=manifest['entry_sha256'][FILE])


def validate_bindings(archive,manifest,session,evidence,tv,scene):
    """Join lifecycle to the native, already validated TV/role/attempt records."""
    import io
    from PIL import Image
    from .mobile_f5a_assisted import STAMP
    require(tv.get('available') and scene.get('available'),'Missing TV/scene extension')
    attempts={a['attempt_id']:a for a in checked_rows(archive,'samples/attempts.jsonl',manifest,
        allow_malformed=loss_declared(session))}
    rows={r['frame_key']:r for r in tv['rows']};roles={r['mt_invocation_id']:r for r in scene['roles']}
    photo_roi=[evidence['photo_roi']['photo_roi_camera_xy'][k] for k in ('left','top','right','bottom')]
    missing=[];names=set(archive.namelist());evidence['mz_by_attempt']={}
    for key,frame in evidence['frames'].items():
        if key not in rows:missing.append('LIFECYCLE_WITHOUT_FRAME:'+key);continue
        require(frame['source_stamp']==rows[key]['source_stamp'],'Lifecycle/TV stamp conflict')
    for aid,a in attempts.items():
        if a.get('mt_executed') is not True:continue
        gid=a.get('mt_invocation_id');call=evidence['mt_invocations'].get(gid)
        require(call is not None,'Attempt without actual MT_STARTED')
        require(call['frame_key']==a['tv_bbox_frame_key'] and all(a[k]==call['source_stamp'][k] for k in STAMP),'Attempt/lifecycle source conflict')
        require([a['roi_'+k] for k in ('left','top','right','bottom')]==photo_roi,'Actual MT ROI differs from frozen photo ROI')
        started=integer(a['mt_started_elapsed_nanos'])
        require(integer(a['attempt_started_elapsed_nanos'])<=started<=call['events']['MT_STARTED']['elapsed_nanos'],'MT timing order')
        entry=a.get('mt_input_evidence_entry');kind=a.get('mt_input_evidence_kind')
        if entry==a.get('evidence_entry'):kind=kind or a.get('evidence_kind')
        input_record=dict(attempt_id=aid,entry=entry,kind=kind,roi_camera_xy=photo_roi,
            input_width=a.get('input_width'),input_height=a.get('input_height'),input_scale=a.get('input_scale'),
            input_pad_x=a.get('input_pad_x'),input_pad_y=a.get('input_pad_y'),status='MISSING')
        if entry and entry in names and kind=='mt_input_roi':
            info=archive.getinfo(safe_path(entry));require(info.file_size<=20*1024*1024,'MT input image exceeds limit')
            checked=checked_hash(archive,entry,manifest)
            with Image.open(io.BytesIO(archive.read(entry))) as image:
                require(image.format=='JPEG' and image.size==(a['input_width'],a['input_height']) and max(image.size)<=12000 and image.width*image.height<=40_000_000,'Actual MT JPEG dimensions/format')
                image.load()
            input_record.update(status='WRITTEN',**checked)
        else:missing.append('ACTUAL_MT_INPUT_EVIDENCE_MISSING:'+aid)
        call.setdefault('inputs',[]).append(input_record)
        mid=a.get('f6_mz_invocation_id')
        if a.get('mz_executed') is True:
            mz=evidence['mz_invocations'].get(mid)
            require(mz and mz['attempt_id']==aid and mz['mt_invocation_id']==gid and mz['source_stamp']==call['source_stamp'],'MZ lifecycle binding')
            evidence['mz_by_attempt'][aid]=mid
    for gid,call in evidence['mt_invocations'].items():
        role=roles.get(gid);decoded=call['events'].get('MT_DECODED')
        call['final_population_complete']=bool(role and role.get('role_proof_status')=='COMPLETE' and role.get('postprocess_status')=='COMPLETED' and decoded)
        if role:
            require(role['source_stamp']==call['source_stamp'] and role['roi']==photo_roi,'Role/lifecycle source ROI conflict')
            if decoded:require(len(role['raw_attempt_ids'])==decoded['details']['raw_detection_count'],'Decoded/RAW population conflict')
        else:missing.append('START_WITHOUT_ROLE:'+gid)
        if not call.get('inputs'):missing.append('START_WITHOUT_INPUT:'+gid)
    for mid,mz in evidence['mz_invocations'].items():
        if mz['attempt_id'] not in attempts:missing.append('MZ_START_WITHOUT_ATTEMPT:'+mid)
    evidence['binding_gaps']=missing
    evidence['completeness']=dict(lifecycle=evidence['status'],tv=tv['status'],scene=scene['status'],
        attempts='PARTIAL' if missing else 'VALID',samples='VALID' if session.get('collection_complete') is True and
        not any(session.get(k) for k in ('dropped_sample_count','dropped_telemetry_count','integrity_loss_count')) else 'PARTIAL')
    if missing or any(v!='VALID' for v in evidence['completeness'].values()):evidence['status']='PARTIAL'
