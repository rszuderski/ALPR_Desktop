"""Evidence-backed distinction between raw MT boxes and post-dedup detections.

No human GT is inferred from IoU or OCR. Full-source telemetry must prove
candidate membership in the same call. Legacy single-rejection evidence is
retained verbatim; multiple rejections require explicit per-ID source evidence.
Never edit original .alprsession, individual MT outputs or stored review GT.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json
import math
import zipfile

DETECTION_STATUSES = frozenset({"VALID_QUAD", "DETECTION_INVALID_QUAD"})


def bbox_iou(a, b):
    keys = ("plate_left", "plate_top", "plate_right", "plate_bottom")
    try:
        u, v = tuple(float(a[k]) for k in keys), tuple(float(b[k]) for k in keys)
    except (KeyError, TypeError, ValueError, OverflowError):
        return 0.0
    if not all(math.isfinite(x) for x in u+v) or u[2]<=u[0] or u[3]<=u[1] or v[2]<=v[0] or v[3]<=v[1]:
        return 0.0
    area_a=(u[2]-u[0])*(u[3]-u[1]); area_b=(v[2]-v[0])*(v[3]-v[1])
    inter=max(0, min(u[2],v[2])-max(u[0],v[0]))*max(0,min(u[3],v[3])-max(u[1],v[1]))
    union=area_a+area_b-inter
    return inter/union if union>0 else 0.0


def _source_integer(value):
    # Full Android attempts.csv is read as strings; JSON telemetry stores ints.
    if type(value) is int and value >= 0:
        return value
    if isinstance(value,str) and value.isascii() and value.isdecimal():
        return int(value)
    return None


def _positive_integer(obj, name):
    return _source_integer(obj.get(name))


@lru_cache(maxsize=20)
def _trace_index(path, source_sha, size, mtime_ns):
    """Only already SHA-validated session ZIP; no extraction and 20 MiB cap."""
    found={}
    if not path or not source_sha:
        return found
    try:
        file=Path(path)
        stat=file.stat()
        if stat.st_size != size or stat.st_mtime_ns != mtime_ns:
            return found
        with zipfile.ZipFile(file) as archive:
            info=archive.getinfo("telemetry/traces.jsonl")
            if info.file_size > 20*1024*1024:
                return found
            with archive.open(info) as stream:
                for line in stream:
                    if len(line) > 50_000:
                        continue
                    try:
                        row=json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if not isinstance(row,dict):
                        continue
                    key=(row.get("source_sequence"),row.get("source_timestamp_ns"),row.get("scene_generation"))
                    if any(v is None for v in key):
                        continue
                    # A repeated key is ambiguous, so no automatic attribution.
                    found.setdefault(key,[]).append(row)
    except (OSError,zipfile.BadZipFile,KeyError):
        return {}
    return found


def post_nms_proof(session, invocation_id):
    """Return exact traced suppression (primary, extra, counters) or None.

    Narrow-by-design: a 2->1 static invocation with one final MZ inference.
    This is the *one* documented F4 case, not a general-purpose box classifier.
    """
    group=session.mt_invocations.get(invocation_id)
    if not group or not group.executed or group.cancelled or group.execution_failed:
        return None
    if group.detection_count != 2 or len(group.records) != 2:
        return None
    rows=sorted(group.records,key=lambda r:int(r.get("mt_detection_index") or -1))
    if [_source_integer(r.get("mt_detection_index")) for r in rows]!=[0,1]:
        return None
    if any(str(r.get("mt_status") or "").upper() not in DETECTION_STATUSES for r in rows):
        return None
    if bbox_iou(*rows) < .90:
        return None
    left,right=rows
    if (str(left.get("mz_status") or "").upper() != "READ" or
            str(right.get("mz_status") or "").upper() != "NOT_RUN" or
            session.sample_ids_by_attempt.get(str(right.get("id") or ""))):
        return None
    try:
        if float(left.get("plate_confidence")) < float(right.get("plate_confidence")):
            return None
    except (TypeError,ValueError):
        return None
    # Same source/identity for both child records, not merely visual overlap.
    fields=("source_sequence","source_timestamp_nanos","scene_generation","mt_invocation_id")
    if any(left.get(k) is None or left.get(k) != right.get(k) for k in fields):
        return None
    path=getattr(session,"path",None)
    bundle=getattr(session,"bundle",None)
    source_sha=getattr(bundle,"source_archive_sha256","")
    if path is None or not source_sha:
        return None
    try:
        stat=Path(path).stat()
    except OSError:
        return None
    index=_trace_index(str(path),source_sha,stat.st_size,stat.st_mtime_ns)
    key=(_source_integer(left["source_sequence"]),_source_integer(left["source_timestamp_nanos"]),
         _source_integer(left["scene_generation"]))
    if any(v is None for v in key):
        return None
    matches=index.get(key,[])
    if len(matches) != 1:
        return None
    trace=matches[0]
    counters=trace.get("counters") or {}
    expected={"plate_detections_raw":2,"plate_detections_after_dedup":1,
              "plate_detections_suppressed":1,"mt_runs_this_frame":1,"mz_runs":1}
    if not isinstance(counters,dict) or any(_positive_integer(counters,k)!=v for k,v in expected.items()):
        return None
    # Trace itself is a provenance record for the *whole invocation*.
    return {"primary_id":str(left["id"]),"suppressed_id":str(right["id"]),
            "mt_invocation_id":invocation_id,"trace_frame_id":trace.get("frame_id"),
            "source_sequence":key[0],"bbox_iou":bbox_iou(left,right),
            "raw":2,"after_dedup":1,"suppressed":1}


def eligible_suppressed_id(session, invocation_id):
    """One physical plate still requires an explicit operator cardinality decision."""
    evidence=post_nms_proof(session,invocation_id)
    if not evidence:
        return ""
    decision=session.invocation_annotation(invocation_id)
    # Canvas inventory has its own independently reviewed physical plates;
    # the diagnostic legacy-subject path MUST NOT modify canvas scoring.
    if isinstance(decision.get("canvas_review_v1"), dict):
        return ""
    if decision.get("visible_plate_count") != "one" or decision.get("evaluable") is False:
        return ""
    # Operator's single-plate count + exact post-NMS telemetry may co-exist
    # with any previous GT sidecar values; no writes or auto-migration.
    return evidence["suppressed_id"]


def canvas_post_nms_proof(session, invocation_id):
    """Keep historical proofs stable; accept complete explicit multi-ID evidence.

    MZ can run only on retained post-dedup candidates in the archived static
    pipeline. If every retained candidate has its own executed MZ/crop record,
    the single remaining raw record is uniquely identified by subtraction.
    NOT_RUN or overlap alone is never sufficient. No NMS is rerun here.
    """
    if (getattr(getattr(session, 'bundle', None), 'scene_evidence', {}) or {}).get('available'):
        return _producer_role_proof(session, invocation_id)
    diagnostic,trace,_ = _role_context(session, invocation_id)
    if diagnostic.get('explicit_role_declaration_present'):
        return _explicit_role_proof(session, invocation_id)
    old = post_nms_proof(session, invocation_id)
    if old:
        return old
    if getattr(session,'path',None) is None:return None
    group = session.mt_invocations.get(invocation_id)
    if not group or not group.executed or group.cancelled or group.execution_failed:
        return None
    from .mobile_source_frame_roi import source_invocation_context
    if not source_invocation_context(group.records, invocation_id, session.entry_names):
        return None
    rows = list(group.records)
    raw = len(rows)
    if raw < 2 or group.detection_count != raw:
        return None
    indices=[_source_integer(r.get('mt_detection_index')) for r in rows]
    if any(i is None for i in indices) or sorted(indices) != list(range(raw)):
        return None
    key = tuple(_source_integer(rows[0].get(k)) for k in ('source_sequence','source_timestamp_nanos','scene_generation'))
    if any(v is None for v in key):return None
    if any(tuple(_source_integer(r.get(k)) for k in ('source_sequence','source_timestamp_nanos','scene_generation')) != key for r in rows):
        return None
    stat = session.path.stat()
    matches = _trace_index(str(session.path),session.bundle.source_archive_sha256,stat.st_size,stat.st_mtime_ns).get(key,[])
    if len(matches) != 1:return None
    trace = matches[0]; counters = trace.get('counters',{})
    expected = dict(plate_detections_raw=raw,plate_detections_after_dedup=raw-1,
                    plate_detections_suppressed=1,mt_runs_this_frame=1,mz_runs=raw-1)
    if any(_source_integer(counters.get(k)) != v for k,v in expected.items()):return None
    if any(_source_integer(r.get(k)) != _source_integer(trace.get(k)) or _source_integer(r.get(k)) is None
           for r in rows for k in ('visual_epoch','camera_transform_generation')):return None
    kept=[];rejected=[]
    for row in rows:
        aid=str(row.get('id') or row.get('attempt_id') or '')
        if row.get('mt_status') not in DETECTION_STATUSES or str(row.get('mt_executed')).lower() != 'true':return None
        samples=[session.samples[sid] for sid in session.sample_ids_by_attempt.get(aid,()) if sid in session.samples]
        ran=str(row.get('mz_executed')).lower() == 'true'
        if (ran and row.get('rectification_status') == 'OK' and row.get('mz_status') in {'READ','NO_CHARACTERS'}
                and samples and all(str(s.get('attempt_id')) == aid and not session.cancelled(s) for s in samples)):
            kept.append(aid)
        elif not ran and row.get('mz_status') == 'NOT_RUN' and row.get('rectification_status') == 'NOT_RUN' and not samples:
            rejected.append(aid)
        else:return None
    if len(kept) != raw-1 or len(rejected) != 1:return None
    return dict(schema='alpr.postdedup.executed_complement.v1',mt_invocation_id=invocation_id,
                source_archive_sha256=session.bundle.source_archive_sha256,trace_frame_id=trace.get('frame_id'),
                source_sequence=key[0],source_timestamp_nanos=key[1],scene_generation=key[2],
                final_ids=sorted(kept),suppressed_id=rejected[0],raw=raw,after_dedup=raw-1,suppressed=1)


def suppressed_ids(proof):
    """Compatibility adapter. Never truncate a proven population to one ID."""
    if not proof:return []
    return list(proof['suppressed_ids']) if 'suppressed_ids' in proof else [proof['suppressed_id']] if proof.get('suppressed_id') else []


def final_ids(proof):
    if not proof:return []
    return list(proof['final_ids']) if 'final_ids' in proof else [proof['primary_id']] if proof.get('primary_id') else []


def _role_context(session,gid):
    """A trace counter is not a per-candidate decision. Validate its scope first."""
    out=dict(proof_status='UNPROVEN',reason='SOURCE_FINAL_ROLE_UNPROVEN',
        reasons=[],raw_count=None,final_count=None,suppressed_count=None,
        final_ids=[],suppressed_ids=[],final_identity_count=0,suppressed_identity_count=0)
    group=session.mt_invocations.get(gid)
    if not group or not group.executed or group.cancelled or group.execution_failed:
        out['reasons']=['MT_NOT_COMPLETED'];return out,None,None
    try:
        from .mobile_source_frame_roi import source_invocation_context
        rows=list(group.records)
        ids=[str(r.get('id') or r.get('attempt_id') or '') for r in rows]
        if not rows or not all(ids) or len(set(ids))!=len(ids):raise ValueError('ATTEMPT_ID_AMBIGUOUS')
        key=tuple(_source_integer(rows[0].get(k)) for k in ('source_sequence','source_timestamp_nanos','scene_generation'))
        if any(v is None for v in key) or any(tuple(_source_integer(r.get(k)) for k in
                ('source_sequence','source_timestamp_nanos','scene_generation'))!=key for r in rows):
            raise ValueError('ATTEMPT_TRACE_IDENTITY_CONFLICT')
        stat=session.path.stat()
        matches=_trace_index(str(session.path),session.bundle.source_archive_sha256,stat.st_size,stat.st_mtime_ns).get(key,[])
        if len(matches)!=1:raise ValueError('TRACE_KEY_MISSING_OR_AMBIGUOUS')
        trace=matches[0];c=trace.get('counters') or {}
        out['explicit_role_declaration_present']='source_roles_v2' in trace
        ctx=source_invocation_context(group.records,gid,session.entry_names)
        if not ctx:raise ValueError('SOURCE_FRAME_ROI_UNPROVEN')
        raw,final,rejected=(_source_integer(c.get(k)) for k in
            ('plate_detections_raw','plate_detections_after_dedup','plate_detections_suppressed'))
        out.update(raw_count=raw,final_count=final,suppressed_count=rejected,trace_frame_id=trace.get('frame_id'),
            source_sequence=key[0],source_timestamp_nanos=key[1],scene_generation=key[2],
            image_entry=ctx['entry'],roi=list(ctx['roi']),attempt_ids=ids,
            mz_runs=_source_integer(c.get('mz_runs')))
        if any(v is None for v in (raw,final,rejected)):raise ValueError('NMS_COUNTERS_MISSING')
        if raw!=final+rejected:
            out['proof_status']='CONFLICT';raise ValueError('NMS_COUNTERS_CONFLICT')
        no_detection=(raw==0 and group.detection_count==0 and len(rows)==1
            and str(rows[0].get('mt_status')).upper()=='NO_DETECTION')
        if any((str(r.get('mt_status') or '').upper() not in DETECTION_STATUSES and not no_detection)
               or str(r.get('mt_executed')).lower()!='true' for r in rows):
            raise ValueError('MT_CANDIDATE_NOT_COMPLETED')
        if not no_detection and (raw!=len(rows) or group.detection_count!=raw
                or sorted(_source_integer(r.get('mt_detection_index')) if _source_integer(r.get('mt_detection_index')) is not None else -1 for r in rows)!=list(range(raw))):
            raise ValueError('RAW_ATTEMPT_POPULATION_INCOMPLETE')
        if no_detection:out['attempt_ids']=[]  # The invocation sentinel is not a detection.
        if _source_integer(c.get('mt_runs_this_frame'))!=1:raise ValueError('MULTIPLE_MT_EXECUTIONS_IN_TRACE')
        if any(_source_integer(r.get(k)) is None or _source_integer(r.get(k))!=_source_integer(trace.get(k))
               for r in rows for k in ('visual_epoch','camera_transform_generation')):
            raise ValueError('TRACE_EPOCH_TRANSFORM_CONFLICT')
        return out,trace,ctx
    except (ValueError,KeyError,TypeError,OSError,AttributeError) as exc:
        out['reasons'].append(str(exc));return out,None,None


def _explicit_role_proof(session,gid):
    """Versioned per-ID source protocol; no ranking, OCR inference or NMS replay.

    The archived v2 research-samples schema does not promise this protocol.
    Absence of these fields is missing evidence, never an implicit empty list.
    Synthetic acceptance fixtures exercise this contract without claiming that
    old Android archives actually exported it.
    """
    evidence,trace,ctx=_role_context(session,gid)
    if trace is None:return None
    declaration=trace.get('source_roles_v2')
    if not isinstance(declaration,dict):return None
    finals=declaration.get('final_ids');rejected=declaration.get('suppressed_ids')
    if (declaration.get('schema')!='alpr.source_nms_roles.v2' or declaration.get('complete') is not True
            or declaration.get('mt_invocation_id')!=gid or declaration.get('source_frame_entry')!=ctx['entry']
            or declaration.get('roi')!=list(ctx['roi']) or not isinstance(finals,list) or not isinstance(rejected,list)
            or any(not isinstance(v,str) or not v for v in finals+rejected)
            or len(set(finals+rejected))!=len(finals+rejected)
            or set(finals+rejected)!=set(evidence['attempt_ids'])
            or len(finals)!=evidence['final_count'] or len(rejected)!=evidence['suppressed_count']):return None
    # Explicit roles allow FINAL with failed rectification/unexecuted MZ.
    # Conversely, a source crop attached to SUPPRESSED contradicts this protocol.
    if any(str(session.attempts[aid].get('mz_executed')).lower()=='true'
           or session.sample_ids_by_attempt.get(aid) for aid in rejected):return None
    result=dict(schema='alpr.postdedup.explicit_ids.v2',mt_invocation_id=gid,
        source_archive_sha256=session.bundle.source_archive_sha256,trace_frame_id=trace.get('frame_id'),
        source_sequence=evidence['source_sequence'],source_timestamp_nanos=evidence['source_timestamp_nanos'],
        scene_generation=evidence['scene_generation'],final_ids=sorted(finals),suppressed_ids=sorted(rejected),
        raw=evidence['raw_count'],after_dedup=len(finals),suppressed=len(rejected),
        evidence_refs=['telemetry/traces.jsonl#'+str(trace.get('frame_id'))+'/source_roles_v2',
                       'samples/attempts.csv',ctx['entry']])
    if len(rejected)==1:result['suppressed_id']=rejected[0]
    return result



def _producer_role_proof(session, gid):
    """Adapt validated Android memberships; no selector or NMS replay."""
    role = getattr(session, 'mt_detection_roles', {}).get(gid)
    group = session.mt_invocations.get(gid)
    if (not role or not group or not group.executed or group.cancelled or group.execution_failed
            or role['stale_or_cancelled'] or role['execution_error']
            or role['role_proof_status'] != 'COMPLETE' or role['raw_attempt_ids'] is None):
        return None
    from .mobile_source_frame_roi import source_invocation_context
    ctx = source_invocation_context(group.records, gid, session.entry_names)
    if not ctx:
        return None
    final, suppressed = list(role['final_attempt_ids']), list(role['suppressed_attempt_ids'])
    return dict(schema='alpr.postdedup.explicit_ids.v2', provenance='ANDROID_MT_DETECTION_ROLES_V1',
        mt_invocation_id=gid, source_archive_sha256=session.bundle.source_archive_sha256,
        scene_event_id=role['scene_event_id'], trace_frame_id=role['trace_frame_id'],
        postprocess_scope_id=role['postprocess_scope_id'], postprocess_status=role['postprocess_status'],
        source_stamp=role['source_stamp'], image_entry=ctx['entry'], roi=list(ctx['roi']),
        final_ids=final, suppressed_ids=suppressed, raw=len(final)+len(suppressed),
        after_dedup=len(final), suppressed=len(suppressed),
        decoder_raw_ids=list(role['raw_attempt_ids']), rejected_before_nms=role['rejected_before_nms'])

def _producer_roles(session, gid):
    role = getattr(session, 'mt_detection_roles', {}).get(gid)
    proof = _producer_role_proof(session, gid)
    complete = proof is not None
    final = list(role['final_attempt_ids']) if role else []
    suppressed = list(role['suppressed_attempt_ids']) if role else []
    return dict(schema='alpr.desktop.source_roles.v2', adopted_date='2026-11-05',
        source_archive_sha256=session.bundle.source_archive_sha256, mt_invocation_id=gid,
        proof_status='PROVEN' if complete else 'UNPROVEN',
        reason='' if complete else 'MT_ROLE_OR_EXECUTION_INCOMPLETE',
        reasons=[] if complete else ['MT_ROLE_OR_EXECUTION_INCOMPLETE'],
        method='ANDROID_MT_DETECTION_ROLES_V1', mz_count_is_final_count=False,
        raw_count=(len(role['raw_attempt_ids']) if role and role['raw_attempt_ids'] is not None else None),
        final_count=len(final) if role else None, suppressed_count=len(suppressed) if role else None,
        final_ids=final if complete else [], suppressed_ids=suppressed if complete else [],
        recorded_final_ids=final, recorded_suppressed_ids=suppressed,
        rejected_before_nms=role['rejected_before_nms'] if role else [],
        unresolved_attempt_ids=role['unresolved_attempt_ids'] if role else [],
        final_identity_count=len(final) if complete else 0,
        suppressed_identity_count=len(suppressed) if complete else 0,
        missing_final_id_count=0 if complete else len(final),
        missing_suppressed_id_count=0 if complete else len(suppressed), proof=proof,
        stale_or_cancelled=role['stale_or_cancelled'] if role else None)


def source_roles_v2(session,gid):
    """Read-only diagnosis, including known counters and missing per-ID proof."""
    if (getattr(getattr(session, 'bundle', None), 'scene_evidence', {}) or {}).get('available'):
        return _producer_roles(session, gid)
    evidence,trace,ctx=_role_context(session,gid)
    evidence.update(schema='alpr.desktop.source_roles.v2',adopted_date='2026-11-02',
        source_archive_sha256=session.bundle.source_archive_sha256,mt_invocation_id=gid,
        method='explicit_per_id_or_unchanged_legacy_single_rejection',
        mz_count_is_final_count=False)
    proof=canvas_post_nms_proof(session,gid)
    if proof:
        finals=final_ids(proof);rejected=suppressed_ids(proof)
        evidence.update(proof_status='PROVEN',reason='',reasons=[],final_ids=finals,suppressed_ids=rejected,
            raw_count=proof['raw'],final_count=proof['after_dedup'],suppressed_count=proof['suppressed'],
            final_identity_count=len(finals),suppressed_identity_count=len(rejected),proof=proof)
    elif trace is not None and evidence['suppressed_count']==0:
        evidence.update(proof_status='PROVEN',reason='',reasons=[],final_ids=evidence['attempt_ids'],
            final_identity_count=evidence['raw_count'])
    elif trace is not None:
        evidence['reasons'].append('PER_CANDIDATE_NMS_IDENTITY_OR_EXECUTION_CONTRACT_MISSING'
            if 'source_roles_v2' not in trace else 'EXPLICIT_ROLE_DECLARATION_CONFLICT')
    evidence['missing_final_id_count']=None if evidence['final_count'] is None else evidence['final_count']-evidence['final_identity_count']
    evidence['missing_suppressed_id_count']=None if evidence['suppressed_count'] is None else evidence['suppressed_count']-evidence['suppressed_identity_count']
    return evidence


def unresolved_filter_reason(session,gid):
    """Known suppression without attributable IDs must not become a new READY."""
    if getattr(session,'path',None) is None:return ''
    group=session.mt_invocations[gid];row=group.records[0];stat=session.path.stat()
    key=tuple(_source_integer(row.get(k)) for k in ('source_sequence','source_timestamp_nanos','scene_generation'))
    matches=_trace_index(str(session.path),session.bundle.source_archive_sha256,stat.st_size,stat.st_mtime_ns).get(key,[])
    research='session.json' in getattr(session,'entry_names',())
    if len(matches)!=1 and research:
        return 'Brak jednoznacznego śladu końcowych wyników MT dla tej klatki.'
    if len(matches)==1:
        c=matches[0].get('counters',{});raw=_source_integer(c.get('plate_detections_raw'));final=_source_integer(c.get('plate_detections_after_dedup'))
        if research and (raw is None or final is None):
            return 'Brak liczników RAW i końcowych wskazań w śladzie klatki.'
        if raw is not None and final is not None and raw>final and not canvas_post_nms_proof(session,gid):
            return 'Źródło potwierdza odrzucone wskazania, ale nie pozwala ustalić ich identyfikatorów.'
    return ''


def suppressed_subject(session, subject):
    """A raw-only record contributes no separate physical-plate denominator."""
    ids=subject.get("attempts",())
    if not ids or subject.get("samples"):
        return False
    for aid in ids:
        gid=session.invocation_by_attempt.get(aid,"")
        if not gid or eligible_suppressed_id(session,gid) != aid:
            return False
    return True
