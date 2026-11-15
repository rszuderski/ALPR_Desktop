"""Presentation provenance for F4; never infer a research variant from a model.

The archived Android fields remain raw. The registry supplies a separate label
only for one verified archive identity, compatible session/scenario/repetition.
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import csv,hashlib,re

REGISTRY=Path(__file__).parent/'protocols/alpr_analysis_1_0/f4_registry.csv'
SCHEMA='alpr.mobile.f4_variant_provenance.v1'


def resolve(*,archive_sha,session_id,source_session_ids,source_scenarios,
            declared_variant=None,collection_variant=None,source_repeats=(),rows=(),source_verified=False):
    result=dict(schema=SCHEMA,status='UNCONFIRMED',provenance=None,archive_sha256=archive_sha,
        session_id=session_id,analysis_variant=None,scenario=None,repeat=None,source_variant=None,
        source_replicate_index=None,declared_android_variant=declared_variant,
        declared_collection_variant=collection_variant,label='NIEPOTWIERDZONE',reason=None)
    def fail(reason,conflict=False):
        result.update(status='CONFLICT' if conflict else 'UNCONFIRMED',
            label='KONFLIKT' if conflict else 'NIEPOTWIERDZONE',reason=reason)
        return result
    if not source_verified or not re.fullmatch(r'[a-f0-9]{64}',str(archive_sha)):
        return fail('SOURCE_SHA_UNVERIFIED')
    ids={str(v).strip() for v in source_session_ids if v is not None and str(v).strip()}
    if not session_id or ids!={session_id}:return fail('SOURCE_SESSION_ID_UNPROVEN')
    matches=[r for r in rows if str(r.get('archive_sha256','')).lower()==archive_sha]
    if len(matches)!=1:return fail('REGISTRY_SHA_DUPLICATE' if matches else 'REGISTRY_SHA_NOT_FOUND')
    row=matches[0]
    if (not all(row.get(k) for k in ('analysis_variant','source_variant','scenario','rNN_folder','source_replicate_index'))
            or not re.fullmatch(r'r[0-9]{2}',row['rNN_folder'])):
        return fail('REGISTRY_ROW_INVALID')
    if row.get('session_id') and row['session_id']!=session_id:return fail('REGISTRY_SESSION_CONFLICT',True)
    scenarios={str(v).strip() for v in source_scenarios if v is not None and str(v).strip()}
    if not scenarios:return fail('SOURCE_SCENARIO_UNPROVEN')
    if scenarios!={row['scenario']}:return fail('SOURCE_SCENARIO_CONFLICT',True)
    try:
        raw_repeats=[v for v in source_repeats if v is not None and str(v).strip()]
        if any(not re.fullmatch(r'[1-9][0-9]*',str(v)) for v in [row['source_replicate_index'],*raw_repeats]):
            raise ValueError('Invalid repeat')
        expected=int(row['source_replicate_index'])
        repeats=[int(v) for v in raw_repeats]
    except (ValueError,TypeError):return fail('SOURCE_REPEAT_INVALID',True)
    if not repeats:return fail('SOURCE_REPEAT_UNPROVEN')
    if any(v!=expected for v in repeats):return fail('SOURCE_REPEAT_CONFLICT',True)
    allowed={row['source_variant'],row['analysis_variant']}
    if any(str(v).strip() not in allowed for v in (declared_variant,collection_variant) if v is not None and str(v).strip()):
        return fail('SOURCE_VARIANT_CONFLICT',True)
    result.update(status='VERIFIED',provenance='F4_REGISTRY_SHA',analysis_variant=row['analysis_variant'],
        scenario=row['scenario'],repeat=row['rNN_folder'],source_variant=row['source_variant'],
        source_replicate_index=row['source_replicate_index'],label=row['analysis_variant']+' · '+row['rNN_folder'])
    return result


@lru_cache(maxsize=128)
def _source_sha(path,size,mtime):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    current=Path(path).stat()
    if (current.st_size,current.st_mtime_ns)!=(size,mtime):return None
    return h.hexdigest()


def from_bundle(bundle,*,session_id,registry_path=REGISTRY):
    experiment=getattr(bundle,'experiment_session',None) or {};collection=getattr(bundle,'collection_session',None) or {}
    digest=str(getattr(bundle,'source_archive_sha256',None) or '').lower()
    verified=False;rows=[];registry_sha=None;error=None
    try:
        p=Path(bundle.path);s=p.stat()
        verified=bool(bundle.validation.ok and _source_sha(str(p.resolve()),s.st_size,s.st_mtime_ns)==digest)
        data=Path(registry_path).read_bytes();registry_sha=hashlib.sha256(data).hexdigest()
        import io
        rows=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig')),delimiter=';'))
    except (OSError,ValueError,TypeError,AttributeError) as exc:
        error=type(exc).__name__
    result=resolve(archive_sha=digest,session_id=session_id,
        source_session_ids=[experiment.get('experiment_session_id'),experiment.get('session_id'),collection.get('session_id')],
        source_scenarios=[experiment.get('scenario_id'),collection.get('scenario_id')],
        declared_variant=experiment.get('variant'),collection_variant=collection.get('variant'),
        source_repeats=[experiment.get('replicate_index'),collection.get('replicate_index')],
        rows=rows,source_verified=verified)
    result.update(registry_path=str(registry_path),registry_sha256=registry_sha,
        source_experiment_variant_id=experiment.get('variant_id'),declared_android_scenario=experiment.get('scenario_id'))
    if error:result['read_error']=error
    return result


def for_session(session,*,registry_path=REGISTRY):
    error=None
    try:session.verify_source()
    except (OSError,ValueError,AttributeError) as exc:error=type(exc).__name__
    result=from_bundle(session.bundle,session_id=session.session_id,registry_path=registry_path)
    if error:
        result.update(status='UNCONFIRMED',provenance=None,label='NIEPOTWIERDZONE',reason='SOURCE_SHA_UNVERIFIED',read_error=error)
    elif (session.review.session_id!=session.session_id
            or session.review.source_archive_sha256!=session.bundle.source_archive_sha256):
        result.update(status='UNCONFIRMED',provenance=None,label='NIEPOTWIERDZONE',reason='REVIEW_SOURCE_IDENTITY_MISMATCH')
    return result


def text(context):
    if context.get('status')=='VERIFIED':
        first=f"Scenariusz F4 (rejestr SHA): {context['scenario']}\nWariant F4 (rejestr SHA): {context['label']}"
    else:
        first='Wariant F4 (rejestr SHA): '+context.get('label','NIEPOTWIERDZONE')+' · '+str(context.get('reason') or 'brak dowodu')
        first+='\nScenariusz zadeklarowany przez Androida: '+str(context.get('declared_android_scenario') or 'brak danych')
    raw=context.get('declared_android_variant')
    first+='\nWariant zadeklarowany przez Androida: '+(str(raw) if raw is not None and str(raw).strip() else 'brak danych')
    if context.get('provenance'):first+='\nprovenance='+context['provenance']
    return first
