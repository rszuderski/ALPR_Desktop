"""Projection of the existing FINAL journal. PC software intervals, never photons.

Interval = PHOTO_VIEW_UPDATED -> request of a successfully observed replacement.
UTC is descriptive only. An absent replacement leaves the end and duration null.
"""
from pathlib import Path
import csv
import hashlib
import json


def digest(raw):return hashlib.sha256(raw).hexdigest()


def register_run(prepared,run_path,state):
    """Locator only: mutable run/journals must still be verified at import."""
    from ..gt_pack import _atomic_write_json
    prepared=Path(prepared).resolve();run_path=Path(run_path).resolve(strict=True)
    if run_path.parents[2]!=prepared or digest((prepared/'manifest.json').read_bytes())!=state['manifest_sha256']:
        raise ValueError('RUN_LOCATOR_IDENTITY_MISMATCH')
    # Hash the key rather than using an untrusted run ID as a filename.
    key=digest((state['manifest_sha256']+'|'+state['run_id']).encode())
    path=prepared.parent/'run_index'/(key+'.json');path.parent.mkdir(exist_ok=True)
    value=dict(schema='alpr.final.run_locator.v1',run_id=state['run_id'],manifest_sha256=state['manifest_sha256'],run_path=str(run_path))
    if path.exists() and json.loads(path.read_text('utf-8'))!=value:raise ValueError('RUN_LOCATOR_CONFLICT')
    _atomic_write_json(path,value)
    return path


def discover_runs(scopes):
    """Read-only discovery in explicit preparation roots, including legacy runs."""
    paths=set();locators={};issues=[]
    for scope in dict.fromkeys(Path(p).resolve() for p in scopes):
        for path in scope.glob('run_index/*.json'):
            try:
                entry=json.loads(path.read_text('utf-8'));target=Path(entry['run_path']).resolve(strict=True)
                if entry.get('schema')!='alpr.final.run_locator.v1' or target.parents[3]!=scope or target.name!='run.json':
                    raise ValueError('RUN_LOCATOR_SCOPE_MISMATCH')
                paths.add(target);locators.setdefault(target,[]).append(entry)
            except (OSError,ValueError,KeyError,IndexError) as exc:issues.append(dict(path=str(path),reason=str(exc)))
        paths.update(p.resolve() for p in scope.glob('prepared_*/runs/run_*/run.json'))
    candidates=[]
    for path in sorted(paths):
        try:
            prepared=path.parents[2];raw=path.read_bytes();manifest_raw=(prepared/'manifest.json').read_bytes()
            manifest=json.loads(manifest_raw);sha=digest(manifest_raw)
            state,events,_=read_run(path,manifest,sha)
            if path.read_bytes()!=raw:raise ValueError('RUN_CHANGED_DURING_DISCOVERY')
            if any(e['run_id']!=state['run_id'] or e['manifest_sha256']!=sha for e in locators.get(path,[])):
                raise ValueError('RUN_LOCATOR_IDENTITY_MISMATCH')
            if state.get('status') not in ('COMPLETED','STOPPED'):raise ValueError('RUN_NOT_CLOSED')
            terminal='RUN_COMPLETED' if state['status']=='COMPLETED' else 'RUN_STOPPED'
            if not events or events[-1]['event']!=terminal or state.get('last_event')!=events[-1]:
                raise ValueError('RUN_TERMINAL_CHECKPOINT_MISMATCH')
            if terminal=='RUN_COMPLETED' and any(r['status']!='COMPLETED' for r in project(manifest,events)):
                raise ValueError('RUN_COMPLETENESS_MISMATCH')
            context=marker_context(state,events,manifest,sha,prepared=prepared)
            if context:
                from . import final180_markers as markers
                tokens={markers.token_key(v) for v in context['observed_tokens']}
                technical={markers.token_key(v) for v in context['technical_tokens']}
            else:
                from . import final180_markers as markers
                # Historical P had no display_token field. Its full identity is
                # fixed by the manifest/run and observed PHOTO_VIEW_UPDATED.
                tokens={markers.token(sha,state['run_id'],e['exposure_id'],'P') for e in events if e['event']=='PHOTO_VIEW_UPDATED'}
                technical=set()
            candidates.append(dict(path=str(path),prepared=str(prepared),manifest=manifest,state=state,tokens=tokens,technical_tokens=technical,
                proof=dict(run_id=state['run_id'],manifest_sha256=sha,run_path=str(path),run_sha256=digest(raw),
                    journals=state['journals'],locator_used=path in locators,status=state['status'])))
        except (OSError,ValueError,KeyError,IndexError) as exc:issues.append(dict(path=str(path),reason=str(exc)))
    return candidates,issues


def verify_run_proof(proof):
    """Reject source changes between discovery and analysis/export."""
    path=Path(proof['run_path']);manifest_path=path.parents[2]/'manifest.json'
    if digest(path.read_bytes())!=proof['run_sha256'] or digest(manifest_path.read_bytes())!=proof['manifest_sha256']:
        raise ValueError('RUN_CHANGED_AFTER_MATCH')
    for entry in proof['journals']:
        raw=(path.parent/entry['file']).read_bytes()
        if len(raw)!=entry['bytes'] or digest(raw)!=entry['sha256']:raise ValueError('RUN_JOURNAL_CHANGED_AFTER_MATCH')


def check_run_tag(prepared,state):
    """Explicit experiment-root scope; collisions fail closed, no inferred run."""
    from . import final180_markers as markers
    if state.get('qr_binding',{}).get('qr_version')==markers.F3_QR:return check_f3_registry(prepared,state)
    prepared=Path(prepared).resolve(strict=True)
    if not (prepared/'manifest.json').is_file() or digest((prepared/'manifest.json').read_bytes())!=state['manifest_sha256']:
        raise ValueError('F2_RUN_REGISTRY_SCOPE_UNAVAILABLE')
    scope=prepared.parent
    tag=state['qr_binding']['run_tag'];identity=(state['manifest_sha256'],state['run_id']);matches=[]
    for path in scope.glob('prepared_*/runs/run_*/run.json'):
        other=json.loads(path.read_text('utf-8'))
        binding=other.get('qr_binding')
        if not binding:continue
        if binding.get('qr_version')==markers.F3_QR:continue
        expected=markers._short('F2_RUN|'+other['manifest_sha256']+'|'+other['run_id'],10)
        if binding.get('run_tag')!=expected:raise ValueError('F2_RUN_REGISTRY_CORRUPT')
        manifest_path=path.parents[2]/'manifest.json'
        if digest(manifest_path.read_bytes())!=other['manifest_sha256']:raise ValueError('F2_REGISTRY_MANIFEST_SHA_MISMATCH')
        if binding['run_tag']==tag and (other['manifest_sha256'],other['run_id'])!=identity:raise ValueError('F2_RUN_TAG_COLLISION')
        if binding['run_tag']==tag:matches.append(path)
    if len(matches)>1:raise ValueError('F2_DUPLICATE_RUN_INDEX')
    for path in scope.glob('calibration_previews/preview_*.json'):
        preview=json.loads(path.read_text('utf-8'));binding=preview.get('test_binding',{})
        if binding.get('qr_version')==markers.F3_QR:continue
        expected=markers._short('F2_RUN|'+preview['manifest_sha256']+'|'+preview['technical_id'],10)
        if binding.get('run_tag')!=expected:raise ValueError('F2_CALIBRATION_REGISTRY_CORRUPT')
        if binding['run_tag']==tag and (preview['manifest_sha256'],preview['technical_id'])!=identity:raise ValueError('F2_RUN_TAG_COLLISION')
    return str(scope)


def f3_registry_scopes(prepared):
    """Same local roots available to the existing one-file Desktop importer."""
    from alpr_runtime import workspace_root
    roots=[Path(prepared).resolve().parent,workspace_root()/'experiments/F6-FINAL180-A12-01']
    return sorted(set(p.resolve() for p in roots if p.is_dir()))


def check_f3_registry(prepared,state):
    """Exhaustive available-run/preview index; reject even an individual tag reuse."""
    from . import final180_markers as markers
    prepared=Path(prepared).resolve(strict=True);sha=state['manifest_sha256']
    raw=(prepared/'manifest.json').read_bytes()
    if digest(raw)!=sha:raise ValueError('F3_REGISTRY_MANIFEST_SHA_MISMATCH')
    m=json.loads(raw);own=state['qr_binding'];test_only=own.get('purpose')=='TEST_ONLY'
    if own!=markers.f3_binding(m,sha,state['run_id'],test_only=test_only):raise ValueError('F3_RUN_INDEX_MISMATCH')
    scopes=f3_registry_scopes(prepared);registered=[]
    for scope in scopes:
        paths=set(scope.glob('prepared_*/runs/run_*/run.json'))
        # Include the locator index, but verify its complete identity and scope.
        for path in scope.glob('run_index/*.json'):
            entry=json.loads(path.read_text('utf-8'));target=Path(entry['run_path']).resolve()
            if not target.is_file():raise ValueError('F3_RUN_LOCATOR_UNAVAILABLE')
            other=json.loads(target.read_text('utf-8'))
            if other.get('qr_binding',{}).get('qr_version')!=markers.F3_QR:continue
            if (entry.get('schema')!='alpr.final.run_locator.v1' or target.parents[3]!=scope
                    or entry.get('run_id')!=other.get('run_id') or entry.get('manifest_sha256')!=other.get('manifest_sha256')):
                raise ValueError('F3_RUN_LOCATOR_IDENTITY_MISMATCH')
            paths.add(target)
        for path in sorted(paths):
            other=json.loads(path.read_text('utf-8'));b=other.get('qr_binding',{})
            if b.get('qr_version')!=markers.F3_QR:continue
            raw=(path.parents[2]/'manifest.json').read_bytes()
            if digest(raw)!=other['manifest_sha256']:raise ValueError('F3_REGISTRY_MANIFEST_SHA_MISMATCH')
            expected=markers.f3_binding(json.loads(raw),other['manifest_sha256'],other['run_id'])
            if b!=expected:raise ValueError('F3_RUN_REGISTRY_CORRUPT')
            registered.append((str(path.resolve()),b))
        for path in scope.glob('calibration_previews/preview_*.json'):
            preview=json.loads(path.read_text('utf-8'));b=preview.get('test_binding',{})
            if b.get('qr_version')!=markers.F3_QR:continue
            raw=(Path(preview['prepared_path'])/'manifest.json').read_bytes()
            if digest(raw)!=preview['manifest_sha256']:raise ValueError('F3_CALIBRATION_MANIFEST_SHA_MISMATCH')
            expected=markers.f3_binding(json.loads(raw),preview['manifest_sha256'],preview['technical_id'],test_only=True)
            if b!=expected or preview.get('research_run') is not False or preview.get('token')!=b['entries'][0]['token']:
                raise ValueError('F3_CALIBRATION_REGISTRY_CORRUPT')
            registered.append((str(path.resolve()),b))
    tags={};pairs={};identities={}
    for path,b in registered:
        identity=(b['manifest_sha256'],b['run_id'],b['purpose'])
        if identity in identities:raise ValueError('F3_DUPLICATE_RUN_INDEX')
        identities[identity]=path
        if b['run_tag'] in tags:raise ValueError('F3_RUN_TAG_COLLISION')
        tags[b['run_tag']]=identity
        for entry in b['entries']:
            if entry['pair_tag'] in pairs:raise ValueError('F3_PAIR_TAG_COLLISION')
            pairs[entry['pair_tag']]=identity
    identity=(own['manifest_sha256'],own['run_id'],own['purpose'])
    if own['run_tag'] in tags and tags[own['run_tag']]!=identity:raise ValueError('F3_RUN_TAG_COLLISION')
    if any(e['pair_tag'] in pairs and pairs[e['pair_tag']]!=identity for e in own['entries']):raise ValueError('F3_PAIR_TAG_COLLISION')
    return [str(p) for p in scopes]


def register_calibration(prepared,identity):
    """Technical token provenance outside the frozen plan; no research run/event."""
    from . import final180_markers as markers
    from ..gt_pack import _atomic_write_json
    prepared=Path(prepared);manifest=json.loads((prepared/'manifest.json').read_text('utf-8'));sha=digest((prepared/'manifest.json').read_bytes())
    if manifest['layout'].get('layout_version') not in markers.BOUND_LAYOUTS:return None
    expected=markers.binding(manifest,sha,identity['technical_id'],test_only=True)
    if identity.get('test_binding')!=expected:raise ValueError('F2_CALIBRATION_IDENTITY_MISMATCH')
    check_run_tag(prepared,dict(manifest_sha256=sha,run_id=identity['technical_id'],qr_binding=expected))
    path=prepared.parent/'calibration_previews'/('preview_'+identity['technical_id']+'.json')
    if path.exists():raise ValueError('F2_CALIBRATION_IDENTITY_ALREADY_EXISTS')
    path.parent.mkdir(exist_ok=True)
    extra=dict(prepared_path=str(prepared.resolve())) if expected['qr_version']==markers.F3_QR else {}
    _atomic_write_json(path,dict(identity,schema='alpr.final.calibration_identity.v1',purpose='TEST_ONLY',research_run=False,**extra))
    return path


def marker_context(state,events,manifest,manifest_sha,*,prepared=None):
    from . import final180_markers as markers
    if manifest['layout'].get('layout_version') not in markers.BOUND_LAYOUTS:return None
    f3=manifest['layout'].get('layout_version')==markers.F3_LAYOUT
    expected=markers.binding(manifest,manifest_sha,state['run_id'])
    if state.get('qr_binding')!=expected:raise ValueError('F3_RUN_INDEX_MISMATCH' if f3 else 'F2_RUN_INDEX_MISMATCH')
    scope_prepared=Path(prepared) if prepared is not None else Path(state['prepared_path'])
    check_run_tag(scope_prepared,state)
    technical=[]
    for path in scope_prepared.resolve().parent.glob('calibration_previews/preview_*.json'):
        preview=json.loads(path.read_text('utf-8'))
        if preview.get('manifest_sha256')!=manifest_sha:continue
        binding=markers.binding(manifest,manifest_sha,preview['technical_id'],test_only=True)
        if preview.get('test_binding')!=binding or preview.get('research_run') is not False:raise ValueError('F2_CALIBRATION_REGISTRY_CORRUPT')
        technical.extend(e['token'] for e in binding['entries'])
    allowed={markers.token_key(e['token']):e for e in expected['entries']};observed=[]
    for event in events:
        value=event.get('display_token')
        if value is None:
            if f3 and (event['event']=='PHOTO_VIEW_UPDATED' or (event['event']=='REPLACEMENT_VIEW_UPDATED' and event.get('phase') in ('T','COUNTDOWN'))):
                raise ValueError('F3_OBSERVED_EVENT_PAIR_MISSING')
            continue
        key=markers.token_key(value)
        if key not in allowed:raise ValueError('F3_EVENT_PAIR_MISMATCH' if f3 else 'F2_EVENT_TOKEN_MISMATCH')
        entry=allowed[key]
        if event['event']=='PHOTO_VIEW_UPDATED' and (entry['phase']!='P' or entry['exposure_id']!=event.get('exposure_id')):
            raise ValueError('F2_EVENT_PHOTO_TOKEN_MISMATCH')
        if event['event']=='REPLACEMENT_VIEW_UPDATED' and entry['phase']!='T':raise ValueError('F2_EVENT_TRANSITION_TOKEN_MISMATCH')
        if event['event'] in ('PHOTO_VIEW_UPDATED','REPLACEMENT_VIEW_UPDATED'):observed.append(key)
    return dict(binding=expected,observed_tokens=sorted(set(observed)),technical_tokens=technical,registry_scope=str(scope_prepared.resolve().parent))


def read_run(path, manifest, manifest_sha, *, recover=False):
    path=Path(path);state=json.loads(path.read_text('utf-8'))
    if state.get('schema')!='alpr.final.presentation_run.v1' or state.get('manifest_sha256')!=manifest_sha:
        raise ValueError('RUN_MANIFEST_MISMATCH')
    journals=state.get('journals') or [dict(file='events.jsonl',sha256=state.get('journal_sha256'))]
    events=[];recovery=[];updated=[];sequence=0;clocks={};shown=set()
    for entry in journals:
        name=entry['file']
        if Path(name).name!=name:raise ValueError('UNSAFE_JOURNAL_PATH')
        raw=(path.parent/name).read_bytes();checkpoint=entry.get('bytes',len(raw))
        if type(checkpoint) is not int or not 0<=checkpoint<=len(raw) or digest(raw[:checkpoint])!=entry['sha256']:
            raise ValueError('RUN_JOURNAL_SHA_MISMATCH')
        if len(raw)!=checkpoint and not recover:raise ValueError('RUN_JOURNAL_UNCHECKPOINTED_TAIL')
        valid=entry.get('valid_bytes',len(raw));offset=0;tail=None
        for line in raw[:valid].splitlines(keepends=True):
            try:
                row=json.loads(line)
                if not isinstance(row,dict):raise ValueError('Expected object')
            except (ValueError,UnicodeError):
                if not recover or raw[offset+len(line):].strip():raise ValueError('RUN_JOURNAL_CORRUPTION')
                tail=offset;break
            if not isinstance(row.get('event'),str) or type(row.get('segment')) is not int:
                raise ValueError('EVENT_STRUCTURE_INVALID')
            if row.get('schema')=='alpr.final.presentation_event.v1':
                if row.get('run_id')!=state['run_id'] or row.get('manifest_sha256')!=manifest_sha:
                    raise ValueError('EVENT_RUN_IDENTITY_MISMATCH')
                if type(row.get('event_seq')) is not int or row['event_seq']<=sequence:raise ValueError('EVENT_SEQUENCE_CONFLICT')
                sequence=row['event_seq'];cid=row['clock_id'];ns=row['pc_perf_counter_ns']
                if type(ns) is not int or row.get('pc_unit')!='ns' or row.get('clock_domain')!='PC_PERF_COUNTER':raise ValueError('EVENT_CLOCK_INVALID')
                if ns<clocks.get(cid,ns):raise ValueError('EVENT_CLOCK_REVERSED')
                clocks[cid]=ns
                if row.get('exposure_id'):
                    p=row.get('position')
                    if type(p) is not int or not 1<=p<=len(manifest['exposures']):raise ValueError('EVENT_POSITION_INVALID')
                    expected=manifest['exposures'][p-1]
                    if row['exposure_id']!=expected['exposure_id'] or row.get('photo_sha256')!=expected['photo_sha256']:
                        raise ValueError('EVENT_PHOTO_IDENTITY_MISMATCH')
            if row.get('event')=='PHOTO_REQUESTED':
                eid=row.get('exposure_id')
                if eid in shown:raise ValueError('DUPLICATE_EXPOSURE_REQUEST_IN_RUN')
                shown.add(eid)
            events.append(row);offset+=len(line)
        valid=tail if tail is not None else valid
        if tail is not None or len(raw)!=checkpoint:
            recovery.append(dict(file=name,checkpoint_bytes=checkpoint,valid_bytes=valid,
                retained_bytes=len(raw),original_sha256=digest(raw),tail_sha256=digest(raw[valid:]),
                reason='MALFORMED_TAIL_RETAINED' if tail is not None else 'UNCHECKPOINTED_VALID_RECORDS'))
        updated.append(dict(file=name,bytes=len(raw),sha256=digest(raw),valid_bytes=valid))
    state['journals']=updated
    marker_context(state,events,manifest,manifest_sha,prepared=path.parents[2])
    return state,events,recovery


def project(manifest, events):
    result=[];dwell=float(manifest['dwell_seconds']);transition=float(manifest['transition_seconds'])
    for e in manifest['exposures']:
        eid=e['exposure_id'];records=[r for r in events if r.get('exposure_id')==eid]
        intent=next((r for r in records if r['event']=='PHOTO_DRAW_INTENT'),None)
        request=next((r for r in records if r['event']=='PHOTO_REQUESTED'),None)
        start=next((r for r in records if r['event']=='PHOTO_VIEW_UPDATED'),None)
        end=next((r for r in events if r['event']=='REPLACEMENT_VIEW_UPDATED' and r.get('replaced_exposure_id')==eid),None)
        row=dict(position=e['position'],exposure_id=eid,file=e.get('source_relative_path',e.get('source_name',e.get('image_id'))),
            photo_sha256=e['photo_sha256'],phase='P',status='NOT_SHOWN',segment=None,segment_id=None,clock_id=None,
            clock_domain='PC_PERF_COUNTER',unit='ns',request_pc_ns=None,start_pc_ns=None,end_pc_ns=None,duration_pc_ns=None,
            planned_start_s=float(manifest['countdown_seconds'])+(e['position']-1)*(dwell+transition),planned_duration_s=dwell,
            start_segment_s=None,end_segment_s=None,duration_pc_s=None,
            schedule_delay_ns=None,draw_duration_ns=None,prepare_duration_ns=None,read_duration_ns=None,
            event_sequences=[r['event_seq'] for r in records if 'event_seq' in r],reason='NOT_REACHED',
            interval_basis='VIEW_UPDATE_COMPLETED_TO_OBSERVED_REPLACEMENT_REQUEST_PC')
        anchor=start or request or intent
        if anchor:
            row.update(segment=anchor.get('segment'),segment_id=anchor.get('segment_id'),clock_id=anchor.get('clock_id'),
                status='UNKNOWN',reason='VIEW_UPDATE_OR_END_NOT_OBSERVED')
        if request:
            row.update(request_pc_ns=request['pc_perf_counter_ns'],schedule_delay_ns=request.get('schedule_delay_ns'),
                prepare_duration_ns=request.get('prepare_duration_ns'),read_duration_ns=request.get('read_duration_ns'))
        if start:
            row.update(start_pc_ns=start['pc_perf_counter_ns'],draw_duration_ns=start.get('draw_duration_ns'),
                start_segment_s=start['segment_elapsed_ns']/1e9)
        if start and end and start.get('clock_id') and start['clock_id']==end.get('clock_id'):
            end_ns=end['replacement_requested_ns']
            if end_ns<start['pc_perf_counter_ns']:raise ValueError('NEGATIVE_PC_INTERVAL')
            elapsed=end_ns-start['pc_perf_counter_ns']
            row.update(end_pc_ns=end_ns,duration_pc_ns=elapsed,duration_pc_s=elapsed/1e9,
                end_segment_s=row['start_segment_s']+elapsed/1e9,
                status='PARTIAL' if end.get('reason')!='ADVANCE' or elapsed<round(dwell*1e9) else 'COMPLETED',reason=end.get('reason',''))
            row['event_sequences']=list(dict.fromkeys([*row['event_sequences'],end.get('event_seq')]))
        result.append(row)
    return result


def export_csv(path, rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['exposure_id','status'],delimiter=';')
        writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
