"""Versioned operator evidence; never modifies the scientific review or scorer.

The canonical review revision is a boundary in the shared Undo/Redo timeline.
This addendum has its own monotonic revision and append-only events.
"""
from __future__ import annotations
import copy, csv, getpass, json, os
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 'alpr.mobile.bbox_spatial_review.v1'
LABELS = dict(ON_PLATE='Na obszarze tablicy', OFF_PLATES_BAD='Błędna lokalizacja — poza tablicami',
              UNDECIDABLE='Nie można ocenić', UNREVIEWED='Brak rozstrzygnięcia',
              NOT_ASSESSABLE_SOURCE='Nie można ocenić źródła')

def path(session):
    return session.sidecar_path.with_suffix('.bbox-spatial-v1.json')

def load(session):
    p = path(session)
    value = json.loads(p.read_text(encoding='utf-8')) if p.exists() else dict(
        schema=SCHEMA, source_archive_sha256=session.bundle.source_archive_sha256,
        session_id=session.session_id, canonical_review_id=session.review.review_id,
        diagnostic_revision=0, records={}, histories={}, events=[])
    if (value.get('schema') != SCHEMA or value.get('source_archive_sha256') != session.bundle.source_archive_sha256
            or value.get('session_id') != session.session_id or value.get('canonical_review_id') != session.review.review_id):
        raise ValueError('Dodatek oceny BBOX dotyczy innego źródła lub weryfikacji.')
    return value

def key(gid, aid):
    return json.dumps([gid, aid, 'BBOX'], separators=(',', ':'))

def record(session, gid, candidate):
    item=load(session)['records'].get(key(gid, candidate['attempt_id']))
    if item:
        identity=item['source_identity'];row=session.attempts[candidate['attempt_id']]
        if (identity['invocation_id']!=gid or identity['attempt_id']!=candidate['attempt_id']
            or identity['detection_index']!=candidate['mt_detection_index'] or identity['geometry_kind']!='BBOX'
            or item['source_bbox']!=[float(row[k]) for k in ('plate_left','plate_top','plate_right','plate_bottom')]):
            raise ValueError('Ocena BBOX nie odpowiada źródłowej geometrii.')
    return item

def _canonical(session, gid):
    from ..ranking import mobile_canvas_review as core, mobile_candidate_stacks as stacks
    scene = session.invocation_annotation(gid).get(core.KEY, {})
    return scene.get(stacks.HISTORY, {}), scene.get('events', [])

def history_choice(session, gid, *, redo=False):
    """Merge action order without inserting evidence into canonical history."""
    doc = load(session); hist = doc['histories'].get(gid, {})
    if not hist:return None
    canonical, events = _canonical(session, gid)
    choices = []
    if redo:
        own_undo = next((e for e in reversed(doc['events']) if e['gid'] == gid and e['action'] == 'undo'), None)
        canonical_undo = next((e for e in reversed(events) if e.get('action') == 'undo_stack_history'), None)
        if hist.get('redo') and own_undo:
            abandoned = any(e.get('action') not in ('undo_stack_history','redo_stack_history') and
                            e.get('review_revision_before', -1) >= own_undo['base_revision'] for e in events)
            if not abandoned: choices.append(((own_undo['base_revision'], 1, own_undo['revision']), 'diagnostic'))
        if canonical.get('redo') and canonical_undo:
            rev = canonical_undo['review_revision_before'] + 1
            abandoned = any(e['gid'] == gid and e['action'] == 'set' and e['base_revision'] >= rev for e in doc['events'])
            if not abandoned: choices.append(((rev, 0, 0), 'canonical'))
    else:
        head = hist.get('head')
        if isinstance(head, int):
            entry = hist['entries'][head]
            # A redo is a new application of an old action, not its old timestamp.
            event = next(e for e in reversed(doc['events']) if e['gid'] == gid and e['index'] == head and e['action'] in ('set','redo'))
            choices.append(((event['base_revision'], 1, event['revision']), 'diagnostic'))
        head = canonical.get('head')
        if isinstance(head, int):
            entry = canonical['entries'][head]
            rev = entry['review_revision_before'] + 1
            last = events[-1] if events else {}
            if last.get('action') == 'redo_stack_history': rev = last['review_revision_before'] + 1
            choices.append(((rev, 0, 0), 'canonical'))
    return max(choices)[1] if choices else None

def _write(session, previous, changed):
    from ..ranking import mobile_human_review as review
    p = path(session); p.parent.mkdir(parents=True, exist_ok=True)
    lock = p.with_suffix(p.suffix + '.lock')
    try: fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError: raise ValueError('Trwa zapis oceny BBOX w innym oknie. Spróbuj ponownie.') from None
    try:
        os.close(fd)
        session.verify_source()
        if load(session) != previous: raise ValueError('Ocena BBOX zmieniła się w innym oknie. Odczytaj ponownie.')
        if session.sidecar_path.exists():
            canonical = json.loads(session.sidecar_path.read_text(encoding='utf-8'))
            if (canonical['review_id'], canonical['review_revision']) != (session.review.review_id, session.review.review_revision):
                raise ValueError('Weryfikacja zmieniła się w innym oknie. Odczytaj ponownie.')
        review._atomic_write(p, json.dumps(changed, ensure_ascii=False, indent=2, allow_nan=False))
    finally: lock.unlink(missing_ok=True)

def set_verdict(session, gid, oid, verdict, *, reason='', expected_revision=None):
    from .mobile_canvas_bbox_marker import describe
    from ..ranking import mobile_canvas_review as core, mobile_candidate_stacks as stacks
    if verdict not in LABELS or verdict == 'NOT_ASSESSABLE_SOURCE': raise ValueError('Nieznana ocena położenia BBOX.')
    s = core.scene(session, gid); state = stacks.snapshot(session, gid, s)
    core.require_source_final(session, gid, oid)
    candidate = next(c for c in state['candidates'] if c['object_id'] == oid)
    marker = describe(session, gid, s, candidate)
    if not marker['marker']: raise ValueError('Brak ocenialnego znacznika BBOX: ' + marker['reason'])
    old = load(session)
    if expected_revision is not None and expected_revision != old['diagnostic_revision']:
        raise ValueError('Ocena BBOX zmieniła się od otwarcia menu.')
    doc = copy.deepcopy(old); revision = old['diagnostic_revision'] + 1; k = key(gid, candidate['attempt_id'])
    confirmed = [dict(plate_id=pid, gt_quad=p['gt_quad']) for pid,p in state['plates'].items() if p.get('confirmed') and p.get('gt_quad')]
    after = dict(source_identity=marker['identity'], source_bbox=marker['bbox'], validity=marker['validity'],
        spatial_verdict=verdict, reason=reason.strip() or LABELS[verdict], reason_code='OPERATOR_' + verdict,
        actor=session.review.reviewer_id or getpass.getuser(), diagnostic_revision=revision,
        canonical_review_revision_at_action=session.review.review_revision,
        confirmed_gt_context=confirmed, existing_mt_gt_id=candidate.get('plate_id'))
    hist = doc['histories'].setdefault(gid, dict(entries=[], head=None, redo=[]))
    hist['entries'].append(dict(key=k, before=copy.deepcopy(doc['records'].get(k)), after=after, previous=hist['head']))
    hist['head'] = len(hist['entries'])-1; hist['redo'] = []; doc['records'][k] = after
    _event(doc, session, gid, 'set', hist['head']); _write(session, old, doc)
    return after

def _event(doc, session, gid, action, index):
    doc['diagnostic_revision'] += 1
    doc['events'].append(dict(gid=gid, action=action, index=index, revision=doc['diagnostic_revision'],
        base_revision=session.review.review_revision, actor=session.review.reviewer_id or getpass.getuser(),
        at=datetime.now(timezone.utc).isoformat()))

def restore(session, gid, *, redo=False):
    if history_choice(session,gid,redo=redo) != 'diagnostic': return False
    old=load(session); doc=copy.deepcopy(old); hist=doc['histories'][gid]
    index=hist['redo'].pop() if redo else hist['head']; entry=hist['entries'][index]
    if redo: hist['head']=index
    else: hist['head']=entry['previous']; hist['redo'].append(index)
    target=entry['after' if redo else 'before']
    if target is None: doc['records'].pop(entry['key'],None)
    else: doc['records'][entry['key']]=copy.deepcopy(target)
    _event(doc,session,gid,'redo' if redo else 'undo',index); _write(session,old,doc)
    return True

def diagnostic_summary(session):
    """FINAL population only. RAW and unproven invocations are explicit gaps."""
    from ..ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks
    from .mobile_canvas_bbox_marker import describe
    rows=[];gaps=[];raw=0
    for gid,group in session.mt_invocations.items():
        try:
            scene=core.scene(session,gid);state=stacks.snapshot(session,gid,scene)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            gaps.append(dict(invocation_id=gid,source_positions=len(group.records),reason=str(exc)));continue
        for candidate in state['candidates']:
            if candidate['role']!='FINAL':
                raw+=candidate['role']=='SUPPRESSED';continue
            d=describe(session,gid,scene,candidate)
            saved=record(session,gid,candidate)
            verdict=saved['spatial_verdict'] if saved else 'UNREVIEWED' if d['marker'] else 'NOT_ASSESSABLE_SOURCE' if d['validity']!='VALID' else None
            rows.append(dict(source_identity=d['identity'] or dict(session_id=session.session_id,
                source_archive_sha256=session.bundle.source_archive_sha256,invocation_id=gid,
                attempt_id=candidate['attempt_id'],detection_index=candidate['mt_detection_index'],geometry_kind='BBOX'),
                source_bbox=d['bbox'],validity=d['validity'],spatial_verdict=verdict,reason=(saved or {}).get('reason',d['reason']),
                **{k:v for k,v in (saved or {}).items() if k not in ('source_identity','source_bbox','validity','spatial_verdict','reason')}))
    counts={v:sum(r['spatial_verdict']==v for r in rows) for v in LABELS}
    return dict(population='Source MT FINAL; geometry diagnostics independent of canonical TP/FP/FN',
        invalid_bbox_trusted_position=sum(r['validity'] in ('INVALID_ZERO_HEIGHT','INVALID_ZERO_WIDTH','INVALID_POINT') for r in rows),
        valid_bbox=sum(r['validity']=='VALID' for r in rows),not_spatially_assessable=counts['NOT_ASSESSABLE_SOURCE'],
        counts=counts,raw_suppressed_excluded=raw,unproven_invocations=gaps,rows=rows)

def export_addendum(session, directory):
    """Only a fresh export directory; no overwrite of any historical addendum."""
    doc=load(session)
    target=Path(directory); target.mkdir(parents=True,exist_ok=True)
    summary=diagnostic_summary(session)
    payload=dict(doc, aggregate={k:v for k,v in summary.items() if k!='rows'},diagnostic_rows=summary['rows'],
                 canonical_review_revision_at_export=session.review.review_revision)
    jp=target/'bbox_spatial_review_v1.json'; cp=target/'bbox_spatial_review_v1.csv'
    if jp.exists() or cp.exists(): raise ValueError('Dodatek BBOX już istnieje; wybierz nowy katalog eksportu.')
    # Exclusive creation prevents replacing a prior approved export.
    with jp.open('x',encoding='utf-8') as f: json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False)
    fields=['session_id','source_archive_sha256','invocation_id','attempt_id','detection_index','geometry_kind',
            'validity','spatial_verdict','reason','actor','diagnostic_revision','canonical_review_revision_at_action','source_bbox','confirmed_gt_context']
    with cp.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for row in summary['rows']:
            combined={**row['source_identity'],**row}
            writer.writerow({k:json.dumps(combined[k],ensure_ascii=False) if isinstance(combined.get(k),(dict,list)) else combined.get(k,'') for k in fields})
    return [jp,cp]
