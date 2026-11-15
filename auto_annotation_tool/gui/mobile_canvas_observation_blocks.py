"""Source-bound navigation only. Never aggregates metrics or writes GT/reviews.

An observation interval is not a proven stable scene. Its SCENE_FRAME and each
MT bitmap keep their original, independent Canvas identities and source_binding.
"""
from __future__ import annotations

from tkinter import ttk, font

from ..ranking.mobile_scene_evidence import STAMP_FIELDS
from ..ranking import mobile_scene_frame_review as scene_review


def _integer(value):
    if isinstance(value, bool):
        raise ValueError('Boolean timestamp')
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise ValueError('Missing or ambiguous timestamp')


def _stamp(row):
    values = tuple(str(row[k]) if k == 'source_timestamp_domain' else _integer(row[k]) for k in STAMP_FIELDS)
    if (not isinstance(row['source_timestamp_domain'], str)
            or row['source_timestamp_domain'] in ('', 'UNKNOWN') or _integer(row['source_sequence']) == 0):
        raise ValueError('Unknown source identity')
    return values


def _true(value):
    return value is True or str(value).lower() == 'true'


def build_blocks(session):
    """Fail closed for ambiguous boundaries; quarantine individual broken links.

    The validated producer registry, not filenames/row numbers or result time,
    determines eligibility and membership. CSV integer strings are accepted.
    """
    disabled = lambda reason: dict(enabled=False, reason=reason, blocks=(), unassigned={})
    bundle = getattr(session, 'bundle', None)
    if not bundle or not getattr(getattr(bundle, 'validation', None), 'ok', False):
        return disabled('Brak zweryfikowanego źródła')
    metadata = getattr(bundle, 'collection_session', None) or {}
    report = getattr(bundle, 'report_payload', None) or {}
    if not isinstance(metadata, dict) or not isinstance(report, dict):
        return disabled('Nieczytelne metadane sesji')
    execution = metadata.get('execution') or {}
    experiment = report.get('experiment') or {}
    if not isinstance(execution, dict) or not isinstance(experiment, dict):
        return disabled('Nieczytelny tryb sesji')
    effective = experiment.get('effective_execution_config') or {}
    if not isinstance(effective, dict):
        return disabled('Nieczytelny tryb sesji')
    modes = {str(v).lower() for v in (execution.get('analysis_mode'), report.get('analysis_mode'),
             effective.get('analysis_mode')) if v}
    if modes != {'static'}:
        return disabled('Tryb dynamiczny, nieznany lub sprzeczny')
    evidence = getattr(bundle, 'scene_evidence', {}) or {}
    rows = evidence.get('scenes', ())
    if not isinstance(rows, (list, tuple)) or any(not isinstance(r, dict) for r in rows):
        return disabled('Nieczytelny rejestr scen')
    if (not evidence.get('available') or not rows or metadata.get('scene_evidence_complete') is not True
            or metadata.get('scene_evidence_pending_loss_unknown') is not False):
        return disabled('Niepełny rejestr granic sceny')
    try:
        if _integer(metadata['scene_evidence_expected']) != len(rows):
            raise ValueError('Missing boundary')
        start, end = _integer(metadata['started_elapsed_nanos']), _integer(metadata['finished_elapsed_nanos'])
        ordered = sorted(rows, key=lambda r: _integer(r['event_elapsed_nanos']))
        ids = [r['scene_event_id'] for r in ordered]
        times = [_integer(r['event_elapsed_nanos']) for r in ordered]
        if (len(set(ids)) != len(ids) or times[0] != start or end <= times[-1]
                or any(a >= b for a, b in zip(times, times[1:]))):
            raise ValueError('Ambiguous boundary interval')
        if ordered[0]['event_type'] != 'SESSION_START' or any(r['event_type'] != 'SCENE_BOUNDARY' for r in ordered[1:]):
            raise ValueError('Ambiguous start')
        generations = set()
        for index, row in enumerate(ordered, 1):
            sid = row['scene_event_id']
            if (row['session_id'] != session.session_id or sid != f'{session.session_id}-s{index:08d}'
                    or session.scene_frames.get(sid) != row):
                raise ValueError('Missing, reordered or foreign scene identity')
            _stamp(row['event_stamp'])
            generation=tuple(row['event_stamp'][k] for k in ('scene_generation','visual_epoch','camera_transform_generation'))
            if generation in generations:
                raise ValueError('Repeated scene generation')
            generations.add(generation)
        if set(ids) != set(session.scene_frames):
            raise ValueError('Scene registry mismatch')
    except (KeyError, TypeError, ValueError):
        return disabled('Nieustalone granice lub koniec obserwacji')

    blocks = [dict(scene_id=r['scene_event_id'], number=i+1, start_ns=times[i],
                   end_ns=times[i+1] if i+1<len(times) else end, session_start_ns=start,
                   stability=None, calls=[]) for i, r in enumerate(ordered)]
    by_scene = {b['scene_id']: b for b in blocks}
    owners = {}
    for row in ordered:
        for gid in row.get('mt_invocation_ids', ()):
            owners.setdefault(gid, []).append(row['scene_event_id'])
    unassigned = {}
    roles = getattr(session, 'mt_detection_roles', {})
    for gid, group in session.mt_invocations.items():
        try:
            role = roles[gid]
            sid = role['scene_event_id']
            boundary = session.scene_frames[sid]
            stamp = role['source_stamp']; event = boundary['event_stamp']
            if owners.get(gid) != [sid] or role['mt_invocation_id'] != gid:
                raise ValueError('Brak jednoznacznego powiązania ze zdarzeniem')
            if any(stamp[k] != event[k] for k in ('scene_generation','visual_epoch','camera_transform_generation','source_timestamp_domain')):
                raise ValueError('Niezgodna generacja źródła')
            if _integer(stamp['source_sequence']) < _integer(event['source_sequence']):
                raise ValueError('Wejście poprzedza zdarzenie źródłowe')
            identity = _stamp(stamp)
            records = list(group.records)
            if not records:
                raise ValueError('Brak rekordów wywołania')
            aids = set()
            for row in records:
                aid = str(row.get('id') or row.get('attempt_id') or '')
                source = session.attempts[aid]
                if (not aid or _stamp(source) != identity or _stamp(row) != identity
                        or source.get('f5a_scene_event_id') not in (None, '', sid)
                        or source.get('mt_invocation_id') != gid):
                    raise ValueError('Niezgodny source_stamp próby MT')
                if source.get('mt_detection_index') not in (None, ''):
                    aids.add(aid)
            if aids != set(role['raw_attempt_ids']):
                raise ValueError('Niepełne powiązanie rekordów MT')
            by_scene[sid]['calls'].append(gid)
        except (KeyError, TypeError, ValueError) as error:
            unassigned[gid] = str(error) if isinstance(error, ValueError) else 'Brak źródłowego powiązania MT'
    return dict(enabled=True, reason='', blocks=blocks, unassigned=unassigned)


def _call_state(session, gid, inventory):
    group = session.mt_invocations[gid]
    role = session.mt_detection_roles.get(gid, {})
    item = inventory['groups'].get(gid, {})
    task = item.get('review_tasks', {})
    if group.cancelled or role.get('stale_or_cancelled'):
        return dict(label='Anulowano · N/A', incomplete=True, pending=0, saved=0, complete=False)
    if (group.execution_failed or not group.executed or role.get('role_proof_status') != 'COMPLETE'
            or role.get('postprocess_status') != 'COMPLETED' or item.get('status') == 'SOURCE_NOT_EVALUABLE'):
        return dict(label='Niepełne · N/A', incomplete=True, pending=0, saved=0, complete=False)
    saved = item.get('frame_review', {}).get('resolved', 0) if item.get('persisted') else 0
    frame = item.get('frame_review', {})
    pending = len(task.get('units', ())) + len(task.get('steps', ())) if task.get('needs_review') else 0
    complete = bool(item.get('persisted') and not task.get('needs_review')
                    and frame.get('complete')
                    and (frame.get('state') != 'NO_REFERENCE_EVIDENCE' or frame.get('empty_confirmed')))
    # Technical N/A without a persisted operator action is not reviewed work.
    if not complete and not pending:
        pending = 1
    return dict(label='Oceniono' if complete else 'Do oceny', incomplete=False,
                pending=pending, saved=saved, complete=complete)


def render(ed, inventory):
    """Update the existing Treeview in place, preserving open branches/selection."""
    model = build_blocks(ed.h.session)
    was_enabled = getattr(ed, 'observation_model', None)
    ed.observation_model = model if model['enabled'] else None
    tree = ed.h.subject_tree
    if not model['enabled']:
        if was_enabled:
            for iid in tree.get_children():
                if iid.startswith('obs:'):
                    tree.delete(iid)
            tree.configure(style='MobileReview.Observations.Treeview')
            tree.heading('#0', text='Obraz  ·  Co zrobić')
        ed.observation_targets = {}
        return False
    session = ed.h.session
    style = f'Blocks{id(ed)}.MobileReview.Observations.Treeview'
    row_font = font.Font(root=tree, font=ttk.Style(tree).lookup('MobileReview.Observations.Treeview', 'font'))
    ttk.Style(tree).configure(style, rowheight=max(48, 2*row_font.metrics('linespace')+10))
    tree.configure(style=style)
    tree.heading('#0', text='Bloki obserwacji sceny')
    desired = set(); targets = {}; gids = {}; roots = {}; states = {}
    previous = tree.selection(); focus = tree.focus(); yview = tree.yview()
    ed.tree_refreshing = True
    def node(iid, parent, text, target=None, *, opened=False, tag='canvas_pending'):
        desired.add(iid)
        if tree.exists(iid):
            tree.item(iid, text=text, values=('',)*6, tags=(tag,))
            tree.move(iid, parent, 'end')
        else:
            tree.insert(parent, 'end', iid=iid, text=text, values=('',)*6, tags=(tag,), open=opened)
        targets[iid] = target
    def call_node(gid, parent, index, state):
        iid = 'obs:mt:' + gid; gids[gid] = iid
        role = session.mt_detection_roles.get(gid, {})
        final = str(len(role.get('final_attempt_ids', ()))) if role.get('role_proof_status') == 'COMPLETE' else 'N/A'
        detail = (f"RAW: {len(role['raw_attempt_ids']) if 'raw_attempt_ids' in role else 'N/A'} · FINAL: {final}"
                  if state['incomplete'] else f"FINAL: {final} · Zapisane: {state['saved']}")
        text = f"MT {index:02d} — {state['label']}\n{detail}"
        node(iid, parent, text, ('mt', gid, ''), tag='canvas_excluded' if state['incomplete'] else
             'canvas_ready' if state['complete'] else 'canvas_pending')
    try:
        assigned_samples = set()
        samples = {}
        for sample_id, row in session.samples.items():
            samples.setdefault(row.get('attempt_id'), []).append(sample_id)
        for block in model['blocks']:
            sid = block['scene_id']; root = 'obs:scene:' + sid; roots[root] = sid
            try:
                scene = scene_review.scene(session, sid, include_history=False)
                checked = bool(scene['checked'])
                scene_label = 'Brak tablic potwierdzony' if checked and scene['presence']=='NO_PLATES' else 'Oceniono' if checked else 'Do oceny'
            except (KeyError, TypeError, ValueError):
                checked = False; scene_label = 'Niepełny dowód · N/A'
            states.update({gid:_call_state(session,gid,inventory) for gid in block['calls']})
            pending = int(not checked) + sum(states[gid]['pending'] for gid in block['calls'])
            missing_links = set(session.scene_frames[sid].get('mt_invocation_ids', ())) - set(block['calls'])
            complete = checked and not missing_links and all(states[g]['complete'] for g in block['calls'])
            label = 'Oceniono' if complete else 'MT do oceny' if checked and pending else 'Niepełne · N/A' if checked else 'Do oceny'
            block.update(pending=pending, complete=complete)
            node(root, '', f"Scena {block['number']:02d} — {label}\nMT: {len(block['calls'])} · Zadań: {pending}",
                 ('scene',sid,''), tag='canvas_ready' if complete else 'canvas_pending')
            node('obs:frame:'+sid, root, 'Pełna klatka — ocena sceny\n'+scene_label, ('scene',sid,''),
                 tag='canvas_ready' if checked else 'canvas_pending')
            normal=[g for g in block['calls'] if not states[g]['incomplete']]
            incomplete=[g for g in block['calls'] if states[g]['incomplete']]
            mt_parent='obs:mt-folder:'+sid
            node(mt_parent,root,f'Wywołania MT/ROI ({len(normal)})',opened=True)
            for index,gid in enumerate(normal,1):call_node(gid,mt_parent,index,states[gid])
            mz_rows=[]
            for gid in block['calls']:
                for row in session.mt_invocations[gid].records:
                    aid=str(row.get('id') or row.get('attempt_id'))
                    if _true(session.attempts[aid].get('mz_executed')) or aid in samples:
                        mz_rows.append((gid,aid));assigned_samples.update(samples.get(aid,()))
            mz_parent='obs:mz-folder:'+sid
            node(mz_parent,root,f'Powiązane wyniki MZ ({len(mz_rows)})')
            for index,(gid,aid) in enumerate(mz_rows,1):
                row=session.attempts[aid]
                visible=session.predictions_visible(row.get('subject_key',''))
                executed=_true(row.get('mz_executed'))
                label=('N/A — nie wykonano' if not executed else 'Wynik ukryty' if not visible else
                       'Zapisany wynik' if row.get('prediction') else 'Brak odczytu')
                node('obs:mz:'+aid,mz_parent,f'MZ {index:02d} — {label}',('mz',gid,aid))
            if incomplete:
                parent='obs:incomplete:'+sid
                node(parent,root,f'Niepełne / anulowane ({len(incomplete)})',opened=True,tag='canvas_excluded')
                for index,gid in enumerate(incomplete,1):call_node(gid,parent,index,states[gid])
        orphan_samples=set(session.samples)-assigned_samples
        represented={str(r.get('id') or r.get('attempt_id')) for g in session.mt_invocations.values() for r in g.records}
        orphan_attempts=set(session.attempts)-represented
        orphan_calls=set(gid for row in session.scene_frames.values() for gid in row.get('mt_invocation_ids',()))-set(session.mt_invocations)
        orphan_subjects=[key for key,s in session.subjects.items() if not s.get('attempts') and not s.get('samples')]
        if model['unassigned'] or orphan_samples or orphan_subjects or orphan_attempts or orphan_calls:
            diagnostic='obs:unassigned'
            node(diagnostic,'','Dowody nieprzypisane\n/ niepełne',tag='canvas_excluded')
            for index,(gid,reason) in enumerate(model['unassigned'].items(),1):
                call_node(gid,diagnostic,index,dict(label='Powiązanie N/A',incomplete=True,saved=0,complete=False))
            for sample_id in sorted(orphan_samples):
                node('obs:orphan-sample:'+sample_id,diagnostic,'Wynik MZ — powiązanie N/A',('sample',sample_id,''),tag='canvas_excluded')
            for aid in sorted(orphan_attempts):
                node('obs:orphan-attempt:'+aid,diagnostic,'Próba MT — powiązanie N/A',('attempt',aid,''),tag='canvas_excluded')
            for gid in sorted(orphan_calls):
                node('obs:orphan-call:'+gid,diagnostic,'Odwołanie MT bez rekordu · N/A',tag='canvas_excluded')
            for key in orphan_subjects:
                node('obs:orphan-subject:'+key,diagnostic,'Zapis źródłowy — powiązanie N/A',('subject',key,''),tag='canvas_excluded')
        def prune(parent=''):
            for iid in tuple(tree.get_children(parent)):
                if iid not in desired:tree.delete(iid)
                else:prune(iid)
        prune()
        ed.observation_targets=targets;ed.observation_mt_ids=gids;ed.observation_states=states
        ed.scene_tree_ids=roots;ed.scene_mt_tree_ids={};ed.linked_tree_rows={};ed.tree_targets={}
        if previous and tree.exists(previous[0]):tree.selection_set(previous)
        if focus and tree.exists(focus):tree.focus(focus)
        if yview:tree.yview_moveto(yview[0])
    finally:
        ed.tree_refreshing=False
    ed.h._schedule_status_pills();ed.update_header(inventory);sync_selection(ed)
    return True


def _active(ed):
    return bool(getattr(ed,'observation_model',None))


def select(ed,iid):
    if not _active(ed):return False
    target=ed.observation_targets.get(iid)
    if ed.tree_refreshing or target is None:return True
    h=ed.h
    if h._saving:
        ed._pending_observation_target=(id(h.session),iid)
        return True
    kind,key,aid=target
    if not ed.prepare_navigation():
        sync_selection(ed);return True
    ed._observation_selected_iid=iid
    if kind=='scene':
        ed.scene_mt_focus=None
        from .mobile_canvas_scene_frames import open_scene
        h.subject_tree.item('obs:scene:'+key,open=True)
        open_scene(ed,key)
    elif kind in ('mt','mz'):
        group=h.session.mt_invocations.get(key)
        if not group or not group.records:return True
        aid=aid or str(group.records[0].get('id') or group.records[0].get('attempt_id'))
        if h.current_invocation==key and (kind=='mt' or h.current_record==('attempt',aid)):
            sync_selection(ed);return True
        if h.current_invocation==key:
            h._source_frame_keep_on_selection=h._image_entry
        ed.scene_mt_focus=None
        h._multi_focus_attempt(aid)
        h._select_record()
        if kind=='mz':
            # Select the exact existing detection, never synthesize a GT object
            # from an OCR result or a suppressed/unresolved source record.
            from .mobile_canvas_navigator import finals, select_result
            index=next((i for i,row in enumerate(finals(ed)) if row['attempt_id']==aid),None)
            if index is not None:select_result(ed,index)
    elif kind in ('sample','subject','attempt'):
        records=h.session.samples if kind=='sample' else h.session.attempts
        row=records.get(key,{}) if kind!='subject' else {}
        subject=row.get('subject_key') if kind!='subject' else key
        if subject in h.session.subjects:
            h.subject_key=subject;h._render_subject()
            if kind!='subject':
                record=next((i for i,r in h._record_ids.items() if r==(kind,key)),None)
                if record:h.record_tree.selection_set(record)
            h._select_record()
    # The host synchronizes intermediate subject/record changes. Restore the
    # requested row only after its real source and concrete detection are ready.
    ed._observation_selected_iid=iid
    sync_selection(ed)
    return True


def resume(ed):
    pending=getattr(ed,'_pending_observation_target',None)
    ed._pending_observation_target=None
    if pending and pending[0]==id(ed.h.session) and ed.h.subject_tree.exists(pending[1]):
        select(ed,pending[1])


def sync_selection(ed):
    if not _active(ed):return False
    if ed.tree_refreshing or getattr(ed,'_pending_observation_target',None):return True
    h=ed.h;tree=h.subject_tree;gid=h.current_invocation
    current=getattr(ed,'_observation_selected_iid','')
    target=ed.observation_targets.get(current)
    matches = target and ((target[0] in ('scene','mt') and target[1]==gid)
        or (target[0]=='mz' and target[1]==gid and h.current_record==('attempt',target[2]))
        or (target[0] in ('sample','attempt') and h.current_record==(target[0],target[1]))
        or (target[0]=='subject' and h.subject_key==target[1]))
    if not matches:
        current=ed.observation_mt_ids.get(gid) or next((i for i,s in ed.scene_tree_ids.items() if s==gid),None)
    if current and tree.exists(current):
        if tree.selection()!=(current,):tree.selection_set(current)
        tree.focus(current)
        ed._observation_selected_iid=current
        parent=current
        while tree.parent(parent):parent=tree.parent(parent)
        ed.tree_branch=parent
    return True


def open_gid(ed,gid):
    if not _active(ed):return None
    iid=ed.observation_mt_ids.get(gid) or next((i for i,s in ed.scene_tree_ids.items() if s==gid),None)
    if not iid:return False
    ed.review_tabs.select(0)
    ed.h.subject_tree.selection_set(iid);ed.h.subject_tree.see(iid)
    select(ed,iid);return True


def navigate(ed,direction):
    if not _active(ed):return False
    roots=list(ed.scene_tree_ids)
    current=getattr(ed,'tree_branch','')
    index=roots.index(current) if current in roots else 0
    target=roots[max(0,min(len(roots)-1,index+direction))]
    if target!=current:
        ed.h.subject_tree.selection_set(target);ed.h.subject_tree.see(target);select(ed,target)
    return True


def next_pending(ed):
    if not _active(ed):return False
    for b in ed.observation_model['blocks']:
        sid=b['scene_id']
        if not scene_review.scene(ed.h.session,sid,include_history=False)['checked']:
            open_gid(ed,sid);return True
        for gid in b['calls']:
            if ed.observation_states[gid]['pending']:
                open_gid(ed,gid);return True
    ed.message('Dostępne zadania rozstrzygnięte; niepełne dowody zachowują N/A.')
    return True
