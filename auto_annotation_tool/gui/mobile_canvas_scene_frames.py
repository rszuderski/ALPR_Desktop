"""Adapters for SCENE_FRAME rows in the existing Z4 tree and Canvas."""
from __future__ import annotations
from ..ranking import mobile_scene_frame_review as frames

def add_rows(ed, desired):
    ed.scene_tree_ids = {}
    ed.scene_mt_tree_ids = {}
    tree = ed.h.subject_tree
    for i, (key, row) in enumerate(getattr(ed.h.session, 'scene_frames', {}).items(), 1):
        iid = 'scene::' + key
        ed.scene_tree_ids[iid] = key
        desired.add(iid)
        if row['write_status'] == 'MISSING':
            label, tag = 'N/A · ' + row['missing_reason'], 'canvas_excluded'
        else:
            s = frames.scene(ed.h.session, key)
            label, tag = ('Zweryfikowana', 'canvas_ready') if s['checked'] else ('Do oceny', 'canvas_pending')
        text = f'Scena {i:02d} · {label}'
        if tree.exists(iid):
            tree.item(iid, text=text, values=('',)*6, tags=(tag,))
        else:
            tree.insert('', 'end', iid=iid, text=text, values=('',)*6, tags=(tag,))
        tree.move(iid, '', i-1)
    from ..ranking.mobile_mt_object_evaluation import evaluate_unit
    redirects=set()
    for gid,role in ed.h.session.mt_detection_roles.items():
        row=ed.h.session.scene_frames.get(role['scene_event_id'])
        if not row or row['write_status']!='WRITTEN' or role['source_stamp']!=row['source_stamp']:continue
        redirects.add(gid)
        iid='scene-mt::'+gid
        ed.scene_mt_tree_ids[iid]=(role['scene_event_id'],gid)
        desired.add(iid)
        index=len(ed.scene_mt_tree_ids)
        scored=evaluate_unit(ed.h.session,gid)
        state=('ocenione' if scored['n_qualified'] else 'anulowano · N/A' if role['stale_or_cancelled']
               else 'role niepełne · N/A' if role['role_proof_status']!='COMPLETE' else 'źródło FINAL')
        final_count=str(len(role['final_attempt_ids'])) if role['role_proof_status']=='COMPLETE' else '—'
        text=f"MT/ROI {index:02d} · {state} · FINAL {final_count}"
        if tree.exists(iid):tree.item(iid,text=text,values=('',)*6,tags=('canvas_ready' if scored['n_qualified'] else 'canvas_pending',))
        else:tree.insert('','end',iid=iid,text=text,values=('',)*6,tags=('canvas_pending',))
    from .mobile_canvas_editor import subject_groups
    for iid,subject in ed.h._subject_ids.items():
        groups=subject_groups(ed.h.session,subject)
        if groups and set(groups)<=redirects:desired.discard(iid)

def open_scene(ed, key):
    h = ed.h
    if h.current_invocation == key and h.current_record == ('scene', key):
        ed.sync();ed.request_overlay()
        return True
    if not ed.prepare_navigation():
        return True
    row = h.session.scene_frames[key]
    h.subject_key = ''
    # Retire the previous subject table before any queued selection callback
    # can run. Its rows are rebuilt by genuine MT/MZ navigation afterwards.
    h._record_ids = {}
    h.record_tree.delete(*h.record_tree.get_children())
    h._retained_image_on_refresh = None
    h._source_frame_keep_on_selection = ''
    h._multi_requested_attempt = ''
    # current_invocation is the existing Canvas active-unit slot; no MT record
    # or mt_invocation_id is created. The record kind and persisted scope are SCENE.
    h.current_record = ('scene', key)
    h.current_invocation = key
    h._image_rows = []
    h.gt_var.set(''); h.note_var.set('')
    if row['write_status'] == 'WRITTEN':
        h._image_rows = [('Pełna scena przed ROI', dict(image_entry=row['source_frame_entry'],
                            source_frame_view=True, scene_frame_view=True, scene_event_id=key))]
        evidence = getattr(h.session, 'tv_bbox_evidence', {})
        for frame in evidence.get('rows', ()):
            if frame['source_stamp'] == row['source_stamp'] and frame['crop_image']['status'] == 'WRITTEN':
                h._image_rows.append(('Wycinek TV — kontekst, GT na pełnej kamerze',
                    dict(image_entry=frame['crop_image']['entry'], context_only=True, tv_bbox_frame_key=frame['frame_key'])))
    h.image_choice.configure(values=[v[0] for v in h._image_rows])
    if h._image_rows:
        h.image_choice.current(0)
    else:
        h.image_choice.set('N/A: ' + str(row['missing_reason']))
    h.record_label.configure(text='Pełna scena · ASSISTED')
    h._load_image()
    ed.sync(); ed.update_report()
    if row['write_status']=='MISSING':
        message='SCENE_FRAME · MISSING · N/A: '+row['missing_reason']
        h.image_help.configure(text=message)
        h._paint_image(message)
    return True

def context(ed, model):
    s = ed._read()
    session = ed.h.session
    rows = frames.summary(session)
    selected = next((o for o in s['objects'] if o['object_id'] == ed.selected), None)
    row = session.scene_frames[s['scene_event_id']]
    note = ('Na pełnej scenie nie ma tablic — potwierdzone' if s['presence']=='NO_PLATES' else
            'Pełna scena zweryfikowana' if s['checked'] else 'D: dodaj tablicę · Spacja: świadomie potwierdź brak tablic')
    cover = frames.coverage(session, s['scene_event_id'])
    note += '\nMT: ' + row['mt_execution_status'] + ((' · ' + row['mt_not_run_reason']) if row['mt_not_run_reason'] else '')
    note += '\nPokrycie ROI: ' + (cover['reason'] or ', '.join(r['coverage'] for r in cover['rows']) or 'brak mianownika')
    note += '\nE2E TV: N/A · SCENE_UNMAPPED'
    model = getattr(ed, 'observation_model', None)
    if model:
        block = next((b for b in model['blocks'] if b['scene_id'] == s['scene_event_id']), None)
        if block:
            start = (block['start_ns'] - block['session_start_ns']) / 1e9
            end = (block['end_ns'] - block['session_start_ns']) / 1e9
            note += f'\nBlok obserwacji: {start:.3f}–{end:.3f} s · Stabilność: N/A'
    object_text = 'Wybierz ramkę na pełnej scenie'
    if selected:
        object_text = 'GTR · fizyczna tablica\nGTZ: ' + (selected['gt_text'] or selected['text_exclusion_reason'] or 'nieuzupełniony')
        if selected['quad'] is None:
            object_text += '\nPolygon IoU / NMCE: N/A — narożniki niedostępne'
        elif selected.get('mt_attempt_links'):
            object_text += '\nPolygon IoU / NMCE: w ocenie powiązanego MT/ROI'
        else:
            object_text += '\nPolygon IoU / NMCE: N/A — brak powiązanej pary MT'
    scene_na = rows['n_missing']
    caption = f"Sceny: {rows['n_assessed']}/{rows['n_scene_events']} ocenione · N/A źródła: {scene_na}"
    caption += f"\nMT/ROI: {len(session.mt_invocations)} wywołań · attempts: {len(session.attempts)}"
    counts = dict(inspected=f"{rows['n_assessed']}/{rows['n_scene_events']}",
                  remaining=str(rows['n_scene_events']-rows['n_assessed']-scene_na), decisions=str(int(not s['checked'])))
    inventory = ed.inventory()
    if inventory.get('operator_review_progress'):
        from ..ranking.mobile_operator_progress import image_counts, evidence_caption
        from ..ranking.mobile_review_status import approval_caption
        progress = inventory['operator_review_progress']
        human = image_counts(progress)
        counts.update(inspected=human['fraction'], remaining=str(human['remaining']) if human['remaining'] is not None else 'Nie ustalono')
        caption = approval_caption(inventory, session.review.review_status, session.review.review_revision)
        caption += '\n' + evidence_caption(progress)
    count = str(len(s['objects'])) if s['presence'] != 'UNRESOLVED' else '—'
    return dict(object=object_text, object_title='WYBRANA TABLICA', image=note, image_note=note,
        image_counts=dict(gt=count,tp='—',fp='—',fn='—'), bbox_counts=dict(tp='—',fp='—',fn='—'),
        bbox_note='Pełna scena jest odrębną populacją od MT/ROI', bbox_scope='SCENE_FRAME · ASSISTED',
        locator_target=None, session=caption, session_note='Brak potwierdzonego mapowania scen TV',
        session_counts=counts)


def source_finals(session, key):
    rows=[]
    source=session.scene_frames[key]['source_stamp']
    for role in session.mt_detection_roles.values():
        if role['scene_event_id']!=key or role['source_stamp']!=source:
            continue
        for aid in role['final_attempt_ids']:
            rows.append(dict(number=len(rows)+1, attempt_id=aid, role=role, row=session.attempts[aid]))
    return rows

def paint_sources(ed):
    ed.c.delete('scene_mt_source')
    from .mobile_canvas_visibility import gt_only
    if gt_only(ed):return
    s=ed._read()
    focus=getattr(ed,'scene_mt_focus',None)
    if not s or s.get('scope')!='SCENE_FRAME' or not focus or ed.mode=='draw' or ed.h._image is None:
        return
    role=ed.h.session.mt_detection_roles.get(focus)
    if not role:return
    roi=role['roi'];a,b=ed.to_canvas(roi[:2]),ed.to_canvas(roi[2:])
    ed.c.create_rectangle(*a,*b,outline='#B98A24',width=1,dash=(6,4),tags=('scene_mt_source',))
    for item in source_finals(ed.h.session,s['scene_event_id']):
        if item['role']['mt_invocation_id']!=focus:continue
        try:box=[float(item['row'][k]) for k in ('plate_left','plate_top','plate_right','plate_bottom')]
        except (TypeError,ValueError,KeyError):continue
        a,b=ed.to_canvas(box[:2]),ed.to_canvas(box[2:])
        ed.c.create_rectangle(*a,*b,outline='#169EC0',width=1,tags=('scene_mt_source',))
        ed.c.create_text(a[0]+3,a[1]-5,text=f"FINAL MT {item['number']}",fill='#169EC0',
                         anchor='sw',tags=('scene_mt_source',))
    ed.c.tag_lower('scene_mt_source')
    if ed.c.find_withtag('evidence_image'):
        ed.c.tag_raise('scene_mt_source','evidence_image')


def paint_tv_bbox(ed):
    """A separate inference-domain outline; all editable GT stays camera-full."""
    ed.c.delete('tv_bbox_domain')
    session = ed.h.session
    evidence = getattr(session, 'tv_bbox_evidence', {})
    if not evidence.get('available') or ed.h._image is None or ed.mode == 'draw':
        return
    state = ed._read()
    if not state or ed.h._image_entry != state.get('image_entry'):
        return
    frozen = evidence['descriptor']
    if tuple(ed.h._image_source_size) != (frozen['source_width_px'], frozen['source_height_px']):
        return
    if state.get('scope') == 'SCENE_FRAME':
        stamp = session.scene_frames[state['scene_event_id']]['source_stamp']
    else:
        group = session.mt_invocations.get(ed.h.current_invocation)
        if not group or not group.records:
            return
        row = group.records[0]
        from ..ranking.mobile_tv_bbox_evidence import STAMP
        # Existing CSV strings are compared exactly to producer integers below.
        stamp = {k: row.get(k) for k in STAMP}
    matches = [r for r in evidence['rows'] if all(str(r['source_stamp'].get(k)) == str(v) for k, v in stamp.items())]
    if len(matches) != 1 or matches[0]['pipeline_status'] in {'REJECTED_GEOMETRY', 'SOURCE_UNAVAILABLE'}:
        return
    box = frozen['tv_bbox_px']
    a, b = ed.to_canvas((box['left'], box['top'])), ed.to_canvas((box['right'], box['bottom']))
    ed.c.create_rectangle(*a, *b, outline='#A672D4', width=1.5, dash=(8, 4), tags=('tv_bbox_domain',))
    ed.c.create_text(a[0]+4, a[1]+4, text=f"TV BBOX · offset ({box['left']},{box['top']}) · {evidence['status']}",
                     anchor='nw', fill='#A672D4', tags=('tv_bbox_domain',))
    if ed.c.find_withtag('evidence_image'):
        ed.c.tag_lower('tv_bbox_domain')
        ed.c.tag_raise('tv_bbox_domain', 'evidence_image')



def show_presence_only(ed):
    from .mobile_canvas_clean_view import sync_image_actions
    if not ed.available() or (ed._read() or {}).get('scope')!='SCENE_FRAME':
        return
    ed.clean_reason_kind='presence'
    ed.clean_exclusion_context=(id(ed.h.session),ed.h.current_invocation)
    ed.clean_reason.set('')
    ed.clean_reason_label.configure(text='Przyczyna braku narożników widocznej tablicy:')
    ed.clean_reason_save.configure(text='Zapisz tablicę')
    ed.clean_exclusion.pack(fill='x',pady=(5,0),before=ed.contexts.image_actions)
    sync_image_actions(ed)
    ed.clean_reason_entry.focus_set()


def sync_empty_action(ed, scene, enabled):
    if scene.get('scope') != 'SCENE_FRAME' or scene['objects']:
        return
    button = ed.clean_gtr_confirm_button
    button.configure(text='Brak tablic na pełnej scenie potwierdzony' if scene['presence']=='NO_PLATES'
        else 'Na pełnej scenie nie ma tablic · Spacja', command=ed.confirm_empty_image,
        state='normal' if enabled and scene['presence']!='NO_PLATES' and ed.mode!='draw' else 'disabled')
    button.pack(fill='x', pady=(4,0))
