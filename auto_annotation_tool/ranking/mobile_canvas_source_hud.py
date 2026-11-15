"""Read-only, source-frame execution counts, separate from quality metrics."""
from .mobile_source_frame_roi import source_invocation_context
from .mobile_mt_postdedup import _trace_index,_source_integer


def _context(session,gid):
    group=session.mt_invocations[gid]
    return source_invocation_context(group.records,gid,session.entry_names)


def _identity(group,ctx):
    fields=('scene_generation','visual_epoch','camera_transform_generation')
    values=[tuple(str(r.get(k,'')).strip() for k in fields) for r in group.records]
    if (not ctx or not ctx['source_sequence'] or not ctx['source_timestamp_nanos']
            or not values or len(set(values))!=1 or not all(values[0])):return None
    return (ctx['entry'],ctx['source_sequence'],ctx['source_timestamp_nanos'],tuple(ctx['size']),*values[0])


def frame_execution(session,gid,*,image_entry=None,image_size=None):
    result=dict(mt=None,final=None,mz=None,reason='brak wiarygodnego śladu',
                mt_invocation_ids=[],final_attempt_ids=[],mz_execution_ids=[],proof=[],frame=None,final_roles_proven=True)
    if session is None or gid not in session.mt_invocations:return result
    session.verify_source()
    group=session.mt_invocations[gid];ctx=_context(session,gid);identity=_identity(group,ctx)
    if (identity is None or (image_entry is not None and image_entry!=ctx['entry'])
            or (image_size is not None and tuple(image_size)!=tuple(ctx['size']))):return result
    result['frame']=dict(entry=ctx['entry'],source_sequence=ctx['source_sequence'],
        source_timestamp_nanos=ctx['source_timestamp_nanos'],size=ctx['size'],source_sha=session.review.source_archive_sha256)
    groups=[]
    for key,other in session.mt_invocations.items():
        other_ctx=_context(session,key)
        if _identity(other,other_ctx)==identity:groups.append((key,other,other_ctx))
        elif any(r.get('source_frame_entry')==ctx['entry'] for r in other.records):
            result['reason']='sprzeczna tożsamość klatki';return result
    from .mobile_canvas_presentation import execution_state
    rows=[];final=0
    for key,other,other_ctx in groups:
        rows.extend(other.records)
        state=execution_state(session,key,image_entry=ctx['entry'],image_size=ctx['size'])
        if state['state']=='MT_NOT_RUN':continue
        if state['state']!='MT_EXECUTED' or other.legacy_identity:
            result['reason']=('anulowano' if other.cancelled else 'błąd wykonania' if other.execution_failed else 'niekompletne dane MT')
            return result
        if any(str(r.get('mt_invocation_id') or '')!=key or str(r.get('mt_executed')).lower() not in ('true','1') for r in other.records):return result
        result['mt_invocation_ids'].append(key)
        # execution_state proves execution/completeness, while roles proves
        # membership after the source NMS. A detection row alone is not FINAL.
        from . import mobile_canvas_review as core,mobile_candidate_stacks as stacks
        try:roles=stacks.roles(session,key,core.scene(session,key))
        except (ValueError,KeyError,TypeError):roles={'?':'UNKNOWN'}
        if any(role=='UNKNOWN' for role in roles.values()):final=None;result['final_roles_proven']=False
        elif final is not None:final+=sum(role=='FINAL' for role in roles.values())
        result['final_attempt_ids'].extend(aid for aid,role in roles.items() if role=='FINAL')
    result.update(mt=len(set(result['mt_invocation_ids'])),final=final)
    result['proof'].append('source_invocation_context + execution_state; unique MT invocation IDs, FINAL after source NMS')
    stat=session.path.stat()
    key=tuple(_source_integer(group.records[0].get(k)) for k in ('source_sequence','source_timestamp_nanos','scene_generation'))
    traces=_trace_index(str(session.path),session.review.source_archive_sha256,stat.st_size,stat.st_mtime_ns).get(key,[])
    trace=traces[0] if len(traces)==1 else None
    counters=(trace or {}).get('counters',{})
    verified=bool(trace and all(str(trace.get(k))==identity[index] for k,index in (('visual_epoch',5),('camera_transform_generation',6)))
        and all(counters.get(k) in (None,ctx['size'][i]) for i,k in enumerate(('source_width','source_height'))))
    if verified:
        mt=_source_integer(counters.get('mt_runs_this_frame'));n=_source_integer(counters.get('plate_detections_after_dedup'))
        if mt!=result['mt'] or (n is not None and result['final'] is not None and n!=result['final']):
            result.update(mt=None,final=None,reason='niekompletny lub sprzeczny ślad MT');return result
        if result['final'] is None and n is not None:result['final']=n
        mz=_source_integer(counters.get('mz_runs'))
        executed={str(r.get('attempt_id') or r.get('id')) for r in rows if str(r.get('mz_executed')).lower() in ('true','1')}
        if mz is not None and mz>=len(executed):
            if mz and all(r.get('mz_status')=='NOT_RUN' for r in rows):
                result['reason']='sprzeczny ślad MZ';return result
            result.update(mz=mz,reason='' if result['final'] is not None else 'brak wiarygodnego śladu FINAL')
            result['proof'].append(dict(trace_frame_id=trace.get('frame_id'),trace_key=key,counter='mz_runs',value=mz))
            return result
        if mz is not None:result['reason']='sprzeczny ślad MZ';return result
    elif traces:
        result['reason']='niejednoznaczny ślad klatki';return result
    # A NOT_RUN record proves zero without inferring execution from OCR text.
    if rows and all(r.get('mz_status')=='NOT_RUN' and str(r.get('mz_executed')).lower() not in ('true','1') for r in rows):
        result.update(mz=0,reason='' if result['final'] is not None else 'brak wiarygodnego śladu FINAL');result['proof'].append('complete source records: MZ NOT_RUN');return result
    executed=[r for r in rows if str(r.get('mz_executed')).lower() in ('true','1')]
    ids={str(r.get('mz_execution_id') or '').strip() for r in executed}
    if executed and '' not in ids and all(r in executed or r.get('mz_status')=='NOT_RUN' for r in rows):
        result.update(mz=len(ids),mz_execution_ids=sorted(ids),reason='' if result['final'] is not None else 'brak wiarygodnego śladu FINAL');result['proof'].append('explicit source mz_execution_id')
    return result


def caption(value):
    number=lambda key:'—' if value[key] is None else str(value[key])
    text=f"Bieżąca klatka · MT wyw.: {number('mt')} · FINAL: {number('final')} · MZ wyw.: {number('mz')}"
    if any(value[k] is None for k in ('mt','final','mz')):text+=' · '+value['reason']
    if not value.get('final_roles_proven',True):text+=' · role ID nieudowodnione · N/A'
    return text


def update(editor):
    h=editor.h
    try:
        stamp=h.session.path.stat() if h.session else None
        key=(id(h.session),h.current_invocation,getattr(stamp,'st_size',None),getattr(stamp,'st_mtime_ns',None),h._image_entry,tuple(h._image_source_size or ()))
        if getattr(editor,'_source_hud_key',None)!=key:
            editor._source_hud_value=frame_execution(h.session,h.current_invocation,image_entry=h._image_entry,image_size=h._image_source_size)
            editor._source_hud_key=key
        value=editor._source_hud_value
    except (ValueError,KeyError,TypeError,OSError):
        value=dict(mt=None,final=None,mz=None,reason='brak wiarygodnego śladu')
    text=caption(value)
    width=max(140,editor.c.winfo_width()-12)
    if editor.source_hud.winfo_pixels(editor.source_hud.cget('wraplength'))!=width:editor.source_hud.configure(wraplength=width)
    if editor.source_hud.cget('text')!=text:editor.source_hud.configure(text=text)
