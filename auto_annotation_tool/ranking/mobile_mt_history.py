"""Read-only evidence for non-editable MT attempts; never a metric eligibility gate.

An attempt cancellation is not proof that the MT backend was interrupted.
Only exact source identities bind telemetry; an archived quad is not proof of
completed postprocessing. All returned coordinates are recorded Pose points.
"""
from __future__ import annotations

import math
from functools import lru_cache

from .mobile_mt_postdedup import _source_integer, _trace_index
from .mobile_source_frame_roi import source_invocation_context, source_pose_quad


@lru_cache(maxsize=64)
def _image_size(path, sha, size, mtime, entry):
    from .mobile_source_frame_roi import read_mobile_source_frame
    from .mobile_package_experiments import read_mobile_sample_image
    reader = read_mobile_source_frame if entry.startswith('samples/frames/') else read_mobile_sample_image
    try:
        image = reader(path, entry)
        result = image.size
        image.close()
        return result
    except (OSError, ValueError, KeyError):
        return None


def _source_evidence(session, group):
    """Reuse the validated-archive reader and exact three-part telemetry index."""
    try:
        session.verify_source()
        stat = session.path.stat()
        sha = session.bundle.source_archive_sha256
    except (AttributeError, OSError, ValueError):
        return {}, None, ['Nie potwierdzono archiwum źródłowego.']
    names = session.entry_names
    entries = {str(r.get(k) or '') for r in group.records
               for k in ('source_frame_entry', 'mt_input_evidence_entry', 'evidence_entry')}
    images = {p:_image_size(str(session.path),sha,stat.st_size,stat.st_mtime_ns,p)
              for p in entries if p in names and p.startswith(('samples/frames/','samples/evidence/'))}
    row = group.records[0]
    key = tuple(_source_integer(row.get(k)) for k in
                ('source_sequence','source_timestamp_nanos','scene_generation'))
    try:
        traces = _trace_index(str(session.path),sha,stat.st_size,stat.st_mtime_ns).get(key,[]) if None not in key else []
    except (TypeError,ValueError):
        traces=[]
    trace = traces[0] if len(traces)==1 else None
    warnings = []
    if trace:
        for field in ('visual_epoch','camera_transform_generation'):
            if _source_integer(trace.get(field)) != _source_integer(row.get(field)):
                warnings.append('Niezgodna tożsamość śladu: '+field)
                trace = None
                break
        if trace:
            counters=trace.get('counters') if isinstance(trace.get('counters'),dict) else {}
            if any(_source_integer(counters.get(k)) != _source_integer(row.get(f))
                   for k,f in (('source_width','source_frame_width_px'),('source_height','source_frame_height_px'))):
                warnings.append('Niezgodne wymiary klatki w śladzie; nie przypisano zakończenia MT.')
                trace=None
    if len(traces)>1:
        warnings.append('Niejednoznaczny ślad tej samej klatki; nie przypisano telemetrii.')
    return images, trace, warnings


def history_evidence(session, gid, *, image_entry=None, image_size=None):
    """Describe only a non-editable invocation, without creating a Canvas scene."""
    group = getattr(session,'mt_invocations',{}).get(gid) if session else None
    if not group:
        return None
    if (group.executed and not group.cancelled and not group.execution_failed and
            source_invocation_context(group.records,gid,session.entry_names)):
        return None
    result = dict(gid=gid, images={}, detections=[], warnings=[], trace=None,
        image_status='Brak zdjęcia', list_label='Brak zdjęcia', view_bound=False,
        mt_status='MT: status zakończenia NIEUSTALONY',
        pipeline_status='Potok: status nieustalony', finality='Końcowy wynik MT: NIEUSTALONY',
        cancellation_stage='NIEUSTALONY', cancellation_timestamp=None,
        reason='Brak potwierdzonego wykonania; tylko inspekcja, bez edycji i oceny.',
        mz_status='MZ: brak potwierdzonego odczytu', metric_eligible=False, roi=None)
    records = group.records
    legacy=bool(group.legacy_identity and not group.executed)
    if (group.mt_invocation_id!=gid or group.session_id!=session.session_id or not records or
        any(str(r.get('session_id') or '')!=session.session_id or
            (str(r.get('mt_invocation_id') or '')!=gid and not
             (legacy and not r.get('mt_invocation_id') and str(r.get('id'))==gid)) or
            session.attempts.get(str(r.get('id') or ''))!=r for r in records)):
        result['warnings'].append('Niezgodne powiązanie wywołania, sesji lub rekordu detekcji.')
        return result
    images, trace, warnings = _source_evidence(session,group)
    result.update(images=images, trace=trace, warnings=warnings)
    ctx = source_invocation_context(records,gid,session.entry_names)
    full_ok = bool(ctx and images.get(ctx['entry'])==tuple(ctx['size']))
    readable = {p:s for p,s in images.items() if s}
    if full_ok:
        result['image_status']='Pełna klatka + ROI · tylko podgląd'
    elif any(p.startswith('samples/frames/') for p in readable):
        result['image_status']='Podgląd klatki · brak pełnego powiązania'
    elif readable:
        result['image_status']='Tylko podgląd ROI / zapisanego obrazu'
    result['list_label'] = ('Anulowany potok' if group.cancelled and full_ok else
        'Tylko podgląd' if any(p.startswith('samples/frames/') for p in readable) else
        'Tylko podgląd ROI' if readable else 'Brak zdjęcia')
    if ctx and not full_ok:
        result['warnings'].append('Klatka niedostępna lub jej rozmiar nie odpowiada metadanym.')
    if not ctx:
        result['warnings'].append('Brak zgodnej pełnej klatki/ROI; nie rysujemy obrysu na tym źródle.')
    reasons = sorted({str(r.get('cancel_reason') or '').strip() for r in records}-{''})
    if group.cancelled:
        result['pipeline_status']='Potok: anulowano / wynik nieaktualny'
        result['reason']=('Scena zastąpiona nowszą (scene_superseded).' if reasons==['scene_superseded'] else
                          'Powód przerwania: '+(', '.join(reasons) or 'nie zapisano')+'.')
        result['reason']+=' Próba poza oceną MT/MZ; nie jest FN. Edycja i zatwierdzanie zablokowane.'
    elif group.execution_failed:
        result['pipeline_status']='Potok: błąd wykonania'
    statuses={str(r.get('mz_status') or '').upper() for r in records}
    if statuses=={'NOT_RUN'}:
        result['mz_status']='MZ: nie uruchomiono'
    saved=[]
    if full_ok:
        indices=[]
        for r in records:
            idx=_source_integer(r.get('mt_detection_index'))
            quad=source_pose_quad(r,ctx['size'])
            if (idx is not None and quad is not None and
                    str(r.get('mt_status') or '').upper() in {'VALID_QUAD','DETECTION_INVALID_QUAD'}):
                indices.append(idx)
                saved.append(dict(attempt_id=r['id'],detection_index=idx,quad=quad,
                    image_entry=ctx['entry'],image_quad=quad))
        if len(indices)!=len(set(indices)):
            saved=[];result['warnings'].append('Powielony indeks detekcji; inspekcja obrysu zablokowana.')
    recorded=sum(str(r.get('mt_status') or '').upper() in {'VALID_QUAD','DETECTION_INVALID_QUAD'} for r in records)
    if group.executed and recorded:
        result['mt_status']=f'MT: zachowany wynik · detekcje w zapisie: {recorded}'
    elif not group.executed:
        result['mt_status']='MT: nie uruchomiono' if all(str(r.get('mt_status')).upper()=='NOT_RUN' for r in records) else result['mt_status']
    # A measured duration alone may describe an interrupted call. Require
    # postprocessing and a complete output inventory before saying completed.
    stage=(trace or {}).get('stage_ms')
    counts=(trace or {}).get('counters')
    stage=stage if isinstance(stage,dict) else {}
    counts=counts if isinstance(counts,dict) else {}
    try:
        duration=stage['plate_inference']
        timed=not isinstance(duration,bool) and math.isfinite(float(duration)) and float(duration)>0
        post=stage['plate_postprocess']
        timed=timed and not isinstance(post,bool) and math.isfinite(float(post)) and float(post)>=0
    except (TypeError,ValueError,KeyError):
        timed=False
    raw=_source_integer(counts.get('plate_detections_raw'))
    final=_source_integer(counts.get('plate_detections_after_dedup'))
    suppressed=_source_integer(counts.get('plate_detections_suppressed'))
    positive_output=final==recorded==len(records)==len(saved) and recorded>0
    zero_output=(final==recorded==0 and len(records)==1 and
        str(records[0].get('mt_status')).upper()=='NO_DETECTION' and
        records[0].get('mt_detection_index') in (None,''))
    final_proof=bool(timed and group.executed and not group.execution_failed and full_ok and
        all(str(r.get('mt_executed')).lower()=='true' for r in records) and
        _source_integer(counts.get('mt_runs_this_frame'))==1 and
        None not in (raw,final,suppressed) and raw==final+suppressed and suppressed==0 and
        (positive_output or zero_output) and
        all(_source_integer(r.get('mt_detection_count'))==recorded for r in records))
    complete=final_proof
    if complete and group.executed and not group.execution_failed:
        result['mt_status']='MT: zakończona inferencja (ślad i kompletny wynik)'
        if group.cancelled:
            result['pipeline_status']='Potok: anulowano; MT ma ukończony pomiar'
    if final_proof:
        result['finality']='Końcowy wynik MT: potwierdzony śladem po deduplikacji'
    else:
        result['finality']='Finalizacja / post-NMS: NIEUSTALONE'
    # Never derive an exact cancellation instant or stage from aggregate times.
    result['mt_completion_confirmed']=complete
    result['final_detection_confirmed']=final_proof
    if trace and str(trace.get('status') or '') not in ('scene_superseded','cancelled','session_stopped') and group.cancelled:
        result['warnings'].append('Status potoku w śladzie nie potwierdza anulowania próby.')
    if image_entry is not None:
        result['view_bound']=bool(full_ok and image_entry==ctx['entry'] and tuple(image_size or ())==tuple(ctx['size']))
        if result['view_bound']:
            result['roi']=ctx['roi'];result['detections']=saved
        elif saved:
            result['warnings'].append('Bieżący obraz/rozmiar nie odpowiada źródłu ramki; nakładka zablokowana.')
    else:
        result['detections']=saved
    return result


def history_assessment(model, evidence, selected=''):
    """Display-only overlay on an unavailable Canvas assessment."""
    model=dict(model, execution=dict(model['execution']))
    model['execution']['title']=evidence['mt_status']
    model.update(pipeline=evidence['pipeline_status'],image_status=evidence['image_status'],
        focus='Kliknij błękitny obrys: inspekcja bez edycji' if evidence['detections'] else 'Brak bezpiecznej ramki do wskazania',
        frame=('Zakończenie inferencji: NIEUSTALONE\n' if not evidence.get('mt_completion_confirmed') else '')+evidence['finality'],
        image_frames='Etap i chwila anulowania: NIEUSTALONE',
        geometry='IoU / NMCE — brak ocenialnej pary obrysów',mz=evidence['mz_status'],
        text='CER / Exact Match — nie dotyczy',outcome=evidence['reason'],
        gt_note='Historia tylko do odczytu. Wskazanie ramki nie zatwierdza obrazu.',
        history_details='',history_selected=selected)
    if evidence['warnings']:
        model['outcome']+='\n'+'\n'.join(evidence['warnings'])
    item=next((d for d in evidence['detections'] if d['attempt_id']==selected),None)
    if item:
        model['focus']=f"MT {item['detection_index']+1} · inspekcja tylko do odczytu"
        model['history_details']=(f"attempt_id: {item['attempt_id']}\nmt_invocation_id: {evidence['gid']}\n"
            f"Indeks detekcji: {item['detection_index']}\nObraz: {item['image_entry']}\n"
            f"ROI: {evidence['roi']}\n{evidence['finality']}\n{evidence['reason']}")
    return model
