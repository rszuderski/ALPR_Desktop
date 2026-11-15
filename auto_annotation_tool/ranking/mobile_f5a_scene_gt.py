"""One reviewed TV exposure, source-bound CameraX geometry, existing sidecar.

A2 supplies content membership, never a geometric transform. This adapter
reuses photograph GT decisions/history, the existing matcher and OCR scorer.
It does not write invocation annotations, A1, A2 or the old A3 layer.
"""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
from collections import Counter
import copy, csv, hashlib, io, json, math, uuid, zipfile

from PIL import Image
from . import mobile_f5a_assisted as a2
from . import photo_reference as photo
from ..registration_text import normalize_registration

KEY='f5a_scene_gt_v1'
SCHEMA='alpr.mobile.f5a_scene_gt.v1'
POLICY=dict(id='f5a_scene_first_input_frame_v1', chronology='RETROSPECTIVE_NOT_PREREGISTERED',
    main='FIRST_A2_VERIFIED_MT_INPUT_BY_CAPTURE_ELAPSED_NS_THEN_FRAME_ID',
    same_frame_calls='ALL_RECORDED_ROIS_IN_ARCHIVE_ORDER_INCLUDING_FAILED_CALLS',
    no_completed_application_output_claim=True, clock_conversion='NONE',
    geometry='IDENTICAL_FULL_CAMERAX_SHA256_AND_DIMENSIONS_ONLY',
    photo_matching='A2_VERIFIED_FRAME_CONTENT_NOT_SCENE_EVENT_ID',
    detector='EXISTING_CONFIDENCE_GREEDY_BBOX_IOU_0_50_NO_REFILTER',
    ocr='FIRST_SOURCE_CALL_AND_DETECTION_FOR_GT_NO_BEST_READ_SELECTION',
    other_frames='CONTENT_ONLY_GEOMETRY_AND_ASSOCIATION_NA_UNLESS_OPERATOR_EXCEPTION',
    scene_success='ALL_CONTENT_GT_DETECTED_AND_FULL_READABLE_TEXT_EXACT_WITH_NO_EXTRA_FINALS',
    no_plates_success='EXECUTED_MAIN_INPUT_WITH_ZERO_FINALS',
    partial_text='NOT_FULL_OCR_GT', optional='DETECTED_AT_LEAST_ONCE_SEPARATE_FROM_MAIN')

EXPOSURE_REVIEW='exposure_review'
EXPOSURE_SCHEMA='alpr.mobile.f5a_exposure_review.v1'
REVIEW_PROTOCOL=dict(id='f5a_recorded_pipeline_review_v1',
    A='EVERY_SOURCE_BOUND_INPUT_OUTPUT_MULTISET_RAW_AND_CONSENSUS_SEPARATELY',
    B='ANY_ARCHIVED_ANDROID_CONFIRMED_EXACT_FULL_GT_TEXT_DIAGNOSTIC_ONLY',
    C='UNCHANGED_FROZEN_FIRST_MT_ANALYSIS',normalization='uppercase_alphanumeric.v1',
    correspondence='TEXT_MULTISET_ONLY_NOT_PHYSICAL_PLATE_ASSOCIATION',
    unknown_gt='FULL_COMPLETENESS_NA_KNOWN_TEXT_COUNTS_RETAINED',
    cancelled='EXECUTION_FAILURE_NO_PUBLISHED_OUTPUT',
    no_mt='KNOWN_NOT_RUN_IS_SYSTEM_FAILURE_FOR_PRESENT_GT_NOT_MISSING_EVIDENCE',
    roi='SOURCE_READ_ONLY_NO_MP_ASSESSMENT',hud_terminal='NOT_RECONSTRUCTED',
    geometry='IOU_NMCE_CONDITIONAL_OCR_SEPARATE_NO_GEOMETRY_TRANSFER')
PROTOCOL_TEXT=(
    'Jedno zdjęcie TV w tej sesji = jedna ocena. GT pochodzi z oryginalnego zdjęcia A1.\n\n'
    'A. Sprawdzamy każdy zapisany komplet odczytów raw i konsensusów dla konkretnego wejścia. '
    'Porównujemy pełne numery i ich krotności; pokazujemy braki oraz dodatkowe odczyty. '
    'Nie wybieramy najlepszego wyniku według GT i nie przypisujemy tekstów do fizycznych tablic.\n\n'
    'B. Osobno: czy wystąpił choć jeden zapisany, potwierdzony przez Androida odczyt zgodny '
    'z pełnym numerem GT. To diagnostyka, nie końcowy wynik telefonu.\n\n'
    'C. Dawna ocena pierwszego wejścia MT pozostaje osobnym porównaniem.\n\n'
    'Niepełny tekst GT ogranicza ocenę kompletności, ale zachowujemy znane numery. '
    'Brak MT i anulowanie nie są dowodem braku tablic. Słaba czytelność CameraX nie wyłącza '
    'automatycznie oceny ALPR względem A1. IoU/NMCE i warunkowa jakość MZ pozostają oddzielne.\n\n'
    'Potwierdź ocenę = sprawdziłem rozliczenie, a nie „model rozpoznał poprawnie”. '
    'Zgłoś błąd zapisuje przypadek do wyjaśnienia; N/A wymaga powodu. '
    'Przeglądanie dowodów i przycisk Dalej nie zatwierdzają zdjęcia.')


def fingerprint(value):return a2.fingerprint(value)
def now():return datetime.now(timezone.utc).isoformat()
def _frame_order(frame):
    value=frame.get('capture_elapsed_nanos')
    return (value if type(value) is int else float('inf'),frame['frame_id'])
def _bbox(quad):return [min(p[0] for p in quad),min(p[1] for p in quad),max(p[0] for p in quad),max(p[1] for p in quad)]


def representative_roi_coverage(store,eid):
    """Selected-ROI evidence only, including zero ROI without an MT call.

    This never evaluates MP or manufactures MT FN / a terminal app result.
    """
    state=store.state(eid);frame=state['frame'];s=store.mobile
    result=dict(status='N/A',reason='SCENE_GT_NOT_APPROVED',rows=[],selected_rois=None,scope='REPRESENTATIVE_CAMERA_FRAME_ONLY')
    if not store.approved(eid) or not frame:return result
    from .mobile_canvas_review import context
    actual=[]
    for gid,call in store.index.calls.items():
        if call['exposure_id']!=eid or not call['frame_id']:continue
        other=store.index.verify_frame(call['frame_id']);group=s.mt_invocations[gid]
        if other['full_frame_sha256']==frame['full_frame_sha256'] and other['source_stamp']==frame['source_stamp'] and group.executed and not group.cancelled:
            ctx=context(s,gid)
            if ctx:actual.append(dict(zip(('left','top','right','bottom'),ctx['roi'])))
    observations=[m for r in s.scene_frames.values() for m in r['mp_observations']
        if m['source_stamp']==frame['source_stamp'] and m['execution_status']=='EXECUTED']
    selections=[m['selected_rois'] for m in observations]
    if actual:
        selections=[actual];result['roi_origin']='ACTUAL_EXECUTED_MT_INPUTS_IN_SOURCE_ORDER'
    else:result['roi_origin']='RECORDED_SELECTED_ROIS_NO_MT_EXECUTION_ASSUMED'
    if not selections:result['reason']='SELECTED_ROI_EVIDENCE_UNPROVEN';return result
    if any(r!=selections[0] for r in selections):result['reason']='SELECTED_ROI_EVIDENCE_CONFLICT';return result
    rois=selections[0];result.update(status='ASSESSED',reason='',selected_rois=copy.deepcopy(rois))
    for p in state['objects']:
        q=p.get('quad');coverage='OUTSIDE_ROI' if not rois else 'N/A'
        if q and rois:
            l,t,r,b=_bbox(q)
            if any(v['left']<=l<r<=v['right'] and v['top']<=t<b<=v['bottom'] for v in rois):coverage='COVERED'
            elif all(r<=v['left'] or l>=v['right'] or b<=v['top'] or t>=v['bottom'] for v in rois):coverage='OUTSIDE_ROI'
            else:coverage='PARTIAL_ROI'
        result['rows'].append(dict(gt_id=p['object_id'],coverage=coverage,mt_fn=None,
            potential_alpr_coverage_loss=coverage=='OUTSIDE_ROI'))
    return result


class Index:
    """Read-only, exact-frame membership for every recorded call, including N/A."""
    def __init__(self, session, reference):
        self.session=session;self.ref=reference;reference.verify();session.verify_source()
        record=reference.sessions.get(session.session_id)
        if not record or record['archive_sha256']!=session.review.source_archive_sha256:
            raise ValueError('Sesja/SHA nie odpowiada źródłom A2.')
        self.exposures={e['exposure_id']:e for e in sorted(reference.exposures.values(),key=lambda e:e['planned_position']) if e['session_id']==session.session_id}
        self.frames={f['frame_id']:f for f in reference.frames.values() if f['session_id']==session.session_id}
        self.calls={};self._verified={};self.diagnostics=[]
        for gid,group in session.mt_invocations.items():
            # Desktop groups cancelled pre-MT attempts under an attempt-id
            # fallback. Preserve those records, but never invent MT calls.
            if not any(str(r.get('mt_invocation_id') or '')==gid for r in group.records):
                self.diagnostics.extend(dict(attempt_id=r['id'],reason='NO_SOURCE_MT_INVOCATION_ID',
                    source_record=copy.deepcopy(session.attempts[r['id']])) for r in group.records)
                continue
            row=dict(mt_invocation_id=gid,frame_id=None,exposure_id=None,reason='A2_FRAME_UNPROVEN')
            try:
                fid=reference.call_frame(session,gid);frame=self.frames[fid]
                eid=reference.frame_exposure.get(fid)
                mapping=reference.calls[(session.session_id,gid)]
                if (not eid or frame.get('status')!='POTWIERDZONE' or frame.get('verified') is not True
                        or mapping.get('status')!='POTWIERDZONE'
                        or mapping.get('photo_sha256')!=frame.get('photo_sha256')
                        or any(reference.frame_exposure.get(other)!=eid or self.frames[other].get('status')!='POTWIERDZONE'
                               for other in mapping['frame_ids'])):
                    raise ValueError('A2_CONTENT_UNPROVEN_OR_OUTSIDE_GT')
                self.verify_frame(fid)
                row.update(frame_id=fid,exposure_id=eid,reason='')
            except (ValueError,KeyError,OSError) as exc:row['reason']=str(exc)
            self.calls[gid]=row
        self.main={};self.default_frames={};self.main_reasons={}
        for eid,e in self.exposures.items():
            ids={r['frame_id'] for r in self.calls.values() if r['exposure_id']==eid}
            fid=min(ids,key=lambda f:_frame_order(self.frames[f])) if ids else None
            ordered=all(type(self.frames[f].get('capture_elapsed_nanos')) is int for f in ids)
            # Selection is frozen before reading any GT state or OCR output.
            self.main[eid]=[g for g,r in self.calls.items() if fid and ordered and r['exposure_id']==eid
                and reference.frame_key(r['frame_id'])==reference.frame_key(fid)]
            self.main_reasons[eid]='' if fid and ordered else 'MAIN_INPUT_ORDER_UNPROVEN' if fid else 'NO_VERIFIED_MT_INPUT'
            candidates=sorted(e['evidence_frame_ids'],key=lambda f:_frame_order(self.frames[f]))
            self.default_frames[eid]=fid or next((f for f in candidates if self.frame_ok(f)),None)

    def frame_ok(self,fid):
        try:self.verify_frame(fid);return self.frames[fid].get('status')=='POTWIERDZONE'
        except (ValueError,KeyError,OSError):return False

    def verify_frame(self,fid):
        self.session.verify_source();f=self.frames[fid]
        if f.get('verified') is not True or f['full_frame_entry'] not in self.session.entry_names:
            raise ValueError('Brak zweryfikowanej pełnej klatki CameraX.')
        entry=f['full_frame_entry'];digest=f['full_frame_sha256']
        if (entry,digest) not in self._verified:
            with zipfile.ZipFile(self.session.path) as z:raw=z.read(entry)
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('SHA klatki nie odpowiada A2.')
            with Image.open(io.BytesIO(raw)) as im:size=im.size
            self._verified[(entry,digest)]=size
        return dict(frame_id=fid,full_frame_entry=entry,full_frame_sha256=digest,
            scene_event_id=f.get('scene_event_id'),coordinate_space='mt_source_bitmap_xy',
            source_stamp=copy.deepcopy(f['source_stamp']),image_size=list(self._verified[(entry,digest)]))

    def frame_image(self,fid):
        self.verify_frame(fid)
        from .mobile_source_frame_roi import read_mobile_source_frame
        return read_mobile_source_frame(self.session.bundle,self.frames[fid]['full_frame_entry'])

    def frame_ids(self,eid):
        result=[];seen=set()
        for fid in sorted(self.exposures[eid]['evidence_frame_ids'],key=lambda f:_frame_order(self.frames[f])):
            key=self.ref.frame_key(fid)
            if key not in seen:result.append(fid);seen.add(key)
        return result


class CameraMaterial:
    def __init__(self,store):self.store=store;self.order=list(store.index.exposures)
    @property
    def images(self):
        result={}
        for eid in self.order:
            state=self.store.state(eid);f=state['frame'];size=f['image_size'] if f else [1,1]
            result[eid]=dict(image_id=eid,relative_path=f['full_frame_entry'] if f else '',
                width=size[0],height=size[1],source_file_sha256=f['full_frame_sha256'] if f else None)
        return result
    def verify(self,eid):
        frame=self.store.state(eid)['frame']
        if not frame:raise ValueError('Brak reprezentatywnej klatki CameraX; N/A.')
        return self.store.index.verify_frame(frame['frame_id'])
    def image(self,eid):return self.store.index.frame_image(self.verify(eid)['frame_id'])


class SceneReference(photo.PhotoReference):
    """Reuse A1 text/geometry operations; persistence is exclusively mobile sidecar."""
    def __init__(self,session,reference):
        self.mobile=session;self.index=Index(session,reference);self.path=session.sidecar_path
        self.material=CameraMaterial(self)
        self.data=copy.deepcopy(session.review.provenance.get(KEY) or dict(schema=SCHEMA,
            binding=reference.binding,session_id=session.session_id,source_archive_sha256=session.review.source_archive_sha256,
            annotation_protocol='CAMERAX_GT_WITH_A1_CONTENT_REFERENCE',geometry_seed='NONE',
            exposure_ids=list(self.index.exposures),
            policy=copy.deepcopy(POLICY),policy_sha256=fingerprint(POLICY),revision=0,
            last_image_id=next(iter(self.index.exposures)),reviews={},events=[]))
        if self.data['binding']!=reference.binding:raise ValueError('Zapis GT scen dotyczy innej wersji A1/A2.')
        if self.data['exposure_ids']!=list(self.index.exposures):raise ValueError('Niezgodny rejestr ekspozycji GT scen.')
        validate_layer(session,session.review)
        op=exposure_review_state(self.data)
        if op['last_context'] and not self.valid_context(op['last_context']):raise ValueError('Nieprawidłowy kontekst wznowienia.')
        for eid,r in op['reviews'].items():
            for v in r['versions'].values():
                state=v['state']
                if state is None:continue
                if ('img-sha256-'+state['photo_sha256'] not in reference.gt_states or
                        state['a2_photo_sha256']!=self.index.exposures[eid]['photo_sha256'] or
                        not self.valid_context(state['context']) or any(not self.valid_context(c) for c in state['viewed_evidence'])):
                    raise ValueError('Ocena ekspozycji odwołuje się do obcego GT lub dowodu.')
        for eid,r in self.data['reviews'].items():
            if eid not in self.index.exposures:raise ValueError('Obca ekspozycja w GT scen.')
            for v in r['versions'].values():
                state=v['state'];f=state['frame']
                if state['photo_sha256']!=self.index.exposures[eid]['photo_sha256']:raise ValueError('Treść GT nie odpowiada fotografii A2.')
                if f and (f!=self.index.verify_frame(f['frame_id']) or reference.frame_exposure.get(f['frame_id'])!=eid):
                    raise ValueError('GT sceny nie odpowiada klatce CameraX A2.')
                for exception in state['exceptions'].values():
                    call=self.index.calls.get(exception['mt_invocation_id'],{})
                    if call.get('exposure_id')!=eid or exception['frame']!=self.index.verify_frame(call['frame_id']):
                        raise ValueError('Ocena wyjątku nie odpowiada wejściu MT z A2.')

    def proposal(self,eid):
        e=self.index.exposures[eid];fid=self.index.default_frames[eid]
        canonical=copy.deepcopy(self.index.ref.gt_states[e['image_id']])
        objects=canonical['objects']
        for obj in objects:
            photo._explicit(obj)
            # Content and stable plate ID only. All JPEG geometry is discarded.
            obj.update(canonical_gt_id=obj['object_id'],quad=None,localization_quad=None,
                geometry_kind='UNAVAILABLE',geometry_status='UNDETERMINED',decision_version=3,
                approximate_region=None,approximate_point=None,geometry_exclusion_reason='Nie ustalono obrysu na CameraX')
        return dict(presence=canonical['presence'],objects=objects,
            frame=self.index.verify_frame(fid) if fid else None,approval=None,exceptions={},
            content_origin='A1_PROPOSAL_NOT_OPERATOR_APPROVAL',photo_sha256=e['photo_sha256'])

    def state(self,eid):
        r=self.data['reviews'].get(eid)
        return copy.deepcopy(r['versions'][r['timeline'][r['cursor']]]['state']) if r else self.proposal(eid)

    def scene(self,eid):
        result=super().scene(eid)
        result.update(scope='F5A_TV_EXPOSURE_CAMERAX',checked=self.approved(eid))
        return result

    def approved(self,eid):return bool(self.state(eid)['approval'])
    def status(self,eid):return 'Gotowe' if self.approved(eid) else 'Uzupełnij' if eid in self.data['reviews'] else 'Do oceny'
    def next_pending(self,key):
        order=self.material.order;i=order.index(key)
        return next((k for k in order[i+1:]+order[:i+1] if not self.approved(k)),None)
    def remember(self,key):pass  # browsing is read-only; last explicit save is the resume point

    def protocol_approval(self):return exposure_review_state(self.data)['approval']

    def _save_operator(self,layer,eid,action,context=None):
        self.index.ref.verify();self.mobile.verify_source()
        changed=copy.deepcopy(self.data);changed[EXPOSURE_REVIEW]=layer
        changed['revision']+=1
        if eid:changed['last_image_id']=eid
        if context:layer['last_context']=list(context)
        layer['events'].append(dict(action=action,exposure_id=eid,at=now(),revision=changed['revision'],origin='OPERATOR_EXPLICIT'))
        review=copy.deepcopy(self.mobile.review);review.provenance[KEY]=changed
        self.mobile._save_change(review);self.data=changed

    def accept_protocol(self,*,approval=None):
        if self.protocol_approval():return
        if approval is not None and (approval.get('binding')!=self.index.ref.binding or
                approval.get('protocol_sha256')!=fingerprint(REVIEW_PROTOCOL) or approval.get('origin')!='OPERATOR_EXPLICIT'):
            raise ValueError('Zatwierdzenie dotyczy innego protokołu lub źródeł.')
        layer=copy.deepcopy(exposure_review_state(self.data))
        layer['approval']=copy.deepcopy(approval) if approval else dict(origin='OPERATOR_EXPLICIT',actor=self.mobile.review.reviewer_id,
            at=now(),protocol_sha256=fingerprint(REVIEW_PROTOCOL),binding=self.index.ref.binding,
            approval_id=uuid.uuid4().hex,approved_in_session=self.mobile.session_id)
        self._save_operator(layer,None,'ACCEPT_PROTOCOL' if approval is None else 'USE_ACCEPTED_PROTOCOL')

    def operator_decision(self,eid):return exposure_decision(self.data,eid)

    def operator_photo(self,eid):
        row=self.operator_decision(eid)
        return row['photo_sha256'] if row else self.index.exposures[eid]['photo_sha256']

    def operator_status(self,eid):
        return {'CONFIRMED':'Potwierdzono','CORRECTED':'Skorygowano','ERROR_REPORTED':'Zgłoszony błąd',
            'NOT_ASSESSABLE':'Nie można ocenić'}.get((self.operator_decision(eid) or {}).get('decision'),'Nieocenione')

    def valid_context(self,context):
        if not context:return False
        kind,key,*rest=context
        if kind=='exposure':return key in self.index.exposures
        if kind=='frame':return key in self.index.frames
        if kind=='call':return key in self.index.calls
        if kind in ('attempt','mz','consensus'):
            return key in self.index.calls and len(rest)==1 and rest[0] in [r['id'] for r in self.mobile.mt_invocations[key].records]
        return False

    def decide_exposure(self,eid,decision,*,reason='',context=None,viewed=(),photo_sha=None,diagnostic_b=None,override_b=False):
        if not self.protocol_approval():raise ValueError('Najpierw zatwierdź protokół oceny.')
        if eid not in self.index.exposures:raise ValueError('Obca ekspozycja.')
        if decision not in ('CONFIRMED','ERROR_REPORTED','NOT_ASSESSABLE','CORRECTED'):raise ValueError('Nieprawidłowa decyzja.')
        if decision!='CONFIRMED' and not reason.strip():raise ValueError('Podaj krótki powód.')
        if override_b and (decision!='CORRECTED' or type(diagnostic_b) is not bool):raise ValueError('Korekta B wymaga jawnego Tak/Nie.')
        if photo_sha and photo_sha!=self.operator_photo(eid) and decision!='CORRECTED':raise ValueError('Zmiana zdjęcia wymaga jawnej korekty.')
        context=context or ('exposure',eid)
        if not self.valid_context(context) or any(not self.valid_context(v) for v in viewed):raise ValueError('Nieprawidłowy dowód źródłowy.')
        sha=photo_sha or self.operator_photo(eid)
        if 'img-sha256-'+sha not in self.index.ref.gt_states:raise ValueError('Zdjęcie nie należy do A1.')
        previous=self.operator_decision(eid)
        if decision=='CONFIRMED' and previous and previous['decision']=='CORRECTED':return  # no duplicate acceptance
        row=dict(origin='OPERATOR_EXPLICIT',decision=decision,reason=reason.strip(),actor=self.mobile.review.reviewer_id,
            at=now(),revision=self.data['revision']+1,protocol_id=REVIEW_PROTOCOL['id'],series_id=self.index.ref.binding['series_id'],session_id=self.mobile.session_id,exposure_id=eid,
            source_archive_sha256=self.mobile.review.source_archive_sha256,a2_photo_sha256=self.index.exposures[eid]['photo_sha256'],
            photo_sha256=sha,binding=self.index.ref.binding,protocol_sha256=fingerprint(REVIEW_PROTOCOL),
            viewed_evidence=[list(v) for v in dict.fromkeys(tuple(v) for v in (*viewed,context))],context=list(context),
            correction=dict(photo_reference=sha!=self.index.exposures[eid]['photo_sha256'],
                diagnostic_b_override=bool(override_b),diagnostic_b=diagnostic_b if override_b else None))
        if previous and all(previous.get(k)==row.get(k) for k in ('decision','reason','photo_sha256','correction')):return
        layer=copy.deepcopy(exposure_review_state(self.data))
        r=layer['reviews'].setdefault(eid,dict(versions={'initial':dict(state=None)},timeline=['initial'],cursor=0))
        vid=uuid.uuid4().hex;r['versions'][vid]=dict(state=row,at=row['at'],action=decision)
        r['timeline']=r['timeline'][:r['cursor']+1]+[vid];r['cursor']+=1
        self._save_operator(layer,eid,'DECIDE_EXPOSURE',context)

    def undo_exposure(self,eid,*,redo=False,context=None):
        layer=copy.deepcopy(exposure_review_state(self.data));r=layer['reviews'].get(eid)
        if not r:return
        cursor=r['cursor']+(1 if redo else -1)
        if not 0<=cursor<len(r['timeline']):return
        r['cursor']=cursor;self._save_operator(layer,eid,'REDO_EXPOSURE' if redo else 'UNDO_EXPOSURE',context or ('exposure',eid))

    def _change(self,eid,action,change):
        before=self.state(eid);state=copy.deepcopy(before);value=change(state)
        state=json.loads(json.dumps(state))
        if state==before:return value
        if action not in ('APPROVE_SCENE','EXCEPTION'):state.update(approval=None,exceptions={})
        self._validate_state(eid,state)
        changed=copy.deepcopy(self.data)
        r=changed['reviews'].setdefault(eid,dict(versions={'initial':dict(state=before,action='A1_CONTENT_PROPOSAL')},timeline=['initial'],cursor=0))
        vid=uuid.uuid4().hex;r['versions'][vid]=dict(state=state,action=action,at=now(),
            actor=self.mobile.review.reviewer_id,geometry_scope='SELECTED_CAMERA_FRAME_ONLY')
        r['timeline']=r['timeline'][:r['cursor']+1]+[vid];r['cursor']+=1
        self._save(changed,eid,action)
        return value

    def _save(self,changed,eid,action):
        self.index.ref.verify();self.material.verify(eid)
        changed['revision']=self.data['revision']+1;changed['last_image_id']=eid
        changed['events'].append(dict(action=action,exposure_id=eid,at=now(),revision=changed['revision']))
        review=copy.deepcopy(self.mobile.review);review.provenance[KEY]=changed
        # Do not relabel earlier reviews, migrate decisions or change A3 status.
        self.mobile._save_change(review);self.data=changed

    def approve(self,eid):
        state=self.state(eid)
        if self.approved(eid):return
        if not state['frame'] or not photo.complete(state):raise ValueError('Uzupełnij GT tablic lub jawne powody N/A.')
        def change(s):s['approval']=dict(at=now(),actor=self.mobile.review.reviewer_id,
            content_and_camera_readability_confirmed=True,source='OPERATOR_EXPLICIT_SCENE',
            content_sha256=fingerprint(dict(presence=s['presence'],objects=s['objects'],frame=s['frame'])))
        self._change(eid,'APPROVE_SCENE',change)

    def empty(self,eid):
        super().empty(eid);self.approve(eid)

    def choose_frame(self,eid,fid):
        if self.index.ref.frame_exposure.get(fid)!=eid or not self.index.frame_ok(fid):raise ValueError('Klatka nie ma potwierdzonej treści tej fotografii.')
        frame=self.index.verify_frame(fid)
        if self.state(eid)['frame']==frame:return
        def change(s):
            s['frame']=frame
            for obj in s['objects']:
                obj.update(quad=None,localization_quad=None,approximate_point=None,approximate_region=None,
                    decision_version=3,geometry_kind='UNAVAILABLE',geometry_status='UNDETERMINED',
                    geometry_exclusion_reason='Zmiana klatki CameraX; poprzednia geometria pozostaje w historii')
        self._change(eid,'CHOOSE_CAMERA_FRAME',change)

    def exception(self,eid,gid,aid,decision,*,gt_id=None,text_readable=None,reason=''):
        if not self.approved(eid):raise ValueError('Najpierw zatwierdź GT sceny.')
        if self.index.calls.get(gid,{}).get('exposure_id')!=eid:raise ValueError('Wywołanie nie należy do tej fotografii.')
        group=self.mobile.mt_invocations[gid]
        if not group.executed or group.cancelled or group.execution_failed:raise ValueError('Model niewykonany/anulowany pozostaje N/A.')
        if aid not in [r['id'] for r in group.records] or a2.detection_role(self.mobile,gid,aid)!='FINAL':raise ValueError('Brak dowodu FINAL.')
        if decision not in ('TP','FP','N/A'):raise ValueError('Nieprawidłowa ocena.')
        if decision=='TP' and gt_id not in {p['object_id'] for p in self.state(eid)['objects']}:raise ValueError('Wybierz tablicę GT sceny.')
        if decision=='N/A' and not reason.strip():raise ValueError('N/A wymaga powodu.')
        def change(s):s['exceptions'][aid]=dict(mt_invocation_id=gid,attempt_id=aid,decision=decision,
            gt_id=gt_id if decision=='TP' else None,text_readable=text_readable,reason=reason,
            frame=self.index.verify_frame(self.index.calls[gid]['frame_id']),actor=self.mobile.review.reviewer_id,at=now())
        self._change(eid,'EXCEPTION',change)


def validate_layer(session,review):
    d=review.provenance.get(KEY)
    if d is None:return
    if (d.get('schema')!=SCHEMA or d.get('policy')!=POLICY or d.get('policy_sha256')!=fingerprint(POLICY)
        or d.get('session_id')!=session.session_id or d.get('source_archive_sha256')!=session.review.source_archive_sha256):
        raise ValueError('Nieprawidłowa warstwa GT scen F5A.')
    progress(d)
    validate_exposure_review(session,d)
    checker=photo.PhotoReference.__new__(photo.PhotoReference)
    for eid,r in d['reviews'].items():
        if not 0<=r['cursor']<len(r['timeline']) or any(v not in r['versions'] for v in r['timeline']):raise ValueError('Uszkodzona historia GT scen.')
        for v in r['versions'].values():
            state=v['state'];f=state['frame']
            if f and (f['full_frame_entry'] not in session.entry_names or len(f['full_frame_sha256'])!=64):raise ValueError('Obca klatka GT scen.')
            w,h=f['image_size'] if f else (1,1)
            checker.material=SimpleNamespace(images={eid:dict(width=w,height=h)})
            checker._validate_state(eid,state)
            approval=state.get('approval')
            if approval and (not f or not photo.complete(state) or approval.get('source')!='OPERATOR_EXPLICIT_SCENE'
                    or approval.get('content_and_camera_readability_confirmed') is not True or approval['content_sha256']!=fingerprint(
                    dict(presence=state['presence'],objects=state['objects'],frame=f))):raise ValueError('Nieaktualne zatwierdzenie GT sceny.')
            for aid,e in state.get('exceptions',{}).items():
                if aid not in session.attempts or e['attempt_id']!=aid or session.attempts[aid]['mt_invocation_id']!=e['mt_invocation_id']:
                    raise ValueError('Obca ocena wyjątku F5A.')
                if (not approval or e.get('decision') not in ('TP','FP','N/A')
                    or (e['decision']=='TP' and e.get('gt_id') not in {p['object_id'] for p in state['objects']})
                    or (e['decision']!='TP' and e.get('gt_id') is not None)
                    or (e['decision']=='N/A' and not str(e.get('reason','')).strip())
                    or e.get('text_readable') not in (None,True,False)):
                    raise ValueError('Nieprawidłowa decyzja wyjątku F5A.')


def progress(layer):
    """Persisted human scene approvals only; not a metric or call denominator."""
    try:
        ids=layer.get('exposure_ids',[])
        if (layer.get('schema')!=SCHEMA or not isinstance(ids,list) or not ids or len(ids)!=len(set(ids))
                or set(layer.get('reviews',{}))-set(ids)):
            raise ValueError('Nieprawidłowy mianownik GT scen.')
        count=0
        for row in layer['reviews'].values():
            state=row['versions'][row['timeline'][row['cursor']]]['state'];approval=state.get('approval')
            if (approval and approval.get('source')=='OPERATOR_EXPLICIT_SCENE' and photo.complete(state)
                and approval.get('content_sha256')==fingerprint(dict(presence=state['presence'],objects=state['objects'],frame=state['frame']))):
                count+=1
        return dict(approved=count,total=len(ids))
    except (KeyError,TypeError,IndexError) as exc:raise ValueError('Uszkodzony postęp GT scen.') from exc


def source_candidates(session,gid):
    from .mobile_source_frame_roi import source_pose_quad
    from .mobile_canvas_review import context
    ctx=context(session,gid)
    result=[]
    for row in session.mt_invocations[gid].records:
        aid=row['id'];role=a2.detection_role(session,gid,aid)
        if row.get('mt_status') not in ('VALID_QUAD','DETECTION_INVALID_QUAD'):continue
        raw=session.attempts[aid]
        try:box=[float(raw['plate_'+k]) for k in ('left','top','right','bottom')]
        except (ValueError,KeyError,TypeError):box=None
        value=raw.get('mt_confidence')
        if value is None or value=='':value=raw.get('plate_confidence')
        try:confidence=float(value)
        except (ValueError,TypeError):confidence=None
        if confidence is not None and not math.isfinite(confidence):confidence=None
        if box and any(not math.isfinite(v) for v in box):box=None
        result.append(dict(attempt_id=aid,index=int(raw['mt_detection_index']),role=role,bbox=box,
            confidence=confidence,pose=source_pose_quad(raw,ctx['size']) if ctx else None,
            raw_source_bbox=[raw.get('plate_'+k) for k in ('left','top','right','bottom')],raw_confidence=value,
            source_pose_status=raw.get('mt_pose_status'),raw_pose=raw.get('mt_pose_corners_norm')))
    return result


def _ocr(session,candidate,plate,*,outcome,readable):
    from .mobile_canvas_review import _prediction
    from .mobile_object_mz_quality import score_readout
    aid=candidate['attempt_id'];a=session.attempts[aid];samples=[session.samples[k] for k in session.sample_ids_by_attempt.get(aid,())]
    sample=samples[0] if len(samples)==1 else {};pred,status,ran=_prediction(session,aid)
    executed={'true':True,'false':False,'1':True,'0':False}.get(str(a.get('mz_executed','')).lower())
    if executed is None and status=='NOT_RUN':executed=False
    crop=a.get('evidence_entry');lineage=all(str(sample[k])==str(a.get(k,'')) for k in ('session_id','mt_invocation_id',*a2.STAMP) if sample.get(k) is not None)
    proven=bool(crop and crop in session.entry_names and a.get('evidence_kind')=='plate_crop' and lineage
        and (not sample or (sample.get('image_entry') or sample.get('evidence_entry'))==crop))
    full=plate is not None and photo.text_kind(plate)=='FULL_TEXT'
    row=score_readout(dict(executed=executed,parent_proven=candidate['role']=='FINAL',source_mt_role=candidate['role'],
        mt_outcome=outcome,operator_decision='NOT_PLATE' if outcome=='FP' else None,crop_parent_proven=proven,
        readout_unambiguous=len(samples)<=1 and status!='AMBIGUOUS_SAMPLES',gtz_confirmed=bool(full and readable),
        gtz=plate['gt_text'] if full else '',raw_prediction=pred))
    if plate and not full:row['quality_reason']='PARTIAL_OR_UNREADABLE_SCENE_TEXT'
    elif plate and not readable:row['quality_reason']='CAMERAX_READABILITY_UNPROVEN'
    row.update(attempt_id=aid,source_sample_ids=list(session.sample_ids_by_attempt.get(aid,())),
        source_mz_status=status,crop_entry=crop,raw_source_readouts=[s.get('raw_prediction',s.get('prediction','')) for s in samples])
    return row


def analyze_call(store,gid):
    from .mobile_canvas_review import context
    from .mt_eval_metrics import METRIC_PROTOCOL,_scene_result,box_iou
    from .mobile_mt_object_evaluation import _geometry
    s=store.mobile;binding=store.index.calls[gid];group=s.mt_invocations[gid];ctx=context(s,gid)
    out=dict(**binding,source_archive_sha256=s.review.source_archive_sha256,session_id=s.session_id,
        attempt_ids=[r['id'] for r in group.records],roi=list(ctx['roi']) if ctx else None,
        source_frame=store.index.verify_frame(binding['frame_id']) if binding['frame_id'] else None,
        execution_status='CANCELLED' if group.cancelled else 'FAILED' if group.execution_failed else 'EXECUTED' if group.executed else 'NOT_RUN',
        status='N/A',geometry_status='N/A',geometry_reason='NO_VERIFIED_CAMERA_GEOMETRY_TRANSFER',
        metric_tp_iou50=None,metric_fp_iou50=None,metric_fn_iou50=None,rows=[],coverage=[],
        content_reference=None,unmatched_gt_ids=[],exceptions=[])
    out['source_execution_flags']=dict(executed=group.executed,cancelled=group.cancelled,
        execution_failed=group.execution_failed,execution_errors=list(group.execution_errors))
    candidates=source_candidates(s,gid)
    out['source_candidates']=copy.deepcopy(candidates)
    eid=binding['exposure_id'];state=store.state(eid) if eid else None
    reason=binding['reason'] or ('SCENE_GT_NOT_APPROVED' if not store.approved(eid) else '')
    if out['execution_status']!='EXECUTED':reason='MT_'+out['execution_status']
    if not ctx:reason=reason or 'MT_SOURCE_CONTEXT_UNPROVEN'
    elif out['source_frame'] and list(ctx['size'])!=out['source_frame']['image_size']:reason='SOURCE_IMAGE_DIMENSIONS_MISMATCH'
    roles=s.mt_detection_roles.get(gid,{})
    if roles.get('role_proof_status')!='COMPLETE' or any(c['role']=='UNKNOWN' for c in candidates):reason=reason or 'FINAL_POPULATION_UNPROVEN'
    source_ready=not reason
    exact=bool(state and state['frame'] and out['source_frame'] and state['frame']['full_frame_sha256']==out['source_frame']['full_frame_sha256']
        and state['frame']['image_size']==out['source_frame']['image_size'])
    finals=[c for c in candidates if c['role']=='FINAL'];by_id={p['object_id']:p for p in state['objects']} if state else {}
    if state and store.approved(eid):out['content_reference']=dict(exposure_id=eid,photo_sha256=state['photo_sha256'],presence=state['presence'],plate_ids=list(by_id),n_plates=len(by_id))
    matches={};auto={};geometry={};eligible=[]
    if not reason and state['presence']=='NO_PLATES':
        auto={c['attempt_id']:dict(outcome='FP',gt_id=None) for c in finals}
        out.update(status='ASSESSED',geometry_status='NOT_REQUIRED_NEGATIVE_SCENE',geometry_reason='',metric_tp_iou50=0,metric_fp_iou50=len(finals),metric_fn_iou50=0)
    elif not reason and exact:
        l,t,r,b=ctx['roi'];reason=''
        for p in by_id.values():
            q=p.get('quad');box=_bbox(q) if q and photo.geometry_status(p)=='EXACT' else None
            coverage='N/A'
            if box:
                if box[2]<=l or box[0]>=r or box[3]<=t or box[1]>=b:coverage='OUTSIDE_ROI'
                elif l<=box[0]<box[2]<=r and t<=box[1]<box[3]<=b:coverage='COVERED';eligible.append(dict(plate_id=p['object_id'],bbox=box,quad=q,pose_eligible=False,pose_exclusion=None))
                else:coverage='PARTIAL_ROI'
            out['coverage'].append(dict(gt_id=p['object_id'],coverage=coverage))
            if coverage in ('N/A','PARTIAL_ROI'):reason='GT_GEOMETRY_NA_OR_PARTIAL_ROI'
        for c in finals:
            box=c['bbox'];w,h=ctx['size'];confidence=c['confidence']
            if (not box or any(not math.isfinite(v) for v in box) or not(0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h)
                or confidence is None or not math.isfinite(confidence) or not 0<=confidence<=1):reason='SOURCE_BBOX_OR_CONFIDENCE_UNPROVEN'
            elif sum(box_iou(box,p['bbox'])>=.5 for p in eligible)>1:reason='AMBIGUOUS_PLATE_ASSOCIATION'
        if not reason:
            protocol=dict(METRIC_PROTOCOL,operational_conf=-1,match_iou=.5,plate_class_ids=[0])
            size=out['source_frame']['image_size']
            result=_scene_result(dict(domain='F5A',width=size[0],height=size[1],predictions=[dict(index=c['index'],confidence=c['confidence'],class_id=0,bbox=c['bbox']) for c in finals]),
                dict(domain='F5A',image=dict(oriented_size=size),plates=eligible,sha256=s.review.source_archive_sha256,file=ctx['entry']),protocol)
            indexes={c['index']:c for c in finals}
            for p in result['matched']:
                c=indexes[p['prediction_index']];pid=p['plate_id'];matches[c['attempt_id']]=pid
                geometry[c['attempt_id']]=_geometry(dict(bbox=c['bbox'],bbox_source_verified=True,pose_quad=c['pose'],pose_source_verified=c['pose'] is not None),dict(quad=by_id[pid]['quad'],bbox=_bbox(by_id[pid]['quad'])),ctx['size'])
            for c in finals:auto[c['attempt_id']]=dict(outcome='TP' if c['attempt_id'] in matches else 'FP',gt_id=matches.get(c['attempt_id']))
            out.update(status='ASSESSED',geometry_status='EXACT_CAMERA_BYTES',geometry_reason='',
                metric_tp_iou50=len(matches),metric_fp_iou50=len(finals)-len(matches),metric_fn_iou50=len(eligible)-len(matches),unmatched_gt_ids=result['unmatched_gt_ids'])
    elif not reason:reason='NO_VERIFIED_CAMERA_GEOMETRY_TRANSFER'
    if not out['coverage'] and by_id:out['coverage']=[dict(gt_id=pid,coverage='N/A') for pid in by_id]
    used=set()
    for c in candidates:
        aid=c['attempt_id'];a=auto.get(aid,{});exception=state['exceptions'].get(aid) if state and store.approved(eid) else None
        # An explicit exception is operational, never a fabricated IoU result.
        override=bool(exception and out['execution_status']=='EXECUTED' and c['role']=='FINAL')
        decision=exception['decision'] if override else a.get('outcome','N/A')
        pid=exception['gt_id'] if override else a.get('gt_id')
        if decision=='TP' and pid in used:decision='DUPLICATE'
        if decision=='TP':used.add(pid)
        plate=by_id.get(pid);readable=bool(exact and plate and photo.text_kind(plate)=='FULL_TEXT')
        if override:readable=exception.get('text_readable') is True
        mz=_ocr(s,c,plate,outcome='FP' if decision=='DUPLICATE' else decision,readable=readable)
        if override and exception.get('text_readable') is False:
            mz.update(quality_reason='OPERATOR_UNREADABLE_CAMERA_TEXT',quality_note=exception['reason'])
        row=dict(attempt_id=aid,source_role=c['role'],gt_id=pid,operational_outcome=decision,
            origin='OPERATOR_EXCEPTION' if override else ('AUTO_SCENE_NO_PLATES' if state['presence']=='NO_PLATES' else 'AUTO_EXACT_CAMERA_GEOMETRY') if aid in auto else 'UNAVAILABLE',
            automatic_iou50_outcome=a.get('outcome'),geometry=geometry.get(aid),mz=mz,
            reason=exception['reason'] if override else reason or ('RAW_SUPPRESSED' if c['role']=='SUPPRESSED' else ''),
            exception_resolved=override)
        out['rows'].append(row)
        if decision=='N/A' and c['role']=='FINAL' and not override:out['exceptions'].append(aid)
    out['operational_status']='ASSESSED' if source_ready and all(r['operational_outcome']!='N/A' for r in out['rows'] if r['source_role']=='FINAL') else 'N/A'
    out['reason']=reason;out['n_mz_executed']=sum(str(s.attempts[r['id']].get('mz_executed')).lower() in ('true','1') for r in group.records)
    return out


def analyze(store):
    from .mobile_object_mz_quality import _execution_counters
    store.index.ref.verify();units=[analyze_call(store,g) for g in store.index.calls];by_gid={u['mt_invocation_id']:u for u in units};exposures=[]
    for eid,e in store.index.exposures.items():
        state=store.state(eid);main=[by_gid[g] for g in store.index.main[eid]]
        all_units=[u for u in units if u['exposure_id']==eid]
        main_rows=[r for u in main for r in u['rows'] if r['source_role']=='FINAL'];first={}
        for row in main_rows:
            if row['operational_outcome']=='TP' and row['gt_id'] not in first:first[row['gt_id']]=row
        ready=bool(store.approved(eid) and main and all(u['operational_status']=='ASSESSED' for u in main))
        pids={p['object_id'] for p in state['objects']};matched=set(first)
        outside={pid for pid in pids if main and all(any(c['gt_id']==pid and c['coverage']=='OUTSIDE_ROI' for c in u['coverage']) for u in main)}
        success=None;why=store.index.main_reasons[eid] or 'MAIN_RESULT_NOT_EVALUABLE'
        if ready:
            if state['presence']=='NO_PLATES':success=not main_rows;why='NEGATIVE_SCENE'
            elif pids-matched or len(main_rows)>len(pids) or any(r['operational_outcome'] in ('FP','DUPLICATE') for r in main_rows):success=False;why='ROI_LOSS_OR_MT_MISS_OR_EXTRA_FINAL'
            elif all(first[pid]['mz']['canonical_ocr_eligible'] for pid in pids):success=all(first[pid]['mz']['exact_match'] for pid in pids);why='FIRST_SOURCE_READOUT_PER_GT'
            else:why='OCR_NOT_EVALUABLE'
        exposures.append(dict(exposure_id=eid,photo_sha256=e['photo_sha256'],approved=store.approved(eid),
            representative_frame=state['frame'],main_call_ids=store.index.main[eid],main_input_frame=main[0]['source_frame'] if main else None,
            main_result_basis='SCENE_CONTENT_AND_AUTO_OR_EXPLICIT_OPERATIONAL_OUTCOMES_NOT_IOU_RATE',
            main_alpr_success=success,main_reason=why,main_outside_all_roi_gt_ids=sorted(outside),
            representative_roi_coverage=representative_roi_coverage(store,eid),
            main_first_readout_by_gt={pid:r['attempt_id'] for pid,r in first.items()},
            any_detected_at_least_once=True if any(r['operational_outcome']=='TP' for u in all_units for r in u['rows']) else False if all_units and all(u['operational_status']=='ASSESSED' for u in all_units) else None))
    fkeys={store.index.ref.frame_key(fid) for fid in store.index.frames}
    known=[e for e in exposures if e['main_alpr_success'] is not None]
    counters=_execution_counters(store.mobile)
    events=[e for e in getattr(store.index.ref,'data',{}).get('scene_events',[]) if e['session_id']==store.mobile.session_id]
    return dict(schema='alpr.mobile.f5a_scene_results.v1',policy=POLICY,policy_sha256=fingerprint(POLICY),
        gt_mode='CAMERAX_GT_WITH_A1_CONTENT_REFERENCE',
        binding=store.index.ref.binding,session_id=store.mobile.session_id,source_archive_sha256=store.mobile.review.source_archive_sha256,
        review_revision=store.mobile.review.review_revision,exposures=exposures,units=units,
        source_frames=copy.deepcopy(list(store.index.frames.values())),source_scene_events=copy.deepcopy(events),
        unbound_attempt_records=copy.deepcopy(store.index.diagnostics),
        denominators=dict(exposures=len(exposures),unique_photographs=len({e['photo_sha256'] for e in exposures}),
            camera_frames=len(fkeys),frame_representations=len(store.index.frames),scene_events=len(events),mt_calls=len(units),
            desktop_record_groups=len(store.mobile.mt_invocations),unbound_attempt_records=len(store.index.diagnostics),
            mt_marked_executed=sum(u['source_execution_flags']['executed'] for u in units),
            mt_cancelled=sum(u['source_execution_flags']['cancelled'] for u in units),
            mt_failed=sum(u['source_execution_flags']['execution_failed'] for u in units),
            mz_executed=counters['n_mz_calls'],mz_count_reason=counters['reason'],
            mz_executed_attempt_records=sum(u['n_mz_executed'] for u in units),approved_scenes=sum(e['approved'] for e in exposures),
            evaluable_mt_calls=sum(u['operational_status']=='ASSESSED' for u in units),na_mt_calls=sum(u['operational_status']!='ASSESSED' for u in units),
            evaluable_iou50_mt_calls=sum(u['status']=='ASSESSED' for u in units),
            evaluable_ocr=sum(r['mz']['canonical_ocr_eligible'] for u in units for r in u['rows']),
            evaluable_main_scenes=len(known),na_main_scenes=len(exposures)-len(known)),
        main_alpr_success_rate=sum(e['main_alpr_success'] for e in known)/len(known) if known else None,
        source_execution_counters=counters,
        outside_gt_calls=[u['mt_invocation_id'] for u in units if u['exposure_id'] is None])


def export(session,directory,*,store=None):
    if KEY not in session.review.provenance:return []
    from .mobile_human_review import _atomic_write
    if store is None:
        binding=session.review.provenance[KEY]['binding'];ref=a2.Reference(binding['mapping_path'],binding['gt_path'],expected=binding)
        store=SceneReference(session,ref)
    payload=analyze(store);payload['gt_layer']=copy.deepcopy(store.data)
    directory=Path(directory);path=directory/'f5a_scene_gt.json'
    _atomic_write(path,json.dumps(payload,ensure_ascii=False,indent=2));paths=[path]
    for name,rows in [('f5a_scenes.csv',payload['exposures']),('f5a_calls.csv',payload['units'])]:
        stream=io.StringIO(newline='');fields=list(dict.fromkeys(k for r in rows for k in r))
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
        out=directory/name;_atomic_write(out,'\ufeff'+stream.getvalue());paths.append(out)
    if EXPOSURE_REVIEW in store.data:
        result=operator_results(store)
        out=directory/'f5a_exposure_reviews.json';_atomic_write(out,json.dumps(result,ensure_ascii=False,indent=2));paths.append(out)
        for name,rows in [('f5a_operator_decisions.csv',[{**(r['operator_decision'] or {}),'exposure_id':r['exposure_id'],'status':r['status']} for r in result['exposures']]),
                ('f5a_recorded_outputs.csv',[dict(exposure_id=r['exposure_id'],**v) for r in result['exposures'] for v in r['automatic']['A']]),
                ('f5a_source_timings.csv',result['source_timing_rows'])]:
            stream=io.StringIO(newline='');fields=list(dict.fromkeys(k for r in rows for k in r))
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
            out=directory/name;_atomic_write(out,'\ufeff'+stream.getvalue());paths.append(out)
    manifest=directory/'f5a_scene_export_manifest.json'
    _atomic_write(manifest,json.dumps(dict(schema='alpr.mobile.f5a_scene_export_manifest.v1',
        session_id=session.session_id,source_archive_sha256=session.review.source_archive_sha256,
        review_revision=session.review.review_revision,binding=store.index.ref.binding,policy_sha256=fingerprint(POLICY),
        files=[dict(path=p.name,sha256=a2.sha(p),bytes=p.stat().st_size) for p in paths]),ensure_ascii=False,indent=2))
    paths.append(manifest)
    return paths


def timeline_records(index):
    """Read-only navigation projection. IDs join records; clocks never join them.

    MZ items are archived readouts of attempts, not invented inference calls.
    A CameraX bitmap can have multiple A2 representations and multiple MTs.
    """
    s=index.session;ref=index.ref;frames={};aliases={};keys={}
    for fid,f in index.frames.items():
        key=ref.frame_key(fid);canonical=keys.setdefault(key,fid);aliases[fid]=canonical
        if canonical not in frames:
            frames[canonical]=dict(f,frame_id=canonical,aliases=[],exposure_id=ref.frame_exposure.get(fid),calls=[])
        frames[canonical]['aliases'].append(fid)
    events={e['scene_event_id']:e for e in getattr(ref,'data',{}).get('scene_events',[]) if e['session_id']==s.session_id}
    calls={};mz={};traces={}
    for i,row in enumerate(getattr(s.bundle,'trace_rows',())):
        if row.get('frame_id') not in (None,''):traces.setdefault(str(row['frame_id']),[]).append(i)
    for gid,binding in index.calls.items():
        fid=aliases.get(binding['frame_id']);roles=s.mt_detection_roles.get(gid,{})
        if not fid:
            # A genuine input outside the A1 photograph set is still evidence.
            try:
                source=ref.call_frame(s,gid);index.verify_frame(source);fid=aliases[source]
            except (ValueError,KeyError,OSError):pass
        row=dict(binding,frame_id=fid,mz=[],trace_ids=[],source_candidates=source_candidates(s,gid),
            stale=s.mt_invocations[gid].cancelled,failed=s.mt_invocations[gid].execution_failed,mz_not_run=bool(s.mt_invocations[gid].records) and all(
                str(s.attempts[r['id']].get('mz_executed','')).lower() in ('false','0') or
                s.attempts[r['id']].get('mz_status')=='NOT_RUN' for r in s.mt_invocations[gid].records))
        # A trace ID must be explicitly attached to this call's source stamp.
        frame=frames.get(fid,{})
        if roles.get('trace_frame_id') not in (None,'') and frame and all(
                str(roles.get('source_stamp',{}).get(k))==str(frame['source_stamp'].get(k)) for k in a2.STAMP):
            row['trace_ids'].append(str(roles['trace_frame_id']))
        for record in s.mt_invocations[gid].records:
            aid=record['id'];raw=s.attempts[aid]
            samples=[s.samples[k] for k in s.sample_ids_by_attempt.get(aid,()) if k in s.samples]
            # Every attempt can retain a consensus; only explicit execution adds MZ.
            executed=str(raw.get('mz_executed','')).lower() in ('true','1')
            linked=[r for r in samples if str(r.get('attempt_id',''))==aid and
                    r.get('mt_invocation_id',gid)==gid and all(
                        str(r[k])==str(raw.get(k)) for k in a2.STAMP if r.get(k) is not None)]
            tids=list(dict.fromkeys(str(r['timing']['frame_id']) for r in linked
                if (r.get('timing') or {}).get('frame_id') not in (None,'')))
            row['trace_ids']=list(dict.fromkeys(row['trace_ids']+tids))
            readouts=[str(r.get('raw_prediction',r.get('prediction','')) or '') for r in linked]
            if not readouts:readouts=[str(raw.get('raw_prediction',raw.get('prediction','')) or '')]
            consensus=[str(r.get('consensus_prediction') or '') for r in linked if 'consensus_prediction' in r]
            if not consensus and 'consensus_prediction' in raw:consensus=[str(raw['consensus_prediction'] or '')]
            flags={source_bool(r.get('track_confirmed')) for r in linked};flags.discard(None)
            item=dict(attempt_id=aid,gid=gid,frame_id=fid,exposure_id=binding['exposure_id'],
                readouts=readouts,sample_ids=[r['id'] for r in linked],trace_ids=tids or row['trace_ids'][:],
                raw_field_present=all('raw_prediction' in r or 'prediction' in r for r in linked) if linked else ('raw_prediction' in raw or 'prediction' in raw),
                consensus=consensus,confirmed=next(iter(flags)) if len(flags)==1 else None,
                executed=executed,entity_id=raw.get('entity_id'),plate_track_id=raw.get('plate_track_id'),
                status=raw.get('mz_status',''),stale=s.cancelled(raw),cancel_reason=raw.get('cancel_reason',''))
            # Kept separately from the MZ invocation population.
            row.setdefault('readout_ids',[]).append(aid)
            if executed:mz[aid]=item;row['mz'].append(aid)
            row.setdefault('readouts',{})[aid]=item
        calls[gid]=row
        if fid:frames[fid]['calls'].append(gid)
    from .mobile_package_experiments import iter_full_event_rows
    changes=[r for r in iter_full_event_rows(s.bundle) if r.get('event_type') in ('consensus_updated','consensus_confirmed','mz_result')]
    reads={a:r for c in calls.values() for a,r in c.get('readouts',{}).items()}
    for item in reads.values():
        # Event clock is displayed separately. Only explicit frame/track IDs join it.
        tracks={str(s.samples[k].get('track_id')) for k in item['sample_ids'] if s.samples[k].get('track_id') is not None}
        item['events']=[r for r in changes if str(r.get('frame_id')) in item['trace_ids'] and str(r.get('track_id')) in tracks]
    return dict(frames=frames,aliases=aliases,calls=calls,mz=mz,events=events,traces=traces,
        readouts=reads,consensus_events=changes)


def source_bool(value):
    return {'true':True,'1':True,'false':False,'0':False}.get(str(value).lower())


def exposure_review_state(layer):
    return layer.get(EXPOSURE_REVIEW) or dict(schema=EXPOSURE_SCHEMA,protocol=REVIEW_PROTOCOL,
        protocol_sha256=fingerprint(REVIEW_PROTOCOL),approval=None,reviews={},last_context=None,events=[])


def exposure_decision(layer,eid):
    r=exposure_review_state(layer)['reviews'].get(eid)
    return copy.deepcopy(r['versions'][r['timeline'][r['cursor']]]['state']) if r else None


def exposure_progress(layer):
    ids=layer['exposure_ids'];value=exposure_review_state(layer)
    if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)):raise ValueError('Nieprawidłowy mianownik oceny zdjęć.')
    if value['reviews'] and (not value['approval'] or value['approval'].get('origin')!='OPERATOR_EXPLICIT'):
        raise ValueError('Oceny bez jawnego zatwierdzenia protokołu.')
    states=[exposure_decision(layer,k) for k in ids]
    if any(s and s.get('origin')!='OPERATOR_EXPLICIT' for s in states):raise ValueError('Nieprawidłowe pochodzenie oceny.')
    counts=Counter((s or {}).get('decision','UNREVIEWED') for s in states)
    if set(counts)-{'UNREVIEWED','CONFIRMED','CORRECTED','ERROR_REPORTED','NOT_ASSESSABLE'}:raise ValueError('Nieznana decyzja zdjęcia.')
    return dict(total=len(layer['exposure_ids']),reviewed=sum(v for k,v in counts.items() if k!='UNREVIEWED'),
        confirmed=counts['CONFIRMED'],corrected=counts['CORRECTED'],errors=counts['ERROR_REPORTED'],
        na=counts['NOT_ASSESSABLE'],unreviewed=counts['UNREVIEWED'])


def validate_exposure_review(session,layer):
    value=layer.get(EXPOSURE_REVIEW)
    if value is None:return
    if (value.get('schema')!=EXPOSURE_SCHEMA or value.get('protocol')!=REVIEW_PROTOCOL or
            value.get('protocol_sha256')!=fingerprint(REVIEW_PROTOCOL)):
        raise ValueError('Nieobsługiwany protokół oceny ekspozycji.')
    approval=value.get('approval')
    if approval and (approval.get('origin')!='OPERATOR_EXPLICIT' or approval.get('protocol_sha256')!=fingerprint(REVIEW_PROTOCOL)
                     or approval.get('binding')!=layer['binding']):raise ValueError('Brak jawnego zatwierdzenia protokołu.')
    for eid,r in value['reviews'].items():
        if eid not in layer['exposure_ids'] or not approval:raise ValueError('Obca ekspozycja lub niezatwierdzony protokół.')
        if not r['timeline'] or not 0<=r['cursor']<len(r['timeline']) or any(v not in r['versions'] for v in r['timeline']):
            raise ValueError('Uszkodzona historia oceny ekspozycji.')
        for v in r['versions'].values():
            s=v['state']
            if s is None:continue
            if (s.get('origin')!='OPERATOR_EXPLICIT' or s.get('session_id')!=session.session_id or s.get('exposure_id')!=eid
                    or s.get('binding')!=layer['binding'] or s.get('protocol_sha256')!=fingerprint(REVIEW_PROTOCOL)
                    or s.get('source_archive_sha256')!=session.review.source_archive_sha256
                    or s.get('decision') not in ('CONFIRMED','CORRECTED','ERROR_REPORTED','NOT_ASSESSABLE')):
                raise ValueError('Nieprawidłowa decyzja ekspozycji.')
            if s['decision']!='CONFIRMED' and not s.get('reason','').strip():raise ValueError('Decyzja wymaga powodu.')
            if len(s.get('photo_sha256',''))!=64 or len(s.get('a2_photo_sha256',''))!=64:raise ValueError('Brak tożsamości zdjęcia.')
            correction=s.get('correction',{})
            if (type(correction.get('diagnostic_b_override')) is not bool or
                    (correction['diagnostic_b_override'] and type(correction.get('diagnostic_b')) is not bool) or
                    s.get('protocol_id')!=REVIEW_PROTOCOL['id'] or type(s.get('revision')) is not int):
                raise ValueError('Nieprawidłowa korekta lub wersja oceny.')
            for ref in s.get('viewed_evidence',[]):
                kind,key,*rest=ref
                if kind in ('call','attempt','mz','consensus') and key not in session.mt_invocations:raise ValueError('Obce MT w dowodach.')
                if rest and rest[0] not in session.attempts:raise ValueError('Obce MZ w dowodach.')


def text_multiset(gt,values,*,reason=''):
    """Text accounting only. It never assigns predictions to physical plates."""
    normalize=normalize_registration
    expected=Counter(normalize(photo.display_text(p)) for p in gt['objects'] if photo.text_kind(p)=='FULL_TEXT')
    actual=Counter(normalize(v) for v in values if normalize(v))
    unknown=sum(photo.text_kind(p)!='FULL_TEXT' for p in gt['objects'])
    if gt['presence']=='UNRESOLVED':reason=reason or 'GT_PRESENCE_UNRESOLVED'
    if unknown:reason=reason or 'PARTIAL_OR_UNREADABLE_GT_TEXT'
    return dict(expected=dict(expected),actual=dict(actual),unknown_gt_text=unknown,
        known_recovered=dict(expected & actual),known_missing=dict(expected-actual),
        unaccounted=dict(actual-expected),duplicate_readouts={k:v for k,v in actual.items() if v>1},
        exact=None if reason else expected==actual,reason=reason,
        physical_plate_association='N/A_NO_OBJECT_CORRESPONDENCE')


def frozen_first_input(reference):
    """Read the already published C comparison; never recompute or overwrite it."""
    base=Path(reference.binding.get('mapping_path','')).parent.parent
    matches=[]
    for manifest in base.glob('F5A_WYNIKI_A1_A2_*/manifest_wynikow.json'):
        try:
            m=json.loads(manifest.read_text(encoding='utf-8'))
            if m.get('binding')!=reference.binding:continue
            row=next(r for r in m['files'] if r['path']=='wyniki.json');path=manifest.parent/row['path']
            if a2.sha(path)!=row['sha256']:continue
            payload=json.loads(path.read_text(encoding='utf-8'))
            if payload.get('binding')!=reference.binding:continue
            matches.append(dict(path=str(path),sha256=row['sha256'],manifest_sha256=a2.sha(manifest),
                protocol=payload['protocol'],exposures={r['exposure_id']:r for r in payload['exposures']}))
        except (OSError,ValueError,KeyError,StopIteration):continue
    return matches[0] if len(matches)==1 else dict(reason='FROZEN_FIRST_INPUT_NOT_UNIQUELY_VERIFIED',exposures={})


def exposure_accounting(store,eid,*,evidence=None,photo_sha=None):
    """A/B proposals over every recorded input, not a selected best/terminal output."""
    ev=evidence or timeline_records(store.index);s=store.mobile;ref=store.index.ref
    sha=photo_sha or store.index.exposures[eid]['photo_sha256'];gt=ref.gt_states['img-sha256-'+sha]
    groups={}
    for gid,c in ev['calls'].items():
        if c['exposure_id']!=eid:continue
        # Exact A2 frame identity only; two ROI calls on this same input stay together.
        groups.setdefault(c['frame_id'] or gid,[]).append(gid)
    sequence=[];confirmed=[];unknown_confirmation=False
    for fid,gids in groups.items():
        raw=[];consensus=[];raw_reason='';cons_reason='';items=[];failures=[]
        for gid in gids:
            call=ev['calls'][gid];g=s.mt_invocations[gid]
            if not g.executed or g.cancelled or g.execution_failed:
                failures.append(dict(mt_invocation_id=gid,reason='CANCELLED' if g.cancelled else 'FAILED' if g.execution_failed else 'NOT_EXECUTED'))
                raw_reason=cons_reason='EXECUTION_FAILURE_NO_PUBLISHED_OUTPUT';continue
            if not call['frame_id']:raw_reason=cons_reason='SOURCE_FRAME_UNPROVEN'
            candidates=call['source_candidates']
            if any(c['role']=='UNKNOWN' for c in candidates):raw_reason=cons_reason='FINAL_ROLE_UNPROVEN'
            for c in candidates:
                if c['role']!='FINAL':continue
                r=call['readouts'].get(c['attempt_id'])
                if not r:raw_reason=cons_reason='READOUT_EVIDENCE_MISSING';continue
                items.append(r)
                if r['stale']:
                    raw_reason=cons_reason='CANCELLED_OUTPUT';continue
                if not r['raw_field_present']:raw_reason='RAW_FIELD_MISSING'
                elif len(r['readouts'])!=1:raw_reason='RAW_READOUT_AMBIGUOUS'
                elif r['executed']:raw.extend(r['readouts'])
                # A skipped MZ is a known lack of fresh output, not fabricated text.
                elif source_bool(s.attempts[r['attempt_id']].get('mz_executed')) is None:raw_reason='MZ_EXECUTION_UNPROVEN'
                if len(r['consensus'])==1:consensus.extend(r['consensus'])
                elif not r['consensus']:cons_reason='CONSENSUS_FIELD_MISSING'
                else:cons_reason='CONSENSUS_AMBIGUOUS'
                if r['confirmed'] is True and len(r['consensus'])==1:
                    confirmed.append(dict(attempt_id=r['attempt_id'],value=r['consensus'][0]))
                elif r['confirmed'] is None:unknown_confirmation=True
        sequence.append(dict(frame_id=fid if fid in ev['frames'] else None,mt_invocation_ids=gids,
            raw=text_multiset(gt,raw,reason=raw_reason),consensus=text_multiset(gt,consensus,reason=cons_reason),
            readout_ids=[r['attempt_id'] for r in items],execution_failures=failures,
            system_failure=bool(failures),origin='PROGRAM_PROPOSAL',terminal_android_result=False))
    # No call: require explicit, same-frame evidence of zero selected ROI.
    zero=[]
    if not groups:
        stamps=[f['source_stamp'] for f in ev['frames'].values() if f['exposure_id']==eid]
        zero=[dict(source_stamp=m['source_stamp'],reason=m.get('reason') or m.get('zero_roi_reason') or 'RECORDED_ZERO_SELECTED_ROIS')
            for f in s.scene_frames.values() for m in f['mp_observations']
            if m['source_stamp'] in stamps and m['execution_status']=='EXECUTED' and m['selected_rois']==[]]
    full=text_multiset(gt,[])['expected'];normalize=normalize_registration
    matched=[r for r in confirmed if normalize(r['value']) in full]
    if not full:b=None;why='NO_FULL_GT_NUMBER'
    elif matched:b=True;why='ARCHIVED_CONFIRMED_EXACT_TEXT_NOT_PHYSICAL_MATCH'
    elif unknown_confirmation:b=None;why='CONFIRMATION_EVIDENCE_MISSING'
    elif groups or zero:b=False;why='NO_CONFIRMED_EXACT_READOUT'
    else:b=None;why='NO_MT_EXECUTION_EVIDENCE'
    return dict(exposure_id=eid,photo_sha256=sha,origin='PROGRAM_PROPOSAL',A=sequence,
        B=dict(value=b,reason=why,matching_source_records=matched,diagnostic_only=True),
        no_mt_reason='RECORDED_ZERO_SELECTED_ROIS' if zero else 'NO_MT_EXECUTION_EVIDENCE' if not groups else '',
        no_mt_evidence=zero,no_mt_system_exact=(gt['presence']=='NO_PLATES' if zero and gt['presence']!='UNRESOLVED' else None),
        counts=dict(mt=len(gids_all:=[g for g,c in ev['calls'].items() if c['exposure_id']==eid]),
            mz=sum(r['executed'] for g in gids_all for r in ev['calls'][g]['readouts'].values())),
        iou=None,nmce=None,conditional_ocr=None,geometry_reason='NO_OBJECT_CORRESPONDENCE_OR_CAMERA_GT')


def operator_results(store,*,evidence=None):
    store.index.ref.verify();store.mobile.verify_source()
    ev=evidence or timeline_records(store.index);frozen=frozen_first_input(store.index.ref);rows=[]
    for eid in store.material.order:
        decision=store.operator_decision(eid);automatic=exposure_accounting(store,eid,evidence=ev);corrected=None
        if decision and decision['decision']=='CORRECTED':
            corrected=exposure_accounting(store,eid,evidence=ev,photo_sha=decision['photo_sha256'])
            corrected['origin']='OPERATOR_EXPLICIT_CORRECTION'
            if decision['correction']['diagnostic_b_override']:
                corrected['B']=dict(value=decision['correction']['diagnostic_b'],reason=decision['reason'],diagnostic_only=True,origin='OPERATOR_EXPLICIT')
        rows.append(dict(exposure_id=eid,automatic=automatic,operator_decision=decision,corrected=corrected,
            C=copy.deepcopy(frozen['exposures'].get(eid)),C_na_reason='' if eid in frozen['exposures'] else frozen['reason'],
            status=decision['decision'] if decision else 'UNREVIEWED'))
    accepted=[r['corrected'] or r['automatic'] for r in rows if r['status'] in ('CONFIRMED','CORRECTED')]
    values=[r['B']['value'] for r in accepted];known=[v for v in values if v is not None]
    summary=dict(scope='THIS_SESSION_ONLY_NOT_360_EXPOSURES',accepted_accounting_exposures=len(accepted),
        B=dict(success=sum(known),denominator=len(known),na=len(values)-len(known),rate=sum(known)/len(known) if known else None),
        A={k:dict(exact=sum(v[k]['exact'] is True for r in accepted for v in r['A']),
            denominator=sum(v[k]['exact'] is not None for r in accepted for v in r['A']),
            na=sum(v[k]['exact'] is None for r in accepted for v in r['A']),unit='RECORDED_INPUT_NOT_EXPOSURE') for k in ('raw','consensus')},
        reported_errors=sum(r['status']=='ERROR_REPORTED' for r in rows),
        operator_na=sum(r['status']=='NOT_ASSESSABLE' for r in rows),
        unreviewed=sum(r['status']=='UNREVIEWED' for r in rows))
    return dict(schema=EXPOSURE_SCHEMA,protocol=REVIEW_PROTOCOL,protocol_approval=store.protocol_approval(),
        session_id=store.mobile.session_id,source_archive_sha256=store.mobile.review.source_archive_sha256,
        session_metadata=copy.deepcopy(store.index.ref.sessions[store.mobile.session_id]),
        binding=store.index.ref.binding,progress=exposure_progress(store.data),exposures=rows,verified_summary=summary,
        frozen_first_input_source={k:v for k,v in frozen.items() if k!='exposures'},
        source_evidence=ev,source_timing_rows=copy.deepcopy(store.mobile.bundle.trace_rows),
        timing_note='Original trace clock and durations; joined by actual frame ID, not by capture/completion time.',
        review_revision=store.mobile.review.review_revision,history=copy.deepcopy(exposure_review_state(store.data)))
