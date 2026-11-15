"""A3: source-bound manual decisions in the EXISTING mobile review sidecar.

A2 is a proposal, the frozen photograph GT is read-only. No JPEG->CameraX
geometry, inference, source prediction edit, or legacy scorer override.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy, csv, hashlib, io, json, uuid

KEY='f5a_assisted_gt_v1'
SCHEMA='alpr.mobile.f5a_assisted_gt.v1'
SERIES='F5A-ROI-A12-01'
STAMP=('scene_generation','visual_epoch','camera_transform_generation','source_sequence','source_timestamp_nanos','source_timestamp_domain')
BUCKETS=('identities','coverage','detections','missing','units','blocks')
COVERAGE=('COVERED','OUTSIDE_ROI','PARTIAL_ROI','ROI_COVERAGE_UNPROVEN')

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class Reference:
    def __init__(self, mapping, gt=None, *, expected=None):
        self.path=Path(mapping).resolve();self.root=self.path.parent
        manifest_path=self.root/'artifacts_manifest.json';manifest=read(manifest_path)
        entries={r['path']:r['sha256'] for r in manifest['files']}
        self.data=read(self.path)
        if self.data.get('schema')!='alpr.f5a.a2.mapping.v1':raise ValueError('Nieobsługiwane mapowanie A2.')
        gt_sha=self.data['summary']['gt_sha256']
        if manifest.get('gt_sha256')!=gt_sha:raise ValueError('Niespójne SHA referencji w manifeście A2.')
        self.gt_path=Path(gt).resolve() if gt else self.root/('reference_frozen_sha256-'+gt_sha+'.json')
        self.pins={str(manifest_path):sha(manifest_path),str(self.path):entries[self.path.name],str(self.gt_path):gt_sha}
        for name in ('inventory.json','source_integrity_before.json'):
            self.pins[str(self.root/name)]=entries[name]
        for path,digest in self.pins.items():
            if sha(path)!=digest:raise ValueError('Niezgodne SHA źródła A2/GT: '+Path(path).name)
        self.gt=read(self.gt_path);inventory=read(self.root/'inventory.json')
        self.photo_root=next(Path(r['path']).parent for r in read(self.root/'source_integrity_before.json') if Path(r['path']).name=='sorting_manifest.json')
        self.photos={p['sha256']:p for p in inventory['photos']}
        for digest,p in self.photos.items():
            path=(self.photo_root/p['sciezka_zrodla']).resolve()
            if not path.is_relative_to(self.photo_root.resolve()) or sha(path)!=digest:raise ValueError('Nieprawidłowe źródło fotografii GT.')
            self.pins[str(path)]=digest
        for p in read(self.root/'source_integrity_before.json'):
            if Path(p['path']).name in ('sorting_manifest.json','manifest_blokow.json'):
                if sha(p['path'])!=p['sha256']:raise ValueError('Manifest materiału zmienił się.')
                self.pins[p['path']]=p['sha256']
        from .photo_reference import complete
        self.gt_states={h:r['versions'][r['timeline'][r['cursor']]]['state'] for h,r in self.gt['reviews'].items()}
        if len(self.photos)!=40 or any(not complete(self.gt_states['img-sha256-'+h]) for h in self.photos):
            raise ValueError('Referencja nie zawiera 40 kompletnych fotografii.')
        self.sessions={s['session_id']:s for s in self.data['sessions']}
        self.exposures={e['exposure_id']:e for e in self.data['exposures']}
        self.frames={f['frame_id']:f for f in self.data['frames']}
        self.calls={(m['session_id'],m['mt_invocation_id']):m for m in self.data['mt_observations']}
        grid={(s['variant'],s['replicate'],s['block']) for s in self.sessions.values()}
        if len(self.sessions)!=36 or grid!={(v,r,'BLOK_'+b) for v in ('R0','R1','R2') for r in (1,2,3) for b in 'ABCD'} or len(self.exposures)!=360:
            raise ValueError('Niepełna albo powtórzona kohorta 36 sesji A2.')
        self.frame_exposure={}
        for e in self.exposures.values():
            for fid in e['evidence_frame_ids']:
                f=self.frames[fid]
                if f['session_id']!=e['session_id'] or f['photo_sha256']!=e['photo_sha256']:raise ValueError('Niespójne powiązania klatek A2.')
                if fid in self.frame_exposure:raise ValueError('Powielona klatka w ekspozycjach A2.')
                self.frame_exposure[fid]=e['exposure_id']
        self.binding=dict(schema=SCHEMA,series_id=SERIES,mapping_path=str(self.path),mapping_sha256=self.pins[str(self.path)],
            gt_path=str(self.gt_path),gt_sha256=gt_sha,artifacts_manifest_sha256=self.pins[str(manifest_path)])
        if expected and self.binding!=expected:raise ValueError('Zmieniono przypiętą referencję A2/GT. Ocen nie przeniesiono.')
        self._stamps={p:self._stamp(p) for p in self.pins}

    @staticmethod
    def _stamp(p):
        s=Path(p).stat();return s.st_size,s.st_mtime_ns

    def verify(self):
        for p,digest in self.pins.items():
            current=self._stamp(p)
            if current!=self._stamps[p]:
                if sha(p)!=digest:raise ValueError('Źródło A2/GT zmieniło się; zapis zablokowany.')
                self._stamps[p]=current

    def attach(self, session):
        self.verify();s=self.sessions.get(session.session_id)
        if not s or s['archive_sha256']!=session.review.source_archive_sha256:raise ValueError('Sesja lub SHA archiwum nie odpowiada A2.')
        old=session.review.provenance.get(KEY)
        if old and old['binding']!=self.binding:raise ValueError('Sidecar zawiera oceny innej wersji A2/GT.')
        session._f5a_reference=self
        validate_layer(session,session.review)
        return self

    def plates(self, photo):return copy.deepcopy(self.gt_states['img-sha256-'+photo]['objects']) if photo else []
    def frame_key(self, fid):
        f=self.frames[fid]
        return 'frame:'+fingerprint([f['session_id'],f['full_frame_sha256'],f['source_stamp']])
    def call_frame(self, session, gid):
        m=self.calls.get((session.session_id,gid))
        if not m or not m['frame_ids']:raise ValueError('Brak dowodu klatki wywołania w A2.')
        ids=m['frame_ids'];f=self.frames[ids[0]]
        if any(self.frame_key(fid)!=self.frame_key(ids[0]) for fid in ids):raise ValueError('Wywołanie ma różne klatki źródłowe.')
        group=session.mt_invocations[gid]
        if any(any(str(row.get(k))!=str(f['source_stamp'].get(k)) for k in STAMP) or row.get('source_frame_entry')!=f['full_frame_entry'] for row in group.records):
            raise ValueError('Stamp albo obraz MT nie odpowiada A2.')
        return ids[0]

def discover(source):
    for parent in Path(source).resolve().parents:
        if (parent/'01_materialy/04_GT_FOTOGRAFII_Z4').is_dir():
            return sorted((parent/'03_wyniki').glob('*/mapping.json'))
    return []

def registry_path():
    from ..config import CONFIG
    return CONFIG.DIR_7_RANKINGS_MOBILE_PACKAGES/'f5a_reference_bindings.json'

def register(reference, path=None):
    """Only source locations/hashes; never operator decisions or GT."""
    from .mobile_human_review import _atomic_write
    path=Path(path or registry_path());data=read(path) if path.exists() else dict(schema='alpr.mobile.reference_bindings.v1',series={})
    data['series'][SERIES]=reference.binding;path.parent.mkdir(parents=True,exist_ok=True)
    _atomic_write(path,json.dumps(data,ensure_ascii=False,indent=2))

def registered(path=None):
    path=Path(path or registry_path())
    if not path.exists():return None
    binding=read(path).get('series',{}).get(SERIES)
    return Reference(binding['mapping_path'],binding['gt_path'],expected=binding) if binding else None

def reference(session):
    ref=getattr(session,'_f5a_reference',None)
    if ref:return ref
    layer=session.review.provenance.get(KEY)
    if not layer:raise ValueError('Najpierw połącz mapowanie A2 i GT dla serii.')
    b=layer['binding'];return Reference(b['mapping_path'],b['gt_path'],expected=b).attach(session)

def state(session):
    layer=session.review.provenance.get(KEY,{})
    return copy.deepcopy(layer.get('state') or {b:{} for b in BUCKETS})

def identity(session,fid,st=None):
    ref=reference(session);st=st or state(session);f=ref.frames[fid]
    row=st['identities'].get(ref.frame_key(fid)) or st['identities'].get(ref.frame_exposure.get(fid))
    if row:return dict(row,approved=row['decision'] in ('CONFIRMED','CORRECTED'))
    return dict(decision='PROPOSED',photo_sha256=f.get('photo_sha256'),approved=False,decision_id=None,origin='A2_AUTOMATIC_EVIDENCE')

def _audit(session,**values):
    return dict(values,decision_id=uuid.uuid4().hex,origin='OPERATOR_EXPLICIT',reviewer_id=session.review.reviewer_id,
        at=datetime.now(timezone.utc).isoformat())

def _change(session,action,mutate,*,resume=None):
    ref=reference(session);ref.verify();session.verify_source()
    before=state(session);after=copy.deepcopy(before);mutate(after)
    if action!='APPROVE_A3_SESSION':after['blocks'].pop('approval:'+session.session_id,None)
    changes=[dict(bucket=b,key=k,before=before[b].get(k),after=after[b].get(k)) for b in BUCKETS
        for k in set(before[b])|set(after[b]) if before[b].get(k)!=after[b].get(k)]
    if not changes:return
    changed=copy.deepcopy(session.review)
    changed.review_status='IN_PROGRESS'
    changed.review_mode='assisted'
    changed.review_mode_origin='F5A_A3_CANONICAL_GT_ASSISTANCE'
    layer=changed.provenance.setdefault(KEY,dict(schema=SCHEMA,binding=ref.binding,state=before,versions={},timeline=[],cursor=-1))
    layer.setdefault('legacy_review_header',dict(review_mode=session.review.review_mode,review_mode_origin=session.review.review_mode_origin,
        review_status=session.review.review_status,review_revision=session.review.review_revision))
    vid=uuid.uuid4().hex;layer['versions'][vid]=dict(action=action,changes=changes)
    layer['timeline']=layer['timeline'][:layer['cursor']+1]+[vid];layer['cursor']+=1;layer['state']=after
    if resume:layer['resume']=resume
    # This explicitly assisted layer never unlocks or changes legacy blind GT.
    layer['review_mode']='ASSISTED_CANONICAL_PHOTO_GT';layer['geometric_metrics']='N/A_NO_CAMERA_GT_TRANSFORM'
    session._save_change(changed,allow_assisted_reveal=True)

def decide_identity(session,scope,decision,*,photo=None,reason=''):
    ref=reference(session)
    if scope in ref.exposures:
        e=ref.exposures[scope]
        if e['session_id']!=session.session_id:raise ValueError('Inna sesja ekspozycji.')
        key=scope;proposal=e['photo_sha256'];ids=e['evidence_frame_ids']
    else:
        f=ref.frames[scope]
        if f['session_id']!=session.session_id:raise ValueError('Inna sesja klatki.')
        key=ref.frame_key(scope);proposal=f.get('photo_sha256');ids=[scope]
    if decision not in ('CONFIRMED','CORRECTED','REJECTED','N/A'):raise ValueError('Nieprawidłowa decyzja tożsamości.')
    if decision=='CONFIRMED':photo=proposal
    if decision in ('CONFIRMED','CORRECTED') and photo not in ref.photos:raise ValueError('Wybierz fotografię z kanonicznego GT.')
    if decision in ('REJECTED','N/A'):
        if not reason.strip():raise ValueError('Odrzucenie/N/A wymaga powodu.')
        photo=None
    value=_audit(session,decision=decision,photo_sha256=photo,reason=reason.strip(),a2_proposal=proposal,evidence_frame_ids=ids)
    _change(session,'PHOTO_IDENTITY',lambda s:s['identities'].__setitem__(key,value),resume=dict(scope=scope))

def confirm_block(session,viewed):
    ref=reference(session);exps=[e for e in ref.exposures.values() if e['session_id']==session.session_id]
    if {e['exposure_id'] for e in exps}-set(viewed):raise ValueError('Najpierw obejrzyj wszystkie 10 ekspozycji bloku.')
    def change(s):
        for e in exps:
            key=e['exposure_id']
            if key not in s['identities']:
                s['identities'][key]=_audit(session,decision='CONFIRMED',photo_sha256=e['photo_sha256'],reason='',a2_proposal=e['photo_sha256'],evidence_frame_ids=e['evidence_frame_ids'])
        s['blocks'][session.session_id]=_audit(session,viewed_exposures=sorted(viewed),scope='PHOTO_IDENTITY_ONLY',mt_mz_approved=False)
    _change(session,'CONFIRM_VIEWED_PHOTO_BLOCK',change)

def _unit(session,gid,st=None):
    ref=reference(session);fid=ref.call_frame(session,gid);i=identity(session,fid,st)
    if not i['approved']:raise ValueError('Najpierw potwierdź albo popraw tożsamość fotografii.')
    return fid,i,{p['object_id']:p for p in ref.plates(i['photo_sha256'])}

def detection_role(session,gid,aid):
    from . import mobile_canvas_review as core
    from .mobile_mt_postdedup import final_ids,suppressed_ids
    try:
        s=core._legacy_scene(session,gid,include_history=False);proof=s.get('post_nms_proof') or core._post_nms_proof(session,gid,s['parents'])
    except (ValueError,KeyError,TypeError):proof=None
    if proof and aid in final_ids(proof):return 'FINAL'
    if proof and aid in suppressed_ids(proof):return 'SUPPRESSED'
    role=session.mt_detection_roles.get(gid,{})
    if role.get('role_proof_status')=='COMPLETE':
        if aid in role.get('final_attempt_ids',()):return 'FINAL'
        if aid in role.get('suppressed_attempt_ids',()):return 'SUPPRESSED'
    return 'UNKNOWN'

def decide_detection(session,gid,aid,outcome,*,gt_id=None,reason='',coverage=None):
    fid,i,plates=_unit(session,gid);group=session.mt_invocations[gid]
    if aid not in {str(r.get('id') or r.get('attempt_id')) for r in group.records}:raise ValueError('Detekcja nie należy do wywołania.')
    if str(session.attempts[aid].get('mt_status')).upper() not in ('VALID_QUAD','DETECTION_INVALID_QUAD'):raise ValueError('Wybrany zapis nie zawiera detekcji.')
    if outcome not in ('TP','FP','DUPLICATE','N/A'):raise ValueError('Nieprawidłowa decyzja MT.')
    if outcome!='N/A' and (not group.executed or group.cancelled or group.execution_failed):raise ValueError('Wywołanie niewykonane/anulowane/błędne wymaga N/A.')
    if outcome!='N/A' and detection_role(session,gid,aid)=='SUPPRESSED':raise ValueError('Źródłowy RAW po NMS nie jest końcową detekcją. Zachowaj N/A diagnostyczne.')
    if outcome in ('TP','DUPLICATE') and gt_id not in plates:raise ValueError('Wskaż tablicę kanonicznego GT.')
    if outcome=='N/A' and not reason.strip():raise ValueError('N/A wymaga powodu.')
    if coverage is not None and coverage not in COVERAGE:raise ValueError('Nieprawidłowe pokrycie.')
    def change(s):
        others=[r for key,r in s['detections'].items() if key!=aid and r['gid']==gid and r.get('gt_id')==gt_id and r['outcome']=='TP' and r['identity_decision_id']==i['decision_id']]
        if outcome=='TP' and others:raise ValueError('Ta tablica ma już TP w tym wywołaniu. Oznacz duplikat.')
        if outcome=='DUPLICATE' and not others:raise ValueError('Najpierw wskaż prawdziwą detekcję tej tablicy w tym wywołaniu.')
        s['units'].pop(gid,None)
        s['detections'][aid]=_audit(session,gid=gid,aid=aid,outcome=outcome,gt_id=gt_id if outcome in ('TP','DUPLICATE') else None,
            photo_sha256=i['photo_sha256'],frame_id=fid,identity_decision_id=i['decision_id'],reason=reason.strip(),source_role=detection_role(session,gid,aid))
        if coverage is not None and gt_id:
            s['coverage'][gid+'|'+gt_id]=_audit(session,gid=gid,gt_id=gt_id,status=coverage,reason='',frame_id=fid,identity_decision_id=i['decision_id'],actual_roi=group.roi_geometry)
    _change(session,'DETECTION_DECISION',change,resume=dict(gid=gid,frame_id=fid))

def batch_single_tp(session,gids,gt_id):
    """Explicit reviewed selection; only one proven FINAL per real call."""
    plan=[]
    for gid in dict.fromkeys(gids):
        fid,i,plates=_unit(session,gid);group=session.mt_invocations[gid]
        if gt_id not in plates or not group.executed or group.cancelled or group.execution_failed:raise ValueError('Zaznaczone wywołania muszą dotyczyć tej samej tablicy i mieć poprawny przebieg.')
        aids=[str(r.get('id') or r.get('attempt_id')) for r in group.records if r.get('mt_status') in ('VALID_QUAD','DETECTION_INVALID_QUAD')]
        finals=[aid for aid in aids if detection_role(session,gid,aid)=='FINAL']
        if len(finals)!=1 or any(detection_role(session,gid,aid)=='UNKNOWN' for aid in aids):raise ValueError('Ocena seryjna wymaga dokładnie jednego źródłowo potwierdzonego FINAL w każdym wywołaniu.')
        plan.append((gid,fid,i,finals[0]))
    if not plan:raise ValueError('Zaznacz wywołania MT do oceny seryjnej.')
    def change(s):
        for gid,fid,i,aid in plan:
            if any(r['gid']==gid and key!=aid and r.get('gt_id')==gt_id and r['outcome']=='TP' for key,r in s['detections'].items()):raise ValueError('Konflikt istniejącej decyzji TP.')
            s['units'].pop(gid,None)
            s['detections'][aid]=_audit(session,gid=gid,aid=aid,outcome='TP',gt_id=gt_id,photo_sha256=i['photo_sha256'],
                frame_id=fid,identity_decision_id=i['decision_id'],reason='',source_role='FINAL',batch_scope=list(dict.fromkeys(gids)))
            s['coverage'][gid+'|'+gt_id]=_audit(session,gid=gid,gt_id=gt_id,status='COVERED',reason='',frame_id=fid,identity_decision_id=i['decision_id'],actual_roi=session.mt_invocations[gid].roi_geometry)
    _change(session,'EXPLICIT_REVIEWED_SINGLE_FINAL_BATCH',change)

def text_na(session,aid,reason):
    if not reason.strip():raise ValueError('N/A odczytu wymaga powodu.')
    def change(s):
        if aid not in s['detections']:raise ValueError('Najpierw oceń wskazanie MT.')
        s['units'].pop(s['detections'][aid]['gid'],None)
        s['detections'][aid]['text_exclusion_reason']=reason.strip()
    _change(session,'READOUT_NA',change)

def decide_coverage(session,gid,gt_id,status,reason=''):
    fid,i,plates=_unit(session,gid)
    if gt_id not in plates or status not in COVERAGE:raise ValueError('Nieprawidłowa tablica lub decyzja ROI.')
    if status=='ROI_COVERAGE_UNPROVEN' and not reason.strip():raise ValueError('Pokrycie N/A wymaga powodu.')
    value=_audit(session,gid=gid,gt_id=gt_id,status=status,reason=reason.strip(),frame_id=fid,identity_decision_id=i['decision_id'],actual_roi=session.mt_invocations[gid].roi_geometry)
    def change(s):
        s['units'].pop(gid,None);s['coverage'][gid+'|'+gt_id]=value
    _change(session,'ROI_DECISION',change,resume=dict(gid=gid,frame_id=fid))

def decide_missing(session,gid,gt_id,reason='',*,confirm_coverage=False):
    fid,i,plates=_unit(session,gid);s=state(session);coverage=s['coverage'].get(gid+'|'+gt_id,{})
    group=session.mt_invocations[gid]
    if not group.executed or group.cancelled or group.execution_failed:raise ValueError('FN wymaga wykonanego MT. Dla anulowanego/brakującego wyniku użyj N/A.')
    if gt_id not in plates or not confirm_coverage and (coverage.get('status')!='COVERED' or coverage.get('identity_decision_id')!=i['decision_id']):
        raise ValueError('FN wywołania wymaga jawnie potwierdzonego pokrycia tablicy przez użyty ROI.')
    if any(r['gid']==gid and r.get('gt_id')==gt_id and r['outcome']=='TP' and r['identity_decision_id']==i['decision_id'] for r in s['detections'].values()):raise ValueError('Tablica ma już TP w tym wywołaniu.')
    def change(st):
        st['units'].pop(gid,None)
        if confirm_coverage:st['coverage'][gid+'|'+gt_id]=_audit(session,gid=gid,gt_id=gt_id,status='COVERED',reason='',frame_id=fid,identity_decision_id=i['decision_id'],actual_roi=session.mt_invocations[gid].roi_geometry)
        st['missing'][gid+'|'+gt_id]=_audit(session,gid=gid,gt_id=gt_id,outcome='FN',reason=reason,frame_id=fid,identity_decision_id=i['decision_id'])
    _change(session,'MANUAL_FN',change)

def unit_na(session,gid,reason):
    if not reason.strip():raise ValueError('N/A wymaga powodu.')
    ref=reference(session);fid=ref.call_frame(session,gid)
    ident=identity(session,fid)
    _change(session,'UNIT_NA',lambda s:s['units'].__setitem__(gid,_audit(session,status='N/A',reason=reason.strip(),frame_id=fid,identity_decision_id=ident['decision_id'])),resume=dict(gid=gid,frame_id=fid))

def confirm_empty_unit(session,gid):
    fid,i,plates=_unit(session,gid)
    from .mobile_mt_invocations import DETECTION_STATUSES
    group=session.mt_invocations[gid]
    if not group.executed or group.cancelled or group.execution_failed:raise ValueError('Brak poprawnie wykonanego MT — użyj N/A, nie poprawnego pustego wyniku.')
    if plates or any(r.get('mt_status') in DETECTION_STATUSES for r in session.mt_invocations[gid].records):raise ValueError('To wywołanie nie jest pustym wynikiem dla NO_PLATES.')
    _change(session,'CONFIRM_EMPTY_UNIT',lambda s:s['units'].__setitem__(gid,_audit(session,status='EMPTY_REVIEWED',frame_id=fid,identity_decision_id=i['decision_id'])))

def undo(session,*,redo=False):
    ref=reference(session);ref.verify();layer=session.review.provenance.get(KEY)
    if not layer:return
    pos=layer['cursor']+(1 if redo else 0)
    if pos<0 or pos>=len(layer['timeline']):return
    changed=copy.deepcopy(session.review);target=changed.provenance[KEY]
    for c in target['versions'][target['timeline'][pos]]['changes']:
        value=c['after' if redo else 'before'];bucket=target['state'][c['bucket']]
        if value is None:bucket.pop(c['key'],None)
        else:bucket[c['key']]=copy.deepcopy(value)
    target['cursor']+=1 if redo else -1
    session._save_change(changed)

def validate_layer(session,review):
    layer=review.provenance.get(KEY)
    if not layer:return
    if layer.get('schema')!=SCHEMA or layer.get('binding',{}).get('series_id')!=SERIES:raise ValueError('Nieobsługiwana warstwa A3.')
    if review.review_mode!='assisted':raise ValueError('A3 korzysta z gotowego GT i predykcji — zachowaj jawny tryb asystowany.')
    if set(layer['state'])!=set(BUCKETS):raise ValueError('Uszkodzona warstwa A3.')
    if not -1<=layer['cursor']<len(layer['timeline']):raise ValueError('Uszkodzona historia A3.')
    for vid in layer['timeline']:
        if vid not in layer['versions']:raise ValueError('Brak wersji historii A3.')
    for bucket in BUCKETS:
        for r in layer['state'][bucket].values():
            if r.get('origin')!='OPERATOR_EXPLICIT' or not r.get('decision_id'):raise ValueError('Brak jawnej decyzji operatora A3.')
            if r.get('gid') and r['gid'] not in session.mt_invocations:raise ValueError('A3 odwołuje się do obcego wywołania.')
    for aid,r in layer['state']['detections'].items():
        if aid not in session.attempts or r['aid']!=aid or session.invocation_by_attempt.get(aid)!=r['gid']:raise ValueError('Obcy wynik MT w A3.')
        if r.get('outcome') not in ('TP','FP','DUPLICATE','N/A') or r.get('outcome')=='N/A' and not r.get('reason','').strip():raise ValueError('Nieprawidłowe rozstrzygnięcie MT A3.')
    for key,r in layer['state']['coverage'].items():
        if r.get('status') not in COVERAGE or key!=r['gid']+'|'+r['gt_id'] or r['actual_roi']!=session.mt_invocations[r['gid']].roi_geometry:raise ValueError('Niespójne źródłowe ROI w A3.')
        if r['status']=='ROI_COVERAGE_UNPROVEN' and not r.get('reason','').strip():raise ValueError('N/A ROI wymaga powodu.')
    for gid,r in layer['state']['units'].items():
        if gid not in session.mt_invocations or r.get('status') not in ('N/A','EMPTY_REVIEWED'):raise ValueError('Nieprawidłowa ocena wywołania A3.')
        if r['status']=='N/A' and not r.get('reason','').strip():raise ValueError('N/A wywołania wymaga powodu.')
    ref=getattr(session,'_f5a_reference',None)
    if ref:
        ref.verify()
        if layer['binding']!=ref.binding:raise ValueError('Zmieniono źródła A3.')
        for r in layer['state']['identities'].values():
            if r['decision'] not in ('CONFIRMED','CORRECTED','REJECTED','N/A'):raise ValueError('Nieprawidłowy status tożsamości A3.')
            if r['decision'] in ('CONFIRMED','CORRECTED') and r.get('photo_sha256') not in ref.photos:raise ValueError('Fotografia spoza referencji.')
            if r['decision'] in ('REJECTED','N/A') and (r.get('photo_sha256') is not None or not r.get('reason','').strip()):raise ValueError('Odrzucone/N/A powiązanie wymaga powodu, bez fotografii zastępczej.')
            if any(fid not in ref.frames or ref.frames[fid]['session_id']!=session.session_id for fid in r['evidence_frame_ids']):raise ValueError('Obca klatka w decyzji A3.')
        for bucket in ('coverage','detections','missing','units'):
            for key,r in layer['state'][bucket].items():
                gid=r.get('gid',key)
                if ref.frame_key(r['frame_id'])!=ref.frame_key(ref.call_frame(session,gid)):raise ValueError('Ocena nie odpowiada źródłowej klatce wywołania.')
        for r in layer['state']['detections'].values():
            if r['outcome'] in ('TP','DUPLICATE') and r['gt_id'] not in {p['object_id'] for p in ref.plates(r['photo_sha256'])}:raise ValueError('Tablica nie należy do wskazanej fotografii.')

def unit_state(session,gid):
    ref=reference(session);st=state(session);manual=st['units'].get(gid,{})
    fid=ref.call_frame(session,gid);current=identity(session,fid,st)
    if manual.get('status')=='N/A' and manual.get('identity_decision_id')==current['decision_id']:
        return dict(complete=True,status='N/A',reason=manual['reason'],rows=[],gt_assessments=[],pending=[])
    try:fid,i,plates=_unit(session,gid,st)
    except ValueError as exc:return dict(complete=False,status='IDENTITY_PENDING',reason=str(exc),rows=[],pending=['PHOTO_IDENTITY'])
    from .mobile_canvas_review import _prediction
    from .mobile_object_mz_quality import score_readout
    pending=[];rows=[];linked=set();gt_rows=[]
    group=session.mt_invocations[gid]
    if not group.executed or group.cancelled or group.execution_failed:
        return dict(complete=False,status='EXECUTION_UNASSESSABLE',reason='Brak poprawnego wykonania MT — wybierz N/A z powodem.',rows=[],gt_assessments=[],pending=['EXECUTION_NA_REQUIRED'])
    for a in session.mt_invocations[gid].records:
        aid=str(a.get('id') or a.get('attempt_id'))
        if a.get('mt_status') not in ('VALID_QUAD','DETECTION_INVALID_QUAD'):continue
        role=detection_role(session,gid,aid);r=st['detections'].get(aid,{})
        valid=r.get('identity_decision_id')==i['decision_id'];outcome=r.get('outcome') if valid else None
        if outcome=='DUPLICATE' and not any(v['gid']==gid and v.get('gt_id')==r.get('gt_id') and v['outcome']=='TP' and v['identity_decision_id']==i['decision_id'] for v in st['detections'].values()):
            valid=False;outcome=None
        if role!='SUPPRESSED' and not outcome:pending.append('MT:'+aid)
        plate=plates.get(r.get('gt_id'),{}) if valid else {};text_state=plate.get('text_state') or ('FULL_TEXT' if plate.get('gt_text') else 'UNREADABLE_TEXT')
        pred,mz_status,ran=_prediction(session,aid)
        flag=str(a.get('mz_executed','')).lower();executed=True if flag in ('true','1') else False if flag in ('false','0') or mz_status=='NOT_RUN' else None
        samples=[session.samples[sid] for sid in session.sample_ids_by_attempt.get(aid,())]
        sample=samples[0] if len(samples)==1 else {}
        crop=a.get('evidence_entry')
        # Same source-crop contract as mobile_object_mz_quality.snapshot.
        # Multiple recorded reads are retained in the archive, never reduced
        # to a best/consensus result for A3 after looking at canonical GT.
        lineage=all(str(sample[k])==str(a.get(k,'')) for k in ('session_id','mt_invocation_id',*STAMP)
                    if sample.get(k) is not None) and (not sample.get('attempt_id') or sample['attempt_id']==aid)
        crop_proven=bool(crop and crop in session.entry_names and a.get('evidence_kind')=='plate_crop' and lineage
                         and (not sample or (sample.get('image_entry') or sample.get('evidence_entry'))==crop))
        scored=score_readout(dict(executed=executed,parent_proven=role=='FINAL',source_mt_role=role,
            mt_outcome=outcome,operator_decision='NOT_PLATE' if outcome=='FP' else None,crop_parent_proven=crop_proven,
            readout_unambiguous=len(samples)<=1 and mz_status!='AMBIGUOUS_SAMPLES',gtz_confirmed=valid and text_state=='FULL_TEXT' and bool(plate.get('gt_text')),
            gtz=plate.get('gt_text','') if text_state=='FULL_TEXT' else '',raw_prediction=pred))
        if text_state=='PARTIAL_TEXT':scored.update(canonical_ocr_eligible=False,exact_match=None,cer=None,quality_reason='PARTIAL_TEXT_NOT_FULL_GT')
        if valid and r.get('text_exclusion_reason'):scored.update(canonical_ocr_eligible=False,exact_match=None,cer=None,edit_distance=None,gt_characters=None,quality_reason='OPERATOR_NA: '+r['text_exclusion_reason'])
        if outcome=='TP':linked.add(r['gt_id'])
        rows.append(dict(scored,mt_invocation_id=gid,attempt_id=aid,photo_sha256=i['photo_sha256'],gt_id=r.get('gt_id') if valid else None,
            operational_outcome=outcome or 'UNRESOLVED',gt_text_state=text_state,polygon_iou=None,nmce=None,tp_at_iou_050=None,
            metric_scope='MANUAL_OPERATIONAL_NOT_GEOMETRIC',source_mz_status=mz_status,source_sample_ids=list(session.sample_ids_by_attempt.get(aid,())),
            source_execution_id=sample.get('mz_execution_id') or a.get('mz_execution_id'),crop_entry=crop))
    for pid in plates:
        c=st['coverage'].get(gid+'|'+pid,{});m=st['missing'].get(gid+'|'+pid,{})
        coverage=c.get('status') if c.get('identity_decision_id')==i['decision_id'] else 'UNRESOLVED'
        missing=m.get('identity_decision_id')==i['decision_id'] and coverage=='COVERED' and pid not in linked
        if coverage=='UNRESOLVED':pending.append('ROI:'+pid)
        elif coverage=='OUTSIDE_ROI' and pid in linked:pending.append('ROI_CONFLICT:'+pid)
        elif coverage=='COVERED' and pid not in linked and not missing:pending.append('TP_OR_FN:'+pid)
        gt_rows.append(dict(mt_invocation_id=gid,gt_id=pid,photo_sha256=i['photo_sha256'],coverage=coverage,coverage_reason=c.get('reason',''),
            operational_outcome='TP' if pid in linked else 'FN' if missing else 'N/A' if coverage in COVERAGE[1:] else 'UNRESOLVED',
            polygon_iou=None,nmce=None,geometry_reason='NO_CAMERA_GT_TRANSFORM'))
    if not plates and not rows and (manual.get('status')!='EMPTY_REVIEWED' or manual.get('identity_decision_id')!=i['decision_id']):pending.append('EMPTY_UNIT_INSPECTION')
    return dict(complete=not pending,status='REVIEWED' if not pending else 'PENDING',reason='',rows=rows,gt_assessments=gt_rows,pending=pending)

def approve_session(session):
    ref=reference(session)
    unresolved=sum(identity(session,fid)['decision']=='PROPOSED' for fid,f in ref.frames.items() if f['session_id']==session.session_id)
    pending=sum(not unit_state(session,g)['complete'] for g in session.mt_invocations)
    if unresolved or pending:raise ValueError(f'A3 nie jest ukończone: nieocenione powiązania klatek {unresolved}, wywołania MT {pending}.')
    _change(session,'APPROVE_A3_SESSION',lambda s:s['blocks'].__setitem__('approval:'+session.session_id,
        _audit(session,status='OPERATOR_APPROVED_A3',binding=ref.binding,scope='A3_ONLY_NO_GEOMETRIC_QUALIFICATION')))

def export(session,directory):
    if KEY not in session.review.provenance:return []
    from .mobile_human_review import _atomic_write
    ref=reference(session);ref.verify();units={g:unit_state(session,g) for g in session.mt_invocations}
    identities_complete=all(identity(session,fid)['decision']!='PROPOSED' for fid,f in ref.frames.items() if f['session_id']==session.session_id)
    payload=dict(schema=SCHEMA,session_id=session.session_id,source_archive_sha256=session.review.source_archive_sha256,
        review_revision=session.review.review_revision,layer=copy.deepcopy(session.review.provenance[KEY]),units=units,
        operator_complete=identities_complete and all(u['complete'] for u in units.values()),photo_identities_complete=identities_complete,
        operator_approved=bool(state(session)['blocks'].get('approval:'+session.session_id)),
        geometric_metrics='N/A_NO_CAMERA_GT_TRANSFORM')
    path=Path(directory)/'f5a_a3_operator_review.json';_atomic_write(path,json.dumps(payload,ensure_ascii=False,indent=2))
    def csv_rows(field):
        return [dict(session_id=session.session_id,source_archive_sha256=session.review.source_archive_sha256,review_revision=session.review.review_revision,
            mapping_sha256=ref.binding['mapping_sha256'],gt_sha256=ref.binding['gt_sha256'],unit_status=u['status'],unit_complete=u['complete'],**r)
            for u in units.values() for r in u.get(field,[])]
    rows=csv_rows('rows');csv_path=Path(directory)/'f5a_a3_operational_results.csv'
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['attempt_id']);writer.writeheader();writer.writerows(rows)
    _atomic_write(csv_path,'\ufeff'+stream.getvalue())
    gt_rows=csv_rows('gt_assessments');roi_path=Path(directory)/'f5a_a3_roi_and_missing.csv'
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(gt_rows[0]) if gt_rows else ['gt_id']);writer.writeheader();writer.writerows(gt_rows)
    _atomic_write(roi_path,'\ufeff'+stream.getvalue());return [path,csv_path,roi_path]
