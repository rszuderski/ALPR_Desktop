"""Original-photograph GT source for the Z4 Canvas, independent of Android.

One atomic, versioned reference document is authoritative. No MobileReviewSession,
MT record, inference, image copy or mutation of the material's GT Pack is involved.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import copy
import hashlib
import json
import math
import os
import stat
import uuid

from PIL import Image, ImageOps
from ..gt_pack import _atomic_write_json
from ..registration_text import normalize_registration
from .mobile_pose_metrics import validate_mt_quad

SCHEMA = 'alpr.photo_reference.v1'
SCHEMA_V2 = 'alpr.photo_reference.v2'
SCHEMA_V3 = 'alpr.photo_reference.v3'
GEOMETRY_REASONS = {
    'CUT_OFF': 'Ucięta krawędzią zdjęcia',
    'OCCLUDED': 'Częściowo zasłonięta',
    'BLURRED': 'Granice niewyraźne',
    'UNDETERMINED': 'Nie można ustalić pełnego obrysu',
}
PROTOCOL = 'MANUAL_SOURCE_PHOTOGRAPH_NO_PREDICTIONS_V1'
OUTPUT_FOLDER = '04_GT_FOTOGRAFII_Z4'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp(path):
    s = Path(path).stat()
    return s.st_size, s.st_mtime_ns


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _walk(root):
    """Never enter a Windows reparse point (junction) or symlink."""
    for entry in sorted(os.scandir(root), key=lambda e: (e.name.casefold(), e.name)):
        info = entry.stat(follow_symlinks=False)
        if entry.is_symlink() or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            continue
        if entry.is_dir(follow_symlinks=False):
            yield from _walk(Path(entry.path))
        elif entry.is_file(follow_symlinks=False) and Path(entry.name).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
            yield Path(entry.path)


class Material:
    """Read-only manifest-qualified material. Hash once, recheck changed files."""
    def __init__(self, root, *, expected=40):
        self.root = Path(root).resolve(strict=True)
        self.sort_path = self.root / 'sorting_manifest.json'
        self.tv_path = self.root.parent / '03_POKAZ_TV' / 'manifest_blokow.json'
        sorting, tv = read_json(self.sort_path), read_json(self.tv_path)
        required = tv.get('presentation_order')
        if not isinstance(required, list) or len(required) != expected:
            raise ValueError(f'Manifest pokazu musi wskazywać {expected} fotografii.')
        hashes = [r.get('sha256') for r in required]
        if len(set(hashes)) != expected or any(not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
            raise ValueError('Manifest zawiera nieprawidłowe lub powtórzone tożsamości fotografii.')
        if set(hashes) != {r.get('candidate_sha256') for r in sorting.get('records', [])}:
            raise ValueError('Manifesty Sortowni i pokazu wskazują różne fotografie.')
        self.proof = {str(p): dict(sha256=digest(p), stamp=stamp(p)) for p in (self.sort_path, self.tv_path)}
        self.independence = copy.deepcopy(sorting.get('independence', {'status': 'NOT_VERIFIED'}))
        self.images = {}
        self.order = []
        files = list(_walk(self.root))
        by_relative = {p.relative_to(self.root).as_posix(): p for p in files}
        by_sha = {}
        self._verified = {}
        for path in files:
            before = stamp(path); sha = digest(path)
            if stamp(path) != before:
                raise ValueError('Fotografia zmieniła się podczas odczytu: ' + path.name)
            self._verified[str(path)] = before
            by_sha.setdefault(sha, []).append(path.relative_to(self.root).as_posix())
        for item in required:
            relative = item.get('sciezka_zrodla', '')
            posix = PurePosixPath(relative)
            if not relative or posix.is_absolute() or '..' in posix.parts or '\\' in relative or ':' in relative:
                raise ValueError('Nieprawidłowa ścieżka fotografii w manifeście.')
            sha = item['sha256']; path = by_relative.get(relative)
            if path is None or relative not in by_sha.get(sha, []):
                raise ValueError('Brak fotografii lub niezgodna zawartość: ' + relative)
            with Image.open(path) as original:
                orientation = original.getexif().get(274, 1)
                image = ImageOps.exif_transpose(original)
                width, height = image.size
                image.load()
            key = 'img-sha256-' + sha
            self.images[key] = dict(image_id=key, relative_path=relative,
                aliases=sorted(by_sha[sha]), source_file_sha256=sha,
                size_bytes=path.stat().st_size, width=width, height=height,
                exif_orientation=orientation, coordinate_space='exif_transposed_source_pixels')
            self.order.append(key)
        if not self.images:
            raise ValueError('Nie znaleziono fotografii do oznaczenia.')

    @property
    def default_output(self):
        return self.root.parent / OUTPUT_FOLDER / 'reference_v1.json'

    def verify(self, key):
        row = self.images[key]; path = self.root / row['relative_path']
        # Reparse changes after opening must also fail closed.
        for part in (path, *path.parents):
            if part == self.root.parent:
                break
            info = part.lstat()
            if part.is_symlink() or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError('Źródło fotografii zostało zastąpione dowiązaniem.')
        current = stamp(path)
        if self._verified.get(str(path)) != current:
            if digest(path) != row['source_file_sha256'] or stamp(path) != current:
                raise ValueError('Fotografia zmieniła się. Zapis jest zablokowany: ' + path.name)
            self._verified[str(path)] = current
        for name, proof in self.proof.items():
            if stamp(name) != proof['stamp']:
                if digest(name) != proof['sha256']:
                    raise ValueError('Manifest materiału zmienił się. Otwórz materiał ponownie.')
                proof['stamp'] = stamp(name)
        return path

    def image(self, key):
        path = self.verify(key)
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert('RGB')


def blank():
    return dict(presence='UNRESOLVED', objects=[])


def text_kind(obj):
    """Legacy text retains its original meaning, regardless of its length."""
    return obj.get('text_state') or ('FULL_TEXT' if obj['gt_text'] else
        'UNREADABLE_TEXT' if obj['text_exclusion_reason'] else 'UNRESOLVED')


def display_text(obj):
    return obj.get('partial_text', '') if text_kind(obj) == 'PARTIAL_TEXT' else obj['gt_text']


def text_complete(obj):
    return bool(display_text(obj) or
        (text_kind(obj) == 'UNREADABLE_TEXT' and obj['text_exclusion_reason'].strip()))


def geometry_kind(obj):
    return obj.get('geometry_kind') or ('EXACT_QUAD' if obj['quad'] is not None else 'UNAVAILABLE')


def geometry_status(obj):
    """Read legacy reasons for presentation; never modify stored decisions."""
    if obj.get('geometry_status'):return obj['geometry_status']
    if obj['quad'] is not None:return 'EXACT'
    reason=obj['geometry_exclusion_reason'].casefold()
    if 'ucięt' in reason:return 'CUT_OFF'
    if 'zasłonięt' in reason:return 'OCCLUDED'
    if 'niewyraźn' in reason or 'rozmyt' in reason:return 'BLURRED'
    return 'UNDETERMINED'


def localization_outline(obj):
    """Canvas location only. Scorers must continue to read exact `quad`."""
    return obj.get('localization_quad') or obj.get('quad')


def _explicit(obj):
    """Upgrade only an edited object; never rewrite historical snapshots."""
    obj.setdefault('text_state', text_kind(obj))
    obj.setdefault('geometry_kind', geometry_kind(obj))
    obj.setdefault('partial_text', '')
    obj.setdefault('approximate_region', None)
    obj.setdefault('approximate_point', None)
    obj.setdefault('decision_version', 2)


def _explicit_geometry(obj):
    _explicit(obj)
    obj.setdefault('geometry_status', geometry_status(obj))
    obj.setdefault('localization_quad', copy.deepcopy(obj['quad']))
    obj['decision_version'] = 3


def _set_geometry_status(obj, status, reason=None):
    if status not in ('EXACT', *GEOMETRY_REASONS):
        raise ValueError('Nieprawidłowy status geometrii.')
    _explicit_geometry(obj)
    if status=='EXACT':
        if not obj['localization_quad']:
            raise ValueError('Najpierw wskaż cztery rzeczywiste narożniki za pomocą D.')
        obj.update(quad=copy.deepcopy(obj['localization_quad']), geometry_kind='EXACT_QUAD',
                   geometry_exclusion_reason='', approximate_region=None, approximate_point=None)
    else:
        obj['quad']=None
        if obj['localization_quad']:obj['geometry_kind']='APPROXIMATE_QUAD'
        obj['geometry_exclusion_reason']=reason if reason is not None else GEOMETRY_REASONS[status]
    obj['geometry_status']=status


def complete(state):
    if state['presence'] == 'NO_PLATES':
        return not state['objects']
    return bool(state['presence'] == 'PRESENT' and state['objects'] and all(
        (o['quad'] is not None or o['geometry_exclusion_reason']) and
        text_complete(o) for o in state['objects']))


@contextmanager
def _write_lock(path):
    """OS releases the advisory lock after a crash; the tiny lock file may remain."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+b')
    locked = False
    try:
        if path.stat().st_size == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError:
            raise ValueError('Oceny są zapisywane w innym oknie. Spróbuj ponownie.') from None
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


class PhotoReference:
    """Source/save adapter, not an Android review or a scoring engine."""
    def __init__(self, material, path=None):
        self.material = material
        self.annotation_protocol = getattr(material, 'annotation_protocol', PROTOCOL)
        self.predictions_exposed = getattr(material, 'predictions_exposed', False)
        self.path = Path(path or material.default_output).resolve()
        if self.path == material.sort_path or self.path == material.tv_path or self.path.suffix != '.json' or self.path.is_relative_to(material.root):
            raise ValueError('Referencję zapisz poza katalogiem oryginalnych fotografii.')
        self._disk_sha = digest(self.path) if self.path.exists() else None
        self.data = read_json(self.path) if self.path.exists() else dict(schema=SCHEMA,
            reference_id=str(uuid.uuid4()), revision=0, last_image_id=material.order[0],
            annotation_protocol=self.annotation_protocol, predictions_exposed=self.predictions_exposed,
            dataset_independence=material.independence,
            source_manifests={Path(p).name:v['sha256'] for p,v in material.proof.items()},
            images=copy.deepcopy(material.images), reviews={}, events=[])
        self._validate(self.data)

    def _validate(self, data):
        if (data.get('schema') not in (SCHEMA, SCHEMA_V2, SCHEMA_V3)
                or data.get('annotation_protocol') != self.annotation_protocol
                or data.get('predictions_exposed') is not self.predictions_exposed):
            raise ValueError('Nieobsługiwany kontrakt referencji. Nie zmieniono istniejącego pliku.')
        if data.get('source_manifests') != {Path(p).name:v['sha256'] for p,v in self.material.proof.items()}:
            raise ValueError('Referencja dotyczy innych manifestów materiału.')
        if data.get('dataset_independence') != self.material.independence:
            raise ValueError('Niezależność zbioru nie odpowiada dowodom w manifeście.')
        if set(data['images']) != set(self.material.images) or set(data['reviews']) - set(data['images']):
            raise ValueError('Referencja dotyczy innego zestawu fotografii.')
        if data['last_image_id'] not in data['images'] or type(data['revision']) is not int:
            raise ValueError('Nieprawidłowa rewizja lub miejsce wznowienia.')
        for key, row in data['images'].items():
            for field in ('image_id','source_file_sha256','relative_path','size_bytes','width','height','coordinate_space','exif_orientation'):
                if row[field] != self.material.images[key][field]:
                    raise ValueError('Niezgodna tożsamość lub geometria fotografii.')
        for key, review in data['reviews'].items():
            if not 0 <= review['cursor'] < len(review['timeline']):
                raise ValueError('Uszkodzona historia referencji.')
            for version in review['versions'].values():
                self._validate_state(key, version['state'])
                if data['schema'] == SCHEMA and any(o.get('decision_version') for o in version['state']['objects']):
                    raise ValueError('Nowe decyzje wymagają wersji v2 referencji.')
                if data['schema'] != SCHEMA_V3 and any(o.get('decision_version')==3 for o in version['state']['objects']):
                    raise ValueError('Oddzielny obrys lokalizacji wymaga wersji v3 referencji.')
            for vid in review['timeline']:
                if vid not in review['versions']:
                    raise ValueError('Brakuje wersji w historii referencji.')

    def _validate_state(self, key, state):
        if state['presence'] not in ('UNRESOLVED','PRESENT','NO_PLATES'):
            raise ValueError('Nieprawidłowa decyzja o obecności tablic.')
        objects=state['objects'];ids=[o['object_id'] for o in objects]
        if len(ids)!=len(set(ids)) or any(not i.startswith('PGT:') for i in ids):
            raise ValueError('Nieprawidłowa tożsamość tablic.')
        if (bool(objects) != (state['presence']=='PRESENT')) or len(objects)>100:
            raise ValueError('Niespójna obecność tablic.')
        for obj in objects:
            for field in ('gt_text','text_exclusion_reason','geometry_exclusion_reason'):
                if not isinstance(obj[field],str) or len(obj[field])>256:
                    raise ValueError('Nieprawidłowy tekst lub powód N/A.')
            if obj['gt_text'] and obj['text_exclusion_reason']:
                raise ValueError('Numer i oznaczenie nieczytelności nie mogą występować razem.')
            if obj['quad'] is not None:
                row=self.material.images[key]
                validate_mt_quad(obj['quad'],row['width'],row['height'])
                if obj['geometry_exclusion_reason']:
                    raise ValueError('Obrys i brak obrysu nie mogą występować razem.')
            elif not obj['geometry_exclusion_reason'].strip():
                raise ValueError('Brak obrysu wymaga podania powodu.')
            if 'decision_version' not in obj:
                if any(f in obj for f in ('text_state','partial_text','geometry_kind','approximate_region','approximate_point','localization_quad','geometry_status')):
                    raise ValueError('Brak wersji rozszerzonych decyzji.')
                continue
            if obj['decision_version'] not in (2,3):
                raise ValueError('Nieobsługiwana wersja decyzji.')
            if obj['decision_version']==2 and any(f in obj for f in ('localization_quad','geometry_status')):
                raise ValueError('Oddzielny obrys lokalizacji wymaga wersji decyzji 3.')
            partial=obj.get('partial_text')
            if not isinstance(partial,str) or len(partial)>256:
                raise ValueError('Nieprawidłowy fragment numeru.')
            kind=obj.get('text_state')
            if kind not in ('FULL_TEXT','PARTIAL_TEXT','UNREADABLE_TEXT','UNRESOLVED'):
                raise ValueError('Nieprawidłowy stan tekstu.')
            if kind == 'FULL_TEXT' and (partial or obj['text_exclusion_reason']):
                raise ValueError('Pełny numer nie może być jednocześnie fragmentem ani N/A.')
            if kind == 'PARTIAL_TEXT' and (obj['gt_text'] or not obj['text_exclusion_reason'].strip()):
                raise ValueError('Fragment nie jest pełnym numerem GT.')
            if kind == 'UNREADABLE_TEXT' and (obj['gt_text'] or partial or not obj['text_exclusion_reason'].strip()):
                raise ValueError('Nieczytelny numer wymaga powodu i nie zawiera tekstu.')
            if kind == 'UNRESOLVED' and (obj['gt_text'] or partial or obj['text_exclusion_reason']):
                raise ValueError('Nierozstrzygnięty tekst nie może zawierać decyzji.')
            geom=obj.get('geometry_kind');region=obj.get('approximate_region');point=obj.get('approximate_point')
            kinds=('EXACT_QUAD','APPROXIMATE_REGION','APPROXIMATE_POINT','UNAVAILABLE')
            if geom not in kinds + (('APPROXIMATE_QUAD',) if obj['decision_version']==3 else ()):
                raise ValueError('Nieprawidłowy sposób lokalizacji.')
            if (geom=='EXACT_QUAD') != (obj['quad'] is not None):
                raise ValueError('Orientacyjna lokalizacja nie może zawierać dokładnego quada.')
            if (geom=='APPROXIMATE_REGION') != (region is not None) or (geom=='APPROXIMATE_POINT') != (point is not None):
                raise ValueError('Niespójna orientacyjna lokalizacja.')
            coordinates=region if region is not None else point
            if coordinates is not None:
                row=self.material.images[key]
                if (not isinstance(coordinates,(list,tuple)) or len(coordinates)!=(4 if region is not None else 2)
                        or any(type(v) not in (int,float) or not math.isfinite(v) for v in coordinates)
                        or any(not 0<=v<=row['width' if i%2==0 else 'height'] for i,v in enumerate(coordinates))):
                    raise ValueError('Lokalizacja musi leżeć na fotografii, we współrzędnych źródłowych.')
                if region is not None and (region[0]>=region[2] or region[1]>=region[3]):
                    raise ValueError('Obszar musi mieć dodatnią szerokość i wysokość.')
            if obj['decision_version']==3:
                status=obj.get('geometry_status');outline=obj.get('localization_quad')
                if status not in ('EXACT', *GEOMETRY_REASONS):
                    raise ValueError('Nieprawidłowy status geometrii.')
                if (outline is not None) != (geom in ('EXACT_QUAD','APPROXIMATE_QUAD')):
                    raise ValueError('Niespójny obrys lokalizacji.')
                if outline is not None:
                    row=self.material.images[key]
                    validate_mt_quad(outline,row['width'],row['height'])
                if status=='EXACT':
                    if geom!='EXACT_QUAD' or obj['quad']!=outline:
                        raise ValueError('Dokładna geometria musi odpowiadać narysowanemu obrysowi.')
                elif obj['quad'] is not None or not obj['geometry_exclusion_reason'].strip():
                    raise ValueError('Orientacyjny obrys nie jest dokładnym GT Pose.')

    def state(self, key):
        review=self.data['reviews'].get(key)
        return copy.deepcopy(review['versions'][review['timeline'][review['cursor']]]['state'] if review else blank())

    def scene(self, key):
        """Transient compatibility view for the existing Canvas gestures only."""
        state=self.state(key);row=self.material.images[key]
        for obj in state['objects']:
            obj.update(parent_id=None,attempt_id=None,kind='manual',error=False,
                text_confirmed=text_complete(obj),
                geometry_state='operator_corrected' if obj['quad'] else 'unavailable')
        return dict(state,scope='SOURCE_PHOTOGRAPH',image_id=key,image_entry=row['relative_path'],
            image_size=[row['width'],row['height']],roi=[0,0,row['width'],row['height']],
            parents=[],checked=complete(state),exclusion_reason='',events=[])

    def status(self, key):
        state=self.state(key)
        if complete(state):return 'Gotowe'
        if state['objects']:return 'Wymaga uzupełnienia'
        return 'W trakcie' if key in self.data['reviews'] else 'Do oznaczenia'

    def _save(self, changed, key, action):
        if getattr(self.material, 'readonly', False):
            raise ValueError('Zamrożona referencja jest tylko do odczytu; przygotuj nową wersję manifestu.')
        self.material.verify(key)
        self._validate(changed)
        with _write_lock(self.path.with_suffix('.write.lock')):
            current=digest(self.path) if self.path.exists() else None
            if current != self._disk_sha:
                raise ValueError('Oceny zmieniły się w innym oknie. Otwórz referencję ponownie; niezapisane zmiany pozostają w tym oknie.')
            if current and self.data['schema']!=changed['schema']:
                # Same lock and exact pre-migration bytes. A failed backup blocks
                # the save. Never overwrite a previously created backup.
                target=changed['schema'].rsplit('.',1)[-1]
                backup=self.path.with_name(f'{self.path.stem}.pre-{target}-{current}.json')
                original=self.path.read_bytes()
                if hashlib.sha256(original).hexdigest()!=current:
                    raise ValueError('Referencja zmieniła się przed wykonaniem kopii.')
                if backup.exists():
                    if backup.read_bytes()!=original:
                        raise ValueError('Istniejąca kopia referencji ma inną zawartość.')
                else:
                    with backup.open('xb') as stream:
                        stream.write(original);stream.flush();os.fsync(stream.fileno())
            changed['revision']=self.data['revision']+1
            changed['last_image_id']=key
            changed['events'].append(dict(at=datetime.now(timezone.utc).isoformat(),action=action,
                image_id=key,revision=changed['revision'],predictions_exposed=False))
            _atomic_write_json(self.path,changed)
            self._disk_sha=digest(self.path)
            self.data=changed

    def _change(self, key, action, change):
        state=self.state(key);result=change(state)
        self._validate_state(key,state)
        if state==self.state(key):return result
        changed=copy.deepcopy(self.data)
        if any(o.get('decision_version')==3 for o in state['objects']):
            changed['schema']=SCHEMA_V3
        elif changed['schema']==SCHEMA and any(o.get('decision_version')==2 for o in state['objects']):
            changed['schema']=SCHEMA_V2
        review=changed['reviews'].setdefault(key,dict(versions={'initial':dict(state=blank())},timeline=['initial'],cursor=0))
        vid='version-'+uuid.uuid4().hex
        review['versions'][vid]=dict(state=state,action=action,annotation_protocol=PROTOCOL,predictions_exposed=False)
        review['timeline']=review['timeline'][:review['cursor']+1]+[vid]
        review['cursor']+=1
        self._save(changed,key,action)
        return result

    def add(self, key, quad=None, geometry_reason='', *, approximate_region=None, approximate_point=None):
        oid='PGT:'+uuid.uuid4().hex
        def change(state):
            obj=dict(object_id=oid,quad=quad,gt_text='',text_exclusion_reason='',geometry_exclusion_reason=geometry_reason)
            _explicit(obj)
            if approximate_region is not None:obj.update(geometry_kind='APPROXIMATE_REGION',approximate_region=approximate_region)
            if approximate_point is not None:obj.update(geometry_kind='APPROXIMATE_POINT',approximate_point=approximate_point)
            _explicit_geometry(obj)
            state['objects'].append(obj)
            state['presence']='PRESENT'
            return oid
        return self._change(key,'ADD_PLATE',change)

    def edit(self, key, oid, *, quad=None, text=None, reason='', geometry_reason=None,
             text_state=None, approximate_region=None, approximate_point=None, localization_quad=None):
        def change(state):
            obj=next((o for o in state['objects'] if o['object_id']==oid),None)
            if obj is None:raise ValueError('Tablica nie jest już aktywna. Wybierz ją ponownie.')
            _explicit(obj)
            if quad is not None:
                _explicit_geometry(obj)
                obj.update(quad=quad,localization_quad=copy.deepcopy(quad),geometry_status='EXACT',geometry_kind='EXACT_QUAD',geometry_exclusion_reason='',approximate_region=None,approximate_point=None)
            elif localization_quad is not None:
                _explicit_geometry(obj)
                obj['localization_quad']=copy.deepcopy(localization_quad)
                _set_geometry_status(obj,obj['geometry_status'],obj['geometry_exclusion_reason'])
            elif approximate_region is not None or approximate_point is not None:
                _explicit_geometry(obj)
                obj.update(quad=None,geometry_kind='APPROXIMATE_REGION' if approximate_region is not None else
                    'APPROXIMATE_POINT',localization_quad=None,
                    approximate_region=approximate_region,approximate_point=approximate_point,
                    geometry_exclusion_reason=(geometry_reason if geometry_reason is not None else obj['geometry_exclusion_reason']).strip())
                obj.pop('geometry_status',None);obj['geometry_status']=geometry_status(obj)
            elif geometry_reason is not None:
                # Legacy API: an uncertainty reason changes accuracy, never
                # discards a drawn location. Explicit replacement is separate.
                geometry_note=geometry_reason.strip()
                status=geometry_status(dict(quad=None,geometry_exclusion_reason=geometry_note))
                _set_geometry_status(obj,status,geometry_note)
            if text is not None or text_state is not None:
                value=normalize_registration(display_text(obj) if text is None else text)
                kind=text_state or ('UNREADABLE_TEXT' if reason else
                    'PARTIAL_TEXT' if obj['text_state']=='PARTIAL_TEXT' else 'FULL_TEXT')
                if kind=='UNREADABLE_TEXT' and text is not None and value:
                    raise ValueError('Numer i oznaczenie nieczytelności nie mogą występować razem.')
                obj.update(text_state=kind,gt_text=value if kind=='FULL_TEXT' else '',
                    partial_text=value if kind=='PARTIAL_TEXT' else '',
                    text_exclusion_reason=('Znany tylko fragment numeru' if kind=='PARTIAL_TEXT' else reason.strip() if kind=='UNREADABLE_TEXT' else ''))
            return oid
        return self._change(key,'EDIT_PLATE',change)

    def set_geometry_status(self, key, oid, status):
        def change(state):
            obj=next((o for o in state['objects'] if o['object_id']==oid),None)
            if obj is None:raise ValueError('Tablica nie jest już aktywna. Wybierz ją ponownie.')
            _set_geometry_status(obj,status)
        self._change(key,'SET_GEOMETRY_STATUS',change)

    def remove(self, key, oid):
        def change(state):
            state['objects']=[o for o in state['objects'] if o['object_id']!=oid]
            state['presence']='PRESENT' if state['objects'] else 'UNRESOLVED'
        self._change(key,'REMOVE_PLATE',change)

    def empty(self, key):
        def change(state):
            if state['objects']:raise ValueError('Na fotografii są oznaczone tablice. Usuń błędne oznaczenia przed potwierdzeniem ich braku.')
            state['presence']='NO_PLATES'
        self._change(key,'NO_PLATES',change)

    def restore(self, key, *, redo=False):
        review=self.data['reviews'].get(key)
        if not review:return
        target=review['cursor']+(1 if redo else -1)
        if not 0 <= target < len(review['timeline']):return
        changed=copy.deepcopy(self.data);changed['reviews'][key]['cursor']=target
        self._save(changed,key,'REDO' if redo else 'UNDO')

    def remember(self, key):
        if key!=self.data['last_image_id']:
            self._save(copy.deepcopy(self.data),key,'NAVIGATE')

    def next_pending(self, key):
        i=self.material.order.index(key);order=self.material.order[i+1:]+self.material.order[:i+1]
        return next((k for k in order if not complete(self.state(k))),None)
