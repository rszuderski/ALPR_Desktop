"""Lossless FINAL input adapter for native Z2 XML and PhotoReference.

Original XML/images are read only. Each exposure has its own identity, even
when two slots use the same image bytes. No model or GT decision is inferred.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import hashlib
import math
from decimal import Decimal, InvalidOperation
import json
import os
import shutil
import uuid
import xml.etree.ElementTree as ET

from PIL import Image, ImageOps
from . import photo_reference as photo
from ..gt_pack import _atomic_write_json
from ..plate_ground_truth import get_plate_ground_truth
from ..registration_text import NORMALIZATION_POLICY, normalize_registration

SCHEMA = 'alpr.final.presentation.v1'
SCHEMA_V2 = 'alpr.final.presentation.v2'
REFERENCE_PROTOCOL = 'Z2_SAVED_XML_REFERENCE_V1'
CONTRACT = 'F6_FINAL180_SPLIT_V1'
DEFAULT_DWELL_SECONDS = 5.0


def parse_dwell(value):
    """Operator seconds, with decimal comma; bounded by existing 3600 s cap."""
    try:
        if isinstance(value,bool):raise ValueError()
        number=Decimal(str(value).strip().replace(',','.'))
        if not number.is_finite() or not Decimal('0.000000001')<=number<=3600:
            raise ValueError()
        if number*1_000_000_000!=(number*1_000_000_000).to_integral_value():
            raise ValueError()
        return float(number)
    except (ValueError,InvalidOperation):
        raise ValueError('Czas zdjęcia: wpisz dodatnią, skończoną liczbę od 0,000000001 do 3600 s, maksymalnie 9 miejsc po przecinku.') from None


def plan_times(count,dwell_seconds,*,countdown_seconds=5,transition_seconds=.5):
    seconds=parse_dwell(dwell_seconds)
    photos=count*seconds
    return dict(exposures=count,dwell_seconds=seconds,planned_photo_seconds=photos,
        countdown_seconds=countdown_seconds,transition_seconds=transition_seconds,
        planned_total_seconds=float(countdown_seconds)+photos+max(count-1,0)*float(transition_seconds))


def format_seconds(seconds):
    minutes=int(seconds//60);rest=seconds-minutes*60
    return f'{minutes}:{rest:012.9f}'.rstrip('0').rstrip('.').replace('.',',')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
POLICY = dict(id='f6_first_source_mt_bbox_iou50_v1',
    selection='FIRST_IDENTIFIED_ACTIVE_SOURCE_FRAME_WITH_STARTED_MT',
    order='capture_elapsed_nanos_then_source_sequence_then_frame_key',
    same_frame='ALL_SOURCE_BOUND_MT_INVOCATIONS_INCLUDING_FAILURES',
    detection='EXISTING_CONFIDENCE_DESC_INDEX_ASC_IOU_DESC_GT_ID_ASC_GREEDY',
    final_index_order='MT_INVOCATION_ID_THEN_SOURCE_DETECTION_INDEX_THEN_ATTEMPT_ID',
    geometry='CONTINUOUS_FULL_CAMERAX_BBOX_XYXY_FROM_TRANSFORMED_GT_QUAD',
    iou_threshold=0.5, no_refilter=True, normalization=NORMALIZATION_POLICY,
    full_text='FULL_TEXT_ONLY', unreadable='DETECTION_ONLY',
    system_denominator='ALL_GEOMETRICALLY_EVALUABLE_FULL_TEXT_GT_IN_IDENTIFIED_SELECTED_INPUTS_INCLUDING_FAILURES',
    no_clock_alignment=True, no_final_hud_claim=True,
    technical_tolerance_status='SYNTHETIC_VALIDATION_ONLY_NOT_A12_TV_ACCEPTANCE')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def files(root, *, recursive=True):
    # Reuse the existing safe walk for supported formats; add no traversal of links.
    root = Path(root).resolve(strict=True)
    result = []
    for base, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if not Path(base, d).is_symlink() and not Path(base, d).is_junction()) if recursive else []
        for name in sorted(names):
            path = Path(base, name)
            if path.suffix.lower() in EXTENSIONS and not path.is_symlink():
                result.append(path)
    return sorted(result, key=lambda p: (p.relative_to(root).as_posix().casefold(), p.name))


def orient_points(points, size, orientation):
    """Continuous image-edge coordinates, corresponding to Pillow EXIF transpose."""
    w, h = size
    operations = {1: lambda x,y:(x,y), 2:lambda x,y:(w-x,y), 3:lambda x,y:(w-x,h-y),
        4:lambda x,y:(x,h-y), 5:lambda x,y:(y,x), 6:lambda x,y:(h-y,x),
        7:lambda x,y:(h-y,w-x), 8:lambda x,y:(y,w-x)}
    if orientation not in operations:raise ValueError('EXIF_ORIENTATION_UNSUPPORTED')
    return [list(operations[orientation](float(x), float(y))) for x,y in points]


def audit(xml_path, positive_dir, negative_dir=None, *, negative_declared=False,
          unreadable_plate_ids=(), expected_positive=None, expected_negative=10,
          xml_space='exif_transposed_source_pixels'):
    from ..gui.z2_preview_workflow import _parse_cvat_preview_annotations
    xml_path, positive_dir = Path(xml_path).resolve(strict=True), Path(positive_dir).resolve(strict=True)
    xml_sha = photo.digest(xml_path)
    inventory = ET.parse(xml_path).getroot().findall('.//image')
    helper = SimpleNamespace(_bbox_from_polygon=lambda p: (min(x for x,y in p),min(y for x,y in p),max(x for x,y in p),max(y for x,y in p)))
    parsed = _parse_cvat_preview_annotations(helper, xml_path)
    annotations = {a.filename: a for a in parsed}
    errors, warnings, images, blank_ids = [], [], [], []
    export_manifest=xml_path.parent/'annotation_export_manifest.json'
    if export_manifest.is_file() and photo.read_json(export_manifest).get('annotation_scope')=='approved_ok_only':
        errors.append('APPROVED_ONLY_XML_NOT_ALLOWED_FOR_FINAL')
    positives = files(positive_dir)
    negatives = files(negative_dir) if negative_dir else []
    if len(inventory) != len(parsed) or len(annotations) != len(inventory):errors.append('XML_RECORDS_INVALID_OR_DUPLICATED')
    if expected_positive is not None and len(positives) != expected_positive:errors.append(f'POSITIVE_COUNT:{len(positives)}/{expected_positive}')
    if expected_negative is not None and len(negatives) != expected_negative:errors.append(f'NEGATIVE_COUNT:{len(negatives)}/{expected_negative}')
    if negatives and not negative_declared:errors.append('NEGATIVE_DECLARATION_MISSING')
    if xml_space not in ('raw_file_pixels', 'exif_transposed_source_pixels'):raise ValueError('XML_COORDINATE_SPACE_REQUIRED')
    relative_positive = {p.relative_to(positive_dir).as_posix(): p for p in positives}
    xml_names = {el.get('name') for el in inventory}
    for name in sorted(xml_names - set(relative_positive)):
        errors.append('XML_IMAGE_NOT_IN_POSITIVE_INVENTORY:' + str(name))
    for name in sorted(set(relative_positive) - xml_names):errors.append('POSITIVE_IMAGE_MISSING_XML:' + name)
    ids = set()
    for role, paths, root in (('POSITIVE', positives, positive_dir), ('NEGATIVE', negatives, Path(negative_dir).resolve() if negative_dir else None)):
        for path in paths:
            relative = path.relative_to(root).as_posix()
            before = photo.stamp(path); sha = photo.digest(path)
            with Image.open(path) as opened:
                raw_size = list(opened.size); orientation = int(opened.getexif().get(274, 1))
                oriented = ImageOps.exif_transpose(opened); size = list(oriented.size)
            if before != photo.stamp(path):raise ValueError('SOURCE_CHANGED_DURING_AUDIT')
            ann = annotations.get(relative) if role == 'POSITIVE' else None
            state = photo.blank()
            if role == 'NEGATIVE' and negative_declared:state['presence'] = 'NO_PLATES'
            shapes = [d for d in ann.detections if d.label == 'plate'] if ann else []
            if role == 'POSITIVE' and not shapes:errors.append('POSITIVE_WITHOUT_PLATE:' + relative)
            xml_shapes = next((len(el.findall('polygon')) for el in inventory if el.get('name') == relative), 0)
            if role == 'POSITIVE' and xml_shapes != len(shapes):errors.append('UNPARSED_XML_SHAPE:' + relative)
            if ann and [ann.width, ann.height] != (raw_size if xml_space == 'raw_file_pixels' else size):
                errors.append('XML_DIMENSIONS_CONFLICT:' + relative)
            for index, detection in enumerate(shapes):
                attrs = detection.attributes
                pid = str(attrs.get('plate_annotation_id') or '')
                if not pid or pid in ids:errors.append('PLATE_ID_MISSING_OR_DUPLICATED:' + relative)
                ids.add(pid)
                pid = pid or 'unresolved-' + fingerprint([xml_sha, relative, index])
                q = [list(p) for p in detection.polygon]
                if xml_space == 'raw_file_pixels':q = orient_points(q, raw_size, orientation)
                reason = ''
                try:photo.validate_mt_quad(q, *size)
                except (ValueError, TypeError):q = None;reason = 'INVALID_XML_QUAD';errors.append('INVALID_GEOMETRY:' + pid)
                text = get_plate_ground_truth(attrs)
                declared_kind = attrs.get('text_state') or attrs.get('ground_truth_text_state')
                kind = 'PARTIAL_TEXT' if declared_kind in ('PARTIAL', 'PARTIAL_TEXT') else 'FULL_TEXT' if text else 'UNREADABLE_TEXT' if pid in unreadable_plate_ids else 'UNRESOLVED'
                if not text:blank_ids.append(pid)
                if kind == 'UNRESOLVED':errors.append('UNRESOLVED_TEXT:' + pid)
                partial = text if kind == 'PARTIAL_TEXT' else ''
                obj = dict(object_id='PGT:' + pid, source_plate_annotation_id=pid,
                    source_xml_attributes=deepcopy(attrs), quad=q, gt_text=text if kind == 'FULL_TEXT' else '',
                    text_exclusion_reason=('Deklaracja operatora: nieczytelny numer' if kind == 'UNREADABLE_TEXT' else 'Jawny tekst częściowy' if partial else ''),
                    geometry_exclusion_reason=reason, decision_version=2, text_state=kind, partial_text=partial,
                    geometry_kind='EXACT_QUAD' if q else 'UNAVAILABLE', approximate_region=None, approximate_point=None)
                state['objects'].append(obj)
                state['presence'] = 'PRESENT'
            images.append(dict(path=str(path), relative_path=relative, source_role=role,
                image_id='img-sha256-'+sha+'-'+fingerprint([role,relative])[:12],
                sha256=sha, bytes=before[0], mtime_ns=before[1], raw_size=raw_size, size=size,
                exif_orientation=orientation, xml_coordinate_space=xml_space, state=state))
    by_sha = {}
    def content(state):
        return dict(presence=state['presence'],objects=sorted(
            [dict(quad=o['quad'],text=o['gt_text'],kind=o['text_state'],partial=o['partial_text']) for o in state['objects']],
            key=fingerprint))
    for row in images:
        old = by_sha.setdefault(row['sha256'], row)
        if content(old['state']) != content(row['state']):errors.append('SAME_BYTES_CONFLICTING_REFERENCE:' + row['relative_path'])
    if len(by_sha) != len(images):warnings.append('REPEATED_BYTES_RETAINED_AS_SEPARATE_EXPOSURES')
    objects = [o for r in images for o in r['state']['objects']]
    summary = dict(positive_images=len(positives), negative_images=len(negatives), xml_image_records=len(inventory),
        images_with_plates=sum(bool(r['state']['objects']) for r in images), plates=len(objects),
        full_text=sum(photo.text_kind(o) == 'FULL_TEXT' for o in objects),
        unreadable=sum(photo.text_kind(o) == 'UNREADABLE_TEXT' for o in objects),
        partial=sum(photo.text_kind(o) == 'PARTIAL_TEXT' for o in objects), empty_text=len(blank_ids),
        no_plate_positive=[r['relative_path'] for r in images if r['source_role']=='POSITIVE' and not r['state']['objects']])
    if photo.digest(xml_path) != xml_sha:raise ValueError('XML_CHANGED_DURING_AUDIT')
    return dict(schema='alpr.final.input_audit.v1', xml_path=str(xml_path), xml_sha256=xml_sha,
        positive_dir=str(positive_dir), negative_dir=str(negative_dir) if negative_dir else None,
        negative_declared=negative_declared, unreadable_plate_ids=sorted(set(unreadable_plate_ids)),
        blank_plate_ids=blank_ids, images=images, summary=summary, errors=sorted(set(errors)), warnings=warnings,
        status='PASS' if not errors else 'NO_GO')


class FrozenMaterial:
    annotation_protocol = REFERENCE_PROTOCOL
    predictions_exposed = True  # Z2 auto-assisted origins are retained, never relabeled blind GT.
    readonly = True

    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        self.sort_path = self.directory/'material.json'
        self.tv_path = self.directory/'source_annotations.xml'
        self.data = photo.read_json(self.sort_path)
        self.annotation_protocol=self.data.get('annotation_protocol',REFERENCE_PROTOCOL)
        self.predictions_exposed=self.data.get('predictions_exposed',True)
        self.root = self.directory/'images'
        self.images = self.data['images']; self.order = list(self.images)
        manifest_path=self.directory/'manifest.json'
        if manifest_path.is_file():
            ordered=list(dict.fromkeys(e['image_id'] for e in photo.read_json(manifest_path).get('exposures',[])))
            if set(ordered)==set(self.images):self.order=ordered
        self.independence = dict(status='NOT_VERIFIED', scope='SHA_IDENTIFICATION_NOT_VISUAL_TRAINING_DISJOINTNESS')
        sources=[self.sort_path]+([self.tv_path] if self.data.get('source_xml_sha256') and self.tv_path.is_file() else [])
        self.proof = {str(p):dict(sha256=photo.digest(p),stamp=photo.stamp(p)) for p in sources}

    @property
    def default_output(self):return self.directory/'reference.json'

    def verify(self, key):
        row=self.images[key];relative=PurePosixPath(row['relative_path'])
        if relative.is_absolute() or '..' in relative.parts or ':' in str(relative) or '\\' in str(relative):raise ValueError('UNSAFE_FROZEN_PATH')
        path=self.root/relative
        if path.is_symlink() or photo.digest(path)!=row['source_file_sha256']:raise ValueError('FROZEN_IMAGE_SHA_MISMATCH')
        for name, proof in self.proof.items():
            if photo.digest(name)!=proof['sha256']:raise ValueError('FROZEN_MANIFEST_CHANGED')
        return path

    def image(self,key):
        with Image.open(self.verify(key)) as image:return ImageOps.exif_transpose(image).convert('RGB')


def display_ready(manifest):
    """V2 presentation needs valid material, independently of reference coverage."""
    errors=manifest['input_errors']
    if manifest.get('schema')==SCHEMA_V2:return not manifest.get('material_errors',[])
    return not errors or (manifest.get('layout_version')=='final_f2_v1' and manifest.get('mode')=='TECHNICAL_NON_FINAL'
        and all(e.startswith('POSITIVE_WITHOUT_PLATE:') for e in errors))


def freeze(audited, root, *, screen=(1920,1080), dwell_seconds=DEFAULT_DWELL_SECONDS, configuration=None, technical=False, layout_version=None, reference_aware=False):
    from .final180_markers import layout, GEOMETRY_POLICY
    dwell_seconds=parse_dwell(dwell_seconds)
    if layout_version is not None:
        from .final180_markers import versioned_layout
        versioned_layout(*screen,version=layout_version)
    if photo.digest(audited['xml_path'])!=audited['xml_sha256']:raise ValueError('AUDIT_XML_CHANGED')
    if reference_aware and not any(r['state']['presence']!='UNRESOLVED' or r['state']['objects'] for r in audited['images']):
        reference_faults=('XML_RECORDS_INVALID_OR_DUPLICATED','XML_IMAGE_NOT_IN_POSITIVE_INVENTORY:','XML_DIMENSIONS_CONFLICT:','UNPARSED_XML_SHAPE:')
        if not any(e.startswith(reference_faults) for e in audited['errors']):
            target=freeze_photographs(audited['positive_dir'],root,screen=screen,dwell_seconds=dwell_seconds,technical=technical,additional_source=audited.get('negative_dir'),layout_version=layout_version)
            doc=photo.read_json(target/'manifest.json');doc['reference_import_note']=dict(source_xml_sha256=audited['xml_sha256'],status='NO_REFERENCE_DECISIONS_IN_XML')
            doc['configuration']=deepcopy(configuration or {})
            _write_plan(target,doc);return target
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    directory=root/('prepared_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    directory.mkdir(exist_ok=False);(directory/'images').mkdir()
    shutil.copy2(audited['xml_path'],directory/'source_annotations.xml')
    images, exposures = {}, []
    for position, row in enumerate(audited['images'],1):
        path=Path(row['path'])
        if photo.digest(path)!=row['sha256']:raise ValueError('AUDITED_PHOTO_CHANGED:' + str(path))
        key=row['image_id']
        if key not in images:
            name=f'{len(images)+1:04d}'+path.suffix.lower()
            shutil.copy2(path,directory/'images'/name)
            images[key]=dict(image_id=key,relative_path=name,source_relative_path=row['relative_path'],
                source_file_sha256=row['sha256'],size_bytes=row['bytes'],width=row['size'][0],height=row['size'][1],
                exif_orientation=row['exif_orientation'],raw_size=row['raw_size'],
                coordinate_space='exif_transposed_source_pixels')
        exposures.append(dict(exposure_id=f'e{position:04d}',position=position,image_id=key,
            photo_sha256=row['sha256'],source_role=row['source_role']))
    _atomic_write_json(directory/'material.json',dict(schema='alpr.final.frozen_material.v1',images=images,
        source_xml_sha256=audited['xml_sha256'],audit=audited))
    material=FrozenMaterial(directory);reference=photo.PhotoReference(material)
    reference.data['schema']=photo.SCHEMA_V2
    for row in audited['images']:
        key=row['image_id']
        reference.data['reviews'][key]=dict(versions={'xml':dict(state=row['state'],action='IMPORT_SAVED_XML',
            annotation_protocol=REFERENCE_PROTOCOL,predictions_exposed=True)},timeline=['xml'],cursor=0)
    reference.data['import_declarations']=dict(blank_text_unreadable=dict(
        statement='Puste teksty przy wskazanych istniejących ramkach oznaczają nieczytelny numer',
        source_xml_sha256=audited['xml_sha256'],plate_ids=audited['unreadable_plate_ids']),
        negative_images=[r['sha256'] for r in audited['images'] if r['source_role']=='NEGATIVE'] if audited['negative_declared'] else [])
    reference._validate(reference.data)
    _atomic_write_json(reference.path,reference.data)
    document=dict(schema=SCHEMA,contract=CONTRACT,manifest_id=uuid.uuid4().hex,created_utc=utc(),
        mode='TECHNICAL_NON_FINAL' if technical else 'FINAL',dwell_seconds=float(dwell_seconds),
        planned_photo_seconds=len(exposures)*float(dwell_seconds),transition_seconds=.5,countdown_seconds=5,
        material_sha256=photo.digest(directory/'material.json'),reference_sha256=photo.digest(reference.path),
        source_xml_sha256=audited['xml_sha256'],exposures=exposures,layout=layout(*screen),
        policy=deepcopy(POLICY),geometry_policy=deepcopy(GEOMETRY_POLICY),configuration=configuration or {},
        input_audit=audited['summary'],input_errors=audited['errors'],
        acceptance=dict(desktop='PREPARED',android_contract='NOT_VERIFIED',a12_tv='NOT_VERIFIED'),
        status='TECHNICAL_ONLY' if technical and not audited['errors'] else 'NO_GO')
    if layout_version is not None:_f2_layout(document,material,screen,layout_version)
    if technical and display_ready(document):document['status']='TECHNICAL_ONLY'
    if reference_aware:bind_reference_metadata(document,directory)
    _write_plan(directory,document)
    return directory


def load(directory,*,strict_reference=True,reference_path=None):
    directory=Path(directory).resolve(strict=True);doc=photo.read_json(directory/'manifest.json')
    if doc.get('schema') not in (SCHEMA,SCHEMA_V2) or doc.get('contract')!=CONTRACT:raise ValueError('FINAL_MANIFEST_SCHEMA_MISMATCH')
    dwell=parse_dwell(doc['dwell_seconds'])
    if not math.isclose(doc['planned_photo_seconds'],len(doc['exposures'])*dwell,rel_tol=0,abs_tol=1e-9):
        raise ValueError('FROZEN_PLAN_TIME_MISMATCH')
    from .final180_markers import GEOMETRY_POLICY
    if doc.get('policy')!=POLICY or doc.get('geometry_policy')!=GEOMETRY_POLICY:
        raise ValueError('FROZEN_POLICY_DIFFERS_FROM_IMPLEMENTATION')
    if photo.digest(directory/'material.json')!=doc['material_sha256']:raise ValueError('FROZEN_MATERIAL_CHANGED')
    material=FrozenMaterial(directory)
    ids=[e['exposure_id'] for e in doc['exposures']]
    if len(ids)!=len(set(ids)) or [e['position'] for e in doc['exposures']]!=list(range(1,len(ids)+1)):
        raise ValueError('EXPOSURE_ORDER_INVALID')
    for e in doc['exposures']:
        material.verify(e['image_id'])
        if material.images[e['image_id']]['source_file_sha256']!=e['photo_sha256']:raise ValueError('EXPOSURE_SOURCE_MISMATCH')
    if doc['layout'].get('layout_version'):
        expected=deepcopy(doc);_f2_layout(expected,material,(doc['layout']['width'],doc['layout']['height']),doc['layout']['layout_version'])
        if any(doc.get(k)!=expected[k] for k in ('layout','layout_version','qr_version','exposures')):raise ValueError('F2_FROZEN_LAYOUT_OR_PLACEMENT_MISMATCH')
    from . import final180_markers as markers
    if doc['layout'].get('layout_version')==markers.F3_LAYOUT:
        if doc.get('qr_identity_contract')!=expected['qr_identity_contract']:raise ValueError('F3_FROZEN_IDENTITY_CONTRACT_MISMATCH')
        if photo.read_json(directory/'qr_plan_index.json')!=markers.f3_plan_index(doc,photo.digest(directory/'manifest.json')):
            raise ValueError('F3_FROZEN_PLAN_INDEX_MISMATCH')
    reference,error,_,_=_open_reference(directory,doc,material,reference_path)
    if error and strict_reference:raise ValueError(error)
    return doc,material,reference


def _open_reference(directory,doc,material,override=None):
    """One native reader. No file search, synthetic GT, or prediction fallback."""
    path=Path(override).resolve() if override is not None else Path(directory)/'reference.json'
    declared_none=doc.get('reference_mode')=='none'
    if override is None and declared_none:return None,'',None,None
    try:
        if not path.is_file():raise ValueError('REFERENCE_FILE_MISSING')
        sha=photo.digest(path)
        if override is None and sha!=doc.get('reference_sha256'):raise ValueError('FROZEN_REFERENCE_CHANGED')
        if doc.get('source_xml_sha256'):
            xml=Path(directory)/'source_annotations.xml'
            if not xml.is_file() or photo.digest(xml)!=doc['source_xml_sha256']:raise ValueError('FROZEN_XML_CHANGED')
        # A later explicit reference may conservatively record prior access to
        # model results. Never downgrade an already exposed source to blind GT.
        candidate=photo.read_json(path)
        if not isinstance(candidate,dict):raise ValueError('REFERENCE_OBJECT_REQUIRED')
        binding=candidate.get('presentation_binding',{})
        if override is not None and doc.get('material_binding') and binding:
            if (binding.get('material_id')!=doc['material_binding'].get('material_id') or
                    binding.get('manifest_sha256')!=photo.digest(Path(directory)/'manifest.json')):
                raise ValueError('MATERIAL_REFERENCE_BINDING_MISMATCH')
            if candidate.get('annotation_protocol') not in (photo.PROTOCOL,photo.ASSISTED_PROTOCOL,REFERENCE_PROTOCOL):
                raise ValueError('MATERIAL_REFERENCE_PROTOCOL_UNSUPPORTED')
            from copy import copy
            material=copy(material);material.annotation_protocol=candidate['annotation_protocol']
        if override is not None and candidate.get('predictions_exposed') is True and material.predictions_exposed is False:
            from copy import copy
            material=copy(material);material.predictions_exposed=True
        reference=photo.PhotoReference(material,path)
        if photo.digest(path)!=sha:raise ValueError('REFERENCE_CHANGED_DURING_READ')
        return reference,'',sha,str(path)
    except (ValueError,KeyError,TypeError,OSError,AttributeError,IndexError) as exc:return None,str(exc),None,str(path)


def reference_preflight(directory,*,reference_path=None):
    """Native completeness plus immutable image identity; separate from MT evidence."""
    doc,material,_=load(directory,strict_reference=False,reference_path=reference_path)
    reference,error,sha,path=_open_reference(directory,doc,material,reference_path)
    return _reference_coverage(doc,reference,error,sha,path,override=reference_path is not None)


def _reference_coverage(doc,reference,error,sha,path,*,override=False):
    from collections import Counter
    legacy=(doc.get('schema')==SCHEMA and not doc.get('reference_binding')) or doc.get('reference_binding',{}).get('origin')=='VALIDATED_LEGACY_FROZEN_REFERENCE'
    rows=[];seen={};conflicts=[];sha_states={}
    if reference:
        integrity=('XML_RECORDS_INVALID_OR_DUPLICATED','XML_IMAGE_NOT_IN_POSITIVE_INVENTORY:','PLATE_ID_MISSING_OR_DUPLICATED:',
            'INVALID_GEOMETRY:','XML_DIMENSIONS_CONFLICT:','UNPARSED_XML_SHAPE:','SAME_BYTES_CONFLICTING_REFERENCE:')
        conflicts.extend(x for x in doc.get('input_errors',[]) if x.startswith(integrity))
    if reference:
        for key in reference.material.order:
            state=reference.state(key)
            if photo.complete(state):
                sha_key=reference.material.images[key].get('source_file_sha256')
                content=fingerprint(dict(presence=state['presence'],objects=sorted([dict(quad=o['quad'],text=o['gt_text'],kind=photo.text_kind(o),partial=o.get('partial_text','')) for o in state['objects']],key=fingerprint)))
                if sha_key and sha_states.setdefault(sha_key,content)!=content:conflicts.append('SAME_BYTES_CONFLICTING_REFERENCE:'+sha_key)
            for obj in state['objects']:
                previous=seen.setdefault(obj['object_id'],key)
                if previous!=key:conflicts.append('DUPLICATE_REFERENCE_OBJECT_ID:'+obj['object_id'])
    if conflicts:error='; '.join(conflicts)
    for e in doc['exposures']:
        state=reference.state(e['image_id']) if reference is not None else None
        objects=state['objects'] if state else [];reason=error
        if error:status='GT_CONFLICT'
        elif state is None or (state['presence']=='UNRESOLVED' and not objects):status='GT_MISSING';reason='NO_REFERENCE_FOR_PHOTOGRAPH'
        else:
            review=reference.data['reviews'].get(e['image_id'],{});current=(review.get('versions') or {}).get((review.get('timeline') or [''])[review.get('cursor',0)],{})
            imported=any(v.get('action')=='IMPORT_SAVED_XML' for v in review.get('versions',{}).values())
            declared_negative=state['presence']=='NO_PLATES' and e['photo_sha256'] in reference.data.get('import_declarations',{}).get('negative_images',[])
            if imported and not legacy and not declared_negative:status='GT_INCOMPLETE';reason='XML_SCENE_COMPLETENESS_NOT_CONFIRMED'
            elif photo.complete(state) and all(photo.text_kind(o)!='FULL_TEXT' or normalize_registration(o['gt_text']) for o in objects):status='GT_COMPLETE';reason=''
            else:status='GT_INCOMPLETE';reason='UNRESOLVED_REFERENCE_DECISIONS'
        rows.append(dict(exposure_id=e['exposure_id'],image_id=e['image_id'],photo_sha256=e['photo_sha256'],status=status,reason=reason,
            plate_count=len(objects) if status=='GT_COMPLETE' else None,known_objects=len(objects) if state else None,
            no_plates=bool(status=='GT_COMPLETE' and state['presence']=='NO_PLATES'),
            full_text=sum(photo.text_kind(o)=='FULL_TEXT' for o in objects) if status=='GT_COMPLETE' else None,
            unreadable=sum(photo.text_kind(o)=='UNREADABLE_TEXT' for o in objects) if status=='GT_COMPLETE' else None))
    counts=Counter(r['status'] for r in rows)
    declared=doc.get('reference_binding',{}).get('coverage')
    if not override and not error and declared is not None and declared!=rows:
        error='FROZEN_REFERENCE_COVERAGE_MISMATCH'
        for row in rows:row.update(status='GT_CONFLICT',reason=error,plate_count=None,no_plates=False,full_text=None,unreadable=None)
        counts=Counter(r['status'] for r in rows)
    mode='invalid' if counts['GT_CONFLICT'] else 'complete' if rows and counts['GT_COMPLETE']==len(rows) else 'partial' if counts['GT_COMPLETE'] or counts['GT_INCOMPLETE'] else 'none'
    return dict(schema='alpr.final.reference_preflight.v1',mode=mode,reference_sha256=sha,reference_path=path,
        annotation_protocol=reference.annotation_protocol if reference else None,predictions_exposed=reference.predictions_exposed if reference else None,
        reference_schema=reference.data.get('schema') if reference else None,reference_id=reference.data.get('reference_id') if reference else None,reference_revision=reference.data.get('revision') if reference else None,
        provenance='EXPLICIT_ANALYSIS_OVERRIDE' if override else 'FROZEN_MANIFEST',
        counts={k:counts[k] for k in ('GT_COMPLETE','GT_INCOMPLETE','GT_MISSING','GT_CONFLICT')},coverage=rows,
        complete_gt_plates=sum(r['plate_count'] or 0 for r in rows),full_text=sum(r['full_text'] or 0 for r in rows),
        unreadable=sum(r['unreadable'] or 0 for r in rows),negative_images=sum(r['no_plates'] for r in rows),error=error)


def bind_reference_metadata(doc,directory,*,legacy=False):
    """Called only while constructing a NEW plan; never rewrites old material."""
    doc['schema']=SCHEMA_V2
    if legacy:doc['reference_binding']=dict(origin='VALIDATED_LEGACY_FROZEN_REFERENCE')
    material=FrozenMaterial(directory);reference,error,sha,path=_open_reference(directory,doc,material)
    coverage=_reference_coverage(doc,reference,error,sha,path)
    doc['reference_mode']=coverage['mode'];doc['reference_binding']=dict(schema='alpr.final.reference_binding.v1',
        origin='VALIDATED_LEGACY_FROZEN_REFERENCE' if legacy else 'NATIVE_PHOTO_REFERENCE',reference_sha256=sha,
        reference_schema=coverage['reference_schema'],reference_id=coverage['reference_id'],reference_revision=coverage['reference_revision'],
        coverage=coverage['coverage'],counts=coverage['counts'])
    doc['material_errors']=[]
    return coverage


def audit_reference_preflight(audited):
    """Same readiness verifier before freezing an explicitly selected Z2 XML."""
    images={r['image_id']:dict(width=r['size'][0],height=r['size'][1]) for r in audited['images']}
    states={r['image_id']:r['state'] for r in audited['images']}
    adapter=SimpleNamespace(material=SimpleNamespace(order=list(images),images=images),annotation_protocol=REFERENCE_PROTOCOL,predictions_exposed=True,
        state=lambda key:deepcopy(states[key]),data=dict(schema='Z2_XML_PREFLIGHT',reviews={key:dict(cursor=0,timeline=['xml'],versions={'xml':dict(action='IMPORT_SAVED_XML',state=state)}) for key,state in states.items()},
        import_declarations=dict(negative_images=[r['sha256'] for r in audited['images'] if r['state']['presence']=='NO_PLATES'])))
    error=''
    try:
        for key,state in states.items():photo.PhotoReference._validate_state(adapter,key,state)
    except (ValueError,KeyError,TypeError) as exc:error=str(exc)
    doc=dict(schema=SCHEMA_V2,input_errors=audited['errors'],exposures=[dict(exposure_id=f'e{i:04d}',image_id=r['image_id'],photo_sha256=r['sha256']) for i,r in enumerate(audited['images'],1)])
    return _reference_coverage(doc,adapter,error,audited['xml_sha256'],audited['xml_path'])


def working_reference(directory,target,*,source_reference=None,predictions_exposed=False):
    """Native editable draft outside the immutable preparation; source stays locked."""
    from copy import copy
    directory=Path(directory).resolve(strict=True);target=Path(target).resolve()
    if target.is_relative_to(directory) or target.exists() or target.suffix!='.json':raise ValueError('GT_WORK_COPY_REQUIRES_NEW_EXTERNAL_JSON')
    _,material,_=load(directory,strict_reference=False)
    material=copy(material);material.readonly=False
    target.parent.mkdir(parents=True,exist_ok=True)
    if source_reference is not None:
        source=Path(source_reference).resolve(strict=True);before=photo.digest(source)
        with target.open('xb') as stream:stream.write(source.read_bytes())
        if photo.digest(target)!=before or photo.digest(source)!=before:raise ValueError('REFERENCE_CHANGED_DURING_COPY')
    if predictions_exposed:
        material.predictions_exposed=True
        if target.exists():
            data=photo.read_json(target);data['predictions_exposed']=True
            data['review_context']='MANUAL_EDITOR_OPENED_FROM_MODEL_RESULT_REPORT; ORIGINAL_REFERENCE_UNCHANGED'
            _atomic_write_json(target,data)
    return photo.PhotoReference(material,target)


def freeze_photographs(source,root,*,screen,dwell_seconds=DEFAULT_DWELL_SECONDS,technical=False,additional_source=None,configuration=None,layout_version=None,source_material=None):
    """Presentation without XML or reference.json; absence is metadata, never GT."""
    from . import final180_markers as markers
    if source_material is not None and additional_source is not None:raise ValueError('MATERIAL_REQUIRES_ONE_EXPLICIT_SNAPSHOT')
    if not str(source).strip():raise ValueError('PHOTO_DIRECTORY_REQUIRED')
    source=Path(source).resolve(strict=True);sources=[source]+([Path(additional_source).resolve(strict=True)] if additional_source else [])
    paths=[];seen=set()
    for root_index,source_root in enumerate(sources):
        for path in ([source_material.verify(key) for key in source_material.order] if source_material is not None else files(source_root)):
            if path in seen:continue
            seen.add(path);relative=path.relative_to(source_root).as_posix()
            paths.append((path,(str(root_index)+'/' if len(sources)>1 else '')+relative))
    seconds=parse_dwell(dwell_seconds);version=layout_version or markers.F2_LAYOUT;board=markers.versioned_layout(*screen,version=version)
    if not paths:raise ValueError('NO_PHOTOGRAPHS')
    target=Path(root).resolve()/('prepared_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    target.mkdir(parents=True,exist_ok=False);(target/'images').mkdir();images={};exposures=[]
    for i,(path,relative) in enumerate(paths,1):
        before=photo.stamp(path);sha=photo.digest(path);key='img-sha256-'+sha+'-'+fingerprint(relative)[:12]
        if source_material is not None and key!=source_material.order[i-1]:raise ValueError('MATERIAL_IMAGE_IDENTITY_CHANGED')
        with Image.open(path) as image:raw=list(image.size);orientation=int(image.getexif().get(274,1));w,h=ImageOps.exif_transpose(image).size
        dest=target/'images'/f'{i:04d}{path.suffix.lower()}';shutil.copy2(path,dest)
        if photo.stamp(path)!=before or photo.digest(dest)!=sha:raise ValueError('SOURCE_CHANGED_DURING_FREEZE')
        images[key]=dict(image_id=key,relative_path=dest.name,source_relative_path=relative,source_file_sha256=sha,size_bytes=before[0],width=w,height=h,
            raw_size=raw,exif_orientation=orientation,coordinate_space='exif_transposed_source_pixels')
        exposures.append(dict(position=i,exposure_id=f'e{i:04d}',image_id=key,photo_sha256=sha,source_role='UNKNOWN'))
    _atomic_write_json(target/'material.json',dict(schema='alpr.final.frozen_material.v1',images=images,source_xml_sha256=None,
        annotation_protocol=photo.PROTOCOL,predictions_exposed=False,source_directory=str(source),source_directories=[str(p) for p in sources]))
    doc=dict(schema=SCHEMA_V2,contract=CONTRACT,manifest_id=uuid.uuid4().hex,created_utc=utc(),mode='TECHNICAL_NON_FINAL' if technical else 'FINAL',
        dwell_seconds=seconds,planned_photo_seconds=len(exposures)*seconds,transition_seconds=.5,countdown_seconds=5,
        material_sha256=photo.digest(target/'material.json'),reference_sha256=None,source_xml_sha256=None,reference_mode='none',exposures=exposures,
        layout=board,policy=deepcopy(POLICY),geometry_policy=deepcopy(markers.GEOMETRY_POLICY),configuration=deepcopy(configuration or {}),input_audit=dict(photographs=len(exposures)),input_errors=[],material_errors=[],
        acceptance=dict(desktop='PREPARED',android_contract='NOT_VERIFIED',a12_tv='NOT_VERIFIED'),status='TECHNICAL_ONLY' if technical else 'NO_GO')
    _f2_layout(doc,FrozenMaterial(target),screen,version);bind_reference_metadata(doc,target);_write_plan(target,doc);load(target)
    return target


def _f2_layout(doc,material,screen,version):
    from . import final180_markers as markers
    board=markers.versioned_layout(*screen,version=version)
    doc.update(layout=board,layout_version=version,qr_version=board['qr_version'])
    if version==markers.F3_LAYOUT:
        doc['qr_identity_contract']=dict(schema='alpr.final.f3_pair_contract.v1',left='L+RUN4+PAIR5',right='R+IDX3+PHASE1+PAIR5',
            display_token='ORDERED_ARRAY_LEFT_RIGHT',same_full_source_frame=True,independent_decodes_required=2,
            full_identity='RUN_JSON_MANIFEST_SHA256_AND_OBSERVED_PC_JOURNAL',collision_policy='REJECT_WITHOUT_OVERWRITE',
            test_only='SEPARATE_TECHNICAL_IDENTITY_NOT_A_RESEARCH_RUN')
    else:doc.pop('qr_identity_contract',None)
    for e in doc['exposures']:
        row=material.images[e['image_id']];fit=markers.photo_placement((row['width'],row['height']),doc['layout'])
        e.update(actual_image_rect=fit['rect'],photo_placement=fit)


def _write_plan(directory,doc):
    from . import final180_markers as markers
    _atomic_write_json(Path(directory)/'manifest.json',doc)
    if doc['layout'].get('layout_version')==markers.F3_LAYOUT:
        _atomic_write_json(Path(directory)/'qr_plan_index.json',markers.f3_plan_index(doc,photo.digest(Path(directory)/'manifest.json')))


def relayout(directory,screen,*,root=None,dwell_seconds=None,reference_aware=False,ignore_reference=False,layout_version=None):
    """New F2 plan; preserve every material/GT byte, order, and old run."""
    from . import final180_markers as markers
    directory=Path(directory).resolve(strict=True);parent,material,_=load(directory,strict_reference=not ignore_reference)
    parent_sha=photo.digest(directory/'manifest.json')
    seconds=parse_dwell(parent['dwell_seconds'] if dwell_seconds is None else dwell_seconds)
    version=layout_version or parent['layout'].get('layout_version') or markers.F2_LAYOUT
    board=markers.versioned_layout(*screen,version=version)  # Validate before creating/copying anything.
    target_root=Path(root).resolve() if root is not None else directory.parent
    target=target_root/('prepared_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    target.mkdir(parents=True,exist_ok=False);(target/'images').mkdir()
    for name in ('material.json','reference.json','source_annotations.xml'):
        if ignore_reference and name=='reference.json':continue
        if (directory/name).is_file():shutil.copy2(directory/name,target/name)
    for key,row in material.images.items():
        dest=target/'images'/row['relative_path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(material.verify(key),dest)
        if photo.digest(dest)!=row['source_file_sha256']:raise ValueError('RELAYOUT_IMAGE_COPY_MISMATCH')
    doc=deepcopy(parent);doc.update(manifest_id=uuid.uuid4().hex,created_utc=utc(),dwell_seconds=seconds,planned_photo_seconds=len(parent['exposures'])*seconds,
        parent_plan=dict(path=str(directory),manifest_sha256=parent_sha,manifest_id=parent['manifest_id'],change='LAYOUT_QR_F2_VERSIONED',previous_dwell_seconds=parent['dwell_seconds']),
        acceptance=dict(desktop='PREPARED',android_contract='NOT_VERIFIED',a12_tv='NOT_VERIFIED'),status='TECHNICAL_ONLY' if parent['mode']=='TECHNICAL_NON_FINAL' else 'NO_GO')
    if photo.digest(directory/'manifest.json')!=parent_sha:raise ValueError('RELAYOUT_PARENT_CHANGED')
    _f2_layout(doc,material,screen,version)
    if version==markers.F3_LAYOUT:doc['parent_plan']['change']='LAYOUT_QR_F3_PAIR_VERSIONED'
    if ignore_reference:
        doc.update(reference_mode='none',reference_sha256=None);doc.pop('reference_binding',None)
        doc['parent_plan']['change']='EXPLICIT_REFERENCE_DETACHMENT_AND_VERSIONED_LAYOUT'
    if reference_aware or ignore_reference:
        legacy=not ignore_reference and (parent.get('schema')==SCHEMA or parent.get('reference_binding',{}).get('origin')=='VALIDATED_LEGACY_FROZEN_REFERENCE')
        bind_reference_metadata(doc,target,legacy=legacy)
    _write_plan(target,doc);load(target)
    return target


def replan(directory,dwell_seconds,*,root=None):
    """Version only the timing manifest; copy the exact frozen material/GT bytes."""
    seconds=parse_dwell(dwell_seconds);directory=Path(directory).resolve(strict=True)
    parent,material,_=load(directory)
    if seconds==parent['dwell_seconds']:return directory
    parent_sha=photo.digest(directory/'manifest.json')
    target_root=Path(root).resolve() if root is not None else directory.parent
    target_root.mkdir(parents=True,exist_ok=True)
    target=target_root/('prepared_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    target.mkdir(exist_ok=False);(target/'images').mkdir()
    for name in ('material.json','reference.json','source_annotations.xml'):
        if (directory/name).is_file():shutil.copy2(directory/name,target/name)
    for key,row in material.images.items():
        source=material.verify(key);dest=target/'images'/row['relative_path'];dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        if photo.digest(dest)!=row['source_file_sha256']:raise ValueError('REPLAN_IMAGE_COPY_MISMATCH')
    doc=deepcopy(parent)
    doc.update(manifest_id=uuid.uuid4().hex,created_utc=utc(),dwell_seconds=seconds,
        planned_photo_seconds=len(parent['exposures'])*seconds,
        parent_plan=dict(path=str(directory),manifest_sha256=parent_sha,manifest_id=parent['manifest_id'],
            change='OPERATOR_DWELL_SECONDS',previous_dwell_seconds=parent['dwell_seconds']))
    if photo.digest(directory/'manifest.json')!=parent_sha:raise ValueError('REPLAN_PARENT_CHANGED')
    _write_plan(target,doc)
    load(target)
    return target
