"""Native TV_BBOX_CROP v1 evidence validation; no inference or TV homography.

Wire decimals are retained for SetupJson/BigDecimal canonical hashes. Float32
rounding below validates the producer's translation; it never rewrites MT/GT.
"""
from __future__ import annotations

import csv
from datetime import datetime
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import zipfile

from .mobile_scene_evidence import safe_path, stamp, integer, _unique_fields

MODE = 'TV_BBOX_CROP'
FULL = 'CAMERA_FULL_FRAME'
CAPABILITY = 'tv_bbox_crop_v1'
SCHEMA = 'alpr.tv_bbox_crop.v1'
ROW_SCHEMA = 'alpr.tv_bbox_input.v1'
INDEX = 'samples/tv_bbox_inputs.jsonl'
EXPORT_FILE = 'tv_bbox_input_evidence.json'
NAMESPACE = 'tv_bbox_input_evidence'
EXPORT_SCHEMA = 'alpr.desktop_tv_bbox_input_evidence.v1'
STAMP = ('scene_generation', 'visual_epoch', 'camera_transform_generation',
         'source_sequence', 'source_timestamp_nanos', 'source_timestamp_domain')
CAMERA_SPACE, LOCAL_SPACE = 'camera_full_bitmap_xy', 'tv_bbox_local_xy'


def require(ok, reason):
    if not ok:
        raise ValueError('TV_BBOX_INPUT_INVALID: ' + reason)


def loads(value):
    def constant(token):
        raise ValueError('Non-finite JSON: ' + token)
    return json.loads(value, parse_float=Decimal, parse_constant=constant, object_pairs_hook=_unique_fields)


def plain(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


def quote(value):
    # JSONObject.quote, including its non-ASCII escapes and </ handling.
    out, previous = [], ''
    escapes = {'"': '\\"', '\\': '\\\\', '\b': '\\b', '\t': '\\t',
               '\n': '\\n', '\f': '\\f', '\r': '\\r'}
    for ch in value:
        n = ord(ch)
        if ch in escapes:
            out.append(escapes[ch])
        elif ch == '/' and previous == '<':
            out.append('\\/')
        elif n < 32 or 0x80 <= n < 0xa0 or 0x2000 <= n < 0x2100:
            out.append('\\u%04x' % n)
        else:
            out.append(ch)
        previous = ch
    return '"' + ''.join(out) + '"'


def canonical(value):
    """SetupJson.canonical: sorted UTF-16 keys, plain BigDecimal, UTF-8 JSON."""
    if value is None:
        return 'null'
    if type(value) is bool:
        return 'true' if value else 'false'
    if type(value) is int:
        return str(value)
    if isinstance(value, (Decimal, float)):
        number = value if isinstance(value, Decimal) else Decimal(str(value))
        require(number.is_finite() and math.isfinite(float(number)) and abs(number.adjusted()) <= 1000,
                'Unsafe canonical number')
        if number == 0:
            return '0'
        text = format(number, 'f')
        return text.rstrip('0').rstrip('.') if '.' in text else text
    if isinstance(value, str):
        return quote(value)
    if isinstance(value, list):
        return '[' + ','.join(canonical(v) for v in value) + ']'
    require(isinstance(value, dict) and all(isinstance(k, str) for k in value), 'Canonical object')
    return '{' + ','.join(quote(k) + ':' + canonical(value[k])
        for k in sorted(value, key=lambda k: k.encode('utf-16-be', 'surrogatepass'))) + '}'


def digest(value):
    require(isinstance(value, dict), 'Digest object')
    return hashlib.sha256(canonical({k: v for k, v in value.items() if k != 'digest_sha256'}).encode('utf-8')).hexdigest()


def number(value):
    require(type(value) in (int, float, Decimal) and math.isfinite(float(value)), 'Finite number required')
    return float(value)


def int32(value, *, positive=False):
    require(type(value) is int and (0 < value < 2**31 if positive else 0 <= value < 2**31), 'Integer dimension/boundary')
    return value


def utc(value):
    require(isinstance(value, str), 'UTC timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0, 'UTC timestamp')
    return parsed


def geometry(value):
    require(isinstance(value, dict), 'Source geometry')
    for key in ('source_image_width', 'source_image_height', 'buffer_width', 'buffer_height', 'preview_width', 'preview_height'):
        require(int32(value[key], positive=True) <= 32768, 'Geometry dimension')
    w, h = value['source_image_width'], value['source_image_height']
    require(max(w, h) <= 12000 and w * h <= 40_000_000, 'Source image budget')
    for key in ('rotation_degrees', 'display_rotation_degrees'):
        require(type(value[key]) is int and value[key] in (0, 90, 180, 270), 'Camera rotation')
    for key in ('binding_generation', 'viewport_revision'):
        integer(value[key])
    require(isinstance(value['camera_id'], str) and value['camera_id']
            and isinstance(value['lens_facing'], str) and value['lens_facing'], 'Camera identity')
    require(value.get('physical_camera_id') is None or isinstance(value['physical_camera_id'], str), 'Physical camera ID')
    require(number(value['zoom_ratio']) > 0 and type(value['mirrored']) is bool, 'Camera zoom/mirroring')
    c, m, u = value['crop_rect_px'], value['source_to_view_matrix'], value['usable_view_rect']
    require(isinstance(c, list) and len(c) == 4 and isinstance(m, list) and len(m) == 9
            and isinstance(u, list) and len(u) == 4, 'Geometry arrays')
    c, m, u = list(map(number, c)), list(map(number, m)), list(map(number, u))
    require(0 <= c[0] < c[2] <= value['buffer_width'] and 0 <= c[1] < c[3] <= value['buffer_height'], 'Camera cropRect')
    a, b = c[2] - c[0], c[3] - c[1]
    if value['rotation_degrees'] % 180:
        a, b = b, a
    require((w, h) == (math.floor(a + .5), math.floor(b + .5)), 'CameraX crop/rotation dimensions')
    determinant = m[0]*(m[4]*m[8]-m[5]*m[7])-m[1]*(m[3]*m[8]-m[5]*m[6])+m[2]*(m[3]*m[7]-m[4]*m[6])
    require(math.isfinite(determinant) and abs(determinant) >= 1e-12, 'Singular source-to-view matrix')
    require(value['mirrored'] == (m[0]*m[4]-m[1]*m[3] < 0), 'Mirroring conflict')
    require(0 <= u[0] < u[2] <= value['preview_width'] and 0 <= u[1] < u[3] <= value['preview_height'], 'Usable preview rectangle')
    projected = []
    for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
        z = m[6]*x+m[7]*y+m[8]
        require(math.isfinite(z) and abs(z) >= 1e-12, 'Singular preview mapping')
        projected.append(((m[0]*x+m[1]*y+m[2])/z, (m[3]*x+m[4]*y+m[5])/z))
    require(u[0] >= min(p[0] for p in projected)-1e-4 and u[2] <= max(p[0] for p in projected)+1e-4
            and u[1] >= min(p[1] for p in projected)-1e-4 and u[3] <= max(p[1] for p in projected)+1e-4,
            'Usable rectangle outside source pixels')


def same_geometry(a, b):
    # Match CameraPreviewGeometry.sameGeometry, including its numeric tolerances.
    exact = ('source_image_width', 'source_image_height', 'buffer_width', 'buffer_height', 'rotation_degrees',
             'display_rotation_degrees', 'camera_id', 'physical_camera_id', 'lens_facing', 'binding_generation',
             'preview_width', 'preview_height', 'mirrored')
    return (all(a.get(k) == b.get(k) for k in exact) and abs(number(a['zoom_ratio'])-number(b['zoom_ratio'])) < 1e-5
            and a['crop_rect_px'] == b['crop_rect_px']
            and all(abs(number(x)-number(y)) <= 1e-4 for k in ('source_to_view_matrix', 'usable_view_rect') for x, y in zip(a[k], b[k])))


def validate_descriptor(value):
    require(isinstance(value, dict) and value.get('schema') == SCHEMA and digest(value) == value.get('digest_sha256'), 'Descriptor digest')
    require(value.get('input_mode') == MODE and value.get('processing_transform') == 'CROP_ONLY_TRANSLATION'
            and value.get('H_TV') == 'NOT_APPLIED', 'TV transform/homography')
    require(value.get('source_coordinate_space') == CAMERA_SPACE and value.get('input_coordinate_space') == LOCAL_SPACE,
            'Descriptor coordinate spaces')
    c, g, setup = value['calibration_snapshot'], value['source_geometry'], value['camera_setup']
    require(isinstance(setup, dict), 'Camera setup object')
    require(isinstance(c, dict) and digest(c) == c.get('digest_sha256') == value['calibration_digest_sha256'], 'Calibration digest')
    require(c.get('schema') == 'alpr.tv_setup_calibration.v1' and c.get('status') == 'READY'
            and c.get('calibration_source') == 'operator_manual_tv_corners'
            and c.get('coordinate_space') == 'source_frame_normalized_v1'
            and c.get('corner_order') == 'TL_TR_BR_BL', 'Approved calibration required')
    utc(c['confirmed_at_utc'])
    utc(c['frozen_at_utc'])
    require(c.get('reason_codes') == [], 'READY calibration has invalidation reasons')
    geometry(g)
    geometry(c['source_geometry'])
    require(same_geometry(g, c['source_geometry']), 'Stale calibration geometry')
    require(setup.get('schema') == 'alpr.camera_setup.v1' and setup.get('logical_camera_id') == g['camera_id']
            and setup.get('physical_camera_id') == g.get('physical_camera_id')
            and abs(number(setup['zoom_ratio'])-number(g['zoom_ratio'])) < 1e-5, 'Frozen camera_setup conflict')
    w, h = int32(value['source_width_px'], positive=True), int32(value['source_height_px'], positive=True)
    require((w, h) == (g['source_image_width'], g['source_image_height']), 'Descriptor source dimensions')
    points = value['corners_norm']
    require(points == c['corners_norm'] and isinstance(points, list) and len(points) == 4, 'Calibration corners')
    xy = [(number(p['x']), number(p['y'])) for p in points]
    require(all(0 <= x <= 1 and 0 <= y <= 1 for x, y in xy), 'Corners outside source')
    for i in range(4):
        a, b, d = xy[i], xy[(i+1)%4], xy[(i+2)%4]
        require((b[0]-a[0])*(d[1]-b[1])-(b[1]-a[1])*(d[0]-b[0]) > 1e-12, 'Convex TL/TR/BR/BL order')
    expected = (max(0, math.floor(min(p[0] for p in xy)*w)), max(0, math.floor(min(p[1] for p in xy)*h)),
                min(w, math.ceil(max(p[0] for p in xy)*w)), min(h, math.ceil(max(p[1] for p in xy)*h)))
    box = tuple(int32(value['tv_bbox_px'][k]) for k in ('left', 'top', 'right', 'bottom'))
    require(box == expected and box[2] > box[0] and box[3] > box[1], 'BBOX floor/ceil boundaries')
    require((int32(value['input_width_px'], positive=True), int32(value['input_height_px'], positive=True))
            == (box[2]-box[0], box[3]-box[1]), 'Crop dimensions')
    require((int32(value['offset_to_camera_full']['x']), int32(value['offset_to_camera_full']['y'])) == box[:2], 'BBOX offset')
    return value


def f32(value):
    return struct.unpack('>f', struct.pack('>f', number(value)))[0]


def validate_detection(local, attempt, frozen):
    require(isinstance(local, dict) and local.get('coordinate_space') == LOCAL_SPACE, 'Local MT geometry space')
    box, points = local['bbox'], local['corners_px']
    require(isinstance(box, list) and len(box) == 4 and isinstance(points, list), 'Local MT geometry')
    l, t, r, b = map(number, box)
    require(0 <= l < r <= frozen['input_width_px'] and 0 <= t < b <= frozen['input_height_px'], 'Local MT bbox bounds')
    ox, oy = frozen['offset_to_camera_full']['x'], frozen['offset_to_camera_full']['y']
    for i, key in enumerate(('plate_left', 'plate_top', 'plate_right', 'plate_bottom')):
        require(f32(f32(box[i]) + (ox if i % 2 == 0 else oy)) == f32(attempt[key]), 'MT bbox missing/double offset')
    translated = []
    for point in points:
        require(isinstance(point, list) and len(point) == 2, 'Local MT point')
        require(0 <= number(point[0]) <= frozen['input_width_px'] and 0 <= number(point[1]) <= frozen['input_height_px'], 'Local MT point bounds')
        translated.append((f32(f32(point[0])+ox), f32(f32(point[1])+oy)))
    if attempt.get('mt_pose_status') == 'VALID_QUAD':
        w, h = frozen['source_width_px'], frozen['source_height_px']
        require(attempt.get('mt_pose_coordinate_space') == 'source_frame_normalized_v1'
                and attempt.get('mt_pose_corner_order') == 'TL_TR_BR_BL'
                and attempt.get('mt_pose_source_width_px') == w and attempt.get('mt_pose_source_height_px') == h,
                'MT Pose full-source dimensions/space')
        encoded = [tuple(map(float, p.split(':'))) for p in attempt['mt_pose_corners_norm'].split(';')]
        require(len(encoded) == len(translated) == 4, 'MT Pose point count')
        remaining = [(x/w, y/h) for x, y in translated]
        for point in encoded:
            require(len(point) == 2 and all(math.isfinite(v) and 0 <= v <= 1 for v in point), 'Full MT Pose')
            matches = [p for p in remaining if all(abs(a-b) <= 2*math.ulp(b) for a, b in zip(point, p))]
            require(len(matches) == 1, 'MT Pose missing/double offset')
            remaining.remove(matches[0])


def validate_letterbox(attempt, roi_width, roi_height):
    """Check captured preprocessing metadata; raw MT outputs are not regenerated."""
    width, height = attempt.get('input_width'), attempt.get('input_height')
    if not width or not height:
        require(attempt.get('mt_detection_index') is None, 'Detection without input dimensions')
        return  # failed/NOT_RUN preprocessing may have no model input evidence
    int32(width, positive=True)
    int32(height, positive=True)
    scale = min(f32(width/roi_width), f32(height/roi_height))
    padx = f32(f32(width-f32(roi_width*scale))/2)
    pady = f32(f32(height-f32(roi_height*scale))/2)
    require(f32(attempt['input_scale']) == scale and f32(attempt['input_pad_x']) == padx
            and f32(attempt['input_pad_y']) == pady, 'Inverse-letterbox metadata conflict')


def absent():
    return dict(available=False, status='ABSENT', rows=[], dependencies={})


def read_extension(archive, descriptor, session, manifest, report, *, f6=None):
    caps = descriptor.get('capabilities') or {}
    require(isinstance(caps, dict) and type(caps.get(CAPABILITY, False)) is bool, 'TV capability boolean')
    names = archive.namelist()
    require(len(names) == len(set(names)), 'Duplicate ZIP entry')
    claimed = (session.get('input_mode') == MODE or 'tv_bbox_crop' in session or INDEX in names
               or (session.get('execution') or {}).get('input_mode') == MODE
               or (report.get('research_collection') or {}).get('input_mode') == MODE)
    if not caps.get(CAPABILITY, False):
        require(not claimed, 'Undeclared TV BBOX input')
        return absent()
    require(descriptor.get('tv_bbox_inputs_file') == INDEX, 'TV index path')
    deps = {}
    active_f6=bool(f6 and f6.get('available'))

    def read(name, limit=32*1024*1024):
        safe_path(name)
        info = archive.getinfo(name)
        require(0 <= info.file_size <= limit, 'Missing/oversized entry: ' + name)
        data = archive.read(info)
        hashed = hashlib.sha256(data).hexdigest()
        require(len(data) == info.file_size and manifest.get('entry_sha256', {}).get(name) == hashed, 'Manifest SHA/size: ' + name)
        deps[name] = dict(sha256=hashed, bytes=len(data))
        return data

    def document(name):
        return loads(read(name).decode('utf-8-sig'))

    def rows(name):
        if active_f6:
            from .mobile_f6_execution_evidence import checked_rows, loss_declared
            values=list(checked_rows(archive,name,manifest,loader=loads,
                allow_malformed=name=='samples/attempts.jsonl' and loss_declared(session)))
            deps[name]=dict(sha256=manifest['entry_sha256'][name],bytes=archive.getinfo(name).file_size)
            return values
        text = read(name).decode('utf-8')
        require(not text.startswith('\ufeff'), 'JSONL BOM')
        values = [loads(line) for line in text.splitlines() if line.strip()]
        require(all(isinstance(v, dict) for v in values), 'JSONL row object')
        return values

    require(plain(document('session.json')) == session and plain(document('samples/schema.json')) == descriptor, 'Metadata conflict')
    s, r = document('session.json'), document('report.json')['research_collection']
    require(s['input_mode'] == MODE and s['experiment_type'] == 'roi_budget', 'TV input mode/experiment')
    frozen = validate_descriptor(s['tv_bbox_crop'])
    execution = s['execution']
    require(execution['input_mode'] == MODE and canonical(execution['tv_bbox_crop']) == canonical(frozen), 'Execution descriptor conflict')
    require(execution['feature_flags']['autozoom'] is False, 'TV input requires frozen zoom / AZ OFF')
    require(canonical(execution['camera_setup']) == canonical(frozen['camera_setup']), 'Execution camera_setup')
    require(s['tv_bbox_crop_frozen_at_utc'] == s['started_at'], 'START freeze binding')
    utc(s['started_at'])
    require(utc(frozen['calibration_snapshot']['confirmed_at_utc']) <= utc(s['started_at']), 'Calibration confirmed after START')
    if 'tv_bbox_crop_frozen_at_utc' in execution:
        require(execution['tv_bbox_crop_frozen_at_utc'] == s['started_at'], 'Execution freeze binding')
    for key in ('input_mode', 'tv_bbox_crop', 'tv_bbox_crop_frozen_at_utc', 'tv_bbox_input_evidence', 'tv_bbox_inputs_pending', 'execution'):
        require(canonical(s.get(key)) == canonical(r.get(key)), 'Research collection conflict: ' + key)
    require(plain(r) == report.get('research_collection'), 'Research metadata conflict')
    require(type(s['tv_bbox_inputs_pending']) is bool, 'Pending flag')
    attempts, scenes, roles = {}, {}, {}
    for dest, name, key in ((attempts, 'samples/attempts.jsonl', 'attempt_id'),
                            (scenes, 'samples/scene_evidence.jsonl', 'scene_event_id'),
                            (roles, 'samples/mt_detection_roles.jsonl', 'mt_invocation_id')):
        for row in rows(name):
            identity = row.get(key)
            require(isinstance(identity, str) and identity and identity not in dest and row.get('session_id') == s['session_id'], 'Source identity: ' + key)
            dest[identity] = row
    if active_f6:
        from .mobile_f6_execution_evidence import checked_hash
        deps['samples/attempts.csv']=checked_hash(archive,'samples/attempts.csv',manifest)
        with archive.open('samples/attempts.csv') as stream:
            csv_rows=list(csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8-sig')))
    else:
        csv_rows = list(csv.DictReader(io.StringIO(read('samples/attempts.csv').decode('utf-8-sig'))))
    require(len(csv_rows) == len(attempts) and {a['attempt_id'] for a in csv_rows} == set(attempts), 'Attempt CSV identities')
    for row in csv_rows:
        for key in ('session_id', 'mt_invocation_id', *STAMP):
            require(row.get(key, '') == str(attempts[row['attempt_id']].get(key, '')), 'Attempt CSV identity: ' + key)
    records, keys, linked = rows(INDEX), set(), set()
    require(active_f6 or len(records) <= 4096, 'TV frame limit')
    written = missing = 0
    for row in records:
        require(row['schema'] == ROW_SCHEMA and row['session_id'] == s['session_id'], 'TV frame schema/session')
        require(row['tv_bbox_crop_digest_sha256'] == frozen['digest_sha256'] and row['processing_transform'] == 'CROP_ONLY_TRANSLATION'
                and row['H_TV'] == 'NOT_APPLIED', 'Frame transform/descriptor')
        st = row['source_stamp']
        stamp(st)
        require(set(st) == set(STAMP), 'Full SourceStamp fields')
        key = digest(dict(source_stamp=st, snapshot=frozen['digest_sha256']))
        require(row['frame_key'] == key and key not in keys, 'Frame key/SourceStamp or duplicate index')
        keys.add(key)
        require(type(row['stale_or_cancelled']) is bool and isinstance(row['cancel_reason'], str)
                and isinstance(row['pipeline_status'], str) and row['pipeline_status'], 'Frame status/cancellation')
        for name in ('crop_prepare_elapsed_nanos', 'captured_elapsed_nanos'):
            integer(row[name])
        scene_id = row.get('scene_event_id')
        if scene_id is not None:
            require(isinstance(scene_id, str) and scene_id in scenes, 'Unknown scene event')
            event = scenes[scene_id].get('event_stamp')
            if event:
                require(all(event[k] == st[k] for k in STAMP[:3]), 'Scene generation conflict')
        rejected = row['pipeline_status'] == 'REJECTED_GEOMETRY'
        for field, space, origin, width, height, prefix in (
            ('source_image', CAMERA_SPACE, 'camera_bitmap_after_camerax_crop_rotation_before_tv_bbox',
             frozen['source_width_px'], frozen['source_height_px'], 'tv-camera-'),
            ('crop_image', LOCAL_SPACE, 'crop_only_translation_from_camera_bitmap',
             frozen['input_width_px'], frozen['input_height_px'], 'tv-input-')):
            image = row[field]
            require(image['encoding'] == 'JPEG94' and image['exact_model_input'] is False
                    and image['coordinate_space'] == space and image['origin'] == origin, 'Image context/coordinate space')
            if 'write_elapsed_nanos' in image:
                integer(image['write_elapsed_nanos'])
            if image['status'] == 'MISSING':
                require(image['entry'] is None and image['sha256'] is None
                        and isinstance(image['missing_reason'], str) and image['missing_reason'], 'Missing image reason')
                require((image['width_px'] is None) == (image['height_px'] is None), 'Missing image dimensions')
                if image['width_px'] is not None:
                    int32(image['width_px'], positive=True)
                    int32(image['height_px'], positive=True)
                    require((field == 'source_image' and rejected) or (image['width_px'], image['height_px']) == (width, height), 'Missing image dimensions conflict')
                missing += 1
                continue
            require(image['status'] == 'WRITTEN' and image['missing_reason'] is None, 'Unfinished/final image status')
            expected = 'samples/frames/' + prefix + key + '.jpg'
            shared = field == 'crop_image' and frozen['input_width_px'] == frozen['source_width_px'] and frozen['input_height_px'] == frozen['source_height_px']
            require(image['entry'] == expected or (shared and image['entry'] == row['source_image']['entry']), 'Image identity path')
            data = read(image['entry'], 20*1024*1024)
            require(hashlib.sha256(data).hexdigest() == image['sha256'], 'TV image SHA')
            from PIL import Image
            with Image.open(io.BytesIO(data)) as decoded:
                require(decoded.format == 'JPEG' and max(decoded.size) <= 12000 and decoded.width*decoded.height <= 40_000_000, 'JPEG format/size')
                require(decoded.size == (int32(image['width_px'], positive=True), int32(image['height_px'], positive=True)), 'JPEG metadata dimensions')
                require((field == 'source_image' and rejected) or decoded.size == (width, height), 'Frozen image dimensions')
                decoded.load()
            written += 1
        require(isinstance(row['attempt_links'], list) and (not rejected or not row['attempt_links']), 'Rejected geometry invoked MT')
        for link in row['attempt_links']:
            aid = link['attempt_id']
            require(aid in attempts and aid not in linked, 'Unknown/duplicate attempt link')
            linked.add(aid)
            a = attempts[aid]
            require(a['tv_bbox_frame_key'] == key and a['tv_bbox_coordinate_space'] == LOCAL_SPACE, 'Attempt/frame/space binding')
            if scene_id is not None and a.get('f5a_scene_event_id'):
                require(a.get('f5a_scene_event_id') == scene_id, 'Attempt scene identity conflict')
            stamp(a)
            require(all(a[k] == st[k] for k in STAMP), 'Attempt SourceStamp conflict')
            for field in ('mt_invocation_id', 'mt_detection_index', 'mt_executed'):
                require(a.get(field) == link.get(field), 'MT identity conflict: ' + field)
            require(type(link['mt_executed']) is bool and isinstance(link['mt_invocation_id'], str), 'MT identity types')
            if link['mt_detection_index'] is not None:
                integer(link['mt_detection_index'])
            roi = a['tv_bbox_roi_local_xy']
            require(roi == link['roi_local_xy'], 'ROI link conflict')
            l, t, right, bottom = [int32(roi[k]) for k in ('left', 'top', 'right', 'bottom')]
            require(l < right <= frozen['input_width_px'] and t < bottom <= frozen['input_height_px'], 'ROI outside TV')
            ox, oy = frozen['offset_to_camera_full']['x'], frozen['offset_to_camera_full']['y']
            require([a['roi_' + k] for k in ('left', 'top', 'right', 'bottom')] == [l+ox, t+oy, right+ox, bottom+oy], 'ROI translation')
            if link['mt_executed']:
                validate_letterbox(a, right-l, bottom-t)
            require(a.get('tv_bbox_local_geometry') == link.get('local_geometry'), 'Local geometry link')
            if link['mt_detection_index'] is not None:
                validate_detection(a.get('tv_bbox_local_geometry'), a, frozen)
            if link['mt_executed']:
                gid = link['mt_invocation_id']
                require(gid in roles and link['role_index_file'] == 'samples/mt_detection_roles.jsonl', 'Unknown MT role identity')
                require(roles[gid]['source_stamp'] == st, 'Role SourceStamp conflict')
                if scene_id is not None and roles[gid].get('scene_event_id'):
                    require(roles[gid]['scene_event_id'] == scene_id, 'Role scene identity conflict')
                if link['mt_detection_index'] is not None:
                    require(aid in (roles[gid].get('raw_attempt_ids') or []), 'Detection absent from source RAW')
            else:
                require(link['mt_invocation_id'] == '' and link['mt_detection_index'] is None, 'Invented MT invocation')
    require(linked == set(attempts), 'Attempt without TV input proof')
    summary = s['tv_bbox_input_evidence']
    require(summary['schema'] == ROW_SCHEMA, 'TV summary schema')
    for field, count in (('frames', len(records)), ('images_written', written), ('images_missing', missing)):
        require(type(summary[field]) is int and summary[field] == count, 'TV summary count: ' + field)
    require(type(summary['complete']) is bool and type(summary['pending_loss_unknown']) is bool, 'TV summary flags')
    require(not s['tv_bbox_inputs_pending'] or summary['pending_loss_unknown'], 'Pending loss omitted')
    require(not summary['complete'] or (records and missing == 0 and not summary['pending_loss_unknown']), 'False complete claim')
    if 'samples/tv_bbox_inputs_journal.jsonl' in names:
        if active_f6:
            from .mobile_f6_execution_evidence import checked_hash
            deps['samples/tv_bbox_inputs_journal.jsonl']=checked_hash(archive,'samples/tv_bbox_inputs_journal.jsonl',manifest)
        else:read('samples/tv_bbox_inputs_journal.jsonl')  # recovery log is never a substitute for final index
    return dict(available=True, status='VALID' if summary['complete'] else 'PARTIAL',
                descriptor=plain(frozen), rows=plain(records), summary=plain(summary), dependencies=deps)


def input_context(session, evidence):
    if not evidence.get('available'):
        return {}
    d, execution = evidence['descriptor'], session.get('execution') or {}
    return dict(input_mode=MODE, status=evidence['status'], series_id=session.get('series_id'),
        variant=execution.get('variant'), roi_budget_policy=execution.get('roi_budget_policy'),
        descriptor_sha256=d['digest_sha256'], calibration_sha256=d['calibration_digest_sha256'],
        camera_setup=d['camera_setup'], source_geometry=d['source_geometry'], tv_bbox_px=d['tv_bbox_px'],
        offset_to_camera_full=d['offset_to_camera_full'], frozen_at_utc=session['tv_bbox_crop_frozen_at_utc'],
        models=execution.get('stages') or {}, source_coordinate_space=CAMERA_SPACE)


def cohort_conflict(contexts):
    contexts = list(contexts)
    modes = {c.get('input_mode', FULL) for c in contexts}
    if len(modes) > 1:
        return 'INPUT_MODE_COHORT_MISMATCH'
    if modes == {MODE}:
        keys = set()
        active_mp = set()
        for context in contexts:
            models = context.get('models') or {}
            model_key = {k: models.get(k) for k in ('mt', 'mz')}
            if (models.get('mp') or {}).get('enabled') is True:
                active_mp.add(canonical(models['mp']))
            keys.add(canonical(dict(snapshot=context.get('descriptor_sha256'), series=context.get('series_id'), models=model_key)))
        if len(keys) > 1 or len(active_mp) > 1 or any(not c.get('descriptor_sha256') for c in contexts):
            return 'TV_BBOX_FROZEN_INPUT_COHORT_MISMATCH'
    return None


def diagnostic_text(evidence):
    if not evidence.get('available'):
        return ''
    d = evidence['descriptor']
    lines = [f"TV_BBOX_CROP: {evidence['status']} · pełna kamera pozostaje przestrzenią GT.",
        f"BBOX {d['tv_bbox_px']} · offset {d['offset_to_camera_full']} · crop {d['input_width_px']}×{d['input_height_px']}",
        'CROP_ONLY_TRANSLATION · H_TV=NOT_APPLIED · JPEG94: kontekst · E2E TV: SCENE_UNMAPPED / N/A',
        'Descriptor SHA-256: ' + d['digest_sha256'], 'Kalibracja SHA-256: ' + d['calibration_digest_sha256']]
    for row in evidence['rows']:
        lines.append(f"{row['frame_key']} · {row['pipeline_status']} · kamera={row['source_image']['status']} / TV={row['crop_image']['status']} · links={len(row['attempt_links'])}")
    return '\n'.join(lines)


def decorate_conditions(session, info):
    evidence = getattr(session, 'tv_bbox_evidence', {})
    if evidence.get('available'):
        info['input_mode'] = MODE
        info['input_context'] = input_context(session.bundle.collection_session, evidence)
    return info


def export(session, directory):
    evidence = session.tv_bbox_evidence
    if not evidence.get('available'):
        return []
    directory, paths, deps = Path(directory), [], {}
    with zipfile.ZipFile(session.path) as archive:
        data = archive.read('manifest.json')
        originals = dict(evidence['dependencies'], **{'manifest.json': dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))})
        for name, expected in originals.items():
            data = archive.read(name)
            require(len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], 'Source dependency changed')
            target = directory / NAMESPACE / safe_path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            paths.append(target)
            deps[target.relative_to(directory).as_posix()] = expected
    doc = dict(schema=EXPORT_SCHEMA, session_id=session.session_id, source_archive_sha256=session.review.source_archive_sha256,
               review_revision=session.review.review_revision, status=evidence['status'], input_mode=MODE, dependencies=deps)
    target = directory / EXPORT_FILE
    target.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
    return paths + [target]


def load_export(directory, manifest, source_session):
    directory = Path(directory)
    path = directory / EXPORT_FILE
    covered = {r['path'] for r in manifest.get('files', [])}
    expected = (source_session or {}).get('input_mode') == MODE or 'tv_bbox_crop' in (source_session or {})
    if not path.exists():
        require(not expected and EXPORT_FILE not in covered and not (directory / NAMESPACE).exists(), 'Missing exported TV evidence')
        return absent()
    require(EXPORT_FILE in covered, 'Unmanifested TV descriptor')
    doc = loads(path.read_text(encoding='utf-8'))
    require(doc['schema'] == EXPORT_SCHEMA and doc['input_mode'] == MODE and all(doc.get(k) == manifest.get(k)
        for k in ('session_id', 'source_archive_sha256', 'review_revision')), 'Export TV identity')
    payloads = {}
    for name, value in doc['dependencies'].items():
        safe_path(name)
        target = directory / name
        require(name.startswith(NAMESPACE + '/') and name in covered and target.is_file() and not target.is_symlink()
                and target.stat().st_size == value['bytes'] and 0 <= value['bytes'] <= 32*1024*1024, 'TV dependency closure/size')
        data = target.read_bytes()
        require(hashlib.sha256(data).hexdigest() == value['sha256'], 'TV dependency SHA')
        payloads[name[len(NAMESPACE)+1:]] = data
    actual = {f.relative_to(directory).as_posix() for f in (directory / NAMESPACE).rglob('*') if f.is_file()}
    require(actual == set(doc['dependencies']), 'Unreferenced TV dependency')

    class Archive:
        def namelist(self): return list(payloads)
        def getinfo(self, name): return SimpleNamespace(filename=name, file_size=len(payloads[name]))
        def read(self, name): return payloads[name if isinstance(name, str) else name.filename]

    session, descriptor, report = [plain(loads(payloads[n])) for n in ('session.json', 'samples/schema.json', 'report.json')]
    require(source_session == session, 'Review/source TV session conflict')
    evidence = read_extension(Archive(), descriptor, session, plain(loads(payloads['manifest.json'])), report)
    require(evidence['status'] == doc['status'] and evidence['available'], 'Export TV status conflict')
    require(set(payloads) == set(evidence['dependencies']) | {'manifest.json'}, 'Export TV dependency closure')
    return evidence
