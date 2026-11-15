# Producer F5A/12A validator adapted to native Desktop with explicit errors.
# Contract SHA256: 78bd4dc0a1f4c1f8101f1a33b17a7ec3a53ee510f92ca2bf5433d5af18d0fa59
"""Strict validation of optional F5A producer evidence in native Desktop."""
import hashlib
import json
import struct
from pathlib import PurePosixPath

def safe_path(value):
    if not (isinstance(value, str) and value):
        raise ValueError('Missing image path')
    path = PurePosixPath(value)
    if not (not path.is_absolute() and (not any((p in {'', '.', '..'} for p in value.split('/'))))):
        raise ValueError('Unsafe evidence path')
    if not ('\\' not in value and ':' not in value and ('\x00' not in value)):
        raise ValueError('Unsafe evidence path')
    return value

def jpeg_dimensions(data):
    if not data[:2] == b'\xff\xd8':
        raise ValueError('Invalid JPEG signature')
    pos = 2
    while pos < len(data):
        if not data[pos] == 255:
            raise ValueError('Invalid JPEG marker')
        while pos < len(data) and data[pos] == 255:
            pos += 1
        if not pos < len(data):
            raise ValueError('Truncated JPEG')
        marker = data[pos]
        pos += 1
        if marker in {216, 1} or 208 <= marker <= 215:
            continue
        if not marker not in {217, 218}:
            raise ValueError('JPEG dimensions missing')
        if not pos + 2 <= len(data):
            raise ValueError('Truncated JPEG')
        size = struct.unpack_from('>H', data, pos)[0]
        if not (size >= 2 and pos + size <= len(data)):
            raise ValueError('Truncated JPEG segment')
        if marker in {192, 193, 194, 195, 197, 198, 199, 201, 202, 203, 205, 206, 207}:
            if not size >= 8:
                raise ValueError('Invalid JPEG SOF')
            height, width = struct.unpack_from('>HH', data, pos + 3)
            if not (width > 0 and height > 0):
                raise ValueError('Invalid JPEG dimensions')
            return (width, height)
        pos += size
    raise ValueError('JPEG dimensions missing')

def integer(value):
    if not (type(value) is int and 0 <= value < 2 ** 63):
        raise ValueError('Invalid int64')
    return value

def stamp(value):
    if not isinstance(value, dict):
        raise ValueError('Missing source stamp')
    for key in ('scene_generation', 'visual_epoch', 'camera_transform_generation', 'source_sequence', 'source_timestamp_nanos'):
        integer(value[key])
    if not value['source_timestamp_domain'] in {'CAMERAX_SENSOR', 'PREVIEW_INHERITED_CAMERA', 'RUNTIME_UPTIME', 'UNKNOWN'}:
        raise ValueError('F5A evidence contract violation')

def ids(value):
    if not (isinstance(value, list) and all((isinstance(x, str) and x for x in value))):
        raise ValueError('Invalid identity array')
    if not len(set(value)) == len(value):
        raise ValueError('Duplicate identity')
    return set(value)

def _validate_contract(archive, descriptor, session, attempts, f6=None):
    caps = descriptor.get('capabilities', {})
    if not caps.get('scene_evidence_v1') and (not caps.get('mt_detection_roles_v1')):
        return {'available': False}
    if not (caps.get('scene_evidence_v1') is True and caps.get('mt_detection_roles_v1') is True):
        raise ValueError('F5A evidence contract violation')
    names = archive.namelist()
    if not len(names) == len(set(names)):
        raise ValueError('Duplicate ZIP entry')
    for name in names:
        safe_path(name.rstrip('/'))
    if not archive.testzip() is None:
        raise ValueError('ZIP CRC mismatch')
    if not descriptor['scene_evidence_file'] == 'samples/scene_evidence.jsonl':
        raise ValueError('F5A evidence contract violation')
    if not descriptor['mt_detection_roles_file'] == 'samples/mt_detection_roles.jsonl':
        raise ValueError('F5A evidence contract violation')

    def rows(name):
        if f6 and f6.get('available'):
            from .mobile_f6_execution_evidence import checked_rows
            return list(checked_rows(archive,name,json.loads(archive.read('manifest.json'))))
        info = archive.getinfo(name)
        if not info.file_size <= 32 * 1024 * 1024:
            raise ValueError('Evidence index too large')
        content = archive.read(name).decode('utf-8')
        if not not content.startswith('\ufeff'):
            raise ValueError('Unexpected BOM')
        values = [json.loads(line) for line in content.splitlines() if line.strip()]
        if not all((isinstance(row, dict) for row in values)):
            raise ValueError('Invalid evidence row')
        return values
    scenes = rows(descriptor['scene_evidence_file'])
    roles = rows(descriptor['mt_detection_roles_file'])
    scene_ids = ids([row['scene_event_id'] for row in scenes])
    ids([row['attempt_id'] for row in attempts])
    invocations = {}
    for row in attempts:
        if str(row['mt_executed']).lower() == 'true':
            invocations.setdefault(row['mt_invocation_id'], []).append(row)
    role_ids = ids([row['mt_invocation_id'] for row in roles])
    if not role_ids <= invocations.keys():
        raise ValueError('Role references a nonexistent MT')
    written = 0
    scene_calls = set()
    for row in scenes:
        if not (row['schema'] == 'alpr.scene_evidence.v1' and row['session_id'] == session['session_id']):
            raise ValueError('F5A evidence contract violation')
        if not row['event_type'] in {'SESSION_START', 'SCENE_BOUNDARY'}:
            raise ValueError('F5A evidence contract violation')
        if not row['scene_event_id'].startswith(session['session_id'] + '-s'):
            raise ValueError('F5A evidence contract violation')
        if not (isinstance(row['event_reason'], str) and row['event_reason']):
            raise ValueError('F5A evidence contract violation')
        if not row['settling_status'] == 'NOT_ASSESSED':
            raise ValueError('F5A evidence contract violation')
        if not row['source_frame_coordinate_space'] == 'mt_source_bitmap_xy':
            raise ValueError('F5A evidence contract violation')
        if not row['source_frame_origin'] == 'pipeline_bitmap_before_roi':
            raise ValueError('F5A evidence contract violation')
        integer(row['event_elapsed_nanos'])
        if row['event_stamp'] is not None:
            stamp(row['event_stamp'])
        if row['source_stamp'] is not None:
            stamp(row['source_stamp'])
        if row['write_status'] == 'WRITTEN':
            written += 1
            if not row['missing_reason'] is None:
                raise ValueError('F5A evidence contract violation')
            entry = safe_path(row['source_frame_entry'])
            if not (entry.startswith('samples/frames/') and entry.endswith('.jpg')):
                raise ValueError('F5A evidence contract violation')
            if archive.getinfo(entry).file_size > 20 * 1024 * 1024:
                raise ValueError('Scene JPEG exceeds size limit')
            payload = archive.read(entry)
            if not hashlib.sha256(payload).hexdigest() == row['source_frame_sha256']:
                raise ValueError('Scene SHA mismatch')
            if not jpeg_dimensions(payload) == (row['source_frame_width_px'], row['source_frame_height_px']):
                raise ValueError('Scene dimension mismatch')
            stamp(row['source_stamp'])
            event, source = (row['event_stamp'], row['source_stamp'])
            for key in ('scene_generation', 'visual_epoch', 'camera_transform_generation', 'source_timestamp_domain'):
                if not source[key] == event[key]:
                    raise ValueError('Scene source identity conflict')
            if not source['source_sequence'] >= event['source_sequence']:
                raise ValueError('Stale scene image')
            integer(row['capture_elapsed_nanos'])
            integer(row['capture_duration_nanos'])
        else:
            if not (row['write_status'] == 'MISSING' and row['missing_reason']):
                raise ValueError('Missing image needs a reason')
            if not (row['source_frame_entry'] is None and row['source_frame_sha256'] is None):
                raise ValueError('F5A evidence contract violation')
        calls = ids(row['mt_invocation_ids'])
        known_starts = set(invocations) | (set(f6.get('mt_invocations', {})) if f6 and f6.get('available') else set())
        if not calls <= known_starts:
            raise ValueError('Scene references a nonexistent MT')
        if not not scene_calls & calls:
            raise ValueError('Invocation assigned to multiple scenes')
        scene_calls |= calls
        if not row['mt_execution_status'] in {'EXECUTED', 'NOT_RUN', 'UNKNOWN'}:
            raise ValueError('F5A evidence contract violation')
        if not bool(calls) == (row['mt_execution_status'] == 'EXECUTED'):
            raise ValueError('F5A evidence contract violation')
        if row['mt_execution_status'] == 'NOT_RUN':
            if not row['mt_not_run_reason']:
                raise ValueError('NOT_RUN needs a reason')
        elif not row['mt_not_run_reason'] is None:
            raise ValueError('F5A evidence contract violation')
        if not row['links_status'] in {'COMPLETE', 'INCOMPLETE'}:
            raise ValueError('F5A evidence contract violation')
        for mp in row['mp_observations']:
            needed=set(('source_stamp trace_frame_id execution_status selected_rois').split())
            if mp.get('execution_status') in ('EXECUTED','NOT_RUN'):
                needed.update(('raw_count','rejected_class_count','rejected_class_ids'))
            if mp.get('execution_status')=='NOT_RUN':needed.add('not_run_reason')
            missing=needed-set(mp)
            if missing:
                raise ValueError('MP_OBSERVATION_FIELDS_MISSING: '+','.join(sorted(missing)))
            stamp(mp['source_stamp'])
            integer(mp['trace_frame_id'])
            if not mp['execution_status'] in {'EXECUTED', 'NOT_RUN', 'FAILED'}:
                raise ValueError('F5A evidence contract violation')
            if mp['execution_status'] == 'EXECUTED':
                integer(mp['raw_count'])
                integer(mp['rejected_class_count'])
                if not len(mp['rejected_class_ids']) == mp['rejected_class_count'] <= mp['raw_count']:
                    raise ValueError('F5A evidence contract violation')
                if not all((type(x) is int for x in mp['rejected_class_ids'])):
                    raise ValueError('F5A evidence contract violation')
            elif mp['execution_status'] == 'NOT_RUN':
                if not (mp['not_run_reason'] and mp['raw_count'] is None and (mp['rejected_class_count'] is None) and (mp['rejected_class_ids'] is None)):
                    raise ValueError('F5A evidence contract violation')
            for roi in mp['selected_rois']:
                for key in ('left', 'top', 'right', 'bottom', 'entity_id', 'vehicle_track_id'):
                    integer(roi[key])
                if not (roi['right'] > roi['left'] and roi['bottom'] > roi['top']):
                    raise ValueError('F5A evidence contract violation')
    if not session['scene_evidence_expected'] == len(scenes):
        raise ValueError('F5A evidence contract violation')
    if not session['scene_evidence_written'] == written:
        raise ValueError('F5A evidence contract violation')
    if not session['scene_evidence_missing'] == len(scenes) - written:
        raise ValueError('F5A evidence contract violation')
    if session['scene_evidence_complete']:
        if not (written == len(scenes) > 0 and session['scene_evidence_integrity_loss_count'] == 0):
            raise ValueError('F5A evidence contract violation')
        if not all((row['links_status'] == 'COMPLETE' for row in scenes)):
            raise ValueError('F5A evidence contract violation')
    role_complete = True
    for row in roles:
        gid = row['mt_invocation_id']
        if not (row['schema'] == 'alpr.mt_detection_roles.v1' and row['session_id'] == session['session_id']):
            raise ValueError('F5A evidence contract violation')
        stamp(row['source_stamp'])
        if not (row['raw_scope'] == 'DECODER_OUTPUT' and row['role_scope'] == 'MERGED_MT_DEDUP'):
            raise ValueError('F5A evidence contract violation')
        if not row['postprocess_scope_id'] == session['session_id'] + '/frame-' + str(integer(row['trace_frame_id'])):
            raise ValueError('F5A evidence contract violation')
        if not row['postprocess_status'] in {'COMPLETED', 'BYPASSED', 'NOT_REACHED'}:
            raise ValueError('F5A evidence contract violation')
        if not row['role_proof_status'] in {'COMPLETE', 'INCOMPLETE'}:
            raise ValueError('F5A evidence contract violation')
        if not (row['scene_event_id'] in scene_ids or (row['scene_event_id'] is None and row['role_proof_status'] == 'INCOMPLETE')):
            raise ValueError('F5A evidence contract violation')
        source_rows = invocations[gid]
        for source in source_rows:
            for key, value in row['source_stamp'].items():
                if not str(source[key]) == str(value):
                    raise ValueError('Role stamp conflict')
            if not [int(source['roi_' + key]) for key in ('left', 'top', 'right', 'bottom')] == row['roi']:
                raise ValueError('Role ROI conflict')
            if not source['source_frame_entry'] == (row['source_frame_entry'] or ''):
                raise ValueError('Role image conflict')
        if row['source_frame_entry'] in names:
            width, height = jpeg_dimensions(archive.read(safe_path(row['source_frame_entry'])))
            if not (width == int(source_rows[0]['source_frame_width_px']) and height == int(source_rows[0]['source_frame_height_px'])):
                raise ValueError('Invocation image dimensions conflict')
            left, top, right, bottom = row['roi']
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ValueError('ROI outside its source bitmap')
        expected = {r['attempt_id'] for r in source_rows if r['mt_detection_index'] != ''}
        final, suppressed = (ids(row['final_attempt_ids']), ids(row['suppressed_attempt_ids']))
        rejected = ids([item['attempt_id'] for item in row['rejected_before_nms']])
        unresolved = ids(row['unresolved_attempt_ids'])
        partitions = [final, suppressed, rejected, unresolved]
        if not all((item['reason'] == 'INSUFFICIENT_KEYPOINTS' for item in row['rejected_before_nms'])):
            raise ValueError('F5A evidence contract violation')
        if not all((str(source.get('mz_executed')).lower() != 'true' and source.get('evidence_kind') != 'plate_crop' for source in source_rows if source['attempt_id'] in suppressed)):
            raise ValueError('Suppressed candidate has downstream MZ evidence')
        if not sum(map(len, partitions)) == len(set.union(*partitions)):
            raise ValueError('Role populations overlap')
        if not set.union(*partitions) <= expected:
            raise ValueError('Foreign attempt in roles')
        if row['raw_attempt_ids'] is not None:
            if not ids(row['raw_attempt_ids']) == expected == set.union(*partitions):
                raise ValueError('Incomplete RAW population')
        if row['role_proof_status'] == 'COMPLETE':
            if not (row['raw_attempt_ids'] is not None and (not unresolved) and (row['postprocess_status'] != 'NOT_REACHED')):
                raise ValueError('F5A evidence contract violation')
        else:
            role_complete = False
        if row['stale_or_cancelled']:
            if not row['cancel_reason']:
                raise ValueError('F5A evidence contract violation')
    if session['mt_detection_roles_complete']:
        if not (role_ids == invocations.keys() == scene_calls and role_complete):
            raise ValueError('F5A evidence contract violation')
    manifest = json.loads(archive.read('manifest.json'))
    report = json.loads(archive.read('report.json'))['research_collection']
    for key in ('scene_evidence_expected', 'scene_evidence_written', 'scene_evidence_missing', 'scene_evidence_integrity_loss_count', 'scene_evidence_complete', 'mt_detection_roles_complete'):
        if not manifest[key] == session[key] == report[key]:
            raise ValueError('Extension summary conflict')
    return {'available': True, 'scenes': len(scenes), 'written': written, 'missing': len(scenes) - written, 'mt_role_records': len(roles), 'scene_evidence_complete': session['scene_evidence_complete'], 'mt_detection_roles_complete': session['mt_detection_roles_complete']}

CONTRACT_SHA256 = '78bd4dc0a1f4c1f8101f1a33b17a7ec3a53ee510f92ca2bf5433d5af18d0fa59'
SUMMARY_FIELDS = ('scene_evidence_expected', 'scene_evidence_written', 'scene_evidence_missing',
                  'scene_evidence_integrity_loss_count', 'scene_evidence_complete', 'mt_detection_roles_complete')
STAMP_FIELDS = ('scene_generation', 'visual_epoch', 'camera_transform_generation', 'source_sequence',
                'source_timestamp_nanos', 'source_timestamp_domain')

def _require(condition, reason):
    if not condition:
        raise ValueError(reason)

def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON field: ' + key)
        result[key] = value
    return result

def _checked_bytes(archive, name, manifest, limit):
    name = safe_path(name)
    info = archive.getinfo(name)
    _require(info.file_size <= limit, 'F5A evidence exceeds size limit: ' + name)
    payload = archive.read(info)
    digest = manifest.get('entry_sha256', {}).get(name)
    _require(isinstance(digest, str) and len(digest) == 64
             and hashlib.sha256(payload).hexdigest() == digest, 'F5A manifest SHA mismatch: ' + name)
    return payload

def _index(archive, name, manifest):
    payload = _checked_bytes(archive, name, manifest, 32 * 1024 * 1024)
    text = payload.decode('utf-8')
    _require(not text.startswith('\ufeff'), 'F5A JSONL BOM forbidden')
    return [json.loads(line, object_pairs_hook=_unique_fields) for line in text.splitlines() if line.strip()]

def read_extension(archive, descriptor, session, manifest, report, *, f6=None):
    """Native reader extension; absent legacy capability carries no scene claim."""
    import csv
    import io
    import re
    from datetime import datetime
    from PIL import Image
    caps = descriptor.get('capabilities') or {}
    declared = [caps.get(k, False) for k in ('scene_evidence_v1', 'mt_detection_roles_v1')]
    _require(all(type(v) is bool for v in declared), 'F5A capability must be boolean')
    if not any(declared):
        return dict(available=False, status='ABSENT', scenes=(), roles=())
    _require(all(declared), 'F5A v1 producer declares scene and role capabilities together')
    _require(session.get('experiment_type') == 'roi_budget', 'F5A extension outside roi_budget')
    active_f6=bool(f6 and f6.get('available'))
    if active_f6:
        from .mobile_f6_execution_evidence import checked_hash, checked_rows
        checked_hash(archive,'samples/attempts.csv',manifest)
        with archive.open('samples/attempts.csv') as stream:
            attempts=list(csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8-sig')))
        scenes=list(checked_rows(archive,descriptor['scene_evidence_file'],manifest))
        roles=list(checked_rows(archive,descriptor['mt_detection_roles_file'],manifest))
    else:
        attempts_text = _checked_bytes(archive, 'samples/attempts.csv', manifest, 32 * 1024 * 1024).decode('utf-8-sig')
        attempts = list(csv.DictReader(io.StringIO(attempts_text)))
        scenes = _index(archive, descriptor['scene_evidence_file'], manifest)
        roles = _index(archive, descriptor['mt_detection_roles_file'], manifest)
    summary = _validate_contract(archive, descriptor, session, attempts, f6=f6)
    pending=session.get('scene_evidence_pending_loss_unknown', (session.get('f6_evidence') or {}).get('pending_loss_unknown') if active_f6 else None)
    _require(type(pending) is bool, 'Missing pending-loss qualification')
    for key in SUMMARY_FIELDS:
        value = session.get(key)
        if key.endswith('_complete'):
            _require(type(value) is bool, 'Invalid extension completion flag')
        else:
            integer(value)
        _require(manifest.get(key) == value == report.get('research_collection', {}).get(key), 'Extension summary conflict')
    _require(session['scene_evidence_written'] + session['scene_evidence_missing'] == session['scene_evidence_expected'], 'Scene denominator conflict')
    if pending:
        _require(not session['scene_evidence_complete'], 'Unknown pending loss cannot be complete')
    by_call = {}
    last = -1
    for row in scenes:
        required = set(('schema session_id scene_event_id event_type event_reason transition_revision event_stamp source_stamp observed_at_utc event_elapsed_nanos capture_elapsed_nanos capture_duration_nanos source_frame_entry source_frame_sha256 source_frame_width_px source_frame_height_px source_frame_coordinate_space source_frame_origin write_status missing_reason settling_status mp_observations mt_invocation_ids mt_execution_status mt_not_run_reason links_status').split())
        _require(required <= row.keys(), 'Incomplete scene record')
        match = re.fullmatch(re.escape(session['session_id']) + r'-s(\d{8,})', row['scene_event_id'])
        _require(match is not None and int(match[1]) > last, 'Scene IDs are not unique monotonic producer IDs')
        last = int(match[1])
        if row['transition_revision'] is not None:
            integer(row['transition_revision'])
        observed = datetime.fromisoformat(row['observed_at_utc'].replace('Z', '+00:00'))
        _require(observed.utcoffset() is not None and observed.utcoffset().total_seconds() == 0, 'Scene observation needs UTC')
        for key in ('source_stamp', 'event_stamp'):
            if row[key] is not None:
                _require(set(row[key]) == set(STAMP_FIELDS), 'Unexpected source stamp fields')
        if row['event_type'] == 'SESSION_START':
            _require(row['event_reason'] == 'session_start', 'Invalid SESSION_START reason')
        if row['write_status'] == 'WRITTEN':
            _require(row['source_frame_entry'] == 'samples/frames/scene-' + row['scene_event_id'] + '.jpg', 'Scene image identity path conflict')
            data = _checked_bytes(archive, row['source_frame_entry'], manifest, 20 * 1024 * 1024)
            width, height = row['source_frame_width_px'], row['source_frame_height_px']
            _require(type(width) is int and type(height) is int and 0 < max(width, height) <= 12000
                     and width * height <= 40_000_000, 'Unsafe scene dimensions')
            with Image.open(io.BytesIO(data)) as image:
                _require(image.format == 'JPEG' and image.size == (width, height), 'Scene JPEG mismatch')
                image.load()
        for mp in row['mp_observations']:
            _require(set(mp['source_stamp']) == set(STAMP_FIELDS), 'Invalid MP stamp')
            for roi in mp['selected_rois']:
                if mp['source_stamp'] == row['source_stamp'] and row['write_status'] == 'WRITTEN':
                    _require(0 <= roi['left'] < roi['right'] <= row['source_frame_width_px']
                             and 0 <= roi['top'] < roi['bottom'] <= row['source_frame_height_px'], 'MP ROI outside its bitmap')
        for call in row['mt_invocation_ids']:
            by_call[call] = row['scene_event_id']
    verified_images = set()
    for role in roles:
        _require(set(role['source_stamp']) == set(STAMP_FIELDS), 'Invalid invocation stamp fields')
        _require(type(role['stale_or_cancelled']) is bool, 'Invalid cancellation flag')
        _require(all(isinstance(role[k], str) for k in ('cancel_reason', 'execution_error', 'processing_error')), 'Invalid execution qualification')
        _require(type(role['roi']) is list and len(role['roi']) == 4
                 and all(type(v) is int for v in role['roi']), 'Invalid invocation ROI')
        if role['scene_event_id'] is not None:
            _require(by_call.get(role['mt_invocation_id']) == role['scene_event_id'], 'Scene/invocation link conflict')
        if role['postprocess_status'] == 'NOT_REACHED':
            _require(not role['final_attempt_ids'] and not role['suppressed_attempt_ids'], 'NOT_REACHED cannot claim FINAL')
        entry = role['source_frame_entry']
        members = [r for r in attempts if r['mt_invocation_id'] == role['mt_invocation_id']]
        written = any(r.get('source_frame_status') == 'WRITTEN' for r in members)
        if written:
            _require(entry in archive.namelist(), 'Declared WRITTEN invocation image missing')
        if entry and entry in archive.namelist() and entry not in verified_images:
            data = _checked_bytes(archive, entry, manifest, 20 * 1024 * 1024)
            with Image.open(io.BytesIO(data)) as image:
                width,height=image.size
                _require(image.format=='JPEG' and max(width,height)<=12000 and width*height<=40_000_000,
                         'Invalid invocation JPEG')
                image.load()
            verified_images.add(entry)
    summary.update(status='VALID' if summary['scene_evidence_complete'] and summary['mt_detection_roles_complete'] else 'PARTIAL',
                   scenes=tuple(scenes), roles=tuple(roles), contract_sha256=CONTRACT_SHA256,
                   pending_loss_unknown=pending)
    return summary
