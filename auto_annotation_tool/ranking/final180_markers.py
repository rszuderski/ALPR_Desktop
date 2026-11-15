"""Standard OpenCV ArUco geometry + versioned QR exposure identity.

Decodes recorded full camera images offline. It does not recognize plates,
modify inference pixels, align clocks, or use GT text for identification.
"""
import json
import math
import hashlib
import base64
import re
import cv2
import numpy as np
from PIL import Image, ImageOps

GEOMETRY_POLICY = dict(id='f6_aruco4_redundant_corners_v1', dictionary='DICT_4X4_50',
    marker_ids=[0,1,2,3], min_marker_edge_camera_px=18, max_reprojection_px=2.5,
    max_leave_one_marker_out_px=4.0, max_normalized_condition=10000,
    min_photo_camera_area_fraction=.02, marker_crop_clearance_px=2,
    detector_min_marker_distance_rate=.05,corner_refinement='SUBPIX',
    calibration='FIXED_BEFORE_FINAL; SYNTHETIC_TESTED; A12_TV_NOT_VERIFIED')

F2_LAYOUT='final_f2_v1'
F2_SAFE_LAYOUT='final_f2_v2_roi_gap'
F2_CLEAR_LAYOUT='final_f2_v3_marker_clearance'
F2_LAYOUTS=(F2_LAYOUT,F2_SAFE_LAYOUT,F2_CLEAR_LAYOUT)
F2_EDGE_LAYOUTS=(F2_SAFE_LAYOUT,F2_CLEAR_LAYOUT)
F2_QR='f2_short_v1'
F3_LAYOUT='final_f3_v1_pairqr'
F3_QR='f3_pair_v1'
BOUND_LAYOUTS=(*F2_LAYOUTS,F3_LAYOUT)
EDGE_LAYOUTS=(*F2_EDGE_LAYOUTS,F3_LAYOUT)


def layout_f3(width,height):
    """Keep F2v3 photos/ArUco and enlarge both QR raster and active symbol."""
    board=layout_f2(width,height,version=F2_CLEAR_LAYOUT)
    old=board['qr_rects'];side=35*board['qr']['module_px'];module=side/29
    first=math.ceil(4*module-.5);last=math.ceil(25*module-.5)-1
    if last-first+1<25*board['qr']['module_px']:raise ValueError('F3_ACTIVE_SYMBOL_SHRUNK')
    # Preserve the inner QR edges and the entire 57 px photo clearance at 1080p.
    # Nearest-neighbor module rasterization uses 7/8 px at 1080p: the active
    # 21x21 symbol is 152 px (previously 150), the complete QR is 210 (was 198).
    # Its outer edge stays inside the ArUco quiet-zone envelope, at x=33>29.
    top=(board['height']-side)//2
    rects=[[old[0][2]-side,top,old[0][2],top+side],
           [old[1][0],top,old[1][0]+side,top+side]]
    outer=board['aruco'][0]['rect'][0]-board['aruco'][0]['quiet']
    if rects[0][0]<outer or rects[1][2]>board['width']-outer:
        raise ValueError('F3_QR_EXCEEDS_SAFE_MARKER_ENVELOPE')
    board.update(layout_version=F3_LAYOUT,qr_version=F3_QR,qr_rects=rects,
        qr=dict(version=1,ecc='H',mode='ALPHANUMERIC',max_characters=10,module_count=21,
            quiet_modules=4,raster_modules=29,module_px=module,
            raster_px=side,active_symbol_px=last-first+1,symbol_bounds_px=[first,first,last,last],
            module_pixel_widths=[math.floor(module),math.ceil(module)],rasterization='NEAREST_NO_ANTIALIAS',
            encoder='OPENCV_V1_H_ALPHANUMERIC_BOTH_FORMAT_COPIES_VERIFIED',
            version_evidence='21x21_MODULES; VERSION_BITS_ABSENT_BELOW_VERSION_7'),
        renderer_version='final_f3_pair_renderer_v1',decoder_version='f3_independent_geometry_bound_v1',
        edge_phase_policy='REGISTERED_TEST_ONLY_PREROLL_AND_RUN_BOUND_POSTROLL_T')
    return board


def versioned_layout(width,height,*,version):
    return layout_f3(width,height) if version==F3_LAYOUT else layout_f2(width,height,version=version)


def token_key(value):
    """Hashable pair key; never collapse two F3 symbols to a single read."""
    if isinstance(value,(list,tuple)) and len(value)==2 and all(isinstance(v,str) for v in value):return tuple(value)
    if isinstance(value,str):return value
    raise ValueError('INVALID_DISPLAY_TOKEN_STRUCTURE')


def _f3_slots(manifest,*,test_only=False):
    if test_only:return [(0,'TEST_ONLY','T')]
    exposures=manifest['exposures'];n=len(exposures)
    if n+2>=36**3 or [e['position'] for e in exposures]!=list(range(1,n+1)):
        raise ValueError('F3_INDEX_OUT_OF_RANGE_OR_ORDER')
    return [(0,'COUNTDOWN','T')]+[(e['position'],e['exposure_id'] if phase=='P' else 'BEFORE:'+e['exposure_id'],phase)
        for e in exposures for phase in ('P','T')]+[(n+1,'PREROLL','T'),(n+2,'POSTROLL','T')]


def f3_plan_index(manifest,manifest_sha):
    """Frozen slot index. Actual short codes are assigned only at explicit START."""
    return dict(schema='alpr.final.f3_plan_index.v1',qr_version=F3_QR,manifest_sha256=manifest_sha,
        run_id=None,assignment='FULL_RUN_BINDING_CREATED_AT_EXPLICIT_START',
        entries=[dict(index=i,index36=_index36(i),identity=identity,phase=phase,
            exposure_id=identity if phase=='P' else None) for i,identity,phase in _f3_slots(manifest)],
        test_only=dict(index=0,phase='T',separate_technical_identity=True,research_run=False))


def f3_binding(manifest,manifest_sha,run_id,*,test_only=False):
    if not re.fullmatch('[0-9a-f]{64}',manifest_sha) or not re.fullmatch('[0-9a-f]{32}',run_id):
        raise ValueError('F3_FULL_IDENTITY_INVALID')
    domain='TEST_ONLY' if test_only else 'RUN'
    tag=_short('|'.join(('F3_RUN',domain,manifest_sha,run_id)),4);entries=[];pairs=set()
    for index,identity,phase in _f3_slots(manifest,test_only=test_only):
        pair=_short('|'.join(('F3_PAIR',domain,manifest_sha,run_id,str(index),identity,phase)),5)
        if pair in pairs:raise ValueError('F3_PAIR_TAG_COLLISION')
        pairs.add(pair)
        entries.append(dict(token=['L'+tag+pair,'R'+_index36(index)+phase+pair],pair_tag=pair,
            index=index,identity=identity,phase=phase,exposure_id=identity if phase=='P' else None))
    return dict(qr_version=F3_QR,run_tag=tag,run_id=run_id,manifest_sha256=manifest_sha,
        purpose='TEST_ONLY' if test_only else manifest.get('mode','FINAL'),entries=entries,
        check='F3_DOMAIN_SEPARATED_SHA256_BASE32_RUN20_PAIR25; COLLISIONS_REJECTED; FULL_PC_PROOF_REQUIRED')


def binding(manifest,manifest_sha,run_id,*,test_only=False):
    factory=f3_binding if manifest['layout'].get('layout_version')==F3_LAYOUT else f2_binding
    return factory(manifest,manifest_sha,run_id,test_only=test_only)


def qr_f3(value):
    if not re.fullmatch(r'(L[A-Z2-7]{9}|R[0-9A-Z]{3}[PT][A-Z2-7]{5})',value):raise ValueError('F3_TOKEN_SYNTAX')
    params=cv2.QRCodeEncoder_Params();params.version=1
    params.correction_level=cv2.QRCodeEncoder_CORRECT_LEVEL_H;params.mode=cv2.QRCodeEncoder_MODE_ALPHANUMERIC
    encoded=cv2.QRCodeEncoder_create(params).encode(value)
    if encoded.shape!=(25,25) or any(np.any(edge!=255) for edge in (encoded[:2],encoded[-2:],encoded[:,:2],encoded[:,-2:])):
        raise ValueError('F3_ENCODER_NOT_VERSION_1_OR_BORDER')
    symbol=encoded[2:-2,2:-2];n=21
    vertical=[(i if i<6 else i+1 if i<8 else n-15+i,8) for i in range(15)]
    horizontal=[(8,n-i-1 if i<8 else 7 if i==8 else 14-i) for i in range(15)]
    formats=[]
    for coords in (vertical,horizontal):
        bits=sum(int(symbol[y,x]==0)<<i for i,(y,x) in enumerate(coords))^0x5412
        data=bits>>10;remainder=bits
        for shift in range(14,9,-1):
            if remainder&(1<<shift):remainder^=0x537<<(shift-10)
        if remainder or data>>3!=2:raise ValueError('F3_ENCODER_ECC_NOT_H')
        formats.append(data)
    if formats[0]!=formats[1]:raise ValueError('F3_ENCODER_FORMAT_COPIES_DISAGREE')
    return np.pad(symbol,4,constant_values=255)


def f3_pair_syntax(decoded):
    return (isinstance(decoded,(list,tuple)) and len(decoded)==2
        and all(isinstance(v,str) for v in decoded)
        and bool(re.fullmatch(r'L[A-Z2-7]{9}',decoded[0]))
        and bool(re.fullmatch(r'R[0-9A-Z]{3}[PT][A-Z2-7]{5}',decoded[1])))


def qualify_f3(decoded,manifest,manifest_sha,run_id,context):
    na=lambda reason:dict(status='N/A',reason=reason,pair_verified=False)
    if context is None:return na('F3_VERIFIED_RUN_INDEX_REQUIRED')
    expected=f3_binding(manifest,manifest_sha,run_id)
    if context.get('binding')!=expected:return na('F3_RUN_INDEX_MISMATCH')
    if len(decoded)!=2 or not all(decoded):return na('EXPOSURE_CODE_UNREADABLE')
    left,right=decoded
    if not f3_pair_syntax(decoded):
        return na('F3_LEFT_RIGHT_SYNTAX_OR_POSITION')
    if left[5:]!=right[5:]:return na('F3_PAIR_TAG_MISMATCH')
    key=token_key(decoded)
    if key in {token_key(v) for v in context.get('technical_tokens',[])}:
        return dict(status='N/A',reason='TECHNICAL_TRANSITION_FRAME',pair_verified=True,phase='T',
            identity='TEST_ONLY',technical_purpose='TEST_ONLY')
    if left[1:5]!=expected['run_tag']:return na('FOREIGN_MANIFEST_OR_RUN')
    entry=next((e for e in expected['entries'] if token_key(e['token'])==key),None)
    if entry is None:return na('F3_CHECK_OR_POSITION_MISMATCH')
    if key not in {token_key(v) for v in context.get('observed_tokens',[])}:return na('F3_PAIR_NOT_OBSERVED_IN_PC_LOG')
    evidence=dict(pair_verified=True,phase=entry['phase'],identity=entry['identity'],
        identity_evidence='TWO_DIFFERENT_SPATIALLY_SEPARATE_QR_SAME_SOURCE_FRAME_FULL_PC_BINDING')
    if entry['phase']!='P':return dict(evidence,status='N/A',reason='TECHNICAL_TRANSITION_FRAME',technical_purpose=entry['identity'])
    return dict(evidence,status='PASS',reason='',exposure_id=entry['exposure_id'])


def layout_f2(width,height,*,version=F2_LAYOUT):
    """Independent version; no silent resizing of an existing manifest."""
    width,height=int(width),int(height)
    if version not in F2_LAYOUTS:raise ValueError('UNKNOWN_LAYOUT_VERSION')
    if width<1280 or height<720:raise ValueError('F2_MONITOR_MINIMUM_1280x720')
    scale=min(width/1920,height/1080);side=round(210*scale);top=round(100*scale)
    module=max(4,int(6*scale));qr=33*module;marker=round(120*scale);quiet=max(12,round(16*scale))
    x=(side-marker)//2;y=round(48*scale);qx=(side-qr)//2;qy=(height-qr)//2
    photo_side=round(250*scale) if version==F2_SAFE_LAYOUT else side
    if version==F2_CLEAR_LAYOUT:
        # Align the complete QR (including its quiet zone) with the outer
        # black ArUco edges. Move the photo field too, preserving ROI clearance.
        qx=x;photo_side=round(300*scale)
    board=dict(layout_version=version,qr_version=F2_QR,width=width,height=height,
        photo_rect=[photo_side,top,width-photo_side,height-top],image_inset_px=max(12,round(16*scale)),
        aruco=[dict(id=i,rect=[a,b,a+marker,b+marker],quiet=quiet) for i,(a,b) in enumerate(
            [(x,y),(width-x-marker,y),(width-x-marker,height-y-marker),(x,height-y-marker)])],
        qr_rects=[[qx,qy,qx+qr,qy+qr],[width-qx-qr,qy,width-qx,qy+qr]],qr_positions=['LEFT','RIGHT'],
        qr=dict(version=2,ecc='H',mode='ALPHANUMERIC',module_count=25,quiet_modules=4,raster_modules=33,module_px=module,
            encoder='OPENCV_VERSION_ECC_MODE_ENFORCED_AND_FORMAT_BITS_VERIFIED'),
        renderer_version='final_f2_renderer_v1',decoder_version='f2_independent_geometry_bound_v1',
        background='BLACK',photo_fit='EXIF_TRANSPOSE_CONTAIN_NO_STRETCH_WITH_SAFETY_INSET',
        marker_dictionary='DICT_4X4_50',transition='EXPLICIT_T_PHASE_WITHOUT_PHOTO')
    if version in F2_EDGE_LAYOUTS:
        board['edge_phase_policy']='REGISTERED_TEST_ONLY_PREROLL_AND_RUN_BOUND_POSTROLL_T'
    symbols=[[a['rect'][0]-quiet,a['rect'][1]-quiet,a['rect'][2]+quiet,a['rect'][3]+quiet] for a in board['aruco']]+board['qr_rects']
    regions=[*symbols,board['photo_rect']]
    for i,a in enumerate(regions):
        if not (0<=a[0]<a[2]<=width and 0<=a[1]<a[3]<=height):raise ValueError('F2_LAYOUT_OUTSIDE_SCREEN')
        for b in regions[i+1:]:
            if min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1]):raise ValueError('F2_LAYOUT_COLLISION')
    if qx<4 or width-2*photo_side<640 or height-2*top<400:raise ValueError('F2_UNSAFE_MARGINS')
    return board


def _short(value,length):
    return base64.b32encode(hashlib.sha256(value.encode('utf-8')).digest()).decode('ascii')[:length]


def _index36(value):
    if not 0<=value<36**3:raise ValueError('F2_INDEX_OUT_OF_RANGE')
    chars='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';result=''
    for _ in range(3):result=chars[value%36]+result;value//=36
    return result


def f2_binding(manifest,manifest_sha,run_id,*,test_only=False):
    """Deterministic projection stored in run.json, not a second GT store."""
    if not re.fullmatch('[0-9a-f]{64}',manifest_sha) or not re.fullmatch('[0-9a-f]{32}',run_id):raise ValueError('F2_FULL_IDENTITY_INVALID')
    tag=_short('F2_RUN|'+manifest_sha+'|'+run_id,10);entries=[]
    identities=[(0,'TEST_ONLY','T')] if test_only else [(0,'COUNTDOWN','T')]+[
        (e['position'],e['exposure_id'] if phase=='P' else 'BEFORE:'+e['exposure_id'],phase)
        for e in manifest['exposures'] for phase in ('P','T')]
    if not test_only and manifest['layout'].get('layout_version') in F2_EDGE_LAYOUTS:identities.append((0,'POSTROLL','T'))
    for index,identity,phase in identities:
        check=_short('|'.join(('F2_CHECK',manifest_sha,run_id,identity,phase,tag)),4)
        entries.append(dict(token='F2'+tag+_index36(index)+phase+check,index=index,identity=identity,phase=phase,
            exposure_id=identity if phase=='P' else None))
    return dict(qr_version=F2_QR,run_tag=tag,run_id=run_id,manifest_sha256=manifest_sha,
        purpose='TEST_ONLY' if test_only else manifest.get('mode','FINAL'),entries=entries,
        check='FIRST_20_BITS_SHA256_BASE32_OF_F2_CHECK_PIPE_MANIFEST_PIPE_RUN_PIPE_IDENTITY_PIPE_PHASE_PIPE_RUN10')


def payload(manifest,manifest_sha,run_id,exposure_id,phase,*,test_only=False):
    if manifest['layout'].get('layout_version')==F3_LAYOUT:
        index=f3_binding(manifest,manifest_sha,run_id,test_only=test_only)
        identity=exposure_id if phase=='P' or exposure_id in ('COUNTDOWN','TEST_ONLY','PREROLL','POSTROLL') else 'BEFORE:'+exposure_id
        return next(e['token'] for e in index['entries'] if e['identity']==identity and e['phase']==phase)
    if manifest['layout'].get('layout_version') not in F2_LAYOUTS:return token(manifest_sha,run_id,exposure_id,phase)
    binding=f2_binding(manifest,manifest_sha,run_id,test_only=test_only)
    identity=exposure_id if phase=='P' or exposure_id in ('COUNTDOWN','TEST_ONLY','POSTROLL') else 'BEFORE:'+exposure_id
    return next(e['token'] for e in binding['entries'] if e['identity']==identity and e['phase']==phase)


def qr_f2(payload):
    if not re.fullmatch(r'F2[A-Z2-7]{10}[A-Z0-9]{3}[PT][A-Z2-7]{4}',payload):raise ValueError('F2_TOKEN_SYNTAX')
    params=cv2.QRCodeEncoder_Params();params.version=2;params.correction_level=cv2.QRCodeEncoder_CORRECT_LEVEL_H
    params.mode=cv2.QRCodeEncoder_MODE_ALPHANUMERIC
    encoded=cv2.QRCodeEncoder_create(params).encode(payload)
    if encoded.shape!=(29,29) or np.any(encoded[:2]!=255) or np.any(encoded[-2:]!=255) or np.any(encoded[:,:2]!=255) or np.any(encoded[:,-2:]!=255):raise ValueError('F2_ENCODER_VERSION_OR_BORDER')
    symbol=encoded[2:-2,2:-2]  # OpenCV's 2-module border -> explicit 4-module quiet zone.
    coords=[(i if i<6 else i+1 if i<8 else 25-15+i,8) for i in range(15)]
    bits=sum(int(symbol[y,x]==0)<<i for i,(y,x) in enumerate(coords))^0x5412
    data=bits>>10;remainder=bits
    for shift in range(14,9,-1):
        if remainder&(1<<shift):remainder^=0x537<<(shift-10)
    if remainder or data>>3!=2:raise ValueError('F2_ENCODER_ECC_NOT_H')
    return np.pad(symbol,4,constant_values=255)


def qualify_f2(decoded,manifest,manifest_sha,run_id,context):
    if context is None:return dict(status='N/A',reason='F2_VERIFIED_RUN_INDEX_REQUIRED')
    binding=f2_binding(manifest,manifest_sha,run_id)
    if context.get('binding')!=binding:return dict(status='N/A',reason='F2_RUN_INDEX_MISMATCH')
    if not re.fullmatch(r'F2[A-Z2-7]{10}[A-Z0-9]{3}[PT][A-Z2-7]{4}',decoded):return dict(status='N/A',reason='F2_TOKEN_SYNTAX')
    if decoded in context.get('technical_tokens',[]):return dict(status='N/A',reason='TECHNICAL_TRANSITION_FRAME',technical_purpose='TEST_ONLY')
    if decoded[2:12]!=binding['run_tag']:return dict(status='N/A',reason='FOREIGN_MANIFEST_OR_RUN')
    entry=next((e for e in binding['entries'] if e['token']==decoded),None)
    if entry is None:return dict(status='N/A',reason='F2_CHECK_OR_POSITION_MISMATCH')
    if decoded not in context.get('observed_tokens',[]):return dict(status='N/A',reason='F2_TOKEN_NOT_OBSERVED_IN_PC_LOG')
    if entry['phase']!='P':return dict(status='N/A',reason='TECHNICAL_TRANSITION_FRAME')
    return dict(status='PASS',reason='',exposure_id=entry['exposure_id'],identity_evidence='TWO_SPATIALLY_SEPARATE_QR_CODES_AGREE')


def layout(width, height):
    width,height=int(width),int(height)
    if width<640 or height<480:raise ValueError('MONITOR_TOO_SMALL')
    border=max(110,int(min(width,height)*.22))
    marker=int(border*.60);qr=int(border*.90);quiet=max(12,marker//4);margin=quiet+8
    corners=[(margin,margin),(width-marker-margin,margin),
             (width-marker-margin,height-marker-margin),(margin,height-marker-margin)]
    return dict(width=width,height=height,photo_rect=[border,border,width-border,height-border],
        aruco=[dict(id=i,rect=[x,y,x+marker,y+marker],quiet=quiet) for i,(x,y) in enumerate(corners)],
        qr_rects=[[int((width-qr)/2),6,int((width+qr)/2),6+qr],
                  [int((width-qr)/2),height-qr-6,int((width+qr)/2),height-6]],
        background='BLACK',photo_fit='EXIF_TRANSPOSE_CONTAIN_NO_STRETCH',
        marker_dictionary='DICT_4X4_50',transition='EXPLICIT_T_PHASE_WITHOUT_PHOTO')


def rect_points(rect, *, pixel_corners=False):
    l,t,r,b=map(float,rect)
    if pixel_corners:r-=1;b-=1
    return np.array([[l,t],[r,t],[r,b],[l,b]],dtype=np.float64)


def project(points, matrix):
    points=np.asarray(points,dtype=np.float64)
    result=cv2.perspectiveTransform(points.reshape(-1,1,2),np.asarray(matrix,dtype=np.float64)).reshape(-1,2)
    if not np.isfinite(result).all():raise ValueError('NONFINITE_TRANSFORM')
    return result


def photo_placement(size, board):
    l,t,r,b=board['photo_rect'];w,h=size
    inset=board.get('image_inset_px',0);l+=inset;t+=inset;r-=inset;b-=inset
    if w/h>(r-l)/(b-t):out_w,out_h=r-l,round(h/w*(r-l))
    else:out_w,out_h=round(w/h*(b-t)),b-t
    x=l+(r-l-out_w)//2;y=t+(b-t-out_h)//2
    return dict(rect=[x,y,x+out_w,y+out_h],source_size=[w,h],scale_x=out_w/w,scale_y=out_h/h,
        letterbox=[x-board['photo_rect'][0],y-board['photo_rect'][1],board['photo_rect'][2]-x-out_w,board['photo_rect'][3]-y-out_h],
        oriented_photo_to_tv=[[out_w/w,0,x],[0,out_h/h,y],[0,0,1]])


def token(manifest_sha, run_id, exposure_id, phase):
    if phase not in ('P','T'):raise ValueError('INVALID_MARKER_PHASE')
    return '|'.join(('F6v1',manifest_sha,run_id,exposure_id,phase))


def render(image, board, payload):
    """One composed frame; only this raster is handed to Tk, never GT/names."""
    canvas=Image.new('RGB',(board['width'],board['height']),'black')
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for item in board['aruco']:
        l,t,r,b=item['rect'];quiet=item['quiet']
        canvas.paste('white',(l-quiet,t-quiet,r+quiet,b+quiet))
        marker=cv2.aruco.generateImageMarker(dictionary,item['id'],r-l)
        canvas.paste(Image.fromarray(marker).convert('RGB'),(l,t))
    paired=board.get('qr_version')==F3_QR
    if paired and (not isinstance(payload,(list,tuple)) or len(payload)!=2 or payload[0]==payload[1]):
        raise ValueError('F3_TWO_DIFFERENT_SYMBOLS_REQUIRED')
    encoded=None if board.get('qr_version') in (F2_QR,F3_QR) else cv2.QRCodeEncoder_create().encode(payload)
    # Encoder output has a quiet zone; use integer modules, then center on white.
    for index,(l,t,r,b) in enumerate(board['qr_rects']):
        if board.get('qr_version')==F2_QR:encoded=qr_f2(payload)  # Each symbol independently encoded.
        if paired:encoded=qr_f3(payload[index])
        factor=min((r-l)//encoded.shape[1],(b-t)//encoded.shape[0])
        if factor<2:raise ValueError('QR_MODULES_TOO_SMALL_FOR_DISPLAY')
        raster_size=(r-l,b-t) if paired else (encoded.shape[1]*factor,encoded.shape[0]*factor)
        qr=Image.fromarray(encoded).resize(raster_size,Image.Resampling.NEAREST)
        canvas.paste('white',(l,t,r,b));canvas.paste(qr,(l+(r-l-qr.width)//2,t+(b-t-qr.height)//2))
    placement=None
    if image is not None:
        oriented=ImageOps.exif_transpose(image).convert('RGB')
        placement=photo_placement(oriented.size,board);l,t,r,b=placement['rect']
        canvas.paste(oriented.resize((r-l,b-t),Image.Resampling.LANCZOS),(l,t))
    return canvas,placement


def geometry(image, board, policy=GEOMETRY_POLICY):
    pixels=np.asarray(image.convert('RGB'));gray=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
    parameters=cv2.aruco.DetectorParameters()
    parameters.minMarkerDistanceRate=policy['detector_min_marker_distance_rate']
    parameters.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),parameters)
    corners,ids,_=detector.detectMarkers(gray)
    ids=[] if ids is None else ids.reshape(-1).tolist()
    expected={row['id']:row for row in board['aruco']}
    found={i:[c.reshape(4,2) for c,j in zip(corners,ids) if j==i] for i in expected}
    if any(len(v)!=1 for v in found.values()):return dict(status='N/A',reason='ARUCO_MISSING_OR_DUPLICATED',detected_ids=ids)
    src=np.vstack([rect_points(expected[i]['rect'],pixel_corners=True) for i in sorted(expected)])
    dst=np.vstack([found[i][0] for i in sorted(expected)]).astype(np.float64)
    edge=min(np.linalg.norm(q-np.roll(q,1,axis=0),axis=1).min() for q in (v[0] for v in found.values()))
    if edge<policy['min_marker_edge_camera_px']:return dict(status='N/A',reason='MARKER_RESOLUTION_TOO_LOW',min_edge_px=float(edge))
    matrix,_=cv2.findHomography(src,dst,0)
    if matrix is None or not np.isfinite(matrix).all():return dict(status='N/A',reason='HOMOGRAPHY_UNAVAILABLE')
    reprojection=float(np.linalg.norm(project(src,matrix)-dst,axis=1).max())
    held_out=[]
    for marker in range(4):
        keep=np.ones(16,dtype=bool);keep[marker*4:(marker+1)*4]=False
        other,_=cv2.findHomography(src[keep],dst[keep],0)
        held_out.append(float(np.linalg.norm(project(src[~keep],other)-dst[~keep],axis=1).max()) if other is not None else math.inf)
    h,w=gray.shape
    normalized=np.diag([1/w,1/h,1])@matrix@np.diag([board['width'],board['height'],1])
    condition=float(np.linalg.cond(normalized));quad=project(rect_points(board['photo_rect']),matrix)
    signed=float(cv2.contourArea(quad.astype(np.float32),oriented=True));area=signed/(w*h)
    visible=bool(np.all(quad[:,0]>=0) and np.all(quad[:,0]<=w) and np.all(quad[:,1]>=0) and np.all(quad[:,1]<=h))
    status='PASS';reason=''
    if signed<=0:status='N/A';reason='MIRRORED_OR_INVERTED_HOMOGRAPHY'
    elif condition>policy['max_normalized_condition']:status='N/A';reason='ILL_CONDITIONED_HOMOGRAPHY'
    elif reprojection>policy['max_reprojection_px'] or max(held_out)>policy['max_leave_one_marker_out_px']:
        status='N/A';reason='REPROJECTION_ERROR'
    elif not visible or area<policy['min_photo_camera_area_fraction']:status='N/A';reason='PHOTO_VISIBILITY_INSUFFICIENT'
    return dict(status=status,reason=reason,matrix_tv_to_camera=matrix.tolist(),
        inverse_camera_to_tv=np.linalg.inv(matrix).tolist(),reprojection_max_px=reprojection,
        leave_one_marker_out_max_px=max(held_out),normalized_condition=condition,
        photo_camera_area_fraction=area,photo_visible=visible,marker_corners_camera=dst.tolist(),
        policy=policy,source_image_size=[w,h],detected_ids=ids,
        ignored_ids=[i for i in ids if i not in expected])


def decode_codes(image, board, verified_geometry):
    """Pixel-only QR diagnostics, before manifest/run/phase qualification.

    Each region is independently projected from the four required markers.
    The bounded fallback changes only a temporary decoder image, not source
    pixels, the display layout, model input or the accepted geometry policy.
    No expected token, GT, run identity or neighboring QR enters this function.
    """
    if verified_geometry.get('status')!='PASS':raise ValueError('QR_GEOMETRY_UNPROVEN')
    if len(board['qr_rects'])!=2:raise ValueError('TWO_QR_REGIONS_REQUIRED')
    pixels=np.asarray(image.convert('RGB'));matrix=np.asarray(verified_geometry['inverse_camera_to_tv'])
    size=(board['width'],board['height'])
    linear=cv2.warpPerspective(pixels,matrix,size,flags=cv2.INTER_LINEAR)
    cubic=None;codes=[]
    def crop(frame,rect,pad):
        l,t,r,b=map(int,rect)
        return frame[max(0,t-pad):min(frame.shape[0],b+pad),max(0,l-pad):min(frame.shape[1],r+pad)]
    for index,rect in enumerate(board['qr_rects']):
        attempts=[]
        def read(patch,method):
            value,points,_=cv2.QRCodeDetector().detectAndDecode(patch)
            attempts.append(dict(method=method,localized=points is not None,decoded=bool(value),
                decoder_image_size=[patch.shape[1],patch.shape[0]]))
            return value
        value=read(crop(linear,rect,2),'LINEAR_RECTIFIED_NATIVE')
        if not value and board.get('qr_version') in (F2_QR,F3_QR):
            # ArUco independently establishes this symbol's location. OpenCV's
            # finder can fail on a valid short QR after subpixel rectification;
            # decode the same pixels at the frozen symbol bounds, never an
            # expected payload or the other QR's result.
            patch=crop(linear,rect,0);module=board['qr']['module_px'];q=4*module;end=q+board['qr']['module_count']*module-1
            if board.get('qr_version')==F3_QR:q,_,end,_=board['qr']['symbol_bounds_px']
            points=np.array([[[q,q],[end,q],[end,end],[q,end]]],dtype=np.float32)
            try:value=cv2.QRCodeDetector().decode(patch,points)[0]
            except cv2.error:value=''
            attempts.append(dict(method='F3_ARUCO_BOUND_SYMBOL_DECODE' if board.get('qr_version')==F3_QR else 'F2_ARUCO_BOUND_SYMBOL_DECODE',localized=True,decoded=bool(value),
                decoder_image_size=[patch.shape[1],patch.shape[0]],bounds_source='FROZEN_LAYOUT_AND_FOUR_ARUCO'))
        if not value:
            if cubic is None:cubic=cv2.warpPerspective(pixels,matrix,size,flags=cv2.INTER_CUBIC)
            gray=cv2.cvtColor(crop(cubic,rect,4),cv2.COLOR_RGB2GRAY)
            sharpened=cv2.addWeighted(gray,2,cv2.GaussianBlur(gray,(0,0),1),-1,0)
            enlarged=cv2.resize(sharpened,None,fx=6,fy=6,interpolation=cv2.INTER_CUBIC)
            value=read(enlarged,'CUBIC_RECTIFIED_UNSHARP_SIGMA1_X6')
        quad=project(rect_points(rect),verified_geometry['matrix_tv_to_camera'])
        codes.append(dict(index=index,position=board.get('qr_positions',['TOP','BOTTOM'])[index],rect_tv=list(rect),
            quad_camera=quad.tolist(),camera_edge_px=np.linalg.norm(quad-np.roll(quad,1,axis=0),axis=1).tolist(),
            value=value,status='DECODED' if value else 'LOCALIZED_NOT_DECODED' if any(a['localized'] for a in attempts) else 'NOT_LOCALIZED',
            attempts=attempts))
    return dict(decoded_tokens=[r['value'] for r in codes],qr_codes=codes,qr_decoder_policy=board.get('decoder_version','f6_qr_independent_rectified_fallback_v2'))


def decode(image, manifest, manifest_sha, run_id, *, policy=None, run_context=None):
    board=manifest['layout'];result=geometry(image,board,policy or manifest['geometry_policy'])
    if result['status']!='PASS':return result
    result.update(decode_codes(image,board,result));decoded=result['decoded_tokens']
    if board.get('layout_version')==F3_LAYOUT:
        return dict(result,**qualify_f3(decoded,manifest,manifest_sha,run_id,run_context))
    if not all(decoded):return dict(result,status='N/A',reason='EXPOSURE_CODE_UNREADABLE')
    if len(set(decoded))!=1:return dict(result,status='N/A',reason='TRANSITION_OR_CONFLICTING_CODES')
    if board.get('layout_version') in F2_LAYOUTS:
        return dict(result,**qualify_f2(decoded[0],manifest,manifest_sha,run_id,run_context))
    parts=decoded[0].split('|')
    if len(parts)!=5 or parts[:3]!=['F6v1',manifest_sha,run_id]:return dict(result,status='N/A',reason='FOREIGN_MANIFEST_OR_RUN')
    if parts[4]!='P':return dict(result,status='N/A',reason='TECHNICAL_TRANSITION_FRAME')
    if parts[3] not in {e['exposure_id'] for e in manifest['exposures']}:
        return dict(result,status='N/A',reason='UNKNOWN_EXPOSURE')
    return dict(result,exposure_id=parts[3],identity_evidence='TWO_SPATIALLY_SEPARATE_QR_CODES_AGREE')


def input_exclusion(decoded, board, crop_rect, rois=(), *, clearance=None):
    """Require marker pixels OUTSIDE every recorded model source region."""
    if decoded.get('status')!='PASS':return dict(status='N/A',reason='GEOMETRY_UNPROVEN')
    clearance=GEOMETRY_POLICY['marker_crop_clearance_px'] if clearance is None else clearance
    symbols=[[r['rect'][0]-r['quiet'],r['rect'][1]-r['quiet'],
              r['rect'][2]+r['quiet'],r['rect'][3]+r['quiet']] for r in board['aruco']]+board['qr_rects']
    regions=[crop_rect,*rois]
    intersections=[]
    for rect in symbols:
        marker=project(rect_points(rect),decoded['matrix_tv_to_camera']).astype(np.float32)
        for region in regions:
            l,t,r,b=region;expanded=rect_points([l-clearance,t-clearance,r+clearance,b+clearance]).astype(np.float32)
            area,_=cv2.intersectConvexConvex(marker,expanded)
            if area>0:intersections.append(dict(marker_rect_tv=rect,input_region=region,intersection_area_px=float(area)))
    return dict(status='PASS' if not intersections else 'N/A',reason='' if not intersections else 'MARKER_INSIDE_MODEL_INPUT',
        intersections=intersections,scope='SPATIAL_EXCLUSION_FROM_VALIDATED_CROP_AND_ROI',
        exact_tensor_pixels='NOT_RECORDED_BY_LEGACY_JPEG94_CONTEXT')


def transform_gt(quad, image_size, board, decoded):
    fit=photo_placement(image_size,board)
    matrix=np.asarray(decoded['matrix_tv_to_camera'])@np.asarray(fit['oriented_photo_to_tv'])
    return project(quad,matrix).tolist(),dict(photo_fit=fit,matrix_photo_to_camera=matrix.tolist(),
        inference_transform='UNCHANGED_CROP_ONLY_TRANSLATION',H_TV_in_inference='NOT_APPLIED')
