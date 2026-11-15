"""Opt-in private A12 source; no room photographs or archives in the repository."""
import hashlib,json,os,zipfile
from pathlib import Path
import pytest
import cv2,numpy as np
from PIL import Image
from auto_annotation_tool.ranking import final180_markers as markers
from auto_annotation_tool.ranking.mobile_package_experiments import read_mobile_report_bundle
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession

SOURCE_SHA='031993976c126cae15aad56905964dc63a338ff4882400b0f5a1020c21b90739'

@pytest.fixture
def real_source():
    path=os.environ.get('ALPR_F6_REAL_SESSION')
    if not path:pytest.skip('Private real A12 archive not supplied')
    path=Path(path);assert hashlib.sha256(path.read_bytes()).hexdigest()==SOURCE_SHA
    return path

def test_real_native_import_keeps_r0_final_identity_and_no_detection(real_source,tmp_path):
    b=read_mobile_report_bundle(real_source)
    assert b.validation.ok,b.validation.errors
    assert b.validation.checked_hashes==36 and b.validation.skipped_hashes==0
    assert b.collection_session['execution']['variant']=='R0_FINAL'
    s=MobileReviewSession(b,sidecar_path=tmp_path/'review.json')
    assert len(s.mt_invocations)==1 and not s.samples
    assert b.f6_evidence['event_counts']==dict(FRAME_OPENED=1,MT_STARTED=1,MT_RETURNED=1,MT_DECODED=1,SCENE_OBSERVATION=1,FRAME_FINISHED=1)
    assert b.f6_evidence['status']=='VALID' and not b.f6_evidence['mz_invocations']
    actual=next(iter(b.f6_evidence['mt_invocations'].values()))['inputs'][0]
    assert actual['roi_camera_xy']==[150,523,778,786]
    link=b.tv_bbox_evidence['rows'][0]['attempt_links'][0];offset=b.tv_bbox_evidence['descriptor']['offset_to_camera_full']
    assert [link['roi_local_xy'][k]+offset['x' if k in ('left','right') else 'y'] for k in ('left','top','right','bottom')]==actual['roi_camera_xy']
    assert not s.sidecar_path.exists()

@pytest.mark.parametrize('fault',['mp_on','wrong_budget','missing_budget','different_variant','session_conflict'])
def test_final_label_does_not_bypass_execution_contract(real_source,tmp_path,fault):
    with zipfile.ZipFile(real_source) as z:data={n:z.read(n) for n in z.namelist()}
    session=json.loads(data['session.json']);report=json.loads(data['report.json']);execution=session['execution']
    if fault=='mp_on':execution['stages']['mp']['enabled']=True
    elif fault=='wrong_budget':execution['roi_budget_policy']='r1_one_roi'
    elif fault=='missing_budget':execution.pop('roi_budget_policy')
    elif fault=='different_variant':execution['variant']='R2_FINAL'
    else:session['variant']='R1'
    report['research_collection']=session
    data['session.json']=json.dumps(session).encode();data['report.json']=json.dumps(report).encode()
    manifest=json.loads(data['manifest.json'])
    for name in ('session.json','report.json'):manifest['entry_sha256'][name]=hashlib.sha256(data[name]).hexdigest()
    data['manifest.json']=json.dumps(manifest).encode()
    path=tmp_path/'negative.alprsession'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name,value in data.items():z.writestr(name,value)
    b=read_mobile_report_bundle(path)
    assert not b.validation.ok and any('F6 is recorded R0/MP OFF' in e for e in b.validation.errors)

@pytest.fixture
def real_board(real_source):
    import io
    path=os.environ.get('ALPR_F6_PREPARED')
    if not path:pytest.skip('Matching private/read-only F6 preparation not supplied')
    p=Path(path)/'manifest.json';raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    assert sha=='b0acd187f8cee444752efecdc360176c9a40f22b0977052afccfc26c4ab2d9d5'
    m=json.loads(raw);b=read_mobile_report_bundle(real_source);assert b.validation.ok
    entry=b.tv_bbox_evidence['rows'][0]['source_image']
    with zipfile.ZipFile(real_source) as z:raw=z.read(entry['entry'])
    assert hashlib.sha256(raw).hexdigest()==entry['sha256']=='ab62cdbf443fd7410406896d2a39187b663e813a4e7913538c59d977da0c725c'
    return Image.open(io.BytesIO(raw)).convert('RGB'),m,sha

def test_real_two_qr_independent_geometry_and_technical_phase(real_board):
    image,m,sha=real_board;before=image.tobytes();g=markers.geometry(image,m['layout'],m['geometry_policy'])
    assert g['status']=='PASS' and set(g['detected_ids'])=={0,1,2,3,17} and g['ignored_ids']==[17]
    qr=markers.decode_codes(image,m['layout'],g)
    assert all(q['status']=='DECODED' for q in qr['qr_codes'])
    expected='F6v1|'+sha+'|62df0673e13a402aabdc59c77b6cd8de|TEST_ONLY|T'
    assert qr['decoded_tokens']==[expected,expected] # assert only; never an input to pixel decoding
    assert all(q['attempts'][0]['decoded'] is False and q['attempts'][-1]['decoded'] is True for q in qr['qr_codes'])
    assert image.tobytes()==before
    d=markers.decode(image,m,sha,qr['decoded_tokens'][0].split('|')[2])
    assert d['reason']=='TECHNICAL_TRANSITION_FRAME' and not d.get('exposure_id')
    assert markers.decode(image,m,sha,'a_different_run')['reason']=='FOREIGN_MANIFEST_OR_RUN'
    roi=[150,523,778,786] # independently recorded test expectation, not decoder coordinates
    assert markers.input_exclusion(g,m['layout'],roi,[roi])['status']=='PASS'
    assert markers.input_exclusion(g,m['layout'],[39,417,888,888])['reason']=='MARKER_INSIDE_MODEL_INPUT'

@pytest.mark.parametrize('erased',[0,1])
def test_real_erasing_either_qr_never_borrows_other_token(real_board,erased):
    image,m,sha=real_board;g=markers.geometry(image,m['layout'],m['geometry_policy'])
    q=markers.project(markers.rect_points(m['layout']['qr_rects'][erased]),g['matrix_tv_to_camera'])
    pixels=np.array(image);cv2.fillConvexPoly(pixels,np.round(q).astype(np.int32),[255,255,255])
    d=markers.decode(Image.fromarray(pixels),m,sha,'unused_identity')
    assert d['reason']=='EXPOSURE_CODE_UNREADABLE'
    assert d['decoded_tokens'][erased]=='' and d['decoded_tokens'][1-erased].startswith('F6v1|')

def test_different_run_ids_in_two_qr_are_conflicting_even_with_valid_pixels():
    from test_final180_desktop import board_image
    m,image,_=board_image()
    other,_=markers.render(None,m['layout'],markers.token('a'*64,'c'*32,'e0001','P'))
    rect=m['layout']['qr_rects'][1];image.paste(other.crop(rect),tuple(rect[:2]))
    d=markers.decode(image,m,'a'*64,'b'*32)
    assert d['reason']=='TRANSITION_OR_CONFLICTING_CODES' and len(set(d['decoded_tokens']))==2

def test_low_level_qr_requires_verified_geometry_and_two_regions():
    from test_final180_desktop import board_image
    m,image,_=board_image()
    with pytest.raises(ValueError,match='QR_GEOMETRY_UNPROVEN'):markers.decode_codes(image,m['layout'],dict(status='N/A'))
    g=markers.geometry(image,m['layout']);m['layout']['qr_rects'].pop()
    with pytest.raises(ValueError,match='TWO_QR_REGIONS_REQUIRED'):markers.decode_codes(image,m['layout'],g)
