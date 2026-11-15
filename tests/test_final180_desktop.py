"""Desktop-only FINAL tests; synthetic data is not Android/A12 acceptance."""
from copy import deepcopy
from pathlib import Path
from _private_test_data import private_test_path
from types import SimpleNamespace as NS
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
import pytest

from auto_annotation_tool.ranking import final180_reference as ref
from auto_annotation_tool.ranking import final180_markers as marker
from auto_annotation_tool.ranking import final180_mobile as mobile
from auto_annotation_tool.ranking import photo_reference as photo
from auto_annotation_tool.gui.final180_tv import PresentationRun,PreparedFrames


def material(tmp_path,n=2,negative=1,empty=(),unreadable=(),duplicate=False):
    pos=tmp_path/'positive';neg=tmp_path/'negative';pos.mkdir(parents=True);neg.mkdir()
    root=ET.Element('annotations')
    for i in range(n):
        name=f'p{i:03d}.png';Image.new('RGB',(100,80),(i%250,30,70)).save(pos/name)
        el=ET.SubElement(root,'image',name=name,width='100',height='80',id=str(i))
        if i not in empty:
            shape=ET.SubElement(el,'polygon',label='plate',source='manual',points='20,20;80,20;80,45;20,45')
            ET.SubElement(shape,'attribute',name='plate_annotation_id').text=f'plate-{i}'
            if i not in unreadable:ET.SubElement(shape,'attribute',name='ground_truth_text').text='TEST123'
            ET.SubElement(shape,'attribute',name='text').text='PREDICTION_IS_NOT_GT'
    for i in range(negative):Image.new('RGB',(100,80),(255,100+i,10)).save(neg/f'n{i:03d}.png')
    if duplicate:shutil.copyfile(pos/'p000.png',pos/'p001.png')
    xml=tmp_path/'annotations.xml';ET.ElementTree(root).write(xml,encoding='utf-8',xml_declaration=True)
    a=ref.audit(xml,pos,neg,negative_declared=True,unreadable_plate_ids=[f'plate-{i}' for i in unreadable],expected_negative=negative)
    return a


def frozen(tmp_path,*,dwell_seconds=30,**kwargs):
    a=material(tmp_path/'input',**kwargs)
    return ref.freeze(a,tmp_path/'experiment',dwell_seconds=dwell_seconds,technical=True)


@pytest.mark.parametrize('n',[166,170])
def test_full_xml_inventory_and_variable_operator_selection(tmp_path,n):
    a=material(tmp_path,n=n,negative=10,empty=(3,),unreadable=(1,2))
    assert a['summary']['xml_image_records']==n
    assert a['summary']['plates']==n-1 and a['summary']['unreadable']==2
    assert len(a['images'])==n+10
    assert 'POSITIVE_WITHOUT_PLATE:p003.png' in a['errors']
    assert a['images'][3]['state']['presence']=='UNRESOLVED'
    assert all(r['state']['presence']=='NO_PLATES' for r in a['images'][-10:])


def test_unreadability_is_scoped_and_prediction_never_replaces_gt(tmp_path):
    a=material(tmp_path,n=2,unreadable=(0,))
    assert a['images'][0]['state']['objects'][0]['gt_text']==''
    again=ref.audit(a['xml_path'],a['positive_dir'],a['negative_dir'],negative_declared=True,expected_negative=1)
    assert 'UNRESOLVED_TEXT:plate-0' in again['errors']
    assert 'PREDICTION_IS_NOT_GT' not in [o['gt_text'] for r in again['images'] for o in r['state']['objects']]


def test_duplicate_bytes_have_separate_exposure_and_source_annotation_ids(tmp_path):
    d=frozen(tmp_path,duplicate=True)
    m,mat,r=ref.load(d)
    assert m['exposures'][0]['photo_sha256']==m['exposures'][1]['photo_sha256']
    assert m['exposures'][0]['exposure_id']!=m['exposures'][1]['exposure_id']
    assert r.state(m['exposures'][0]['image_id'])['objects'][0]['source_plate_annotation_id']=='plate-0'
    assert r.state(m['exposures'][1]['image_id'])['objects'][0]['source_plate_annotation_id']=='plate-1'


def test_freeze_uses_native_photo_reference_and_protects_xml(tmp_path):
    a=material(tmp_path/'source',unreadable=(0,));before=photo.digest(a['xml_path'])
    d=ref.freeze(a,tmp_path/'out');m,mat,r=ref.load(d)
    assert r.data['schema']==photo.SCHEMA_V2 and r.data['predictions_exposed'] is True
    assert r.data['import_declarations']['blank_text_unreadable']['plate_ids']==['plate-0']
    assert photo.digest(a['xml_path'])==before
    with pytest.raises(ValueError,match='tylko do odczytu'):r.add(mat.order[0],[[1,1],[10,1],[10,10],[1,10]])


@pytest.mark.parametrize('target',['reference.json','material.json','image'])
def test_frozen_source_change_is_detected(tmp_path,target):
    d=frozen(tmp_path)
    p=next((d/'images').iterdir()) if target=='image' else d/target
    p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError):ref.load(d)


def test_approved_only_package_is_rejected(tmp_path):
    a=material(tmp_path)
    Path(tmp_path/'annotation_export_manifest.json').write_text(json.dumps(dict(annotation_scope='approved_ok_only')))
    again=ref.audit(a['xml_path'],a['positive_dir'],a['negative_dir'],negative_declared=True,expected_negative=1)
    assert 'APPROVED_ONLY_XML_NOT_ALLOWED_FOR_FINAL' in again['errors']


def test_all_saved_export_preserves_empty_records_without_declaring_negative():
    from auto_annotation_tool.gui.z2_run_io_runtime import _get_plate_annotation_export_annotations
    from auto_annotation_tool.data_models import ImageAnnotation,Detection
    records=[ImageAnnotation('empty.jpg',100,100),ImageAnnotation('plate.jpg',100,100,[Detection('plate',1,(1,1,10,10))])]
    owner=NS(current_annotation_run_dir=Path('run'),current_annotations=records,_paths_equivalent=lambda a,b:True,
        _get_plate_detections=lambda a:a.plates)
    assert len(_get_plate_annotation_export_annotations(owner,Path('run')))==1
    assert len(_get_plate_annotation_export_annotations(owner,Path('run'),include_empty=True))==2
    assert records[0].detections==[]


def test_native_cvat_export_keeps_empty_image_record(tmp_path):
    from auto_annotation_tool.exporters.cvat_exporter import CVATExporter
    from auto_annotation_tool.data_models import ImageAnnotation,AnnotationStatus
    p=tmp_path/'saved.xml'
    assert CVATExporter().export([ImageAnnotation('empty.png',100,80,status=AnnotationStatus.NO_PLATE)],p,only_successful=False)
    assert len(ET.parse(p).getroot().findall('image'))==1


def test_three_plates_with_partial_and_reordered_finals():
    a=plate('a');b=plate('b','',[[50,10],[80,10],[80,30],[50,30]],kind='PARTIAL_TEXT')
    b['partial_text']='TE';c=plate('c','LAST',[[10,50],[40,50],[40,70],[10,70]])
    finals=[candidate('z',[10,50,40,70],'LAST'),candidate('y',[50,10,80,30],'UNTRUSTED'),candidate('x',[10,10,40,30],'TEST')]
    e,s,f=scoring_state([a,b,c],finals)
    r=mobile.score_frame(e,s,f);assert (r['tp'],r['fp'],r['fn'])==(3,0,0)
    assert r['readable_gt']==2 and r['exact_reads']==2 and r['plates'][1]['text_outcome']=='N/A'
    f['finals'].reverse();assert mobile.score_frame(e,s,f)['final_order']==r['final_order']


def test_native_final_panel_uses_existing_viewer_and_variable_count(tmp_path):
    from auto_annotation_tool.gui.final180_tv import make_app,legacy_module,LEGACY_DEFAULT
    if not LEGACY_DEFAULT.is_file():pytest.skip('Original optional F5A viewer not installed')
    d=frozen(tmp_path,n=2,negative=1)
    args=NS(xml=None,positives=None,negatives=None,output=tmp_path,dwell=30,negatives_declared=True,unreadable_ids=[])
    app=make_app(legacy_module(LEGACY_DEFAULT),args)
    try:
        app.loaded(d);app.window.update()
        assert len(app.manifest['exposures'])==3
        assert app.manifest['planned_photo_seconds']==90
        assert app.viewer is None and not (d/'runs').exists()
        assert 'HUD' in app.window.title() or 'klatki' in app.window.title()
    finally:app._close()


@pytest.mark.parametrize('orientation,expected',[ (1,[10,20]),(2,[90,20]),(3,[90,60]),(4,[10,60]),
    (5,[20,10]),(6,[60,10]),(7,[60,90]),(8,[20,90])])
def test_exif_coordinate_chain(orientation,expected):
    assert ref.orient_points([[10,20]],(100,80),orientation)==[expected]


def board_image(phase='P',eid='e0001'):
    board=marker.layout(1920,1080)
    m=dict(layout=board,geometry_policy=marker.GEOMETRY_POLICY,exposures=[dict(exposure_id='e0001'),dict(exposure_id='e0002')])
    im,fit=marker.render(Image.new('RGB',(1000,700),'gray') if phase=='P' else None,board,marker.token('a'*64,'b'*32,eid,phase))
    return m,im,fit


def test_standard_markers_decode_with_redundant_geometric_validation():
    m,im,_=board_image();d=marker.decode(im,m,'a'*64,'b'*32)
    assert d['status']=='PASS' and d['exposure_id']=='e0001'
    assert d['leave_one_marker_out_max_px']<marker.GEOMETRY_POLICY['max_leave_one_marker_out_px']
    assert len(d['marker_corners_camera'])==16


@pytest.mark.parametrize('fault',['damaged','different_code','transition','foreign_run','mirrored'])
def test_unreadable_conflicting_transition_or_foreign_markers_never_guess(fault):
    m,im,_=board_image()
    if fault=='damaged':im.paste('white',tuple(m['layout']['qr_rects'][0]))
    elif fault=='different_code':
        _,other,_=board_image(eid='e0002');r=m['layout']['qr_rects'][1];im.paste(other.crop(r),(r[0],r[1]))
    elif fault=='transition':m,im,_=board_image('T')
    elif fault=='mirrored':im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    d=marker.decode(im,m,'a'*64,'c'*32 if fault=='foreign_run' else 'b'*32)
    assert d['status']=='N/A' and not d.get('exposure_id')


def test_model_crop_excludes_entire_technical_border_and_preserves_translation():
    m,im,fit=board_image();d=marker.decode(im,m,'a'*64,'b'*32)
    assert marker.input_exclusion(d,m['layout'],m['layout']['photo_rect'])['status']=='PASS'
    assert marker.input_exclusion(d,m['layout'],[0,0,1920,1080])['status']=='N/A'
    q,chain=marker.transform_gt([[0,0],[1000,0],[1000,700],[0,700]],(1000,700),m['layout'],d)
    assert np.asarray(q)==pytest.approx(marker.rect_points(fit['rect']),abs=1)
    assert chain['H_TV_in_inference']=='NOT_APPLIED'


def test_perspective_camera_image_and_jpeg_context():
    m,im,_=board_image();matrix=cv2.getPerspectiveTransform(np.float32([[0,0],[1919,0],[1919,1079],[0,1079]]),
        np.float32([[30,50],[1450,80],[1420,900],[70,920]]))
    camera=cv2.warpPerspective(np.asarray(im),matrix,(1500,1000))
    import io
    stream=io.BytesIO();Image.fromarray(camera).save(stream,format='JPEG',quality=94)
    d=marker.decode(Image.open(io.BytesIO(stream.getvalue())),m,'a'*64,'b'*32)
    assert d['status']=='PASS',d


def plate(pid,text='TEST',quad=None,kind=None):
    return dict(object_id=pid,source_plate_annotation_id=pid,quad=quad or [[10,10],[40,10],[40,30],[10,30]],
        gt_text=text,text_state=kind or ('FULL_TEXT' if text else 'UNREADABLE_TEXT'),partial_text='',
        text_exclusion_reason='' if text else 'operator unreadable',geometry_exclusion_reason='')


def candidate(aid,box,text='TEST',confidence=.9):
    return dict(attempt_id=aid,bbox=box,confidence=confidence,mz=dict(executed=True,status='EXECUTED',
        source_bound=True,raw_field_present=True,unambiguous=True,raw_prediction=text))


def scoring_state(objects,finals,execution='COMPLETED'):
    exposure=dict(exposure_id='e0001',position=1,photo_sha256='a'*64,source_role='POSITIVE' if objects else 'NEGATIVE')
    state=dict(presence='PRESENT' if objects else 'NO_PLATES',objects=objects)
    frame=dict(frame_key='source',source_stamp=dict(source_sequence=1),captured_elapsed_nanos=1,
        mt_invocation_ids=['call'],identity_status='PASS',exposure_id='e0001',geometry_status='PASS',
        projected_gt=[dict(plate_id=o['object_id'],quad=o['quad'],bbox=mobile._bbox(o['quad'])) for o in objects],
        input_region=[0,0,100,100],image_size=[100,100],execution_status=execution,final_population_complete=True,finals=finals)
    return exposure,state,frame


def test_wrong_text_does_not_turn_detection_tp_into_fp_and_unreadable_is_detected():
    objects=[plate('p1'),plate('p2','',[[50,10],[80,10],[80,30],[50,30]])]
    e,s,f=scoring_state(objects,[candidate('a2',[50,10,80,30],'ANY'),candidate('a1',[10,10,40,30],'WRONG')])
    r=mobile.score_frame(e,s,f)
    assert (r['tp'],r['fp'],r['fn'])==(2,0,0)
    assert r['readable_gt']==1 and r['wrong_reads']==1
    assert r['plates'][1]['text_outcome']=='N/A'


def test_duplicates_count_fp_and_text_never_drives_geometry_matching():
    e,s,f=scoring_state([plate('p1')],[candidate('a1',[10,10,40,30],'WRONG',.9),candidate('a2',[10,10,40,30],'TEST',.8)])
    r=mobile.score_frame(e,s,f)
    assert (r['tp'],r['fp'],r['fn'])==(1,1,0) and r['wrong_reads']==1


@pytest.mark.parametrize('execution',['ERROR','CANCELLED','UNPROVEN'])
def test_failed_first_input_is_not_replaced_by_later_success(execution):
    e,s,f=scoring_state([plate('p1')],[],execution);later=deepcopy(f)
    later.update(frame_key='later',captured_elapsed_nanos=2,execution_status='COMPLETED',finals=[candidate('a1',[10,10,40,30])])
    first,reason=mobile.select_first('e0001',[later,f]);assert first is f
    r=mobile.score_frame(e,s,first)
    assert r['tp'] is None and r['fn'] is None
    assert r['status']==('N/A' if execution=='UNPROVEN' else 'SYSTEM_FAILURE')


def test_zero_negative_false_positive_and_missing_archive_are_distinct():
    e,s,f=scoring_state([],[]);r=mobile.score_frame(e,s,f)
    assert r['status']=='MEASURED' and r['negative_false_positive'] is False
    f['finals']=[candidate('bad',[10,10,30,30])];r=mobile.score_frame(e,s,f)
    assert r['fp']==1 and r['negative_false_positive'] is True
    r=mobile.score_frame(e,s,None);assert r['negative_false_positive'] is None


def test_text_without_geometry_is_not_a_fake_box_or_correct_silence():
    e,s,f=scoring_state([],[]);f['text_without_geometry']=[dict(raw_prediction='EXTRA')]
    r=mobile.score_frame(e,s,f)
    assert r['fp']==0 and r['negative_text_only'] is True and r['negative_correct_silence'] is False


def test_run_journal_tampering_is_rejected(tmp_path):
    d=frozen(tmp_path);m,_,_=ref.load(d);run=PresentationRun(d,m);run.stop()
    with (run.directory/'events.jsonl').open('a') as stream:stream.write('{}\n')
    with pytest.raises(ValueError,match='JOURNAL_'):
        mobile.analyze_archive(d,tmp_path/'absent.zip',run.directory/'run.json')


def test_missed_readable_plate_stays_in_system_denominator_and_no_ocr_correction():
    e,s,f=scoring_state([plate('p1','O1'),plate('p2','MORE',[[50,10],[80,10],[80,30],[50,30]])],
        [candidate('a',[10,10,40,30],'01')])
    r=mobile.score_frame(e,s,f)
    assert r['readable_gt']==2 and r['wrong_reads']==1 and r['missing_reads']==1 and r['exact_reads']==0


def test_missing_mz_record_is_na_not_empty_read():
    e,s,f=scoring_state([plate('p1')],[candidate('a',[10,10,40,30])]);f['finals'][0]['mz']['raw_field_present']=False
    r=mobile.score_frame(e,s,f);assert r['unknown_reads']==1 and r['missing_reads']==0


def test_not_run_and_late_completion_use_source_frame_identity():
    e,s,f=scoring_state([plate('p1')],[]);f['completion_exposure_id']='e0002'
    first,_=mobile.select_first('e0001',[f]);assert first is f
    f['mt_invocation_ids']=[]
    assert mobile.select_first('e0001',[f])==(None,'NOT_RUN_OBSERVED_NO_STARTED_MT')


def test_stop_resume_and_accelerated_long_run_never_skip_or_overwrite(tmp_path):
    d=frozen(tmp_path,n=166,negative=10);m,_,_=ref.load(d);now=[0]
    run=PresentationRun(d,m,clock=lambda:now[0]);run.show(0,{})
    journal_before=(run.directory/'events.jsonl').read_bytes();run.stop()
    resumed=PresentationRun(d,m,resume=run.directory,clock=lambda:now[0]);assert resumed.next_index==1
    assert (run.directory/'events.jsonl').read_bytes().startswith(journal_before)
    for index in range(1,176):
        resumed.present(index,{},lambda:None)
        now[0]+=int(m['dwell_seconds']*1e9)+987654
        assert resumed.due();resumed.end_photo()
        if index==175:resumed.complete(lambda:None)
        else:resumed.replace(lambda:None)
    from auto_annotation_tool.ranking.final180_timeline import read_run
    _,events,_=read_run(run.directory/'run.json',m,photo.digest(d/'manifest.json'))
    assert [e['position'] for e in events if e['event']=='PHOTO_REQUESTED']==list(range(1,177))
    assert photo.read_json(run.directory/'run.json')['status']=='COMPLETED'
    assert any(e['event']=='RUN_STOPPED' and e['partial_position']==1 for e in events)


def test_prefetch_has_at_most_two_frames(tmp_path):
    d=frozen(tmp_path,n=5);m,mat,_=ref.load(d);cache=PreparedFrames(mat,m,photo.digest(d/'manifest.json'),'b'*32)
    try:
        for i in range(len(m['exposures'])):
            cache.request(i);cache.get(i);cache.request(i+1)
        assert cache.max_resident<=2
    finally:cache.close()


def test_invalid_native_archive_preserves_whole_plan_and_na(tmp_path):
    d=frozen(tmp_path);m,_,_=ref.load(d);run=PresentationRun(d,m);run.stop()
    bad=tmp_path/'bad.alprsession';bad.write_bytes(b'not a zip')
    report=mobile.analyze_archive(d,bad,run.directory/'run.json')
    assert len(report['rows'])==3 and all(r['status']=='N/A' for r in report['rows'])
    assert sum(len(r['plates']) for r in report['rows'])==2
    assert report['summary']['detection']['tp'] is None and report['summary']['negatives_with_false_detection'] is None
    assert report['go']=='NO_GO'
    assert report==mobile.analyze_archive(d,bad,run.directory/'run.json')


def test_versioned_report_export_preserves_na_plate_rows_and_hashes(tmp_path):
    d=frozen(tmp_path/'fixture');m,_,_=ref.load(d);run=PresentationRun(d,m);run.stop()
    bad=tmp_path/'empty.alprsession';bad.write_bytes(b'')
    result=mobile.analyze_archive(d,bad,run.directory/'run.json')
    first=mobile.export_report(result,tmp_path/'results');second=mobile.export_report(result,tmp_path/'results')
    assert first!=second and (first/'evaluation.json').read_bytes()==(second/'evaluation.json').read_bytes()
    manifest=photo.read_json(first/'manifest.json')
    assert all(photo.digest(first/name)==sha for name,sha in manifest['files'].items())
    import csv
    with (first/'plates.csv').open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream,delimiter=';'))
    assert len(rows)==2 and all(row['detection']=='N/A' for row in rows)


def test_existing_native_android_archive_without_final_markers_is_not_a_final_success(tmp_path):
    archive=private_test_path('F5A-ROI-A12-01/02_sesje/R0/PAIRED_ROI_01/r01/F5A_R0_r01_BLOK_A.alprsession')
    if not archive.is_file():pytest.skip('Optional legacy producer archive unavailable')
    d=frozen(tmp_path);m,_,_=ref.load(d);run=PresentationRun(d,m);run.stop();before=photo.digest(archive)
    report=mobile.analyze_archive(d,archive,run.directory/'run.json')
    assert report['acceptance']['desktop_import']=='NATIVE_READER_VALIDATED'
    assert report['summary']['identified_exposures']==0 and all(r['status']=='N/A' for r in report['rows'])
    assert report['acceptance']['producer_contract'].startswith('NOT_VERIFIED') and photo.digest(archive)==before
