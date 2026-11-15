from copy import deepcopy
from dataclasses import asdict
import hashlib

import cv2
import numpy as np
import pytest

from auto_annotation_tool.rectification import plate_rectifier as rect
from auto_annotation_tool.ranking.e2e_eval_metrics import score_pair, validate_reference
from auto_annotation_tool.ranking.mobile_human_review import align_plate_text
from auto_annotation_tool.ranking.mz_whole_plate_evaluation import read_plate_records,aggregate_plate_metrics
from auto_annotation_tool.registration_text import normalize_registration


@pytest.mark.parametrize('rotation',[0,1,2,3])
def test_asymmetric_rotation_preserves_raw_k0_without_screen_sort(rotation):
    image=np.zeros((200,200,3),np.uint8)
    points=np.array([[20.,20.],[160.,20.],[160.,60.],[20.,60.]])
    for point,color in zip(points,[(0,0,255),(0,255,0),(255,0,0),(0,255,255)]):
        cv2.circle(image,tuple(point.astype(int)),8,color,-1)
    expected,evidence=rect.PlateRectifier.rectify_predicted_quad(image,points.tolist())
    for _ in range(rotation):
        image=np.rot90(image).copy();points=np.stack((points[:,1],199-points[:,0]),axis=1)
    before=deepcopy(points.tolist())
    actual,result=rect.PlateRectifier.rectify_predicted_quad(image,points.tolist())
    assert result['permutation']==[0,1,2,3] and points.tolist()==before
    assert actual.shape==expected.shape==(91,320,3)
    assert np.mean(abs(actual.astype(float)-expected.astype(float)))<1
    assert result['pixel_sha256']==hashlib.sha256(actual.tobytes()).hexdigest()


@pytest.mark.parametrize('w,h',[(40,100),(100,100),(100,40)])
def test_portrait_square_landscape_aspect_and_no_automatic_rotation(w,h):
    image=np.zeros((150,150,3),np.uint8)
    points=[[10,10],[10+w,10],[10+w,10+h],[10,10+h]]
    crop,info=rect.PlateRectifier.rectify_predicted_quad(image,points)
    assert crop.shape[1]/crop.shape[0]==pytest.approx(w/h,abs=.005)
    assert max(crop.shape[:2])==320
    assert info['permutation']==[0,1,2,3]


def test_reverse_cycle_has_explicit_permutation_but_raw_is_unchanged():
    points=[[10,10],[10,40],[110,40],[110,10]];before=deepcopy(points)
    crop,info=rect.PlateRectifier.rectify_predicted_quad(np.zeros((150,150,3),np.uint8),points)
    assert points==before and info['raw_quad_status']=='REVERSED_WINDING'
    assert info['permutation']==[0,3,2,1] and crop is not None


@pytest.mark.parametrize('points,status',[
    (None,'KEYPOINTS_MISSING'), ([[10,10],[100,10],[10,50],[100,50]],'SELF_INTERSECTING'),
    ([[10,10],[100,10],[30,20],[10,50]],'CONCAVE'),
    ([[10,10],[10,10],[100,50],[10,50]],'DEGENERATE'),
    ([[10,10],[100,10],[float('nan'),50],[10,50]],'DEGENERATE'),
    ([[-2,10],[100,10],[100,50],[10,50]],'OUT_OF_BOUNDS')])
def test_invalid_raw_never_hull_repairs_or_bbox_falls_back(points,status):
    crop,info=rect.PlateRectifier.rectify_predicted_quad(np.zeros((150,150,3),np.uint8),points)
    assert crop is None and info['status']=='QUAD_INVALID' and info['raw_quad_status']==status
    assert info['permutation'] is None and info['pixel_sha256'] is None


def test_missing_opencv_is_runtime_block_not_a_whole_scene_crop(monkeypatch):
    monkeypatch.setattr(rect,'CV2_AVAILABLE',False)
    with pytest.raises(RuntimeError,match='OPENCV_REQUIRED'):
        rect.PlateRectifier.rectify_predicted_quad(np.zeros((20,20,3),np.uint8),[[0,0],[10,0],[10,10],[0,10]])


def test_homography_error_has_explicit_failure_without_mz(monkeypatch):
    def fail(*a,**k):raise cv2.error('TEST_HOMOGRAPHY')
    monkeypatch.setattr(rect.cv2,'getPerspectiveTransform',fail)
    crop,info=rect.PlateRectifier.rectify_predicted_quad(np.zeros((20,20,3),np.uint8),[[0,0],[10,0],[10,10],[0,10]])
    assert crop is None and info['status']=='RECTIFICATION_FAILED'


def test_two_row_order_unicode_and_empty_micro_cer_use_shared_contract():
    detections=[{'character':'2','bbox':[20,50,30,70]}, {'character':'B','bbox':[20,0,30,20]},
                {'character':'1','bbox':[0,50,10,70]}, {'character':'A','bbox':[0,0,10,20]}]
    assert read_plate_records(detections)['prediction']=='AB12'
    assert normalize_registration(' łódź-01 ')=='ŁÓDŹ01'
    rows=[]
    for gt,pred in [('AB',''),('12345678','12345678')]:
        row=asdict(align_plate_text(gt,pred));row['no_read']=not pred;rows.append(row)
    assert rows[0]['missing_characters']==2 and aggregate_plate_metrics(rows)['cer']==.2


def full_reference():
    scenes=[]
    for domain,scene_count,readable,unreadable,sevens in [('DAY',199,230,5,142),('NIGHT',197,219,25,216)]:
        domain_scenes=[{'sha256':f'TEST-{domain}-{i}', 'domain':domain, 'image':{'oriented_size':[320,200]},'plates':[]} for i in range(scene_count)]
        for i in range(readable+unreadable):
            excluded=domain=='DAY' and i==readable
            domain_scenes[i%scene_count]['plates'].append({'plate_id':f'TEST-{domain}-plate-{i}', 'bbox':[10,10,60,30],
                'quad':[[10,10],[60,10],[60,30],[10,30]], 'bbox_eligible':True,'pose_eligible':not excluded,
                'pose_exclusion':{'reason':'TEST_ONLY'} if excluded else None,
                'pose_evaluation_status':'EXCLUDED_BY_OPERATOR' if excluded else 'ELIGIBLE',
                'ocr_eligible':i<readable,'text_decision':'approved' if i<readable else 'excluded_unreadable',
                'ground_truth_text':('A'* (7 if i<sevens else 6)) if i<readable else None})
        scenes.extend(domain_scenes)
    return scenes


def test_full_denominators_all_readable_count_even_with_no_mt():
    reference=full_reference();counts=validate_reference(reference)
    assert counts['ALL']=={'scenes':396,'bbox':479,'pose':478,'ocr':449,'characters':3052}
    assert counts['DAY']['characters']==1522 and counts['NIGHT']['characters']==1530
    rows=[{'scene_sha256':s['sha256'],'width':320,'height':200,'status':'COMPLETE','detections':[]} for s in reference]
    result=score_pair(rows,reference)
    for domain,plates,characters in [('ALL',449,3052),('DAY',230,1522),('NIGHT',219,1530)]:
        m=result['domains'][domain]
        assert m['plates']==plates and m['ground_truth_characters']==characters
        assert m['no_read']==plates and m['cer']==1 and m['exact_match_rate']==0
    assert result['domains']['ALL']['box_fn']==479


def test_missing_or_duplicate_gt_never_silently_changes_denominator():
    reference=full_reference();reference[0]['plates'][0].pop('ground_truth_text')
    with pytest.raises(ValueError,match='GT_TEXT_MISSING'):validate_reference(reference)
    reference=full_reference();reference.append(deepcopy(reference[0]))
    with pytest.raises(ValueError,match='GT_SCENE_INVALID'):validate_reference(reference)
