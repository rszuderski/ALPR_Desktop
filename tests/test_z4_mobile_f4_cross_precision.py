#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4LIB020: scientific F4 FP32↔INT8 comparison is manifest-only and fail-closed."""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / 'auto_annotation_tool/gui/z4_model_export.py'
DIRECTORY = ROOT / 'auto_annotation_tool/gui'

def load(filename):
    spec = importlib.util.spec_from_file_location('_z4_20_'+filename, DIRECTORY / (filename+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

compare = load('z4_mobile_f4_comparability')
inventory = load('z4_mobile_f4_inventory')
PINS = {'MT':'a'*64,'MZ':'b'*64}


def model(role, precision, suffix='', *, labels=None):
    size = 512 if role == 'MT' else 320
    names = labels if labels is not None else (['plate'] if role=='MT' else ['0','1','Q'])
    role_name = 'plate' if role=='MT' else 'character'
    mf = {
       'schema':'alpr.model.v1','role':role_name,'task':'pose' if role=='MT' else 'detect',
       'source':{'checkpoint_sha256':PINS[role]}, 'model_id':role+'-'+precision+suffix,
       'labels':names,
       'variants':[{'runtime':'tflite','precision':precision,
                    'graph_precision':{'verified':True}}],
       'input':{'width':size,'height':size,'channels':3,'layout':'NHWC','color':'RGB',
                'data_type':'INT8' if precision=='int8' else 'FLOAT32'},
       'output':{'class_count':len(names), 'keypoint_count':4 if role=='MT' else 0,
                 'keypoint_dimensions':2 if role=='MT' else 0,
                 'confidence_threshold':0.25,'iou_threshold':0.45,
                 'nms_required':True,'nms_in_graph':False,
                 'normalized_coordinates':False,'has_objectness':False,'end2end_output':False,
                 'decoder':'yolo_raw','output_format':'raw_yolo',
                 'box_format':'xywh','tensor_layout':'channels_first'},
    }
    if precision=='int8':
        mf['calibration_evidence']={'file':'calib.json','mode':'exploratory','status':'EXPLORATORY_ONLY'}
        mf['research_calibration']={'final_test_status':'NOT_VERIFIED'}
    return SimpleNamespace(kind='MODEL', role=role, schema='alpr.model.v1',
             state='DISCOVERED_NOT_VERIFIED',error='', manifest=mf,
             path=f'/test/{role}_{precision}{suffix}.alprmodel', name=f'{role}_{precision}{suffix}.alprmodel')


class CrossPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.rows=[model(role,precision) for precision in ('fp32','int8') for role in ('MT','MZ')]

    def audit(self, rows=None, *, truncated=False):
        r=self.rows if rows is None else rows
        inv=inventory.build_f4_inventory(r, checkpoint_sha_by_role=PINS,scan_truncated=truncated)
        return compare.compare_f4_cross_precision(r,inv)

    def test_full_metadata_matching_is_not_android_or_graph_approval(self):
        got=self.audit()
        self.assertEqual(got['status'],'MANIFEST_CROSS_PRECISION_MATCH_RUNTIME_NOT_VERIFIED')
        self.assertFalse(got['graph_checked'])
        self.assertFalse(got['android_approved'])
        self.assertFalse(got['f4_experiment_performed'])
        self.assertEqual([got['roles'][r]['role_state'] for r in ('MT','MZ')],
                         ['MANIFEST_MATCH_RUNTIME_NOT_VERIFIED']*2)

    def test_class_label_order_is_required_even_with_matching_checkpoint(self):
        self.rows[-1].manifest['labels']=['0','Q','1']
        got=self.audit()
        self.assertEqual(got['status'],'BLOCKED_METADATA_MISMATCH')
        self.assertIn('ordered_labels_differ',got['roles']['MZ']['checked_pairs'][0]['blockers'])

    def test_mismatch_confidence_fails(self):
        self.rows[-1].manifest['output']['confidence_threshold']=0.32
        got=self.audit()
        self.assertIn('output_confidence_threshold_differs',got['roles']['MZ']['checked_pairs'][0]['blockers'])

    def test_iou_float_tolerance(self):
        self.rows[-1].manifest['output']['iou_threshold']=0.45000001
        got=self.audit()
        self.assertEqual(got['status'],'MANIFEST_CROSS_PRECISION_MATCH_RUNTIME_NOT_VERIFIED')

    def test_output_normalization_mismatch_is_blocker(self):
        self.rows[-1].manifest['output']['normalized_coordinates']=True
        got=self.audit()
        self.assertIn('output_normalized_coordinates_differs',got['roles']['MZ']['checked_pairs'][0]['blockers'])

    def test_decoder_differences_require_runtime_parity_not_false_approval(self):
        self.rows[-1].manifest['output']['decoder']='raw_split_ltrb_scores_v1'
        got=self.audit()
        row=got['roles']['MZ']['checked_pairs'][0]
        self.assertEqual(row['state'],'RUNTIME_PARITY_REVIEW_REQUIRED')
        self.assertEqual(got['status'],'RUNTIME_PARITY_REVIEW_REQUIRED')
        self.assertIn('runtime_decoder_differs',row['decoder_review'])
        self.assertIn('Wymaga testu dekodowania',compare.describe_f4_cross_precision(got))

    def test_missing_threshold_is_inconclusive(self):
        del self.rows[-1].manifest['output']['iou_threshold']
        got=self.audit()
        self.assertEqual(got['status'],'INCONCLUSIVE_METADATA')
        self.assertIn('output_iou_threshold_missing',got['roles']['MZ']['checked_pairs'][0]['unknown'])

    def test_missing_input_layout_is_inconclusive(self):
        del self.rows[-1].manifest['input']['layout']
        got=self.audit()
        self.assertEqual(got['status'],'INCONCLUSIVE_METADATA')

    def test_source_checkpoint_tamper_is_not_comparable(self):
        self.rows[-1].manifest['source']['checkpoint_sha256']='9'*64
        got=self.audit()
        self.assertEqual(got['status'],'BLOCKED_NO_BOTH_PRECISIONS')

    def test_raw_research_tflite_not_treated_as_int8_package(self):
        raw=SimpleNamespace(kind='RESEARCH_TFLITE',role='MZ',schema='',
                 state='RESEARCH_ONLY_NOT_PACKAGED',path='/research/clean.tflite',manifest={})
        rows=[r for r in self.rows if not (r.role=='MZ' and r.manifest['variants'][0]['precision']=='int8')]+[raw]
        got=self.audit(rows)
        self.assertEqual(got['status'],'BLOCKED_NO_BOTH_PRECISIONS')

    def test_truncated_scan_never_comparable(self):
        self.assertEqual(self.audit(truncated=True)['status'],'BLOCKED_INCOMPLETE_SCAN')

    def test_multiple_good_candidates_ambiguous(self):
        alt=copy.deepcopy(self.rows[2]);alt.path='/test/MT_int8_alt.alprmodel';alt.name='MT_int8_alt.alprmodel'
        got=self.audit(self.rows+[alt])
        self.assertEqual(got['status'],'AMBIGUOUS_SELECT_EXPLICITLY')
        self.assertEqual(got['roles']['MT']['total_combinations'],2)

    def test_max_combinations_bounded(self):
        big=[r for r in self.rows if r.role=='MZ']
        for n in range(9):
            x=copy.deepcopy(self.rows[0]);x.path=f'/test/MT_fp32_{n}.alprmodel';big.append(x)
        for n in range(9):
            x=copy.deepcopy(self.rows[2]);x.path=f'/test/MT_int8_{n}.alprmodel';big.append(x)
        got=self.audit(big)
        self.assertEqual(got['status'],'INCONCLUSIVE_METADATA')
        self.assertEqual(got['roles']['MT']['role_state'],'INCOMPLETE_TOO_MANY_CANDIDATES')
        self.assertEqual(got['roles']['MT']['checked_pairs'],[])

    def test_role_specific_pose_keypoint_contract(self):
        self.rows[2].manifest['output']['keypoint_count']=5
        got=self.audit()
        self.assertIn('output_keypoint_count_differs',got['roles']['MT']['checked_pairs'][0]['blockers'])

    def test_gui_shows_audit_in_existing_f4_matrix_without_export(self):
        body=GUI.read_text(encoding='utf-8-sig')
        self.assertIn('compare_f4_cross_precision(rows, report)',body)
        self.assertIn('describe_f4_cross_precision(comparison)',body)
        self.assertIn('results.put((True, matrix_text, report))',body)
        self.assertIn('threading.Thread(target=scan, name="z4-f4-readonly-inventory"',body)

if __name__=='__main__':unittest.main()
