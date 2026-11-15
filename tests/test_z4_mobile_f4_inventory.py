#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4LIB015: F4 inventory is SHA-pinned and manifest-only, never Android approval."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/'auto_annotation_tool/gui/z4_mobile_f4_inventory.py'
GUI=ROOT/'auto_annotation_tool/gui/z4_model_export.py'
spec=importlib.util.spec_from_file_location('z4_f4_inventory_unit',HELPER)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
MT_SHA='1'*64
MZ_SHA='2'*64


def row(role, precision, *, sha=None, kind='MODEL', bad_input=False,
        status='DISCOVERED_NOT_VERIFIED', variants=None, calibration_mode='strict'):
    actual = 'plate' if role=='MT' else 'character'
    graph = {'verified': True} if precision=='int8' else {'verified': True}
    v = (variants if variants is not None else [{'runtime':'tflite','precision':precision,'graph_precision':graph}])
    labels = ['plate'] if role=='MT' else ['A', 'B']
    manifest = {'schema':'alpr.model.v1','role':actual,
       'task':'pose' if role=='MT' else 'detect','source':{'checkpoint_sha256':sha or (MT_SHA if role=='MT' else MZ_SHA)},
       'input': {'width':512 if role=='MT' else 320,'height':512 if role=='MT' else 320,
                 'data_type': ('FLOAT32' if precision=='int8' else 'INT8') if bad_input else ('INT8' if precision=='int8' else 'FLOAT32')},
       'output':{'class_count':len(labels),'keypoint_count':4 if role=='MT' else 0}, 'labels':labels,
       'variants':v}
    if precision=='int8':
        manifest['calibration_evidence']={'file':'calib.json','mode':calibration_mode,
                 'status':'READY' if calibration_mode=='strict' else 'EXPLORATORY_ONLY'}
        if calibration_mode=='exploratory':
            manifest['research_calibration']={'final_test_status':'NOT_VERIFIED'}
    return SimpleNamespace(role=role,kind=kind,schema='alpr.model.v1',state=status,error='',
                           manifest=manifest,path=f'/test/{role}_{precision}_{sha or "a"}.alprmodel',
                           name=f'{role}_{precision}.alprmodel')


class InventoryTests(unittest.TestCase):
    def report(self, rows, **kwargs):
        return mod.build_f4_inventory(rows,checkpoint_sha_by_role=kwargs.pop('pins',{'MT':MT_SHA,'MZ':MZ_SHA}),**kwargs)

    def test_int8_pair_manifest_candidates_not_approved(self):
        report=self.report([row('MT','int8'),row('MZ','int8')])
        pair=report['pairs']['int8']
        self.assertEqual(pair['pair_count_manifest_only'],1)
        self.assertEqual(pair['state'],'MANIFEST_CANDIDATES_REQUIRE_FULL_VALIDATION')
        self.assertFalse(report['android_import_approved'])
        self.assertFalse(report['f4_experiment_performed'])

    def test_fp32_pair_independent_from_int8(self):
        report=self.report([row('MT','fp32'),row('MZ','fp32')])
        self.assertEqual(report['pairs']['fp32']['pair_count_manifest_only'],1)
        self.assertEqual(report['pairs']['int8']['state'],'BLOCKED_NO_MATCHING_PACKAGE')

    def test_missing_mz_int8_blocks_even_with_raw_research(self):
        raw=SimpleNamespace(kind='RESEARCH_TFLITE',role='MZ',path='/x/clean.tflite',state='RESEARCH_ONLY_NOT_PACKAGED')
        report=self.report([row('MT','int8'),raw])
        self.assertEqual(report['pairs']['int8']['missing_roles'],['MZ'])
        self.assertEqual(len(report['research_mz_tflite']),1)
        self.assertIn('nie .alprmodel',mod.describe_f4_inventory(report))

    def test_wrong_checkpoint_sha_not_a_match(self):
        report=self.report([row('MT','int8',sha='3'*64),row('MZ','int8')])
        self.assertEqual(report['pairs']['int8']['missing_roles'],['MT'])

    def test_incomplete_scan_is_never_ready(self):
        report=self.report([row('MT','int8'),row('MZ','int8')],scan_truncated=True)
        self.assertEqual(report['pairs']['int8']['state'],'SCAN_INCOMPLETE_NO_READINESS_CLAIM')

    def test_fp32_masquerading_as_int8_rejected(self):
        report=self.report([row('MT','int8',bad_input=True),row('MZ','int8')])
        self.assertEqual(report['pairs']['int8']['pair_count_manifest_only'],0)

    def test_mixed_package_cannot_count_as_single_precision(self):
        mixed=[{'runtime':'tflite','precision':'int8','graph_precision':{'a':1}},
               {'runtime':'tflite','precision':'fp32'}]
        report=self.report([row('MT','int8',variants=mixed),row('MZ','int8')])
        self.assertEqual(report['pairs']['int8']['pair_count_manifest_only'],0)

    def test_not_graph_proven_int8_rejected(self):
        variants=[{'runtime':'tflite','precision':'int8','graph_precision':{}}]
        report=self.report([row('MT','int8',variants=variants),row('MZ','int8')])
        self.assertEqual(report['pairs']['int8']['pair_count_manifest_only'],0)

    def test_invalid_artifact_state_excluded(self):
        report=self.report([row('MT','int8',status='REGISTERED_SHA_MISMATCH'),row('MZ','int8')])
        self.assertEqual(report['pairs']['int8']['missing_roles'],['MT'])

    def test_missing_checkpoint_reference_fails_closed(self):
        report=self.report([row('MT','int8'),row('MZ','int8')],pins={'MT':MT_SHA,'MZ':'not-a-sha'})
        self.assertEqual(report['pairs']['int8']['state'],'BLOCKED_NO_SELECTED_CHECKPOINT_SHA')

    def test_exploratory_int8_labeled_not_final_test_approved(self):
        report=self.report([row('MT','int8'),row('MZ','int8',calibration_mode='exploratory')])
        item=report['pairs']['int8']['candidates']['MZ'][0]
        self.assertEqual(item['calibration'],'EXPLORATORY_ONLY')
        self.assertFalse(report['android_import_approved'])

    def test_compound_package_never_used_as_single_model(self):
        r=row('MT','fp32',kind='PACKAGE')
        report=self.report([r,row('MZ','fp32')])
        self.assertEqual(report['pairs']['fp32']['pair_count_manifest_only'],0)

    def test_metadata_rejection_does_not_read_image_or_pt_files(self):
        report=self.report([row('MT','fp32',bad_input=True),row('MZ','fp32')])
        self.assertEqual(report['pairs']['fp32']['pair_count_manifest_only'],0)

    def test_gui_exposes_read_only_inventory_button(self):
        code=GUI.read_text(encoding='utf-8-sig')
        self.assertIn('Sprawdź gotowe paczki (FP32 / INT8)',code)
        self.assertIn('from .z4_mobile_f4_inventory import build_f4_inventory, describe_f4_inventory',code)
        self.assertIn('threading.Thread(target=scan, name="z4-f4-readonly-inventory", daemon=True)',code)
        self.assertIn('scan_truncated=truncated',code)

if __name__=='__main__': unittest.main()
