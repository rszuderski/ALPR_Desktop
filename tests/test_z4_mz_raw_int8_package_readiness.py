"""Z4LIB025 — no approval elevation and no accidental re-conversion."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / 'auto_annotation_tool/exporters/mz_raw_int8_package_readiness.py'
EXPORTER = REPO / 'auto_annotation_tool/exporters/mobile_model_exporter.py'
spec = importlib.util.spec_from_file_location('z4lib025_gate_test', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_contract():
    io = {
        'input': {'shape': [1, 320, 320, 3], 'data_type': 'INT8', 'port': 0, 'tensor_index': 0},
        'outputs': [
            {'port': 0, 'tensor_index': 646, 'role': 'ltrb_normalized_distances_after_dfl',
             'shape': [1, 4, 2100], 'data_type': 'INT8',
             'quantization': {'scale': .006233580876141787, 'zero_point': -126}},
            {'port': 1, 'tensor_index': 634, 'role': 'sigmoid_class_scores',
             'shape': [1, 36, 2100], 'data_type': 'INT8',
             'quantization': {'scale': .00390625, 'zero_point': -128}},
        ],
    }
    return {
        'schema': module.RESEARCH_SCHEMA,
        'status': 'EXPLORATORY_ONLY',
        'contract_state': 'PROVISIONAL_T2B_NO_ANDROID_DECODER',
        'full_integer_graph': True,
        'graph_static_precision': 'INT8',
        'execution_precision_android': 'UNVERIFIED',
        'android_decoder': 'NOT_IMPLEMENTED',
        'android_import_approved': False,
        'alprmodel_export_approved': False,
        'final_test': 'NOT_VERIFIED',
        'draft_output_contract': {
            'format': 'raw_split_ltrb_scores_v1',
            'number_of_classes': 36,
            'approval': 'DRAFT_NOT_ANDROID_COMPATIBLE',
            'decoder': 'yolo26_detect_ltrb_scores_topk_v1',
            'labels_class_order': 'REQUIRES_CHECKPOINT_OR_APPROVED_MANIFEST',
            'calibration_manifest': 'REQUIRES_LIVE_REVALIDATION_BEFORE_PACKAGE',
            'io': io,
        },
        'graph': {'operator_count': 388, 'tensor_count': 647},
        'sources': {key: {'path': rel, 'sha256': sha}
                    for key, (rel, sha, _arc) in module.SOURCES.items()},
    }


class StageUnitTests(unittest.TestCase):
    def setUp(self):
        self.contract = fixture_contract()

    def test_correct_provisional_contract_accepted_for_audit_only(self):
        module._validate_staged_flags(self.contract)

    def test_cannot_replace_research_status_with_approved(self):
        for key, value in [('status', 'READY'), ('android_import_approved', True),
                           ('alprmodel_export_approved', True), ('final_test', 'PASS')]:
            with self.subTest(key=key):
                doc = copy.deepcopy(self.contract)
                doc[key] = value
                with self.assertRaises(module.MzPackagingNotReady):
                    module._validate_staged_flags(doc)

    def test_output_shape_and_decoder_are_pinned(self):
        for change in ('shape', 'decoder', 'classes', 'quantization'):
            with self.subTest(change=change):
                doc = copy.deepcopy(self.contract)
                output = doc['draft_output_contract']
                if change == 'shape':
                    output['io']['outputs'][1]['shape'] = [1, 35, 2100]
                elif change == 'decoder':
                    output['decoder'] = 'ultralytics_detect_raw_v1'
                elif change == 'classes':
                    output['number_of_classes'] = 35
                else:
                    output['io']['outputs'][1]['quantization']['zero_point'] = 0
                with self.assertRaises(module.MzPackagingNotReady):
                    module._validate_staged_flags(doc)

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(module.MzPackagingNotReady):
                module._safe_managed_path(Path(root), '../escape', exists=False)

    def test_redirected_research_model_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            top = Path(root)
            (top/'actual').write_bytes(b'123')
            try:
                (top/'link').symlink_to(top/'actual')
            except (OSError, NotImplementedError):
                self.skipTest('No symlink permissions')
            with self.assertRaisesRegex(module.MzPackagingNotReady, 'REDIRECTED_PATH'):
                module._safe_managed_path(top, 'link')


class LiveStageTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.contract = fixture_contract()
        staged = self.root / module.STAGED_REL
        staged.parent.mkdir(parents=True)
        self.staged = staged
        self.source_bytes = {}
        replacements = {}
        for key, (relative, _expected, arc) in module.SOURCES.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            value = b'FAKE_FIXTURE_' + key.encode()
            path.write_bytes(value)
            self.source_bytes[key] = value
            replacements[key] = (relative, hashlib.sha256(value).hexdigest(), arc)
            self.contract['sources'][key]['sha256'] = hashlib.sha256(value).hexdigest()
        self.staged.write_text(json.dumps(self.contract), encoding='utf-8')
        self.pinned = hashlib.sha256(self.staged.read_bytes()).hexdigest()
        self.patch_source = mock.patch.object(module, 'SOURCES', replacements)
        self.patch_sha = mock.patch.object(module, 'STAGED_SHA256', self.pinned)
        self.patch_source.start()
        self.patch_sha.start()
        self.addCleanup(self.patch_source.stop)
        self.addCleanup(self.patch_sha.stop)
        self.model_path = self.root / module.SOURCES['clean_raw_tflite'][0]
        self.checkpoint_path = self.root / module.SOURCES['checkpoint'][0]

    def inspect(self, **kwargs):
        return module.assess_research_mz_int8(
            self.root, _research_inspector=lambda **_: copy.deepcopy(self.contract), **kwargs,
        )

    def test_live_audit_is_read_only_and_never_approves(self):
        original_files = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result = self.inspect()
        self.assertEqual(result['research_artifact_integrity'], 'PASS_PINNED_SHA_AND_LIVE_GRAPH_EVIDENCE')
        self.assertEqual(result['status'], 'BLOCKED_NOT_ANDROID_DEPLOYABLE')
        self.assertFalse(result['android_import_approved'])
        self.assertFalse(result['alprmodel_export_approved'])
        self.assertIn('LIVE_TRAIN_CALIBRATION_MANIFEST_REQUIRED', result['blockers'])
        self.assertIn('CHECKPOINT_36_CLASS_ORDER_NOT_VERIFIED', result['blockers'])
        self.assertFalse(result['conversion_performed'])
        self.assertFalse(result['packaging_performed'])
        self.assertEqual(original_files, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_mismatching_live_report_is_rejected(self):
        def forged_inspector(**_):
            modified = copy.deepcopy(self.contract)
            modified['status'] = 'READY'
            return modified
        with self.assertRaisesRegex(module.MzPackagingNotReady, 'EVIDENCE_DIFFERS'):
            module.assess_research_mz_int8(self.root, _research_inspector=forged_inspector)

    def test_mutation_during_live_inspection_rejected(self):
        def modifying_inspector(**_):
            self.model_path.write_bytes(b'changed')
            return copy.deepcopy(self.contract)
        with self.assertRaisesRegex(module.MzPackagingNotReady, 'SOURCE_CHANGED'):
            module.assess_research_mz_int8(self.root, _research_inspector=modifying_inspector)

    def test_calibration_must_be_explicit_and_live_validated(self):
        relative = Path('output/mz_day_int8_raw_probe/calibration.json')
        (self.root/relative).write_text('{}')
        def validator(path, **kwargs):
            self.assertEqual(path, self.root/relative)
            self.assertEqual(kwargs['role'], 'character')
            self.assertEqual(kwargs['image_size'], 320)
            self.assertEqual(kwargs['confidence_threshold'], .25)
            return {'status': 'EXPLORATORY_ONLY', 'input_fingerprint': 'a'*64}
        result = self.inspect(calibration_manifest=relative, _calibration_validator=validator)
        self.assertEqual(result['calibration']['status'], 'EXPLORATORY_ONLY_REVALIDATED')
        self.assertIn('ANDROID_DUAL_OUTPUT_DECODER_NOT_APPROVED', result['blockers'])
        self.assertNotIn('LIVE_TRAIN_CALIBRATION_MANIFEST_REQUIRED', result['blockers'])

    def test_calibration_status_cannot_be_forged(self):
        relative = Path('output/mz_day_int8_raw_probe/calibration.json')
        (self.root/relative).write_text('{}')
        with self.assertRaisesRegex(module.MzPackagingNotReady, 'EVIDENCE_INCOMPLETE'):
            self.inspect(calibration_manifest=relative,
                         _calibration_validator=lambda *_, **__: {'status': 'READY', 'input_fingerprint': 'a'*64})

    def test_labels_must_come_from_checkpoint_reader(self):
        with self.assertRaisesRegex(module.MzPackagingNotReady, "CHECKPOINT_36_CLASS_ORDER_NOT_VERIFIED"):
            self.inspect(verify_checkpoint_labels=True, _labels_reader=lambda _: list('ABCDEFGHIJKLMN') + [str(i) for i in range(10)])

    def test_labels_36_verified_but_android_still_blocked(self):
        labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + list('0123456789')
        result = self.inspect(verify_checkpoint_labels=True, _labels_reader=lambda _: labels)
        self.assertEqual(result['class_order']['status'], 'CHECKPOINT_VERIFIED')
        self.assertEqual(result['class_order']['ordered_labels'], labels)
        self.assertIn('ALPRMODEL_EXPORT_NOT_APPROVED_BY_RESEARCH_CONTRACT', result['blockers'])

    def test_research_bundle_has_distinct_extension_and_no_overwrite(self):
        with mock.patch.object(module, 'assess_research_mz_int8', return_value={'model_sha256': 'a'*64}):
            result = module.stage_research_bundle(self.root)
            self.assertFalse(result['android_import_approved'])
            dest = self.root/module.BUNDLE_REL
            self.assertTrue(dest.is_file())
            self.assertEqual(dest.suffix, '.alprresearch')
            self.assertEqual(list(self.root.rglob('*.alprmodel')), [])
            with zipfile.ZipFile(dest) as zip_file:
                self.assertEqual(set(zip_file.namelist()),
                                 {'research_contract.json', 'model.tflite', 'clean_evidence.json', 'train_parity.json'})
                self.assertEqual(zip_file.read('model.tflite'), self.source_bytes['clean_raw_tflite'])
            with self.assertRaisesRegex(module.MzPackagingNotReady, 'ALREADY_EXISTS'):
                module.stage_research_bundle(self.root)

    def test_archive_race_cannot_replace_existing_destination(self):
        with mock.patch.object(module, 'assess_research_mz_int8', return_value={'model_sha256': 'a'*64}):
            with mock.patch.object(module.os, 'link', side_effect=FileExistsError('race')):
                with self.assertRaisesRegex(module.MzPackagingNotReady, 'DESTINATION_RACE'):
                    module.stage_research_bundle(self.root)
            self.assertFalse((self.root/module.BUNDLE_REL).exists())

    def test_module_does_not_change_normal_export_blocker(self):
        source = EXPORTER.read_text(encoding='utf-8-sig')
        self.assertIn('MZ_RAW_INT8_CONTRACT_NOT_READY', source)
        self.assertIn('def assess_existing_mz_day_int8_packaging(', source)
        self.assertIn('def _mz_raw_int8_export_block_reason(', source)


if __name__ == '__main__':
    unittest.main()
