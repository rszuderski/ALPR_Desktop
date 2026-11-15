#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4LIB016: full fail-closed tests for manifest-only inventory pair selection."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = ROOT / 'auto_annotation_tool' / 'gui'
PKG = '_z4lib016_test_package'
if PKG not in sys.modules:
    fake = types.ModuleType(PKG)
    fake.__path__ = [str(GUI_DIR)]
    sys.modules[PKG] = fake


def load(name):
    full_name = PKG + '.' + name
    spec = importlib.util.spec_from_file_location(full_name, GUI_DIR / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


inventory = load('z4_mobile_f4_inventory')
registry = load('z4_mobile_artifact_registry')
pairs = load('z4_mobile_f4_pair_selection')


class PairChoiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoints = {}
        self.packages = {}
        for role in ('MT','MZ'):
            ckpt = self.root / f'{role}_best.pt'
            ckpt.write_bytes((role + '_weights').encode())
            self.checkpoints[role] = ckpt
        self.imgsz = {'MT': 512, 'MZ': 320}
        self.create_pair('int8')
        self.create_pair('fp32')
        self.report = self.scan()

    def manifest(self, role: str, precision: str):
        size = self.imgsz[role]
        labels = ['plate'] if role == 'MT' else ['A','B']
        manifest = {
            'schema': 'alpr.model.v1',
            'role': 'plate' if role == 'MT' else 'character',
            'task': 'pose' if role == 'MT' else 'detect',
            'model_id': f'{role}-{precision}',
            'source': {'checkpoint_sha256': registry.sha256_file(self.checkpoints[role])},
            'input': {'width':size,'height':size,'data_type':'INT8' if precision=='int8' else 'FLOAT32'},
            'output': {'class_count':len(labels),'keypoint_count':4 if role=='MT' else 0},
            'labels': labels,
            'variants': [{'runtime':'tflite','precision':precision,'graph_precision':{'artifact_sha256':'a'*64}}],
        }
        if precision == 'int8':
            manifest['calibration_evidence'] = {'file':'calibration.json','mode':'exploratory','status':'EXPLORATORY_ONLY'}
            manifest['research_calibration'] = {'final_test_status':'NOT_VERIFIED'}
        return manifest

    def write_package(self, role: str, precision: str, manifest=None, path=None):
        dest = path or (self.root / f'{role}_{precision}.alprmodel')
        with zipfile.ZipFile(dest,'w') as out:
            out.writestr('manifest.json',json.dumps(manifest or self.manifest(role,precision)))
        return dest

    def create_pair(self, precision):
        self.packages[precision] = {role:self.write_package(role,precision) for role in ('MT','MZ')}

    def scan(self, *, truncated=False):
        rows = [registry.read_archive(p) for role_map in self.packages.values() for p in role_map.values() if p.is_file()]
        pins = {role:registry.sha256_file(self.checkpoints[role]) for role in ('MT','MZ')}
        return inventory.build_f4_inventory(rows,checkpoint_sha_by_role=pins,scan_truncated=truncated)

    def run_choice(self, precision='int8', **kwargs):
        return pairs.validate_f4_pair_choice(
            kwargs.get('report',self.report), precision=precision,
            package_paths=kwargs.get('paths', self.packages.get(precision, {})),
            checkpoint_paths=kwargs.get('checkpoints', self.checkpoints),
            expected_image_size=kwargs.get('sizes', self.imgsz),
            read_archive=kwargs.get('reader', registry.read_archive),
            sha256_file=kwargs.get('hasher', registry.sha256_file),
        )

    def test_int8_pair_selected_but_not_graph_or_android_approved(self):
        result = self.run_choice()
        self.assertEqual(result['precision'],'int8')
        self.assertEqual(result['roles']['MT']['image_size'],512)
        self.assertEqual(result['roles']['MZ']['image_size'],320)
        self.assertEqual(result['roles']['MT']['package_sha256_at_selection'],registry.sha256_file(self.packages['int8']['MT']))
        self.assertFalse(result['desktop_graph_verified'])
        self.assertFalse(result['android_import_approved'])
        self.assertFalse(result['export_started'])

    def test_fp32_pair_independent(self):
        result = self.run_choice('fp32')
        self.assertEqual(result['precision'],'fp32')
        self.assertEqual(Path(result['roles']['MZ']['path']),self.packages['fp32']['MZ'].resolve())

    def test_reject_truncated_scan(self):
        with self.assertRaisesRegex(pairs.F4PairBindingError, 'niekompletny'):
            self.run_choice(report=self.scan(truncated=True))

    def test_reject_raw_tflite_even_if_artifact_exists(self):
        raw=self.root/'MZ_int8.tflite';raw.write_bytes(b'raw-graph')
        with self.assertRaisesRegex(pairs.F4PairBindingError,'alprmodel'):
            self.run_choice(paths={**self.packages['int8'],'MZ':raw})

    def test_reject_different_model_from_outside_inventory(self):
        outside=self.write_package('MT','int8',path=self.root/'outside.alprmodel')
        with self.assertRaisesRegex(pairs.F4PairBindingError,'paczka nie pochodzi z aktualnego przeglądu modeli'):
            self.run_choice(paths={**self.packages['int8'],'MT':outside})

    def test_reject_checkpoint_changed_after_scan(self):
        self.checkpoints['MT'].write_bytes(b'changed')
        with self.assertRaisesRegex(pairs.F4PairBindingError,'checkpoint zmienił'):
            self.run_choice()

    def test_reject_mz_checkpoint_changed_after_scan(self):
        self.checkpoints['MZ'].write_bytes(b'changed')
        with self.assertRaisesRegex(pairs.F4PairBindingError,'MZ: checkpoint zmienił'):
            self.run_choice()

    def test_reject_input_size_drift(self):
        with self.assertRaisesRegex(pairs.F4PairBindingError,'imgsz'):
            self.run_choice(sizes={'MT':320,'MZ':320})

    def test_reject_modified_manifest_after_scan(self):
        manifest=self.manifest('MZ','int8')
        manifest['role']='vehicle'
        self.write_package('MZ','int8',manifest=manifest)
        with self.assertRaisesRegex(pairs.F4PairBindingError,'ponownej kontroli manifestu'):
            self.run_choice()

    def test_reject_mixed_precision_after_scan(self):
        manifest=self.manifest('MZ','int8')
        manifest['variants'].append({'runtime':'tflite','precision':'fp32'})
        self.write_package('MZ','int8',manifest=manifest)
        with self.assertRaisesRegex(pairs.F4PairBindingError,'not_single_variant'):
            self.run_choice()

    def test_reject_wrong_package_role_swapped(self):
        paths={'MT':self.packages['int8']['MZ'],'MZ':self.packages['int8']['MT']}
        with self.assertRaisesRegex(pairs.F4PairBindingError,'paczka nie pochodzi z aktualnego przeglądu modeli'):
            self.run_choice(paths=paths)

    def test_missing_graph_evidence_blocks_before_graph_validation(self):
        manifest=self.manifest('MZ','int8')
        manifest['variants'][0]['graph_precision'] = {}
        self.write_package('MZ','int8',manifest=manifest)
        with self.assertRaisesRegex(pairs.F4PairBindingError,'missing_static_graph_evidence'):
            self.run_choice()

    def test_reject_invalid_record_schema(self):
        with self.assertRaisesRegex(pairs.F4PairBindingError,'wiarygodnego wyniku'):
            self.run_choice(report={'schema':'foreign'})

    def test_reject_missing_role_candidate(self):
        report=copy.deepcopy(self.report)
        report['pairs']['int8']['state']='BLOCKED_NO_MATCHING_PACKAGE'
        with self.assertRaisesRegex(pairs.F4PairBindingError,'Nie znaleziono obu modeli'):
            self.run_choice(report=report)

    def test_reject_unknown_precision(self):
        with self.assertRaisesRegex(pairs.F4PairBindingError,'FP32 albo INT8'):
            self.run_choice('int4')

    def test_reject_suspicious_sha_result(self):
        def hasher(path):
            return 'invalid'
        with self.assertRaisesRegex(pairs.F4PairBindingError,'checkpoint zmienił'):
            self.run_choice(hasher=hasher)

    def test_reject_package_changed_during_manifest_read(self):
        target=self.packages['int8']['MZ']
        def reader(path):
            out=registry.read_archive(path)
            if Path(path)==target:
                target.write_bytes(target.read_bytes()+b'padding')
            return out
        with self.assertRaisesRegex(pairs.F4PairBindingError,'zmieniła się podczas kontroli'):
            self.run_choice(reader=reader)

    def test_reject_foreign_reader_path(self):
        def reader(path):
            from dataclasses import replace
            row=registry.read_archive(path)
            if Path(path).name.startswith('MT'):
                return replace(row,path='/tmp/notselected.alprmodel')
            return row
        with self.assertRaisesRegex(pairs.F4PairBindingError,'innego pliku'):
            self.run_choice(reader=reader)

    def test_reject_missing_checkpoint(self):
        self.checkpoints['MZ'].unlink()
        with self.assertRaisesRegex(pairs.F4PairBindingError,'nie istnieje'):
            self.run_choice()

    def test_reject_missing_package(self):
        self.packages['int8']['MT'].unlink()
        with self.assertRaisesRegex(pairs.F4PairBindingError,'nie istnieje'):
            self.run_choice()

    def test_research_raw_is_never_selected(self):
        self.packages['int8']['MZ'].unlink()
        report=self.scan()
        self.assertEqual(report['pairs']['int8']['missing_roles'],['MZ'])
        with self.assertRaises(pairs.F4PairBindingError):
            self.run_choice(report=report)

    def test_report_not_mutated(self):
        snapshot=copy.deepcopy(self.report)
        self.run_choice()
        self.assertEqual(self.report,snapshot)

    def test_gui_has_selector_and_no_export_action(self):
        gui=(GUI_DIR/'z4_model_export.py').read_text(encoding='utf-8-sig')
        snippet=gui.split('        def show_f4_inventory() -> None:',1)[1].split('        checks_shell = tk.Frame(',1)[0]
        for token in ('validate_f4_pair_choice(', 'fmt_dropdown.bind(', 'selection_notice',
                      'Przypisz MT + MZ do eksportu (bez konwersji)', 'f4_no_conversion.set(True)',
                      'read_archive=read_archive', 'sha256_file=sha256_file', '_sync_executor_buttons()'):
            self.assertIn(token,snippet)
        self.assertNotIn('exporter.export(',snippet)
        self.assertNotIn('package_exporter.export(',snippet)
        self.assertNotIn('write_bytes(',snippet)

    def test_f4_selection_does_not_pretend_graph_check(self):
        result=self.run_choice()
        for role in ('MT','MZ'):
            self.assertEqual(result['roles'][role]['authority'],'MANIFEST_RECHECK_ONLY_NOT_GRAPH_VALIDATED')


if __name__ == '__main__': unittest.main()
