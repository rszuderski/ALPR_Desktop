#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4LIB023: exact zero must not be silently replaced by default conf/IoU."""
from __future__ import annotations
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui/z4_model_export.py'
MODULE = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))


def _isolated_function(name: str, environment: dict):
    matches = [node for node in ast.walk(MODULE) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f'Expected one {name}, got {len(matches)}')
    node = matches[0]
    code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), 'exec')
    exec(code, environment)
    return environment[name]


class V:
    def __init__(self, value): self.value = value
    def get(self): return self.value


class ExportRequest:
    def __init__(self, **fields): self.__dict__.update(fields)


class Builder:
    def _safe_model_export_slug(self, value, fallback='model'): return value.replace(' ', '-')
    def _build_mobile_export_metadata(self, run, target, weights):
        return {'model': {}, 'source': {}, 'training': {}, 'project': {}}
    def _json_safe_training_value(self, value): return value


def _environment():
    """Run the real AST-extracted GUI builders without importing Tk/Ultralytics."""
    env = {
        'math': math, 'Path': Path, 'CONFIG': SimpleNamespace(
            DEFAULT_IMG_SIZE=512, DEFAULT_CONFIDENCE=0.25, DEFAULT_IOU=0.45),
        'MobileExportRequest': ExportRequest, 'MobileExportError': ValueError,
        '_mobile_role_from_training_target': lambda x: 'plate',
        '_selected_export_formats_and_quantizations': lambda opt: (['litert'], ['fp32']),
        '_selected_export_format_quantizations': lambda opt: {'litert': ('fp32',)},
        '_mobile_export_candidate_params_millions': lambda candidate: (None, ''),
        '_mobile_export_candidate_training_provenance': lambda candidate: {},
        '_mobile_export_provenance_known_total': lambda profile: None,
        '_mobile_export_candidate_total_epochs': lambda candidate: 0,
        '_mobile_export_candidate_manifest_snapshot': lambda candidate: {},
        '_mobile_export_target_marker': lambda candidate: 'MT',
        '_state_reuses_existing_mt': lambda state: False,
        '_state_reuses_existing_mz': lambda state: False,
        'calibration_request_options': lambda options: {},
        'self': Builder(), 'imgsz_var': V(512), 'conf_var': V(0.25),
        'iou_var': V(0.45), 'calibration_var': V(''), 'destination_var': V(''),
    }
    _isolated_function('_mobile_export_threshold_or_default', env)
    return env


class ZeroThresholdRegression(unittest.TestCase):
    def test_explicit_zero_int_float_string(self):
        f = _environment()['_mobile_export_threshold_or_default']
        for v in (0, 0.0, '0', '0.0'):
            with self.subTest(value=v):
                self.assertEqual(f(v, .25), 0.0)

    def test_default_only_when_value_is_missing(self):
        f = _environment()['_mobile_export_threshold_or_default']
        for v in (None, '', '  '):
            with self.subTest(value=v):
                self.assertEqual(f(v, .45), .45)

    def test_preserve_existing_clamp(self):
        f = _environment()['_mobile_export_threshold_or_default']
        self.assertEqual(f(-.1, .25), 0.0)
        self.assertEqual(f(1.3, .25), 1.0)
        self.assertEqual(f(.12, .25), .12)

    def test_nonfinite_values_rejected(self):
        f = _environment()['_mobile_export_threshold_or_default']
        for v in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=v), self.assertRaises(ValueError):
                f(v, .25)

    def test_real_request_builder_keeps_zero_in_request_and_snapshot(self):
        env = _environment()
        make = _isolated_function('_build_mobile_export_request_for_candidate', env)
        candidate = {'role':'plate','target':'plate','best_weights':Path('best.pt'),
                     'img_size':512,'target_label':'MT','model_label':'MT-DAY'}
        opts = {'conf':V(0.0),'iou':V(0.0), 'imgsz':V(512),
                'calibration':V(''), 'calibration_choice':V('')}
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'model.alprmodel'
            for with_metadata in (False, True):
                with self.subTest(metadata=with_metadata):
                    request = make(candidate, destination=output,
                                   include_metadata=with_metadata, option_vars=opts)
                    self.assertEqual(request.confidence_threshold, 0.0)
                    self.assertEqual(request.iou_threshold, 0.0)
                    if with_metadata:
                        settings = request.metadata['export_settings']
                        self.assertEqual(settings['confidence_threshold'], 0.0)
                        self.assertEqual(settings['iou_threshold'], 0.0)

    def test_actual_slot_snapshot_keeps_zero(self):
        env = _environment()
        make = _isolated_function('_state_export_manifest_snapshot', env)
        state = {'marker':'MT', 'candidate':{'role':'plate','target':'plate'},
                 'conf':V(0.0), 'iou':V(0.0), 'imgsz':V(512)}
        snapshot = make(state)
        self.assertEqual(snapshot['confidence_threshold'], 0.0)
        self.assertEqual(snapshot['iou_threshold'], 0.0)

    def test_preflight_and_metadata_use_same_helper(self):
        functions = {node.name: node for node in ast.walk(MODULE) if isinstance(node, ast.FunctionDef)}
        for name in ('_build_mobile_export_request_for_candidate', '_state_export_manifest_snapshot'):
            node = functions[name]
            thresholds = [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_mobile_export_threshold_or_default']
            with self.subTest(function=name):
                self.assertGreaterEqual(len(thresholds), 2)
        content = SOURCE.read_text(encoding='utf-8-sig')
        self.assertNotIn('or CONFIG.DEFAULT_CONFIDENCE', content)
        self.assertNotIn('or CONFIG.DEFAULT_IOU', content)


if __name__ == '__main__':
    unittest.main()
