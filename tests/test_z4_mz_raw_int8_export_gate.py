"""Direct regression checks of MZ INT8 preconversion gate; no exports."""
import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'auto_annotation_tool/exporters/mobile_model_exporter.py'
GUI = ROOT / 'auto_annotation_tool/gui/z4_model_export.py'
DIGEST = '631bb1453554cdb6b15eca724baa392e2acffb350899107f892dc11f937dbbf8'


def _load_gate(monkeypatch, digest='unrelated'):
    tree = ast.parse(EXPORT.read_text(encoding='utf-8-sig'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_mz_raw_int8_export_block_reason')
    constant = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == '_MZ_DAY_YOLO26S_KNOWN_MIXED_CHECKPOINT_SHA256'
                            for t in n.targets))
    scope = {
        'MobileExportRequest': object,
        'MobileExportError': ValueError,
        'Path': Path,
        'exploratory': SimpleNamespace(MODE='exploratory'),
        'sha256_file': lambda _p: digest,
        '_normalize_role': lambda role: {'char': 'character', 'character': 'character',
                                          'plate': 'plate', 'vehicle': 'vehicle'}.get(role, role),
        '_request_format_quantization_map': lambda req: req.format_quantizations,
    }
    exec(compile(ast.Module(body=[constant, fn], type_ignores=[]), str(EXPORT), 'exec'), scope)
    return scope['_mz_raw_int8_export_block_reason']


def _request(role='character', mode='exploratory', precision=('int8',), path=None):
    return SimpleNamespace(role=role, calibration_mode=mode,
        format_quantizations={'litert': precision}, checkpoint=path or Path(__file__))


def test_research_mz_int8_blocks_even_if_checksum_not_the_known_case(monkeypatch):
    fn = _load_gate(monkeypatch)
    message = fn(_request())
    assert 'MZ_RAW_INT8_CONTRACT_NOT_READY' in message
    assert 'FINAL_TEST_NOT_VERIFIED' in message
    assert 'fallbacku FP32' in message


def test_known_mz_mixed_checkpoint_blocks_even_in_strict_mode(monkeypatch):
    fn = _load_gate(monkeypatch, DIGEST)
    message = fn(_request(mode='strict'))
    assert 'znany checkpoint MZ-DAY' in message


def test_character_fp32_not_blocked(monkeypatch):
    fn = _load_gate(monkeypatch, DIGEST)
    assert fn(_request(precision=('fp32',))) == ''


def test_plate_int8_not_blocked(monkeypatch):
    fn = _load_gate(monkeypatch, DIGEST)
    assert fn(_request(role='plate')) == ''


def test_other_strict_character_int8_not_globally_disabled(monkeypatch):
    fn = _load_gate(monkeypatch)
    assert fn(_request(mode='strict')) == ''


def test_non_litert_character_not_blocked(monkeypatch):
    fn = _load_gate(monkeypatch)
    request = _request()
    request.format_quantizations = {'onnx': ('int8',)}
    assert fn(request) == ''


def test_preflight_wires_gate_and_checks_it_before_conversion():
    tree = ast.parse(EXPORT.read_text(encoding='utf-8-sig'))
    exporter = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MobileModelExporter')
    preflight = next(n for n in exporter.body if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
    runner = next(n for n in exporter.body if isinstance(n, ast.FunctionDef) and n.name == '_run_ultralytics_export')
    for method in (preflight, runner):
        calls = [n for n in ast.walk(method) if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Name) and n.func.id == '_mz_raw_int8_export_block_reason']
        assert len(calls) == 1, method.name
    assert next(n for n in exporter.body if isinstance(n, ast.FunctionDef) and n.name == 'export')


def test_gui_does_not_conflate_preflight_with_proven_precision():
    source = GUI.read_text(encoding='utf-8-sig')
    assert 'Eksport gotowy. Możesz zapisać gotowy plik .alprmodel.' not in source
    assert 'Kontrola wstępna PASS. Warianty świeżego eksportu wymagają weryfikacji po konwersji.' in source


def test_no_fp32_fallback_inside_gate():
    tree = ast.parse(EXPORT.read_text(encoding='utf-8-sig'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_mz_raw_int8_export_block_reason')
    assert not any(isinstance(n, (ast.Assign, ast.AugAssign, ast.NamedExpr))
                   and isinstance(getattr(n, 'target', None), ast.Attribute)
                   for n in ast.walk(fn))
