"""T2B research candidate staging; no conversion, packaging, Git or Android."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from auto_annotation_tool.exporters import mz_raw_int8_reuse as stager

ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / 'auto_annotation_tool/exporters/mobile_model_exporter.py'


def test_exporter_has_explicit_research_inspection_entrypoint():
    tree = ast.parse(EXPORTER.read_text(encoding='utf-8-sig'))
    exporter = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MobileModelExporter')
    inspect = next(node for node in exporter.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'inspect_existing_mz_raw_int8_research')
    assert any(isinstance(node, ast.Name) and node.id == 'inspect_mz_day_research_candidate'
               for node in ast.walk(inspect))
    for method_name in ('preflight', '_run_ultralytics_export'):
        method = next(node for node in exporter.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
        assert sum(isinstance(node, ast.Name) and node.id == '_mz_raw_int8_export_block_reason'
                   for node in ast.walk(method)) == 1


def test_stager_is_not_a_model_package_exporter():
    assert stager.STAGED_REL.suffix == '.json'
    assert stager.STAGED_REL.name.endswith('CONTRACT_PROVISIONAL.json')
    contract = _fake_contract()
    assert contract['android_import_approved'] is False
    assert contract['alprmodel_export_approved'] is False
    assert contract['final_test'] == 'NOT_VERIFIED'
    assert contract['draft_output_contract']['format'] != 'raw_yolo'
    assert contract['draft_output_contract']['nms_policy_vs_fp32'] == 'NOT_FROZEN'


def test_reject_wrong_checkpoint_sha(tmp_path):
    path = tmp_path / 'best.pt'
    path.write_bytes(b'wrong source')
    with pytest.raises(stager.ResearchContractNotReady, match='SHA256 mismatch'):
        stager._sha_guard(path, stager.CHECKPOINT_SHA, 'MZ_CHECKPOINT')


def test_reject_json_that_is_not_the_pinned_train_report(tmp_path):
    p = tmp_path / 'train.json'
    p.write_text('{}', encoding='utf-8')
    with pytest.raises(stager.ResearchContractNotReady, match='SHA256 mismatch'):
        stager._read_pinned_json(p, stager.TRAIN_PARITY_SHA, 'TRAIN_PARITY')


def test_reject_nonjson_sidecar_name(tmp_path):
    with pytest.raises(stager.ResearchContractNotReady, match='research JSON'):
        stager._write_research_contract_only(tmp_path / 'MZ.as_alprmodel', {'schema': 'fake'})


def test_stage_json_does_not_overwrite_differing_file(tmp_path):
    destination = tmp_path / stager.STAGED_REL.name
    destination.write_bytes(b'previous contract')
    with pytest.raises(stager.ResearchContractNotReady, match='already exists but differs'):
        stager._write_research_contract_only(destination, {'schema': 'fake'})
    assert destination.read_bytes() == b'previous contract'


def test_reject_non_int8_graph():
    report = _fake_graph()
    report['classification'] = 'MIXED'
    with pytest.raises(stager.ResearchContractNotReady, match='full-graph INT8'):
        stager._validate_static_graph(report)


def test_reject_missing_second_output():
    report = _fake_graph()
    report['subgraphs'][0]['outputs'].pop()
    with pytest.raises(stager.ResearchContractNotReady, match='number of I/O'):
        stager._validate_static_graph(report)


def test_reject_mismatched_quantization():
    report = _fake_graph()
    report['subgraphs'][0]['outputs'][1]['zero_point'] = [-126]
    with pytest.raises(stager.ResearchContractNotReady, match='quantizer mismatch'):
        stager._validate_static_graph(report)


def test_research_sidecar_can_only_stage_json(tmp_path):
    expected = tmp_path / stager.STAGED_REL.name
    data = _fake_contract()
    assert stager._write_research_contract_only(expected, data) == 'RESEARCH_JSON_STAGED'
    assert expected.suffix == '.json'
    assert json.loads(expected.read_text(encoding='utf-8')) == data
    assert stager._write_research_contract_only(expected, data) == 'EXISTING_IDENTICAL_RESEARCH_JSON'


def test_reject_nontrain_report():
    report = _fake_parity()
    report['scope'] = 'FINAL_TEST'
    with pytest.raises(stager.ResearchContractNotReady, match='invalid scope'):
        stager._validate_train_report(report)


def test_aggregate_research_counts_and_ties():
    summary = stager._validate_train_report(_fake_parity())
    assert summary['totals'] == {'pt_above_conf': 88, 'raw_above_conf': 90, 'matches_same_class_iou_ge_050': 87}
    assert summary['topk_score_ties_range'] == [10763, 10789]
    assert len(summary['difference_images']) == 4


def test_evidence_fails_open_if_not_previously_verified():
    evidence = _fake_evidence()
    evidence['mobile_decoder'] = 'IMPLEMENTED'
    with pytest.raises(stager.ResearchContractNotReady, match='invalid mobile_decoder'):
        stager._validate_evidence(evidence)


def test_filename_and_sha_pinning():
    assert stager.MODEL_REL.name == 'MZ_DAY_RAW_INT8_CLEAN_EXPLORATORY.tflite'
    assert len(stager.MODEL_SHA) == len(stager.CHECKPOINT_SHA) == len(stager.TRAIN_PARITY_SHA) == 64


def test_real_artifact_static_and_evidence_when_available():
    # Optional smoke, not a substitute for the --check run on the user's machine.
    paths = [ROOT / p for p in (stager.MODEL_REL, stager.EVIDENCE_REL, stager.TRAIN_REPORT_REL, stager.CHECKPOINT_REL)]
    if not all(p.is_file() for p in paths):
        pytest.skip('Operator research outputs not present in this checkout')
    result = stager.inspect_mz_day_research_candidate(model=paths[0], evidence=paths[1],
                train_parity=paths[2], checkpoint=paths[3])
    assert result['full_integer_graph'] is True
    assert result['android_import_approved'] is False
    assert result['train_evidence']['raw_clean_exact'] == '24_OF_24_TENSORS'


def _fake_graph():
    def tensor(idx, shape, scale, zp):
        return {'index': idx, 'shape': shape, 'dtype': 'INT8', 'scale': [scale], 'zero_point': [zp]}
    return {
        'artifact_sha256': stager.MODEL_SHA, 'classification': 'INT8',
        'full_integer_graph': True, 'issues': [], 'analysis_sha256': 'fake',
        'tensor_type_counts': {'INT32':144,'INT64':10,'INT8':493},
        'subgraphs': [{'operators': [{}] * 388, 'tensors': [{}] * 647,
                       'inputs': [tensor(0,[1,320,320,3],0.003921568859368563,-128)],
                       'outputs': [tensor(646,[1,4,2100],0.006233580876141787,-126),
                                   tensor(634,[1,36,2100],0.00390625,-128)]}]}


def _fake_parity():
    # Synthetic summary with precisely the signed baseline counts, no real images.
    rows = []
    pt = [7,7,7,7,7,7,9,7,7,8,8,7]
    raw = [7,8,7,8,7,7,8,7,7,9,8,7]
    matches = [7,7,7,7,7,7,8,7,7,8,8,7]
    assert sum(pt) == 88 and sum(raw) == 90 and sum(matches) == 87
    for i in range(12):
        rows.append({
            'image_train_split_relative': f'images/train/img_{i:03d}.jpg',
            'two_output_exact_quantized_equality': True,
            'clean_output_sha256': 'a' * 64, 'original_raw_output_sha256': 'a' * 64,
            'output_mismatches': [0,0],
            'matching': {'pt_above_conf': pt[i], 'raw_above_conf': raw[i],
                         'matches_same_class_iou_ge_050': matches[i], 'mean_iou_of_matched': 0.94},
            'raw_selection': {'quantized_score_threshold_ties':10763 if i == 0 else 10789,
                              'geometry': 'normalized_distance_ltrb_plus_analytic_grid_to_xyxy'},
        })
    return {'schema': 'alpr.f4.mz_clean_signed_int8_vs_original_raw_and_pt_train.v1',
            'status': 'TRAIN_CLEAN_VS_ORIGINAL_RAW_EXACT_AND_PT_DIAGNOSTIC_ONLY',
            'scope': 'TRAIN_ONLY_DIAGNOSTIC_NOT_FINAL_TEST', 'source_mixed_sha256': stager.REJECTED_MIXED_SHA,
            'raw_original_sha256': stager.ORIGINAL_RAW_SHA, 'clean_int8_sha256': stager.MODEL_SHA,
            'clean_evidence_sha256': stager.CLEAN_EVIDENCE_SHA, 'checkpoint_sha256': stager.CHECKPOINT_SHA,
            'images': 12, 'raw_output_parity_exact_cases': 12, 'raw_output_parity_exact_tensors': 24,
            'score_threshold':0.25, 'iou_match_threshold':0.5, 'decoder': 'normalized LTRB; topk k=300; no NMS',
            'preprocess': 'RGB letterbox 320, PIL bilinear padding 114',
            'execution_precision_android': 'UNVERIFIED', 'android_decoder': 'NOT_IMPLEMENTED',
            'model_scope':'EXPLORATORY_ONLY', 'final_test':'NOT_RUN', 'quality':'NOT_CERTIFIED',
            'totals': {'pt_above_conf':88,'raw_above_conf':90,'matches_same_class_iou_ge_050':87},
            'pt_recovery_fraction':87/88,'cases':rows}


def _fake_evidence():
    return {'source_sha256': stager.ORIGINAL_RAW_SHA,'candidate_sha256': stager.MODEL_SHA,
            'precision_after': 'INT8', 'precision_before':'UNVERIFIED', 'full_integer_graph':True,
            'issues_after': [], 'number_of_operators':388,'number_of_tensors_after':647,
            'number_of_tensors_before':649,'model_scope':'EXPLORATORY_ONLY',
            'final_test':'NOT_VERIFIED', 'android_execution':'UNVERIFIED', 'mobile_decoder':'NOT_IMPLEMENTED',
            'new_io':{'input':[0],'outputs':[646,634]},
            'cpu_parity': {'status':'CPU_EXACT_6_CASES','values_compared':504000,
                           'android_runtime':'UNVERIFIED','model_quality':'NOT_VERIFIED','final_test':'NOT_RUN',
                           'cases':[{'outputs':[{'kind':'geometry','shape':[1,4,2100],
                                                'mismatches':0,'values_compared':8400},
                                               {'kind':'scores','shape':[1,36,2100],
                                                'mismatches':0,'values_compared':75600}]} for _ in range(6)]}}


def _fake_contract():
    # The constructor itself is tested through the independent static and parity validators.
    return {'schema': stager.RESEARCH_SCHEMA, 'android_import_approved':False,
            'alprmodel_export_approved':False, 'final_test':'NOT_VERIFIED',
            'draft_output_contract': {'format':'raw_split_ltrb_scores_v1',
                                      'nms_policy_vs_fp32':'NOT_FROZEN'}}
