"""Read-only identity from existing Android metadata, independent of F4's registry.

This is not a parser for the pending Android scene-evidence extension.  It never
rewrites source metadata, infers a variant from models, or certifies scene GT.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from .mobile_mz_input_evidence import DIAGNOSTIC_PURPOSES, is_diagnostic_purpose


def _dict(value):
    return value if isinstance(value, dict) else {}


def from_bundle(bundle):
    raw = _dict(getattr(bundle, 'report_payload', None))
    index = _dict(getattr(bundle, 'experiment_session', None))
    collection = _dict(getattr(bundle, 'collection_session', None))
    experiment = _dict(raw.get('experiment'))
    recorded = _dict(experiment.get('session'))
    configs = [_dict(raw.get('research_execution_config')),
               _dict(collection.get('execution')),
               _dict(experiment.get('effective_execution_config'))]
    sources = [collection, index, recorded, experiment, *configs]
    issues = []

    def field(name, aliases=(), normalizer=str):
        values = []
        for source in sources:
            for key in (name, *aliases):
                value = source.get(key)
                if value is None or str(value).strip() == '':
                    continue
                try:
                    value = normalizer(value)
                except (ValueError, TypeError):
                    issues.append(name.upper() + '_INVALID')
                    continue
                if value not in values:
                    values.append(value)
        if len(values) > 1:
            issues.append(name.upper() + '_CONFLICT')
        return values[0] if len(values) == 1 else None

    def variant(value):
        match = re.fullmatch(r'(R[012])(?:\s+[—–-].*)?', str(value).strip(), re.I)
        if not match:
            raise ValueError('Not an explicit ROI variant')
        return match[1].upper()

    def positive_integer(value):
        if isinstance(value, bool) or not str(value).isdigit() or int(value) <= 0:
            raise ValueError('Invalid repeat')
        return int(value)

    # Unrelated experiments retain their existing F4/legacy routing.
    kinds = {str(s[k]).strip().lower() for s in sources for k in ('experiment_type', 'type')
             if s.get(k) not in (None, '')}
    series_values = {str(s['series_id']).strip() for s in sources if s.get('series_id')}
    if any(is_diagnostic_purpose(s.get('measurement_purpose')) for s in sources):
        def purpose_value(value):
            if not is_diagnostic_purpose(value):
                raise ValueError('Invalid diagnostic purpose')
            return value
        purpose = field('measurement_purpose', normalizer=purpose_value)
        series = field('series_id')
        scenario = field('scenario_id')
        selected = field('variant')
        repeat = field('replicate_index', normalizer=positive_integer)
        if not purpose or not series or not series.startswith(DIAGNOSTIC_PURPOSES.get(purpose, '\x00')):
            issues.append('DIAGNOSTIC_PURPOSE_SERIES_CONFLICT')
        if not bool(getattr(getattr(bundle, 'validation', None), 'ok', False)):
            issues.append('SOURCE_UNVERIFIED')
        status = 'CONFLICT' if issues else 'DIAGNOSTIC_ONLY'
        return dict(experiment=purpose if status == 'DIAGNOSTIC_ONLY' else 'UNPROVEN', family='DIAGNOSTIC',
                    measurement_purpose=purpose or '', experiment_type=(purpose or 'diagnostic_conflict').lower(),
                    series_id=series, scenario_id=scenario,
                    variant=selected, variant_display=purpose or 'DIAGNOSTIC_CONFLICT',
                    replicate_index=repeat,
                    roi_budget_policy=field('roi_budget_policy'), is_pilot=False,
                    status=status, issues=issues, execution=next((c for c in configs if c), {}),
                    provenance='ANDROID_RECORDED_METADATA')
    applicable = 'roi_budget' in kinds or any(s.upper().startswith('F5A-') for s in series_values)
    if not applicable:
        return dict(experiment='UNPROVEN', status='NOT_APPLICABLE', issues=[], is_pilot=False)

    kind = field('experiment_type', ('type',), lambda v: str(v).strip().lower())
    series = field('series_id', normalizer=lambda v: str(v).strip())
    scenario = field('scenario_id', normalizer=lambda v: str(v).strip())
    selected = field('variant', ('experiment_variant',), variant)
    policy = field('roi_budget_policy', ('effective_roi_budget_policy',), lambda v: str(v).strip().lower())
    repeat = field('replicate_index', normalizer=positive_integer)
    policy_variants = {'r0_full_frame': 'R0', 'full_frame': 'R0',
                       'r1_one_roi': 'R1', 'r2_two_roi': 'R2'}
    if policy and policy not in policy_variants:
        issues.append('ROI_POLICY_UNRECOGNIZED')
    elif policy and selected and policy_variants[policy] != selected:
        issues.append('VARIANT_POLICY_CONFLICT')
    if kind != 'roi_budget':
        issues.append('EXPERIMENT_TYPE_UNPROVEN')
    if not series or not series.upper().startswith('F5A-'):
        issues.append('F5A_SERIES_UNPROVEN')
    for key, value in (('SCENARIO', scenario), ('VARIANT', selected), ('ROI_POLICY', policy)):
        if not value:
            issues.append(key + '_UNPROVEN')
    verified = bool(getattr(getattr(bundle, 'validation', None), 'ok', False))
    if not verified:
        issues.append('SOURCE_UNVERIFIED')
    issues = list(dict.fromkeys(issues))
    pilot = bool(series and re.search(r'(?:^|[-_])PILOT(?:$|[-_])', series, re.I))
    status = 'CONFLICT' if any('CONFLICT' in v or 'INVALID' in v for v in issues) else 'UNCONFIRMED' if issues else 'VERIFIED'
    execution = next((c for c in configs if c), {})
    return dict(experiment='F5A' if status == 'VERIFIED' else 'UNPROVEN',
                family='F5A' if any(s.upper().startswith('F5A-') for s in series_values) else 'ROI',
                experiment_type=kind, variant=selected, series_id=series,
                scenario_id=scenario, replicate_index=repeat, roi_budget_policy=policy,
                is_pilot=pilot, status=status, issues=issues, execution=execution,
                variant_display=('PILOT · ' if pilot else '') + (selected or 'NIEPOTWIERDZONE'),
                provenance='ANDROID_RECORDED_METADATA' if status == 'VERIFIED' else None)


def is_f5a(bundle):
    return from_bundle(bundle).get('experiment') == 'F5A'


def from_review_provenance(provenance, *, verified):
    """Resolve the same recorded fields after an existing review export/reload."""
    provenance = _dict(provenance)
    return from_bundle(SimpleNamespace(
        report_payload={'research_execution_config': provenance.get('research_execution_config')},
        experiment_session=provenance.get('experiment_session'),
        collection_session=provenance.get('collection_session'),
        validation=SimpleNamespace(ok=verified)))
