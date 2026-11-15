"""UI labels/order only. Renderer keys, figures and export names stay intact."""
from collections import Counter
from dataclasses import dataclass
from typing import Any


# Explicitly ordered by chart family. IoU is polygon_iou_mean_pct;
# micro-CER is sum(text_errors)/sum(gt_characters), not a mean of plate CERs.
SUMMARY_LABELS = {
    'mt_p50_ms': 'Czasy · MT — mediana (p50) [ms]',
    'mt_p95_ms': 'Czasy · MT — p95 [ms]',
    'mz_p50_ms': 'Czasy · MZ — mediana (p50) [ms]',
    'mz_p95_ms': 'Czasy · MZ — p95 [ms]',
    'pipeline_p50_ms': 'Czasy · ALPR — mediana (p50) [ms]',
    'pipeline_p95_ms': 'Czasy · ALPR — p95 [ms]',
    'assisted_exact': 'Jakość MZ · pełne zgodne numery [%]',
    'assisted_cer': 'Jakość MZ · micro-CER [%]',
    'assisted_iou': 'Geometria MT · średnie IoU [%]',
    'assisted_iou_distribution': 'Geometria MT · rozkład IoU [%]',
    'assisted_nmce_distribution': 'Geometria MT · rozkład NMCE [%]',
    'assisted_iou_vs_cer_pairs': 'Zależność · IoU a CER (tablice)',
    'assisted_nmce_vs_cer_pairs': 'Zależność · NMCE a CER (tablice)',
}
TRACE_LABELS = {
    'oś czasu · plate_inference_ms': 'Czasy · MT — przebieg sesji [ms]',
    'oś czasu · character_inference_ms': 'Czasy · MZ — suma w śladzie [ms]',
    'oś czasu · total_ms': 'Czasy · ALPR — przebieg sesji [ms]',
}


@dataclass(frozen=True)
class PlotChoice:
    key: str
    label: str
    figure: Any


def _description(key):
    if key in TRACE_LABELS:
        return TRACE_LABELS[key], 6 + list(TRACE_LABELS).index(key), ''
    for index, (suffix, label) in enumerate(SUMMARY_LABELS.items()):
        if key.endswith('_'+suffix):
            scenario = key[:-len(suffix)-1]
            return label, index if index < 6 else index+3, scenario
    # A future renderer stays selectable and retains its identity.
    return 'Inne · '+key.replace('_',' '), 100, ''


def plot_choices(items):
    """One option per unchanged key; first occurrence owns that renderer.

    Different scenario prefixes remain explicit when shortening would give
    identical labels. Display text is never used to identify the active plot.
    """
    unique = {}
    for key, figure in items:
        unique.setdefault(key, figure)
    descriptions = {key: _description(key) for key in unique}
    counts = Counter(label for label, _, _ in descriptions.values())
    result = []
    for key in sorted(unique, key=lambda k: (descriptions[k][1], descriptions[k][2], k)):
        label, _, scenario = descriptions[key]
        if counts[label] > 1:
            label += ' · '+(scenario or key)
        result.append(PlotChoice(key, label, unique[key]))
    # Also handle a future key whose literal text collides with a qualified label.
    collisions = Counter(option.label for option in result)
    return [PlotChoice(o.key, o.label+f' [{o.key}]' if collisions[o.label]>1 else o.label, o.figure)
            for o in result]
