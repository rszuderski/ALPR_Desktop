"""Z4LIB029: human-readable summary of MT/MZ export sources, no I/O.

UI information only. The preflight/exporters are authoritative; this module
never approves a model, precision comparison, calibration, Android, or F4.
"""
from __future__ import annotations
from pathlib import Path


def summarize_export_plan(rows: list[dict]) -> tuple[str, bool]:
    """Return (status text, reuse-only) for explicitly selected MT and MZ.

    Each row: role ('MT'/'MZ'), reuse(bool), path(str), precision(str),
    formats(dict of booleans), image_size(int). No filesystem reads.
    """
    if len(rows) != 2 or {row.get('role') for row in rows} != {'MT', 'MZ'}:
        raise ValueError('Zestaw wymaga dokładnie jednego MT i jednego MZ.')
    by_role = {row['role']: row for row in rows}
    lines = []
    for role in ('MT', 'MZ'):
        row = by_role[role]
        if row.get('reuse'):
            precision = str(row.get('precision') or '?').upper()
            filename = Path(str(row.get('path') or '')).name
            if filename:
                lines.append(f'{role}: użyję paczki z biblioteki ({precision}) — {filename}')
            else:
                lines.append(f'{role}: paczka z biblioteki ({precision}) — WYBIERZ PLIK')
        else:
            formats = row.get('formats') or {}
            choices = []
            for key, label in (
                ('litert', 'TFLite FP32'), ('int8', 'TFLite INT8'),
                ('onnx', 'ONNX FP32'), ('onnx_int8', 'ONNX INT8'),
                ('ncnn', 'NCNN FP32'),
            ):
                if formats.get(key):
                    choices.append(label)
            readable = ', '.join(choices) if choices else 'NIE WYBRANO FORMATU'
            size = row.get('image_size') or '?'
            lines.append(f'{role}: przygotuję nowy model z best.pt — {readable}, imgsz={size}')
    return '\n'.join(lines), all(bool(by_role[role].get('reuse')) for role in ('MT', 'MZ'))
