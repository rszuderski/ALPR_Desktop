# -*- coding: utf-8 -*-
"""Re-render per-session Android traces for paper-ready PNG/SVG figures.

The existing Z4 Mobile report browser draws its latency plot on a Tk Canvas;
exporting a vector figure must redraw from original trace rows, not screenshot
or save postscaled UI pixels. This module accepts bundle.trace_rows directly.
"""
from __future__ import annotations
from collections.abc import Iterable, Mapping
import math
from pathlib import Path

SERIES = {
    "plate_inference_ms": ("MT — czas inferencji łącznie w śladzie", "Łączny czas inferencji MT na ślad [ms]"),
    "character_inference_ms": ("MZ — czas inferencji łącznie w śladzie", "Łączny czas inferencji MZ na ślad [ms]"),
    "total_ms": ("ALPR — zarejestrowany czas przebiegu", "Czas przebiegu [ms]"),
}


def _valid_number(x):
    if x is None or x == "" or isinstance(x, bool): return None
    try:v=float(x)
    except (ValueError,TypeError,OverflowError):return None
    return v if math.isfinite(v) and v >= 0 else None


def trace_points(trace_rows: Iterable[Mapping], stage: str):
    if stage not in SERIES:raise ValueError(f"Invalid stage {stage}")
    pts=[]
    for row in trace_rows:
        if not isinstance(row,Mapping):continue
        status=str(row.get('status') or '').lower()
        if status in ('scene_superseded','runtime_error','pipeline_error','cancelled','not_run'):
            continue
        time=_valid_number(row.get('elapsed_ms'))
        value=_valid_number(row.get(stage))
        if time is None or value is None:continue
        # Literal zero is not evidence of an inference run. Real measured
        # TFLite calls are timed positively; skip absent/unexecuted stages.
        if stage in ("plate_inference_ms", "character_inference_ms") and value <= 0:
            continue
        if stage == "character_inference_ms":
            try:
                mz_runs = int(row.get("mz_runs") or 0)
            except (ValueError, TypeError, OverflowError):
                mz_runs = 0
            if row.get("mz_runs") not in (None, "") and mz_runs <= 0:
                continue
        # For pipeline, require the stage actually ran, otherwise a cancelled
        # frame with a nonzero overhead could be counted as a completed run.
        if stage=='total_ms' and _valid_number(row.get('plate_inference_ms')) is None:
            continue
        pts.append((time/1000,value))
    pts.sort(key=lambda item:item[0]);return pts


def trace_figures(trace_rows, *, session_label, sample_note=""):
    import matplotlib.pyplot as plt
    rows=list(trace_rows)
    for key,(title,ylabel) in SERIES.items():
        values=trace_points(rows,key)
        fig,ax=plt.subplots(figsize=(11,5.4))
        fig.subplots_adjust(left=.11,right=.97,top=.72,bottom=.24)
        if values:
            xs=[v[0] for v in values];ys=[v[1] for v in values]
            ax.plot(xs,ys,marker='o',linewidth=1.3,markersize=3.8)
            ax.set_ylim(bottom=0)
        else:
            ax.text(.5,.5,'Brak wykonanych pomiarów dla tej metryki',
                    ha='center',va='center',transform=ax.transAxes)
        ax.set_title(f"{title}\n{session_label}", fontsize=10, pad=12)
        ax.set_xlabel('Czas od początku sesji [s]')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        note=(f"N={len(values)} rzeczywiście zarejestrowanych wartości. Nie podstawiono zera za brak próby."
             +(" Czas MT jest sumą w obrębie śladu przy wielu ROI/fallbacku." if key=='plate_inference_ms' else " MZ jest sumą w obrębie śladu przy wielu detekcjach." if key=='character_inference_ms' else "")
             +(f" {sample_note}" if sample_note else ""))
        import textwrap
        fig.text(.055,.055,'\n'.join(textwrap.wrap(note,110)),fontsize=8,va='bottom')
        yield key,fig
