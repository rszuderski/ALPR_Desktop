# -*- coding: utf-8 -*-
"""AKT85: read-only chart comparison from reports already imported into ALPR Desktop.

Source of runtime plots: actual validated Android traces; source of quality plots:
validated, explicitly assisted Desktop Canvas reviews only.  Never infer a
missing measurement or implicitly merge DAY with NIGHT.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import textwrap

from .mobile_comparison_core import (
    InvalidExport, Latency, Session, figures, load_session_export,
    validate_selection,
)
from .mobile_import_trace_charts import trace_points, trace_figures
from .mobile_compare_review_gate import paired_geometry_text
from .mobile_model_identity import identity_from_bundle

SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,160}$")
_SHA_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_STAGES = {"mt": "plate_inference_ms", "mz": "character_inference_ms", "pipeline": "total_ms"}


def _get(value: Any, key: str, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _nested(obj: Any, *keys):
    for key in keys:
        if isinstance(obj, dict):
            obj=obj.get(key)
        else:
            return None
        if obj is None:
            return None
    return obj


def _first(*values):
    for item in values:
        if item not in (None, "", {}, []):
            return item
    return None


def _sha_file(path: Path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(512*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(rows: Iterable[float], q: float) -> float | None:
    numbers=sorted(x for x in rows if x is not None and math.isfinite(x) and x >= 0)
    if not numbers:
        return None
    pos=(len(numbers)-1)*q
    lo=int(math.floor(pos))
    hi=int(math.ceil(pos))
    return numbers[lo] + (numbers[hi]-numbers[lo])*(pos-lo)


def _extract_model_ref(refs, role):
    if not isinstance(refs, dict):
        return {}
    aliases={"mt": ("mt", "plate", "MT", "Plate"), "mz": ("mz", "character", "MZ", "Character")}[role]
    for key in aliases:
        value=refs.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _model_metadata(report, bundle, raw, coll, role):
    stage=("plate" if role=="mt" else "character")
    choices=[
        _nested(coll, "execution", "stages", role),
        _nested(coll, "execution", "stages", stage),
        _nested(coll, "execution", stage),
        _nested(raw, "execution", stage),
        _nested(raw, "research_execution_config", "stages", role),
        _extract_model_ref(_get(bundle,"model_refs",{}),role),
        _extract_model_ref(_get(bundle,"pipeline_manifests",{}),role),
    ]
    merged={}
    for item in choices:
        if isinstance(item,dict):
            for key,value in item.items():
                if key not in merged and value not in (None, "", {}, []):
                    merged[key]=value
    return merged


def _read_all_traces(bundle):
    """Use complete trace stream. No misleading metrics from a preview limit."""
    preview=list(_get(bundle,"trace_rows",()) or ())
    total=int(_get(bundle,"trace_total",0) or len(preview))
    if total<=len(preview):
        return preview, "complete"
    try:
        # The official Android source reader streams the complete rows from
        # the archive. It does NOT start another measurement or write to disk.
        from ..ranking.mobile_package_experiments import iter_full_trace_rows
        full=list(iter_full_trace_rows(bundle))
        if len(full)<total:
            return [], f"INCOMPLETE_TRACE_SOURCE: {len(full)}/{total}"
        return full,"complete"
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, TypeError) as exc:
        return [], f"INCOMPLETE_TRACE_SOURCE: {type(exc).__name__} ({len(preview)}/{total})"


def _discover_review(root: Path | None, session_id: str, source_sha: str, review_root=None):
    from ..config import CONFIG
    from .mobile_report_canvas_view import selected_canvas_export
    review_root = Path(review_root or getattr(CONFIG, "DIR_7_RANKINGS_MOBILE_PACKAGES", CONFIG.DIR_7_RANKINGS/"mobile_packages"))
    gate, review = selected_canvas_export(session_id, source_sha, root, review_root)
    return review, [] if gate.ready or gate.state == "NONE" else [gate.reason]


def _identity(report, bundle):
    raw=_get(bundle,"report_payload",{}) or _get(report,"raw",{}) or {}
    coll=_get(bundle,"collection_session",{}) or {}
    experiment=_get(bundle,"experiment_session",{}) or _get(report,"experiment_index",{}) or {}
    rex=_get(raw,"experiment_session",{}) or _get(raw,"experiment",{}) or {}
    sid=str(_first(_get(coll,"session_id"),_get(experiment,"session_id"),
                   _get(experiment,"experiment_session_id"),_get(rex,"session_id"),
                   _get(report,"report_id")) or "")
    if not SESSION_ID_PATTERN.fullmatch(sid):
        raise ValueError("Nieprawidłowy identyfikator session_id w raporcie.")
    scenario=str(_first(_get(coll,"scenario_id"),_get(experiment,"scenario_id"),
                        _get(rex,"scenario_id")) or "NIEOKREŚLONY")
    replicate=_first(_get(coll,"replicate_index"),_get(experiment,"replicate_index"),
                     _get(rex,"replicate_index"))
    try:
        recorded_repeat=f"r{int(replicate):02d}" if int(replicate)>0 else "r?"
    except (ValueError,TypeError,OverflowError):
        recorded_repeat="r?"
    # F4 source metadata can contain repeat 4--6 in the NIGHT FP32 files,
    # while the verified external F4 folder is r01--r03. Use the actual
    # 02_sesje/.../rNN locator when available, never silently rewrite source.
    actual_path=Path(str(_get(bundle,"path","") or ""))
    repeat=recorded_repeat
    parent_parts=[p.name.lower() for p in actual_path.parents]
    if "02_sesje" in parent_parts:
        for part in parent_parts[:parent_parts.index("02_sesje")]:
            if re.fullmatch(r"r[0-9]{2}",part):
                repeat=part
                break
    device=_first(_get(raw,"device"),_get(report,"device"),{}) or {}
    device_name=str(_first(_get(device,"model"),_get(device,"name"),_get(device,"device_name")) or "unknown")
    android=str(_first(_get(device,"android_version"),_get(device,"android")) or "unknown")
    build=_first(_get(coll,"app_build"),_get(raw,"app_build"),{}) or {}
    apk=str(_first(_nested(build,"apk_identity","base_apk_sha256"),_get(build,"apk_sha256")) or "").lower()
    if apk and not _SHA_PATTERN.fullmatch(apk):
        apk=""  # missing attestation is marked, never fabricated
    mt=_model_metadata(report,bundle,raw,coll,"mt")
    mz=_model_metadata(report,bundle,raw,coll,"mz")
    pmt=str(_get(mt,"precision") or "").lower()
    pmz=str(_get(mz,"precision") or "").lower()
    precision=pmt if pmt==pmz and pmt in ("fp32","int8") else "unknown"
    def model_id(m):
        return str(_first(_get(m,"checkpoint_sha256"),_nested(m,"checkpoint","sha256")) or "")
    from ..ranking.mobile_experiment_identity import from_bundle
    experiment_identity = from_bundle(bundle)
    extra = {}
    from ..ranking.mobile_tv_bbox_evidence import input_context as tv_input_context
    tv_context = tv_input_context(getattr(bundle, 'collection_session', {}) or {}, getattr(bundle, 'tv_bbox_evidence', {}) or {})
    if tv_context:
        extra.update(input_mode=tv_context['input_mode'], input_context=tv_context)
    if experiment_identity.get('family') in {'F5A', 'DIAGNOSTIC'}:
        extra.update(experiment_type=experiment_identity.get('experiment_type') or '',
                     series_id=experiment_identity.get('series_id') or '',
                     roi_variant=experiment_identity.get('variant') or '',
                     is_pilot=experiment_identity['is_pilot'], identity_status=experiment_identity['status'],
                     measurement_purpose=experiment_identity.get('measurement_purpose', ''))
    return dict(session_id=sid,scenario=scenario,repeat_label=repeat,device=device_name,
                android=android,apk_sha256=apk,precision=precision,
                mt_checkpoint=model_id(mt),mz_checkpoint=model_id(mz),
                mt_model_sha=str(_get(mt,"model_sha256") or ""),
                mz_model_sha=str(_get(mz,"model_sha256") or ""),
                recorded_repeat=recorded_repeat, **extra)


def record_from_import(report, bundle, results_root=None, *, review_root=None, include_review=True):
    """Return (verified or incomplete Session, trace rows, explicit warnings)."""
    if bundle is None:
        raise ValueError("Raport nie ma podłączonego oryginalnego archiwum.")
    valid=_get(bundle,"validation",None)
    if not bool(_get(valid,"ok",False)):
        raise ValueError("Archiwum nie przeszło walidacji źródła.")
    sha=str(_get(bundle,"source_archive_sha256","") or "").lower()
    expected=str(_get(report,"source_archive_sha256","") or "").lower()
    if not _SHA_PATTERN.fullmatch(sha) or expected != sha:
        raise ValueError("Niezgodność SHA-256 pomiędzy wybranym raportem i archiwum.")
    if str(_get(_get(bundle,"report"),"report_id","")) != str(_get(report,"report_id","")):
        raise ValueError("Archiwum ma inny report_id niż importowana sesja.")
    ident=_identity(report,bundle)
    model_identity=identity_from_bundle(bundle)
    recorded_repeat=ident.pop("recorded_repeat",ident["repeat_label"])
    experiment_fields = {k: ident[k] for k in ('experiment_type', 'series_id', 'roi_variant', 'is_pilot', 'identity_status', 'measurement_purpose', 'input_mode', 'input_context') if k in ident}
    rows,source_state=_read_all_traces(bundle)
    lat={}
    for stage,col in _STAGES.items():
        values=[ms for _,ms in trace_points(rows,col)]
        lat[stage]=Latency(len(values),_percentile(values,.5),_percentile(values,.95))
    warnings=[]
    if ident["repeat_label"] != recorded_repeat:
        warnings.append(f"Telefon zapisuje powtórzenie {recorded_repeat}, folder źródła wskazuje {ident['repeat_label']} — zachowano oba odniesienia.")
    if source_state!="complete":
        warnings.append(source_state + ": pominięto niepełne statystyki czasów.")
    review, more=(_discover_review(results_root,ident["session_id"],sha,review_root) if include_review else (None,[]))
    warnings.extend(more)
    if review is not None:
        if ident["apk_sha256"] and review.apk_sha256 and review.apk_sha256 != ident["apk_sha256"]:
            warnings.append("Eksport Canvas ma inny SHA APK — nie dołączono oceny GT.")
            review=None
    if review is not None:
        # Metrics always come from the exact imported archive, never from a
        # summary from another run. GT retains its ASSISTED designation.
        return replace(review,latency=lat,scenario=ident["scenario"],precision=ident["precision"],
                       repeat_label=ident["repeat_label"],device=ident["device"],android=ident["android"],
                       apk_sha256=ident["apk_sha256"],mt_checkpoint=ident["mt_checkpoint"] or review.mt_checkpoint,
                       mz_checkpoint=ident["mz_checkpoint"] or review.mz_checkpoint,
                       model_identity=model_identity,
                        **experiment_fields,
                       source_evidence_status="verified Android archive + reviewed export"),rows,warnings
    result=Session(directory=Path(results_root or Path.cwd())/ident["session_id"],
        session_id=ident["session_id"],review_revision=0,review_status="UNREVIEWED",
        review_mode="NOT_EVALUABLE",source_sha256=sha,
        manifest_sha256="",latency=lat,canvas=None,
        scientific_quality_status="NOT_EVALUABLE",source_evidence_status=source_state,
        model_identity=model_identity,
        **{k:v for k,v in ident.items() if k!="session_id"})
    return result,rows,warnings


def infer_results_root(entries):
    """Only infer the established 03_wyniki sibling of an existing 02_sesje."""
    candidates=set()
    for _,bundle in entries:
        path=Path(str(_get(bundle,"path","") or ""))
        if not path.is_file():
            continue
        for parent in path.parents:
            if parent.name=="02_sesje":
                results=parent.parent/"03_wyniki"
                if results.is_dir():candidates.add(results.resolve())
                break
    return next(iter(candidates)) if len(candidates)==1 else None


def _safe_output_name(name):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,120}",name):
        raise ValueError("Niebezpieczna nazwa wykresu.")
    return name


def export_imported_charts(records, row_sets, results_root, *, comparison_scope=None):
    """Versioned export, staging + atomic rename, no source/session mutations."""
    import matplotlib.pyplot as plt
    from .mobile_comparison_scope import validate_comparison, comparison_warnings
    records=list(records); rows=[list(x) for x in row_sets]
    scope=validate_comparison(records,rows,comparison_scope=comparison_scope,require_sources=comparison_scope is not None)
    warnings=comparison_warnings(records,scope)
    root=Path(results_root)
    if not root.is_dir() or root.name.lower()!="03_wyniki":
        raise ValueError("Wskaż istniejący katalog 03_wyniki.")
    single=len(records)==1
    if single:
        parent=root/records[0].session_id
        if not parent.exists():parent.mkdir(mode=0o755)
        if not parent.is_dir():raise ValueError("Katalog wynikowy sesji nie jest katalogiem.")
        stem=f"wykresy_r{records[0].review_revision:06d}"
    else:
        parent=root/"porownanie"
        if not parent.is_dir():raise ValueError("Nie znaleziono istniejącego folderu porownanie.")
        stem="porownanie_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage=Path(tempfile.mkdtemp(prefix=".alpr_chart_stage_",dir=parent))
    dest=None
    try:
        outputs=[]
        for title,fig in figures(records,comparison_scope=scope):
            try:
                safe=_safe_output_name(title)
                for ext in ("png","svg"):
                    path=stage/(safe+"."+ext)
                    fig.savefig(path,dpi=220 if ext=="png" else None,facecolor="white")
                    outputs.append(path)
            finally:
                plt.close(fig)
        if single:
            for title,fig in trace_figures(rows[0],session_label=records[0].label):
                try:
                    safe=_safe_output_name("czas_seria_"+title)
                    for ext in ("png","svg"):
                        path=stage/(safe+"."+ext)
                        fig.savefig(path,dpi=220 if ext=="png" else None,facecolor="white")
                        outputs.append(path)
                finally:
                    plt.close(fig)
        data=stage/"dane_porownania.csv"
        with data.open("w",encoding="utf-8-sig",newline="") as handle:
            writer=csv.writer(handle,delimiter=';')
            writer.writerow(["session_id","scenario","repeat","precision","apk_sha256","source_sha256","review_revision",
                             "stage","n","p50_ms","p95_ms","canvas_assisted","exact_pct","micro_cer_pct","geometry_n","polygon_iou_pct",
                             "comparison_scope","quality_availability","report_id"])
            for s in records:
                for stage_name,met in s.latency.items():
                    c=s.canvas or {}
                    quality=scope!='performance' and s.canvas_ready
                    writer.writerow([s.session_id,s.scenario,s.repeat_label,s.precision,s.apk_sha256,s.source_sha256,
                                     s.review_revision,stage_name,met.n,met.p50_ms,met.p95_ms,quality if scope!='performance' else '',
                                     s.canvas_exact_pct if quality else '',s.canvas_cer_pct if quality else '',c.get('geometry_n') if quality else '',
                                     c.get('polygon_iou_mean_pct') if quality else '',scope,
                                     'ASSISTED' if quality else 'N/A — poza zakresem lub brak pełnej oceny',
                                     (s.comparison_source or {}).get('report',{}).get('report_id','')])
        outputs.append(data)
        # This is the essential scientific link: original MT geometry and the
        # MZ error concern exactly the same annotated plate. No synthetic
        # re-inference on the operator-corrected ROI is implied.
        if scope!='performance':
            paired = stage / "geometria_MT_a_odczyt_MZ.csv"
            with paired.open("w",encoding="utf-8-sig",newline="") as handle:
                writer=csv.writer(handle,delimiter=';')
                writer.writerow(["session_id","source_archive_sha256","review_revision","scenario",
                    "precision","repeat","image_entry","object_id","polygon_iou_pct",
                    "nmce_pct","cer_pct","exact_match","review_origin"])
                for sess in records:
                    if not sess.canvas_ready:
                        continue
                    for row in paired_geometry_text(sess.canvas):
                        writer.writerow([sess.session_id,sess.source_sha256,sess.review_revision,
                            sess.scenario,sess.precision,sess.repeat_label,row["image_entry"],
                            row["object_id"],row["iou_pct"],row["nmce_pct"],row["cer_pct"],
                            int(row["exact"]),"ASSISTED_CANVAS_SOURCE_ROI"])
            outputs.append(paired)
        manifest={
            "schema":"alpr.mobile.import_charts.v1",
            "generated_utc":datetime.now(timezone.utc).isoformat(),
            "scope":"single_session" if single else "comparison",
            "comparison_scope":scope,
            "quality_availability":"N/A — poza zakresem" if scope=='performance' else 'Wyłącznie kompletne oceny ASSISTED',
            "comparison_kind":"descriptive",
            "warnings":warnings,
            "sources":[{"session_id":s.session_id,"source_archive_sha256":s.source_sha256,
                        "apk_sha256":s.apk_sha256,"scenario":s.scenario,"variant":s.precision,
                        "review_revision":s.review_revision,"review_export_manifest_sha256":s.manifest_sha256,
                        "canvas_assisted_ready":s.canvas_ready,
                        "report_id":(s.comparison_source or {}).get('report',{}).get('report_id'),
                        "source_path":(s.comparison_source or {}).get('path'),
                        "trace_sha256":(s.comparison_source or {}).get('trace_sha256'),
                        "latency":{k:dict(n=v.n,p50_ms=v.p50_ms,p95_ms=v.p95_ms,
                            availability='MEASURED' if v.n else 'N/A') for k,v in s.latency.items()},
                        "package_and_models":s.model_identity} for s in records],
            "files":[{"name":p.name,"bytes":p.stat().st_size,"sha256":_sha_file(p)} for p in outputs],
            "limitations":["Pełna jakość MT/MZ poza zakresem; pola jakości pozostają puste." if scope=='performance' else "Canvas ASSISTED nie jest niezależnym GT.",
                           "DAY/NIGHT rozdzielono. Brak wartości nie jest zerem.",
                           "Porównania p95 między sesjami opisowe; nie są sparowanym testem klatek.",
                           "MZ dotyczy sumy czasu w śladzie, nie pojedynczego wywołania MZ.",
                           "Zgodność scenariusza i różne SHA nie dowodzą identyczności zdjęć ani kontrolowanych warunków."],
        }
        (stage/"wykresy_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        for number in range(1,100000):
            candidate=parent/f"{stem}_{number:03d}"
            if not candidate.exists():
                dest=candidate
                break
        if dest is None:
            raise RuntimeError("Brak wolnego numeru eksportu.")
        # Rendering can take time. Do not publish an export invalidated meanwhile.
        validate_comparison(records,rows,comparison_scope=scope,require_sources=comparison_scope is not None)
        stage.rename(dest)
        return dest
    finally:
        if stage.exists():shutil.rmtree(stage,ignore_errors=True)
