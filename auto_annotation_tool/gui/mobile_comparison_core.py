# -*- coding: utf-8 -*-
"""F4: read-only session review comparison and publication-oriented figure exports.

Read-only comparison backend integrated with the ALPR Desktop import browser.
Reads *derived, already exported* review folders (AKT83), not original .alprsession.
Never edits sources. Critical: Canvas assisted results are not independent GT.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .mobile_compare_review_gate import inspect_canvas, paired_geometry_text
from .mobile_model_identity import recorded_model_identity, chart_session_label, chart_identity_caption

CHART_SCHEMA = "alpr.f4.chart_comparison.v1"
ALLOWED_PRECISION = {"fp32", "int8"}
_LATENCY_KEYS = {"mt": "MT — lokalizacja tablicy", "mz": "MZ — rozpoznawanie znaków",
                 "pipeline": "Potok ALPR (według telemetrii telefonu)"}
_SHA = re.compile(r"^[a-f0-9]{64}$")
_SAFE = re.compile(r"^[a-zA-Z0-9_.-]+$")


class InvalidExport(ValueError):
    pass


@dataclass(frozen=True)
class Latency:
    n: int
    p50_ms: float | None
    p95_ms: float | None


@dataclass(frozen=True)
class Session:
    directory: Path
    session_id: str
    review_revision: int
    review_status: str
    review_mode: str
    source_sha256: str
    manifest_sha256: str
    scenario: str
    repeat_label: str
    device: str
    android: str
    apk_sha256: str
    precision: str
    mt_checkpoint: str
    mz_checkpoint: str
    mt_model_sha: str
    mz_model_sha: str
    latency: dict[str, Latency]
    canvas: dict[str, Any] | None
    scientific_quality_status: str
    source_evidence_status: str
    model_identity: dict[str, Any] = field(default_factory=dict)
    operator_approved: bool = False
    mt_evaluation: dict[str, Any] | None = None
    mt_object_evaluation: dict[str, Any] | None = None
    comparison_source: dict[str, Any] | None = None
    experiment_type: str = ''
    series_id: str = ''
    roi_variant: str = ''
    is_pilot: bool = False
    identity_status: str = 'UNCONFIRMED'
    scene_assessment: dict[str, Any] | None = None
    mz_input_evidence: dict[str, Any] | None = None
    measurement_purpose: str = ''
    input_mode: str = 'CAMERA_FULL_FRAME'
    input_context: dict[str, Any] | None = None
    tv_bbox_evidence: dict[str, Any] | None = None

    @property
    def label(self) -> str:
        label = chart_session_label(self.model_identity, scenario=self.scenario,
            repeat=self.repeat_label, session_id=self.session_id, precision=self.precision)
        from ..ranking.mobile_mz_input_evidence import is_diagnostic_purpose
        if self.input_mode == 'TV_BBOX_CROP':
            label = 'TV_BBOX_CROP · pełna kamera dla GT\n' + label
        if is_diagnostic_purpose(self.measurement_purpose):
            return self.measurement_purpose + ' · ' + self.series_id + '\n' + label
        if self.experiment_type == 'roi_budget' and self.series_id.upper().startswith('F5A-'):
            return f"F5A · {'PILOT · ' if self.is_pilot else ''}{self.roi_variant} · {self.series_id}\n" + label
        return label

    @property
    def canvas_ready(self) -> bool:
        # Unified operator gate: signs, source MT corners, corrected four
        # corners, FN accounting and annotated MZ must all be complete.
        return inspect_canvas(self.canvas, self.review_status)[0]

    @property
    def canvas_exact_pct(self) -> float | None:
        if not self.canvas_ready:
            return None
        n = self.canvas["evaluated_reads"]
        return (100 * self.canvas["exact_reads"] / n) if n else None

    @property
    def canvas_cer_pct(self) -> float | None:
        if not self.canvas_ready:
            return None
        n = self.canvas.get("gt_characters", 0)
        return (100 * self.canvas["text_errors"] / n) if n else None


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(512 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise InvalidExport(f"Expected JSON object: {path}")
    return data


def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _int(value, default=0) -> int:
    try:
        return max(0, int(value))
    except (ValueError, TypeError, OverflowError):
        return default


def _trace_step(diagnostics, operation, path):
    if diagnostics is not None:
        diagnostics.update(operation=operation, file=str(path))


def _verify_manifest(export_dir: Path, manifest: dict, *, diagnostics=None) -> None:
    _trace_step(diagnostics, "validate_manifest_schema", export_dir / "export_manifest.json")
    entries = manifest.get("files")
    if manifest.get("schema") != "alpr_mobile_review_export_manifest_v1" or not isinstance(entries, list):
        raise InvalidExport("Unknown or incomplete export manifest")
    seen = set()
    _trace_step(diagnostics, "resolve_export_directory", export_dir)
    root = export_dir.resolve()
    for row in entries:
        if not isinstance(row, dict):
            raise InvalidExport("Malformed manifest entry")
        name = row.get("path")
        if not isinstance(name, str) or not name or "\\" in name:
            raise InvalidExport("Unsafe or missing relative path")
        _trace_step(diagnostics, "resolve_manifest_file", root / name)
        target = (root / name).resolve()
        _trace_step(diagnostics, "validate_file_path", target)
        if target == root or root not in target.parents or target in seen or not target.is_file():
            raise InvalidExport(f"Missing, repeated or unsafe file: {name}")
        seen.add(target)
        expected_size = row.get("bytes")
        expected_hash = row.get("sha256", "")
        _trace_step(diagnostics, "stat_file_size", target)
        if not isinstance(expected_size, int) or expected_size != target.stat().st_size:
            raise InvalidExport(f"Wrong size: {name}")
        if not isinstance(expected_hash, str) or not _SHA.fullmatch(expected_hash):
            raise InvalidExport(f"Missing SHA-256: {name}")
        _trace_step(diagnostics, "read_and_verify_sha256", target)
        if _file_hash(target) != expected_hash:
            raise InvalidExport(f"SHA-256 mismatch: {name}")
        if diagnostics is not None:
            diagnostics["files_verified"] = diagnostics.get("files_verified", 0) + 1


def load_session_export(directory: Path, *, verify=True, diagnostics=None) -> Session:
    directory = Path(directory)
    if diagnostics is not None:
        diagnostics.update(directory=str(directory.absolute()), loader_module=__name__,
                           loader_module_path=__file__, verify=verify, files_verified=0)
    _trace_step(diagnostics, "is_export_directory", directory)
    if not directory.is_dir():
        raise InvalidExport(f"Missing export directory: {directory}")
    mf_path = directory / "export_manifest.json"
    _trace_step(diagnostics, "read_manifest_json", mf_path)
    mf = _json(mf_path)
    if diagnostics is not None:
        diagnostics.update(manifest_session_id=mf.get("session_id"),
                           manifest_source_sha256=mf.get("source_archive_sha256"),
                           manifest_revision=mf.get("review_revision"),
                           files_expected=len(mf["files"]) if isinstance(mf.get("files"), list) else None)
    if verify:
        _verify_manifest(directory, mf, diagnostics=diagnostics)
    _trace_step(diagnostics, "validate_manifest_identity", mf_path)
    id_ = str(mf.get("session_id") or "")
    if not _SAFE.fullmatch(id_):
        raise InvalidExport("Invalid session_id in manifest")
    _trace_step(diagnostics, "find_scientific_summary", directory)
    sci_files = sorted(directory.glob("ocena_naukowa_*.json"))
    if len(sci_files) != 1:
        raise InvalidExport("Expected one scientific summary JSON")
    _trace_step(diagnostics, "read_scientific_summary_json", sci_files[0])
    sci = _json(sci_files[0])
    _trace_step(diagnostics, "read_review_json", directory / "review.json")
    review = _json(directory / "review.json")
    _trace_step(diagnostics, "validate_export_identity", directory)
    digest = str(mf.get("source_archive_sha256", ""))
    if not _SHA.fullmatch(digest):
        raise InvalidExport("Missing source archive SHA-256")
    for name, doc in [("review", review), ("science", sci)]:
        if doc.get("session_id") != id_ or doc.get("source_archive_sha256") != digest:
            raise InvalidExport(f"{name}: inconsistent source identity")
        if int(doc.get("review_revision", -1)) != int(mf.get("review_revision", -2)):
            raise InvalidExport(f"{name}: review revision mismatch")
    if sci.get("review_status") != mf.get("review_status"):
        raise InvalidExport("Review status mismatch")
    if 'review_status' in review and review['review_status']!=mf.get('review_status'):
        raise InvalidExport('Review document status mismatch')
    if review.get('review_id') and mf.get('review_id') and review['review_id']!=mf['review_id']:
        raise InvalidExport('Review document ID mismatch')
    operator_approved=(review.get('schema')=='alpr.mobile_human_review.v1'
        and review.get('review_status')==mf.get('review_status')=='COMPLETED'
        and bool(review.get('review_id')) and review['review_id']==mf.get('review_id'))
    mt_files=[r['path'] for r in mf.get('files',[]) if str(r.get('path','')).endswith('/mt_detection_evaluation.json')]
    mt_evaluation=_json(directory/mt_files[0]) if len(mt_files)==1 else None
    if mt_evaluation and any(mt_evaluation.get(k)!=mf.get(k) for k in ('session_id','source_archive_sha256','review_revision')):
        raise InvalidExport('MT evaluation identity mismatch')
    object_files=[r['path'] for r in mf.get('files',[]) if str(r.get('path','')).endswith('/mt_object_evaluation.json')]
    mt_objects=None
    if object_files:
        if len(object_files)!=1:raise InvalidExport('Ambiguous MT object evaluation namespace')
        from ..ranking.mobile_mt_object_export import replay
        mt_objects=replay((directory/object_files[0]).parent)
        if any(mt_objects.get(k)!=mf.get(k) for k in ('session_id','source_archive_sha256','review_revision')):
            raise InvalidExport('MT object evaluation identity mismatch')
    canvas_file = directory / "canvas_review_summary.json"
    _trace_step(diagnostics, "read_canvas_summary_json", canvas_file)
    canvas = _json(canvas_file) if canvas_file.exists() else None
    _trace_step(diagnostics, "validate_canvas_identity", canvas_file)
    if canvas and (canvas.get("source_archive_sha256") != digest or canvas.get("review_revision") != mf.get("review_revision")):
        raise InvalidExport("Canvas review identity mismatch")
    conditions = sci.get("conditions") or {}
    from ..ranking.mobile_mz_input_evidence import load_export as load_mz_export
    try:
        mz_input_evidence = load_mz_export(directory, mf, (review.get('provenance') or {}).get('collection_session'))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise InvalidExport('MZ input evidence: ' + str(exc)) from exc
    from ..ranking.mobile_tv_bbox_evidence import load_export as load_tv_export, input_context as tv_input_context
    try:
        source_session = (review.get('provenance') or {}).get('collection_session') or {}
        tv_bbox_evidence = load_tv_export(directory, mf, source_session)
        input_context = tv_input_context(source_session, tv_bbox_evidence)
    except (ValueError, KeyError, TypeError, AttributeError, IndexError, ArithmeticError, OSError) as exc:
        raise InvalidExport('TV BBOX evidence: ' + str(exc)) from exc
    scene_file = directory / 'scene_frame_assessment.json'
    scene_assessment = _json(scene_file) if scene_file.exists() else None
    if scene_assessment:
        if 'scene_frame_assessment.json' not in {r.get('path') for r in mf.get('files',[])}:
            raise InvalidExport('Scene-frame assessment is not covered by manifest')
        if (any(scene_assessment.get(k) != mf.get(k) for k in ('session_id','source_archive_sha256','review_revision'))
                or scene_assessment.get('scene_annotations') != review.get('scene_annotations', {})):
            raise InvalidExport('Scene-frame review identity mismatch')
    models = conditions.get("models") or {}
    mt, mz = models.get("MT") or {}, models.get("MZ") or {}
    mt_p = str(mt.get("precision") or "").lower()
    mz_p = str(mz.get("precision") or "").lower()
    prec = mt_p if mt_p == mz_p and mt_p in ALLOWED_PRECISION else "unknown"
    review_provenance = review.get("provenance") or {}
    provenance = review_provenance.get("model_provenance") or {}
    app_build = ((review_provenance.get("collection_session") or {}).get("app_build") or {})
    apk_sha = str((app_build.get("apk_identity") or {}).get("base_apk_sha256") or app_build.get("apk_sha256") or "")
    if apk_sha and not _SHA.fullmatch(apk_sha):
        raise InvalidExport("Bad APK SHA")
    chk_mt = str((provenance.get("plate") or {}).get("checkpoint_sha256") or "")
    chk_mz = str((provenance.get("character") or {}).get("checkpoint_sha256") or "")
    if any(s and not _SHA.fullmatch(s) for s in (chk_mt, chk_mz)):
        raise InvalidExport("Bad checkpoint SHA")
    perf = ((sci.get("performance") or {}).get("latency") or {})
    stages = {}
    for k in _LATENCY_KEYS:
        raw = perf.get(k) or {}
        stages[k] = Latency(_int(raw.get("n")), _number(raw.get("p50_ms")), _number(raw.get("p95_ms")))
    source_path = str(review.get("source_archive_path") or "").replace("/", "\\")
    match = re.search(r"\\(r[0-9]{2})(?:\\[^\\]+)?$", source_path, re.IGNORECASE)
    repeat_label = match.group(1).lower() if match else "r?"
    from ..ranking.mobile_experiment_identity import from_review_provenance
    roi_identity = from_review_provenance(review_provenance, verified=verify)
    roi_metadata = {}
    if roi_identity.get('family') in {'F5A', 'DIAGNOSTIC'}:
        roi_metadata = dict(experiment_type=roi_identity.get('experiment_type') or '',
            series_id=roi_identity.get('series_id') or '',
            roi_variant=roi_identity.get('variant') or 'UNPROVEN',
            is_pilot=roi_identity['is_pilot'], identity_status=roi_identity['status'],
            measurement_purpose=roi_identity.get('measurement_purpose', ''))
        repeat = roi_identity.get('replicate_index')
        repeat_label = f'r{repeat:02d}' if repeat else 'r?'
    _trace_step(diagnostics, "read_manifest_sha256", mf_path)
    result = Session(
        directory=directory, session_id=id_, review_revision=int(mf["review_revision"]),
        review_status=str(mf.get("review_status")), review_mode=str(mf.get("review_mode")),
        operator_approved=operator_approved,mt_evaluation=mt_evaluation,mt_object_evaluation=mt_objects,
        scene_assessment=scene_assessment,
        mz_input_evidence=mz_input_evidence,
        tv_bbox_evidence=tv_bbox_evidence, input_mode=input_context.get('input_mode', 'CAMERA_FULL_FRAME'),
        input_context=input_context or None,
        source_sha256=digest, manifest_sha256=_file_hash(mf_path),
        scenario=str(conditions.get("scenario") or "UNSPECIFIED"), repeat_label=repeat_label,
        device=str(conditions.get("device") or "unknown"), android=str(conditions.get("android") or "unknown"),
        apk_sha256=apk_sha, precision=prec, mt_checkpoint=chk_mt, mz_checkpoint=chk_mz,
        mt_model_sha=str(mt.get("model_sha256") or ""), mz_model_sha=str(mz.get("model_sha256") or ""),
        latency=stages, canvas=canvas,
        scientific_quality_status=str(sci.get("scientific_state") or "UNKNOWN"),
        source_evidence_status="manifest verified" if verify else "UNVERIFIED",
        model_identity=recorded_model_identity(review_provenance, verified=verify,
            recorded_package_id=(mf.get("experiment_session_as_recorded") or {}).get("package_id", "")),
        **roi_metadata,
    )
    if diagnostics is not None:
        diagnostics.update(operation="complete", manifest_verified=bool(verify),
                           manifest_sha256=result.manifest_sha256)
    return result


def scan_exports(root: Path) -> list[Session]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"No such directory: {root}")
    # To prevent scanning files in thesis source copies, walk only session/* export folders.
    entries = []
    for sessdir in sorted(root.iterdir()):
        if not sessdir.is_dir() or sessdir.name.lower() == "porownanie":
            continue
        for export_dir in sorted(sessdir.iterdir()):
            if export_dir.is_dir() and (export_dir / "export_manifest.json").exists():
                try:
                    entries.append(load_session_export(export_dir))
                except InvalidExport as exc:
                    raise InvalidExport(f"{export_dir}: {exc}") from exc
    return entries


def validate_selection(records: Iterable[Session]) -> list[str]:
    rows = list(records); warnings = []
    from ..ranking.mobile_tv_bbox_evidence import cohort_conflict as tv_conflict
    conflict = tv_conflict(getattr(r, 'input_context', None) or {'input_mode': getattr(r, 'input_mode', 'CAMERA_FULL_FRAME')} for r in rows)
    if conflict:
        raise ValueError('NOT_COMPARABLE / N/A: ' + conflict)
    from ..ranking.mobile_mz_input_evidence import is_diagnostic_purpose, diagnostic_cohort_conflict
    purposes = {r.measurement_purpose for r in rows}
    if diagnostic_cohort_conflict(purposes):
        raise ValueError('NOT_COMPARABLE / N/A: diagnostyka wymaga osobnej kohorty; rozdziel cele urządzeniowe, syntetyczne i oficjalne F4/F5A.')
    for purpose in purposes:
        if is_diagnostic_purpose(purpose):
            warnings.append(purpose + ': dane diagnostyczne poza oficjalnymi kohortami F4/F5A.')
    if not rows:
        return ["Nie wybrano żadnej sesji."]
    keys = [r.session_id for r in rows]
    if len(keys) != len(set(keys)):
        warnings.append("Powtórna rewizja tej samej sesji — nie licz jej jako nowego pomiaru.")
    shas = [r.source_sha256 for r in rows]
    if len(shas) != len(set(shas)):
        warnings.append("Powtórzono to samo archiwum źródłowe — różne kopie nie są oddzielnymi próbami.")
    scenarios = {r.scenario for r in rows}
    if len(scenarios) > 1:
        warnings.append("Wybrane różne scenariusze DAY/NIGHT: wyniki wymagają oddzielnych wykresów i wniosków.")
    devices = {r.device for r in rows if r.device != "unknown"}
    if len(devices) > 1:
        warnings.append("Różne urządzenia — czasów nie należy przypisywać wyłącznie kwantyzacji.")
    apks = {r.apk_sha256 for r in rows if r.apk_sha256}
    if len(apks) > 1:
        warnings.append("Różne APK! Wyniki nie są kontrolowanym porównaniem FP32/INT8 tej samej wersji aplikacji.")
    if any(not r.apk_sha256 for r in rows):
        warnings.append("Brak dowodu tej samej wersji APK dla wszystkich sesji.")
    unknown = [r.label for r in rows if r.precision == "unknown"]
    if unknown:
        warnings.append("Niepotwierdzona zgodność precyzji MT/MZ: " + ", ".join(unknown))
    mt_ck = {r.mt_checkpoint for r in rows if r.mt_checkpoint}
    mz_ck = {r.mz_checkpoint for r in rows if r.mz_checkpoint}
    if len(mt_ck) > 1 or len(mz_ck) > 1:
        warnings.append("Różne checkpointy modeli — nie jest to izolowany test FP32 vs INT8.")
    if any(not r.mt_checkpoint or not r.mz_checkpoint for r in rows):
        warnings.append("Brak dowodu wspólnego checkpointu co najmniej jednego modelu.")
    if not all(r.canvas_ready for r in rows):
        warnings.append("Nie wszystkie sesje mają ukończoną ocenę Canvas z GT — braków nie traktować jako 0%.")
    if any(r.scientific_quality_status != "VERIFIED" for r in rows):
        warnings.append("Ocena naukowa nie jest formalnie zweryfikowana; jakości używaj wyłącznie z Canvas (ASSISTED).")
    if len(rows) > 1:
        warnings.append("Zgodność scenariusza nie dowodzi identyczności klatek; parowanie zdjęcie-do-zdjęcia wymaga manifestu/hash.")
        warnings.append("Powtórzenia tej samej planszy nie są niezależnymi obserwacjami jakości.")
    return warnings


def _plot(record_pairs, *, title: str, y_label: str, value_fn, note: str, bounds=None, compact_labels=False):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, max(5.2, 1.1*len(record_pairs)+3)))
    fig.subplots_adjust(left=.37, right=.965, top=.85, bottom=.25)
    fig.suptitle(chart_identity_caption(record_pairs), fontsize=9, y=.98)
    if compact_labels:
        from .mobile_measurement_availability import polish_model_caption
        fig.subplots_adjust(left=.22)
        caption=polish_model_caption(chart_identity_caption(record_pairs))
        fig.suptitle('\n'.join(textwrap.wrap(caption,95)),fontsize=8,y=.98)
    names=[];values=[];notes=[]
    for rec in record_pairs:
        value, detail = value_fn(rec)
        if value is not None and math.isfinite(value):
            name=(f'{rec.scenario} · {rec.repeat_label}\n{rec.precision.upper()} · {rec.session_id[-8:]}'
                  if compact_labels else rec.label)
            names.append(name);values.append(value);notes.append(detail)
    if not names:
        ax.text(.5,.5,"Brak ocenialnych danych (nie zastępujemy ich zerem)",ha="center",va="center",transform=ax.transAxes)
        ax.set_xlim(0, 1);ax.set_yticks([])
    else:
        positions=list(range(len(names)))
        ax.barh(positions, values, height=.42)
        ax.set_yticks(positions, names)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim(len(names)-.07, -.93)
        high=max(values) if values else 1
        ax.set_xlim(0, max(1, high*1.35) if bounds is None else bounds[1])
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        for i, (v,d) in enumerate(zip(values,notes)):
            ax.text(v + (ax.get_xlim()[1]*.018), i, f"{v:.2f}  {d}",ha="left", va="center",fontsize=9)
    ax.set_title(title,pad=15)
    ax.set_xlabel(y_label)
    wrapped="\n".join(textwrap.wrap(note, width=120))
    fig.text(.04,.065,wrapped,fontsize=8,va="bottom",ha="left")
    return fig


def _geometry_text_figure(group: list[Session], *, scenario: str, measure: str):
    """Scatter of IoU/NMCE vs CER from *the same* detected plate.

    Descriptive only: the MT pose was not experimentally changed in this view.
    GT was corrected while seeing predictions, so no causal claims.
    """
    if measure == "nmce_pct":
        from .mobile_geometry_scatter import nmce_cer_figure
        return nmce_cer_figure(group, scenario=scenario)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    fig.subplots_adjust(left=.12, right=.96, top=.84, bottom=.28)
    fig.suptitle(chart_identity_caption(group), fontsize=9, y=.98)
    observed = 0
    for s in group:
        if not s.canvas_ready:
            continue
        rows = list(paired_geometry_text(s.canvas))
        if not rows:
            continue
        x = [r[measure] for r in rows]
        y = [r["cer_pct"] for r in rows]
        ax.scatter(x, y, label=f"{s.label}  (n={len(rows)})", s=50, alpha=.72)
        observed += len(rows)
    ax.set_xlabel("Polygon IoU MT–korekta [%]" if measure == "iou_pct" else "NMCE MT–korekta [%]")
    ax.set_ylabel("CER znaków dla tej samej tablicy [%]")
    ax.set_title(f"Geometria MT a błędy MZ · Canvas ASSISTED · {scenario}")
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)
    if measure == "iou_pct":
        ax.set_xlim(0, 102)
    else:
        ax.set_xlim(left=0)
    ax.set_ylim(bottom=-1)
    if not observed:
        ax.text(.5,.5,"Brak ocenialnych par IoU/NMCE + CER",transform=ax.transAxes,ha="center",va="center")
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
        fig.set_size_inches(12, 6.8)
        fig.subplots_adjust(right=.60)
    msg=("Każdy punkt to jedna tablica: oryginalna ramka MT vs ramka operatora oraz "
         "oryginalny odczyt MZ. N="+str(observed)+" par. GT ASSISTED; powtarzane sceny "
         "nie są niezależne. Korelacja nie dowodzi przyczynowości. "
         "Wpływ poprawionej ramki wymaga osobnego ponownego uruchomienia MZ.")
    fig.text(.05,.055,"\n".join(textwrap.wrap(msg,width=120)),fontsize=8,va="bottom")
    return fig


def _geometry_spread_figure(group: list[Session], *, scenario: str, measure: str):
    """Full distributions of plate-wise MT geometry mismatch, not means only."""
    import matplotlib.pyplot as plt
    series=[]; labels=[]
    for s in group:
        if not s.canvas_ready:
            continue
        values=[_number(r.get(measure))
                for r in (s.canvas or {}).get("rows",())
                if isinstance(r,dict) and r.get("included") and
                not r.get("post_nms_suppressed_raw") and r.get("outcome")=="LINKED_MT"]
        values=[v for v in values if v is not None]
        if values:
            series.append(values); labels.append(s.label+f"  (n={len(values)})")
    fig,ax=plt.subplots(figsize=(12,max(5.2,3+1.1*len(series))))
    fig.subplots_adjust(left=.37,right=.96,top=.82,bottom=.27)
    fig.suptitle(chart_identity_caption(group), fontsize=9, y=.98)
    if series:
        try:
            ax.boxplot(series,orientation='horizontal',tick_labels=labels,patch_artist=False,showfliers=True)
        except TypeError:  # matplotlib < 3.9
            ax.boxplot(series,vert=False,labels=labels,patch_artist=False,showfliers=True)
        if measure=="polygon_iou_pct":ax.set_xlim(0,102)
        else:ax.set_xlim(left=0)
        ax.grid(axis="x",alpha=.19)
        ax.tick_params(axis="y",labelsize=8)
    else:
        ax.text(.5,.5,"Brak ocenialnych ramek",transform=ax.transAxes,ha="center",va="center")
        ax.set_yticks([])
    ax.set_title(f"Rozrzut {'Polygon IoU' if measure=='polygon_iou_pct' else 'NMCE'} MT · Canvas ASSISTED · {scenario}")
    ax.set_xlabel("Polygon IoU względem korekty [%]" if measure=="polygon_iou_pct" else "NMCE względem korekty [%]")
    fig.text(.05,.06,"Obserwacje: sparowane narożniki MT i operatora. N pokazuje ocenione ramki; "
             "pominięcia MT (FN) raportowane oddzielnie. Ocena asystowana, nie ślepy test.",fontsize=8,va="bottom")
    return fig


def figures(records: Iterable[Session], *, comparison_scope=None):
    """Each figure is independent and captioned. No pseudo-comparison if n=1."""
    rows=list(records)
    if not rows:
        raise ValueError("Select at least one session")
    from .mobile_comparison_scope import scope_value
    scope = scope_value(comparison_scope, len(rows))
    for scenario in sorted({r.scenario for r in rows}):
        group=[r for r in rows if r.scenario==scenario]
        tag=re.sub(r"[^a-zA-Z0-9_-]","_",scenario)
        for stage,stage_label in _LATENCY_KEYS.items():
            for pct in ("p50_ms","p95_ms"):
                label="mediana (p50)" if pct=="p50_ms" else "95. percentyl (p95)"
                fig = _plot(group,title=f"{stage_label} — {label} · {scenario}",y_label="Czas [ms]",
                    value_fn=lambda r, stage=stage,pct=pct: (getattr(r.latency[stage],pct),f"(n={r.latency[stage].n} śladów)"),
                    note=("Czasy natywne Mobile: MT to czas inferencji, MZ to suma w obrębie śladu; "
                          "potok to czas rejestrowany na ślad. p95 liczony per sesja. "
                          "Różna liczba śladów nie oznacza porównania sparowanego."),compact_labels=scope=='performance')
                yield f"{tag}_{stage}_{pct}",fig
        if scope != 'performance' and any(r.canvas_ready for r in group):
            yield f"{tag}_assisted_exact",_plot(group,title=f"Dokładny odczyt MZ — Canvas ASSISTED · {scenario}",
                y_label="Exact Match [% ocenionych odczytów MZ]",
                value_fn=lambda r: (r.canvas_exact_pct,f"(n={r.canvas['evaluated_reads']})" if r.canvas_ready else ""),
                note="GT utworzono przy widocznej predykcji; nie jest to wynik niezależnego testu ALPR.",bounds=(0,120))
            yield f"{tag}_assisted_cer",_plot(group,title=f"CER znaków — Canvas ASSISTED · {scenario}",
                y_label="Micro-CER [% znaków GT]",
                value_fn=lambda r: (r.canvas_cer_pct,f"({r.canvas['text_errors']}/{r.canvas['gt_characters']} znaków)" if r.canvas_ready else ""),
                note="Micro-CER = suma błędów / suma znaków GT. Wyłącznie warunkowe odczyty MZ z Canvas.",bounds=None)
            yield f"{tag}_assisted_iou",_plot(group,title=f"Korekta ramki MT — Canvas ASSISTED · {scenario}",
                y_label="Średnie polygon IoU [%]",
                value_fn=lambda r: (r.canvas.get('polygon_iou_mean_pct') if r.canvas_ready and r.canvas.get('geometry_n',0)>0 else None,
                     f"(n={r.canvas['geometry_n']} korekt)" if r.canvas_ready else ""),
                note="IoU predykcja vs korekta operatora, nie niezależne GT geometrii.",bounds=(0,120))
            yield f"{tag}_assisted_iou_distribution", _geometry_spread_figure(
                group,scenario=scenario,measure="polygon_iou_pct")
            yield f"{tag}_assisted_nmce_distribution", _geometry_spread_figure(
                group,scenario=scenario,measure="nmce_pct")
            yield f"{tag}_assisted_iou_vs_cer_pairs", _geometry_text_figure(
                group,scenario=scenario,measure="iou_pct")
            yield f"{tag}_assisted_nmce_vs_cer_pairs", _geometry_text_figure(
                group,scenario=scenario,measure="nmce_pct")


def _csv_summary(records, path:Path):
    columns=("session_id","review_revision","scenario","repeat","precision","source_sha256","apk_sha256",
             "MT_n","MT_p50_ms","MT_p95_ms","MZ_n","MZ_p50_ms","MZ_p95_ms",
             "ALPR_n","ALPR_p50_ms","ALPR_p95_ms","canvas_assisted","canvas_exact_n",
             "canvas_reads_n","canvas_exact_pct","canvas_cer_pct","canvas_pose_n","canvas_iou_mean_pct")
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader()
        for r in records:
            c=r.canvas or {}
            writer.writerow({
                "session_id":r.session_id,"review_revision":r.review_revision,"scenario":r.scenario,
                "repeat":r.repeat_label,"precision":r.precision,"source_sha256":r.source_sha256,"apk_sha256":r.apk_sha256,
                "MT_n":r.latency['mt'].n,"MT_p50_ms":r.latency['mt'].p50_ms,"MT_p95_ms":r.latency['mt'].p95_ms,
                "MZ_n":r.latency['mz'].n,"MZ_p50_ms":r.latency['mz'].p50_ms,"MZ_p95_ms":r.latency['mz'].p95_ms,
                "ALPR_n":r.latency['pipeline'].n,"ALPR_p50_ms":r.latency['pipeline'].p50_ms,
                "ALPR_p95_ms":r.latency['pipeline'].p95_ms,"canvas_assisted":r.canvas_ready,
                "canvas_exact_n":c.get('exact_reads') if r.canvas_ready else None,
                "canvas_reads_n":c.get('evaluated_reads') if r.canvas_ready else None,
                "canvas_exact_pct":r.canvas_exact_pct,"canvas_cer_pct":r.canvas_cer_pct,
                "canvas_pose_n":c.get('geometry_n') if r.canvas_ready else None,
                "canvas_iou_mean_pct":c.get('polygon_iou_mean_pct') if r.canvas_ready else None,
            })


def export_figures(records: Iterable[Session], root: Path, *, single_session=False) -> Path:
    """Create versioned PNG/SVG/CSV with provenance manifest; source data read-only."""
    import matplotlib.pyplot as plt
    rows=list(records)
    if not rows: raise ValueError("Select at least one report")
    if len({r.session_id for r in rows}) != len(rows) or len({r.source_sha256 for r in rows}) != len(rows):
        raise ValueError("The same source session cannot be counted twice")
    root=Path(root)
    if not root.is_dir():
        raise ValueError("Results root is missing")
    if single_session and len(rows)==1:
        parent=root/rows[0].session_id
        if not parent.is_dir():
            parent.mkdir(mode=0o755, parents=False, exist_ok=False)  # only under an existing 03_wyniki root
        stem=f"wykresy_r{rows[0].review_revision:06d}"
    else:
        parent=root/"porownanie"
        if not parent.is_dir():
            raise ValueError("Comparison folder is missing; do not silently create it")
        stem="porownanie_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Create a private staging folder and move it only after all exports succeed.
    staging=Path(tempfile.mkdtemp(prefix=".alpr_figure_export_",dir=str(parent)))
    destination=None
    try:
        output=[]
        for name,fig in figures(rows):
            try:
                for ext in ("png","svg"):
                    path=staging/(name+"."+ext)
                    fig.savefig(path,dpi=240 if ext=="png" else None,facecolor="white")
                    output.append(path)
            finally:
                plt.close(fig)
        csv_path=staging/"porownanie_dane.csv"
        _csv_summary(rows,csv_path)
        output.append(csv_path)
        manifest={
            "schema":CHART_SCHEMA,
            "generated_utc":datetime.now(timezone.utc).isoformat(),
            "scope":"single_session" if single_session else "multi_session",
            "sources":[{
                "session_id":r.session_id,"review_revision":r.review_revision,"export_manifest_sha256":r.manifest_sha256,
                "source_archive_sha256":r.source_sha256,"scenario":r.scenario,
                "precision":r.precision,"repeat":r.repeat_label,"review_mode":r.review_mode,
                "apk_sha256":r.apk_sha256,
                "measurement_purpose":r.measurement_purpose,
                **({'input_mode': r.input_mode, 'input_context': r.input_context} if r.input_mode == 'TV_BBOX_CROP' else {}),
                "quality_evaluable_assisted":r.canvas_ready,"scientific_quality_status":r.scientific_quality_status,
                "mt_checkpoint":r.mt_checkpoint,"mz_checkpoint":r.mz_checkpoint,
                "package_and_models":r.model_identity,
            } for r in rows],
            "warnings":validate_selection(rows),
            "files":[{"name":p.name,"bytes":p.stat().st_size,"sha256":_file_hash(p)} for p in output]
        }
        (staging/"wykresy_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        for k in range(1,1000):
            candidate=parent/(f"{stem}_{k:03d}")
            if not candidate.exists():
                destination=candidate
                break
        if destination is None:
            raise RuntimeError("No available export version")
        staging.rename(destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)
