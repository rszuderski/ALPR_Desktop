"""Human annotations and reproducible mobile quality metrics; independent of Tk."""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import tempfile
import uuid
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Any

from .mobile_package_experiments import (
    MobileReportBundle, _file_sha256, _utc_now_iso, iter_full_attempt_rows,
    iter_full_sample_annotations, iter_full_sample_rows, read_mobile_report_bundle,
    read_mobile_sample_image,
)
from .mobile_mt_invocations import DETECTION_STATUSES, group_mt_invocations, calculate_mt_invocations

REVIEW_SCHEMA = "alpr.mobile_human_review.v1"
NORMALIZATION_POLICY = "uppercase_alphanumeric.v1"
LEGACY_NOTICE = ("Sesja nie zawiera pełnego rejestru prób MT. "
                 "Możliwa jest weryfikacja cropów MZ / odczytu end-to-end.")


def normalize_registration(value: Any) -> str:
    # Matches the desktop OCR cleanup; deliberately no O/0 or I/1 correction.
    return "".join(char for char in str(value or "").upper() if char.isalnum())


@dataclass(frozen=True)
class PlateTextAlignment:
    ground_truth: str
    prediction: str
    exact_match: bool
    correct_characters: int
    incorrect_characters: int
    missing_characters: int
    extra_characters: int
    edit_distance: int
    cer: float | None
    alignment: tuple[dict[str, str], ...]


@lru_cache(maxsize=8192)
def align_plate_text(ground_truth: str, prediction: str) -> PlateTextAlignment:
    gt, pred = normalize_registration(ground_truth), normalize_registration(prediction)
    if len(gt) > 256 or len(pred) > 2048:
        raise ValueError("Tekst przekracza limit długości rejestracji do porównania.")
    # Optimize edit distance, then preserve the most matches. Remaining ties
    # prefer diagonal, missing, extra, so confusion counts are deterministic.
    costs = [[(0, 0)] * (len(pred) + 1) for _ in range(len(gt) + 1)]
    moves = [[""] * (len(pred) + 1) for _ in range(len(gt) + 1)]
    for i in range(1, len(gt) + 1):
        costs[i][0], moves[i][0] = (i, 0), "missing"
    for j in range(1, len(pred) + 1):
        costs[0][j], moves[0][j] = (j, 0), "extra"
    for i in range(1, len(gt) + 1):
        for j in range(1, len(pred) + 1):
            equal = gt[i - 1] == pred[j - 1]
            diagonal = costs[i - 1][j - 1]
            missing, extra = costs[i - 1][j], costs[i][j - 1]
            choices = [((diagonal[0] + (not equal), diagonal[1] - equal), 0, "correct" if equal else "incorrect"),
                       ((missing[0] + 1, missing[1]), 1, "missing"),
                       ((extra[0] + 1, extra[1]), 2, "extra")]
            cost, _, operation = min(choices)
            costs[i][j], moves[i][j] = cost, operation
    alignment = []
    i, j = len(gt), len(pred)
    while i or j:
        operation = moves[i][j]
        alignment.append({"kind": operation, "ground_truth": gt[i - 1] if operation != "extra" else "",
                          "prediction": pred[j - 1] if operation != "missing" else ""})
        i -= operation != "extra"
        j -= operation != "missing"
    alignment.reverse()
    counts = Counter(item["kind"] for item in alignment)
    distance = counts["incorrect"] + counts["missing"] + counts["extra"]
    return PlateTextAlignment(gt, pred, gt == pred, counts["correct"], counts["incorrect"],
                              counts["missing"], counts["extra"], distance,
                              distance / len(gt) if gt else None, tuple(alignment))


def _bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "yes", "1"}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _first(row: dict, *keys: str) -> str:
    return next((str(row[key]).strip() for key in keys if row.get(key) not in (None, "")), "")


def _stamp(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass
class MobileHumanReview:
    source_archive_path: str
    source_archive_sha256: str
    session_id: str
    schema: str = REVIEW_SCHEMA
    review_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    reviewer_id: str = ""
    review_status: str = "NOT_STARTED"
    review_revision: int = 0
    review_mode: str = "blinded_gt_v1"
    review_mode_locked: bool = False
    mt_review_policy: str = "invocation_v1"
    normalization_policy: str = NORMALIZATION_POLICY
    subjects: dict[str, dict] = field(default_factory=dict)
    attempt_annotations: dict[str, dict] = field(default_factory=dict)
    invocation_annotations: dict[str, dict] = field(default_factory=dict)
    sample_annotations: dict[str, dict] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict, *, source_sha256: str, session_id: str) -> "MobileHumanReview":
        if data.get("schema") != REVIEW_SCHEMA:
            raise ValueError("Nieobsługiwany schemat weryfikacji.")
        if data.get("source_archive_sha256") != source_sha256 or data.get("session_id") != session_id:
            raise ValueError("Weryfikacja dotyczy innego archiwum (SHA-256 lub identyfikator sesji).")
        if data.get("normalization_policy", NORMALIZATION_POLICY) != NORMALIZATION_POLICY:
            raise ValueError("Weryfikacja używa innej polityki normalizacji tekstu.")
        values = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        values.setdefault("review_mode", "assisted")
        values.setdefault("mt_review_policy", "legacy_visibility_v1")
        values.setdefault("review_mode_locked", bool(data.get("subjects") or data.get("attempt_annotations") or data.get("sample_annotations")))
        result = cls(**values)
        if result.review_mode not in {"blinded_gt_v1", "assisted"} or result.mt_review_policy not in {"invocation_v1", "legacy_visibility_v1"}:
            raise ValueError("Nieobsługiwany tryb lub kontrakt weryfikacji.")
        result.review_status = result.review_status.upper()
        if result.review_status not in {"NOT_STARTED", "IN_PROGRESS", "COMPLETED"}:
            raise ValueError("Nieprawidłowy status weryfikacji.")
        for mapping in (result.subjects, result.attempt_annotations, result.sample_annotations, result.invocation_annotations):
            if not isinstance(mapping, dict) or any(not isinstance(value, dict) for value in mapping.values()):
                raise ValueError("Nieprawidłowy zapis decyzji operatora.")
        result.review_mode_locked = bool(result.review_mode_locked or any((result.subjects, result.attempt_annotations,
                                                                         result.sample_annotations, result.invocation_annotations)))
        return result


class MobileReviewSession:
    """One validated source, full record indexes and atomically saved decisions."""

    def __init__(self, bundle: MobileReportBundle, *, sidecar_path: Path | None = None):
        from ..config import CONFIG
        if not bundle.validation.ok:
            raise ValueError("Archiwum nie przeszło walidacji: " + "; ".join(bundle.validation.errors))
        self.bundle = bundle
        self.path = Path(bundle.path).resolve()
        if not zipfile.is_zipfile(self.path):
            raise ValueError("Weryfikacja dowodów wymaga źródłowego archiwum .alprsession lub ZIP.")
        self._source_stamp = None
        self.verify_source(force=True)
        self.attempts_available = bundle.attempts_available
        self.samples: dict[str, dict] = {}
        self.sample_ids_by_attempt: dict[str, list[str]] = {}
        self.attempts: dict[str, dict] = {}
        self.subjects: dict[str, dict] = {}
        self.warnings: list[str] = [] if self.attempts_available else [LEGACY_NOTICE]
        self.session_id = str(bundle.experiment_session.get("experiment_session_id")
                              or bundle.collection_session.get("session_id")
                              or bundle.report_payload.get("session_id") or bundle.report.report_id)
        rows = list(iter_full_sample_rows(bundle))
        attempt_rows = list(iter_full_attempt_rows(bundle)) if self.attempts_available else []
        source_sessions = {str(row["session_id"]) for row in rows + attempt_rows if row.get("session_id")}
        if len(source_sessions) > 1:
            raise ValueError("Indeks próbek zawiera więcej niż jedną sesję.")
        if source_sessions:
            self.session_id = next(iter(source_sessions))
        annotations = {}
        for row in iter_full_sample_annotations(bundle):
            key = _first(row, "capture_id", "sample_id", "id")
            if not key or key in annotations:
                raise ValueError("Brak lub zduplikowany identyfikator adnotacji próbki.")
            annotations[key] = row
        self.entry_names = {entry.name for entry in bundle.entries}
        for index, row in enumerate(attempt_rows):
            key = _first(row, "attempt_id")
            if not key or key in self.attempts:
                raise ValueError("Brak lub zduplikowany attempt_id w rejestrze prób.")
            self.attempts[key] = dict(row, id=key, order=index)
            self._assign_subject(self.attempts[key], "attempts")
        for index, raw in enumerate(rows):
            key = _first(raw, "capture_id", "sample_id", "id") or f"legacy-row-{index + 1}"
            if key in self.samples:
                raise ValueError(f"Zduplikowana próbka: {key}")
            original = annotations.pop(key, {})
            row = dict(original, **raw, id=key, order=index)
            attempt = self.attempts.get(_first(row, "attempt_id"), {})
            if attempt:
                if row.get("subject_key") and row["subject_key"] != attempt["subject_key"]:
                    raise ValueError("Próba i crop wskazują różne subject_key.")
                row["subject_key"] = attempt["subject_key"]
            image_entry = _first(row, "image", "image_entry", "crop_entry", "crop_path")
            if not image_entry:
                image_entry = next((f"samples/crops/{key}{suffix}" for suffix in (".jpg", ".jpeg", ".png", ".webp")
                                    if f"samples/crops/{key}{suffix}" in self.entry_names), "")
            row["image_entry"] = image_entry
            # New Android attempts preserve the fresh MZ result, even when empty.
            # Do not substitute a previously accumulated consensus for a no-read.
            if "prediction" in attempt:
                row["prediction"] = attempt["prediction"]
            elif "fresh_prediction" in original:
                row["prediction"] = original["fresh_prediction"]
            self.samples[key] = row
            if row.get("attempt_id"):
                self.sample_ids_by_attempt.setdefault(row["attempt_id"], []).append(key)
            self._assign_subject(row, "samples")
        if annotations:
            self.warnings.append(f"Adnotacje poza indeksem cropów: {len(annotations)}. Nie są liczone jako odczyty.")
        self.mt_invocations = group_mt_invocations(self.attempts, self.session_id)
        self.invocation_by_attempt = {row["id"]: key for key, group in self.mt_invocations.items() for row in group.records}
        self.verify_source()
        self.sidecar_path = Path(sidecar_path) if sidecar_path else (
            CONFIG.DIR_7_RANKINGS_MOBILE_PACKAGES / "human_reviews" / f"{bundle.source_archive_sha256}.review.json")
        if self.sidecar_path.resolve() == self.path:
            raise ValueError("Weryfikacja musi być zapisana poza źródłową sesją.")
        provenance = {"model_provenance": copy.deepcopy(bundle.report_payload.get("model_provenance", {})),
                      "model_refs": copy.deepcopy(bundle.model_refs), "pipeline_manifests": copy.deepcopy(bundle.pipeline_manifests),
                      "experiment_session": copy.deepcopy(bundle.experiment_session), "device": copy.deepcopy(bundle.report.device),
                      "runtime": bundle.report.runtime, "variant_id": bundle.report.variant_id,
                      "collection_session": copy.deepcopy(bundle.collection_session or bundle.report_payload.get("research_collection", {})),
                      "capture": copy.deepcopy(bundle.report_payload.get("capture", {})),
                      "research_execution_config": copy.deepcopy(bundle.report_payload.get("research_execution_config", {}))}
        self.review = MobileHumanReview(str(self.path), bundle.source_archive_sha256, self.session_id, provenance=provenance)
        if self.sidecar_path.exists():
            self.review = MobileHumanReview.from_dict(json.loads(self.sidecar_path.read_text(encoding="utf-8")),
                source_sha256=bundle.source_archive_sha256, session_id=self.session_id)
            self._validate_decisions(self.review)
            if self.review.review_status == "COMPLETED" and self.completion_issues():
                raise ValueError("Zapis oznaczono jako ukończony mimo brakujących decyzji.")
            if any(group.executed and not group.cancelled and self.invocation_annotation(key).get("decision_source") == "legacy_cardinality_unknown"
                   for key, group in self.mt_invocations.items()):
                self.warnings.append("Starsza weryfikacja nie zawiera liczby tablic na całym wejściu MT. Te wywołania wymagają nowej oceny, aby wejść do głównej miary MT.")

    @classmethod
    def open(cls, path: Path, **kwargs) -> "MobileReviewSession":
        return cls(read_mobile_report_bundle(path), **kwargs)

    def verify_source(self, *, force=False) -> None:
        before = _stamp(self.path)
        if force or self._source_stamp != before:
            if _file_sha256(self.path) != self.bundle.source_archive_sha256 or _stamp(self.path) != before:
                raise ValueError("Źródłowa sesja zmieniła się. Weryfikacja nie zostanie do niej podłączona.")
            self._source_stamp = before

    def _assign_subject(self, row: dict, kind: str) -> None:
        key = _first(row, "subject_key")
        legacy = not key
        if not key:
            scene = _first(row, "scene_generation") or "unknown-scene"
            identity = next((f"{name}-{row[name]}" for name in ("entity_id", "plate_track_id", "track_id", "vehicle_track_id")
                             if row.get(name) not in (None, "", "0", 0, "-1", -1)), f"{kind}-{row['id']}")
            key = f"{self.session_id}/legacy/{scene}/{identity}"
        row["subject_key"] = key
        subject = self.subjects.setdefault(key, {"key": key, "legacy_identity": legacy, "samples": [], "attempts": []})
        subject[kind].append(row["id"])

    def image(self, row: dict):
        self.verify_source()
        result = read_mobile_sample_image(self.bundle, _first(row, "image_entry", "evidence_entry"))
        self.verify_source()
        return result

    def _validate_decisions(self, review: MobileHumanReview) -> None:
        for decisions, records in ((review.subjects, self.subjects), (review.attempt_annotations, self.attempts),
                                   (review.sample_annotations, self.samples), (review.invocation_annotations, self.mt_invocations)):
            if set(decisions) - set(records):
                raise ValueError("Weryfikacja zawiera identyfikatory spoza źródłowej sesji.")
            for value in decisions.values():
                for name in ("evaluable", "is_plate", "gt_confirmed"):
                    if name in value and value[name] is not None and not isinstance(value[name], bool):
                        raise ValueError("Decyzja operatora musi mieć wartość logiczną.")
                if "plate_visibility" in value and value["plate_visibility"] not in {"visible", "invisible", "uncertain"}:
                    raise ValueError("Nieprawidłowa ocena widoczności tablicy.")
                if "visible_plate_count" in value and value["visible_plate_count"] not in {"none", "one", "multiple", "uncertain"}:
                    raise ValueError("Nieprawidłowa ocena liczby tablic na wejściu MT.")
        for subject in review.subjects.values():
            if subject.get("evaluable") is True and not normalize_registration(subject.get("ground_truth")):
                raise ValueError("Oceniana tablica wymaga niepustego GT.")

    def _save_change(self, changed: MobileHumanReview) -> None:
        self.verify_source()
        self._validate_decisions(changed)
        if self.review.review_mode_locked and changed.review_mode != self.review.review_mode:
            raise ValueError("Tryb weryfikacji jest stały po pierwszej zapisanej decyzji.")
        changed.review_mode_locked = self.review.review_mode_locked or any((changed.subjects, changed.attempt_annotations,
                                                                          changed.sample_annotations, changed.invocation_annotations))
        if self.sidecar_path.exists():
            saved = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
            if (saved.get("review_id") != self.review.review_id
                    or saved.get("review_revision") != self.review.review_revision
                    or saved.get("source_archive_sha256") != self.review.source_archive_sha256):
                raise ValueError("Weryfikacja została zmieniona w innym oknie. Otwórz ją ponownie przed zapisem.")
        changed.review_revision = self.review.review_revision + 1
        changed.updated_at = _utc_now_iso()
        _atomic_write(self.sidecar_path, json.dumps(changed.to_dict(), ensure_ascii=False, indent=2))
        self.review = changed

    def set_subject(self, key: str, *, ground_truth: str = "", evaluable: bool | None = True, note: str = "") -> None:
        if key not in self.subjects:
            raise KeyError(key)
        gt = normalize_registration(ground_truth)
        if len(gt) > 256:
            raise ValueError("GT jest zbyt długie.")
        changed = copy.deepcopy(self.review)
        changed.subjects[key] = {"ground_truth": gt, "evaluable": evaluable, "note": note,
                                 "gt_confirmed": bool(gt) and (evaluable is True or self.review.subjects.get(key, {}).get("gt_confirmed", False)),
                                 "legacy_identity": self.subjects[key]["legacy_identity"]}
        changed.review_status = "IN_PROGRESS"
        self._save_change(changed)

    def predictions_visible(self, key: str) -> bool:
        subject = self.review.subjects.get(key, {})
        return self.review.review_mode == "assisted" or bool(subject.get("gt_confirmed") and subject.get("ground_truth"))

    def subject_draft(self, key: str) -> str:
        subject = self.review.subjects.get(key, {})
        return str(subject.get("draft_ground_truth", subject.get("ground_truth", "")))

    def save_subject_draft(self, key: str, *, ground_truth: str, note: str = "") -> None:
        if key not in self.subjects:
            raise KeyError(key)
        gt = normalize_registration(ground_truth)
        if len(gt) > 256:
            raise ValueError("GT jest zbyt długie.")
        if self.predictions_visible(key):
            self.set_subject(key, ground_truth=ground_truth, note=note, evaluable=True if gt else None)
            return
        changed = copy.deepcopy(self.review)
        decision = dict(changed.subjects.get(key, {}))
        decision.update(draft_ground_truth=str(ground_truth), note=note, legacy_identity=self.subjects[key]["legacy_identity"])
        if gt != normalize_registration(self.subject_draft(key)):
            decision["evaluable"] = None
        changed.subjects[key] = decision
        changed.review_status = "IN_PROGRESS"
        self._save_change(changed)

    def set_review_mode(self, mode: str) -> None:
        if mode not in {"blinded_gt_v1", "assisted"}:
            raise ValueError("Nieznany tryb weryfikacji.")
        changed = copy.deepcopy(self.review)
        changed.review_mode = mode
        self._save_change(changed)

    def annotate_invocation(self, key: str, *, visible_plate_count: str, evaluable: bool = True) -> None:
        changed = copy.deepcopy(self.review)
        changed.invocation_annotations[key] = {"visible_plate_count": visible_plate_count, "evaluable": evaluable}
        changed.review_status = "IN_PROGRESS"
        self._save_change(changed)

    def invocation_annotation(self, key: str) -> dict:
        if key in self.review.invocation_annotations:
            return self.review.invocation_annotations[key]
        if self.review.mt_review_policy != "legacy_visibility_v1":
            return {}
        group = self.mt_invocations[key]
        # Old per-record visibility never established the count in the full MT
        # input, even when the backend returned only one detection. Preserve the
        # historical review, but do not manufacture a one-plate decision.
        excluded = (all(self.review.subjects.get(s, {}).get("evaluable") is False for s in group.subject_keys)
                    or all(self.review.attempt_annotations.get(row["id"], {}).get("evaluable") is False
                           or self.review.attempt_annotations.get(row["id"], {}).get("plate_visibility") == "uncertain" for row in group.records))
        historical_complete = self.review.review_status == "COMPLETED"
        if historical_complete or excluded or any(row["id"] in self.review.attempt_annotations for row in group.records):
            return {"visible_plate_count": "uncertain" if historical_complete or excluded else None,
                    "evaluable": False if historical_complete or excluded else None, "decision_source": "legacy_cardinality_unknown"}
        return {}

    def invocation_completion_issues(self, key: str) -> list[str]:
        group = self.mt_invocations[key]
        if group.cancelled or not group.executed or group.execution_failed:
            return []
        decision = self.invocation_annotation(key)
        if decision.get("evaluable") is False or decision.get("visible_plate_count") == "uncertain":
            return []
        if decision.get("visible_plate_count") not in {"none", "one", "multiple"}:
            return [f"Brak oceny liczby tablic w wywołaniu MT: {key}"]
        evidence = any(entry in self.entry_names for entry in group.evidence_entries)
        if not evidence:
            return [f"Brak dowodu wejścia MT; oznacz wywołanie jako nie do oceny: {key}"]
        if self.review.mt_review_policy == "legacy_visibility_v1" and key not in self.review.invocation_annotations:
            return []
        if decision.get("visible_plate_count") == "none":
            return []
        return [f"Brak oceny detekcji: {row['id']}" for row in group.records
                if row.get("mt_status") in DETECTION_STATUSES
                and self.review.attempt_annotations.get(row["id"], {}).get("evaluable") is not False
                and self.review.attempt_annotations.get(row["id"], {}).get("is_plate") not in (True, False)]

    def annotate(self, kind: str, key: str, **decision) -> None:
        if kind not in {"attempt", "sample"}:
            raise ValueError("Nieznany rodzaj dowodu.")
        changed = copy.deepcopy(self.review)
        target = changed.attempt_annotations if kind == "attempt" else changed.sample_annotations
        target[key] = dict(target.get(key, {}), **decision)
        changed.review_status = "IN_PROGRESS"
        self._save_change(changed)

    def set_reviewer(self, reviewer_id: str) -> None:
        changed = copy.deepcopy(self.review)
        changed.reviewer_id = str(reviewer_id).strip()
        self._save_change(changed)

    def cancelled(self, row: dict) -> bool:
        attempt = self.attempts.get(_first(row, "attempt_id"), {})
        return _bool(row.get("stale_or_cancelled")) or _bool(attempt.get("stale_or_cancelled"))

    def attempt_has_evidence(self, row: dict) -> bool:
        return row.get("evidence_entry") in self.entry_names or any(
            self.samples[key].get("image_entry") in self.entry_names
            for key in self.sample_ids_by_attempt.get(row["id"], ()))

    def subject_completion_issues(self, key: str) -> list[str]:
        issues = []
        subject = self.subjects[key]
        rows = [self.samples[k] for k in subject["samples"]] + [self.attempts[k] for k in subject["attempts"]]
        if rows and all(self.cancelled(row) for row in rows):
            return issues
        decision = self.review.subjects.get(key, {})
        if decision.get("evaluable") is not False and (not decision.get("evaluable") or not normalize_registration(decision.get("ground_truth"))):
            issues.append(f"Brak GT lub decyzji „nie do oceny”: {key}")
        elif decision.get("evaluable") is True and not self.predictions_visible(key):
            issues.append(f"GT wymaga jawnego zatwierdzenia: {key}")
        for invocation in dict.fromkeys(self.invocation_by_attempt[attempt_id] for attempt_id in subject["attempts"]):
            issues.extend(self.invocation_completion_issues(invocation))
        return issues

    def completion_issues(self) -> list[str]:
        if not self.subjects:
            return ["Sesja nie zawiera próbek ani prób do weryfikacji."]
        return list(dict.fromkeys(issue for key in self.subjects for issue in self.subject_completion_issues(key)))

    def complete(self) -> None:
        issues = self.completion_issues()
        if issues:
            raise ValueError(f"Pozostałe decyzje: {len(issues)}. {issues[0]}")
        changed = copy.deepcopy(self.review)
        changed.review_status = "COMPLETED"
        self._save_change(changed)

    def statistics(self) -> dict:
        return calculate_review_statistics(self)

    def export(self, directory: Path) -> list[Path]:
        self.verify_source(force=True)
        directory = Path(directory)
        paths = [directory / name for name in ("review.json", "summary.csv", "subjects.csv", "character_confusion.csv")]
        if self.path in {path.resolve() for path in paths}:
            raise ValueError("Eksport nie może zastąpić źródłowej sesji.")
        stats = self.statistics()
        payload = self.review.to_dict()
        payload["derived_metrics"] = stats
        _atomic_write(paths[0], json.dumps(payload, ensure_ascii=False, indent=2))
        summary = dict(stats["summary"], source_archive_sha256=self.review.source_archive_sha256,
                       session_id=self.session_id, review_status=self.review.review_status,
                       review_revision=self.review.review_revision, provenance=self.review.provenance)
        for path, rows, columns in ((paths[1], [summary], list(summary)),
                (paths[2], stats["subjects"], list(stats["subjects"][0]) if stats["subjects"] else ["subject_key"]),
                (paths[3], stats["character_confusion"], ["ground_truth", "prediction", "count"])):
            output = io.StringIO(newline="")
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})
            _atomic_write(path, "\ufeff" + output.getvalue())
        return paths


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def calculate_review_statistics(session: MobileReviewSession) -> dict:
    review = session.review
    summary = Counter()
    confusion = Counter()
    reads, subjects, mt_rows = [], [], []
    mt_summary, invocations = calculate_mt_invocations(session)
    invocation_results = {row["mt_invocation_id"]: row for row in invocations}
    misses_by_subject = Counter(key for row in invocations if row["included"] and row["outcome"] == "no_detection" for key in row["subject_keys"])
    summary.update(subject_count=len(session.subjects), attempt_count=len(session.attempts), crop_count=len(session.samples))
    for key, subject in session.subjects.items():
        decision = review.subjects.get(key, {})
        gt = normalize_registration(decision.get("ground_truth"))
        all_rows = [session.samples[k] for k in subject["samples"]] + [session.attempts[k] for k in subject["attempts"]]
        all_cancelled = all(session.cancelled(row) for row in all_rows)
        evaluable = decision.get("evaluable") is True and bool(gt) and not all_cancelled and session.predictions_visible(key)
        pending = len(session.subject_completion_issues(key))
        reviewed = not pending
        summary["reviewed_subjects" if reviewed else "not_reviewed_subjects"] += 1
        summary["evaluable_subjects"] += evaluable
        summary["pending_decisions"] += pending
        subject_reads = []
        for sample_id in subject["samples"]:
            sample = session.samples[sample_id]
            annotation = review.sample_annotations.get(sample_id, {})
            attempt = session.attempts.get(_first(sample, "attempt_id"), {})
            attempt_annotation = review.attempt_annotations.get(attempt.get("id", ""), {})
            excluded = (not evaluable or annotation.get("evaluable") is False or annotation.get("is_plate") is False
                        or attempt_annotation.get("evaluable") is False or attempt_annotation.get("is_plate") is False
                        or attempt_annotation.get("plate_visibility") in {"invisible", "uncertain"} or session.cancelled(sample)
                        or str(attempt.get("mz_status", sample.get("mz_status", ""))).upper() == "NOT_RUN")
            has_image = sample.get("image_entry") in session.entry_names
            summary["missing_crop_count"] += not has_image
            if excluded or not has_image:
                continue
            aligned = align_plate_text(gt, str(sample.get("prediction", sample.get("text", "")) or ""))
            outcome = "no_read" if not aligned.prediction else "exact" if aligned.exact_match else "incorrect"
            result = dict(asdict(aligned), sample_id=sample_id, subject_key=key, outcome=outcome)
            reads.append(result)
            subject_reads.append(result)
            summary["evaluable_reads"] += 1
            summary[{"no_read": "no_reads", "exact": "exact_reads", "incorrect": "incorrect_reads"}[outcome]] += 1
            summary["gt_characters"] += len(gt)
            for name in ("correct_characters", "incorrect_characters", "missing_characters", "extra_characters"):
                summary[name] += getattr(aligned, name)
            for item in aligned.alignment:
                if item["kind"] == "incorrect":
                    confusion[item["ground_truth"], item["prediction"]] += 1
        valid_attempts = []
        false_sample_attempts = {session.samples[s].get("attempt_id") for s in subject["samples"]
                                 if review.sample_annotations.get(s, {}).get("is_plate") is False}
        for attempt_id in subject["attempts"]:
            row = session.attempts[attempt_id]
            annotation = review.attempt_annotations.get(attempt_id, {})
            linked_false = attempt_id in false_sample_attempts
            status = _first(row, "mt_status").upper()
            cancelled = session.cancelled(row)
            false_detection = (not cancelled and status in {"VALID_QUAD", "DETECTION_INVALID_QUAD"}
                               and (annotation.get("is_plate") is False or linked_false))
            invocation = invocation_results[session.invocation_by_attempt[attempt_id]]
            mt_rows.append({"attempt_id": attempt_id, "subject_key": key, "mt_status": status,
                            "mt_invocation_id": invocation["mt_invocation_id"], "invocation_included": invocation["included"],
                            "false_detection": false_detection})
            if (not cancelled and annotation.get("evaluable") is not False and not false_detection
                    and annotation.get("plate_visibility") not in {"invisible", "uncertain"}):
                valid_attempts.append(row)
        result = {"subject_key": key, "ground_truth": gt, "evaluable": evaluable,
                  "legacy_identity": subject["legacy_identity"], "reviewed": reviewed,
                  "pending_decisions": pending, "evaluable_reads": len(subject_reads),
                  "mt_no_detection_invocations": misses_by_subject[key] if session.attempts_available else None,
                  "sample_count": len(subject["samples"]), "attempt_count": len(subject["attempts"]),
                  "subject_exact_success": None, "attempts_to_first_exact": None, "time_to_first_exact_ms": None,
                  "time_basis": None, "first_exact_capture_source": None, "first_exact_camera_zoom_ratio": None,
                  "baseline_exact_before_az": None, "exact_only_after_az": None, "consensus_repaired_result": None}
        if evaluable:
            candidates = valid_attempts if session.attempts_available else [session.samples[r["sample_id"]] for r in subject_reads]
            time_field = next((name for name in ("attempt_started_elapsed_nanos", "captured_at_ms")
                               if candidates and all(_number(row.get(name)) is not None for row in candidates)), None)
            candidates = sorted(candidates, key=lambda row: _number(row[time_field]) if time_field else row["order"])
            exact_records = []
            for index, row in enumerate(candidates):
                mz_run = str(row.get("mz_status", "READ")).upper() not in {"NOT_RUN"}
                raw_exact = mz_run and normalize_registration(row.get("prediction")) == gt
                consensus = normalize_registration(row.get("consensus_prediction"))
                consensus_exact = bool(consensus) and consensus == gt
                if raw_exact or consensus_exact:
                    exact_records.append((index, row, consensus_exact and not raw_exact))
            success = bool(exact_records) or any(row["exact_match"] for row in subject_reads)
            result["subject_exact_success"] = success
            summary["subjects_with_exact_read" if success else "subjects_without_exact_read"] += 1
            if any("consensus_prediction" in row for row in candidates):
                result["consensus_repaired_result"] = any(repaired for _, _, repaired in exact_records)
            if exact_records:
                index, first, _ = exact_records[0]
                result["first_exact_capture_source"] = first.get("capture_source") or None
                result["first_exact_camera_zoom_ratio"] = _number(first.get("camera_zoom_ratio"))
                if session.attempts_available:
                    result["attempts_to_first_exact"] = index + 1
                    if time_field:
                        divisor = 1_000_000 if time_field.endswith("nanos") else 1
                        result["time_to_first_exact_ms"] = (_number(first[time_field]) - _number(candidates[0][time_field])) / divisor
                        result["time_basis"] = "first_evaluable_subject_attempt_start"
            if candidates and all(row.get("capture_source") for row in candidates):
                def az(row):
                    return str(row.get("capture_source", "")).lower() in {"auto_zoom", "az"}
                first_az = next((index for index, row in enumerate(candidates) if az(row)), len(candidates))
                result["baseline_exact_before_az"] = any(index < first_az for index, _, _ in exact_records)
                result["exact_only_after_az"] = bool(exact_records) and all(az(row) for _, row, _ in exact_records)
        subjects.append(result)
    for name in ("reviewed_subjects", "not_reviewed_subjects", "evaluable_subjects", "subjects_with_exact_read",
                 "subjects_without_exact_read", "evaluable_reads", "exact_reads", "incorrect_reads", "no_reads",
                 "gt_characters", "correct_characters", "incorrect_characters", "missing_characters", "extra_characters"):
        summary.setdefault(name, 0)
    summary = dict(summary)
    summary.update(subject_success_rate=_rate(summary["subjects_with_exact_read"], summary["evaluable_subjects"]),
                   exact_read_rate=_rate(summary["exact_reads"], summary["evaluable_reads"]),
                   no_read_rate=_rate(summary["no_reads"], summary["evaluable_reads"]),
                   cer=_rate(sum(summary[name] for name in ("incorrect_characters", "missing_characters", "extra_characters")), summary["gt_characters"]),
                   mt_available=session.attempts_available, review_mode=review.review_mode)
    summary.update(mt_summary)
    summary["mt_invocation_count"] = sum(group.executed for group in session.mt_invocations.values()) if session.attempts_available else None
    times = [row["time_to_first_exact_ms"] for row in subjects if row["time_to_first_exact_ms"] is not None]
    attempts = [row["attempts_to_first_exact"] for row in subjects if row["attempts_to_first_exact"] is not None]
    summary.update(median_time_to_first_exact_ms=_percentile(times, .5), p90_time_to_first_exact_ms=_percentile(times, .9),
                   median_attempts_to_first_exact=_percentile(attempts, .5))
    return {"summary": summary, "subjects": subjects, "reads": reads, "mt_attempts": mt_rows, "mt_invocations": invocations,
            "character_confusion": [{"ground_truth": gt, "prediction": pred, "count": count}
                                    for (gt, pred), count in sorted(confusion.items(), key=lambda item: (-item[1], item[0]))]}
