#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Historia treningĂłw.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from ..config import CONFIG, logger
from ..utils import safe_load_yaml


class TrainingStatus(Enum):
    """Status treningu."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingRun:
    """Pojedynczy przebieg treningu."""
    id: str
    name: str
    created_at: str
    status: str = "pending"
    
    # Konfiguracja
    dataset_path: str = ""
    base_model: str = "yolo11s-pose.pt"
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    device: str = "auto"
    lr0: float = 0.01
    
    # PostÄ™p
    current_epoch: int = 0
    best_map50: float = 0.0
    best_map50_95: float = 0.0
    
    # ĹšcieĹĽki
    output_dir: str = ""
    best_weights: str = ""
    last_weights: str = ""
    
    # Czasy
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    paused_at: Optional[str] = None
    
    # Metryki
    metrics_history: List[Dict] = field(default_factory=list)
    
    # BĹ‚Ä™dy
    error_message: str = ""

    # Raporty
    report_html: str = ""
    plots_dir: str = ""
    resource_report: str = ""
    resource_summary: str = ""

    # Rodowód runu
    lineage_mode: str = "new"
    parent_run_id: str = ""
    parent_model_path: str = ""
    parent_model_name: str = ""
    parent_model_target: str = ""
    parent_dataset_path: str = ""
    parent_best_map50: float = 0.0
    parent_best_map50_95: float = 0.0
    training_target: str = ""

    # Zamrozone dane provenance. Pola sa opcjonalne, zeby starsze historie
    # treningow pozostaly czytelne bez migracji destrukcyjnej.
    training_dataset_snapshot: Dict = field(default_factory=dict)
    training_dataset_input_snapshot: Dict = field(default_factory=dict)
    dataset_preparation: Dict = field(default_factory=dict)
    input_checkpoint_snapshot: Dict = field(default_factory=dict)
    output_checkpoint_snapshot: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingRun':
        # Filtruj tylko znane pola
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
    
    @property
    def duration_str(self) -> str:
        """Czas trwania."""
        if not self.started_at:
            return "-"
        
        start = datetime.fromisoformat(self.started_at)
        
        if self.finished_at:
            end = datetime.fromisoformat(self.finished_at)
        elif self.paused_at:
            end = datetime.fromisoformat(self.paused_at)
        else:
            end = datetime.now()
        
        delta = end - start
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    @property
    def progress_percent(self) -> float:
        if self.epochs == 0:
            return 0.0
        return (self.current_epoch / self.epochs) * 100


class TrainingHistory:
    """ZarzÄ…dza historiÄ… treningĂłw."""
    
    HISTORY_FILE = "training_history.json"
    TARGET_DIRS = {
        "chars": "char",
        "plates": "plate",
        "vehicles": "vehicle",
    }
    
    def __init__(self, history_dir: Path = None):
        self.history_dir = Path(history_dir) if history_dir else Path(CONFIG.DEFAULT_TRAINING_DIR)
        self.history_file = self.history_dir / self.HISTORY_FILE
        self.runs: Dict[str, TrainingRun] = {}

        self._load()

    def _get_target_scope(self) -> str:
        return self.TARGET_DIRS.get(self.history_dir.name.lower(), "")

    def _load_runs_from_file(self, history_file: Path) -> Dict[str, TrainingRun]:
        runs: Dict[str, TrainingRun] = {}
        if not history_file.exists():
            return runs

        with open(history_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        for run_id, run_data in data.get("runs", {}).items():
            runs[run_id] = TrainingRun.from_dict(run_data)

        return runs

    @staticmethod
    def _parse_iso_datetime(value: str | None) -> Optional[datetime]:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except Exception:
            return None

    def _get_run_activity_timestamp(self, run: TrainingRun) -> Optional[datetime]:
        candidates: List[datetime] = []
        for raw in (
            getattr(run, "paused_at", None),
            getattr(run, "finished_at", None),
            getattr(run, "started_at", None),
            getattr(run, "created_at", None),
        ):
            parsed = self._parse_iso_datetime(raw)
            if parsed is not None:
                candidates.append(parsed)

        try:
            output_dir = Path(str(getattr(run, "output_dir", "") or "").strip())
            if output_dir.exists():
                candidates.append(datetime.fromtimestamp(output_dir.stat().st_mtime))
        except Exception:
            pass

        return max(candidates) if candidates else None

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            text = str(value if value is not None else "").strip()
            if not text:
                return default
            return float(text)
        except Exception:
            return default

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(float(str(value if value is not None else "").strip()))
        except Exception:
            return default

    def _read_results_metrics(self, train_dir: Path) -> List[Dict]:
        results_path = train_dir / "results.csv"
        if not results_path.exists():
            return []

        rows: List[Dict] = []
        try:
            with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for idx, raw_row in enumerate(reader, start=1):
                    row = {str(k or "").strip(): v for k, v in dict(raw_row or {}).items()}
                    epoch = self._safe_int(row.get("epoch") or row.get("Epoch"), idx)
                    loss = sum(
                        self._safe_float(row.get(key), 0.0)
                        for key in (
                            "train/box_loss",
                            "train/pose_loss",
                            "train/kobj_loss",
                            "train/cls_loss",
                            "train/dfl_loss",
                            "train/rle_loss",
                        )
                    )
                    rows.append(
                        {
                            "epoch": epoch,
                            "timestamp": "",
                            "loss": loss,
                            "map50": self._safe_float(row.get("metrics/mAP50(B)") or row.get("metrics/mAP50")),
                            "map50_95": self._safe_float(row.get("metrics/mAP50-95(B)") or row.get("metrics/mAP50-95")),
                            "precision": self._safe_float(row.get("metrics/precision(B)") or row.get("metrics/precision")),
                            "recall": self._safe_float(row.get("metrics/recall(B)") or row.get("metrics/recall")),
                            "box_map50": self._safe_float(row.get("metrics/mAP50(B)") or row.get("metrics/mAP50")),
                            "box_map50_95": self._safe_float(row.get("metrics/mAP50-95(B)") or row.get("metrics/mAP50-95")),
                            "box_precision": self._safe_float(row.get("metrics/precision(B)") or row.get("metrics/precision")),
                            "box_recall": self._safe_float(row.get("metrics/recall(B)") or row.get("metrics/recall")),
                            "pose_map50": self._safe_float(row.get("metrics/mAP50(P)") or row.get("metrics/mAP50")),
                            "pose_map50_95": self._safe_float(row.get("metrics/mAP50-95(P)") or row.get("metrics/mAP50-95")),
                            "pose_precision": self._safe_float(row.get("metrics/precision(P)") or row.get("metrics/precision")),
                            "pose_recall": self._safe_float(row.get("metrics/recall(P)") or row.get("metrics/recall")),
                        }
                    )
        except Exception as e:
            logger.debug(f"Nie udalo sie odczytac results.csv runu treningu: {e}")
            return []
        return rows

    def _apply_results_metrics_to_run(self, run: TrainingRun, train_dir: Path) -> bool:
        metrics = self._read_results_metrics(train_dir)
        if not metrics:
            return False

        changed = False
        last_epoch = max(self._safe_int(item.get("epoch"), 0) for item in metrics)
        if last_epoch and int(getattr(run, "current_epoch", 0) or 0) != last_epoch:
            run.current_epoch = last_epoch
            changed = True

        best_map50 = max((self._safe_float(item.get("map50"), 0.0) for item in metrics), default=0.0)
        best_map50_95 = max((self._safe_float(item.get("map50_95"), 0.0) for item in metrics), default=0.0)
        if best_map50 and float(getattr(run, "best_map50", 0.0) or 0.0) != best_map50:
            run.best_map50 = best_map50
            changed = True
        if best_map50_95 and float(getattr(run, "best_map50_95", 0.0) or 0.0) != best_map50_95:
            run.best_map50_95 = best_map50_95
            changed = True

        if not getattr(run, "metrics_history", None):
            run.metrics_history = metrics
            changed = True

        return changed

    def _read_worker_training_end_success(self, run: TrainingRun) -> Optional[bool]:
        output_dir = str(getattr(run, "output_dir", "") or "").strip()
        if not output_dir:
            return None
        events_path = Path(output_dir) / "_ipc" / "events.jsonl"
        if not events_path.exists():
            return None

        result: Optional[bool] = None
        try:
            with events_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = str(line or "").strip()
                    if not raw:
                        continue
                    try:
                        event = json.loads(raw)
                    except Exception:
                        continue
                    if str((event or {}).get("type") or "").strip().lower() == "training_end":
                        result = bool(event.get("success", False))
        except Exception:
            return None
        return result

    def _reconcile_stale_running_runs(self) -> bool:
        changed = False
        now = datetime.now()

        for run in self.runs.values():
            status_value = str(getattr(run, "status", "") or "").strip().lower()
            if status_value not in {TrainingStatus.RUNNING.value, TrainingStatus.PENDING.value}:
                continue

            terminal_success = self._read_worker_training_end_success(run)
            last_activity = self._get_run_activity_timestamp(run)
            if terminal_success is None and last_activity is not None:
                age_seconds = max(0.0, (now - last_activity).total_seconds())
                if age_seconds < 180.0:
                    continue

            train_dir = Path(str(getattr(run, "output_dir", "") or "").strip()) / "train"
            weights_dir = train_dir / "weights"
            best_weights = weights_dir / "best.pt"
            last_weights = weights_dir / "last.pt"

            if self._apply_results_metrics_to_run(run, train_dir):
                changed = True

            completed_on_disk = bool(
                terminal_success is True
                or (
                    best_weights.exists()
                    and last_weights.exists()
                    and int(getattr(run, "epochs", 0) or 0) > 0
                    and int(getattr(run, "current_epoch", 0) or 0) >= int(getattr(run, "epochs", 0) or 0)
                )
            )

            run.status = TrainingStatus.COMPLETED.value if completed_on_disk else TrainingStatus.FAILED.value
            run.finished_at = (last_activity or now).isoformat()
            run.best_weights = str(best_weights) if completed_on_disk and best_weights.exists() else ""
            run.last_weights = str(last_weights) if last_weights.exists() else str(getattr(run, "last_weights", "") or "")
            report_html = Path(str(getattr(run, "output_dir", "") or "").strip()) / "training_report.html"
            plots_dir = Path(str(getattr(run, "output_dir", "") or "").strip()) / "plots"
            if report_html.exists():
                run.report_html = str(report_html)
            if plots_dir.exists():
                run.plots_dir = str(plots_dir)

            if completed_on_disk:
                run.error_message = ""
            elif not str(getattr(run, "error_message", "") or "").strip():
                if last_weights.exists():
                    run.error_message = (
                        "Osierocony wpis historii: aktywny trening juz nie istnieje. "
                        "Zachowano checkpoint do wznowienia."
                    )
                else:
                    run.error_message = (
                        "Osierocony wpis historii: aktywny trening juz nie istnieje "
                        "i nie pozostawil checkpointu."
                    )

            changed = True
            logger.warning(
                "Domknieto osierocony run treningu: "
                f"{run.id} | status={run.status} | output={getattr(run, 'output_dir', '')}"
            )

        return changed

    def _reconcile_checkpoint_paths(self) -> bool:
        """Recover checkpoint paths that exist on disk but are missing in history.

        A failed/paused run may still have ``last.pt`` and can be resumed, but it
        must not expose ``best.pt`` as a finished project model.
        """
        changed = False

        for run in self.runs.values():
            status_value = str(getattr(run, "status", "") or "").strip().lower()
            if status_value not in {
                TrainingStatus.COMPLETED.value,
                TrainingStatus.FAILED.value,
                TrainingStatus.PAUSED.value,
                TrainingStatus.CANCELLED.value,
            }:
                continue

            output_dir = str(getattr(run, "output_dir", "") or "").strip()
            if not output_dir:
                continue
            weights_dir = Path(output_dir) / "train" / "weights"
            best_weights = weights_dir / "best.pt"
            last_weights = weights_dir / "last.pt"

            if status_value == TrainingStatus.COMPLETED.value:
                if best_weights.exists() and str(getattr(run, "best_weights", "") or "").strip() != str(best_weights):
                    run.best_weights = str(best_weights)
                    changed = True
                if last_weights.exists() and str(getattr(run, "last_weights", "") or "").strip() != str(last_weights):
                    run.last_weights = str(last_weights)
                    changed = True
                continue

            if str(getattr(run, "best_weights", "") or "").strip():
                run.best_weights = ""
                changed = True
            if last_weights.exists() and str(getattr(run, "last_weights", "") or "").strip() != str(last_weights):
                run.last_weights = str(last_weights)
                changed = True

        return changed

    def _get_legacy_history_files(self) -> List[Path]:
        target_scope = self._get_target_scope()
        if not target_scope:
            return []

        parent_history = self.history_dir.parent / self.HISTORY_FILE
        if parent_history == self.history_file or not parent_history.exists():
            return []

        return [parent_history]

    @staticmethod
    def _infer_target_from_text(text: str) -> str:
        text = str(text or "").strip().lower()
        if not text:
            return ""

        if "pose" in text or any(token in text for token in ("plate", "plates", "tablica", "tablic", "rejestr")):
            return "plate"
        if any(token in text for token in ("char", "chars", "character", "characters", "ocr", "znak", "znaki")):
            return "char"
        if any(token in text for token in ("vehicle", "vehicles", "pojazd", "pojazdy", "samochod", "car", "cars")):
            return "vehicle"
        return ""

    def _infer_target_from_args_file(self, output_dir: str) -> str:
        args_path = Path(output_dir) / "train" / "args.yaml"
        if not args_path.exists():
            return ""

        try:
            args_cfg = safe_load_yaml(args_path)
        except Exception:
            return ""

        task = str(args_cfg.get("task", "") or "").strip().lower()
        model = str(args_cfg.get("model", "") or "")
        data = str(args_cfg.get("data", "") or "")
        project = str(args_cfg.get("project", "") or "")
        merged = " ".join((task, model, data, project))

        if task == "pose":
            return "plate"

        inferred = self._infer_target_from_text(merged)
        if inferred:
            return inferred

        if task == "detect":
            return "char"

        return ""

    def _infer_run_target(self, run: TrainingRun) -> str:
        explicit = str(getattr(run, "training_target", "") or "").strip().lower()
        if explicit in {"plate", "char", "vehicle"}:
            return explicit
        inferred = self._infer_target_from_args_file(getattr(run, "output_dir", ""))
        if inferred:
            return inferred

        merged = " ".join(
            str(value or "")
            for value in (
                getattr(run, "dataset_path", ""),
                getattr(run, "base_model", ""),
                getattr(run, "name", ""),
                getattr(run, "output_dir", ""),
            )
        )
        return self._infer_target_from_text(merged)

    def _matches_current_scope(self, run: TrainingRun) -> bool:
        target_scope = self._get_target_scope()
        if not target_scope:
            return True

        run_target = self._infer_run_target(run)
        return bool(run_target == target_scope)
    
    def _load(self):
        """Laduje historie."""
        try:
            self.runs = self._load_runs_from_file(self.history_file)

            imported_legacy = 0
            if not self.runs:
                for legacy_file in self._get_legacy_history_files():
                    for run_id, run in self._load_runs_from_file(legacy_file).items():
                        if run_id in self.runs or not self._matches_current_scope(run):
                            continue
                        self.runs[run_id] = run
                        imported_legacy += 1

            if imported_legacy:
                logger.info(
                    f"Zaimportowano {imported_legacy} treningow ze starszej struktury do: {self.history_dir}"
                )
                self._save()

            changed = False
            if self._reconcile_stale_running_runs():
                changed = True
            if self._reconcile_checkpoint_paths():
                changed = True
            if changed:
                self._save()

            logger.info(f"Zaladowano {len(self.runs)} treningow z: {self.history_dir}")

        except Exception as e:
            logger.error(f"Blad ladowania historii: {e}")
            self.runs = {}

    def _save(self):
        """Zapisuje historiÄ™."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "runs": {rid: run.to_dict() for rid, run in self.runs.items()}
            }
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"BĹ‚Ä…d zapisywania: {e}")
    
    def create_run(self, name: str, **kwargs) -> TrainingRun:
        """Tworzy nowy trening."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir = self.history_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        run = TrainingRun(
            id=run_id,
            name=name,
            created_at=datetime.now().isoformat(),
            output_dir=str(output_dir),
            **kwargs
        )
        
        self.runs[run_id] = run
        self._save()
        
        logger.info(f"Utworzono trening: {run_id}")
        return run
    
    def update_run(self, run_id: str, **kwargs):
        """Aktualizuje trening."""
        if run_id not in self.runs:
            return
        
        run = self.runs[run_id]
        for key, value in kwargs.items():
            if hasattr(run, key):
                setattr(run, key, value)
        
        self._save()
    
    def add_metrics(self, run_id: str, epoch: int, metrics: Dict):
        """Dodaje metryki."""
        if run_id not in self.runs:
            return
        
        run = self.runs[run_id]
        run.metrics_history.append({
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        })
        run.current_epoch = epoch
        
        if metrics.get("map50", 0) > run.best_map50:
            run.best_map50 = metrics["map50"]
        if metrics.get("map50_95", 0) > run.best_map50_95:
            run.best_map50_95 = metrics["map50_95"]
        
        self._save()
    
    def get_run(self, run_id: str) -> Optional[TrainingRun]:
        return self.runs.get(run_id)
    
    def get_all_runs(self) -> List[TrainingRun]:
        return sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)

    @staticmethod
    def _has_nonfinite_checkpoint_error(run: TrainingRun) -> bool:
        text = str(getattr(run, "error_message", "") or "").strip().lower()
        if not text:
            return False
        needles = (
            "nan",
            "nan/inf",
            "inf weights",
            "non-finite",
            "non finite",
            "not finite",
        )
        return any(needle in text for needle in needles)
    
    def get_resumable_runs(self) -> List[TrainingRun]:
        return [
            run for run in self.runs.values()
            if run.status in [TrainingStatus.PAUSED.value, TrainingStatus.FAILED.value]
            and run.last_weights and Path(run.last_weights).exists()
            and not self._has_nonfinite_checkpoint_error(run)
        ]
    
    def delete_run(self, run_id: str, delete_files: bool = False):
        """Usuwa trening."""
        if run_id not in self.runs:
            return
        
        run = self.runs[run_id]
        
        if delete_files and run.output_dir:
            import shutil
            output_path = Path(run.output_dir)
            if output_path.exists():
                shutil.rmtree(output_path)
        
        del self.runs[run_id]
        self._save()
