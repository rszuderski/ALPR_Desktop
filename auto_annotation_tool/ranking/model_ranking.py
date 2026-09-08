#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ranking modeli.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from ..config import CONFIG, logger

PLATE_POSE_TASK_LABEL = "Tablice (Pose)"


def is_plate_pose_model_path(path_like) -> bool:
    """Rozpoznaje kandydatów modelu tablic bez opierania się wyłącznie na nazwie ``pose``."""
    raw = str(path_like or "").strip()
    if not raw:
        return False
    try:
        path = Path(raw)
    except Exception:
        path = Path(str(raw))

    name = path.name.lower()
    parts = {str(part).lower() for part in path.parts}
    if name.startswith("char_") or "chars" in parts or "characters" in parts:
        return False
    return (
        name.startswith("plate_")
        or "pose" in name
        or "plate" in parts
        or "plates" in parts
        or "tablic" in name
    )


def format_ranking_model_label(model_name: str, model_path: str = "", task_type: str = "") -> str:
    """Zwraca domenową etykietę modelu, żeby UI nie sugerował błędnie toru ``char``."""
    raw_name = str(model_name or "").strip()
    try:
        path_name = Path(str(model_path or "")).name
    except Exception:
        path_name = ""
    name = path_name or raw_name or "model.pt"
    task = str(task_type or "").strip().lower()
    is_plate_task = "tablic" in task or "plate" in task or "pose" in task
    if not is_plate_task and not is_plate_pose_model_path(model_path or name):
        return raw_name or name

    stem = Path(name).stem
    lower = stem.lower()
    if lower.startswith("yolo") and "pose" in lower:
        return f"YOLO Pose · {name}"
    if lower.startswith("best_pose"):
        return f"Model tablic Pose · {name}"

    date_matches = re.findall(r"20\d{6}_\d{6}", stem)
    map_match = re.search(r"map(\d{2,3})\b", lower)
    details: list[str] = []
    if date_matches:
        details.append(date_matches[-1])
    if map_match:
        try:
            details.append(f"mAP {int(map_match.group(1)) / 100:.2f}")
        except Exception:
            details.append(f"mAP {map_match.group(1)}")
    if not details:
        details.append(name)
    return "Model tablic Pose · " + " · ".join(details)


def _percent_metric(value) -> float:
    try:
        numeric = float(value or 0.0)
    except Exception:
        return 0.0
    if 0.0 <= numeric <= 1.0:
        return numeric * 100.0
    return numeric


@dataclass
class ModelRankingEntry:
    """Wpis rankingu."""
    model_name: str
    model_path: str
    date_evaluated: str
    reference_name: str = ""
    reference_path: str = ""
    
    task_type: str = "Tablice (Pose)" 
    
    total_images: int = 0
    total_auto_plates: int = 0
    total_corrected_plates: int = 0
    
    plates_unchanged: int = 0
    plates_minor_fix: int = 0
    plates_major_fix: int = 0
    plates_added: int = 0
    plates_removed: int = 0
    
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    split_name: str = ""
    metrics_source: str = ""
    
    @property
    def f1_score(self) -> float:
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def ranking_score(self) -> float:
        if str(self.metrics_source or "").strip().lower() == "yolo val":
            return _percent_metric(self.map50_95)
        if _percent_metric(self.map50_95) > 0:
            return _percent_metric(self.map50_95)
        return self.f1_score
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["f1_score"] = round(self.f1_score, 2)
        d["ranking_score"] = round(self.ranking_score, 2)
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelRankingEntry':
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**data)


class ModelRanking:
    """Zarządza rankingiem modeli."""
    
    RANKING_FILE = "model_ranking.json"
    
    def __init__(self, ranking_dir: Path = None):
        self.ranking_dir = Path(ranking_dir) if ranking_dir else Path(CONFIG.DEFAULT_RANKING_DIR)
        self.ranking_file = self.ranking_dir / self.RANKING_FILE
        self.entries: List[ModelRankingEntry] = []

        self._load()

    @staticmethod
    def _path_key(path_like) -> str:
        raw = str(path_like or "").strip()
        if not raw:
            return ""
        try:
            return str(Path(raw).resolve()).lower()
        except Exception:
            return str(Path(raw)).lower()

    @classmethod
    def _entry_identity(cls, entry: ModelRankingEntry) -> tuple[str, str, str, str]:
        model_key = cls._path_key(getattr(entry, "model_path", ""))
        if not model_key:
            model_key = str(getattr(entry, "model_name", "") or "").strip().lower()
        reference_key = cls._path_key(getattr(entry, "reference_path", ""))
        if not reference_key:
            reference_key = str(getattr(entry, "reference_name", "") or "").strip().lower()
        return (
            model_key,
            str(getattr(entry, "task_type", "") or "").strip().lower(),
            reference_key,
            str(getattr(entry, "split_name", "") or "").strip().lower(),
        )

    @staticmethod
    def _entry_time_key(entry: ModelRankingEntry) -> str:
        return str(getattr(entry, "date_evaluated", "") or "").strip()

    @classmethod
    def _unique_entries(cls, entries: List[ModelRankingEntry]) -> List[ModelRankingEntry]:
        by_identity: Dict[tuple[str, str, str, str], ModelRankingEntry] = {}
        fallback: List[ModelRankingEntry] = []
        for entry in list(entries or []):
            key = cls._entry_identity(entry)
            if not key[0]:
                fallback.append(entry)
                continue
            current = by_identity.get(key)
            if current is None or cls._entry_time_key(entry) >= cls._entry_time_key(current):
                by_identity[key] = entry
        return [*by_identity.values(), *fallback]

    def _dedupe_entries(self) -> None:
        self.entries = self._unique_entries(self.entries)
    
    def _load(self):
        if self.ranking_file.exists():
            try:
                with open(self.ranking_file, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                
                self.entries = [
                    ModelRankingEntry.from_dict(e) 
                    for e in data.get("entries", [])
                ]
                
                self._dedupe_entries()
                self._sort()
                logger.info(f"Załadowano ranking: {len(self.entries)} modeli")
                
            except Exception as e:
                logger.error(f"Błąd: {e}")
                self.entries = []
    
    def _save(self):
        self.ranking_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "entries": [e.to_dict() for e in self.entries]
        }
        
        with open(self.ranking_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _sort(self):
        self.entries.sort(key=lambda e: e.ranking_score, reverse=True)
    
    def add_entry(self, 
                  model_name: str,
                  model_path: str,
                  comparison_stats: Dict,
                  task_type: str = "Tablice (Pose)",
                  reference_name: str = "",
                  reference_path: str = "",
                  save: bool = True) -> ModelRankingEntry:
        """Dodaje wpis do bazy, obsługując kategorie zadań."""
        entry = ModelRankingEntry(
            model_name=model_name,
            model_path=model_path,
            date_evaluated=datetime.now().isoformat(),
            reference_name=str(reference_name or "").strip(),
            reference_path=str(reference_path or "").strip(),
            task_type=task_type,  # ZAPISUJE ZADANIE
            total_images=comparison_stats.get("total_images", 0),
            total_auto_plates=comparison_stats.get("total_auto_plates", 0),
            total_corrected_plates=comparison_stats.get("total_corrected_plates", 0),
            plates_unchanged=comparison_stats.get("plates_unchanged", 0),
            plates_minor_fix=comparison_stats.get("plates_minor_fix", 0),
            plates_major_fix=comparison_stats.get("plates_major_fix", 0),
            plates_added=comparison_stats.get("plates_added", 0),
            plates_removed=comparison_stats.get("plates_removed", 0),
            accuracy=comparison_stats.get("accuracy", 0),
            precision=comparison_stats.get("precision", 0),
            recall=comparison_stats.get("recall", 0),
            map50=_percent_metric(comparison_stats.get("map50", 0)),
            map50_95=_percent_metric(comparison_stats.get("map50_95", 0)),
            split_name=str(comparison_stats.get("split_name", "") or "").strip(),
            metrics_source=str(comparison_stats.get("metrics_source", "") or "").strip(),
        )
        
        new_identity = self._entry_identity(entry)
        if new_identity[0]:
            self.entries = [
                current for current in self.entries
                if self._entry_identity(current) != new_identity
            ]
        self.entries.append(entry)
        self._sort()
        if save:
            self._save()
        
        return entry

    def flush(self):
        self._save()
    
    def get_ranking(self) -> List[ModelRankingEntry]:
        return self.entries

    def get_unique_entries(self) -> List[ModelRankingEntry]:
        return self._unique_entries(self.entries)
    
    def get_best_model(self) -> Optional[ModelRankingEntry]:
        if not self.entries:
            return None
        return self.entries[0]
    
    def delete_entry(self, index: int):
        if 0 <= index < len(self.entries):
            del self.entries[index]
            self._save()
    
    def generate_report(self) -> str:
        report = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                              RANKING MODELI                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
"""
        
        if not self.entries:
            report += "║  Brak danych                                                                ║\n"
        else:
            report += "║  #   Model                Kategoria             Precyzja  Czułość   F1 Score     ║\n"
            report += "║  ──────────────────────────────────────────────────────────────────────────────  ║\n"
            
            for i, e in enumerate(self.entries[:10], 1):
                name = e.model_name[:20].ljust(20)
                cat = e.task_type[:18].ljust(18)
                report += f"║  {i:2d}. {name} {cat} {e.precision:6.1f}%   {e.recall:6.1f}%   {e.f1_score:6.1f}%    ║\n"
        
        report += "╚══════════════════════════════════════════════════════════════════════════════╝\n"
        
        return report
