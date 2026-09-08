#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Planer ingestii dla iteracji kampanii.

MVP:
- czyta ground truth z nazw plików źródłowych,
- liczy balans znaków z zaakceptowanej wiedzy projektu,
- proponuje kolejną porcję danych z master pool,
- zapisuje wynik w postaci prostych, serializowalnych struktur.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from .config import CONFIG, logger


CHAR_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_SET = set(CHAR_ALPHABET)
GROUND_TRUTH_PATTERN = re.compile(r"[A-Z0-9]{4,}")


@dataclass
class IngestCandidate:
    name: str
    source_path: str
    source_key: str
    ground_truth_texts: list[str]
    char_histogram: dict[str, int]
    score: float = 0.0
    score_details: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_path": self.source_path,
            "source_key": self.source_key,
            "ground_truth_texts": list(self.ground_truth_texts),
            "char_histogram": dict(self.char_histogram),
            "score": float(self.score),
            "score_details": dict(self.score_details or {}),
        }


class CampaignIngestPlanner:
    """
    Planuje kolejne paczki ingestii dla iteracji projektu.

    Strategia MVP:
    - obecny balans znaków pochodzi z zaakceptowanego materiału treningowego,
    - kandydaci z master pool są oceniani względem braków w tym balansie,
    - wybór odbywa się zachłannie, aby każda kolejna próbka aktualizowała
      roboczy histogram i uzupełniała kolejne luki.
    """

    planner_version = "balance_v1"

    def __init__(self, image_extensions: Iterable[str] | None = None):
        self.image_extensions = {
            str(ext).lower()
            for ext in (image_extensions or CONFIG.IMAGE_EXTENSIONS)
        }

    # =========================================================
    # Ground truth z nazw plików
    # =========================================================

    def extract_true_texts_from_filename(self, filename: str) -> list[str]:
        stem = Path(str(filename or "")).stem.upper()
        return GROUND_TRUTH_PATTERN.findall(stem)

    def build_char_histogram(self, texts: Iterable[str]) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for text in texts or []:
            for ch in str(text).upper():
                if ch in CHAR_SET:
                    counter[ch] += 1
        return {
            ch: int(counter[ch])
            for ch in CHAR_ALPHABET
            if counter.get(ch, 0) > 0
        }

    def make_source_key(self, image_path: Path, master_pool_dir: Path | None = None) -> str:
        try:
            resolved = image_path.resolve()
        except Exception:
            resolved = image_path.absolute()

        if master_pool_dir is not None:
            try:
                base = master_pool_dir.resolve()
            except Exception:
                base = master_pool_dir.absolute()
            try:
                return resolved.relative_to(base).as_posix().lower()
            except Exception:
                pass

        return resolved.as_posix().lower()

    # =========================================================
    # Statystyki z danych treningowych projektu
    # =========================================================

    def _is_valid_dataset_dir(self, dataset_dir: Path | None) -> bool:
        if dataset_dir is None:
            return False
        try:
            return (
                dataset_dir.exists()
                and dataset_dir.is_dir()
                and (dataset_dir / "data.yaml").exists()
                and (dataset_dir / "images").exists()
                and (dataset_dir / "labels").exists()
            )
        except Exception:
            return False

    def _looks_like_character_dataset(self, dataset_dir: Path | None) -> bool:
        if not self._is_valid_dataset_dir(dataset_dir):
            return False

        data_yaml = dataset_dir / "data.yaml"
        try:
            content = data_yaml.read_text(encoding="utf-8", errors="ignore").upper()
        except Exception:
            return False

        if "NC: 36" in content:
            return True

        quoted_hits = 0
        for ch in CHAR_ALPHABET:
            if f"'{ch}'" in content or f"\"{ch}\"" in content:
                quoted_hits += 1
        return quoted_hits >= 20

    def collect_char_balance_from_dataset(self, dataset_dir: Path) -> dict[str, Any]:
        counter: Counter[str] = Counter()
        label_files = 0

        labels_root = dataset_dir / "labels"
        for label_file in labels_root.rglob("*.txt"):
            if not label_file.is_file():
                continue
            try:
                lines = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue

            label_files += 1
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls_id = int(float(parts[0]))
                except Exception:
                    continue
                if 0 <= cls_id < len(CHAR_ALPHABET):
                    counter[CHAR_ALPHABET[cls_id]] += 1

        return {
            "source_type": "dataset",
            "source_path": str(dataset_dir),
            "label_files": label_files,
            "total_characters": int(sum(counter.values())),
            "char_balance": self._counter_to_full_balance(counter),
        }

    def collect_char_balance_from_metadata_pools(self, pool_dirs: Iterable[Path]) -> dict[str, Any]:
        counter: Counter[str] = Counter()
        seen_plate_keys: set[str] = set()
        used_sources: list[str] = []

        for pool_dir in pool_dirs:
            if pool_dir is None or not pool_dir.exists() or not pool_dir.is_dir():
                continue
            used_sources.append(str(pool_dir))

            for meta_path in pool_dir.rglob("metadata.json"):
                try:
                    with open(meta_path, "r", encoding="utf-8") as handle:
                        metadata = json.load(handle)
                except Exception as exc:
                    logger.debug(f"Nie udało się odczytać {meta_path}: {exc}")
                    continue

                if not isinstance(metadata, dict):
                    continue

                for plate_id, plate_data in metadata.items():
                    if not isinstance(plate_data, dict):
                        continue

                    plate_key = self._build_plate_key(meta_path, plate_id, plate_data)
                    if plate_key in seen_plate_keys:
                        continue
                    seen_plate_keys.add(plate_key)

                    for char_rec in plate_data.get("characters", []) or []:
                        ch = str(char_rec.get("character", "")).upper().strip()
                        if ch in CHAR_SET:
                            counter[ch] += 1

        return {
            "source_type": "metadata_pools",
            "source_path": used_sources,
            "label_files": 0,
            "total_characters": int(sum(counter.values())),
            "char_balance": self._counter_to_full_balance(counter),
        }

    def collect_project_training_balance(self, project_root: Path) -> dict[str, Any]:
        datasets_dir = project_root / "4_training_datasets"
        chars_datasets_dir = datasets_dir / "chars"
        chars_dir = project_root / "3_cropped_characters"

        dataset_roots: list[Path] = []
        for candidate in (chars_datasets_dir, datasets_dir):
            if candidate.exists() and candidate.is_dir() and candidate not in dataset_roots:
                dataset_roots.append(candidate)

        for dataset_root in dataset_roots:
            merged_dir = dataset_root / "char_merged_pool"
            if self._looks_like_character_dataset(merged_dir):
                result = self.collect_char_balance_from_dataset(merged_dir)
                result["source_type"] = "merged_dataset"
                return result

        dataset_candidates: list[Path] = []
        for dataset_root in dataset_roots:
            try:
                dataset_candidates.extend(
                    path
                    for path in dataset_root.iterdir()
                    if path.is_dir() and self._looks_like_character_dataset(path)
                )
            except Exception:
                continue

        if dataset_candidates:
            dataset_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return self.collect_char_balance_from_dataset(dataset_candidates[0])

        pool_dirs = [
            chars_dir / "manual_char_pool",
            chars_dir / "gold_char_pool",
        ]
        meta_result = self.collect_char_balance_from_metadata_pools(pool_dirs)
        if meta_result.get("total_characters", 0) > 0:
            return meta_result

        return {
            "source_type": "empty",
            "source_path": "",
            "label_files": 0,
            "total_characters": 0,
            "char_balance": self._counter_to_full_balance(Counter()),
        }

    # =========================================================
    # Planowanie ingestii
    # =========================================================

    def _select_candidates(self, candidates, current_counter, limit):
        """Update greedy scores in arrays; resolve ties with the original scorer."""
        import numpy as np

        if not candidates or limit <= 0:
            return [], Counter(current_counter)
        histogram = np.zeros((len(candidates), len(CHAR_ALPHABET)), dtype=np.float64, order="F")
        char_index = {ch: index for index, ch in enumerate(CHAR_ALPHABET)}
        signatures = []
        for row, candidate in enumerate(candidates):
            signatures.append(tuple(candidate.char_histogram.items()))
            for ch, count in candidate.char_histogram.items():
                histogram[row, char_index[ch]] = count
        balance = np.array([current_counter.get(ch, 0) for ch in CHAR_ALPHABET], dtype=np.float64)
        weights = 1.0 + balance.max() - balance + 1.0 / (balance + 1.0)
        totals = histogram.sum(axis=1)
        scores = np.zeros(len(candidates), dtype=np.float64)
        for column in range(len(CHAR_ALPHABET)):
            scores += histogram[:, column] * weights[column]
        scores += np.count_nonzero(histogram, axis=1) * 0.05
        selected = []
        working_counter = Counter(current_counter)
        for _ in range(min(limit, len(candidates))):
            best = float(scores.max())
            # Incremental floating-point sums can differ in their last digits.
            # Scalar rescoring preserves the original stable tie/order contract.
            near = np.flatnonzero(np.isclose(scores, best, rtol=1e-10, atol=1e-8))
            exact = {}
            chosen_index, best_score, best_details = -1, -1.0, {}
            for index in near:
                signature = signatures[index]
                if signature not in exact:
                    exact[signature] = self.score_candidate(candidates[index].char_histogram, working_counter)
                score, details = exact[signature]
                if score > best_score:
                    chosen_index, best_score, best_details = index, score, details
            chosen = candidates[chosen_index]
            chosen.score = round(best_score, 6)
            chosen.score_details = best_details
            selected.append(chosen)
            working_counter.update(chosen.char_histogram)
            scores[chosen_index] = -np.inf
            old_max = float(balance.max())
            new_balance = balance + histogram[chosen_index]
            new_max = float(new_balance.max())
            if new_max != old_max:
                scores += totals * (new_max - old_max)
            for ch in chosen.char_histogram:
                column = char_index[ch]
                delta = (balance[column] - new_balance[column]
                         + 1.0 / (new_balance[column] + 1.0) - 1.0 / (balance[column] + 1.0))
                scores += histogram[:, column] * delta
            balance = new_balance
        return selected, working_counter

    def _pool_images(self, root):
        """Use directory entry metadata, avoiding a Windows stat per image."""
        pending = [(Path(root), "")]
        while pending:
            directory, prefix = pending.pop()
            children = []
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if entry.is_dir(follow_symlinks=False):
                            children.append((Path(entry.path), prefix + entry.name + "/"))
                        elif os.path.splitext(entry.name)[1].lower() in self.image_extensions and entry.is_file():
                            path = Path(entry.path)
                            key = self.make_source_key(path, root) if entry.is_symlink() else (prefix + entry.name).lower()
                            yield path, key
            except OSError:
                continue
            pending.extend(reversed(children))

    def score_candidate(
        self,
        candidate_hist: dict[str, int],
        current_balance: Counter[str],
    ) -> tuple[float, dict[str, float]]:
        if not candidate_hist:
            return 0.0, {}

        current_values = [int(current_balance.get(ch, 0)) for ch in CHAR_ALPHABET]
        max_count = max(current_values) if current_values else 0

        score = 0.0
        details: dict[str, float] = {}
        for ch, occurrences in candidate_hist.items():
            curr = int(current_balance.get(ch, 0))
            deficit = max(0, max_count - curr)
            rarity = 1.0 / (curr + 1.0)
            contribution = float(occurrences) * (1.0 + float(deficit) + rarity)
            score += contribution
            details[ch] = round(contribution, 4)

        score += 0.05 * len(candidate_hist)
        return score, details

    def plan_from_master_pool(
        self,
        master_pool_dir: Path,
        current_balance: dict[str, int] | None = None,
        used_source_keys: Iterable[str] | None = None,
        used_filenames: Iterable[str] | None = None,
        batch_size: int = 200,
    ) -> dict[str, Any]:
        master_pool_dir = Path(master_pool_dir)
        if not master_pool_dir.exists() or not master_pool_dir.is_dir():
            raise FileNotFoundError(f"Master pool nie istnieje: {master_pool_dir}")

        used_keys = {str(key).strip().lower() for key in (used_source_keys or []) if str(key).strip()}
        used_names = {str(name).strip().lower() for name in (used_filenames or []) if str(name).strip()}
        try:
            requested_batch = int(batch_size) if batch_size is not None else 0
        except Exception:
            requested_batch = 0
        select_all_remaining = requested_batch <= 0

        current_counter = Counter()
        for ch in CHAR_ALPHABET:
            current_counter[ch] = int((current_balance or {}).get(ch, 0))

        candidates: list[IngestCandidate] = []
        skipped_used = 0
        skipped_invalid_gt = 0

        for image_path, source_key in self._pool_images(master_pool_dir):
            if source_key in used_keys or image_path.name.lower() in used_names:
                skipped_used += 1
                continue

            texts = self.extract_true_texts_from_filename(image_path.name)
            if not texts:
                skipped_invalid_gt += 1
                continue

            char_hist = self.build_char_histogram(texts)
            if not char_hist:
                skipped_invalid_gt += 1
                continue

            candidates.append(
                IngestCandidate(
                    name=image_path.name,
                    source_path=str(image_path),
                    source_key=source_key,
                    ground_truth_texts=texts,
                    char_histogram=char_hist,
                )
            )

        selected, working_counter = self._select_candidates(
            candidates, current_counter, len(candidates) if select_all_remaining else requested_batch,
        )

        selected_hist = Counter()
        for candidate in selected:
            selected_hist.update(candidate.char_histogram)

        return {
            "planner_version": self.planner_version,
            "generated_at": datetime.now().isoformat(),
            "master_pool_dir": str(master_pool_dir),
            "batch_size": len(selected) if select_all_remaining else requested_batch,
            "candidates_total": len(candidates),
            "selected_total": len(selected),
            "skipped_used": skipped_used,
            "skipped_invalid_ground_truth": skipped_invalid_gt,
            "current_balance": self._counter_to_full_balance(current_counter),
            "selected_balance": self._counter_to_full_balance(selected_hist),
            "predicted_balance_after": self._counter_to_full_balance(working_counter),
            "selected": [candidate.to_dict() for candidate in selected],
        }

    # =========================================================
    # Helpers
    # =========================================================

    def _counter_to_full_balance(self, counter: Counter[str]) -> dict[str, int]:
        return {ch: int(counter.get(ch, 0)) for ch in CHAR_ALPHABET}

    def _build_plate_key(self, meta_path: Path, plate_id: str, plate_data: dict[str, Any]) -> str:
        source_image = str(plate_data.get("source_image", "") or "").strip().lower()
        bbox = plate_data.get("source_bbox") or []
        if isinstance(bbox, list) and len(bbox) >= 4:
            rounded = [str(int(float(v) // 5)) for v in bbox[:4]]
            bbox_key = "_".join(rounded)
        else:
            bbox_key = "na"
        if source_image:
            return f"{source_image}|{bbox_key}"
        return f"{meta_path.as_posix().lower()}|{str(plate_id).strip().lower()}"
