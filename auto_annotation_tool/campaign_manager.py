#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Menadżer Kampanii ALPR (Active Learning Wizard) - Wersja Multi-Project.
Zarządza listą projektów, iteracjami i fizycznym czyszczeniem dysku.
"""

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from time import perf_counter
from typing import Dict, Any, List

from .config import CONFIG, logger
from .campaign_project_registry import bind_campaign_project_registry_methods
from .campaign_stage_state import bind_campaign_stage_state_methods
from .campaign_project_history import bind_campaign_project_history_methods
from .project_cache import PROJECT_CACHE
from .image_directory_index import IMAGE_DIRECTORIES


class CampaignManager:
    def __init__(self):
        self.state_file = CONFIG.WORKSPACE_DIR / "campaigns_registry.json"
        self.state = self._load_state()
        self._plate_approved_stats_cache: Dict[tuple[str, int, int], Dict[str, Any]] = {}
        self._plate_approved_manifest_runtime_cache: Dict[str, Dict[str, Any]] = {}
        self._plate_approved_stats_runtime_cache: Dict[str, Dict[str, Any]] = {}
        self._artifact_registry_cache: Dict[tuple[str, int, int], Dict[str, Any]] = {}
        self._artifact_registry_runtime_cache: Dict[str, Dict[str, Any]] = {}
        self._ingest_manifest_cache: Dict[tuple[str, int, int, int], Dict[str, Any]] = {}
        self._latest_ingest_plan_summary_cache: Dict[tuple[str, int, int], Dict[str, Any]] = {}
        self._latest_ingest_plan_summary_runtime_cache: Dict[str, Dict[str, Any]] = {}
        self._iteration_image_count_cache: Dict[tuple[str, int], int] = {}
        self._iteration_image_source_dir_cache: Dict[tuple[str, int], str] = {}
        self._step3_char_source_state_cache: Dict[tuple[str, int, str, str, str, str, str], Dict[str, Any]] = {}
        self._step3_char_source_state_runtime_cache: Dict[tuple[str, int], Dict[str, Any]] = {}

    # Project registry/default methods are bound after class creation.

    # --- GETTERY I SETTERY (dla AKTYWNEGO projektu) ---

    # Campaign step/status state methods are bound after class creation.

    def _clone_iteration_ingest_manifest(
        self,
        source_iteration: int,
        target_iteration: int,
        target_raw_dir: Path,
        project_name: str,
    ) -> bool:
        manifest = self.load_ingest_manifest(source_iteration, project_name)
        if not manifest:
            return False

        selected_images = []
        for item in manifest.get("selected_images", []) or []:
            if not isinstance(item, dict):
                continue
            cloned_item = dict(item)
            item_name = str(cloned_item.get("name", "") or "").strip()
            if not item_name:
                target_path = str(cloned_item.get("target_path", "") or "").strip()
                if target_path:
                    item_name = Path(target_path).name
            if not item_name:
                continue
            actual_image_path = ""
            for key in ("target_path", "source_path"):
                candidate = str(cloned_item.get(key, "") or "").strip()
                if not candidate:
                    continue
                try:
                    candidate_path = Path(candidate)
                except Exception:
                    continue
                if candidate_path.exists() and candidate_path.is_file():
                    actual_image_path = str(candidate_path.resolve())
                    break
            logical_target_path = target_raw_dir / item_name
            if actual_image_path:
                cloned_item["target_path"] = actual_image_path
            cloned_item["iteration_target_path"] = str(logical_target_path.resolve())
            selected_images.append(cloned_item)

        cloned_manifest = dict(manifest)
        cloned_manifest["iteration"] = int(target_iteration)
        cloned_manifest["created_at"] = datetime.now().isoformat()
        cloned_manifest["selection_mode"] = "iteration_reuse"
        cloned_manifest["reused_from_iteration"] = int(source_iteration)
        cloned_manifest["target_dir"] = str(target_raw_dir.resolve())
        cloned_manifest["manifest_only"] = True
        cloned_manifest["selected_images"] = selected_images
        cloned_manifest["selected_count"] = int(
            len(selected_images) or manifest.get("selected_count", 0) or 0
        )
        cloned_manifest["image_set_token"] = self.build_image_name_set_token(
            [str(item.get("name") or "").strip() for item in selected_images]
        )

        return bool(self.save_ingest_manifest(cloned_manifest, target_iteration, project_name))

    def _carry_iteration_input_forward(
        self,
        source_iteration: int,
        target_iteration: int,
        project_name: str,
    ) -> Dict[str, Any]:
        source_raw_dir = self.get_iteration_image_source_dir(source_iteration, project_name)
        target_raw_dir = self.get_iteration_raw_dir(target_iteration, project_name)
        if source_raw_dir is None or target_raw_dir is None:
            return {"ok": False, "reason": "missing_project_dirs"}

        if not source_raw_dir.exists() or not source_raw_dir.is_dir():
            return {"ok": False, "reason": "missing_source_dir"}

        manifest_image_files = []
        try:
            manifest_image_files = list(
                self.get_iteration_manifest_image_paths(source_iteration, project_name) or []
            )
        except Exception:
            manifest_image_files = []
        if manifest_image_files:
            image_files = [
                Path(path)
                for path in manifest_image_files
                if Path(path).exists()
                and Path(path).is_file()
                and Path(path).suffix.lower() in CONFIG.IMAGE_EXTENSIONS
            ]
        else:
            image_files = [
                path for path in source_raw_dir.iterdir()
                if path.is_file() and path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS
            ]
        if not image_files:
            return {"ok": False, "reason": "missing_images"}

        manifest_cloned = self._clone_iteration_ingest_manifest(
            source_iteration=source_iteration,
            target_iteration=target_iteration,
            target_raw_dir=target_raw_dir,
            project_name=project_name,
        )
        if not manifest_cloned:
            selected_images = []
            for source_path in image_files:
                logical_target_path = target_raw_dir / source_path.name
                selected_images.append(
                    {
                        "name": source_path.name,
                        "source_path": str(source_path.resolve()),
                        "target_path": str(source_path.resolve()),
                        "iteration_target_path": str(logical_target_path.resolve()),
                    }
                )
            manifest_cloned = bool(
                self.save_ingest_manifest(
                    {
                        "project": project_name,
                        "iteration": int(target_iteration),
                        "created_at": datetime.now().isoformat(),
                        "selection_mode": "iteration_reuse",
                        "reused_from_iteration": int(source_iteration),
                        "source_dir": str(source_raw_dir.resolve()),
                        "target_dir": str(target_raw_dir.resolve()),
                        "selected_count": int(len(selected_images)),
                        "selected_images": selected_images,
                        "manifest_only": True,
                        "image_set_token": self.build_image_name_set_token(
                            [str(item.get("name") or "").strip() for item in selected_images]
                        ),
                        "proposal_summary": {
                            "source_iteration": int(source_iteration),
                            "source_kind": "iteration_manifest",
                            "current_iteration_package_count": int(len(selected_images)),
                        },
                    },
                    target_iteration,
                    project_name,
                )
            )
        if not manifest_cloned:
            return {
                "ok": False,
                "reason": "manifest_save_failed",
                "source_dir": str(source_raw_dir.resolve()),
                "target_dir": str(target_raw_dir.resolve()),
                "copied_images": 0,
                "manifest_images": 0,
                "manifest_cloned": False,
                "source_kind": "iteration",
            }

        return {
            "ok": True,
            "source_dir": str(source_raw_dir.resolve()),
            "target_dir": str(target_raw_dir.resolve()),
            "copied_images": 0,
            "manifest_images": int(len(image_files)),
            "manifest_cloned": bool(manifest_cloned),
            "source_kind": "iteration",
        }

    def _is_broad_image_source_dir(self, candidate: str | Path | None, project_name: str = None) -> bool:
        if candidate is None:
            return False
        try:
            path = Path(str(candidate or "").strip()).resolve()
        except Exception:
            return False

        broad_roots: list[Path] = []
        for root_candidate in (
            Path.cwd(),
            CONFIG.WORKSPACE_DIR,
            self.get_project_root_dir(project_name) if project_name else None,
        ):
            if root_candidate is None:
                continue
            try:
                broad_roots.append(Path(root_candidate).resolve())
            except Exception:
                pass

        for root in broad_roots:
            if path == root:
                return True

        # Nie pozwalamy, aby katalog aplikacji albo jego rodzic udawal katalog zdjec.
        # W przeciwnym razie rekurencyjne liczenie obrazow podlapuje artefakty z calego Workspace.
        try:
            workspace_root = Path(CONFIG.WORKSPACE_DIR).resolve()
            return workspace_root.is_relative_to(path) and workspace_root != path
        except Exception:
            try:
                return path in workspace_root.parents
            except Exception:
                return False

    def _resolve_valid_image_source_dir(
        self,
        candidate: str | Path | None,
        *,
        project_name: str = None,
        recursive: bool = True,
        reject_broad: bool = True,
    ) -> str:
        if candidate is None:
            return ""
        try:
            path = Path(str(candidate or "").strip())
        except Exception:
            return ""
        if not path.exists() or not path.is_dir():
            return ""
        if reject_broad and self._is_broad_image_source_dir(path, project_name):
            return ""
        try:
            iterator = path.rglob("*") if recursive else path.iterdir()
            for image_path in iterator:
                if image_path.is_file() and image_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS:
                    return str(path.resolve())
        except Exception:
            return ""
        return ""

    def _resolve_previous_iteration_image_source_for_e1(
        self,
        *,
        project_name: str,
        current_iteration: int,
        project_data: Dict[str, Any],
    ) -> str:
        stage_state = self.get_manual_plate_stage_images_state(
            source_iteration=current_iteration,
            project_name=project_name,
        )
        if int(stage_state.get("image_count", 0) or 0) > 0:
            stage_images_dir = str(stage_state.get("images_dir") or "").strip()
            if stage_images_dir:
                return stage_images_dir

        # E4->E1 nie moze utozsamiac pustego stage z pusta pula projektu.
        # Po torze znakow najczesciej chcemy odtworzyc katalog zdjec z E1,
        # mimo ze stage tablic jest pusty.
        master_pool = self._resolve_valid_image_source_dir(
            project_data.get("master_pool_dir", ""),
            project_name=project_name,
        )
        if master_pool:
            return master_pool

        try:
            manifest = self.load_ingest_manifest(current_iteration, project_name)
        except Exception:
            manifest = {}
        if isinstance(manifest, dict):
            for key in ("master_pool_dir", "source_dir", "target_dir"):
                source = self._resolve_valid_image_source_dir(
                    manifest.get(key, ""),
                    project_name=project_name,
                )
                if source:
                    return source

        raw_source = self._resolve_valid_image_source_dir(
            self.get_iteration_raw_dir(current_iteration, project_name),
            project_name=project_name,
            recursive=False,
            reject_broad=False,
        )
        if raw_source:
            return raw_source
        return ""

    def _resolve_current_iteration_image_source_for_e1(
        self,
        *,
        project_name: str,
        current_iteration: int,
    ) -> str:
        candidates: list[Any] = []

        try:
            latest_plan = self.load_latest_ingest_plan(project_name)
        except Exception:
            latest_plan = {}
        if isinstance(latest_plan, dict):
            try:
                plan_iteration = int(latest_plan.get("iteration", 0) or 0)
            except Exception:
                plan_iteration = 0
            plan_project = str(latest_plan.get("project", "") or "").strip()
            if plan_iteration == int(current_iteration or 0) and (
                not plan_project or plan_project == str(project_name or "").strip()
            ):
                for key in ("master_pool_dir", "source_dir", "target_dir"):
                    candidates.append(latest_plan.get(key, ""))

        try:
            manifest = self.load_ingest_manifest(current_iteration, project_name)
        except Exception:
            manifest = {}
        if isinstance(manifest, dict):
            for key in ("master_pool_dir", "source_dir", "target_dir"):
                candidates.append(manifest.get(key, ""))

        for candidate in candidates:
            valid_source = self._resolve_valid_image_source_dir(
                candidate,
                project_name=project_name,
            )
            if valid_source:
                return valid_source
        return ""

    def get_manual_plate_stage_images_state(
        self,
        source_iteration: int = None,
        project_name: str = None,
    ) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {"ok": False, "image_count": 0, "images_dir": "", "stage_dir": ""}

        try:
            iter_value = int(source_iteration or self.state["projects"][project_name].get("current_iteration", 1) or 1)
        except Exception:
            iter_value = 1

        stage_root = self.get_staging_dir("plate_stage")
        if stage_root is None:
            return {"ok": False, "image_count": 0, "images_dir": "", "stage_dir": ""}

        stage_root = Path(stage_root)
        stage_candidates = [
            (stage_root / f"Iteracja_{iter_value:03d}", stage_root / f"Iteracja_{iter_value:03d}" / "images"),
            (stage_root, stage_root / "images"),
        ]

        for stage_dir, images_dir in stage_candidates:
            try:
                if not images_dir.exists() or not images_dir.is_dir():
                    continue
                manifest_path = stage_dir / "stage_manifest.json"
                if manifest_path.exists():
                    manifest = self._read_json_file(manifest_path)
                    if isinstance(manifest, dict):
                        try:
                            pending_images = int(
                                manifest.get("pending_images", manifest.get("stage_images_total", 0)) or 0
                            )
                        except Exception:
                            pending_images = 0
                        entries = manifest.get("entries", {})
                        if pending_images <= 0 and (not isinstance(entries, dict) or not entries):
                            continue
                count = 0
                for image_path in images_dir.iterdir():
                    if image_path.is_file() and image_path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS:
                        count += 1
                if count > 0:
                    return {
                        "ok": True,
                        "image_count": int(count),
                        "images_dir": str(images_dir.resolve()),
                        "stage_dir": str(stage_dir.resolve()),
                        "iteration": int(iter_value),
                    }
            except Exception:
                continue

        return {
            "ok": False,
            "image_count": 0,
            "images_dir": "",
            "stage_dir": str((stage_root / f"Iteracja_{iter_value:03d}").resolve()),
            "iteration": int(iter_value),
        }

    def ensure_step1_image_source_restored_from_previous_iteration(self, project_name: str = None) -> str:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return ""
        project_data = self.state.get("projects", {}).get(project_name)
        if not isinstance(project_data, dict):
            return ""
        try:
            current_step = int(project_data.get("current_step", 1) or 1)
            current_iteration = int(project_data.get("current_iteration", 1) or 1)
        except Exception:
            return ""
        if current_step != 1:
            return ""
        current_source = str(project_data.get("master_pool_dir", "") or "").strip()
        if current_source:
            valid_current_source = self._resolve_valid_image_source_dir(
                current_source,
                project_name=project_name,
            )
            if valid_current_source:
                if valid_current_source != current_source:
                    project_data["master_pool_dir"] = valid_current_source
                    self.save_state()
                return ""
            project_data["master_pool_dir"] = ""
            project_data["step1_restored_image_source_dir"] = ""
        try:
            manual_clear_iteration = int(project_data.get("step1_source_manual_clear_iteration", 0) or 0)
        except Exception:
            manual_clear_iteration = 0
        if manual_clear_iteration == current_iteration:
            return ""

        current_iteration_source = self._resolve_current_iteration_image_source_for_e1(
            project_name=project_name,
            current_iteration=current_iteration,
        )
        if current_iteration_source:
            project_data["master_pool_dir"] = current_iteration_source
            project_data["step1_restored_image_source_dir"] = current_iteration_source
            project_data["step1_source_manual_clear_iteration"] = 0
            self.save_state()
            return current_iteration_source

        if current_iteration <= 1:
            return ""

        previous_source = self._resolve_previous_iteration_image_source_for_e1(
            project_name=project_name,
            current_iteration=current_iteration - 1,
            project_data=project_data,
        )
        if not previous_source:
            return ""

        project_data["master_pool_dir"] = previous_source
        project_data["step1_restored_image_source_dir"] = previous_source
        project_data["step1_source_manual_clear_iteration"] = 0
        self.save_state()
        return previous_source

    def _seed_iteration_from_master_pool(
        self,
        *,
        source_iteration: int,
        target_iteration: int,
        project_name: str,
    ) -> Dict[str, Any]:
        started_at = perf_counter()
        master_pool_dir = self.get_master_pool_dir(project_name)
        target_raw_dir = self.get_iteration_raw_dir(target_iteration, project_name)
        if master_pool_dir is None or target_raw_dir is None:
            return {"ok": False, "reason": "missing_project_dirs"}
        if not master_pool_dir.exists() or not master_pool_dir.is_dir():
            return {"ok": False, "reason": "missing_master_pool"}

        from .campaign_ingest_planner import CampaignIngestPlanner, CHAR_ALPHABET

        planner = CampaignIngestPlanner()
        used_registry = self.get_used_image_registry(project_name)
        balance_snapshot = self.refresh_ingest_balance_snapshot(project_name)
        source_manifest = self.load_ingest_manifest(source_iteration, project_name)

        preferred_batch_size = 0
        try:
            preferred_batch_size = int(
                source_manifest.get("selected_count", 0)
                or dict(source_manifest.get("proposal_summary") or {}).get("current_iteration_package_count", 0)
                or 0
            )
        except Exception:
            preferred_batch_size = 0

        if preferred_batch_size <= 0:
            try:
                source_raw_dir = self.get_iteration_raw_dir(source_iteration, project_name)
                if source_raw_dir is not None and source_raw_dir.exists():
                    preferred_batch_size = int(
                        len(
                            [
                                path for path in source_raw_dir.iterdir()
                                if path.is_file() and path.suffix.lower() in CONFIG.IMAGE_EXTENSIONS
                            ]
                        )
                    )
            except Exception:
                preferred_batch_size = 0

        if preferred_batch_size <= 0:
            preferred_batch_size = 200

        plan = planner.plan_from_master_pool(
            master_pool_dir=master_pool_dir,
            current_balance=balance_snapshot.get("char_balance", {}),
            used_source_keys=used_registry.get("source_keys", []),
            used_filenames=used_registry.get("filenames", []),
            batch_size=preferred_batch_size,
        )

        selected_items = list(plan.get("selected", []) or [])
        if not selected_items:
            return {
                "ok": False,
                "reason": "missing_remaining_images",
                "source_dir": str(master_pool_dir.resolve()),
                "target_dir": str(target_raw_dir.resolve()),
                "copied_images": 0,
                "manifest_images": 0,
                "manifest_cloned": False,
            }

        linked = 0
        selected_images: list[Dict[str, Any]] = []
        total_hist: Dict[str, int] = {ch: 0 for ch in CHAR_ALPHABET}

        for item in selected_items:
            try:
                source_path = Path(str(item.get("source_path", "") or "").strip())
            except Exception:
                continue
            if not source_path.exists() or not source_path.is_file():
                continue

            logical_target_path = target_raw_dir / source_path.name
            linked += 1

            char_hist = {
                str(ch): int(value)
                for ch, value in dict(item.get("char_histogram") or {}).items()
                if int(value or 0) > 0
            }
            for ch, value in char_hist.items():
                total_hist[ch] = total_hist.get(ch, 0) + int(value)

            selected_images.append(
                {
                    "name": source_path.name,
                    "source_path": str(source_path.resolve()),
                    "source_key": str(item.get("source_key", "") or "").strip(),
                    "target_path": str(source_path.resolve()),
                    "iteration_target_path": str(logical_target_path.resolve()),
                    "ground_truth_texts": list(item.get("ground_truth_texts", []) or []),
                    "char_histogram": char_hist,
                    "score": float(item.get("score", 0.0) or 0.0),
                    "score_details": dict(item.get("score_details", {}) or {}),
                }
            )

        if linked <= 0:
            return {
                "ok": False,
                "reason": "missing_remaining_images",
                "source_dir": str(master_pool_dir.resolve()),
                "target_dir": str(target_raw_dir.resolve()),
                "copied_images": 0,
                "manifest_images": 0,
                "manifest_cloned": False,
            }

        manifest = {
            "project": project_name,
            "iteration": int(target_iteration),
            "created_at": datetime.now().isoformat(),
            "selection_mode": "pool_reuse",
            "reused_from_iteration": int(source_iteration),
            "source_dir": str(master_pool_dir.resolve()),
            "target_dir": str(target_raw_dir.resolve()),
            "master_pool_dir": str(master_pool_dir.resolve()),
            "selected_count": len(selected_images),
            "manifest_only": True,
            "image_set_token": self.build_image_name_set_token(
                [str(item.get("name") or "").strip() for item in selected_images]
            ),
            "char_histogram": {k: int(v) for k, v in total_hist.items() if int(v) > 0},
            "selected_images": selected_images,
            "proposal_summary": {
                "planner_version": str(plan.get("planner_version", "") or ""),
                "generated_at": str(plan.get("generated_at", "") or ""),
                "selected_total": int(plan.get("selected_total", 0) or 0),
                "batch_size": int(plan.get("batch_size", plan.get("selected_total", 0)) or 0),
                "skipped_used": int(plan.get("skipped_used", 0) or 0),
                "source_iteration": int(source_iteration),
                "current_iteration_package_count": int(len(selected_images)),
                "source_kind": "master_pool",
            },
        }

        manifest_saved = bool(self.save_ingest_manifest(manifest, target_iteration, project_name))
        if not manifest_saved:
            return {
                "ok": False,
                "reason": "manifest_save_failed",
                "source_dir": str(master_pool_dir.resolve()),
                "target_dir": str(target_raw_dir.resolve()),
                "copied_images": 0,
                "manifest_images": 0,
                "manifest_cloned": False,
                "source_kind": "master_pool",
            }
        try:
            plan["ok"] = True
            plan["project"] = project_name
            plan["iteration"] = int(target_iteration)
            self.save_latest_ingest_plan(plan, project_name)
        except Exception:
            pass

        elapsed_ms = max(0.0, (perf_counter() - started_at) * 1000.0)
        if elapsed_ms >= 40.0:
            logger.debug(
                "[CampaignManager][PERF] seed_iteration_from_master_pool: "
                f"{elapsed_ms:.1f} ms | batch={int(preferred_batch_size)} "
                f"candidates={int(plan.get('candidates_total', 0) or 0)} "
                f"selected={int(plan.get('selected_total', 0) or 0)} manifest_only=1"
            )

        return {
            "ok": True,
            "source_dir": str(master_pool_dir.resolve()),
            "target_dir": str(target_raw_dir.resolve()),
            "copied_images": 0,
            "manifest_images": int(linked),
            "manifest_cloned": manifest_saved,
            "source_kind": "master_pool",
        }

    def _seed_iteration_from_stage(
        self,
        *,
        source_iteration: int,
        target_iteration: int,
        project_name: str,
    ) -> Dict[str, Any]:
        started_at = perf_counter()
        target_raw_dir = self.get_iteration_raw_dir(target_iteration, project_name)
        stage_root = self.get_staging_dir("plate_stage")
        if target_raw_dir is None or stage_root is None:
            return {"ok": False, "reason": "missing_project_dirs"}

        stage_root = Path(stage_root)
        stage_candidates = [
            (stage_root / f"Iteracja_{int(source_iteration):03d}", stage_root / f"Iteracja_{int(source_iteration):03d}" / "images"),
            (stage_root, stage_root / "images"),
        ]

        stage_iteration_dir = None
        stage_images_dir = None
        for candidate_dir, candidate_images_dir in stage_candidates:
            if candidate_images_dir.exists() and candidate_images_dir.is_dir():
                stage_iteration_dir = candidate_dir
                stage_images_dir = candidate_images_dir
                break

        if stage_iteration_dir is None or stage_images_dir is None:
            missing_images_dir = stage_candidates[0][1]
            return {
                "ok": False,
                "reason": "missing_stage_images",
                "source_dir": str(missing_images_dir.resolve()),
                "target_dir": str(target_raw_dir.resolve()),
                "copied_images": 0,
                "manifest_images": 0,
                "manifest_cloned": False,
            }

        # These are manifest links. Resolve the two parent directories once;
        # resolving every future target repeatedly traversed missing Windows paths.
        source_root = stage_images_dir.resolve()
        target_root = target_raw_dir.resolve()
        selected_images = []
        try:
            with os.scandir(source_root) as entries:
                for entry in entries:
                    try:
                        if os.path.splitext(entry.name)[1].lower() not in CONFIG.IMAGE_EXTENSIONS or not entry.is_file():
                            continue
                        source_path = Path(entry.path)
                        if entry.is_symlink():
                            source_path = source_path.resolve()
                        selected_images.append({
                            "name": entry.name,
                            "source_path": str(source_path),
                            "target_path": str(source_path),
                            "iteration_target_path": str(target_root / entry.name),
                        })
                    except OSError:
                        continue
        except OSError:
            selected_images = []

        if not selected_images:
            return {
                "ok": False,
                "reason": "missing_stage_images",
                "source_dir": str(stage_images_dir.resolve()),
                "target_dir": str(target_raw_dir.resolve()),
                "copied_images": 0,
                "manifest_images": 0,
                "manifest_cloned": False,
            }

        manifest = {
            "project": project_name,
            "iteration": int(target_iteration),
            "created_at": datetime.now().isoformat(),
            "selection_mode": "stage_reuse",
            "reused_from_iteration": int(source_iteration),
            "source_dir": str(stage_images_dir.resolve()),
            "target_dir": str(target_raw_dir.resolve()),
            "selected_count": int(len(selected_images)),
            "selected_images": selected_images,
            "manifest_only": True,
            "image_set_token": self.build_image_name_set_token(
                [str(item.get("name") or "").strip() for item in selected_images]
            ),
            "char_histogram": {},
            "proposal_summary": {
                "source_iteration": int(source_iteration),
                "source_kind": "stage",
                "current_iteration_package_count": int(len(selected_images)),
                "stage_transfer_mode": "manifest_link",
            },
        }
        manifest_saved = bool(self.save_ingest_manifest(manifest, target_iteration, project_name))
        if not manifest_saved:
            return {
                "ok": False,
                "reason": "manifest_save_failed",
                "source_dir": str(stage_images_dir.resolve()),
                "target_dir": str(target_raw_dir.resolve()),
                "copied_images": 0,
                "manifest_images": 0,
                "manifest_cloned": False,
                "source_kind": "stage",
            }

        manifest_path = stage_iteration_dir / "stage_manifest.json"
        if manifest_path.exists():
            try:
                manifest_path.write_text(
                    json.dumps(
                        {
                            "project": project_name,
                            "iteration": int(source_iteration),
                            "updated_at": datetime.now().isoformat(timespec="seconds"),
                            "pending_images": 0,
                            "stage_images_total": 0,
                            "entries": {},
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

        elapsed_ms = max(0.0, (perf_counter() - started_at) * 1000.0)
        if elapsed_ms >= 40.0:
            logger.debug(
                "[CampaignManager][PERF] seed_iteration_from_stage: "
                f"{elapsed_ms:.1f} ms | images={int(len(selected_images))} mode=manifest_link"
            )

        return {
            "ok": True,
            "source_dir": str(stage_images_dir.resolve()),
            "target_dir": str(target_raw_dir.resolve()),
            "copied_images": 0,
            "manifest_images": int(len(selected_images)),
            "manifest_cloned": manifest_saved,
            "source_kind": "stage",
        }

    def advance_to_next_iteration(self, start_mode: str = "new_input") -> Dict[str, Any]:
        act = self.state.get("active_project", "")
        if not act or act not in self.state.get("projects", {}):
            return {"ok": False, "reason": "missing_active_project"}

        start_mode = str(start_mode or "new_input").strip().lower()
        if start_mode not in {"new_input", "reuse_input"}:
            start_mode = "new_input"

        project_data = self.state["projects"][act]
        project_status = str(project_data.get("project_status", "active") or "active").strip().lower()
        current_step = int(project_data.get("current_step", 1) or 1)
        step4_finish_ready = bool(project_data.get("step4_finish_ready", False))
        if project_status in {"paused", "completed"}:
            return {"ok": False, "reason": "project_not_active"}
        if current_step < 5 and not step4_finish_ready:
            return {
                "ok": False,
                "reason": "step4_not_finished",
                "current_step": current_step,
                "step4_finish_ready": step4_finish_ready,
            }
        current_target = self._normalize_iteration_target(project_data.get("iteration_target", ""))
        current_iteration = int(project_data.get("current_iteration", 1) or 1)
        next_iteration = current_iteration + 1
        pending_stage_state = self.get_manual_plate_stage_images_state(
            source_iteration=current_iteration,
            project_name=act,
        )
        pending_stage_images = int(pending_stage_state.get("image_count", 0) or 0)
        previous_image_source = self._resolve_previous_iteration_image_source_for_e1(
            project_name=act,
            current_iteration=current_iteration,
            project_data=project_data,
        )

        result: Dict[str, Any] = {
            "ok": True,
            "requested_mode": start_mode,
            "effective_mode": start_mode,
            "previous_iteration": current_iteration,
            "next_iteration": next_iteration,
            "copied_images": 0,
            "manifest_cloned": False,
            "previous_image_source": previous_image_source,
            "restored_master_pool_dir": "",
            "pending_stage_images": pending_stage_images,
            "needs_new_image_source": False,
        }

        reuse_result: Dict[str, Any] | None = None
        if start_mode == "reuse_input":
            reuse_result = self._seed_iteration_from_stage(
                source_iteration=current_iteration,
                target_iteration=next_iteration,
                project_name=act,
            )
            if not reuse_result.get("ok"):
                reuse_result = self._seed_iteration_from_master_pool(
                    source_iteration=current_iteration,
                    target_iteration=next_iteration,
                    project_name=act,
                )
            if not reuse_result.get("ok"):
                result.update(reuse_result)
                result["ok"] = False
                return result

        project_data["current_iteration"] = next_iteration
        project_data["current_step"] = 1
        project_data["step1_status"] = "pending"
        project_data["step1_source_manual_clear_iteration"] = 0
        project_data["step1_restored_image_source_dir"] = ""
        project_data["project_start_plate_source_run"] = ""
        project_data["project_start_plate_source_xml"] = ""
        project_data["project_start_plate_source_input"] = ""
        project_data["project_start_plate_source_mode"] = ""
        project_data["project_start_plate_source_iteration"] = 0
        project_data["e1_resource_contract_baseline_iteration"] = 0
        project_data["e1_resource_contract_baseline"] = {}
        project_data["e1_resource_contract_last_rollback_at"] = ""
        project_data["e1_resource_contract_last_rollback_from_path"] = ""
        project_data["e1_resource_contract_last_rollback_to_path"] = ""
        project_data["t02_at_review_committed_iteration"] = 0
        project_data["t02_at_review_committed_at"] = ""
        project_data["t02_at_review_committed_run"] = ""
        project_data["t02_at_review_committed_images"] = 0
        project_data["t02_at_review_committed_plates"] = 0
        project_data["project_start_scope_plate_run"] = ""

        if current_target in {"plate", "char"}:
            project_data["last_iteration_target"] = current_target
        project_data["iteration_target"] = ""
        project_data["step4_finish_ready"] = False
        project_data["step4_last_run_id"] = ""
        project_data["step4_last_target"] = ""
        project_data["step4_last_iteration"] = 0
        project_data["step4_without_training_ready"] = False
        project_data["step4_without_training_target"] = ""
        project_data["step4_without_training_iteration"] = 0

        if start_mode == "reuse_input":
            result.update(reuse_result or {})
            result["effective_mode"] = "reuse_input"
        else:
            if previous_image_source:
                project_data["master_pool_dir"] = previous_image_source
                project_data["step1_restored_image_source_dir"] = previous_image_source
                result["restored_master_pool_dir"] = previous_image_source
            else:
                project_data["master_pool_dir"] = ""
                result["needs_new_image_source"] = True

        # Nowa iteracja zaczyna się od pełnego resetu stanów etapów zależnych od danych wejściowych.
        project_data["step2_status"] = "pending"
        project_data["step2_staging_run"] = ""
        project_data["step3_status"] = "pending"
        project_data["step3_substep"] = 1
        project_data["step3_stage1_done"] = False
        project_data["step3_stage2_done"] = False
        project_data["step3_extract_entry_mode"] = ""
        project_data["step3_extract_workflow_step"] = "entry"
        project_data["step3_extract_annotation_run_dir"] = ""
        project_data["step3_extract_xml_path"] = ""
        project_data["step3_extract_images_dir"] = ""
        project_data["project_status"] = "active"
        project_data["project_paused_at"] = ""
        project_data["project_completed_at"] = ""
        try:
            self.ensure_e1_resource_contract_baseline(force=True, project_name=act)
        except Exception:
            pass
        self.save_state()
        try:
            self.clear_project_iteration_ui_snapshots(act)
        except Exception:
            pass
        if start_mode != "reuse_input":
            try:
                self.clear_latest_ingest_plan(act)
            except Exception:
                pass
        return result

    def set_global_model(self, model_type: str, model_path: str):
        act = self.state.get("active_project", "")
        if not act or act not in self.state.get("projects", {}): return
        key = f"best_{model_type}_model"
        self.state["projects"][act][key] = str(model_path)
        self.save_state()

    def get_global_model(self, model_type: str) -> str:
        act = self.state.get("active_project", "")
        if not act or act not in self.state.get("projects", {}): return ""
        key = f"best_{model_type}_model"
        return self.state["projects"][act].get(key, "")

    def set_last_plate_training_source(
        self,
        dataset_path: str = "",
        source_run_path: str = "",
        source_xml_path: str = "",
    ):
        act = self.get_active_project_name()
        if not act:
            return

        project_data = self.state["projects"][act]
        project_data["last_plate_training_dataset"] = str(dataset_path or "").strip()
        project_data["last_plate_training_source_run"] = str(source_run_path or "").strip()
        project_data["last_plate_training_source_xml"] = str(source_xml_path or "").strip()
        self.save_state()

    def get_last_plate_training_source(self) -> Dict[str, str]:
        act = self.get_active_project_name()
        if not act:
            return {
                "dataset_path": "",
                "source_run_path": "",
                "source_xml_path": "",
            }

        project_data = self.state["projects"].get(act, {})
        return {
            "dataset_path": str(project_data.get("last_plate_training_dataset", "") or "").strip(),
            "source_run_path": str(project_data.get("last_plate_training_source_run", "") or "").strip(),
            "source_xml_path": str(project_data.get("last_plate_training_source_xml", "") or "").strip(),
        }

    def set_step4_finish_state(
        self,
        ready: bool,
        *,
        run_id: str = "",
        target: str = "",
        iteration_num: int | None = None,
        selection_confirmed: bool = False,
        model_path: str = "",
    ) -> None:
        act = self.get_active_project_name()
        if not act:
            return

        project_data = self.state["projects"][act]
        is_ready = bool(ready)
        if iteration_num is None:
            try:
                iteration_num = int(project_data.get("current_iteration", 1) or 1)
            except Exception:
                iteration_num = 1
        project_data["step4_finish_ready"] = is_ready
        project_data["step4_last_run_id"] = str(run_id or "").strip() if is_ready else ""
        project_data["step4_last_target"] = self._normalize_iteration_target(target) if is_ready else ""
        project_data["step4_last_iteration"] = int(iteration_num or 0) if is_ready else 0
        project_data["step4_model_choice_confirmed"] = bool(selection_confirmed) if is_ready else False
        project_data["step4_selected_model_path"] = str(model_path or "").strip() if is_ready and selection_confirmed else ""
        if is_ready:
            project_data["step4_without_training_ready"] = False
            project_data["step4_without_training_target"] = ""
            project_data["step4_without_training_iteration"] = 0
        self.save_state()

    def get_step4_finish_state(self) -> Dict[str, Any]:
        act = self.get_active_project_name()
        if not act:
            return {
                "ready": False,
                "run_id": "",
                "target": "",
                "iteration": 0,
                "selection_confirmed": False,
                "model_path": "",
            }

        project_data = self.state["projects"].get(act, {})
        state = {
            "ready": bool(project_data.get("step4_finish_ready", False)),
            "run_id": str(project_data.get("step4_last_run_id", "") or "").strip(),
            "target": self._normalize_iteration_target(project_data.get("step4_last_target", "")),
            "iteration": int(project_data.get("step4_last_iteration", 0) or 0),
            "selection_confirmed": bool(project_data.get("step4_model_choice_confirmed", False)),
            "model_path": str(project_data.get("step4_selected_model_path", "") or "").strip(),
        }
        if not bool(state.get("ready")):
            return state
        run_id = str(state.get("run_id", "") or "").strip()
        iteration_num = int(state.get("iteration", 0) or 0)
        if not bool(state.get("selection_confirmed")):
            return {
                "ready": False,
                "run_id": run_id,
                "target": state.get("target", ""),
                "iteration": iteration_num,
                "selection_confirmed": False,
                "model_path": "",
            }
        if not run_id or iteration_num <= 0:
            return {"ready": False, "run_id": "", "target": "", "iteration": 0, "selection_confirmed": False, "model_path": ""}
        selected_model_path = str(state.get("model_path", "") or "").strip()
        if selected_model_path:
            try:
                selected_model_ready = bool(Path(selected_model_path).exists() and Path(selected_model_path).is_file())
            except Exception:
                selected_model_ready = bool(selected_model_path)
            if selected_model_ready:
                return state

        def _record_matches(record: Dict[str, Any] | None, *, allow_legacy: bool = False) -> bool:
            if not isinstance(record, dict):
                return False
            if str(record.get("run_id", "") or "").strip() != run_id:
                return False
            declared_iteration = 0
            for field in ("trained_iteration", "source_iteration", "iteration"):
                try:
                    declared_iteration = int(record.get(field, 0) or 0)
                except Exception:
                    declared_iteration = 0
                if declared_iteration > 0:
                    break
            if declared_iteration > 0:
                return declared_iteration == iteration_num
            return bool(allow_legacy)

        try:
            iteration_state = self.get_iteration_state(iteration_num=iteration_num, project_name=act)
            if _record_matches(iteration_state.get("step4_training"), allow_legacy=True):
                return state
        except Exception:
            pass
        try:
            bundle = self.get_iteration_artifact_bundle(iteration_num=iteration_num, project_name=act)
            if _record_matches(bundle.get("step4_training"), allow_legacy=False):
                return state
        except Exception:
            pass
        return {
            "ready": False,
            "run_id": "",
            "target": state.get("target", ""),
            "iteration": iteration_num,
            "selection_confirmed": False,
            "model_path": "",
        }

    def set_step4_without_training_decision(
        self,
        ready: bool,
        *,
        target: str = "",
        iteration_num: int | None = None,
    ) -> None:
        act = self.get_active_project_name()
        if not act:
            return

        project_data = self.state["projects"][act]
        is_ready = bool(ready)
        if iteration_num is None:
            try:
                iteration_num = int(project_data.get("current_iteration", 1) or 1)
            except Exception:
                iteration_num = 1
        project_data["step4_without_training_ready"] = is_ready
        project_data["step4_without_training_target"] = self._normalize_iteration_target(target) if is_ready else ""
        project_data["step4_without_training_iteration"] = int(iteration_num or 0) if is_ready else 0
        self.save_state()

    def get_step4_without_training_decision(self) -> Dict[str, Any]:
        act = self.get_active_project_name()
        if not act:
            return {
                "ready": False,
                "target": "",
                "iteration": 0,
            }

        project_data = self.state["projects"].get(act, {})
        return {
            "ready": bool(project_data.get("step4_without_training_ready", False)),
            "target": self._normalize_iteration_target(project_data.get("step4_without_training_target", "")),
            "iteration": int(project_data.get("step4_without_training_iteration", 0) or 0),
        }

    def set_last_plate_manual_source(
        self,
        source_run_path: str = "",
        source_xml_path: str = "",
        source_input_path: str = "",
    ):
        act = self.get_active_project_name()
        if not act:
            return

        project_data = self.state["projects"][act]
        project_data["last_plate_manual_source_run"] = str(source_run_path or "").strip()
        project_data["last_plate_manual_source_xml"] = str(source_xml_path or "").strip()
        project_data["last_plate_manual_source_input"] = str(source_input_path or "").strip()
        self.save_state()

    def get_last_plate_manual_source(self) -> Dict[str, str]:
        act = self.get_active_project_name()
        if not act:
            return {
                "source_run_path": "",
                "source_xml_path": "",
                "source_input_path": "",
            }

        project_data = self.state["projects"].get(act, {})
        return {
            "source_run_path": str(project_data.get("last_plate_manual_source_run", "") or "").strip(),
            "source_xml_path": str(project_data.get("last_plate_manual_source_xml", "") or "").strip(),
            "source_input_path": str(project_data.get("last_plate_manual_source_input", "") or "").strip(),
        }
    def get_project_root_dir(self, project_name: str) -> Path:
        folder_name = self.state["projects"][project_name]["folder_name"]
        root = Path(CONFIG.DIR_9_PROJECTS) / folder_name
        self._ensure_project_workspace_tree(root)
        return root

    def get_project_state_dir(self, project_name: str = None) -> Path | None:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return None

        state_dir = self.get_project_root_dir(project_name) / "_campaign_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir

    def clear_project_iteration_ui_snapshots(self, project_name: str = None) -> bool:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return False

        state_dir = self.get_project_state_dir(project_name)
        if state_dir is None:
            return False

        removed_any = False
        for snapshot_name in (
            "annotation_ui_state.json",
            "wizard_view_cache.json",
        ):
            snapshot_path = state_dir / snapshot_name
            try:
                PROJECT_CACHE.invalidate_json(snapshot_path)
                if snapshot_path.exists():
                    snapshot_path.unlink()
                    removed_any = True
            except Exception as e:
                logger.debug(
                    f"Nie udało się usunąć snapshotu iteracji {snapshot_path}: {e}"
                )

        return removed_any

    def get_project_ingest_state_dir(self, project_name: str = None) -> Path | None:
        state_dir = self.get_project_state_dir(project_name)
        if state_dir is None:
            return None

        ingest_dir = state_dir / "ingest"
        ingest_dir.mkdir(parents=True, exist_ok=True)
        return ingest_dir

    def get_artifact_registry_path(self, project_name: str = None) -> Path | None:
        state_dir = self.get_project_state_dir(project_name)
        if state_dir is None:
            return None
        return state_dir / "artifact_registry.json"

    @staticmethod
    def _safe_registry_path_value(path_like) -> str:
        raw_value = str(path_like or "").strip()
        if not raw_value:
            return ""
        try:
            return str(Path(raw_value).resolve())
        except Exception:
            return raw_value

    @staticmethod
    def _build_registry_path_token(path_like) -> str:
        raw_value = str(path_like or "").strip()
        if not raw_value:
            return ""
        try:
            path = Path(raw_value)
        except Exception:
            return raw_value
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = raw_value
        try:
            stat = path.stat()
            return (
                f"{resolved}|"
                f"{int(getattr(stat, 'st_mtime_ns', 0) or 0)}|"
                f"{int(getattr(stat, 'st_size', 0) or 0)}"
            )
        except Exception:
            return resolved

    @staticmethod
    def _normalize_image_set_name(name_like) -> str:
        raw_value = str(name_like or "").strip()
        if not raw_value:
            return ""
        raw_value = raw_value.replace("\\", "/")
        try:
            return str(Path(raw_value).name or "").strip().lower()
        except Exception:
            return str(raw_value.rsplit("/", 1)[-1] or "").strip().lower()

    def build_image_name_set_token(
        self,
        image_names: List[str] | None = None,
        *,
        images_dir: str | Path | None = None,
    ) -> str:
        normalized_names: list[str] = []

        for raw_name in list(image_names or []):
            normalized = self._normalize_image_set_name(raw_name)
            if normalized:
                normalized_names.append(normalized)

        if not normalized_names and images_dir:
            try:
                return IMAGE_DIRECTORIES.snapshot(images_dir, CONFIG.IMAGE_EXTENSIONS, recursive=True).token
            except (TypeError, ValueError, OSError):
                return ""

        unique_names = sorted(set(normalized_names))
        if not unique_names:
            return ""

        digest = hashlib.sha1("\n".join(unique_names).encode("utf-8")).hexdigest()[:20]
        return f"iset_{len(unique_names):05d}_{digest}"

    def get_iteration_image_set_token(
        self,
        iteration_num: int = None,
        project_name: str = None,
    ) -> str:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return ""

        try:
            iter_value = int(iteration_num or self.state["projects"][project_name].get("current_iteration", 1) or 1)
        except Exception:
            iter_value = 1

        try:
            manifest = self.load_ingest_manifest(iter_value, project_name)
        except Exception:
            manifest = {}

        if isinstance(manifest, dict):
            token = str(manifest.get("image_set_token", "") or "").strip()
            if token:
                return token

            manifest_names = [
                str(item.get("name", "") or "").strip()
                for item in list(manifest.get("selected_images") or [])
                if isinstance(item, dict) and str(item.get("name", "") or "").strip()
            ]
            token = self.build_image_name_set_token(manifest_names)
            if token:
                return token

        try:
            source_dir = self.get_iteration_image_source_dir(iter_value, project_name)
        except Exception:
            source_dir = self.get_iteration_raw_dir(iter_value, project_name)
        return self.build_image_name_set_token(images_dir=source_dir)

    @staticmethod
    def _deep_merge_registry_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in dict(updates or {}).items():
            if isinstance(value, dict):
                current = base.get(key)
                if not isinstance(current, dict):
                    current = {}
                base[key] = CampaignManager._deep_merge_registry_dict(dict(current), value)
            else:
                base[key] = value
        return base

    def load_artifact_registry(self, project_name: str = None) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {
                "project": "",
                "updated_at": "",
                "packages": {},
                "iteration_index": {},
                "iteration_state": {},
            }

        runtime_cached = self._artifact_registry_runtime_cache.get(project_name)
        if isinstance(runtime_cached, dict):
            return dict(runtime_cached)

        registry_path = self.get_artifact_registry_path(project_name)
        if registry_path is None or not registry_path.exists():
            return {
                "project": project_name,
                "updated_at": "",
                "packages": {},
                "iteration_index": {},
                "iteration_state": {},
            }

        cache_key = None
        try:
            stat = registry_path.stat()
            cache_key = (
                project_name,
                int(getattr(stat, "st_mtime_ns", 0) or 0),
                int(getattr(stat, "st_size", 0) or 0),
            )
        except Exception:
            cache_key = None

        if cache_key is not None:
            cached = self._artifact_registry_cache.get(cache_key)
            if isinstance(cached, dict):
                return dict(cached)

        payload = self._read_json_file(registry_path)
        if not isinstance(payload, dict):
            payload = {}
        payload["project"] = str(payload.get("project") or project_name).strip()
        if not isinstance(payload.get("packages"), dict):
            payload["packages"] = {}
        if not isinstance(payload.get("iteration_index"), dict):
            payload["iteration_index"] = {}
        if not isinstance(payload.get("iteration_state"), dict):
            payload["iteration_state"] = {}

        if cache_key is not None:
            self._artifact_registry_cache[cache_key] = dict(payload)
        self._artifact_registry_runtime_cache[project_name] = dict(payload)
        return payload

    def save_artifact_registry(self, payload: Dict[str, Any], project_name: str = None) -> bool:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return False
        registry_path = self.get_artifact_registry_path(project_name)
        if registry_path is None:
            return False
        safe_payload = dict(payload or {})
        safe_payload["project"] = project_name
        safe_payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if not isinstance(safe_payload.get("packages"), dict):
            safe_payload["packages"] = {}
        if not isinstance(safe_payload.get("iteration_index"), dict):
            safe_payload["iteration_index"] = {}
        if not isinstance(safe_payload.get("iteration_state"), dict):
            safe_payload["iteration_state"] = {}
        ok = self._write_json_file(registry_path, safe_payload)
        if ok:
            self._artifact_registry_cache.clear()
            self._artifact_registry_runtime_cache.pop(project_name, None)
            self.invalidate_step3_char_source_state_cache()
        return ok

    def upsert_iteration_state(
        self,
        *,
        iteration_num: int | None = None,
        updates: Dict[str, Any] | None = None,
        project_name: str = None,
    ) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {}

        iter_value = int(iteration_num or self.state["projects"][project_name].get("current_iteration", 1) or 1)
        registry = self.load_artifact_registry(project_name)
        iteration_state = registry.setdefault("iteration_state", {})
        if not isinstance(iteration_state, dict):
            iteration_state = {}
            registry["iteration_state"] = iteration_state

        entry = dict(iteration_state.get(str(iter_value)) or {})
        entry.setdefault("project", project_name)
        entry.setdefault("iteration", iter_value)
        entry.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        entry["updated_at"] = datetime.now().isoformat(timespec="seconds")

        updates_dict = dict(updates or {})
        if updates_dict:
            entry = self._deep_merge_registry_dict(entry, updates_dict)

        iteration_state[str(iter_value)] = entry
        if not self.save_artifact_registry(registry, project_name):
            return {}
        return dict(entry)

    def get_iteration_state(
        self,
        *,
        iteration_num: int | None = None,
        project_name: str = None,
    ) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {}
        iter_value = int(iteration_num or self.state["projects"][project_name].get("current_iteration", 1) or 1)
        registry = self.load_artifact_registry(project_name)
        iteration_state = registry.get("iteration_state", {})
        if not isinstance(iteration_state, dict):
            return {}
        entry = iteration_state.get(str(iter_value))
        return dict(entry) if isinstance(entry, dict) else {}

    def build_iteration_artifact_package_id(
        self,
        images_dir: str | Path | None = None,
        *,
        iteration_num: int | None = None,
        image_set_token: str | None = None,
        project_name: str = None,
    ) -> str:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return ""
        normalized_images_dir = self._safe_registry_path_value(images_dir)
        normalized_image_set_token = str(image_set_token or "").strip()
        if normalized_images_dir and normalized_image_set_token:
            digest = hashlib.sha1(
                f"{normalized_images_dir.lower()}::{normalized_image_set_token}".encode("utf-8")
            ).hexdigest()[:16]
            return f"pkg_{digest}"
        if normalized_images_dir:
            digest = hashlib.sha1(normalized_images_dir.lower().encode("utf-8")).hexdigest()[:16]
            return f"pkg_{digest}"
        iter_value = int(iteration_num or self.state["projects"][project_name].get("current_iteration", 1) or 1)
        digest = hashlib.sha1(f"{project_name.lower()}::{iter_value}".encode("utf-8")).hexdigest()[:16]
        return f"iter_{iter_value:03d}_{digest}"

    def upsert_iteration_artifact_bundle(
        self,
        *,
        images_dir: str | Path | None = None,
        iteration_num: int | None = None,
        image_set_token: str | None = None,
        updates: Dict[str, Any] | None = None,
        project_name: str = None,
    ) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {}

        try:
            iter_value = int(iteration_num or self.state["projects"][project_name].get("current_iteration", 1) or 1)
        except Exception:
            iter_value = 1

        try:
            default_images_dir = self.get_iteration_image_source_dir(iter_value, project_name)
        except Exception:
            default_images_dir = None
        normalized_images_dir = self._safe_registry_path_value(images_dir or default_images_dir or self.get_master_pool_dir(project_name))
        registry = self.load_artifact_registry(project_name)
        packages = registry.setdefault("packages", {})
        iteration_index = registry.setdefault("iteration_index", {})
        existing_package_id = str((iteration_index or {}).get(str(iter_value), "") or "").strip()
        existing_package = dict(packages.get(existing_package_id) or {}) if existing_package_id else {}
        updates_dict = dict(updates or {})
        updates_image_source = dict(updates_dict.get("image_source") or {})
        resolved_image_set_token = str(
            image_set_token
            or updates_image_source.get("image_set_token")
            or dict(existing_package.get("image_source") or {}).get("image_set_token")
            or self.get_iteration_image_set_token(iter_value, project_name)
            or ""
        ).strip()

        package_id = self.build_iteration_artifact_package_id(
            normalized_images_dir,
            iteration_num=iter_value,
            image_set_token=resolved_image_set_token,
            project_name=project_name,
        )
        if not package_id:
            return {}

        package = dict(packages.get(package_id) or existing_package or {})
        package.setdefault("package_id", package_id)
        package.setdefault("project", project_name)
        package.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        package["updated_at"] = datetime.now().isoformat(timespec="seconds")
        package["iteration_first_seen"] = int(package.get("iteration_first_seen", iter_value) or iter_value)
        package["iteration_last_seen"] = int(iter_value)
        package["images_dir"] = normalized_images_dir or str(package.get("images_dir", "") or "").strip()
        package["images_token"] = self._build_registry_path_token(package.get("images_dir"))
        if resolved_image_set_token:
            image_source = package.setdefault("image_source", {})
            if isinstance(image_source, dict):
                image_source.setdefault("image_set_token", resolved_image_set_token)
                try:
                    image_source.setdefault(
                        "image_set_count",
                        int(str(resolved_image_set_token).split("_", 2)[1]),
                    )
                except Exception:
                    pass
        iterations = {
            int(value)
            for value in list(package.get("iterations") or [])
            if str(value).strip().isdigit()
        }
        iterations.add(iter_value)
        package["iterations"] = sorted(iterations)

        if updates_dict:
            package = self._deep_merge_registry_dict(package, updates_dict)

        if existing_package_id and existing_package_id != package_id:
            try:
                packages.pop(existing_package_id, None)
            except Exception:
                pass
        packages[package_id] = package
        iteration_index[str(iter_value)] = package_id
        if not self.save_artifact_registry(registry, project_name):
            return {}
        return dict(package)

    def get_iteration_artifact_bundle(
        self,
        *,
        images_dir: str | Path | None = None,
        iteration_num: int | None = None,
        image_set_token: str | None = None,
        project_name: str = None,
    ) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {}
        registry = self.load_artifact_registry(project_name)
        packages = registry.get("packages", {})
        if not isinstance(packages, dict):
            return {}

        iter_value = int(iteration_num or self.state["projects"][project_name].get("current_iteration", 1) or 1)
        resolved_image_set_token = str(
            image_set_token
            or self.get_iteration_image_set_token(iter_value, project_name)
            or ""
        ).strip()
        normalized_images_dir = self._safe_registry_path_value(
            images_dir or self.get_iteration_image_source_dir(iter_value, project_name)
        )
        if normalized_images_dir:
            package_id = self.build_iteration_artifact_package_id(
                normalized_images_dir,
                iteration_num=iter_value,
                image_set_token=resolved_image_set_token,
                project_name=project_name,
            )
            package = packages.get(package_id)
            if isinstance(package, dict):
                return dict(package)

        package_id = str((registry.get("iteration_index") or {}).get(str(iter_value), "") or "").strip()
        package = packages.get(package_id)
        return dict(package) if isinstance(package, dict) else {}

    @staticmethod
    def _normalize_project_model_target(model_type: str | None) -> str:
        value = str(model_type or "").strip().lower()
        if value in {"plate", "plates", "mt", "pose", "yolo_pose"}:
            return "plate"
        if value in {"char", "chars", "character", "characters", "mz", "detect", "yolo_detect"}:
            return "char"
        return value

    @staticmethod
    def _resolve_existing_model_artifact_path(payload: Dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return ""
        for key in ("best_weights", "path", "model_path", "weights_path", "last_weights"):
            candidate = str(payload.get(key, "") or "").strip()
            if not candidate:
                continue
            try:
                path = Path(candidate)
            except Exception:
                continue
            if path.exists() and path.is_file():
                try:
                    return str(path.resolve())
                except Exception:
                    return str(path)
        return ""

    def _load_training_history_run(
        self,
        run_id: str,
        *,
        project_name: str = None,
    ) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        run_key = str(run_id or "").strip()
        if not project_name or not run_key:
            return {}
        try:
            history_path = self.get_project_root_dir(project_name) / "5_training_runs" / "training_history.json"
        except Exception:
            return {}
        if not history_path.exists():
            return {}
        history = self._read_json_file(history_path)
        runs = history.get("runs") if isinstance(history, dict) else {}
        run = runs.get(run_key) if isinstance(runs, dict) else {}
        return dict(run) if isinstance(run, dict) else {}

    def sync_step4_training_record_from_history(
        self,
        *,
        iteration_num: int | None = None,
        project_name: str = None,
    ) -> Dict[str, Any]:
        """Reconcile the campaign Step 4 record with the authoritative training history.

        The GUI may be closed from Z4 before returning to the graph. If the worker
        already wrote terminal training results, the gate must show a model
        candidate/failure, not a generic interrupted-work warning.
        """

        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {}
        try:
            iter_value = int(iteration_num or self.state["projects"][project_name].get("current_iteration", 1) or 1)
        except Exception:
            iter_value = 1

        try:
            iteration_state = self.get_iteration_state(iteration_num=iter_value, project_name=project_name)
            record = dict(iteration_state.get("step4_training") or {})
        except Exception:
            record = {}
        if not record:
            try:
                bundle = self.get_iteration_artifact_bundle(iteration_num=iter_value, project_name=project_name)
                record = dict(bundle.get("step4_training") or {})
            except Exception:
                record = {}
        run_id = str(record.get("run_id", "") or "").strip()
        if not run_id:
            return record

        terminal_statuses = {
            "completed",
            "failed",
            "paused",
            "cancelled",
        }

        recoverable_statuses = {
            "failed",
            "paused",
            "pending",
            "running",
            "cancelled",
        }

        def _step4_record_target(source: Dict[str, Any] | None = None) -> str:
            payload = dict(source or {})
            target = self._normalize_project_model_target(
                payload.get("target") or record.get("target")
            )
            if target:
                return target
            try:
                return self._normalize_project_model_target(
                    self.state["projects"][project_name].get("iteration_target", "")
                )
            except Exception:
                return ""

        def _set_recoverable_step4_training_session(
            status_value: str,
            source: Dict[str, Any] | None = None,
            *,
            reason: str,
        ) -> None:
            try:
                state = self.get_iteration_state(iteration_num=iter_value, project_name=project_name)
                session = dict(state.get("step4_work_session") or {})
                session_run = str(session.get("run_id", "") or "").strip()
                session_area = str(session.get("work_area", "") or "").strip().lower()
                session_substep = str(session.get("substep", "") or "").strip().lower()
                session_state = str(session.get("state", "") or "").strip().lower()
                is_step4_session = bool(
                    session_area == "z4"
                    and (
                        session_run == run_id
                        or session_substep in {"pz2", "train", "training"}
                        or session_state in {"active", "started", "interrupted", "paused", "dirty"}
                    )
                )
                if session and not is_step4_session:
                    return
                now = datetime.now().isoformat(timespec="seconds")
                session.update(
                    {
                        "active": True,
                        "state": "interrupted",
                        "work_area": "z4",
                        "substep": "train",
                        "target": _step4_record_target(source),
                        "iteration": iter_value,
                        "run_id": run_id,
                        "training_status": status_value,
                        "reason": reason,
                        "updated_at": now,
                    }
                )
                session.setdefault("interrupted_at", now)
                session.setdefault("working_gate_id", "T06")
                self.upsert_iteration_state(
                    iteration_num=iter_value,
                    updates={"step4_work_session": session},
                    project_name=project_name,
                )
            except Exception as exc:
                logger.debug(f"Nie udalo sie oznaczyc przerwanej sesji Z4: {exc}")

        def _close_terminal_step4_session(status_value: str) -> None:
            try:
                state = self.get_iteration_state(iteration_num=iter_value, project_name=project_name)
                session = dict(state.get("step4_work_session") or {})
                session_run = str(session.get("run_id", "") or "").strip()
                session_area = str(session.get("work_area", "") or "").strip().lower()
                session_substep = str(session.get("substep", "") or "").strip().lower()
                session_state = str(session.get("state", "") or "").strip().lower()
                should_close_session = bool(
                    session_area == "z4"
                    and session_state in {"active", "started", "interrupted", "paused", "dirty"}
                    and (session_run == run_id or session_substep in {"pz2", "train", "training"})
                )
                if should_close_session:
                    now = datetime.now().isoformat(timespec="seconds")
                    session.update(
                        {
                            "active": False,
                            "state": f"training_{status_value}",
                            "closed_by": "training_history_reconcile",
                            "closed_at": now,
                            "updated_at": now,
                        }
                    )
                    self.upsert_iteration_state(
                        iteration_num=iter_value,
                        updates={"step4_work_session": session},
                        project_name=project_name,
                    )
            except Exception as exc:
                logger.debug(f"Nie udało się domknąć osieroconej sesji Z4: {exc}")

        record_status = str(record.get("status", "") or "").strip().lower()
        if record_status in terminal_statuses:
            if record_status != "completed":
                _set_recoverable_step4_training_session(
                    record_status,
                    record,
                    reason="training_terminal_status_after_restart",
                )
                return record
            best_weights = str(record.get("best_weights", "") or "").strip()
            if best_weights:
                try:
                    if Path(best_weights).exists():
                        _close_terminal_step4_session(record_status)
                        return record
                except Exception:
                    _close_terminal_step4_session(record_status)
                    return record

        try:
            from .training import TrainingHistory, TrainingStatus

            history = TrainingHistory(history_dir=self.get_project_root_dir(project_name) / "5_training_runs")
            run = history.get_run(run_id)
        except Exception as exc:
            logger.debug(f"Nie udało się zsynchronizować historii treningu Z4: {exc}")
            return record
        if run is None:
            return record

        status_value = str(getattr(run, "status", "") or "").strip().lower()
        payload = dict(record)
        for key, value in {
            "run_id": run_id,
            "status": status_value,
            "dataset_path": str(getattr(run, "dataset_path", "") or "").strip(),
            "output_dir": str(getattr(run, "output_dir", "") or "").strip(),
            "best_weights": str(getattr(run, "best_weights", "") or "").strip(),
            "last_weights": str(getattr(run, "last_weights", "") or "").strip(),
            "current_epoch": int(getattr(run, "current_epoch", 0) or 0),
            "epochs": int(getattr(run, "epochs", 0) or 0),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }.items():
            if key in {"best_weights"} and status_value != TrainingStatus.COMPLETED.value:
                value = ""
            payload[key] = value
        payload.setdefault("iteration", iter_value)
        payload.setdefault("trained_iteration", iter_value)

        changed = payload != record
        if changed:
            try:
                self.upsert_iteration_state(
                    iteration_num=iter_value,
                    updates={"step4_training": payload},
                    project_name=project_name,
                )
            except Exception as exc:
                logger.debug(f"Nie udało się zapisać zsynchronizowanego runu Z4 w iteracji: {exc}")
            try:
                self.upsert_iteration_artifact_bundle(
                    iteration_num=iter_value,
                    updates={"step4_training": payload},
                    project_name=project_name,
                )
            except Exception as exc:
                logger.debug(f"Nie udało się zapisać zsynchronizowanego runu Z4 w paczce artefaktów: {exc}")

        if status_value == TrainingStatus.COMPLETED.value:
            _close_terminal_step4_session(status_value)
        elif status_value in recoverable_statuses:
            _set_recoverable_step4_training_session(
                status_value,
                payload,
                reason="training_requires_attention_after_history_sync",
            )

        return payload

    def _coerce_project_training_model_artifact(
        self,
        payload: Dict[str, Any] | None,
        *,
        target: str,
        iteration_num: int,
        project_name: str,
        source: str,
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        payload_target = self._normalize_project_model_target(payload.get("target"))
        if payload_target != target:
            return {}
        model_path = self._resolve_existing_model_artifact_path(payload)
        if not model_path:
            return {}
        status = str(payload.get("status", "") or "").strip().lower()
        if status and status not in {"completed", "success", "finished", "done", "ready"}:
            return {}

        run_id = str(payload.get("run_id", "") or "").strip()
        history_run = self._load_training_history_run(run_id, project_name=project_name)
        artifact_iteration = 0
        for field in ("trained_iteration", "source_iteration", "iteration"):
            try:
                artifact_iteration = int(payload.get(field, 0) or 0)
            except Exception:
                artifact_iteration = 0
            if artifact_iteration > 0:
                break
        if artifact_iteration <= 0:
            try:
                artifact_iteration = int(iteration_num or 0)
            except Exception:
                artifact_iteration = 0
        result = dict(payload)
        result.update(
            {
                "path": model_path,
                "target": target,
                "iteration": int(artifact_iteration or 0),
                "trained_iteration": int(artifact_iteration or 0),
                "source": source,
                "scope": "trained_previous_iteration",
            }
        )
        if history_run:
            for key in (
                "name",
                "base_model",
                "best_map50",
                "best_map50_95",
                "precision",
                "recall",
                "finished_at",
                "created_at",
            ):
                if key in history_run and key not in result:
                    result[key] = history_run.get(key)
            if not result.get("name"):
                result["name"] = str(history_run.get("name", "") or "").strip()
        return result

    def get_latest_trained_project_model(
        self,
        model_type: str = "plate",
        *,
        before_iteration: int | None = None,
        project_name: str = None,
    ) -> Dict[str, Any]:
        """Return the latest completed project-trained model for the requested target."""

        project_name = self._resolve_project_name(project_name)
        target = self._normalize_project_model_target(model_type)
        if not project_name or target not in {"plate", "char"}:
            return {}
        try:
            before_value = int(before_iteration or 0)
        except Exception:
            before_value = 0

        registry = self.load_artifact_registry(project_name)
        candidates: list[Dict[str, Any]] = []

        def _candidate_allowed_before_iteration(candidate: Dict[str, Any]) -> bool:
            if not before_value:
                return True
            for field in ("trained_iteration", "source_iteration", "iteration"):
                try:
                    value = int(candidate.get(field, 0) or 0)
                except Exception:
                    value = 0
                if value > 0:
                    return value < before_value
            return True

        iteration_state = registry.get("iteration_state", {})
        if isinstance(iteration_state, dict):
            for key, entry in iteration_state.items():
                if not isinstance(entry, dict):
                    continue
                try:
                    iter_value = int(entry.get("iteration", key) or key or 0)
                except Exception:
                    continue
                if before_value and iter_value >= before_value:
                    continue
                candidate = self._coerce_project_training_model_artifact(
                    entry.get("step4_training"),
                    target=target,
                    iteration_num=iter_value,
                    project_name=project_name,
                    source="iteration_state",
                )
                if candidate:
                    if _candidate_allowed_before_iteration(candidate):
                        candidates.append(candidate)

        packages = registry.get("packages", {})
        if isinstance(packages, dict):
            for package in packages.values():
                if not isinstance(package, dict):
                    continue
                iterations = []
                for raw_iter in package.get("iterations", []) or []:
                    try:
                        iterations.append(int(raw_iter))
                    except Exception:
                        pass
                if iterations:
                    iter_value = max(iterations)
                else:
                    try:
                        iter_value = int(package.get("iteration_last_seen", 0) or 0)
                    except Exception:
                        iter_value = 0
                if before_value:
                    previous_iters = [value for value in iterations if value < before_value]
                    if iterations and not previous_iters:
                        continue
                    if previous_iters:
                        iter_value = max(previous_iters)
                    elif iter_value >= before_value:
                        continue
                candidate = self._coerce_project_training_model_artifact(
                    package.get("step4_training"),
                    target=target,
                    iteration_num=iter_value,
                    project_name=project_name,
                    source="artifact_package",
                )
                if candidate:
                    if _candidate_allowed_before_iteration(candidate):
                        candidates.append(candidate)

        if not candidates:
            return {}

        deduped: Dict[str, Dict[str, Any]] = {}

        def _candidate_identity(item: Dict[str, Any]) -> str:
            run_id = str(item.get("run_id", "") or "").strip()
            if run_id:
                return f"run:{run_id}"
            path = str(item.get("path", "") or item.get("best_weights", "") or item.get("output_dir", "") or "").strip()
            if path:
                try:
                    path = str(Path(path).resolve())
                except Exception:
                    pass
                return f"path:{path.lower()}"
            return f"item:{id(item)}"

        def _candidate_iteration(item: Dict[str, Any]) -> int:
            try:
                return int(item.get("iteration", 0) or 0)
            except Exception:
                return 0

        def _prefer_candidate(candidate: Dict[str, Any], existing: Dict[str, Any]) -> bool:
            candidate_source = str(candidate.get("source", "") or "").strip()
            existing_source = str(existing.get("source", "") or "").strip()
            if candidate_source == "iteration_state" and existing_source != "iteration_state":
                return True
            if existing_source == "iteration_state" and candidate_source != "iteration_state":
                return False
            candidate_iter = _candidate_iteration(candidate)
            existing_iter = _candidate_iteration(existing)
            if candidate_iter > 0 and existing_iter > 0:
                return candidate_iter < existing_iter
            if candidate_iter > 0:
                return True
            return False

        for candidate in candidates:
            identity = _candidate_identity(candidate)
            existing = deduped.get(identity)
            if existing is None or _prefer_candidate(candidate, existing):
                deduped[identity] = candidate
        candidates = list(deduped.values())

        def _candidate_sort_key(item: Dict[str, Any]) -> tuple[int, str, str]:
            try:
                iter_value = int(item.get("iteration", 0) or 0)
            except Exception:
                iter_value = 0
            timestamp = str(item.get("finished_at") or item.get("updated_at") or item.get("created_at") or "")
            run_id = str(item.get("run_id", "") or "")
            return iter_value, timestamp, run_id

        return dict(max(candidates, key=_candidate_sort_key))

    def get_effective_project_model(
        self,
        model_type: str = "plate",
        *,
        before_iteration: int | None = None,
        project_name: str = None,
    ) -> Dict[str, Any]:
        """Prefer a model trained in earlier project iterations, then fall back to an explicit project model."""

        project_name = self._resolve_project_name(project_name)
        target = self._normalize_project_model_target(model_type)
        if not project_name or target not in {"plate", "char"}:
            return {}

        trained = self.get_latest_trained_project_model(
            target,
            before_iteration=before_iteration,
            project_name=project_name,
        )
        if trained:
            return trained

        explicit_path = str(self.get_global_model(target) or "").strip()
        if explicit_path:
            try:
                explicit = Path(explicit_path)
            except Exception:
                explicit = None
            if explicit is not None and explicit.exists() and explicit.is_file():
                try:
                    normalized_path = str(explicit.resolve())
                except Exception:
                    normalized_path = explicit_path
                return {
                    "path": normalized_path,
                    "target": target,
                    "iteration": 0,
                    "trained_iteration": 0,
                    "source_iteration": 0,
                    "source": "project_global_model",
                    "scope": "project",
                }
        return {}

    def get_active_project_root_dir(self) -> Path | None:
        act = self.get_active_project_name()
        if not act:
            return None
        return self.get_project_root_dir(act)

    def get_iteration_raw_dir(self, iteration_num: int = None, project_name: str = None) -> Path | None:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return None

        raw_dir = self.get_project_root_dir(project_name) / "1_raw_images"
        raw_dir.mkdir(parents=True, exist_ok=True)
        iter_value = int(iteration_num or self.state["projects"][project_name].get("current_iteration", 1))
        return raw_dir / f"Iteracja_{iter_value:03d}"

    def count_images_in_dir(self, path_like: str | Path | None, *, recursive: bool = False) -> int:
        if path_like is None:
            return 0
        try:
            return IMAGE_DIRECTORIES.snapshot(path_like, CONFIG.IMAGE_EXTENSIONS, recursive=recursive).count
        except (TypeError, ValueError, OSError):
            return 0

    def get_iteration_manifest_image_count(self, iteration_num: int = None, project_name: str = None) -> int:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return 0
        try:
            iter_value = int(iteration_num or self.state["projects"][project_name].get("current_iteration", 1) or 1)
        except Exception:
            iter_value = 1
        cache_key = (project_name, int(iter_value))
        cached_count = self._iteration_image_count_cache.get(cache_key)
        if isinstance(cached_count, int):
            return int(cached_count)
        try:
            summary = self.load_latest_ingest_plan_summary(project_name)
        except Exception:
            summary = {}
        if isinstance(summary, dict) and summary:
            try:
                summary_iter = int(summary.get("iteration", 0) or 0)
            except Exception:
                summary_iter = 0
            summary_project = str(summary.get("project", "") or "").strip()
            if summary_iter == iter_value and (not summary_project or summary_project == project_name):
                for key in ("selected_total", "raw_total", "source_new_to_project_total"):
                    try:
                        count = int(summary.get(key, 0) or 0)
                    except Exception:
                        count = 0
                    if count > 0:
                        self._iteration_image_count_cache[cache_key] = int(count)
                        return int(count)
        try:
            manifest = self.load_ingest_manifest(iter_value, project_name)
        except Exception:
            manifest = {}
        if not isinstance(manifest, dict):
            return 0
        try:
            selected_images = list(manifest.get("selected_images", []) or [])
        except Exception:
            selected_images = []
        if selected_images:
            return int(len(selected_images))
        try:
            return max(0, int(manifest.get("selected_count", 0) or 0))
        except Exception:
            return 0

    def get_iteration_manifest_image_paths(
        self,
        iteration_num: int = None,
        project_name: str = None,
        *,
        base_dir: str | Path | None = None,
    ) -> List[Path]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return []

        try:
            iter_value = int(iteration_num or self.state["projects"][project_name].get("current_iteration", 1) or 1)
        except Exception:
            iter_value = 1

        try:
            manifest = self.load_ingest_manifest(iter_value, project_name)
        except Exception:
            manifest = {}
        if not isinstance(manifest, dict):
            return []

        selected_items = list(manifest.get("selected_images") or [])
        if not selected_items:
            return []

        candidate_roots: list[Path] = []
        for root_candidate in (
            base_dir,
            manifest.get("source_dir"),
            manifest.get("master_pool_dir"),
            manifest.get("target_dir"),
            self.get_iteration_raw_dir(iter_value, project_name),
        ):
            if not root_candidate:
                continue
            try:
                root_path = Path(root_candidate)
            except Exception:
                continue
            try:
                if root_path.exists() and root_path.is_dir():
                    candidate_roots.append(root_path)
            except Exception:
                continue

        image_paths: list[Path] = []
        seen_names: set[str] = set()
        for item in selected_items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            candidates: list[Path] = []
            for key in ("target_path", "source_path", "iteration_target_path"):
                raw_path = str(item.get(key, "") or "").strip()
                if not raw_path:
                    continue
                try:
                    candidates.append(Path(raw_path))
                except Exception:
                    pass
            if name:
                for root_path in candidate_roots:
                    candidates.append(root_path / name)

            resolved_path = None
            for candidate in candidates:
                try:
                    if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in CONFIG.IMAGE_EXTENSIONS:
                        resolved_path = candidate
                        break
                except Exception:
                    continue
            if resolved_path is None:
                continue

            safe_name = str(resolved_path.name or name or "").strip().lower()
            if not safe_name or safe_name in seen_names:
                continue
            seen_names.add(safe_name)
            image_paths.append(resolved_path)

        return image_paths

    def get_iteration_image_count(self, iteration_num: int = None, project_name: str = None) -> int:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return 0
        try:
            iter_value = int(iteration_num or self.state["projects"][project_name].get("current_iteration", 1) or 1)
        except Exception:
            iter_value = 1
        cache_key = (project_name, int(iter_value))
        cached_count = self._iteration_image_count_cache.get(cache_key)
        if isinstance(cached_count, int):
            return int(cached_count)
        try:
            summary = self.load_latest_ingest_plan_summary(project_name)
        except Exception:
            summary = {}
        if isinstance(summary, dict) and summary:
            try:
                summary_iter = int(summary.get("iteration", 0) or 0)
            except Exception:
                summary_iter = 0
            summary_project = str(summary.get("project", "") or "").strip()
            if summary_iter == iter_value and (not summary_project or summary_project == project_name):
                for key in ("selected_total", "raw_total", "source_new_to_project_total"):
                    try:
                        count = int(summary.get(key, 0) or 0)
                    except Exception:
                        count = 0
                    if count > 0:
                        self._iteration_image_count_cache[cache_key] = int(count)
                        return int(count)
        try:
            manifest = self.load_ingest_manifest(iter_value, project_name)
        except Exception:
            manifest = {}
        manifest_mode = str((manifest or {}).get("selection_mode", "") or "").strip().lower() if isinstance(manifest, dict) else ""
        manifest_count = int(self.get_iteration_manifest_image_count(iter_value, project_name) or 0)
        if manifest_count > 0 and (
            bool((manifest or {}).get("manifest_only", False))
            or manifest_mode in {"planned", "planned_manifest", "source_reuse", "pool_reuse", "stage_reuse", "iteration_reuse"}
        ):
            self._iteration_image_count_cache[cache_key] = int(manifest_count)
            return int(manifest_count)

        raw_dir = self.get_iteration_raw_dir(iter_value, project_name)
        physical_count = self.count_images_in_dir(raw_dir, recursive=False)
        if physical_count > 0:
            self._iteration_image_count_cache[cache_key] = int(physical_count)
            return int(physical_count)
        self._iteration_image_count_cache[cache_key] = int(manifest_count)
        return int(manifest_count)

    def get_iteration_image_source_dir(
        self,
        iteration_num: int = None,
        project_name: str = None,
    ) -> Path | None:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return None
        iter_value = int(iteration_num or self.state["projects"][project_name].get("current_iteration", 1))
        cache_key = (project_name, int(iter_value))
        cached_source = self._iteration_image_source_dir_cache.get(cache_key)
        if isinstance(cached_source, str) and cached_source:
            try:
                return Path(cached_source)
            except Exception:
                pass

        raw_dir = self.get_iteration_raw_dir(iter_value, project_name)
        try:
            summary = self.load_latest_ingest_plan_summary(project_name)
        except Exception:
            summary = {}
        if isinstance(summary, dict) and summary:
            try:
                summary_iter = int(summary.get("iteration", 0) or 0)
            except Exception:
                summary_iter = 0
            summary_project = str(summary.get("project", "") or "").strip()
            if summary_iter == iter_value and (not summary_project or summary_project == project_name):
                source_text = str(
                    summary.get("master_pool_dir")
                    or summary.get("source_dir")
                    or summary.get("target_dir")
                    or ""
                ).strip()
                if source_text:
                    try:
                        source_path = Path(source_text)
                        if source_path.exists() and source_path.is_dir():
                            self._iteration_image_source_dir_cache[cache_key] = str(source_path)
                            return source_path
                    except Exception:
                        pass
        try:
            manifest = self.load_ingest_manifest(iter_value, project_name)
        except Exception:
            manifest = {}
        manifest_mode = str((manifest or {}).get("selection_mode", "") or "").strip().lower() if isinstance(manifest, dict) else ""
        if manifest_mode in {"planned", "planned_manifest", "source_reuse", "pool_reuse", "stage_reuse", "iteration_reuse"}:
            for key in ("source_dir", "master_pool_dir", "target_dir"):
                source = self._resolve_valid_image_source_dir(
                    manifest.get(key, "") if isinstance(manifest, dict) else "",
                    project_name=project_name,
                    recursive=(key != "target_dir"),
                    reject_broad=(key != "target_dir"),
                )
                if source:
                    self._iteration_image_source_dir_cache[cache_key] = str(Path(source))
                    return Path(source)

        if self.count_images_in_dir(raw_dir, recursive=False) > 0:
            self._iteration_image_source_dir_cache[cache_key] = str(raw_dir)
            return raw_dir

        if isinstance(manifest, dict):
            for key in ("target_dir", "source_dir", "master_pool_dir"):
                source = self._resolve_valid_image_source_dir(
                    manifest.get(key, ""),
                    project_name=project_name,
                    recursive=(key != "target_dir"),
                    reject_broad=(key != "target_dir"),
                )
                if source:
                    self._iteration_image_source_dir_cache[cache_key] = str(Path(source))
                    return Path(source)

        master_pool = self._resolve_valid_image_source_dir(
            self.state["projects"][project_name].get("master_pool_dir", ""),
            project_name=project_name,
        )
        if master_pool:
            self._iteration_image_source_dir_cache[cache_key] = str(Path(master_pool))
            return Path(master_pool)
        self._iteration_image_source_dir_cache[cache_key] = str(raw_dir)
        return raw_dir

    def get_dir(self, key: str) -> Path | None:
        """Zwraca katalog dla aktywnego projektu."""
        root = self.get_active_project_root_dir()
        if root is None:
            return None

        mapping = {
            "raw": root / "1_raw_images",
            "auto_ann": root / "2_auto_annotations",
            "chars": root / "3_cropped_characters",
            "datasets": root / "4_training_datasets",
            "runs": root / "5_training_runs",
            "models": root / "6_models",
            "rankings": root / "7_rankings",
            "presets": root / "8_presets",
            "ocr_presets": root / "8_presets" / "ocr",
            "detection_pipeline_presets": root / "8_presets" / "detection_pipeline",
            "augmentation_presets": root / "8_presets" / "augmentation",
            "training_presets": root / "8_presets" / "training",
            "legacy_ocr_presets": root / "8_ocr_presets",
        }
        return mapping.get(key) 

    def get_staging_dir(self, key: str):
        """
        Zwraca katalog tymczasowy aktywnego projektu.
        Obecnie wykorzystywany jest staging dla autoanotacji.
        """
        root = self.get_active_project_root_dir()
        if root is None:
            return None

        staging_root = root / "_staging"
        mapping = {
            "auto_ann": staging_root / "auto_annotations",
            "plate_stage": staging_root / "plate_manual_stage",
        }
        return mapping.get(key)   

    # --- INGESTIA / MASTER POOL ---

    def get_master_pool_dir(self, project_name: str = None) -> Path | None:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return None

        path_value = str(self.state["projects"][project_name].get("master_pool_dir", "") or "").strip()
        if not path_value:
            return None
        return Path(path_value)

    def set_master_pool_dir(self, path: str | Path, project_name: str = None) -> bool:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return False

        target = Path(path).expanduser()
        resolved_target = self._resolve_valid_image_source_dir(
            target,
            project_name=project_name,
        )
        if not resolved_target:
            return False
        project_data = self.state["projects"][project_name]
        try:
            self.ensure_e1_resource_contract_baseline(project_name=project_name)
        except Exception:
            pass
        project_data["master_pool_dir"] = str(resolved_target)
        try:
            project_data["master_pool_selected_iteration"] = int(project_data.get("current_iteration", 1) or 1)
        except Exception:
            project_data["master_pool_selected_iteration"] = 1
        project_data["step1_source_manual_clear_iteration"] = 0
        project_data["step1_restored_image_source_dir"] = ""
        try:
            self._latest_ingest_plan_summary_cache.clear()
            self._latest_ingest_plan_summary_runtime_cache.clear()
            self._iteration_image_count_cache.clear()
            self._iteration_image_source_dir_cache.clear()
        except Exception:
            pass
        self.save_state()
        return True

    def get_master_pool_selected_iteration(self, project_name: str = None) -> int:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return 0
        try:
            return int(self.state["projects"][project_name].get("master_pool_selected_iteration", 0) or 0)
        except Exception:
            return 0

    def clear_master_pool_dir(self, project_name: str = None) -> bool:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return False
        project_data = self.state["projects"][project_name]
        try:
            self.ensure_e1_resource_contract_baseline(project_name=project_name)
        except Exception:
            pass
        project_data["master_pool_dir"] = ""
        project_data["master_pool_selected_iteration"] = 0
        project_data["step1_restored_image_source_dir"] = ""
        try:
            project_data["step1_source_manual_clear_iteration"] = int(project_data.get("current_iteration", 1) or 1)
        except Exception:
            project_data["step1_source_manual_clear_iteration"] = 0
        try:
            self._latest_ingest_plan_summary_cache.clear()
            self._latest_ingest_plan_summary_runtime_cache.clear()
            self._iteration_image_count_cache.clear()
            self._iteration_image_source_dir_cache.clear()
        except Exception:
            pass
        self.save_state()
        return True

    def get_ingest_batch_size(self, project_name: str = None) -> int:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return 200
        try:
            value = int(self.state["projects"][project_name].get("ingest_batch_size", 200))
        except Exception:
            value = 200
        return max(1, value)

    def set_ingest_batch_size(self, batch_size: int, project_name: str = None) -> bool:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return False
        try:
            value = max(1, int(batch_size))
        except Exception:
            value = 200
        self.state["projects"][project_name]["ingest_batch_size"] = value
        self.save_state()
        return True

    def get_ingest_manifest_path(self, iteration_num: int = None, project_name: str = None) -> Path | None:
        ingest_dir = self.get_project_ingest_state_dir(project_name)
        if ingest_dir is None:
            return None
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return None
        iter_value = int(iteration_num or self.state["projects"][project_name].get("current_iteration", 1))
        return ingest_dir / f"iter_{iter_value:03d}_manifest.json"

    def get_ingest_balance_snapshot_path(self, project_name: str = None) -> Path | None:
        ingest_dir = self.get_project_ingest_state_dir(project_name)
        if ingest_dir is None:
            return None
        return ingest_dir / "balance_snapshot.json"

    def get_latest_ingest_plan_path(self, project_name: str = None) -> Path | None:
        ingest_dir = self.get_project_ingest_state_dir(project_name)
        if ingest_dir is None:
            return None
        return ingest_dir / "latest_plan.json"

    def get_latest_ingest_plan_summary_path(self, project_name: str = None) -> Path | None:
        ingest_dir = self.get_project_ingest_state_dir(project_name)
        if ingest_dir is None:
            return None
        return ingest_dir / "latest_plan_summary.json"

    def _build_latest_ingest_plan_summary(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            return {}

        count_keys = (
            "ok",
            "planner_version",
            "generated_at",
            "project",
            "iteration",
            "master_pool_dir",
            "source_dir",
            "target_dir",
            "batch_size",
            "raw_total",
            "candidates_total",
            "selected_total",
            "new_to_project_total",
            "source_new_to_project_total",
            "skipped_used",
            "skipped_duplicate_filenames",
            "skipped_duplicate_approved_filenames",
            "project_overlap_filenames",
            "pending_iteration_overlap_filenames",
            "project_pool_total_before_iteration",
            "project_pool_total_after_iteration",
            "skipped_invalid_ground_truth",
            "source_reuse",
            "selection_mode",
        )
        summary: Dict[str, Any] = {
            key: plan.get(key)
            for key in count_keys
            if key in plan
        }
        for key in ("current_balance", "selected_balance", "predicted_balance_after"):
            value = plan.get(key)
            if isinstance(value, dict):
                summary[key] = dict(value)
        selected = plan.get("selected")
        if isinstance(selected, list):
            summary["selected_total"] = int(plan.get("selected_total", len(selected)) or len(selected))
            summary["selected_sample"] = [
                {
                    "name": str(item.get("name", "") or ""),
                    "source_key": str(item.get("source_key", "") or ""),
                }
                for item in selected[:5]
                if isinstance(item, dict)
            ]
        summary["summary_only"] = True
        return summary

    def load_latest_ingest_plan_summary(self, project_name: str = None) -> Dict[str, Any]:
        resolved_project = str(self._resolve_project_name(project_name) or "").strip()
        if resolved_project:
            runtime_cached = self._latest_ingest_plan_summary_runtime_cache.get(resolved_project)
            if isinstance(runtime_cached, dict):
                return dict(runtime_cached)
        summary_path = self.get_latest_ingest_plan_summary_path(resolved_project or project_name)
        if summary_path is None or not summary_path.exists():
            return {}
        cache_key = None
        try:
            stat = summary_path.stat()
            cache_key = (
                resolved_project,
                int(getattr(stat, "st_mtime_ns", 0) or 0),
                int(getattr(stat, "st_size", 0) or 0),
            )
        except Exception:
            cache_key = None
        if cache_key is not None:
            cached = self._latest_ingest_plan_summary_cache.get(cache_key)
            if isinstance(cached, dict):
                return dict(cached)
        summary = self._read_json_file(summary_path)
        result = summary if isinstance(summary, dict) else {}
        if cache_key is not None:
            if len(self._latest_ingest_plan_summary_cache) > 32:
                self._latest_ingest_plan_summary_cache.clear()
            self._latest_ingest_plan_summary_cache[cache_key] = dict(result)
        if resolved_project:
            self._latest_ingest_plan_summary_runtime_cache[resolved_project] = dict(result)
        return result

    def load_latest_ingest_plan(self, project_name: str = None) -> Dict[str, Any]:
        plan_path = self.get_latest_ingest_plan_path(project_name)
        if plan_path is None or not plan_path.exists():
            return {}
        return self._read_json_file(plan_path)

    def save_latest_ingest_plan(self, plan: Dict[str, Any], project_name: str = None) -> Path | None:
        plan_path = self.get_latest_ingest_plan_path(project_name)
        if plan_path is None:
            return None
        if self._write_json_file(plan_path, plan):
            try:
                self._latest_ingest_plan_summary_cache.clear()
                self._latest_ingest_plan_summary_runtime_cache.clear()
                self._iteration_image_count_cache.clear()
                self._iteration_image_source_dir_cache.clear()
            except Exception:
                pass
            summary_path = self.get_latest_ingest_plan_summary_path(project_name)
            if summary_path is not None:
                try:
                    self._write_json_file(summary_path, self._build_latest_ingest_plan_summary(plan))
                except Exception:
                    pass
            return plan_path
        return None

    def clear_latest_ingest_plan(self, project_name: str = None) -> bool:
        plan_path = self.get_latest_ingest_plan_path(project_name)
        if plan_path is None:
            return False
        try:
            if plan_path.exists():
                plan_path.unlink()
            summary_path = self.get_latest_ingest_plan_summary_path(project_name)
            if summary_path is not None and summary_path.exists():
                summary_path.unlink()
            try:
                self._latest_ingest_plan_summary_cache.clear()
                self._latest_ingest_plan_summary_runtime_cache.clear()
                self._iteration_image_count_cache.clear()
                self._iteration_image_source_dir_cache.clear()
            except Exception:
                pass
            return True
        except Exception:
            return False

    def get_plate_approved_set_path(self, project_name: str = None) -> Path | None:
        state_dir = self.get_project_state_dir(project_name)
        if state_dir is None:
            return None
        return state_dir / "plate_approved_set.json"

    def load_plate_approved_set(self, project_name: str = None) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {}

        runtime_cached = self._plate_approved_manifest_runtime_cache.get(project_name)
        if isinstance(runtime_cached, dict):
            entries = runtime_cached.get("entries", {})
            return {
                "project": str(runtime_cached.get("project", "") or project_name).strip(),
                "updated_at": str(runtime_cached.get("updated_at", "") or "").strip(),
                "entries": dict(entries) if isinstance(entries, dict) else {},
            }

        manifest_path = self.get_plate_approved_set_path(project_name)
        if manifest_path is None or not manifest_path.exists():
            return {
                "project": project_name,
                "updated_at": "",
                "entries": {},
            }

        payload = self._read_json_file(manifest_path)
        if not isinstance(payload, dict):
            payload = {}

        entries = payload.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}

        result = {
            "project": project_name,
            "updated_at": str(payload.get("updated_at", "") or "").strip(),
            "entries": entries,
        }
        self._plate_approved_manifest_runtime_cache[project_name] = {
            "project": result["project"],
            "updated_at": result["updated_at"],
            "entries": dict(entries),
        }
        return result

    def list_plate_approved_entries(self, project_name: str = None) -> List[Dict[str, Any]]:
        manifest = self.load_plate_approved_set(project_name)
        entries = manifest.get("entries", {})
        if not isinstance(entries, dict):
            return []

        result: List[Dict[str, Any]] = []
        for entry_key, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            normalized = dict(entry)
            normalized.setdefault("entry_key", str(entry_key or "").strip())
            result.append(normalized)

        result.sort(
            key=lambda item: (
                str(item.get("approved_at", "") or "").strip(),
                str(item.get("image_name", "") or "").strip().lower(),
            )
        )
        return result

    @staticmethod
    def _load_annotation_run_manifest_file(run_dir: Path | None) -> Dict[str, Any]:
        try:
            safe_run_dir = Path(run_dir) if run_dir is not None else None
        except Exception:
            safe_run_dir = None
        if safe_run_dir is None:
            return {}
        manifest_path = safe_run_dir / "run_manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _load_preview_metadata_source_state(preview_dir: Path | None) -> Dict[str, Any]:
        try:
            safe_preview_dir = Path(preview_dir) if preview_dir is not None else None
        except Exception:
            safe_preview_dir = None
        if safe_preview_dir is None:
            return {}

        meta_path = safe_preview_dir / "metadata.json"
        images_dir = safe_preview_dir / "images"
        if not meta_path.exists() or not images_dir.exists():
            return {}

        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(loaded, dict) or not loaded:
            return {}

        total_plates = 0
        source_names: set[str] = set()
        plates_by_source: dict[str, int] = {}
        for pid, payload in loaded.items():
            pid_text = str(pid or "").strip()
            if not pid_text:
                continue
            total_plates += 1
            source_key = ""
            if isinstance(payload, dict):
                source_info = payload.get("source_info") or {}
                if not isinstance(source_info, dict):
                    source_info = {}
                source_key = str(
                    payload.get("source_image")
                    or payload.get("source_name")
                    or source_info.get("image_name")
                    or pid_text
                ).strip()
            if not source_key:
                source_key = pid_text
            normalized_source = CampaignManager._normalize_image_set_name(source_key)
            if not normalized_source:
                normalized_source = str(source_key or pid_text).strip().lower()
            if not normalized_source:
                continue
            source_names.add(normalized_source)
            plates_by_source[normalized_source] = int(plates_by_source.get(normalized_source, 0) or 0) + 1

        images_with_plates = int(len(source_names) or total_plates or 0)
        total_plates = int(total_plates or 0)
        if total_plates <= 0:
            return {}

        return {
            "source_scope": "step3_preview",
            "run_dir": str(safe_preview_dir.resolve()) if safe_preview_dir.exists() else str(safe_preview_dir),
            "run_name": str(safe_preview_dir.name or "").strip(),
            "images_with_plates": images_with_plates,
            "total_plates": total_plates,
            "source_names": set(source_names),
            "plates_by_source": dict(plates_by_source),
        }

    @staticmethod
    def _load_run_plate_counts_by_image(run_dir: Path | None, *, image_names: set[str] | None = None) -> Dict[str, int]:
        try:
            safe_run_dir = Path(run_dir) if run_dir is not None else None
        except Exception:
            safe_run_dir = None
        if safe_run_dir is None:
            return {}
        xml_path = safe_run_dir / "annotations.xml"
        if not xml_path.exists():
            return {}

        wanted_names = {
            CampaignManager._normalize_image_set_name(name)
            for name in set(image_names or set())
            if CampaignManager._normalize_image_set_name(name)
        }
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            return {}

        counts: dict[str, int] = {}
        for image_node in root.findall(".//image"):
            image_name = CampaignManager._normalize_image_set_name(image_node.get("name", ""))
            if not image_name:
                continue
            if wanted_names and image_name not in wanted_names:
                continue
            plate_count = 0
            for tag_name in ("polygon", "box"):
                for det_node in image_node.findall(tag_name):
                    if str(det_node.get("label", "") or "").strip().lower() == "plate":
                        plate_count += 1
            if plate_count <= 0:
                continue
            counts[image_name] = int(plate_count)
        return counts

    def get_step3_char_source_state(
        self,
        *,
        iteration_num: int | None = None,
        project_name: str = None,
    ) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {}

        iter_value = int(iteration_num or self.state["projects"][project_name].get("current_iteration", 1) or 1)

        runtime_key = (project_name, int(iter_value))
        runtime_cached = self._step3_char_source_state_runtime_cache.get(runtime_key)
        if isinstance(runtime_cached, dict):
            return dict(runtime_cached)

        try:
            bundle = dict(self.get_iteration_artifact_bundle(iteration_num=iter_value, project_name=project_name) or {})
        except Exception:
            bundle = {}

        preview_entry = dict(bundle.get("step3_preview_source") or {})
        preview_dir_raw = str(preview_entry.get("preview_dir") or "").strip()
        if not preview_dir_raw:
            try:
                preview_dir_raw = str(self.state["projects"][project_name].get("step3_preview_dir", "") or "").strip()
            except Exception:
                preview_dir_raw = ""

        step2_entry = dict(bundle.get("step2_active_run") or bundle.get("plate_source") or {})
        step2_run_raw = str(step2_entry.get("run_dir") or "").strip()
        step2_run_dir = Path(step2_run_raw) if step2_run_raw else None

        def _fast_path_token(path_like) -> str:
            raw_value = str(path_like or "").strip()
            if not raw_value:
                return ""
            try:
                path = Path(raw_value)
                stat = path.stat()
                return (
                    f"{raw_value}|"
                    f"{int(getattr(stat, 'st_mtime_ns', 0) or 0)}|"
                    f"{int(getattr(stat, 'st_size', 0) or 0)}"
                )
            except Exception:
                return raw_value

        try:
            preview_meta_path = Path(preview_dir_raw) / "metadata.json" if preview_dir_raw else None
        except Exception:
            preview_meta_path = None
        step2_manifest_path = step2_run_dir / "run_manifest.json" if step2_run_dir is not None else None
        step2_xml_path = step2_run_dir / "annotations.xml" if step2_run_dir is not None else None
        approved_path = self.get_plate_approved_set_path(project_name)
        registry_path = self.get_artifact_registry_path(project_name)
        cache_key = (
            project_name,
            int(iter_value),
            _fast_path_token(registry_path),
            _fast_path_token(approved_path),
            _fast_path_token(preview_meta_path),
            _fast_path_token(step2_manifest_path),
            _fast_path_token(step2_xml_path),
        )
        cached_state = self._step3_char_source_state_cache.get(cache_key)
        if isinstance(cached_state, dict):
            self._step3_char_source_state_runtime_cache[runtime_key] = dict(cached_state)
            return dict(cached_state)

        preview_state = self._load_preview_metadata_source_state(Path(preview_dir_raw) if preview_dir_raw else None)
        step2_manifest = self._load_annotation_run_manifest_file(step2_run_dir)
        approved_names = {
            self._normalize_image_set_name(name)
            for name in list(step2_manifest.get("approved_filenames") or [])
            if self._normalize_image_set_name(name)
        }

        step2_counts = self._load_run_plate_counts_by_image(step2_run_dir, image_names=approved_names)

        approved_entries = list(self.list_plate_approved_entries(project_name) or [])
        project_approved_names = {
            self._normalize_image_set_name(entry.get("image_name", ""))
            for entry in approved_entries
            if isinstance(entry, dict) and self._normalize_image_set_name(entry.get("image_name", ""))
        }
        project_approved_plates_by_source: dict[str, int] = {}
        for entry in approved_entries:
            if not isinstance(entry, dict):
                continue
            safe_name = self._normalize_image_set_name(entry.get("image_name", ""))
            if not safe_name:
                continue
            valid_plate_count = 0
            for plate_entry in list(entry.get("plates") or []):
                if not isinstance(plate_entry, dict):
                    continue
                polygon = list(plate_entry.get("polygon") or [])
                if len(polygon) >= 4:
                    valid_plate_count += 1
            if valid_plate_count <= 0:
                valid_plate_count = int(entry.get("plate_count", 0) or 0)
            if valid_plate_count <= 0:
                continue
            project_approved_plates_by_source[safe_name] = int(valid_plate_count)

        union_source_names = {
            str(name or "").strip().lower()
            for name in set(project_approved_plates_by_source.keys())
            if str(name or "").strip()
        }
        union_plates_by_source = {
            str(name or "").strip().lower(): int(count or 0)
            for name, count in dict(project_approved_plates_by_source).items()
            if str(name or "").strip() and int(count or 0) > 0
        }

        for name in set(preview_state.get("source_names") or set()):
            safe_name = str(name or "").strip().lower()
            if safe_name:
                union_source_names.add(safe_name)
        for name, count in dict(preview_state.get("plates_by_source") or {}).items():
            safe_name = str(name or "").strip().lower()
            plate_count = int(count or 0)
            if safe_name and plate_count > 0:
                union_source_names.add(safe_name)
                union_plates_by_source[safe_name] = max(
                    int(union_plates_by_source.get(safe_name, 0) or 0),
                    plate_count,
                )

        pending_added_names: set[str] = set()
        pending_added_plates_by_source: dict[str, int] = {}
        for image_name, plate_count in dict(step2_counts or {}).items():
            safe_name = str(image_name or "").strip().lower()
            if not safe_name or plate_count <= 0:
                continue
            if safe_name in project_approved_names:
                continue
            if safe_name in union_source_names:
                continue
            union_source_names.add(safe_name)
            union_plates_by_source[safe_name] = int(plate_count)
            pending_added_names.add(safe_name)
            pending_added_plates_by_source[safe_name] = int(plate_count)

        if not union_source_names and not union_plates_by_source:
            if len(self._step3_char_source_state_cache) > 32:
                self._step3_char_source_state_cache.clear()
            self._step3_char_source_state_cache[cache_key] = {}
            self._step3_char_source_state_runtime_cache[runtime_key] = {}
            return {}

        union_project_names = set(union_source_names) & set(project_approved_names)
        union_project_plates = int(
            sum(int(project_approved_plates_by_source.get(name, 0) or 0) for name in union_project_names)
        )
        union_total_plates = int(sum(int(count or 0) for count in union_plates_by_source.values()))
        pending_images = int(len(pending_added_names))
        pending_plates = int(sum(int(count or 0) for count in pending_added_plates_by_source.values()))
        current_images = max(0, int(len(union_source_names)) - int(len(union_project_names)))
        current_plates = max(0, int(union_total_plates) - int(union_project_plates))

        min_char_route_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_CHAR_PLATES", 10) or 10)
        result = {
            "source_scope": (
                "step3_pending_union"
                if pending_images > 0
                else (
                    "campaign_approved_set"
                    if union_project_names
                    else str(preview_state.get("source_scope") or "step3_preview")
                )
            ),
            "images_with_plates": int(len(union_source_names)),
            "total_plates": int(union_total_plates),
            "project_images_with_plates": int(len(union_project_names)),
            "project_total_plates": int(union_project_plates),
            "current_images_with_plates": int(current_images),
            "current_total_plates": int(current_plates),
            "preview_images_with_plates": int(preview_state.get("images_with_plates", 0) or 0),
            "preview_total_plates": int(preview_state.get("total_plates", 0) or 0),
            "pending_images_with_plates": int(pending_images),
            "pending_total_plates": int(pending_plates),
            "source_names": set(union_source_names),
            "preview_source_names": set(preview_state.get("source_names") or set()),
            "pending_source_names": set(pending_added_names),
            "run_name": str(preview_state.get("run_name") or "").strip(),
            "run_dir": str(preview_state.get("run_dir") or "").strip(),
            "ready": bool(int(union_total_plates) >= int(min_char_route_plates)),
            "has_source": bool(int(union_total_plates) > 0),
            "needs_more_tables": bool(
                int(union_total_plates) > 0
                and int(union_total_plates) < int(min_char_route_plates)
            ),
        }
        if len(self._step3_char_source_state_cache) > 32:
            self._step3_char_source_state_cache.clear()
        self._step3_char_source_state_cache[cache_key] = dict(result)
        self._step3_char_source_state_runtime_cache[runtime_key] = dict(result)
        return result

    def invalidate_step3_char_source_state_cache(self) -> None:
        try:
            self._step3_char_source_state_cache.clear()
            self._step3_char_source_state_runtime_cache.clear()
        except Exception:
            pass

    def get_plate_approved_set_stats(self, project_name: str = None) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {
                "project": "",
                "updated_at": "",
                "images": 0,
                "plates": 0,
                "manual_images": 0,
                "manual_plates": 0,
                "auto_accepted_images": 0,
                "auto_accepted_plates": 0,
            }

        runtime_cached = self._plate_approved_stats_runtime_cache.get(project_name)
        if isinstance(runtime_cached, dict):
            return dict(runtime_cached)

        cache_key = None
        manifest_path = self.get_plate_approved_set_path(project_name)
        if manifest_path is not None and manifest_path.exists():
            try:
                stat = manifest_path.stat()
                cache_key = (project_name, int(getattr(stat, "st_mtime_ns", 0) or 0), int(getattr(stat, "st_size", 0) or 0))
            except Exception:
                cache_key = None
        if cache_key is not None:
            cached = self._plate_approved_stats_cache.get(cache_key)
            if isinstance(cached, dict):
                return dict(cached)

        manifest = self.load_plate_approved_set(project_name)
        entries = manifest.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}

        stats = {
            "project": str(manifest.get("project", "") or "").strip(),
            "updated_at": str(manifest.get("updated_at", "") or "").strip(),
            "images": 0,
            "plates": 0,
            "manual_images": 0,
            "manual_plates": 0,
            "auto_accepted_images": 0,
            "auto_accepted_plates": 0,
        }

        for entry in entries.values():
            if not isinstance(entry, dict):
                continue

            valid_plate_count = 0
            for plate_entry in list(entry.get("plates") or []):
                if not isinstance(plate_entry, dict):
                    continue
                polygon = list(plate_entry.get("polygon") or [])
                if len(polygon) >= 4:
                    valid_plate_count += 1

            if valid_plate_count <= 0:
                valid_plate_count = int(entry.get("plate_count", 0) or 0)
            if valid_plate_count <= 0:
                continue

            stats["images"] += 1
            stats["plates"] += int(valid_plate_count)

            origin = str(entry.get("annotation_origin", "") or "").strip().lower()
            if origin == "auto_accepted":
                stats["auto_accepted_images"] += 1
                stats["auto_accepted_plates"] += int(valid_plate_count)
            else:
                stats["manual_images"] += 1
                stats["manual_plates"] += int(valid_plate_count)

        if cache_key is not None:
            self._plate_approved_stats_cache[cache_key] = dict(stats)
        self._plate_approved_stats_runtime_cache[project_name] = dict(stats)

        return stats

    def get_plate_approved_set_iteration_stats(
        self,
        iteration_num: int = None,
        project_name: str = None,
    ) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {
                "project": "",
                "iteration": int(iteration_num or 0),
                "images": 0,
                "plates": 0,
                "image_names": [],
            }

        try:
            iter_value = int(
                iteration_num
                or self.state["projects"][project_name].get("current_iteration", 1)
                or 1
            )
        except Exception:
            iter_value = 1

        stats = {
            "project": project_name,
            "iteration": int(iter_value),
            "images": 0,
            "plates": 0,
            "image_names": [],
        }
        image_names: set[str] = set()

        for entry in self.list_plate_approved_entries(project_name):
            if not isinstance(entry, dict):
                continue
            raw_iteration = (
                entry.get("first_approved_iteration")
                or entry.get("approved_iteration")
                or 0
            )
            try:
                entry_iteration = int(raw_iteration or 0)
            except Exception:
                entry_iteration = 0
            if entry_iteration != int(iter_value):
                continue

            valid_plate_count = 0
            for plate_entry in list(entry.get("plates") or []):
                if not isinstance(plate_entry, dict):
                    continue
                polygon = list(plate_entry.get("polygon") or [])
                if len(polygon) >= 4:
                    valid_plate_count += 1
            if valid_plate_count <= 0:
                valid_plate_count = int(entry.get("plate_count", 0) or 0)
            if valid_plate_count <= 0:
                continue

            stats["images"] += 1
            stats["plates"] += int(valid_plate_count)
            image_name = str(entry.get("image_name", "") or "").strip()
            if image_name:
                image_names.add(image_name)

        stats["image_names"] = sorted(image_names)
        return stats

    def upsert_plate_approved_entries(
        self,
        entries: List[Dict[str, Any]],
        project_name: str = None,
    ) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {"ok": False, "reason": "missing_project"}

        manifest_path = self.get_plate_approved_set_path(project_name)
        if manifest_path is None:
            return {"ok": False, "reason": "missing_manifest_path"}

        manifest = self.load_plate_approved_set(project_name)
        stored_entries = manifest.get("entries", {})
        if not isinstance(stored_entries, dict):
            stored_entries = {}

        added = 0
        updated = 0
        for raw_entry in list(entries or []):
            if not isinstance(raw_entry, dict):
                continue

            image_name = str(raw_entry.get("image_name", "") or "").strip()
            entry_key = str(raw_entry.get("entry_key", "") or "").strip().lower() or image_name.lower()
            if not image_name or not entry_key:
                continue

            normalized = dict(raw_entry)
            normalized["entry_key"] = entry_key
            normalized["image_name"] = image_name

            existing_entry = stored_entries.get(entry_key)
            if not isinstance(existing_entry, dict):
                existing_entry = {}

            first_iteration = (
                existing_entry.get("first_approved_iteration")
                or existing_entry.get("approved_iteration")
                or normalized.get("first_approved_iteration")
                or normalized.get("approved_iteration")
            )
            try:
                first_iteration = int(first_iteration or 0)
            except Exception:
                first_iteration = 0
            if first_iteration > 0:
                normalized["first_approved_iteration"] = int(first_iteration)

            first_approved_at = (
                str(existing_entry.get("first_approved_at", "") or "").strip()
                or str(existing_entry.get("approved_at", "") or "").strip()
                or str(normalized.get("first_approved_at", "") or "").strip()
                or str(normalized.get("approved_at", "") or "").strip()
            )
            if first_approved_at:
                normalized["first_approved_at"] = first_approved_at

            if entry_key in stored_entries:
                updated += 1
            else:
                added += 1
            stored_entries[entry_key] = normalized

        manifest["project"] = project_name
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["entries"] = stored_entries

        if not self._write_json_file(manifest_path, manifest):
            return {"ok": False, "reason": "save_failed"}

        try:
            self._plate_approved_stats_cache.clear()
            self._plate_approved_manifest_runtime_cache.pop(project_name, None)
            self._plate_approved_stats_runtime_cache.pop(project_name, None)
            self.invalidate_step3_char_source_state_cache()
        except Exception:
            pass

        return {
            "ok": True,
            "added": int(added),
            "updated": int(updated),
            "total": int(len(stored_entries)),
            "manifest_path": str(manifest_path),
        }

    def remove_plate_approved_entries(
        self,
        image_names: List[str],
        project_name: str = None,
        *,
        approved_from_run: str | None = None,
    ) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {"ok": False, "reason": "missing_project", "removed": 0}

        manifest_path = self.get_plate_approved_set_path(project_name)
        if manifest_path is None:
            return {"ok": False, "reason": "missing_manifest_path", "removed": 0}

        wanted_names = {
            self._normalize_image_set_name(name)
            for name in list(image_names or [])
            if self._normalize_image_set_name(name)
        }
        if not wanted_names:
            return {"ok": True, "removed": 0, "total": 0}

        run_token = ""
        if approved_from_run:
            try:
                run_token = str(Path(approved_from_run).resolve()).strip().lower()
            except Exception:
                run_token = str(approved_from_run or "").strip().lower()

        manifest = self.load_plate_approved_set(project_name)
        stored_entries = manifest.get("entries", {})
        if not isinstance(stored_entries, dict):
            stored_entries = {}

        removed = 0
        for entry_key, entry in list(stored_entries.items()):
            if not isinstance(entry, dict):
                continue

            entry_name = self._normalize_image_set_name(entry.get("image_name", ""))
            key_name = self._normalize_image_set_name(entry_key)
            if entry_name not in wanted_names and key_name not in wanted_names:
                continue

            if run_token:
                entry_run = str(entry.get("approved_from_run", "") or "").strip()
                try:
                    entry_run = str(Path(entry_run).resolve()).strip().lower() if entry_run else ""
                except Exception:
                    entry_run = entry_run.lower()
                if entry_run != run_token:
                    continue

            stored_entries.pop(entry_key, None)
            removed += 1

        if removed <= 0:
            return {"ok": True, "removed": 0, "total": int(len(stored_entries))}

        manifest["project"] = project_name
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["entries"] = stored_entries

        if not self._write_json_file(manifest_path, manifest):
            return {"ok": False, "reason": "save_failed", "removed": 0}

        try:
            self._plate_approved_stats_cache.clear()
            self._plate_approved_manifest_runtime_cache.pop(project_name, None)
            self._plate_approved_stats_runtime_cache.pop(project_name, None)
            self.invalidate_step3_char_source_state_cache()
        except Exception:
            pass

        return {
            "ok": True,
            "removed": int(removed),
            "total": int(len(stored_entries)),
            "manifest_path": str(manifest_path),
        }

    def load_ingest_manifest(self, iteration_num: int = None, project_name: str = None) -> Dict[str, Any]:
        manifest_path = self.get_ingest_manifest_path(iteration_num, project_name)
        if manifest_path is None or not manifest_path.exists():
            return {}
        cache_key = None
        try:
            stat = manifest_path.stat()
            cache_key = (
                str(self._resolve_project_name(project_name) or ""),
                int(getattr(stat, "st_mtime_ns", 0) or 0),
                int(getattr(stat, "st_size", 0) or 0),
                int(iteration_num or 0),
            )
        except Exception:
            cache_key = None
        if cache_key is not None:
            cached = self._ingest_manifest_cache.get(cache_key)
            if isinstance(cached, dict):
                return dict(cached)

        payload = self._read_json_file(manifest_path)
        result = payload if isinstance(payload, dict) else {}
        if cache_key is not None:
            if len(self._ingest_manifest_cache) > 16:
                self._ingest_manifest_cache.clear()
            self._ingest_manifest_cache[cache_key] = dict(result)
        return result

    def save_ingest_manifest(
        self,
        manifest: Dict[str, Any],
        iteration_num: int = None,
        project_name: str = None,
    ) -> Path | None:
        manifest_path = self.get_ingest_manifest_path(iteration_num, project_name)
        if manifest_path is None:
            return None
        if self._write_json_file(manifest_path, manifest):
            try:
                self._ingest_manifest_cache.clear()
                self._iteration_image_count_cache.clear()
                self._iteration_image_source_dir_cache.clear()
            except Exception:
                pass
            return manifest_path
        return None

    def save_ingest_balance_snapshot(self, snapshot: Dict[str, Any], project_name: str = None) -> Path | None:
        snapshot_path = self.get_ingest_balance_snapshot_path(project_name)
        if snapshot_path is None:
            return None
        if self._write_json_file(snapshot_path, snapshot):
            return snapshot_path
        return None

    def load_ingest_balance_snapshot(self, project_name: str = None) -> Dict[str, Any]:
        snapshot_path = self.get_ingest_balance_snapshot_path(project_name)
        if snapshot_path is None or not snapshot_path.exists():
            return {}
        return self._read_json_file(snapshot_path)

    def get_used_image_registry(self, project_name: str = None) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {
                "source_keys": [],
                "filenames": [],
                "total_source_keys": 0,
                "total_filenames": 0,
            }

        source_keys = set()
        filenames = set()
        master_pool_dir = self.get_master_pool_dir(project_name)

        try:
            from .campaign_ingest_planner import CampaignIngestPlanner

            planner = CampaignIngestPlanner()
        except Exception:
            planner = None

        for entry in self.list_plate_approved_entries(project_name):
            if not isinstance(entry, dict):
                continue

            image_name = str(entry.get("image_name", "") or "").strip().lower()
            if image_name:
                filenames.add(image_name)

            source_image_path = str(entry.get("source_image_path", "") or "").strip()
            if not source_image_path:
                continue

            try:
                source_path = Path(source_image_path)
            except Exception:
                continue

            if planner is not None:
                try:
                    source_key = planner.make_source_key(source_path, master_pool_dir=master_pool_dir)
                except Exception:
                    source_key = ""
            else:
                try:
                    source_key = str(source_path.resolve()).strip().lower()
                except Exception:
                    source_key = str(source_path).strip().lower()

            if source_key:
                source_keys.add(source_key)

        return {
            "source_keys": sorted(source_keys),
            "filenames": sorted(filenames),
            "total_source_keys": len(source_keys),
            "total_filenames": len(filenames),
        }

    def get_project_packet_filename_registry(
        self,
        project_name: str = None,
        *,
        exclude_iteration_num: int | None = None,
    ) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {
                "filenames": [],
                "total_filenames": 0,
            }

        filenames = set()
        try:
            raw_root = self.get_dir("raw") if project_name == self.get_active_project_name() else self.get_project_root_dir(project_name) / "1_raw_images"
        except Exception:
            raw_root = None

        if raw_root is not None:
            try:
                raw_root = Path(raw_root)
            except Exception:
                raw_root = None

        if raw_root is not None and raw_root.exists() and raw_root.is_dir():
            try:
                for image_path in raw_root.rglob("*"):
                    if not image_path.is_file():
                        continue
                    if exclude_iteration_num is not None:
                        try:
                            excluded_dir = raw_root / f"Iteracja_{int(exclude_iteration_num):03d}"
                            if excluded_dir in image_path.parents:
                                continue
                        except Exception:
                            pass
                    if image_path.suffix.lower() not in CONFIG.IMAGE_EXTENSIONS:
                        continue
                    filename = str(image_path.name or "").strip().lower()
                    if filename:
                        filenames.add(filename)
            except Exception:
                pass

        try:
            ingest_dir = self.get_project_ingest_state_dir(project_name)
        except Exception:
            ingest_dir = None
        if ingest_dir is not None and ingest_dir.exists():
            try:
                for manifest_path in ingest_dir.glob("iter_*_manifest.json"):
                    try:
                        match = re.search(r"iter_(\d+)_manifest\.json$", manifest_path.name)
                        manifest_iter = int(match.group(1)) if match else 0
                    except Exception:
                        manifest_iter = 0
                    if exclude_iteration_num is not None and manifest_iter == int(exclude_iteration_num):
                        continue
                    manifest = self._read_json_file(manifest_path)
                    for item in list((manifest or {}).get("selected_images") or []):
                        if not isinstance(item, dict):
                            continue
                        filename = str(item.get("name") or "").strip().lower()
                        if not filename:
                            for key in ("target_path", "source_path", "iteration_target_path"):
                                candidate = str(item.get(key) or "").strip()
                                if candidate:
                                    filename = Path(candidate).name.strip().lower()
                                    break
                        if filename:
                            filenames.add(filename)
            except Exception:
                pass

        return {
            "filenames": sorted(filenames),
            "total_filenames": len(filenames),
        }

    def refresh_ingest_balance_snapshot(self, project_name: str = None) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {}

        project_root = self.get_project_root_dir(project_name)
        used_registry = self.get_used_image_registry(project_name)

        from .campaign_ingest_planner import CampaignIngestPlanner

        planner = CampaignIngestPlanner()
        balance_info = planner.collect_project_training_balance(project_root)
        snapshot = {
            "project": project_name,
            "generated_at": datetime.now().isoformat(),
            "iteration": int(self.state["projects"][project_name].get("current_iteration", 1)),
            "master_pool_dir": str(self.get_master_pool_dir(project_name) or ""),
            "used_source_images": int(used_registry.get("total_source_keys", 0)),
            "used_filenames": int(used_registry.get("total_filenames", 0)),
            **balance_info,
        }
        self.save_ingest_balance_snapshot(snapshot, project_name)
        return snapshot

    def build_ingest_plan(
        self,
        batch_size: int = None,
        project_name: str = None,
    ) -> Dict[str, Any]:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return {"ok": False, "error": "Brak aktywnego projektu."}

        master_pool_dir = self.get_master_pool_dir(project_name)
        if master_pool_dir is None:
            return {"ok": False, "error": "Nie skonfigurowano master pool dla projektu."}
        if not master_pool_dir.exists() or not master_pool_dir.is_dir():
            return {"ok": False, "error": f"Master pool nie istnieje: {master_pool_dir}"}

        from .campaign_ingest_planner import CampaignIngestPlanner

        planner = CampaignIngestPlanner()
        used_registry = self.get_used_image_registry(project_name)
        balance_snapshot = self.refresh_ingest_balance_snapshot(project_name)
        effective_batch = max(1, int(batch_size or self.get_ingest_batch_size(project_name)))

        plan = planner.plan_from_master_pool(
            master_pool_dir=master_pool_dir,
            current_balance=balance_snapshot.get("char_balance", {}),
            used_source_keys=used_registry.get("source_keys", []),
            used_filenames=used_registry.get("filenames", []),
            batch_size=effective_batch,
        )
        plan["ok"] = True
        plan["project"] = project_name
        plan["iteration"] = int(self.state["projects"][project_name].get("current_iteration", 1))

        self.save_latest_ingest_plan(plan, project_name)

        return plan

    def record_iteration_ingest(
        self,
        source_dir: str | Path,
        selected_source_files: List[str | Path],
        selection_mode: str = "manual",
        proposal_summary: Dict[str, Any] | None = None,
        project_name: str = None,
        progress_callback=None,
        selected_source_metadata: List[Dict[str, Any]] | None = None,
    ) -> Path | None:
        project_name = self._resolve_project_name(project_name)
        if not project_name:
            return None

        source_dir = Path(source_dir)
        if not source_dir.exists() or not source_dir.is_dir():
            return None

        iteration = int(self.state["projects"][project_name].get("current_iteration", 1))
        target_dir = self.get_iteration_raw_dir(iteration, project_name)
        if target_dir is None:
            return None
        target_dir.mkdir(parents=True, exist_ok=True)

        from .campaign_ingest_planner import CampaignIngestPlanner

        planner = CampaignIngestPlanner()
        master_pool_dir = self.get_master_pool_dir(project_name)

        selected_images = []
        total_hist: Dict[str, int] = {ch: 0 for ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
        total_files = int(len(selected_source_files or []) or 0)
        last_progress_emit = perf_counter()
        metadata_by_name: Dict[str, Dict[str, Any]] = {}
        metadata_by_path: Dict[str, Dict[str, Any]] = {}
        for raw_meta in list(selected_source_metadata or []):
            if not isinstance(raw_meta, dict):
                continue
            name_key = str(raw_meta.get("name") or "").strip().lower()
            if name_key:
                metadata_by_name[name_key] = raw_meta
            path_key = str(raw_meta.get("source_path") or "").strip()
            if path_key:
                try:
                    metadata_by_path[str(Path(path_key).resolve()).lower()] = raw_meta
                except Exception:
                    metadata_by_path[path_key.lower()] = raw_meta

        def _progress(value: float, message: str = "", detail: str = "", *, force: bool = False) -> None:
            nonlocal last_progress_emit
            if not callable(progress_callback):
                return
            now = perf_counter()
            if not force and now - last_progress_emit < 0.12:
                return
            last_progress_emit = now
            try:
                progress_callback(value, message, detail=detail, force=force)
            except TypeError:
                try:
                    progress_callback(value, message)
                except Exception:
                    pass
            except Exception:
                pass

        _progress(
            6,
            "Przygotowuję manifest obrazów E1.",
            f"Do zapisania: {total_files} obrazów.",
            force=True,
        )

        for processed_count, item in enumerate(selected_source_files, start=1):
            source_path = Path(item)
            if not source_path.is_absolute():
                source_path = source_dir / source_path.name
            if not source_path.exists() or not source_path.is_file():
                _progress(
                    6.0 + 74.0 * (processed_count / max(1, total_files)),
                    "Przygotowuję manifest obrazów E1.",
                    f"Przetworzono {processed_count}/{total_files}. Do manifestu: {len(selected_images)}.",
                )
                continue

            try:
                resolved_source_path = source_path.resolve()
            except Exception:
                resolved_source_path = source_path.absolute()
            meta = metadata_by_name.get(str(source_path.name or "").strip().lower())
            if meta is None:
                meta = metadata_by_path.get(str(resolved_source_path).lower())
            true_texts = list((meta or {}).get("ground_truth_texts") or [])
            if not true_texts:
                true_texts = planner.extract_true_texts_from_filename(source_path.name)
            char_hist = dict((meta or {}).get("char_histogram") or {})
            if not char_hist:
                char_hist = planner.build_char_histogram(true_texts)
            for ch, value in char_hist.items():
                total_hist[ch] = total_hist.get(ch, 0) + int(value)

            logical_target_path = target_dir / source_path.name
            actual_image_path = logical_target_path if logical_target_path.exists() else source_path
            selected_images.append({
                "name": source_path.name,
                "source_path": str(resolved_source_path),
                "source_key": str((meta or {}).get("source_key") or "").strip() or planner.make_source_key(source_path, master_pool_dir=master_pool_dir),
                "target_path": str(actual_image_path.resolve()),
                "iteration_target_path": str(logical_target_path.resolve()),
                "ground_truth_texts": true_texts,
                "char_histogram": char_hist,
            })
            _progress(
                6.0 + 74.0 * (processed_count / max(1, total_files)),
                "Przygotowuję manifest obrazów E1.",
                f"Przetworzono {processed_count}/{total_files}. Do manifestu: {len(selected_images)}.",
            )

        normalized_selection_mode = str(selection_mode or "manual").strip() or "manual"
        _progress(
            84,
            "Buduję plik manifestu E1.",
            f"Manifest obejmie {len(selected_images)} obrazów.",
            force=True,
        )
        manifest = {
            "project": project_name,
            "iteration": iteration,
            "created_at": datetime.now().isoformat(),
            "selection_mode": normalized_selection_mode,
            "source_dir": str(source_dir.resolve()),
            "target_dir": str(target_dir.resolve()),
            "master_pool_dir": str(master_pool_dir.resolve()) if master_pool_dir else "",
            "selected_count": len(selected_images),
            "manifest_only": bool(normalized_selection_mode in {"planned", "planned_manifest", "source_reuse", "pool_reuse", "stage_reuse", "iteration_reuse"}),
            "image_set_token": self.build_image_name_set_token(
                [str(item.get("name") or "").strip() for item in selected_images]
            ),
            "char_histogram": {k: int(v) for k, v in total_hist.items() if int(v) > 0},
            "selected_images": selected_images,
            "proposal_summary": proposal_summary or {},
        }

        _progress(
            92,
            "Zapisuję manifest E1 na dysku.",
            "Kończę zapis JSON i odświeżam pamięć podręczną projektu.",
            force=True,
        )
        manifest_path = self.save_ingest_manifest(manifest, iteration, project_name)
        _progress(
            100,
            "Manifest E1 zapisany.",
            f"Zapisano {len(selected_images)} obrazów.",
            force=True,
        )
        return manifest_path

# Singleton Menadżera
bind_campaign_project_registry_methods(CampaignManager)
bind_campaign_stage_state_methods(CampaignManager)
bind_campaign_project_history_methods(CampaignManager)
CAMPAIGN = CampaignManager()
