#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wspolny, lekki cache dla odczytow projektowych.

Na pierwszy etap skupiamy sie na:
- plikach JSON / manifestach,
- listach katalogow run_*.

Cache ma byc bezpieczny:
- JSON jest walidowany po stat(path),
- listy runow maja krotki TTL, zeby nie "zamrozic" widoku,
- wyniki sa kopiowane przed zwroceniem, zeby caller nie mutowal cache.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from threading import RLock
from typing import Any


class ProjectCache:
    def __init__(self) -> None:
        self._lock = RLock()
        self._json_cache: dict[str, dict[str, Any]] = {}
        self._run_dir_cache: dict[tuple[str, bool], dict[str, Any]] = {}
        self._run_dir_ttl_seconds = 2.0

    @staticmethod
    def _normalize_path_key(path_value: Path | str | None) -> str:
        if path_value is None:
            return ""
        try:
            return str(Path(path_value).resolve())
        except Exception:
            return str(Path(path_value))

    @staticmethod
    def _read_stat_signature(path: Path | str | None) -> tuple[int, int] | None:
        if path is None:
            return None
        try:
            stat = Path(path).stat()
        except Exception:
            return None
        return (
            int(getattr(stat, "st_mtime_ns", 0) or 0),
            int(getattr(stat, "st_size", 0) or 0),
        )

    def load_json(self, path_value: Path | str | None, *, default: Any = None) -> Any:
        path = Path(path_value) if path_value is not None else None
        if path is None or not path.exists() or not path.is_file():
            return copy.deepcopy(default)

        cache_key = self._normalize_path_key(path)
        signature = self._read_stat_signature(path)
        if not cache_key or signature is None:
            return copy.deepcopy(default)

        with self._lock:
            cached = self._json_cache.get(cache_key)
            if cached is not None and cached.get("signature") == signature:
                return copy.deepcopy(cached.get("payload"))

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return copy.deepcopy(default)

        with self._lock:
            self._json_cache[cache_key] = {
                "signature": signature,
                "payload": copy.deepcopy(payload),
            }

        return copy.deepcopy(payload)

    def invalidate_json(self, path_value: Path | str | None) -> None:
        cache_key = self._normalize_path_key(path_value)
        if not cache_key:
            return
        with self._lock:
            self._json_cache.pop(cache_key, None)

    def list_annotation_run_dirs(
        self,
        root_value: Path | str | None,
        *,
        require_xml: bool = True,
    ) -> list[Path]:
        root = Path(root_value) if root_value is not None else None
        if root is None:
            return []

        try:
            if not root.exists() or not root.is_dir():
                return []
        except Exception:
            return []

        cache_key = (self._normalize_path_key(root), bool(require_xml))
        root_signature = self._read_stat_signature(root)
        now = time.monotonic()

        with self._lock:
            cached = self._run_dir_cache.get(cache_key)
            if (
                cached is not None
                and cached.get("root_signature") == root_signature
                and (now - float(cached.get("saved_at", 0.0) or 0.0)) <= self._run_dir_ttl_seconds
            ):
                return [Path(item) for item in list(cached.get("paths") or [])]

        resolved_paths: list[str] = []
        try:
            for path in root.rglob("run_*"):
                try:
                    if not path.is_dir():
                        continue
                    if require_xml and not (path / "annotations.xml").exists():
                        continue
                    resolved_paths.append(self._normalize_path_key(path))
                except Exception:
                    continue
        except Exception:
            return []

        with self._lock:
            self._run_dir_cache[cache_key] = {
                "root_signature": root_signature,
                "saved_at": now,
                "paths": list(resolved_paths),
            }

        return [Path(item) for item in resolved_paths]

    def invalidate_annotation_run_dirs(self, root_value: Path | str | None) -> None:
        root_key = self._normalize_path_key(root_value)
        if not root_key:
            return
        with self._lock:
            for require_xml in (False, True):
                self._run_dir_cache.pop((root_key, require_xml), None)


PROJECT_CACHE = ProjectCache()
