"""Lightweight append-only project history for campaign runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .config import logger


def get_project_history_path(self, project_name: str = None) -> Path | None:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return None
    try:
        return self.get_project_root_dir(project_name) / "_campaign_state" / "project_history.jsonl"
    except Exception:
        return None


def append_project_history_event(
    self,
    event_type: str,
    title: str = "",
    *,
    transition_id: str = "",
    gate_id: str = "",
    action: str = "",
    status: str = "ok",
    iteration_num: int | None = None,
    step_num: int | None = None,
    resources: Dict[str, Any] | None = None,
    artifacts: Dict[str, Any] | None = None,
    metrics: Dict[str, Any] | None = None,
    details: Dict[str, Any] | None = None,
    project_name: str = None,
) -> bool:
    """Append a compact audit event without changing campaign state."""
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return False

    history_path = self.get_project_history_path(project_name)
    if history_path is None:
        return False

    project_data = self.state.get("projects", {}).get(project_name, {})
    try:
        resolved_iteration = int(iteration_num or project_data.get("current_iteration", 1) or 1)
    except Exception:
        resolved_iteration = 1
    try:
        resolved_step = int(step_num or project_data.get("current_step", 1) or 1)
    except Exception:
        resolved_step = 1

    entry = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project": project_name,
        "iteration": int(resolved_iteration),
        "step": int(resolved_step),
        "event_type": str(event_type or "event").strip() or "event",
        "title": str(title or "").strip(),
        "transition_id": str(transition_id or "").strip(),
        "gate_id": str(gate_id or "").strip(),
        "action": str(action or "").strip(),
        "status": str(status or "ok").strip() or "ok",
        "resources": dict(resources or {}),
        "artifacts": dict(artifacts or {}),
        "metrics": dict(metrics or {}),
        "details": dict(details or {}),
    }

    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return True
    except Exception as exc:
        logger.error(f"Nie udało się dopisać historii projektu {project_name}: {exc}")
        return False


def load_project_history(self, project_name: str = None, *, limit: int = 100) -> List[Dict[str, Any]]:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return []

    history_path = self.get_project_history_path(project_name)
    if history_path is None or not history_path.exists():
        return []

    try:
        max_items = max(1, int(limit or 100))
    except Exception:
        max_items = 100

    try:
        lines = history_path.read_text(encoding="utf-8-sig").splitlines()
    except Exception:
        return []

    entries: List[Dict[str, Any]] = []
    for line in lines[-max_items:]:
        line = str(line or "").strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            entries.append(item)
    return entries


def summarize_project_history(self, project_name: str = None, *, limit: int = 12) -> List[str]:
    rows: List[str] = []
    for item in self.load_project_history(project_name, limit=limit):
        when = str(item.get("created_at", "") or "").strip()
        transition = str(item.get("transition_id") or item.get("gate_id") or "").strip()
        title = str(item.get("title") or item.get("action") or item.get("event_type") or "").strip()
        status = str(item.get("status") or "").strip()
        prefix = f"{when} | " if when else ""
        gate = f"{transition} | " if transition else ""
        suffix = f" [{status}]" if status and status != "ok" else ""
        rows.append(f"{prefix}{gate}{title}{suffix}")
    return rows


def bind_campaign_project_history_methods(cls) -> None:
    for name in (
        "get_project_history_path",
        "append_project_history_event",
        "load_project_history",
        "summarize_project_history",
    ):
        setattr(cls, name, globals()[name])
