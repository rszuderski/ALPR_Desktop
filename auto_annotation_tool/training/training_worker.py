#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Worker treningu uruchamiany w osobnym procesie.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

from ..config import logger
from .trainer import YOLOPoseTrainer
from .training_history import TrainingHistory
from .resource_monitor import TrainingResourceMonitor


def _append_event(event_path: Path, payload: dict) -> None:
    try:
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug(f"Nie udało się zapisać zdarzenia worker-treningu: {e}")


def _read_control(control_path: Path) -> dict:
    try:
        payload = json.loads(control_path.read_text(encoding="utf-8-sig"))
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _run_control_monitor(trainer: YOLOPoseTrainer, control_path: Path, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        control = _read_control(control_path)
        if bool(control.get("stop")):
            trainer.should_stop = True
        if bool(control.get("pause")):
            trainer.should_pause = True
        if not bool(getattr(trainer, "is_training", False)) and (trainer.should_stop or trainer.should_pause):
            break
        time.sleep(0.2)


def run_job(job_path: Path) -> int:
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    history_dir = Path(str(payload.get("history_dir") or "").strip())
    run_id = str(payload.get("run_id") or "").strip()
    event_path = Path(str(payload.get("event_file") or "").strip())
    control_path = Path(str(payload.get("control_file") or "").strip())

    history = TrainingHistory(history_dir=history_dir)
    trainer = YOLOPoseTrainer(history=history)
    trainer.current_run = history.get_run(run_id)
    if trainer.current_run is None:
        _append_event(
            event_path,
            {
                "type": "training_end",
                "success": False,
                "message": f"Nie znaleziono runu treningu: {run_id}",
            },
        )
        return 2

    trainer.is_training = True
    trainer.should_pause = False
    trainer.should_stop = False

    trainer.on_batch_progress = lambda epoch, batch_idx, total_batches, batch_pct: _append_event(
        event_path,
        {
            "type": "batch_progress",
            "epoch": int(epoch),
            "batch_idx": int(batch_idx),
            "total_batches": int(total_batches),
            "batch_pct": float(batch_pct),
        },
    )
    trainer.on_epoch_end = lambda epoch, metrics: _append_event(
        event_path,
        {
            "type": "epoch_end",
            "epoch": int(epoch),
            "metrics": dict(metrics or {}),
        },
    )
    trainer.on_progress = lambda percent, message: _append_event(
        event_path,
        {
            "type": "progress",
            "percent": float(percent or 0.0),
            "message": str(message or "").strip(),
        },
    )
    trainer.on_training_end = lambda success, message: _append_event(
        event_path,
        {
            "type": "training_end",
            "success": bool(success),
            "message": str(message or "").strip(),
        },
    )

    stop_event = threading.Event()
    control_thread = threading.Thread(
        target=_run_control_monitor,
        args=(trainer, control_path, stop_event),
        daemon=True,
    )
    control_thread.start()

    resource_monitor: TrainingResourceMonitor | None = None
    try:
        try:
            resource_monitor = TrainingResourceMonitor(
                run_dir=Path(str(getattr(trainer.current_run, "output_dir", "") or history_dir / run_id)),
                event_sink=lambda event: _append_event(event_path, event),
                interval_s=float(payload.get("resource_interval_s") or 2.0),
                device=payload.get("device"),
            )
            resource_monitor.start()
            _append_event(
                event_path,
                {
                    "type": "resource_monitor_start",
                    "samples_file": str(resource_monitor.samples_path),
                    "report_file": str(resource_monitor.report_path),
                },
            )
        except Exception as e:
            logger.debug(f"Nie udało się uruchomić monitora zasobów treningu: {e}")
            resource_monitor = None

        trainer._training_loop(
            str(payload.get("model_file") or "").strip(),
            str(payload.get("dataset_path") or "").strip(),
            int(payload.get("epochs") or 0),
            int(payload.get("batch_size") or 0),
            int(payload.get("img_size") or 0),
            payload.get("device"),
            float(payload.get("lr0") or 0.0),
            str(payload.get("resume_from") or "").strip() or None,
        )
    finally:
        if resource_monitor is not None:
            try:
                resource_monitor.stop()
            except Exception as e:
                logger.debug(f"Nie udało się domknąć monitora zasobów treningu: {e}")
        stop_event.set()

    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if not args:
        print("Brak ścieżki job.json", file=sys.stderr)
        return 2

    job_path = Path(args[0])
    if not job_path.exists():
        print(f"Nie znaleziono pliku job: {job_path}", file=sys.stderr)
        return 2

    return run_job(job_path)


if __name__ == "__main__":
    raise SystemExit(main())
