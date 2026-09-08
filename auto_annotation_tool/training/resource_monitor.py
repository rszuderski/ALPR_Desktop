#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight resource monitoring for training worker processes."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ..config import get_torch_module, logger


try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


def _mib(value: float | int | None) -> float:
    try:
        return round(float(value or 0.0) / (1024.0 * 1024.0), 2)
    except Exception:
        return 0.0


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _windows_virtual_memory() -> dict:
    if platform.system().lower() != "windows":
        return {}

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except Exception:
        ok = False
    if not ok:
        return {}

    total = int(status.ullTotalPhys)
    available = int(status.ullAvailPhys)
    used = max(0, total - available)
    percent = (used / total) * 100.0 if total else 0.0
    return {
        "ram_total_mib": _mib(total),
        "ram_available_mib": _mib(available),
        "ram_used_mib": _mib(used),
        "ram_percent": round(percent, 1),
    }


def sample_system_memory() -> dict:
    """Zwraca lekki snapshot RAM bez uruchamiania pełnego monitora treningu."""
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return {
                "ram_total_mib": _mib(getattr(vm, "total", 0)),
                "ram_available_mib": _mib(getattr(vm, "available", 0)),
                "ram_used_mib": _mib(getattr(vm, "used", 0)),
                "ram_percent": round(float(getattr(vm, "percent", 0.0) or 0.0), 1),
            }
        except Exception:
            pass
    return _windows_virtual_memory()


def _windows_process_memory(pid: int) -> dict:
    if platform.system().lower() != "windows":
        return {}

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    handle = None
    try:
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
            False,
            int(pid),
        )
        if not handle:
            return {}
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return {}
        return {
            "rss_mib": _mib(int(counters.WorkingSetSize)),
            "peak_rss_mib": _mib(int(counters.PeakWorkingSetSize)),
            "pagefile_mib": _mib(int(counters.PagefileUsage)),
        }
    except Exception:
        return {}
    finally:
        try:
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            pass


def _format_mib(value: float | int | None) -> str:
    try:
        mib = float(value or 0.0)
    except Exception:
        mib = 0.0
    if mib >= 1024.0:
        return f"{mib / 1024.0:.2f} GB"
    return f"{mib:.0f} MB"


def format_resource_sample_line(sample: dict) -> str:
    system = dict((sample or {}).get("system") or {})
    process = dict((sample or {}).get("process") or {})
    gpu = dict((sample or {}).get("gpu") or {})

    parts: list[str] = []
    ram_pct = system.get("ram_percent")
    ram_used = system.get("ram_used_mib")
    ram_total = system.get("ram_total_mib")
    if ram_pct is not None and ram_total:
        parts.append(f"RAM {float(ram_pct):.1f}% ({_format_mib(ram_used)}/{_format_mib(ram_total)})")

    rss = process.get("rss_mib")
    if rss:
        parts.append(f"proces {_format_mib(rss)}")

    cpu = system.get("cpu_percent")
    if cpu is not None:
        parts.append(f"CPU {float(cpu):.1f}%")

    if bool(gpu.get("available")):
        allocated = gpu.get("allocated_mib")
        reserved = gpu.get("reserved_mib")
        total = gpu.get("total_mib")
        free = gpu.get("free_mib")
        if total:
            parts.append(
                "VRAM "
                f"alloc {_format_mib(allocated)} | rez {_format_mib(reserved)} | "
                f"wolne {_format_mib(free)}/{_format_mib(total)}"
            )
        else:
            parts.append(f"VRAM alloc {_format_mib(allocated)} | rez {_format_mib(reserved)}")

    return " | ".join(parts) if parts else "brak danych zasobow"


@dataclass
class ResourceAccumulator:
    samples: int = 0
    started_at: str = field(default_factory=_now_iso)
    finished_at: str = ""
    max_ram_percent: float = 0.0
    max_ram_used_mib: float = 0.0
    max_process_rss_mib: float = 0.0
    max_process_peak_rss_mib: float = 0.0
    max_cpu_percent: float = 0.0
    max_gpu_allocated_mib: float = 0.0
    max_gpu_reserved_mib: float = 0.0
    min_gpu_free_mib: float = 0.0
    gpu_total_mib: float = 0.0
    last_sample: dict = field(default_factory=dict)

    def update(self, sample: dict) -> None:
        self.samples += 1
        self.last_sample = sample

        system = dict(sample.get("system") or {})
        process = dict(sample.get("process") or {})
        gpu = dict(sample.get("gpu") or {})

        self.max_ram_percent = max(self.max_ram_percent, float(system.get("ram_percent") or 0.0))
        self.max_ram_used_mib = max(self.max_ram_used_mib, float(system.get("ram_used_mib") or 0.0))
        self.max_cpu_percent = max(self.max_cpu_percent, float(system.get("cpu_percent") or 0.0))
        self.max_process_rss_mib = max(self.max_process_rss_mib, float(process.get("rss_mib") or 0.0))
        self.max_process_peak_rss_mib = max(
            self.max_process_peak_rss_mib,
            float(process.get("peak_rss_mib") or 0.0),
        )
        self.max_gpu_allocated_mib = max(
            self.max_gpu_allocated_mib,
            float(gpu.get("allocated_mib") or 0.0),
        )
        self.max_gpu_reserved_mib = max(
            self.max_gpu_reserved_mib,
            float(gpu.get("reserved_mib") or 0.0),
        )
        gpu_free = float(gpu.get("free_mib") or 0.0)
        if gpu_free:
            self.min_gpu_free_mib = gpu_free if not self.min_gpu_free_mib else min(self.min_gpu_free_mib, gpu_free)
        self.gpu_total_mib = max(self.gpu_total_mib, float(gpu.get("total_mib") or 0.0))

    def report(self) -> dict:
        self.finished_at = self.finished_at or _now_iso()
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "samples": int(self.samples),
            "max_ram_percent": round(self.max_ram_percent, 1),
            "max_ram_used_mib": round(self.max_ram_used_mib, 2),
            "max_process_rss_mib": round(self.max_process_rss_mib, 2),
            "max_process_peak_rss_mib": round(self.max_process_peak_rss_mib, 2),
            "max_cpu_percent": round(self.max_cpu_percent, 1),
            "max_gpu_allocated_mib": round(self.max_gpu_allocated_mib, 2),
            "max_gpu_reserved_mib": round(self.max_gpu_reserved_mib, 2),
            "min_gpu_free_mib": round(self.min_gpu_free_mib, 2),
            "gpu_total_mib": round(self.gpu_total_mib, 2),
            "last_sample": self.last_sample,
        }


class TrainingResourceMonitor:
    """Samples system, process and CUDA memory while a training worker is alive."""

    def __init__(
        self,
        *,
        run_dir: Path,
        event_sink: Optional[Callable[[dict], None]] = None,
        interval_s: float = 2.0,
        device=None,
    ):
        self.run_dir = Path(run_dir)
        self.event_sink = event_sink
        self.interval_s = max(0.5, float(interval_s or 2.0))
        self.device = device
        self.pid = os.getpid()
        self.samples_path = self.run_dir / "resource_samples.jsonl"
        self.report_path = self.run_dir / "resource_report.json"
        self.accumulator = ResourceAccumulator()
        self._started_perf = time.perf_counter()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._psutil_process = None
        self._last_cpu_prime = False

    def start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.samples_path.write_text("", encoding="utf-8")
        except Exception:
            pass
        if psutil is not None:
            try:
                self._psutil_process = psutil.Process(self.pid)
                self._psutil_process.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None)
                self._last_cpu_prime = True
            except Exception:
                self._psutil_process = None
        self._thread = threading.Thread(target=self._run, name="training-resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            try:
                thread.join(timeout=max(0.2, self.interval_s + 0.5))
            except Exception:
                pass
        sample = self.sample()
        self.accumulator.update(sample)
        self._append_sample(sample)
        report = self.accumulator.report()
        report["report_path"] = str(self.report_path)
        report["samples_path"] = str(self.samples_path)
        report["summary_text"] = self.format_report_line(report)
        try:
            self.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Nie udalo sie zapisac raportu zasobow treningu: {e}")
        if self.event_sink:
            try:
                self.event_sink({"type": "resource_report", "report": report})
            except Exception:
                pass
        return report

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self.sample()
            self.accumulator.update(sample)
            self._append_sample(sample)
            if self.event_sink:
                try:
                    self.event_sink({"type": "resource_sample", "sample": sample})
                except Exception:
                    pass
            self._stop_event.wait(self.interval_s)

    def _append_sample(self, sample: dict) -> None:
        try:
            with self.samples_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _sample_system(self) -> dict:
        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                result = {
                    "ram_total_mib": _mib(getattr(vm, "total", 0)),
                    "ram_available_mib": _mib(getattr(vm, "available", 0)),
                    "ram_used_mib": _mib(getattr(vm, "used", 0)),
                    "ram_percent": round(float(getattr(vm, "percent", 0.0) or 0.0), 1),
                }
                try:
                    result["cpu_percent"] = round(float(psutil.cpu_percent(interval=None)), 1)
                except Exception:
                    pass
                return result
            except Exception:
                pass
        return _windows_virtual_memory()

    def _sample_process(self) -> dict:
        if self._psutil_process is not None:
            try:
                mem = self._psutil_process.memory_info()
                result = {
                    "pid": int(self.pid),
                    "rss_mib": _mib(getattr(mem, "rss", 0)),
                    "vms_mib": _mib(getattr(mem, "vms", 0)),
                }
                try:
                    result["cpu_percent"] = round(float(self._psutil_process.cpu_percent(interval=None)), 1)
                except Exception:
                    pass
                return result
            except Exception:
                pass
        result = {"pid": int(self.pid)}
        result.update(_windows_process_memory(self.pid))
        return result

    def _sample_gpu(self) -> dict:
        torch = get_torch_module()
        if torch is None:
            return {"available": False}
        try:
            cuda = getattr(torch, "cuda", None)
            if cuda is None or not bool(cuda.is_available()):
                return {"available": False}

            device_index = 0
            raw_device = self.device
            try:
                if raw_device not in (None, "auto", "cpu", ""):
                    device_index = int(raw_device)
            except Exception:
                device_index = 0

            result = {
                "available": True,
                "device_index": int(device_index),
                "device_name": str(cuda.get_device_name(device_index)),
                "allocated_mib": _mib(cuda.memory_allocated(device_index)),
                "reserved_mib": _mib(cuda.memory_reserved(device_index)),
                "max_allocated_mib": _mib(cuda.max_memory_allocated(device_index)),
                "max_reserved_mib": _mib(cuda.max_memory_reserved(device_index)),
            }
            try:
                free_bytes, total_bytes = cuda.mem_get_info(device_index)
                result["free_mib"] = _mib(free_bytes)
                result["total_mib"] = _mib(total_bytes)
                result["used_mib"] = round(float(result["total_mib"]) - float(result["free_mib"]), 2)
            except Exception:
                pass
            return result
        except Exception as e:
            return {"available": False, "error": str(e)}

    def sample(self) -> dict:
        sample = {
            "timestamp": _now_iso(),
            "elapsed_s": round(max(0.0, time.perf_counter() - self._started_perf), 3),
            "system": self._sample_system(),
            "process": self._sample_process(),
            "gpu": self._sample_gpu(),
        }
        sample["summary_text"] = format_resource_sample_line(sample)
        return sample

    @staticmethod
    def format_report_line(report: dict) -> str:
        parts = []
        if float(report.get("max_ram_percent") or 0.0):
            parts.append(
                f"RAM max {float(report.get('max_ram_percent') or 0.0):.1f}% "
                f"({_format_mib(report.get('max_ram_used_mib'))})"
            )
        if float(report.get("max_process_rss_mib") or 0.0):
            parts.append(f"proces max {_format_mib(report.get('max_process_rss_mib'))}")
        if float(report.get("max_cpu_percent") or 0.0):
            parts.append(f"CPU max {float(report.get('max_cpu_percent') or 0.0):.1f}%")
        if float(report.get("max_gpu_reserved_mib") or 0.0):
            gpu_total = float(report.get("gpu_total_mib") or 0.0)
            if gpu_total:
                parts.append(
                    f"VRAM rez max {_format_mib(report.get('max_gpu_reserved_mib'))}/"
                    f"{_format_mib(gpu_total)}"
                )
            else:
                parts.append(f"VRAM rez max {_format_mib(report.get('max_gpu_reserved_mib'))}")
        return " | ".join(parts) if parts else "brak danych zasobow"
