"""Quiet, atomic downloads of official detector weights. No Tk or subprocesses."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import os
import tempfile
import threading
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


class DownloadCancelled(Exception):
    pass


@dataclass(frozen=True)
class DownloadProgress:
    source: str
    received: int = 0
    total: int | None = None
    bytes_per_second: float = 0.0
    complete: bool = False

    @property
    def percent(self) -> float | None:
        if self.complete:
            return 100.0
        return min(99.9, 100 * self.received / self.total) if self.total else None


def official_asset_url(asset_name: str) -> str:
    # Use the release selected by the installed Ultralytics, without invoking
    # its console downloader or automatic dependency installation.
    from ultralytics.utils.downloads import attempt_download_asset

    defaults = inspect.signature(attempt_download_asset).parameters
    repo = defaults["repo"].default
    release = defaults["release"].default
    return f"https://github.com/{repo}/releases/download/{quote(str(release), safe='')}/{quote(Path(asset_name).name)}"


def download_model(url: str, target: Path, cancel: threading.Event, progress,
                   *, opener=urlopen, clock=time.monotonic) -> Path:
    """Keep partial transfers invisible to model discovery, even on cancellation."""
    target = Path(target)
    if target.is_file() and target.stat().st_size > 0:
        size = target.stat().st_size
        progress(DownloadProgress(url, size, size, complete=True))
        return target
    if cancel.is_set():
        raise DownloadCancelled()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = None
    try:
        progress(DownloadProgress(url))
        request = Request(url, headers={"User-Agent": "ALPR-Desktop/model-download", "Accept-Encoding": "identity"})
        with opener(request, timeout=15) as response:
            raw_total = response.headers.get("Content-Length")
            total = int(raw_total) if raw_total and str(raw_total).isdigit() else None
            total = total if total and total > 0 else None
            received = 0
            started = last_report = clock()
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".part", delete=False) as output:
                partial = Path(output.name)
                while True:
                    if cancel.is_set():
                        raise DownloadCancelled()
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    received += len(chunk)
                    now = clock()
                    if now - last_report >= 0.1:
                        progress(DownloadProgress(url, received, total, received / max(0.001, now - started)))
                        last_report = now
                if cancel.is_set():
                    raise DownloadCancelled()
                if not received or (total is not None and received != total):
                    raise OSError(f"Niekompletny plik modelu: {received} / {total} B")
                output.flush()
                os.fsync(output.fileno())
            if cancel.is_set():
                raise DownloadCancelled()
            os.replace(partial, target)
            partial = None
            progress(DownloadProgress(url, received, total, received / max(0.001, clock() - started), True))
        return target
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
