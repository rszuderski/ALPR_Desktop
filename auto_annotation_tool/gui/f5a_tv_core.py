# -*- coding: utf-8 -*-
r"""F5A — automatyczna przeglądarka bloków TV (jeden plik, tylko standardowy Tkinter + Pillow).

  py -3 "$HOME\Desktop\plan\F5A_08_AUTOMATYCZNY_POKAZ_TV_20261104.py" --check
  py -3 "$HOME\Desktop\plan\F5A_08_AUTOMATYCZNY_POKAZ_TV_20261104.py"
  py -3 "$HOME\Desktop\plan\F5A_08_AUTOMATYCZNY_POKAZ_TV_20261104.py" --demo

Wczytuje wyłącznie zamrożony 03_POKAZ_TV/manifest_blokow.json z F5A/07.
10 obrazów/blok; przy normalnym pokazie 30 s/obraz, 5 min/blok.
Nie uruchamia Androida, nie edytuje zdjęć, GT ani Sortowni.
Log przebiegu (czas planowy i czas żądania wyświetlenia) trafia do 00_protokol/pokaz_TV_logi.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from alpr_runtime import workspace_root
DEFAULT_ROOT = workspace_root() / 'experiments' / 'F5A-ROI-A12-01'
TV_REL = Path("01_materialy") / "03_POKAZ_TV"
BLOCKS = ("BLOK_A", "BLOK_B", "BLOK_C", "BLOK_D")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stop(message: str) -> None:
    raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def validate_material(root: Path) -> tuple[Path, str, dict[str, list[dict]]]:
    base = root / TV_REL
    manifest_path = base / "manifest_blokow.json"
    if not base.is_dir() or base.is_symlink():
        stop(f"Brak katalogu 03_POKAZ_TV lub dowiązanie: {base}")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        stop(f"Brak manifestu z F5A/07: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if payload.get("schema") != "alpr.f5a.tv_blocks.v1":
        stop("Nieznany schemat manifestu bloków")
    if payload.get("experiment_id") != "F5A-ROI-A12-01":
        stop("Nieprawidłowy identyfikator eksperymentu")
    if payload.get("scene_count") != 40 or payload.get("blocks") != {b: 10 for b in BLOCKS}:
        stop("Manifest nie deklaruje dokładnie 4 bloków po 10 zdjęć")
    rows = payload.get("presentation_order")
    if not isinstance(rows, list) or len(rows) != 40:
        stop("Manifest musi zawierać dokładnie 40 rekordów kolejności")
    grouped: dict[str, list[dict]] = {block: [] for block in BLOCKS}
    known_sha: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            stop("Nieprawidłowy rekord manifestu")
        block = row.get("blok")
        slot = row.get("pozycja")
        digest = row.get("sha256")
        filename = row.get("nazwa_w_pokazie")
        rel = row.get("sciezka_w_pokazie")
        if block not in grouped or type(slot) is not int or not 1 <= slot <= 10:
            stop(f"Błędny blok lub numer pozycji: {block}, {slot}")
        if not isinstance(filename, str) or not filename.startswith(f"{slot:02d}_"):
            stop(f"Błędny prefiks kolejności: {filename}")
        if not isinstance(rel, str) or rel != f"{block}/{filename}":
            stop(f"Niebezpieczna albo niezgodna ścieżka obrazu: {rel}")
        if Path(filename).name != filename or "\\" in filename or filename in (".", ".."):
            stop(f"Nieprawidłowa nazwa obrazu: {filename}")
        if Path(filename).suffix.lower() not in IMG_EXTS:
            stop(f"Nieobsługiwane rozszerzenie: {filename}")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            stop(f"Błędny SHA-256: {filename}")
        if digest in known_sha:
            stop(f"Powtórzony SHA-256 sceny: {filename}")
        known_sha.add(digest)
        path = base / block / filename
        if not path.is_file() or path.is_symlink():
            stop(f"Brak obrazu lub dowiązanie: {path}")
        before = path.stat()
        actual_sha = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            stop(f"Obraz zmienił się podczas kontroli: {filename}")
        if actual_sha != digest or ("bytes" in row and after.st_size != row["bytes"]):
            stop(f"SHA/rozmiar nie zgadza się z manifestem: {filename}")
        grouped[block].append(dict(row, _path=path))
    for block in BLOCKS:
        expected = {row["nazwa_w_pokazie"] for row in grouped[block]}
        folder = base / block
        if not folder.is_dir() or folder.is_symlink():
            stop(f"Brak katalogu bloku: {folder}")
        found = {p.name for p in folder.iterdir() if p.is_file() and not p.is_symlink()}
        if found != expected or len(list(folder.iterdir())) != 10:
            stop(f"Zawartość {block} nie zgadza się z manifestem; brak lub obcy plik")
        positions = sorted(row["pozycja"] for row in grouped[block])
        if positions != list(range(1, 11)):
            stop(f"Nieciągła kolejność 01–10 w {block}")
        grouped[block].sort(key=lambda row: row["pozycja"])
    if manifest_sha != sha256_file(manifest_path):
        stop("Manifest zmienił się podczas kontroli")
    return base, manifest_sha, grouped


def choose_monitors(root_window):
    """Win32 bez dodatkowych bibliotek; w innym OS: ekran wykrywany przez Tk."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            class RECT(ctypes.Structure):
                _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                            ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
            class MONITORINFOEXW(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT),
                            ("rcWork", RECT), ("dwFlags", wintypes.DWORD),
                            ("szDevice", ctypes.c_wchar * 32)]
            result = []
            CALLBACK = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                                          ctypes.POINTER(RECT), wintypes.LPARAM)
            user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.c_void_p]
            user32.GetMonitorInfoW.restype = wintypes.BOOL
            user32.EnumDisplayMonitors.argtypes = [wintypes.HDC, ctypes.POINTER(RECT), CALLBACK, wintypes.LPARAM]
            user32.EnumDisplayMonitors.restype = wintypes.BOOL
            def enum_callback(hmon, hdc, rect, lparam):
                info = MONITORINFOEXW()
                info.cbSize = ctypes.sizeof(MONITORINFOEXW)
                if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
                    r = info.rcMonitor
                    result.append({"name": info.szDevice, "x": int(r.left), "y": int(r.top),
                                   "w": int(r.right-r.left), "h": int(r.bottom-r.top),
                                   "primary": bool(info.dwFlags & 1)})
                return True
            callback = CALLBACK(enum_callback)
            if user32.EnumDisplayMonitors(None, None, callback, 0) and result:
                return sorted(result, key=lambda x: (not x["primary"], x["name"]))
        except Exception:
            pass
    return [{"name": "Ekran systemowy", "x": 0, "y": 0,
             "w": int(root_window.winfo_screenwidth()), "h": int(root_window.winfo_screenheight()),
             "primary": True}]


def set_dpi_aware():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class TvApp:
    def __init__(self, root_path: Path, demo: bool):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageOps, ImageTk
        self.tk, self.ttk = tk, ttk
        self.Image, self.ImageOps, self.ImageTk = Image, ImageOps, ImageTk
        self.root_path = root_path
        self.base, self.manifest_sha, self.blocks = validate_material(root_path)
        self.demo = demo
        self.dwell = 3 if demo else 30
        self.countdown = 3 if demo else 5
        set_dpi_aware()
        self.window = tk.Tk()
        self.window.title("F5A · Automatyczny pokaz TV" + (" · DEMO" if demo else ""))
        self.window.geometry("720x500")
        self.window.minsize(650, 450)
        self.monitors = choose_monitors(self.window)
        self.monitor_names = [f"{i+1}. {m['name']} — {m['w']}×{m['h']}" +
                              (" (główny)" if m['primary'] else " (zewnętrzny)")
                              for i, m in enumerate(self.monitors)]
        self.screen_choice = tk.StringVar(value=self.monitor_names[1 if len(self.monitors)>1 else 0])
        self.block_choice = tk.StringVar(value=BLOCKS[0])
        self.variant_choice = tk.StringVar(value="R0")
        self.repeat_choice = tk.StringVar(value="r01")
        self.status = tk.StringVar(value="Wybierz monitor i wczytaj blok.")
        self.clock = tk.StringVar(value="—")
        self.mode = "IDLE"
        self.viewer = None
        self.canvas = None
        self.photos = []
        self.current_index = 0
        self.start_mono = None
        self.count_start = None
        self.run = None
        self.monitor = None
        self.pending_id = None
        self._last_countdown_number = None
        self._build_ui()
        self.window.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self):
        t, ttk = self.tk, self.ttk
        frame = ttk.Frame(self.window, padding=18)
        frame.pack(fill="both", expand=True)
        title = "F5A — Pokaz slajdów na TV" + (" (TRYB TESTOWY)" if self.demo else "")
        ttk.Label(frame, text=title, font=("Segoe UI", 17, "bold")).pack(anchor="w", pady=(0, 8))
        ttk.Label(frame, text="4 bloki × 10 obrazów · 30 s/obraz · 5 min/blok" if not self.demo else
                  "DEMO — 3 s/obraz, bez sesji badawczej", foreground="#6b7280").pack(anchor="w")
        forms = ttk.Frame(frame)
        forms.pack(fill="x", pady=(18, 8))
        for col, weight in enumerate([0, 1, 0, 1]):
            forms.columnconfigure(col, weight=weight)
        ttk.Label(forms, text="Monitor TV").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        self.monitor_cb = ttk.Combobox(forms, textvariable=self.screen_choice, values=self.monitor_names, state="readonly", width=49)
        self.monitor_cb.grid(row=0, column=1, columnspan=3, sticky="ew", pady=6)
        ttk.Label(forms, text="Blok").grid(row=1, column=0, sticky="w", pady=6)
        self.block_cb = ttk.Combobox(forms, textvariable=self.block_choice, values=BLOCKS, state="readonly", width=16)
        self.block_cb.grid(row=1, column=1, sticky="w", pady=6)
        ttk.Label(forms, text="Wariant").grid(row=1, column=2, sticky="e", padx=(10, 8), pady=6)
        self.variant_cb = ttk.Combobox(forms, textvariable=self.variant_choice, values=("R0", "R1", "R2"), state="readonly", width=10)
        self.variant_cb.grid(row=1, column=3, sticky="w", pady=6)
        ttk.Label(forms, text="Powtórzenie").grid(row=2, column=0, sticky="w", pady=6)
        self.repeat_cb = ttk.Combobox(forms, textvariable=self.repeat_choice, values=("r01", "r02", "r03"), state="readonly", width=16)
        self.repeat_cb.grid(row=2, column=1, sticky="w", pady=6)
        ttk.Label(forms, text="Ekspozycja").grid(row=2, column=2, sticky="e", padx=(10, 8), pady=6)
        ttk.Label(forms, text=f"{self.dwell} s × 10", font=("Segoe UI", 10, "bold")).grid(row=2, column=3, sticky="w", pady=6)
        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=(16, 12))
        self.prepare_btn = ttk.Button(actions, text="1. Przygotuj pokaz na TV", command=self.prepare)
        self.prepare_btn.pack(side="left", padx=(0, 8))
        self.start_btn = ttk.Button(actions, text="2. START (odliczanie)", command=self.begin_countdown, state="disabled")
        self.start_btn.pack(side="left", padx=(0, 8))
        self.abort_btn = ttk.Button(actions, text="PRZERWIJ", command=self.abort, state="disabled")
        self.abort_btn.pack(side="left")
        ttk.Label(frame, textvariable=self.status, font=("Segoe UI", 12, "bold"), wraplength=660).pack(anchor="w", pady=(14, 4))
        ttk.Label(frame, textvariable=self.clock, font=("Segoe UI", 20, "bold")).pack(anchor="w", pady=(0, 8))
        self.next_btn = ttk.Button(frame, text="Następny blok →", command=self.next_block, state="disabled")
        self.next_btn.pack(anchor="w", pady=(2, 10))
        ttk.Separator(frame).pack(fill="x", pady=10)
        ttk.Label(frame, text=(
            "Obsługa: najpierw ustaw telefon i TV. Kliknij START: 5→1→GO. "
            "Na GO rusza pierwsze zdjęcie — wtedy uruchom pomiar na Androidzie. "
            "Pokaz zmieni 9 zdjęć, zakończy się po 5 min i pozostawi ostatni obraz na ekranie. "
            "ESC = przerwanie; brak pauzy, żeby nie fałszować czasu."
            if not self.demo else
            "DEMO: tylko sprawdzenie działania. Nie uruchamiaj pomiaru Androida. "
            "ESC = przerwanie; 3 s na zdjęcie."
        ), wraplength=670, justify="left", foreground="#52525b").pack(anchor="w")

    def _select_monitor(self):
        try:
            return self.monitors[self.monitor_names.index(self.screen_choice.get())]
        except ValueError:
            stop("Nieprawidłowy monitor")

    def _sync_buttons(self):
        busy = self.mode in ("COUNTDOWN", "RUNNING")
        self.prepare_btn.configure(state="disabled" if busy else "normal")
        self.start_btn.configure(state="normal" if self.mode == "READY" else "disabled")
        self.abort_btn.configure(state="normal" if self.mode in ("READY", "COUNTDOWN", "RUNNING") else "disabled")
        self.next_btn.configure(state="normal" if self.mode == "COMPLETED" and self.block_choice.get()!=BLOCKS[-1] else "disabled")
        for cb in (self.monitor_cb, self.block_cb, self.variant_cb, self.repeat_cb):
            cb.configure(state="disabled" if busy else "readonly")

    def _dispose_viewer(self):
        if self.viewer is not None:
            try:
                self.viewer.destroy()
            except self.tk.TclError:
                pass
        self.viewer = None
        self.canvas = None
        self.photos = []

    def _make_viewer(self, screen):
        t = self.tk
        self.viewer = t.Toplevel(self.window)
        self.viewer.withdraw()
        self.viewer.overrideredirect(True)
        self.viewer.configure(background="black")
        self.viewer.geometry(f"{screen['w']}x{screen['h']}{screen['x']:+d}{screen['y']:+d}")
        self.viewer.attributes("-topmost", True)
        self.canvas = t.Canvas(self.viewer, highlightthickness=0, bg="black", bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.viewer.bind("<Escape>", lambda e: self.abort())
        self.viewer.bind("<space>", lambda e: self.begin_countdown())
        # Optional FINAL adapter: verify native placement while still hidden.
        if hasattr(self, "_position_viewer"):
            self._position_viewer(screen)
        self.viewer.deiconify()
        self.viewer.lift()
        self.viewer.focus_set()
        self.viewer.update_idletasks()

    def _show_photo(self, index):
        self.canvas.delete("all")
        self.canvas.create_image(self.monitor["w"]//2, self.monitor["h"]//2,
                                 image=self.photos[index], anchor="center")
        self.canvas.update_idletasks()

    def _show_countdown(self, number):
        self.canvas.delete("all")
        self.canvas.create_text(self.monitor["w"]//2, self.monitor["h"]//2,
                                text=str(number), fill="white", anchor="center",
                                font=("Segoe UI", max(60, self.monitor["h"]//5), "bold"))
        self.canvas.update_idletasks()

    def prepare(self):
        if self.mode in ("COUNTDOWN", "RUNNING"):
            return
        from tkinter import messagebox
        try:
            base, fresh_sha, fresh_blocks = validate_material(self.root_path)
            if fresh_sha != self.manifest_sha:
                stop("Zmienił się manifest bloków od uruchomienia aplikacji. Zamknij i ponownie sprawdź materiał.")
            self.blocks = fresh_blocks
            block = self.block_choice.get()
            screen = self._select_monitor()
            if block not in BLOCKS:
                stop("Nieznany blok")
            self._dispose_viewer()
            self.monitor = screen
            self._make_viewer(screen)
            self.photos = []
            for row in self.blocks[block]:
                with self.Image.open(row["_path"]) as opened:
                    pic = self.ImageOps.exif_transpose(opened)
                    if pic.mode != "RGB":
                        pic = pic.convert("RGB")
                    fit = self.ImageOps.contain(pic, (screen["w"], screen["h"]), method=self.Image.Resampling.LANCZOS)
                    self.photos.append(self.ImageTk.PhotoImage(fit, master=self.viewer))
            self.mode = "READY"
            self.current_index = 0
            self._show_photo(0)
            self.status.set(f"GOTOWE — {block}, obraz 01/10 na monitorze {screen['name']}. Ustaw kamerę, potem START.")
            total_s = round(self.dwell * 10)
            self.clock.set(f"{total_s//60:02d}:{total_s%60:02d} — planowany czas bloku")
            self._sync_buttons()
        except Exception as exc:
            self._dispose_viewer()
            self.mode = "IDLE"
            self.status.set("STOP — nie udało się przygotować pokazu")
            self._sync_buttons()
            messagebox.showerror("F5A — STOP", str(exc))

    def begin_countdown(self):
        if self.mode != "READY" or self.viewer is None or len(self.photos) != 10:
            return
        from tkinter import messagebox
        try:
            _, sha, _ = validate_material(self.root_path)
            if sha != self.manifest_sha:
                stop("Zmienił się manifest podczas przygotowania.")
        except Exception as exc:
            messagebox.showerror("F5A — STOP", str(exc))
            return
        self.mode = "COUNTDOWN"
        self.count_start = time.perf_counter()
        self.run = {
            "schema": "alpr.f5a.tv_presentation_log.v1", "experiment_id": "F5A-ROI-A12-01",
            "mode": "DEMO_NOT_FOR_RESEARCH" if self.demo else "NORMAL_SCHEDULE",
            "variant": self.variant_choice.get(), "replicate": self.repeat_choice.get(),
            "block": self.block_choice.get(), "monitor": self.monitor,
            "manifest_sha256": self.manifest_sha, "dwell_seconds": self.dwell,
            "countdown_seconds": self.countdown, "countdown_started_utc": utc_now(),
            "start_utc": None, "end_utc": None,
            "result": None, "events": [],
            "limitation": "Czasy zdarzeń mierzone w procesie PC (perf_counter), nie są pomiarem chwili odświeżenia panelu TV ani START Androida.",
        }
        self._show_countdown(self.countdown)
        self._last_countdown_number = self.countdown
        self.status.set("START ZA 5 s — przygotuj uruchomienie Androida" if not self.demo else "START TESTOWY")
        self._sync_buttons()
        self._schedule_tick()

    def _schedule_tick(self):
        self.pending_id = self.window.after(40, self._tick)

    def _tick(self):
        self.pending_id = None
        if self.mode == "COUNTDOWN":
            remaining = self.countdown - (time.perf_counter() - self.count_start)
            if remaining > 0:
                number = max(1, math.ceil(remaining))
                if number != self._last_countdown_number:
                    self._show_countdown(number)
                    self._last_countdown_number = number
                self.clock.set(f"Odliczanie: {number}")
                self._schedule_tick()
                return
            self.start_mono = time.perf_counter()
            self.run["start_utc"] = utc_now()
            self.current_index = 0
            self._show_photo(0)
            self.run["events"].append(self._event(0, 0.0))
            self.mode = "RUNNING"
            self.status.set("GO — uruchom START na Androidzie!  BLOK: " + self.block_choice.get())
            self._sound()
            self._sync_buttons()
            self._schedule_tick()
            return
        if self.mode != "RUNNING":
            return
        elapsed = time.perf_counter() - self.start_mono
        total = 10 * self.dwell
        if elapsed >= total:
            self._finish("COMPLETED")
            return
        expected_index = int(elapsed // self.dwell)
        if expected_index > self.current_index:
            if expected_index != self.current_index + 1:
                self._finish("ABORTED_SCHEDULER_DELAY", "Komputer nie zdążył zmienić zdjęć bez pominięcia pozycji.")
                return
            self.current_index = expected_index
            self._show_photo(self.current_index)
            self.run["events"].append(self._event(self.current_index, self.current_index*self.dwell))
        remaining_s = max(0, math.ceil(total - elapsed))
        self.clock.set(f"Pozostało: {remaining_s//60:02d}:{remaining_s%60:02d}   |   zdjęcie {self.current_index+1:02d}/10")
        self.status.set(f"TRWA — {self.block_choice.get()}, {self.variant_choice.get()}/{self.repeat_choice.get()}")
        self._schedule_tick()

    def _event(self, index, planned_time):
        row = self.blocks[self.block_choice.get()][index]
        elapsed = time.perf_counter()-self.start_mono
        return {"position": index+1, "filename": row["nazwa_w_pokazie"],
                "original_sha256": row["sha256"], "planned_elapsed_s": planned_time,
                "render_request_elapsed_s": round(elapsed, 4),
                "scheduler_delay_ms": round((elapsed-planned_time)*1000, 1),
                "utc": utc_now()}

    def _sound(self):
        if sys.platform == "win32":
            try:
                import winsound
                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
                return
            except Exception:
                pass
        try:
            self.window.bell()
        except Exception:
            pass

    def _persist_log(self):
        if not self.run:
            return ""
        root = self.root_path / "00_protokol" / "pokaz_TV_logi"
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = (f"TV_F5A_{self.run['variant']}_{self.run['replicate']}_{self.run['block']}_"
                f"{stamp}_{uuid.uuid4().hex[:8]}.json")
        path = root / name
        tmp = root / (name + ".tmp")
        try:
            with tmp.open("x", encoding="utf-8") as f:
                json.dump(self.run, f, ensure_ascii=False, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return str(path)

    def _finish(self, result, cause=""):
        if self.mode not in ("COUNTDOWN", "RUNNING"):
            return
        from tkinter import messagebox
        self.mode = "COMPLETED" if result == "COMPLETED" else "ABORTED"
        if self.run:
            self.run["result"] = result
            self.run["end_utc"] = utc_now()
            self.run["cause"] = cause
            self.run["total_elapsed_s"] = (round(time.perf_counter()-self.start_mono, 4)
                                           if self.start_mono is not None else None)
        log_path = None
        try:
            log_path = self._persist_log()
        except Exception as exc:
            self.status.set("STOP: nie udało się zapisać logu")
            messagebox.showerror("F5A — log", f"Nie udało się zapisać logu: {exc}")
        if result == "COMPLETED":
            self.status.set(f"KONIEC — {self.block_choice.get()} ukończony. Ostatnie zdjęcie pozostaje na TV.")
            self.clock.set(f"00:00 · 10/10 · {'DEMO' if self.demo else 'POKAZ ZAKOŃCZONY'}")
            self._sound()
        else:
            self.status.set(f"PRZERWANO — {cause or 'operator zatrzymał pokaz'}")
            self.clock.set("PRZERWANO — nie zaliczać jako kompletny blok")
        self._sync_buttons()
        if log_path and result == "COMPLETED":
            print(f"ZAPIS LOGU: {log_path}", flush=True)

    def abort(self):
        if self.mode in ("COUNTDOWN", "RUNNING"):
            self._finish("ABORTED_BY_OPERATOR", "ESC / przycisk PRZERWIJ")
        elif self.mode == "READY":
            self._dispose_viewer()
            self.mode = "IDLE"
            self.status.set("Podgląd zamknięty bez uruchomienia pokazu.")
            self._sync_buttons()

    def next_block(self):
        if self.mode != "COMPLETED":
            return
        current = BLOCKS.index(self.block_choice.get())
        if current+1 < len(BLOCKS):
            self._dispose_viewer()
            self.mode = "IDLE"
            self.block_choice.set(BLOCKS[current+1])
            self.status.set(f"Wybrano {BLOCKS[current+1]}. Przygotuj i uruchom osobną sesję Androida.")
            self.clock.set("—")
            self._sync_buttons()

    def _close(self):
        if self.mode in ("COUNTDOWN", "RUNNING"):
            from tkinter import messagebox
            if not messagebox.askyesno("F5A — przerwać?", "Pokaz trwa. Przerwać i oznaczyć go jako niekompletny?"):
                return
            self._finish("ABORTED_BY_OPERATOR", "Zamknięto okno sterowania")
        if self.pending_id:
            try:
                self.window.after_cancel(self.pending_id)
            except Exception:
                pass
        self._dispose_viewer()
        self.window.destroy()

    def run_mainloop(self):
        self.window.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Automatyczny pokaz zdjęć F5A na TV")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Katalog eksperymentu F5A-ROI-A12-01")
    parser.add_argument("--check", action="store_true", help="Sprawdź 40 zdjęć i manifest bez uruchamiania GUI")
    parser.add_argument("--demo", action="store_true", help="Test 3 s/obraz, bez pomiaru Androida")
    args = parser.parse_args()
    root_path = args.root.expanduser()
    base, manifest_sha, grouped = validate_material(root_path)
    if args.check:
        print("CHECK PASS — 40/40 fotografii, 4 bloki A/B/C/D po 10, SHA zgodne z manifestem.")
        print(f"KATALOG: {base}")
        print(f"MANIFEST SHA-256: {manifest_sha}")
        for block in BLOCKS:
            print(f"  {block}: {len(grouped[block])} obrazów; " +
                  ", ".join(row["koszyk"] for row in grouped[block]))
        return 0
    try:
        import tkinter
        from PIL import Image, ImageTk, ImageOps
    except ImportError as exc:
        stop(f"Brak biblioteki GUI: {exc}. Jeśli brakuje Pillow, uruchom: py -3 -m pip install pillow")
    TvApp(root_path, args.demo).run_mainloop()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        sys.exit(2)
