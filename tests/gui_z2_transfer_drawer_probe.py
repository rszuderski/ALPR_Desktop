"""Opt-in GUI smoke test. Simulates transfer; never downloads/loads model weights."""
from pathlib import Path
import sys
import threading
import time
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import ImageGrab
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette
from auto_annotation_tool.gui import model_download_dialog as download_ui
from auto_annotation_tool.gui.z2_drawer_slide import PreviewDrawerSlide
from auto_annotation_tool.model_download import DownloadProgress, DownloadCancelled


def main():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    root = tk.Tk()
    root.title("Z2 UI probe (no project data)")
    root.geometry("960x640+80+60")
    palette = get_theme_palette("light_visual_cs")
    root.configure(bg=palette["panel"])
    host = tk.Frame(root, bg=palette["panel"])
    host.pack(fill="both", expand=True)
    canvas = tk.Canvas(host, bg=palette["field"], highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    panel = tk.Frame(host, bg=palette["panel"], highlightbackground=palette["border"], highlightthickness=1)
    tk.Label(panel, text="SZUFLADA Z2", bg=palette["panel"], fg=palette["fg"]).pack(fill="x", pady=12)
    status = tk.Label(panel, text="T04 | 23 tablice [OK]", bg=palette["field"], fg=palette["fg"], pady=15)
    status.pack(fill="x")
    owner = SimpleNamespace(frame=host, canvas_frame=host, preview_canvas=canvas,
                            preview_overlay_dock=panel, app=SimpleNamespace(palette=palette))
    slide = PreviewDrawerSlide(owner)
    errors, heartbeats = [], []
    root.report_callback_exception = lambda *args: errors.append(str(args))
    out = Path("output/z2_ui_probe")
    out.mkdir(parents=True, exist_ok=True)

    def heartbeat():
        heartbeats.append(time.monotonic())
        root.after(40, heartbeat)

    def capture(name):
        ImageGrab.grab(bbox=(root.winfo_rootx(), root.winfo_rooty(),
                            root.winfo_rootx() + root.winfo_width(), root.winfo_rooty() + root.winfo_height())).save(out / name)

    def transfer(url, target, cancel, progress):
        assert threading.current_thread() is not threading.main_thread()
        for i in range(21):
            if cancel.is_set():
                raise DownloadCancelled()
            progress(DownloadProgress(url, i * 1048576, 20 * 1048576, 10 * 1048576, i == 20))
            time.sleep(0.08)
        return target

    def run():
        try:
            slide.place(960, 680, 55, 270, 340, fullscreen=True)
            def wait_paint():
                ready = tk.BooleanVar(root, False)
                root.after(350, lambda: ready.set(True))
                root.wait_variable(ready)
            wait_paint()
            capture("drawer.png")
            slide.toggle()
            wait_paint()
            capture("drawer_hidden.png")
            slide.toggle()
            wait_paint()
            before = len(heartbeats)
            root.after(1200, lambda: capture("download.png"))
            with patch.object(download_ui, "official_asset_url", return_value="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"), \
                 patch.object(download_ui, "download_model", side_effect=transfer):
                assert download_ui.download_models_with_splash(owner, [{"label": "YOLO26n", "asset_name": "yolo26n.pt", "target_path": out / "never_written.pt"}])
            assert len(heartbeats) - before >= 20, "UI heartbeat stalled"
            assert status.cget("text") == "T04 | 23 tablice [OK]"
            assert panel.winfo_manager() == "place"
            assert not errors, errors
            print(f"PASS: drawer, transfer, {len(heartbeats) - before} UI ticks; callback errors: {errors}")
        finally:
            root.destroy()

    root.after(0, heartbeat)
    root.after(150, run)
    root.mainloop()


if __name__ == "__main__":
    main()
