"""Opt-in native Windows probe; shows and closes only its own test windows."""

import ctypes
from ctypes import wintypes
from types import SimpleNamespace, MethodType
import tkinter as tk
from tkinter import ttk
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_annotation_tool.gui import app_theme_runtime, app_window_recovery
from auto_annotation_tool.gui.app_theme_definitions import get_theme_palette


def main():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.restype = wintypes.LONG
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostMessageW.restype = wintypes.BOOL

    root = tk.Tk()
    callback_errors = []
    root.report_callback_exception = lambda kind, error, tb: callback_errors.append((kind, error))
    print("OS:", sys.getwindowsversion(), "Tk:", root.tk.call("package", "provide", "Tk"), flush=True)
    root.title("ALPR - isolated minimize test")
    root.geometry("720x400+50+50")
    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True)
    app = SimpleNamespace(root=root, palette=get_theme_palette(), _iter_loaded_tabs=lambda: [])
    app._center_dialog_window = MethodType(app_theme_runtime._center_dialog_window, app)
    root.bind("<FocusIn>", lambda event: app_window_recovery.on_root_focus_in(app, event), add="+")
    dialog = tk.Toplevel(frame)
    if "--disable-bridge" in sys.argv:
        dialog._aat_native_minimize_unavailable = True
    app_window_recovery.configure_minimizable_modal(dialog)
    dialog.withdraw()
    app_theme_runtime.style_dialog_window(app, dialog, "Zasoby bramki T01 - test", "640x320", parent=frame)
    dialog.withdraw()
    dialog.grab_release()
    dialog.resizable(True, True)
    dialog.minsize(400, 240)
    dialog.wm_transient("")
    body = app_theme_runtime._build_themed_dialog_surface(app, dialog)
    tk.Label(body, text="Native minimize probe", bg=app.palette["panel"]).pack(pady=20)
    dialog.update_idletasks()
    dialog.wm_transient("")
    dialog.deiconify()
    dialog.lift()
    dialog.update()
    dialog.focus_force()
    dialog.grab_set()
    app._free_mode_assistant_context_override_owner = dialog
    outcomes = []

    def snapshot(label):
        hwnd = user32.GetAncestor(dialog.winfo_id(), 2)
        result = dict(label=label, state=dialog.state(), iconic=bool(user32.IsIconic(hwnd)),
                      transient=dialog.transient(), grab=str(root.grab_current()),
                      root_state=root.state(),
                      style=hex(user32.GetWindowLongW(hwnd, -16) & 0xffffffff))
        outcomes.append(result)
        print(result, flush=True)

    for event in ("<Map>", "<Unmap>", "<FocusIn>", "<FocusOut>"):
        dialog.bind(event, lambda e, name=event: snapshot(name) if e.widget is dialog else None, add="+")
    def system_command(command):
        hwnd = user32.GetAncestor(dialog.winfo_id(), 2)
        if not user32.PostMessageW(hwnd, 0x0112, command, 0):
            raise ctypes.WinError(ctypes.get_last_error())

    def recreate_wrapper():
        dialog.wm_transient(root)
        dialog.wm_transient("")

    for cycle in range(3):
        start = 700 + cycle*1800
        if cycle == 1:
            root.after(start-300, lambda: system_command(0xF030))  # maximize
        if cycle == 2:
            root.after(start-300, recreate_wrapper)
        root.after(start-100, lambda c=cycle: snapshot(f"before-{c}"))
        if "--release-grab" in sys.argv:
            root.after(start-50, dialog.grab_release)
        root.after(start, lambda: system_command(0xF020))
        for offset in (200, 600):
            root.after(start+offset, lambda c=cycle, t=offset: snapshot(f"minimized-{c}-{t}"))
        root.after(start+900, lambda: system_command(0xF120))  # native restore
        root.after(start+1200, lambda c=cycle: snapshot(f"restored-{c}"))
    child_window = []

    def open_child():
        child = tk.Toplevel(dialog)
        child.title("Child resource test")
        child.geometry("300x120+120+120")
        child.transient(dialog)
        child.grab_set()
        child_window.append(child)

    def close_child():
        child_window[0].grab_release()
        child_window[0].destroy()
        app_window_recovery.restore_visible_modal(dialog)

    root.after(5600, open_child)
    root.after(5800, lambda: system_command(0xF020))
    root.after(6100, lambda: snapshot("child-blocks-parent"))
    root.after(6200, close_child)
    root.after(6400, lambda: snapshot("child-closed"))
    root.after(6600, lambda: system_command(0xF020))
    root.after(6601, dialog.destroy)  # close with a native request pending
    root.after(7000, root.destroy)
    root.mainloop()
    minimized = [row for row in outcomes if row["label"].startswith("minimized-")]
    restored = [row for row in outcomes if row["label"].startswith("restored-")]
    passed = len(minimized) == 6 and len(restored) == 3
    passed = passed and all(row["iconic"] and row["state"] == "iconic" and row["grab"] == "None"
                            and row["root_state"] != "iconic" for row in minimized)
    passed = passed and all(not row["iconic"] and row["state"] in {"normal", "zoomed"}
                            and row["grab"] == str(dialog) for row in restored)
    protected = [row for row in outcomes if row["label"] == "child-blocks-parent"]
    passed = passed and len(protected) == 1 and not protected[0]["iconic"] and protected[0]["grab"] == str(child_window[0])
    returned = [row for row in outcomes if row["label"] == "child-closed"]
    passed = passed and len(returned) == 1 and returned[0]["grab"] == str(dialog)
    passed = passed and not callback_errors
    print("Callback errors:", callback_errors, flush=True)
    print("NATIVE PROBE:", "PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
