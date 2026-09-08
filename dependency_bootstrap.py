#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lekki bootstrap zależności uruchamiany przed importem pełnego GUI.
Dzięki temu aplikacja może jasno zgłosić brakujące biblioteki i spróbować
je doinstalować w interpreterze, z którego została uruchomiona.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = ROOT_DIR / "requirements-runtime.txt"
LAST_PROBE_LINES: list[str] = []
LAST_PROBE_ISSUES: list["DependencyIssue"] = []


@dataclass(frozen=True)
class DependencySpec:
    module_name: str
    package_name: str | None
    display_name: str
    required_for_startup: bool
    note: str = ""


@dataclass(frozen=True)
class DependencyIssue:
    spec: DependencySpec
    error_text: str

    @property
    def is_system_resource_load_error(self) -> bool:
        return _is_system_resource_load_error(self.error_text)

    @property
    def can_auto_install(self) -> bool:
        if self.is_system_resource_load_error:
            return False
        return bool(str(self.spec.package_name or "").strip())


def _is_system_resource_load_error(error_text: str) -> bool:
    normalized = str(error_text or "").lower()
    markers = (
        "winerror 1455",
        "plik stronicowania jest za mały",
        "plik stronicowania jest za maly",
        "paging file is too small",
        "page file is too small",
    )
    return any(marker in normalized for marker in markers)


def _format_dependency_probe_line(spec: DependencySpec, *, ok: bool, error_text: str = "") -> str:
    scope = "start" if bool(spec.required_for_startup) else "opcjonalne"
    package_label = str(spec.package_name or "").strip()
    if package_label:
        package_label = f" | pip={package_label}"
    if ok:
        return f"[BOOTSTRAP] {spec.display_name}: OK | {scope}{package_label}"
    status = "NIEDOSTĘPNE" if _is_system_resource_load_error(error_text) else "BRAK"
    return (
        f"[BOOTSTRAP] {spec.display_name}: {status} | {scope}{package_label} | "
        f"{str(error_text or '').strip()}"
    )


def _probe_line_to_terminal_entry(line: str) -> dict:
    text = str(line or "").strip()
    upper = text.upper()
    if " BRAK " in upper or upper.endswith(": BRAK") or " NIEDOSTĘPNE " in upper or upper.endswith(": NIEDOSTĘPNE"):
        tag = "terminal_warning"
    elif " OK " in upper or upper.endswith(": OK"):
        tag = "terminal_success"
    else:
        tag = "terminal_info"
    return {"text": text, "tag": tag}


def get_last_probe_terminal_entries() -> list[dict]:
    entries: list[dict] = []
    lines = list(LAST_PROBE_LINES or [])
    if not lines:
        return entries
    entries.append({"text": "[BOOTSTRAP] Raport środowiska z etapu startu aplikacji:", "tag": "terminal_header"})
    for line in lines:
        entries.append(_probe_line_to_terminal_entry(line))
    return entries


DEPENDENCY_SPECS: tuple[DependencySpec, ...] = (
    DependencySpec(
        module_name="tkinter",
        package_name=None,
        display_name="Tkinter (GUI)",
        required_for_startup=True,
        note="Tkinter jest częścią instalacji Pythona. Na nowej maszynie trzeba mieć Pythona z komponentem Tcl/Tk.",
    ),
    DependencySpec(
        module_name="numpy",
        package_name="numpy",
        display_name="NumPy",
        required_for_startup=True,
        note="Podstawa dla operacji na obrazach i danych anotacji.",
    ),
    DependencySpec(
        module_name="cv2",
        package_name="opencv-python",
        display_name="OpenCV",
        required_for_startup=True,
        note="Potrzebny do podglądu, geometrii polygonów i przetwarzania obrazów.",
    ),
    DependencySpec(
        module_name="PIL",
        package_name="Pillow",
        display_name="Pillow",
        required_for_startup=True,
        note="Potrzebny do canvasa, miniaturek i podglądu obrazów.",
    ),
    DependencySpec(
        module_name="yaml",
        package_name="PyYAML",
        display_name="PyYAML",
        required_for_startup=False,
        note="Potrzebny do datasetów YOLO, data.yaml i części walidacji treningu.",
    ),
    DependencySpec(
        module_name="torch",
        package_name="torch",
        display_name="PyTorch",
        required_for_startup=False,
        note="Potrzebny do inferencji i treningu modeli YOLO. Domyślnie pip zainstaluje wariant CPU, jeśli nie ma specjalnej konfiguracji GPU.",
    ),
    DependencySpec(
        module_name="ultralytics",
        package_name="ultralytics",
        display_name="Ultralytics YOLO",
        required_for_startup=False,
        note="Potrzebny do modeli YOLO w Z2, Z3 i Z4.",
    ),
    DependencySpec(
        module_name="easyocr",
        package_name="easyocr",
        display_name="EasyOCR",
        required_for_startup=False,
        note="Potrzebny do OCR znaków tablic w Z3.",
    ),
)

HEAVY_OPTIONAL_MODULES = {"torch", "ultralytics", "easyocr"}


def probe_runtime_dependencies(line_callback=None) -> list[DependencyIssue]:
    global LAST_PROBE_LINES, LAST_PROBE_ISSUES
    issues: list[DependencyIssue] = []
    probe_lines: list[str] = []

    def emit(line: str) -> None:
        probe_lines.append(str(line or ""))
        if callable(line_callback):
            line_callback(line)

    for spec in DEPENDENCY_SPECS:
        try:
            if (not bool(spec.required_for_startup)) and spec.module_name in HEAVY_OPTIONAL_MODULES:
                # Nie ładujemy tu natywnych DLL PyTorch/CUDA/EasyOCR. Sam import potrafi
                # zużyć dużo pagefile na Windowsie jeszcze przed startem GUI.
                if importlib.util.find_spec(spec.module_name) is None:
                    raise ImportError(f"No module named '{spec.module_name}'")
            else:
                importlib.import_module(spec.module_name)
            emit(_format_dependency_probe_line(spec, ok=True))
        except Exception as exc:  # pragma: no cover - zależne od środowiska
            error_text = f"{type(exc).__name__}: {exc}"
            emit(_format_dependency_probe_line(spec, ok=False, error_text=error_text))
            issues.append(
                DependencyIssue(
                    spec=spec,
                    error_text=error_text,
                )
            )
    LAST_PROBE_LINES = probe_lines
    LAST_PROBE_ISSUES = list(issues)
    return issues


def _get_blocking_issues(issues: Iterable[DependencyIssue]) -> list[DependencyIssue]:
    return [issue for issue in issues if bool(issue.spec.required_for_startup)]


def _get_auto_install_packages(issues: Iterable[DependencyIssue]) -> list[str]:
    packages: list[str] = []
    for issue in issues:
        if not issue.can_auto_install:
            continue
        package_name = str(issue.spec.package_name or "").strip()
        if package_name and package_name not in packages:
            packages.append(package_name)
    return packages


def _format_windows_command(parts: list[str]) -> str:
    try:
        return subprocess.list2cmdline(parts)
    except Exception:
        return " ".join(parts)


def build_install_command(
    issues: Iterable[DependencyIssue] | None = None,
    *,
    use_requirements_file: bool = False,
) -> str:
    cmd = [sys.executable, "-m", "pip", "install"]
    if use_requirements_file:
        cmd.extend(["-r", str(REQUIREMENTS_FILE)])
    else:
        packages = _get_auto_install_packages(list(issues or []))
        if not packages:
            return ""
        cmd.extend(packages)
    return _format_windows_command(cmd)


def _build_system_resource_report(issues: list[DependencyIssue]) -> str:
    lines: list[str] = []
    lines.append("Wykryto problem z załadowaniem bibliotek środowiska dla Auto-Annotation Tool.")
    lines.append("")
    lines.append(f"Interpreter: {sys.executable}")
    lines.append(f"Folder aplikacji: {ROOT_DIR}")
    lines.append("")
    lines.append("To nie wygląda na brak instalacji pakietów.")
    lines.append("Biblioteki są zainstalowane albo wykryte, ale Windows nie może załadować natywnych DLL PyTorcha.")
    lines.append("")
    lines.append("Niedostępne moduły:")
    for issue in issues:
        package_label = str(issue.spec.package_name or "").strip()
        if package_label:
            package_label = f" | pip: {package_label}"
        lines.append(f"- {issue.spec.display_name} (import: {issue.spec.module_name}{package_label})")
        lines.append(f"  Błąd: {issue.error_text}")
    lines.append("")
    lines.append("Najbardziej prawdopodobna przyczyna:")
    lines.append("- Plik stronicowania Windows jest za mały dla bibliotek PyTorch/CUDA (WinError 1455).")
    lines.append("")
    lines.append("Co zrobić:")
    lines.append("- Nie uruchamiaj teraz `pip install torch ultralytics easyocr`, bo to najpewniej nie naprawi problemu.")
    lines.append("- Zamknij ciężkie aplikacje i zwiększ plik stronicowania Windows albo ustaw go jako zarządzany przez system.")
    lines.append("- Po zmianie zrestartuj komputer i uruchom aplikację ponownie w tym samym interpreterze.")
    lines.append("- Dopiero jeśli WinError 1455 zniknie, a import nadal będzie padał, sprawdzamy instalację PyTorch.")
    return "\n".join(lines)


def _build_dependency_report(issues: list[DependencyIssue]) -> str:
    if issues and all(issue.is_system_resource_load_error for issue in issues):
        return _build_system_resource_report(issues)

    blockers = _get_blocking_issues(issues)
    optional = [issue for issue in issues if issue not in blockers]
    lines: list[str] = []
    lines.append("Wykryto brakujące biblioteki środowiska dla Auto-Annotation Tool.")
    lines.append("")
    lines.append(f"Interpreter: {sys.executable}")
    lines.append(f"Folder aplikacji: {ROOT_DIR}")
    lines.append("")

    if blockers:
        lines.append("Braki blokujące start aplikacji:")
        for issue in blockers:
            package_label = str(issue.spec.package_name or "").strip()
            if package_label:
                package_label = f" | pip: {package_label}"
            lines.append(
                f"- {issue.spec.display_name} (import: {issue.spec.module_name}{package_label})"
            )
            lines.append(f"  Błąd: {issue.error_text}")
            if issue.spec.note:
                lines.append(f"  Uwaga: {issue.spec.note}")
        lines.append("")

    if optional:
        lines.append("Braki funkcjonalne, które nie muszą blokować samego startu, ale ograniczą część modułów:")
        for issue in optional:
            package_label = str(issue.spec.package_name or "").strip()
            if package_label:
                package_label = f" | pip: {package_label}"
            lines.append(
                f"- {issue.spec.display_name} (import: {issue.spec.module_name}{package_label})"
            )
            lines.append(f"  Błąd: {issue.error_text}")
            if issue.spec.note:
                lines.append(f"  Uwaga: {issue.spec.note}")
        lines.append("")

    auto_packages = _get_auto_install_packages(issues)
    if auto_packages:
        lines.append("Szybka komenda dla brakujących pakietów:")
        lines.append(build_install_command(issues))
        lines.append("")
        if REQUIREMENTS_FILE.exists():
            lines.append("Alternatywa pełnej instalacji runtime:")
            lines.append(build_install_command(use_requirements_file=True))
            lines.append("")

    manual_issues = [issue for issue in issues if not issue.can_auto_install and not issue.is_system_resource_load_error]
    if manual_issues:
        lines.append("Pakiety wymagające ręcznego przygotowania środowiska:")
        for issue in manual_issues:
            lines.append(f"- {issue.spec.display_name}: {issue.spec.note or 'Brak automatycznej instalacji przez pip.'}")
        lines.append("")

    lines.append("Po instalacji aplikacja może zostać uruchomiona ponownie w tym samym interpreterze.")
    return "\n".join(lines)


def _restart_current_process(app_argv: list[str]) -> None:
    argv = [sys.executable] + list(app_argv or sys.argv)
    os.execv(sys.executable, argv)


def _run_pip_install(packages: list[str], line_callback) -> int:
    if not packages:
        return 0

    cmd = [sys.executable, "-m", "pip", "install", *packages]
    line_callback(f"> {_format_windows_command(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        if process.stdout is not None:
            for line in process.stdout:
                line_callback(line.rstrip())
    finally:
        return process.wait()


def _show_cli_dependency_assistant(issues: list[DependencyIssue], app_argv: list[str]) -> bool:
    report = _build_dependency_report(issues)
    print("\n" + "=" * 72)
    print(report)
    print("=" * 72 + "\n")

    auto_packages = _get_auto_install_packages(issues)
    blockers = _get_blocking_issues(issues)

    if auto_packages:
        try:
            answer = input("Czy zainstalować brakujące pakiety teraz w tym interpreterze? [t/N]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer in {"t", "tak", "y", "yes"}:
            return_code = _run_pip_install(auto_packages, print)
            print("\n[BOOTSTRAP] Ponowne sprawdzenie środowiska po instalacji:")
            refreshed = probe_runtime_dependencies(print)
            if return_code == 0 and not refreshed:
                print("\nInstalacja zakończona powodzeniem. Uruchamiam aplikację ponownie...\n")
                _restart_current_process(app_argv)
                return False

            refreshed_blockers = _get_blocking_issues(refreshed)
            if not refreshed_blockers:
                print("\nBraki blokujące start zostały usunięte.")
                try:
                    answer = input("Uruchomić aplikację mimo pozostałych braków opcjonalnych? [T/n]: ").strip().lower()
                except EOFError:
                    answer = ""
                return answer not in {"n", "nie", "no"}

    if blockers:
        return False

    try:
        answer = input("Uruchomić aplikację mimo braków opcjonalnych? [T/n]: ").strip().lower()
    except EOFError:
        answer = ""
    return answer not in {"n", "nie", "no"}


def _show_tk_dependency_assistant(issues: list[DependencyIssue], app_argv: list[str]) -> bool:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk

    result = {"continue": False}
    system_resource_only = bool(issues) and all(issue.is_system_resource_load_error for issue in issues)
    root = tk.Tk()
    root.title("Brakujące biblioteki środowiska")
    root.geometry("920x700")
    if system_resource_only:
        root.title("Problem ładowania bibliotek środowiska")
    root.minsize(760, 560)

    try:
        root.iconname("Auto-Annotation Tool")
    except Exception:
        pass

    shell = ttk.Frame(root, padding=14)
    shell.pack(fill=tk.BOTH, expand=True)

    title = ttk.Label(
        shell,
        text="Ta maszyna nie ma pełnego środowiska dla Auto-Annotation Tool",
        font=("Segoe UI", 12, "bold"),
        anchor=tk.W,
        justify=tk.LEFT,
    )
    title.pack(fill=tk.X)
    if system_resource_only:
        title.configure(text="Windows nie może załadować bibliotek PyTorch")

    intro = ttk.Label(
        shell,
        text=(
            "Aplikacja wykryła brakujące biblioteki. Możesz je doinstalować w bieżącym interpreterze "
            "albo skopiować gotową komendę pip."
        ),
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=860,
    )
    intro.pack(fill=tk.X, pady=(8, 10))
    if system_resource_only:
        intro.configure(
            text=(
                "To wygląda na za mały plik stronicowania Windows, a nie na brak pakietów. "
                "Najpierw popraw pamięć wirtualną systemu, potem uruchom aplikację ponownie."
            )
        )

    report_box = scrolledtext.ScrolledText(
        shell,
        wrap=tk.WORD,
        font=("Consolas", 10),
        height=24,
    )
    report_box.pack(fill=tk.BOTH, expand=True)
    report_box.insert("1.0", _build_dependency_report(issues))
    report_box.configure(state=tk.DISABLED)

    status_var = tk.StringVar(
        value="Status: oczekiwanie na decyzję użytkownika."
    )
    status_lbl = ttk.Label(
        shell,
        textvariable=status_var,
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=860,
    )
    status_lbl.pack(fill=tk.X, pady=(10, 0))

    btn_row = ttk.Frame(shell)
    btn_row.pack(fill=tk.X, pady=(12, 0))

    install_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    install_in_progress = {"value": False}
    issues_state = {"items": list(issues)}

    def append_report(text: str) -> None:
        report_box.configure(state=tk.NORMAL)
        report_box.insert(tk.END, text.rstrip() + "\n")
        report_box.see(tk.END)
        report_box.configure(state=tk.DISABLED)

    def copy_install_command() -> None:
        command = build_install_command(issues_state["items"])
        if not command.strip():
            messagebox.showinfo(
                "Brak komendy",
                "Dla aktualnych braków nie ma komendy pip do skopiowania. Sprawdź sekcję z uwagami ręcznymi.",
                parent=root,
            )
            return
        try:
            root.clipboard_clear()
            root.clipboard_append(command)
            status_var.set("Status: skopiowano komendę pip do schowka.")
        except Exception as exc:
            status_var.set(f"Status: nie udało się skopiować komendy ({exc}).")

    def close_dialog() -> None:
        root.destroy()

    def continue_with_optional_gaps() -> None:
        result["continue"] = True
        root.destroy()

    def restart_after_success() -> None:
        root.destroy()
        _restart_current_process(app_argv)

    def refresh_button_states(current_issues: list[DependencyIssue] | None = None) -> None:
        active_issues = list(current_issues or issues_state["items"])
        blockers = _get_blocking_issues(active_issues)
        can_install = bool(_get_auto_install_packages(active_issues)) and not install_in_progress["value"]
        btn_install.configure(state=("normal" if can_install else "disabled"))
        btn_copy.configure(
            state=("normal" if bool(_get_auto_install_packages(active_issues)) and not install_in_progress["value"] else "disabled")
        )
        btn_close.configure(state=("normal" if not install_in_progress["value"] else "disabled"))
        btn_continue.configure(
            state=(
                "normal"
                if (not blockers and not install_in_progress["value"])
                else "disabled"
            )
        )

    def poll_install_queue() -> None:
        try:
            while True:
                event_type, payload = install_queue.get_nowait()
                if event_type == "line":
                    append_report(payload)
                elif event_type == "done":
                    install_in_progress["value"] = False
                    print("\n[BOOTSTRAP] Ponowne sprawdzenie środowiska po instalacji:")
                    refreshed_issues = probe_runtime_dependencies(print)
                    issues_state["items"] = list(refreshed_issues)
                    refresh_button_states(refreshed_issues)
                    if payload == "0" and not refreshed_issues:
                        status_var.set("Status: instalacja zakończona powodzeniem. Można uruchomić aplikację ponownie.")
                        append_report("Instalacja zakończona powodzeniem. Wszystkie zależności runtime są już dostępne.")
                        if messagebox.askyesno(
                            "Instalacja zakończona",
                            "Biblioteki zostały doinstalowane. Uruchomić aplikację ponownie teraz?",
                            parent=root,
                        ):
                            restart_after_success()
                            return
                    else:
                        refreshed_blockers = _get_blocking_issues(refreshed_issues)
                        if not refreshed_blockers:
                            status_var.set(
                                "Status: braki blokujące start zostały usunięte. Możesz uruchomić aplikację mimo pozostałych braków opcjonalnych."
                            )
                        else:
                            status_var.set(
                                "Status: część zależności nadal jest niedostępna. Sprawdź log instalacji poniżej."
                            )
        except queue.Empty:
            pass

        if root.winfo_exists():
            root.after(120, poll_install_queue)

    def install_missing_packages() -> None:
        packages = _get_auto_install_packages(issues_state["items"])
        if not packages:
            messagebox.showinfo(
                "Brak pakietów do instalacji",
                "Aktualne braki wymagają ręcznej konfiguracji środowiska albo nie ma czego instalować przez pip.",
                parent=root,
            )
            return

        install_in_progress["value"] = True
        refresh_button_states(issues_state["items"])
        status_var.set("Status: trwa instalacja pakietów przez pip...")
        append_report("")
        append_report("=" * 72)
        append_report("Rozpoczynam instalację brakujących pakietów runtime...")

        def worker() -> None:
            return_code = _run_pip_install(
                packages,
                lambda line: install_queue.put(("line", line)),
            )
            install_queue.put(("done", str(return_code)))

        threading.Thread(target=worker, daemon=True).start()

    btn_install = ttk.Button(
        btn_row,
        text="Zainstaluj brakujące biblioteki",
        command=install_missing_packages,
    )
    btn_install.pack(side=tk.LEFT)
    if system_resource_only:
        btn_install.configure(text="Pip nie naprawi WinError 1455")

    btn_copy = ttk.Button(
        btn_row,
        text="Kopiuj komendę pip",
        command=copy_install_command,
    )
    btn_copy.pack(side=tk.LEFT, padx=(8, 0))

    btn_continue = ttk.Button(
        btn_row,
        text="Uruchom mimo braków opcjonalnych",
        command=continue_with_optional_gaps,
    )
    btn_continue.pack(side=tk.RIGHT, padx=(8, 0))
    if system_resource_only:
        btn_continue.configure(text="Uruchom bez modułów ML")

    btn_close = ttk.Button(
        btn_row,
        text="Zamknij",
        command=close_dialog,
    )
    btn_close.pack(side=tk.RIGHT)

    refresh_button_states(issues)
    root.after(120, poll_install_queue)
    root.mainloop()
    return bool(result["continue"])


def ensure_runtime_dependencies(app_argv: list[str] | None = None) -> bool:
    issues = probe_runtime_dependencies(print)
    if not issues:
        return True

    argv = list(app_argv or sys.argv)
    try:
        import tkinter  # noqa: F401
    except Exception:
        return _show_cli_dependency_assistant(issues, argv)

    return _show_tk_dependency_assistant(issues, argv)
