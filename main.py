#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import traceback


for env_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

# Pomaga ograniczyć fragmentację alokacji CUDA w dłuższych sesjach treningowych.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    try:
        from dependency_bootstrap import ensure_runtime_dependencies, get_last_probe_terminal_entries
        if not ensure_runtime_dependencies(app_argv=list(sys.argv)):
            return

        # Importujemy Twoją aplikację jako zewnętrzny moduł
        import tkinter as tk
        from auto_annotation_tool.gui.app import AutoAnnotationApp
        from auto_annotation_tool.config import CONFIG
        
        # Inicjalizacja głównego okna
        root = tk.Tk()
        
        # Ustawienia początkowe okna
        root.geometry("1400x900")
        root.minsize(1024, 768)
        root.title(f"{CONFIG.APP_NAME} ver. {CONFIG.VERSION}")
        
        # Odpalenie interfejsu
        app = AutoAnnotationApp(root)
        try:
            bootstrap_entries = list(get_last_probe_terminal_entries() or [])
            if bootstrap_entries and hasattr(app, "append_global_terminal_entries"):
                app.append_global_terminal_entries(bootstrap_entries)
        except Exception:
            pass
        
        # Start pętli zdarzeń
        try:
            root.mainloop()
        finally:
            try:
                shutdown_cleanup = getattr(app, "_release_runtime_references", None)
                if callable(shutdown_cleanup):
                    shutdown_cleanup()
            except Exception:
                pass
            try:
                from auto_annotation_tool.utils import cleanup_gpu_memory
                cleanup_gpu_memory()
            except Exception:
                pass
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ KRYTYCZNY BŁĄD URUCHOMIENIA APLIKACJI ❌")
        print("="*60)
        traceback.print_exc()
        print("="*60)
        input("Naciśnij ENTER, aby zamknąć...")

if __name__ == "__main__":
    main()
