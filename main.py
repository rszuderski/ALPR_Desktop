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

# Ta optymalizacja alokatora nie jest obsługiwana przez używane buildy PyTorch
# na Windows. Włączamy ją domyślnie tylko na Linuksie, przed importem torch.
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    try:
        from alpr_runtime import initialize_runtime
        initialize_runtime()
        # The dependency bootstrap can itself create Tk windows. Select native
        # Windows pixels before that happens, not only in the Canvas test host.
        from windows_dpi import enable_windows_dpi_awareness
        enable_windows_dpi_awareness()

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
        from alpr_runtime import report_startup_error
        report_startup_error(e)
        return 1
    return 0

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    from alpr_runtime import initialize_runtime, run_worker_if_requested
    initialize_runtime()
    if len(sys.argv)==3 and sys.argv[1]=='--alpr-acceptance':
        from release_acceptance import main as acceptance_main
        raise SystemExit(acceptance_main(sys.argv[2]))
    elif not run_worker_if_requested():
        raise SystemExit(main())
