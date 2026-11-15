"""Explicit Canvas decisions, including MT children with no drawable polygon."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from ..ranking import mobile_canvas_review as core
from ..ranking.mobile_canvas_resolution import (
    exclude_missing_source_pose, link_manual_to_mt, missing_mt_geometry,
    resolution_detail, valid_quad,
)


class CanvasResolutionDialog(tk.Toplevel):
    def __init__(self, editor):
        super().__init__(editor.h.window)
        self.editor = editor
        self.gid = editor.h.current_invocation
        self.session = editor.h.session
        self.revision = editor.h.session.review.review_revision
        self.scene = core.scene(editor.h.session, self.gid)
        self.title("Rozstrzygnij wskazanie MT")
        self.geometry("740x580")
        self.minsize(620, 540)
        self.protocol("WM_DELETE_WINDOW", self.close)
        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        self.intro = ttk.Label(body, text=resolution_detail(self.scene) or
            "Wskaż detekcję MT i ręczną ramkę. Połączenie wymaga Twojej decyzji, "
            "że dotyczą tej samej fizycznej tablicy.", wraplength=690, justify="left")
        self.intro.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(body, text="Wskazanie MT (także bez widocznej ramki)").grid(row=1, column=0, sticky="w")
        self.targets = [o for o in self.scene["objects"] if o.get("parent_id")]
        parents = {p["parent_id"]: p for p in self.scene["parents"]}
        labels = []
        for obj in self.targets:
            parent = parents[obj["parent_id"]]
            state = ("RAW poza oceną" if obj.get("post_nms_suppressed") else "oznaczono FP" if obj["error"]
                     else "Pose poprawna" if valid_quad(self.scene, parent.get("quad")) else "brak poprawnej Pose")
            labels.append(f"MT {int(parent['detection_index'])+1} · {state} · {obj['attempt_id']}")
        self.mt = ttk.Combobox(body, state="readonly", values=labels)
        self.mt.grid(row=2, column=0, sticky="ew", pady=(3, 8))
        if labels:
            self.mt.current(0)
        ttk.Label(body, text="Ręczna ramka referencyjna").grid(row=3, column=0, sticky="w")
        self.manuals = [o for o in self.scene["objects"] if o.get("kind") == "manual" and not o.get("parent_id")]
        self.manual = ttk.Combobox(body, state="readonly", values=[o["object_id"] for o in self.manuals])
        self.manual.grid(row=4, column=0, sticky="ew", pady=(3, 8))
        if len(self.manuals) == 1:
            self.manual.current(0)
        self.details = ttk.Label(body, wraplength=690, justify="left")
        self.details.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.link_button = ttk.Button(body, text="Ta sama tablica — użyj ręcznej ramki jako korekty MT", command=self.link)
        self.link_button.grid(row=6, column=0, sticky="ew", pady=3)
        self.fp_button = ttk.Button(body, command=self.toggle_fp)
        self.fp_button.grid(row=7, column=0, sticky="ew", pady=3)
        self.remove_button = ttk.Button(body, text="Usuń błędnie dodaną ręczną ramkę", command=self.remove_manual)
        self.remove_button.grid(row=8, column=0, sticky="ew", pady=3)
        self.exclude_button = ttk.Button(body, text="Brak danych do oceny — wyłącz cały obraz", command=self.exclude)
        self.exclude_button.grid(row=9, column=0, sticky="ew", pady=3)
        ttk.Label(body, text="Każda decyzja wymaga potwierdzenia. Oryginalny wynik MT pozostaje bez zmian. "
            "Zmiany można cofnąć przez Ctrl+Z; ponowiona edycja wymaga sprawdzenia obrazu.",
            wraplength=690, justify="left").grid(row=10, column=0, sticky="ew", pady=10)
        ttk.Button(body, text="Zamknij bez zmian", command=self.close).grid(row=11, column=0, sticky="e")
        body.bind("<Configure>", lambda e: [w.configure(wraplength=max(300, e.width-28))
            for w in body.winfo_children() if isinstance(w, ttk.Label)])
        self.mt.bind("<<ComboboxSelected>>", self.refresh)
        self.manual.bind("<<ComboboxSelected>>", self.refresh)
        self.bind("<Escape>", lambda e: self.close())
        self.refresh()

    def chosen(self):
        return (self.targets[self.mt.current()] if self.mt.current() >= 0 else None,
                self.manuals[self.manual.current()] if self.manual.current() >= 0 else None)

    def refresh(self, event=None):
        target, manual = self.chosen()
        parent = next((p for p in self.scene["parents"] if target and p["parent_id"] == target["parent_id"]), None)
        raw = bool(target and (target.get("post_nms_suppressed") or self.scene.get("_post_nms_pending")))
        can_link = bool(parent and manual and not raw and not target["error"] and not manual["error"]
                        and valid_quad(self.scene, parent.get("quad")))
        self.link_button.configure(state="normal" if can_link else "disabled")
        self.fp_button.configure(text="Cofnij oznaczenie FP wskazania MT" if target and target["error"] else
            "To błędne wskazanie MT — oznacz FP (ręczna ramka pozostaje osobno)",
            state="normal" if target and not raw else "disabled")
        self.remove_button.configure(state="normal" if manual and not self.scene.get("_post_nms_pending") else "disabled")
        self.exclude_button.configure(state="normal" if missing_mt_geometry(self.scene)
            and not self.scene.get("_post_nms_pending") else "disabled")
        self.details.configure(text=(f"MZ: {parent['mz_status']}. " if parent else "") +
            ("Brak źródłowych narożników MT: połączenie nie utworzy poprawnej pary IoU/NMCE. "
             "Jeśli to ta sama tablica, dostępny jest jawny wyjątek dla całego obrazu. "
             "FP wybierz wyłącznie po stwierdzeniu rzeczywiście błędnego wskazania."
             if parent and not valid_quad(self.scene, parent.get("quad")) else
             "Połączenie zachowuje GT detekcji MT. Ręczna ramka i jej historyczne GT pozostają w historii decyzji; "
             "nie są kopiowane jako nowy odczyt modelu."))

    def apply(self, question, action, description):
        ed = self.editor
        if ed.h.session is not self.session or ed.h.current_invocation != self.gid or ed.h.session.review.review_revision != self.revision:
            messagebox.showinfo("Dane uległy zmianie", "Otwórz rozstrzygnięcie ponownie dla aktualnego obrazu.", parent=self)
            self.close()
            return
        if messagebox.askyesno("Potwierdź decyzję operatora", question, parent=self, default="no"):
            if ed._commit(action, description):
                self.close()

    def link(self):
        target, manual = self.chosen()
        if not target or not manual:
            return
        self.apply("Czy te dwie ramki dotyczą tej samej fizycznej tablicy?\n"
            "Ręczna geometria zastąpi korektę wskazanego MT. GT MT pozostanie bez zmian; "
            "cały ręczny obiekt trafi do historii. Obraz będzie wymagał ponownego sprawdzenia.",
            lambda: link_manual_to_mt(self.editor.h.session, self.gid, manual["object_id"], target["object_id"]),
            "Powiązano ręczną korektę z MT")

    def toggle_fp(self):
        target, _ = self.chosen()
        if not target:
            return
        self.apply("Cofnąć oznaczenie FP?" if target["error"] else
            "Czy wskazanie MT jest rzeczywiście błędne (FP)?\n"
            "Nie wybieraj FP tylko dlatego, że brakuje narożników. Ręczna ramka pozostanie "
            "odrębnym pominięciem (FN) po ponownym sprawdzeniu obrazu.",
            lambda: core.toggle_error(self.editor.h.session, self.gid, target["object_id"]),
            "Zmieniono decyzję FP; sprawdź cały obraz")

    def remove_manual(self):
        _, manual = self.chosen()
        if not manual:
            return
        self.apply("Usunąć tę ręczną ramkę z bieżącej oceny? Pełny zapis pozostanie w historii; "
            "Ctrl+Z przywróci ramkę. Brakujące dane MT nadal wymagają rozstrzygnięcia.",
            lambda: core.remove_operator_geometry(self.editor.h.session, self.gid, manual["object_id"]),
            "Usunięto ręczną ramkę; historia zachowana")

    def exclude(self):
        self.apply("Wyłączyć CAŁY obraz z oceny jakości z powodu brakujących źródłowych narożników MT?\n"
            "Dotyczy to wszystkich tablic na obrazie. Ramki i GT pozostaną w historii; "
            "obraz nie dostarczy TP/FP/FN, CER ani IoU/NMCE. Nie jest to potwierdzenie poprawności modelu.",
            lambda: exclude_missing_source_pose(self.editor.h.session, self.gid),
            "Wyłączono cały obraz z powodu braku danych źródłowych MT")

    def close(self):
        self.destroy()
