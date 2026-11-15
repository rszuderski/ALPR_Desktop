"""Small local Tk presentation components used only by mobile review.

No global theme changes, no session writes, no inference. Colors have text
labels as well; no state is communicated by color alone.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def palette(widget, background):
    try:
        rgb = widget.winfo_rgb(background)
        dark = sum(rgb) / (3 * 65535) < .45
    except tk.TclError:
        dark = False
    if dark:
        return dict(bg="#17232D", surface="#21323E", fg="#EBF2F6", muted="#B5C4CE",
                    border="#3A5364", blue="#77CBF0", blue_bg="#193E54",
                    green="#83DEB5", green_bg="#203F36", amber="#FFD087", amber_bg="#493A25",
                    red="#FF9CA7", red_bg="#4A2932", purple="#C8B4FF", purple_bg="#35334C")
    return dict(bg="#F2F5F8", surface="#FFFFFF", fg="#213547", muted="#5E7283",
                border="#D7E1E8", blue="#14628B", blue_bg="#E5F2FA",
                green="#176B4D", green_bg="#E4F4EC", amber="#86540E", amber_bg="#FFF2D9",
                red="#AB3545", red_bg="#FDECEF", purple="#6952A5", purple_bg="#F0EBFA")


def label(parent, text="", *, p, size=10, bold=False, fg=None, bg=None, wrap=0):
    return tk.Label(parent, text=text, bg=bg or p["surface"], fg=fg or p["fg"],
                    font=("Segoe UI", size, "bold" if bold else "normal"),
                    anchor="w", justify="left", wraplength=wrap)


class MetricCard(tk.Frame):
    def __init__(self, parent, title, *, p, tone="blue", compact=False):
        super().__init__(parent, bg=p["surface"], highlightbackground=p["border"], highlightthickness=1)
        self.p, self.tone = p, tone
        tk.Frame(self, bg=p[tone], height=3).pack(fill="x")
        inside = tk.Frame(self, bg=p["surface"], padx=10 if compact else 14, pady=7 if compact else 11)
        inside.pack(fill="both", expand=True)
        self.title = label(inside, title, p=p, size=9, bold=True, fg=p["muted"])
        self.title.configure(width=1)
        self.title.pack(fill="x")
        self.value = label(inside, "—", p=p, size=18 if compact else 23, bold=True, fg=p[tone])
        self.value.configure(width=1)
        self.value.pack(fill="x", pady=(3, 0))
        self.note = label(inside, "", p=p, size=9, fg=p["muted"])
        self.note.configure(width=1)
        self.note.pack(fill="x", pady=(2, 0))
        self._wrap_width = None
        inside.bind("<Configure>", self._wrap, add="+")

    def _wrap(self, event):
        width = max(55, event.width - 28)
        if width == self._wrap_width:
            return
        self._wrap_width = width
        for w in (self.title, self.value, self.note):
            w.configure(wraplength=width)

    def set(self, value, note=""):
        self.value.configure(text=value)
        self.note.configure(text=note)
        if note:
            self.note.pack(fill="x", pady=(2, 0))
        else:
            self.note.pack_forget()


class ReviewContexts(tk.Frame):
    """Pinned context; the existing sidebar alone scrolls actions and details."""
    def __init__(self, parent, *, p):
        super().__init__(parent, bg=p['bg'], padx=6, pady=4)
        self.fields = {}
        self.data = {}
        for key, title in (('object', 'WYBRANY OBIEKT'), ('image', 'BIEŻĄCY OBRAZ'), ('session', 'CAŁA SESJA')):
            label(self, title, p=p, bg=p['bg'], fg=p['blue'], size=9, bold=True).pack(fill='x', pady=(3, 1))
            body = label(self, p=p, bg=p['bg'], size=9, wrap=260)
            body.configure(width=1)
            body.pack(fill='x', pady=(0, 3))
            self.fields[key] = body
        self.bind('<Configure>', self._wrap)

    def _wrap(self, event):
        for item in self.fields.values():item.configure(wraplength=max(100, event.width-16))

    def render(self, data):
        self.data = data
        for key, item in self.fields.items():item.configure(text=data[key])


class PlateAssessment(tk.Frame):
    """Compact read-only MT/MZ result; GT editing stays on the canvas."""
    def __init__(self, parent, *, p):
        super().__init__(parent, bg=p["surface"], padx=9, pady=7,
                         highlightbackground=p["border"], highlightthickness=1)
        self.p = p
        self.plain = label(self, p=p, size=9, wrap=260)
        self.plain.configure(width=1)
        self.plain.pack(fill='x')
        self.technical = Disclosure(self, 'Szczegóły techniczne i metryki', p=p)
        self.technical.pack(fill='x', pady=(5,0))
        self.fields = {}
        for key, text, bold, tone in (
            ("execution", "", True, "blue"), ("pipeline", "", True, "amber"),
            ("image_status", "", False, "muted"), ("focus", "", False, "muted"),
            ("image_frames", "", False, "muted"), ("frame", "", True, "fg"),
            ("geometry_title", "DOPASOWANIE RAMKI MT", True, "muted"),
            ("reference", "", False, "muted"),
            ("geometry", "", False, "fg"),
            ("text_title", "ODCZYT ZNAKÓW MZ ↔ GT", True, "muted"),
            ("mz", "", True, "fg"), ("gt", "", True, "fg"),
            ("gt_note", "", False, "amber"),
            ("text", "", False, "fg"), ("outcome", "", False, "muted"),
            ("draft", "", False, "amber"),
            ("correction_history", "", False, "muted"),
        ):
            item = label(self.technical.body, text, p=p, bold=bold, fg=p[tone], size=9, wrap=260)
            item.configure(width=1)
            item.pack(fill="x", pady=(5 if key.endswith("title") else 0, 0))
            self.fields[key] = item
        self.bind("<Configure>", self._wrap)
        self.history = Disclosure(self.technical.body,"Pochodzenie ramki · tylko odczyt",p=p)
        self.history_text = label(self.history.body,'',p=p,size=8,wrap=260)
        self.history_text.configure(width=1)
        self.history_text.pack(fill='x')
        self.history_text.bind('<Configure>',lambda e:self.history_text.configure(wraplength=max(80,e.width-6)))
        self._history_selection = ''

    def _wrap(self, event):
        self.plain.configure(wraplength=max(80, event.width - 24))
        for item in self.fields.values():
            item.configure(wraplength=max(80, event.width - 24))

    def render(self, model):
        self.model = model
        operator = model.get('operator', {})
        plain = operator.get('numbers') if operator.get('question') else '\n'.join(filter(None,
            (operator.get('marking'),operator.get('model'),operator.get('numbers'))))
        self.plain.configure(text=plain or model.get('outcome', ''))
        self.fields["execution"].configure(text=model["execution"]["title"])
        for key in ('pipeline','image_status'):
            self.fields[key].configure(text=model.get(key,''))
            if model.get(key):self.fields[key].pack(fill='x',before=self.fields['focus'])
            else:self.fields[key].pack_forget()
        self.fields["focus"].configure(text=model.get("focus", ""))
        self.fields["geometry_title"].configure(text=model.get("geometry_title", "DOPASOWANIE RAMKI MT"))
        self.fields["reference"].configure(text=model.get("reference", ""))
        if model.get("reference"):
            self.fields["reference"].pack(fill="x", before=self.fields["geometry"])
        else:
            self.fields["reference"].pack_forget()
        for key in ("frame", "image_frames", "gt_note", "correction_history"):
            self.fields[key].configure(text=model.get(key, ""))
        for key, before in (("gt_note", "text"), ("correction_history", None)):
            item = self.fields[key]
            if not model.get(key):
                item.pack_forget()
            elif before:
                item.pack(fill="x", before=self.fields[before])
            else:
                item.pack(fill="x")
        for key in ("outcome", "geometry", "mz", "gt", "text"):
            self.fields[key].configure(text=model[key])
        self.fields["draft"].configure(
            text=model.get("summary_note") or ("Poza metrykami jakości" if model["execution"]["state"] != "MT_EXECUTED" else
                "Wynik wstępny — jeszcze nie liczony do podsumowania" if model["draft"] else "Ocena asystowana"),
            fg=self.p.get(model.get("summary_tone", "muted"), self.p["muted"]))
        self.fields["draft"].pack(fill="x", before=self.fields["focus"], pady=(3, 3))
        selected=model.get('history_selected','')
        details=model.get('history_details','')
        if details:
            self.history_text.configure(text=details)
            self.history.pack(fill='x',pady=(8,0))
            if selected!=self._history_selection:
                self.history.opened=bool(selected);self.history._sync()
                if selected:
                    self.technical.opened=True;self.technical._sync()
        else:
            self.history.pack_forget();self.history.opened=False;self.history._sync()
        self._history_selection=selected
        callback = getattr(self, 'on_render', None)
        if callback:callback(model)


class CanvasDecisionPanel(tk.Frame):
    """One contextual decision, no modal and no write on selection/navigation."""
    def __init__(self, parent, editor, *, p):
        super().__init__(parent, bg=p["surface"], padx=8, pady=7,
                         highlightbackground=p["border"], highlightthickness=1)
        self.ed, self.p = editor, p
        self.title = label(self, "Do rozstrzygnięcia", p=p, bold=True, size=9)
        self.title.pack(fill="x")
        self.detail = label(self, p=p, size=9, fg=p["muted"], wrap=260)
        self.detail.configure(width=1); self.detail.pack(fill="x", pady=(3, 3))
        self.detail.bind("<Configure>", lambda e: self.detail.configure(wraplength=max(80,e.width-6)))
        self.pick = ttk.Combobox(self, state="readonly", height=6)
        self.pick.bind("<<ComboboxSelected>>", lambda e: self._choose_pair())
        self.same = editor.make_action_button(self, "Ta sama tablica", self.link, compact=True, wraplength=260)
        self.different = editor.make_action_button(self, "Inna tablica — pozostaw osobno", self.separate, compact=True, wraplength=260)
        self.fp = editor.make_action_button(self, "Oznacz FP", self.toggle_fp, compact=True)
        self.negative = editor.make_action_button(self, "Brak tablic w analizowanym obszarze", self.no_plates, compact=True, wraplength=260)
        self.choose_mt = editor.make_action_button(self, "Wybierz wskazanie MT", self.select_mt, compact=True)
        self.unknown = editor.make_action_button(self, 'Nie można ustalić', self.unresolved, compact=True)
        self.physical_pick = ttk.Combobox(self,state='readonly',height=6)
        self.physical_pick.bind('<<ComboboxSelected>>',lambda e:self.physical_same.configure(state='normal'))
        self.physical_same = editor.make_action_button(self,'Ta sama fizyczna tablica',self.assign_physical,compact=True,wraplength=260)
        self.physical_distinct = editor.make_action_button(self,'To odrębna tablica na tej scenie',self.distinct_physical,compact=True,wraplength=260)
        for button in (self.same,self.different,self.fp,self.negative,self.choose_mt,self.unknown):
            button.configure(pady=2)
        self.consequence = label(self, p=p, size=8, fg=p['muted'], wrap=260)
        self.consequence.configure(width=1)
        self.consequence.bind('<Configure>', lambda e:self.consequence.configure(wraplength=max(80,e.width-6)))
        self.pair = None; self.obj = None; self.options = []
        self._selection_key = None

    def refresh(self, scene, selected, enabled, execution):
        from ..ranking.mobile_canvas_resolution import missing_mt_geometry
        from ..ranking.mobile_canvas_presentation import operator_context
        self.scene, self.gid = scene, self.ed.h.current_invocation
        self.session = self.ed.h.session
        self.revision = self.session.review.review_revision if self.session else None
        self.enabled = enabled and execution.get("state") == "MT_EXECUTED"
        self.execution = execution
        for w in (self.pick,self.same,self.different,self.fp,self.negative,self.choose_mt,self.unknown,self.consequence,self.physical_pick,self.physical_same,self.physical_distinct):w.pack_forget()
        self.pair = None; self.options = []
        self.obj = next((o for o in (scene or {}).get("objects", ()) if o["object_id"] == selected), None)
        if not scene or not self.enabled or scene.get("_post_nms_pending"):
            self.pack_forget(); return
        raw = lambda o: o.get("post_nms_suppressed")
        manuals = [o for o in scene["objects"] if not o.get("parent_id") and not o["error"] and not raw(o)]
        targets = [o for o in scene["objects"] if o.get("parent_id") and not raw(o) and not o["error"]]
        unresolved = missing_mt_geometry(scene)
        self.missing = unresolved
        self.info = operator_context(self.session, self.gid, scene, selected, execution)
        selected = self.info['focus_id']
        self.obj = next((o for o in scene['objects'] if o['object_id'] == selected), None)
        key = (self.gid, selected, self.revision)
        if key != self._selection_key:self.pick.set("")
        self._selection_key = key
        if self.obj and not raw(self.obj) and not self.obj.get('error'):
            if self.obj.get("parent_id"):
                self.options = [(m,self.obj) for m in manuals if self.obj["object_id"] not in m.get("separate_from_mt", ())]
            else:
                self.options = [(self.obj,t) for t in targets if t["object_id"] not in self.obj.get("separate_from_mt", ())]
        if self.options:
            self.detail.configure(text='\n'.join(filter(None,(self.info['marking'],self.info['model'],
                self.info['question'] or 'Czy wskazanie dotyczy tej samej tablicy?'))))
            if len(self.options) == 1:self.pair = self.options[0]
            else:
                self.pick.configure(values=[self._pair_label(pair) for pair in self.options])
                self.pick.pack(fill="x", pady=3)
            self.choose_mt.pack(fill='x', pady=2)
            self.same.pack(fill="x", pady=2);self.different.pack(fill="x", pady=2)
            self._choose_pair()
        elif unresolved:
            self.detail.configure(text='\n'.join(filter(None,(self.info['marking'],self.info['model'],self.info['question']))))
            if self.info['location']:
                self.choose_mt.configure(text='Pokaż wskazanie modelu',state='normal')
                self.choose_mt.pack(fill='x',pady=2)
        else:
            self.detail.configure(text="Potwierdź obecność tablic w analizowanym ROI.")
        target = self.pair[1] if self.pair else self.obj
        source_selected=getattr(self.ed,'_bad_mt_selection',None)==(id(self.session),self.gid,selected)
        if source_selected and self.obj and self.obj.get('parent_id') and target and target.get('parent_id') and not target.get('duplicate_of') and not raw(target) and (unresolved or not scene['checked']):
            self.fp.configure(text='Cofnij błędne wskazanie' if target['error'] else 'Fałszywe wskazanie (nie tablica)',
                state='normal' if target['error'] or self.info['location'] else 'disabled')
            if target['error'] or self.info['location']:self.fp.pack(fill="x", pady=2)
        if self.info['target'] in unresolved:
            self.unknown.configure(state='disabled' if self.info['target'].get('identity_unresolved') else 'normal')
            self.unknown.pack(fill='x',pady=2)
        if not manuals and scene.get("roi_presence") != "NO_PLATES" and (not targets or self.obj and not scene["checked"]):
            self.negative.pack(fill="x", pady=2)
        self.physical_options=[]
        from ..ranking.mobile_canvas_physical_gt import issues as physical_issues
        unresolved_physical=bool(physical_issues(scene))
        if self.obj and self.obj.get('parent_id') and not self.obj['error'] and not self.obj.get('duplicate_of') and (not scene['checked'] or unresolved_physical) and not self.options:
            from ..ranking.mobile_canvas_physical_gt import scope, require_final
            for other in targets:
                if other['object_id']==self.obj['object_id'] or other.get('duplicate_of'):continue
                try:require_final(self.session,self.gid,[self.obj,other])
                except ValueError:continue
                parent=next(p for p in scene['parents'] if p['parent_id']==other['parent_id'])
                self.physical_options.append((self.gid,other['object_id'],f"To samo zdjęcie · wskazanie {int(parent['detection_index'])+1}"))
            from ..ranking import mobile_canvas_review as core
            for gid in self.session.mt_invocations:
                if gid==self.gid:continue
                try:
                    if scope(self.session,gid)!=scope(self.session,self.gid):continue
                    other_scene=core.scene(self.session,gid)
                except (ValueError,KeyError):continue
                registry=other_scene.get('physical_gt_v1',{})
                if not registry.get('confirmed'):continue
                for plate in registry.get('plates',()):
                    self.physical_options.append((gid,plate['reference_object_id'],f"Inne wywołanie · {gid[-9:]} · {plate['gt_id'][-6:]}"))
            if self.physical_options:
                self.detail.configure(text='Jeśli to dodatkowy wynik dla tej samej fizycznej tablicy, wskaż jej właściwą ramkę. Różne tablice pozostaw osobno.')
                self.physical_pick.configure(values=[p[2] for p in self.physical_options]);self.physical_pick.set('')
                self.physical_pick.pack(fill='x',pady=3)
                self.physical_same.configure(state='disabled');self.physical_same.pack(fill='x',pady=2)
                reg=scene.get('physical_gt_v1',{})
                gt=reg.get('associations',{}).get(self.obj['object_id'])
                if any(p['gt_id']==gt and p.get('scene_identity')=='UNRESOLVED_ACROSS_INVOCATIONS' for p in reg.get('plates',())):
                    self.physical_distinct.pack(fill='x',pady=2)
        self.consequence.configure(text=self.info['consequence'] +
            ('\nNie można ustalić: bez oceny trafności.' if unresolved else ''))
        if self.consequence.cget('text'):self.consequence.pack(fill='x',pady=(3,0))
        if any(w.winfo_manager() for w in (self.same,self.fp,self.negative,self.choose_mt,self.unknown,self.physical_same)):
            self.pack(fill="x", after=self.ed.status, pady=(5, 0))
        else:self.pack_forget()

    def _pair_label(self, pair):
        manual,target = pair
        parent = next(p for p in self.scene["parents"] if p["parent_id"] == target["parent_id"])
        return (f"MT {int(parent['detection_index'])+1}" if not self.obj.get("parent_id") else
                f"Ramka ręczna {self.scene['objects'].index(manual)+1}")

    def _choose_pair(self):
        from ..ranking.mobile_canvas_resolution import model_location
        from ..ranking.mobile_canvas_presentation import operator_context
        if len(self.options)>1:
            i=self.pick.current();self.pair=self.options[i] if 0<=i<len(self.options) else None
        location = None
        if self.pair:
            self.info = operator_context(self.session,self.gid,self.scene,self.obj['object_id'],
                self.execution,target_id=self.pair[1]['object_id'])
            self.detail.configure(text='\n'.join(filter(None,(self.info['marking'],self.info['model'],
                self.info['question'] or 'Czy wskazanie dotyczy tej samej tablicy?'))))
            try:location = model_location(self.session,self.gid,self.scene,self.pair[1])
            except (ValueError,TypeError,KeyError):pass
        self.same.configure(state="normal" if location else "disabled")
        self.different.configure(state="normal" if location else "disabled",text='Inna prawdziwa tablica')
        self.choose_mt.configure(text='Pokaż wskazanie modelu',state='normal' if location else 'disabled')
        if not location:
            for w in (self.choose_mt,self.same,self.different):w.pack_forget()
        else:
            self.same.pack(fill='x',pady=2);self.different.pack(fill='x',pady=2)
            self.choose_mt.pack(fill='x',before=self.same,pady=2)
        if self.pair:
            parent=next(p for p in self.scene["parents"] if p["parent_id"]==self.pair[1]["parent_id"])
            self.same.configure(text=f"Tak, to ta tablica — wskazanie {int(parent['detection_index'])+1}")
            if len(self.options) > 1:
                target = self.pair[1]
                if (target in self.missing and self.obj and self.obj.get('parent_id')
                        and getattr(self.ed,'_bad_mt_selection',None)==(id(self.session),self.gid,self.obj['object_id'])):
                    self.fp.configure(text='Fałszywe wskazanie (nie tablica)',state='normal' if location else 'disabled')
                    self.fp.pack(fill='x',pady=2)
                    self.unknown.configure(state='disabled' if target.get('identity_unresolved') else 'normal')
                    self.unknown.pack(fill='x',pady=2)
                self.ed.assessment.model['operator'] = self.info
                self.ed.assessment.render(self.ed.assessment.model)

    def apply(self, action, text):
        if (self.session is not self.ed.h.session or self.gid != self.ed.h.current_invocation
                or self.revision != self.session.review.review_revision):
            self.ed.sync();self.ed.message("Widok zmienił się — wybierz aktualną ramkę.");return False
        return self.ed._commit(action, text)

    def link(self):
        from ..ranking.mobile_canvas_resolution import link_manual_to_mt
        if self.pair:
            manual,target=self.pair
            if self.apply(lambda:link_manual_to_mt(self.session,self.gid,manual["object_id"],target["object_id"]),
                          "Powiązano jedną tablicę z MT; oryginalne Pose zachowano"):
                self.ed._choose_plate(target["object_id"])

    def separate(self):
        from ..ranking.mobile_canvas_resolution import keep_separate
        if self.pair:
            manual,target=self.pair
            if self.apply(lambda:keep_separate(self.session,self.gid,manual["object_id"],target["object_id"]),
                          "Zachowano osobne tablice; oceń wskazanie MT"):
                self.ed._choose_plate(target["object_id"])

    def toggle_fp(self):
        from ..ranking import mobile_canvas_review as core
        target = self.obj
        if target and (not target.get('parent_id') or
                       getattr(self.ed,'_bad_mt_selection',None)!=(id(self.session),self.gid,target['object_id'])):
            self.ed.message(core.GTR_BAD_MESSAGE);return False
        if target:self.apply(lambda:core.toggle_error(self.session,self.gid,target['object_id']),"Zapisano ocenę wskazania modelu")

    def no_plates(self):
        from ..ranking.mobile_canvas_resolution import confirm_no_plates
        self.apply(lambda:confirm_no_plates(self.session,self.gid),"Potwierdzono brak tablic w analizowanym ROI")

    def select_mt(self):
        target = self.pair[1] if self.pair else self.info['target']
        if target:self.ed.show_model_location(target['object_id'])

    def unresolved(self):
        from ..ranking.mobile_canvas_resolution import mark_identity_unresolved
        target = self.pair[1] if self.pair else self.info['target']
        if target:self.apply(lambda:mark_identity_unresolved(self.session,self.gid,target['object_id']),
            'Zapisano: nie można ustalić. Nie przyznano trafienia ani błędu modelu.')

    def assign_physical(self):
        from ..ranking.mobile_canvas_physical_gt import mark_duplicate,share_across_invocations
        index=self.physical_pick.current()
        if not 0<=index<len(self.physical_options):return
        gid,oid,_=self.physical_options[index]
        action=(lambda:mark_duplicate(self.session,self.gid,self.obj['object_id'],oid)) if gid==self.gid else (
            lambda:share_across_invocations(self.session,self.gid,self.obj['object_id'],gid,oid))
        self.apply(action,'Zapisano jedną fizyczną tablicę; wyniki wywołań zachowano osobno.')

    def distinct_physical(self):
        from ..ranking.mobile_canvas_physical_gt import declare_distinct
        if self.obj:self.apply(lambda:declare_distinct(self.session,self.gid,self.obj['object_id']),
            'Potwierdzono odrębną fizyczną tablicę na tej scenie.')

    def close(self):
        self.pack_forget()


class Disclosure(tk.Frame):
    """A full-width section header, not an action-looking CTA button."""
    def __init__(self, parent, title, *, p, opened=False, command=None):
        super().__init__(parent, bg=p["surface"], highlightbackground=p["border"], highlightthickness=1)
        self.opened, self.title, self.command = opened, title, command
        self.header = tk.Label(self, bg=p["blue_bg"], fg=p["blue"], cursor="hand2", takefocus=True,
                               font=("Segoe UI", 10, "bold"), anchor="w", padx=10, pady=9,
                               highlightthickness=1, highlightbackground=p["blue_bg"], highlightcolor=p["blue"])
        self.header.pack(fill="x")
        self.body = tk.Frame(self, bg=p["surface"], padx=10, pady=8)
        for seq in ("<Button-1>", "<Return>", "<space>"):
            self.header.bind(seq, self.toggle)
        self._sync()

    def _sync(self):
        self.header.configure(text=("▾  " if self.opened else "▸  ") + self.title)
        if self.opened:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()

    def toggle(self, event=None):
        self.opened = not self.opened
        self._sync()
        if self.command:
            self.command(self.opened)
        return "break"


class ScrollPane(tk.Frame):
    def __init__(self, parent, *, p):
        super().__init__(parent, bg=p["bg"])
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg=p["bg"], bd=0, highlightthickness=0, yscrollincrement=22)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=bar.set)
        self.body = tk.Frame(self.canvas, bg=p["bg"])
        self.item = self.canvas.create_window(0, 0, window=self.body, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.item, width=e.width))
        self.body.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.bind_wheel(self)

    def wheel(self, event):
        num, delta = getattr(event, "num", None), getattr(event, "delta", 0)
        if num not in (4, 5) and not delta:
            return None
        step = -1 if num == 4 or (num != 5 and delta > 0) else 1
        self.canvas.yview_scroll(step * 3, "units")
        return "break"

    def bind_wheel(self, widget):
        # Called again after dynamic content is created; replace, never double bind.
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(seq, self.wheel)
        for child in widget.winfo_children():
            self.bind_wheel(child)
