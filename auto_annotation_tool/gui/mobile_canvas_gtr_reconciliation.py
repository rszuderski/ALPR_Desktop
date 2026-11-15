"""Explicit same-plate action for overlapping historical manual references."""
import tkinter as tk
from ..ranking import mobile_gtr_reconciliation as data


def install(ed,options):
    ed.gtr_reconciliation_bar=tk.Frame(ed.contexts.sections['image'],bg=ed.p['bg'])
    ed.gtr_reconciliation_options=options
    ed.gtr_reconciliation_key=None
    ed.gtr_reconciliation_buttons=[]


def sync(ed):
    from . import mobile_canvas_clean_view as clean
    if not hasattr(ed,'gtr_reconciliation_bar'):return
    scene=ed._read();state=clean.source(ed)
    choices=data.suggestions(scene,state) if scene and state and not scene.get('exclusion_reason') else []
    if not choices or ed.mode=='draw' or ed.drag:
        ed.gtr_reconciliation_bar.pack_forget();return
    key=(id(ed.h.session),ed.h.current_invocation,ed.h.session.review.review_revision,tuple((r['duplicate_id'],r['canonical_id']) for r in choices))
    if key!=ed.gtr_reconciliation_key:
        for w in ed.gtr_reconciliation_bar.winfo_children():w.destroy()
        ed.gtr_reconciliation_buttons=[]
        tk.Label(ed.gtr_reconciliation_bar,text='Nakładające się GTR. Jeśli to jedna tablica:',bg=ed.p['bg'],fg=ed.p['fg'],
                 font=('Segoe UI',10),anchor='w',wraplength=max(180,ed.sidebar_width-25)).pack(fill='x')
        for choice in choices:
            button=tk.Button(ed.gtr_reconciliation_bar,
                text=f"Jedna tablica: GTR {choice['duplicate_number']} → GTR {choice['canonical_number']}",
                command=lambda choice=choice,key=key:apply(ed,choice,key),**ed.gtr_reconciliation_options)
            button.pack(fill='x',pady=(3,0));ed.gtr_reconciliation_buttons.append(button)
        ed.gtr_reconciliation_key=key
    for button in ed.gtr_reconciliation_buttons:
        button.configure(state='normal' if not ed.h._saving and not ed.busy and ed.entry is None else 'disabled')
    ed.gtr_reconciliation_bar.pack(fill='x',pady=(5,0),before=ed.contexts.image_actions)


def apply(ed,choice,key):
    from . import mobile_canvas_navigator as nav
    if (id(ed.h.session),ed.h.current_invocation,ed.h.session.review.review_revision)!=key[:3]:
        ed.message('Ocena zmieniła się. Wybierz ponownie aktualne GTR.');return
    if not ed.prepare_navigation() or not ed.available():return
    state=ed._read();from ..ranking import mobile_candidate_stacks as stacks
    oid=stacks._state(state)['plates'][choice['canonical_id']]['origin_object_id']
    if ed._commit(lambda:data.merge_manual_duplicate(ed.h.session,ed.h.current_invocation,choice['duplicate_id'],choice['canonical_id'],
                   expected_revision=key[2]),'Połączono duplikat GTR; zachowano numer, obrys i historię tablicy'):
        nav.select_layer(ed,'GTR',oid)
