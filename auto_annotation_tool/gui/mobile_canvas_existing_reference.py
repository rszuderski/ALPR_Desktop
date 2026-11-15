"""Offer an explicit relation to an existing GTR; overlap alone never writes."""
import tkinter as tk


def choices(ed):
    from . import mobile_canvas_clean_view as clean
    from ..ranking.mobile_manual_frame_guard import reservation_records
    row=clean.selected(ed)
    if not row or row['pid'] or not row['candidate'] or row['role']!='FINAL' or row['fp']:return []
    refs=reservation_records(ed._read(),session=ed.h.session,gid=ed.h.current_invocation)
    return [dict(r,oid=row['oid'],mt=row['candidate']['mt_detection_index']+1) for r in refs
            if any(c.get('object_id')==row['oid'] for c in r['contours'])]


def install(ed,options):
    ed.existing_reference_bar=tk.Frame(ed.contexts.sections['image'],bg=ed.p['bg'])
    ed.existing_reference_options=options;ed.existing_reference_key=None;ed.existing_reference_buttons=[]


def sync(ed):
    items=choices(ed) if ed._read() and not ed._read().get('exclusion_reason') else []
    if not items or ed.mode=='draw' or ed.drag:
        ed.existing_reference_bar.pack_forget();return
    key=(id(ed.h.session),ed.h.current_invocation,ed.h.session.review.review_revision,tuple((r['oid'],r['plate_id']) for r in items))
    if key!=ed.existing_reference_key:
        for w in ed.existing_reference_bar.winfo_children():w.destroy()
        ed.existing_reference_buttons=[]
        tk.Label(ed.existing_reference_bar,text='Istnieje obrys operatora bez powiązania z tym MT. Jeśli to ta sama tablica:',
            bg=ed.p['bg'],fg=ed.p['fg'],font=('Segoe UI',10),anchor='w',justify='left',wraplength=max(180,ed.sidebar_width-25)).pack(fill='x')
        for r in items:
            button=tk.Button(ed.existing_reference_bar,text=f"Powiąż MT {r['mt']} z GTR {r['number']}",
                command=lambda r=r,key=key:apply(ed,r,key),**ed.existing_reference_options)
            button.pack(fill='x',pady=(3,0));ed.existing_reference_buttons.append(button)
        ed.existing_reference_key=key
    for button in ed.existing_reference_buttons:
        button.configure(state='normal' if not ed.h._saving and not ed.busy and ed.entry is None else 'disabled')
    ed.existing_reference_bar.pack(fill='x',pady=(5,0),before=ed.approve)


def apply(ed,choice,key):
    if (id(ed.h.session),ed.h.current_invocation,ed.h.session.review.review_revision)!=key[:3]:
        ed.message('Ocena zmieniła się. Wybierz ponownie aktualne powiązanie.');return
    if not ed.prepare_navigation() or not ed.available():return
    if not any(r['oid']==choice['oid'] and r['plate_id']==choice['plate_id'] for r in choices(ed)):return
    from ..ranking.mobile_candidate_stacks import associate
    if ed._commit(lambda:associate(ed.h.session,ed.h.current_invocation,choice['oid'],choice['plate_id']),
        f"Powiązano MT {choice['mt']} z istniejącą GTR {choice['number']}; zachowano obrys i numer GT"):
        from .mobile_canvas_navigator import select_layer
        select_layer(ed,'GTR',choice['oid'])
