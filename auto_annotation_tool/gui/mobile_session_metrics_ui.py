"""Compact, non-focusable live values in the existing session context."""
import tkinter as tk
from .mobile_canvas_ui import MetricCard
from ..ranking.mobile_session_metrics import presentation


class SessionMetrics(tk.Frame):
    def __init__(self,parent,*,p):
        super().__init__(parent,bg=p['bg'],takefocus=0)
        self.cards={}
        self.status=tk.Label(self,bg=p['bg'],fg=p['muted'],font=('Segoe UI',8),
                             anchor='w',justify='left',width=1,takefocus=0,pady=0,bd=0)
        self.status.pack(fill='x',pady=(0,1))
        self.grid_host=tk.Frame(self,bg=p['bg'],takefocus=0)
        self.grid_host.pack(fill='x')
        for i,(key,title,tone) in enumerate((('precision','Precision','blue'),('recall','Recall','blue'),
                ('f1','F1','blue'),('iou','Polygon IoU','blue'),('nmce','NMCE','purple'),('cer','micro-CER','amber'))):
            self.grid_host.columnconfigure(i%3,weight=1,uniform='live_session_metrics')
            card=MetricCard(self.grid_host,title,p=p,tone=tone,compact=True)
            card.grid(row=i//3,column=i%3,sticky='nsew',padx=(0,3 if i%3<2 else 0),pady=(0,1))
            card.winfo_children()[0].configure(height=2)
            card.title.master.configure(padx=3,pady=0)
            card.title.configure(font=('Segoe UI',8,'bold'),pady=0,bd=0)
            card.value.configure(font=('Segoe UI',9,'bold'),pady=0,bd=0)
            card.value.pack_configure(pady=(0,0))
            # MetricCard's generic dashboard margin is too wide for three small
            # cards. Only these added cards use the actual inner label width.
            card.title.master.bind('<Configure>',lambda e,c=card:[
                w.configure(wraplength=max(1,e.width-6)) for w in (c.title,c.value)],add=False)
            self.cards[key]=card
        self.exact=tk.Label(self,bg=p['bg'],fg=p['fg'],font=('Segoe UI',9),
                            anchor='w',justify='left',width=1,takefocus=0)
        self.exact.pack(fill='x')
        self.denominators=tk.Label(self,bg=p['bg'],fg=p['muted'],font=('Segoe UI',8),
                                   anchor='w',justify='left',width=1,takefocus=0)
        self.denominators.pack(fill='x',pady=(1,0))
        def no_focus(widget):
            widget.configure(takefocus=0)
            for child in widget.winfo_children():no_focus(child)
        no_focus(self)
        self.bind('<Configure>',lambda e:[w.configure(wraplength=max(1,e.width))
            for w in (self.status,self.exact,self.denominators)],add='+')
        self.render(None,None)

    def render(self,snapshot,key):
        value=presentation(snapshot,key)
        if getattr(self,'data',None)==value:return
        self.data=value
        for k,card in self.cards.items():card.set(self.data['values'][k])
        for k in ('status','exact','denominators'):
            text=self.data[k]
            if k=='status':
                prefix = ('MT/ROI · '+self.data.get('assessment_scope','ASSISTED')+' · '
                          if (snapshot or {}).get('experiment_family') == 'F5A' else 'MT v2 · ASSISTED · ')
                text=prefix+text
            getattr(self,k).configure(text=text)
