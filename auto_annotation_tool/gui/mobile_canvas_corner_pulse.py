"""Transient vertex-1 cue, independent of collision warnings and review data."""
import tkinter as tk

TAG='gtr_corner_pulse'
GREEN='#35D07F'


class CornerPulse:
    def __init__(self,editor):
        self.ed=editor;self.c=editor.c;self.seen=None;self.target=None;self.pending=None;self.green=False
        self.after_ids=set();self.generation=0
        self.c.bind('<Destroy>',self.destroy,add='+')
        self.c.bind('<FocusOut>',lambda event:self.clear(),add='+')

    def selection(self):
        ed=self.ed
        if ed.h._closed or ed.mode!='edit' or not ed.selected:return None
        from . import mobile_canvas_clean_view as clean,mobile_canvas_geometry_layers as layers
        row=clean.selected(ed);value=layers.model(ed)
        if (not row or not row['gt'] or row['conflict']
                or (not ed.gt_focus_id and value and value['active']!='GTR')):return None
        scene=ed._read()
        if (ed.h._image_entry!=scene['image_entry']
                or tuple(ed.h._image_source_size or ())!=tuple(scene['image_size'])):return None
        return (id(ed.h.session),ed.h.current_invocation,scene['source_binding'],ed.selected,row['pid'])

    def trigger(self):
        target=self.selection();self.seen=target;self.clear()
        if target is None:return
        warning=getattr(self.ed,'collision_warning',None)
        if warning and warning.flashing:
            self.pending=target;return
        if warning:warning.paint()
        self.target=target;self.green=True;generation=self.generation
        def schedule(ms,on):
            def run():
                self.after_ids.discard(token)
                if generation!=self.generation:return
                if self.selection()!=self.target:self.clear();return
                self.green=on
                if ms==700:self.clear()
                else:self.draw()
            token=self.c.after(ms,run);self.after_ids.add(token)
        for ms,on in ((200,False),(350,True),(550,False),(700,False)):schedule(ms,on)
        self.draw()

    def resume(self):
        if self.pending is None:return
        target=self.pending
        if self.selection()==target:self.trigger()
        else:self.clear()

    def paint(self):
        target=self.selection()
        if self.pending is not None and self.pending!=target:self.clear()
        if target!=self.seen:
            self.seen=target;self.trigger()
        self.draw()

    def draw(self):
        self.c.delete(TAG)
        if not self.target:return
        if self.selection()!=self.target:self.clear();return
        if not self.green:return
        from .mobile_canvas_editor import TAG as OVERLAY
        from .mobile_canvas_clean_view import selected
        row=selected(self.ed);q=self.ed.drag['quad'] if self.ed.drag and self.ed.drag['oid']==self.ed.selected else row['gt']
        x,y=self.ed.to_canvas(q[0]);tags=(OVERLAY,TAG,self.ed.selected)
        self.c.create_oval(x-11,y-11,x+11,y+11,fill='#EAF5FC',outline='#EAF5FC',tags=tags)
        self.c.create_oval(x-8,y-8,x+8,y+8,fill=GREEN,outline='#0B633B',width=2,tags=tags+('gtr_vertex_1_green',))
        self.c.create_text(x,y,text='1',fill='#09251B',font=('Segoe UI',7,'bold'),tags=tags+('gtr_vertex_1_label',))
        self.c.tag_raise(TAG)

    def clear(self):
        self.generation+=1
        for token in tuple(self.after_ids):
            try:self.c.after_cancel(token)
            except tk.TclError:pass
        self.after_ids.clear();self.target=None;self.pending=None;self.green=False
        try:self.c.delete(TAG)
        except tk.TclError:pass

    def destroy(self,event):
        if event.widget is self.c:self.clear()
