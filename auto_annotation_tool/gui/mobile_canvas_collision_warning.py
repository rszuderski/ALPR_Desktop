"""Cancellable, view-only GTR warning. Timers never repaint the source image."""
from __future__ import annotations
import copy,time,tkinter as tk
from ..ranking.mobile_manual_frame_guard import conflict_message

TAG='gtr_collision_warning'
RED='#E53935'
NORMAL='#FFD400'

class CollisionWarning:
    def __init__(self,editor):
        self.ed=editor;self.c=editor.c;self.active=None;self.phase='normal'
        self.after_ids=set();self._expiry=None;self._generation=0;self._context=None;self.started_at=None
        self._editing=False;self._mode=None;self._selected=None
        self._flash_complete=True
        self.c.bind('<Destroy>',self._destroy,add='+')
        self.c.bind('<FocusOut>',lambda event:self.clear(),add='+')

    def _key(self):
        ed=self.ed;s=ed._read();h=ed.h;review=getattr(h.session,'review',None)
        return (id(h.session),getattr(review,'source_archive_sha256',None),h.current_invocation,
                getattr(review,'review_revision',None),h._image_entry,s.get('source_binding') if s else None)

    def _valid(self):
        return bool(self.active and self.ed.mode==self._mode and self.c.winfo_exists() and self._context==self._key()
                    and (not self._editing or self.ed.selected==self._selected))

    def _cancel(self,token):
        if token:
            try:self.c.after_cancel(token)
            except tk.TclError:pass
            self.after_ids.discard(token)

    def _after(self,ms,callback):
        generation=self._generation
        def run():
            self.after_ids.discard(token)
            if generation!=self._generation:return
            if not self._valid():self.clear();return
            callback()
        token=self.c.after(ms,run);self.after_ids.add(token);return token

    @property
    def flashing(self):return bool(self._valid() and not self._flash_complete)

    def show(self,reference,*,editing=False):
        same=self._valid() and self.active['plate_id']==reference['plate_id'] and self._editing==editing and not self._flash_complete
        if not same:
            self.clear();self.active=copy.deepcopy(reference);self._context=copy.deepcopy(self._key())
            self._editing=editing;self._mode=self.ed.mode if editing else 'draw';self._selected=getattr(self.ed,'selected',None)
            self.started_at=time.monotonic();self.phase='red';self._flash_complete=False
            pulse=getattr(self.ed,'corner_pulse',None)
            if pulse:pulse.clear()
            for ms,phase in ((150,'normal'),(300,'red')):
                self._after(ms,lambda phase=phase:self._set_phase(phase))
            self._after(450,self._finish_flash)
        self._cancel(self._expiry);self._expiry=self._after(2500,self.clear)
        self.ed.message(conflict_message(reference['number']));self.paint()

    def _set_phase(self,phase):
        self.phase=phase;self.paint()

    def _finish_flash(self):
        self._flash_complete=True;self._set_phase('normal')
        pulse=getattr(self.ed,'corner_pulse',None)
        if pulse:pulse.resume()

    def frames(self):
        if not self.active or not self._valid():return []
        frames=[]
        for ref in self.active.get('choices') or [self.active]:
            for frame in ref.get('contours') or [dict(kind='GTR',quad=ref['quad'])]:
                if not any(f['kind']==frame['kind'] and f['quad']==frame['quad'] for f in frames):frames.append(frame)
        return frames

    def paint(self):
        if not self.active:return
        if not self._valid():self.clear();return
        if self._flash_complete:
            self.c.delete(TAG);return
        width=max(2,min(3,round(2*self.c.winfo_fpixels('1i')/96)))
        frames=self.frames()
        if not frames:return
        coords=[v for p in frames[0]['quad'] for v in self.ed.to_canvas(p)]
        styles=[('gtr_collision_halo','#EAF5FC',width+2,coords,())]
        for index,frame in enumerate(frames):
            coords=[v for p in frame['quad'] for v in self.ed.to_canvas(p)]
            normal=NORMAL if frame['kind']=='GTR' else '#58B8D5'
            styles.append(('gtr_collision_core' if index==0 else f'gtr_collision_shape_{index}',
                RED if self.phase=='red' else normal,{'GTR':width,'POSE':2.5,'BBOX':2}[frame['kind']],coords,(4,3) if frame['kind']=='BBOX' else ()))
        for tag,color,stroke,coords,dash in styles:
            items=self.c.find_withtag(tag)
            if items:
                self.c.coords(items[0],*coords);self.c.itemconfigure(items[0],outline=color,width=stroke,dash=dash)
            else:self.c.create_polygon(*coords,fill='',outline=color,width=stroke,dash=dash,tags=(TAG,tag))
        self.c.tag_raise(TAG)
        pulse=getattr(self.ed,'corner_pulse',None)
        if pulse:pulse.draw()

    def clear(self):
        self._generation+=1
        for token in tuple(self.after_ids):self._cancel(token)
        self._expiry=None;self.active=None;self._context=None;self.phase='normal';self.started_at=None
        self._editing=False;self._mode=None;self._selected=None;self._flash_complete=True
        pulse=getattr(self.ed,'corner_pulse',None)
        if pulse and pulse.pending is not None:pulse.clear()
        try:self.c.delete(TAG)
        except tk.TclError:pass

    def _destroy(self,event):
        if event.widget is self.c:self.clear()
