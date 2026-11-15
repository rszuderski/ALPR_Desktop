"""Hold Shift over this Canvas: noninteractive source/reference contours."""
import tkinter as tk


class ShiftPreview:
    def __init__(self,editor):
        self.ed=editor;self.held=set();self.bindings=[]
        for key in ('Shift_L','Shift_R'):
            self.bind(editor.c,'<KeyPress-'+key+'>',self.press)
            self.bind(editor.c,'<KeyRelease-'+key+'>',self.release)
            self.bind(editor.h.window,'<KeyRelease-'+key+'>',self.release)
        self.bind(editor.c,'<FocusOut>',self.focus_out)
        self.bind(editor.h.window,'<FocusOut>',self.focus_out)
        self.notebook=None;parent=editor.c.master
        while parent is not None and parent is not editor.h.window:
            if parent.winfo_class()=='TNotebook':
                self.notebook=parent;self.bind(parent,'<<NotebookTabChanged>>',self.focus_out);break
            parent=parent.master

    def bind(self,widget,event,callback):
        self.bindings.append((widget,event,widget.bind(event,callback,add='+')))

    @property
    def active(self):
        from .mobile_canvas_visibility import gt_only
        if gt_only(self.ed):return False
        return bool(self.held and self.ed.mode!='draw' and not self.ed.drag)

    def press(self,event):
        if self.ed._is_typing_shortcut(event.widget):return
        before=self.active;self.held.add(event.keysym)
        if self.active!=before:self.ed.request_overlay()

    def release(self,event):
        before=self.active;self.held.discard(event.keysym)
        if self.active!=before:self.ed.request_overlay()

    def clear(self):
        before=self.active;self.held.clear()
        if before:self.ed.request_overlay()

    def focus_out(self,event=None):self.clear()

    def dispose(self):
        self.clear()
        for widget,event,token in self.bindings:
            try:widget.unbind(event,token)
            except tk.TclError:pass
        self.bindings=[]

    def frames(self):
        """All proven geometry on this exact source image, independent of focus."""
        ed=self.ed;s=ed._read();session=ed.h.session;gid=ed.h.current_invocation
        if not s:return []
        key=(id(session),session.review.review_revision,gid,s['source_binding'])
        if getattr(self,'_frames_key',None)==key:return self._frames
        from ..ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks
        from ..ranking.mobile_canvas_operator_view import saved_reference_ids
        from ..ranking.mobile_mt_geometry_layers import source_bbox
        from .mobile_canvas_approval_visuals import RED,POSE,BBOX
        from ..ranking.mobile_canvas_source_hud import _context,_identity
        gids=[gid]
        try:identity=_identity(session.mt_invocations[gid],_context(session,gid))
        except (ValueError,KeyError,TypeError,OSError):identity=None
        if identity is not None:
            for other,group in session.mt_invocations.items():
                if other==gid:continue
                try:same=_identity(group,_context(session,other))==identity
                except (ValueError,KeyError,TypeError,OSError):continue
                if same:gids.append(other)
        frames=[]
        for other in gids:
            try:
                scene=s if other==gid else core.scene(session,other)
                state=stacks.snapshot(session,other,scene)
                from .mobile_canvas_spatial_reference import project
                state=project(session,other,scene,state)
            except (ValueError,KeyError,TypeError,OSError):continue
            from ..ranking.mobile_canvas_operator_view import reference_numbers
            numbers=reference_numbers(state)
            for c in state['candidates']:
                role='FINAL' if c['role']=='FINAL' else 'RAW · odrzucony przez NMS' if c['role']=='SUPPRESSED' else 'rola nieustalona'
                detail=dict(mt_index=c['mt_detection_index']+1,role=c['role'],attempt_id=c['attempt_id'],
                    label=f"MT {c['mt_detection_index']+1} · {role}")
                bad=c.get('operator_bad') or c['association_status']=='UNASSOCIATED_NO_PLATE'
                color=RED if bad else POSE if c['role']=='FINAL' else '#AEB8C2'
                bbox=source_bbox(session,scene,c)
                if bbox:
                    l,t,r,b=bbox;frames.append(dict(kind='BBOX',quad=[(l,t),(r,t),(r,b),(l,b)],color=RED if bad else BBOX if c['role']=='FINAL' else color,oid=c['object_id'],gid=other,**detail))
                else:
                    from .mobile_canvas_bbox_marker import describe,quad
                    marker=describe(session,other,scene,c)
                    if marker['marker']:
                        frames.append(dict(kind='BBOX',quad=quad(marker),marker=marker,color=BBOX if c['role']=='FINAL' else color,oid=c['object_id'],gid=other,**detail))
                if c['mt_quad']:
                    try:core.validate_mt_quad(c['mt_quad'],*scene['image_size'])
                    except (ValueError,TypeError):continue
                    frames.append(dict(kind='POSE',quad=c['mt_quad'],color=color,oid=c['object_id'],gid=other,**detail))
                from .mobile_canvas_reference_seed import project
                seed=project(session,other,scene,state,c)
                if seed:frames.append(dict(kind='GTR',quad=seed['quad'],color='#FFD400',oid=c['object_id'],gid=other,
                    seed=True,role='SEED',label=f"GTR startowy · MT {c['mt_detection_index']+1} · niezapisany",attempt_id=c['attempt_id']))
            saved=saved_reference_ids(scene,state)
            from .mobile_canvas_approval_visuals import approved_ids,GREEN,PENDING
            approved=approved_ids(scene,state)
            for pid,p in state['plates'].items():
                if pid in saved:
                    linked=[str(c['mt_detection_index']+1) for c in state['candidates'] if c['role']=='FINAL' and state['links'].get(c['object_id'],{}).get('plate_id')==pid]
                    frames.append(dict(kind='GTR',quad=p['gt_quad'],color=GREEN if pid in approved else PENDING,oid=pid,gid=other,
                        role='OPERATOR',label=f"GTR {numbers[pid]} · operator · "+('MT '+', '.join(linked) if linked else 'bez powiązania MT')))
        self._frames_key=key;self._frames=frames
        return frames

    def paint(self):
        if not self.active:return
        from . import mobile_canvas_clean_view as clean
        from .mobile_canvas_editor import TAG
        from .mobile_canvas_approval_visuals import polygon
        ed=self.ed
        for frame in sorted(self.frames(),key=lambda frame:frame['kind']=='GTR'):
            q=frame['quad'];kind=frame['kind'];tags=(TAG,'shift_source_preview',frame['gid'],frame['oid'])
            tag={'BBOX':'shift_mt_source_bbox','POSE':'shift_mt_pose','GTR':'shift_saved_gtr'}[kind]
            from .mobile_canvas_selection import accent
            accent(ed,frame,2 if kind=='GTR' else 1.5)
            if frame.get('marker'):
                from .mobile_canvas_bbox_marker import paint
                paint(ed,frame['marker'],color=frame['color'],tags=tags+(tag,))
            else:polygon(ed,[v for p in q for v in ed.to_canvas(p)],color=frame['color'],width=2 if kind=='GTR' else 1.5,
                    dash=(4,3) if kind=='BBOX' else (),tags=tags+(tag,),underlay='shift_contrast_underlay')
            ed.clean_geometry_boxes.append(clean.bounds(ed,q))
        # Every rendered contour has its own identity, including suppressed
        # RAW. Labels are diagnostic only; they do not enter layer navigation.
        import tkinter.font as tkfont
        font=tkfont.Font(root=ed.c,font=('Segoe UI',9,'bold'))
        frames=sorted(self.frames(),key=lambda frame:frame['kind']=='GTR')
        ed.shift_preview_labels=[]
        for index,frame in enumerate(frames):
            label=frame['label']+(' · '+frame['kind'] if frame['kind']!='GTR' else '')
            if frame.get('marker'):label+=' INVALID · '+frame['marker']['reason']
            if len({f['gid'] for f in frames})>1:label+=' · wyw. '+str(1+list(dict.fromkeys(f['gid'] for f in frames)).index(frame['gid']))
            width=min(font.measure(label)+16,max(100,ed.c.winfo_width()-12));height=font.metrics('linespace')+12
            rect=clean.badge_position(ed,clean.bounds(ed,frame['quad']),width,height)
            if rect is None:
                y=6+index*(height+4);rect=(6,y,6+width,y+height)
            ed.clean_label_boxes.append(rect);x,y,r,b=rect
            tags=(TAG,'shift_identity_badge',frame['gid'],frame['oid'],frame['kind'],str(index))
            ed.c.create_rectangle(*rect,fill='#13252D',outline=frame['color'],tags=tags)
            ed.c.create_text(x+8,(y+b)/2,text=label,font=font,anchor='w',fill='#FFFFFF',width=width-16,tags=tags+('shift_identity_text',))
            clean.badge_tether(ed,frame['quad'],rect,frame['oid'],tag='shift_identity_tether')
            ed.shift_preview_labels.append(dict(frame=frame,text=label,rect=rect))
