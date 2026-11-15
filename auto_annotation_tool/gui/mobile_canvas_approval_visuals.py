"""Read-only colors for physical GTR approval and image verification."""
GREEN='#00F03C'
PENDING='#FFBC00'
RED='#FF1744'
POSE='#00DFFF'
BBOX='#659CFF'
DARK='#080808'


def polygon(ed,coords,*,color,width,tags,dash=(),underlay='navigator_contrast_underlay'):
    """One contour with a one-pixel dark edge, on identical coordinates."""
    from .mobile_canvas_editor import TAG
    ed.c.create_polygon(*coords,fill='',outline=DARK,width=width+2,dash=dash,
                        tags=(TAG,'frame_contrast_underlay',underlay))
    return ed.c.create_polygon(*coords,fill='',outline=color,width=width,dash=dash,tags=tags)


def approved_ids(scene,state):
    from ..ranking import mobile_canvas_partial_review as partial
    progress=partial.progress(scene)
    active=partial.manual_plate_ids(scene,state)|{
        link['plate_id'] for oid,link in state['links'].items()
        if link.get('plate_id') and progress['object_decisions'].get(oid)==partial.PLATE}
    conflicted={c['plate_id'] for c in state['conflicts']}
    return {pid for pid in active-set(progress['pending_gtr_ids'])-conflicted
            if state['plates'][pid]['confirmed'] and state['plates'][pid]['gt_quad']}


def color(scene,state,pid):
    return GREEN if pid in approved_ids(scene,state) else PENDING


def paint_image_border(ed,scene):
    """Outline the visible image area, independently of ROI and REF/INF."""
    from ..ranking.mobile_frame_review_state import from_scene,READY,SOURCE_NA,NO_REFERENCE
    from .mobile_canvas_editor import TAG
    if ed.mode=='draw':return
    x,y=ed.to_canvas((0,0));r,b=ed.to_canvas(scene['image_size'])
    # Keep the border visible when zoom clips the image to the Canvas.
    box=(max(2,x+1.5),max(2,y+1.5),min(ed.c.winfo_width()-2,r-1.5),min(ed.c.winfo_height()-2,b-1.5))
    if box[0]>=box[2] or box[1]>=box[3]:return
    state=from_scene(scene)['state'];verified=state==READY;technical=state in {SOURCE_NA,NO_REFERENCE}
    ed.c.create_rectangle(*box,outline=DARK,width=5,tags=(TAG,'image_border_contrast'))
    ed.c.create_rectangle(*box,outline='#8596A6' if technical else GREEN if verified else RED,width=3,
                          tags=(TAG,'image_verification_border','image_na' if technical else 'image_verified' if verified else 'image_pending'))
