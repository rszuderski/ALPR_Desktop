"""Current and total BBOX/Pose/GTR positions, placed beside their badge."""


def paint(ed,context,layer_view):
    from .mobile_canvas_editor import TAG
    from .mobile_canvas_clean_view import intersects
    import tkinter.font as tkfont
    if not hasattr(ed,'_stack_count_font'):
        ed._stack_count_font=tkfont.Font(root=ed.c,font=('Segoe UI',8,'bold'))
    kinds=[layer['kind'] for layer in layer_view['layers']] if layer_view else [context['kind']]
    if context['kind'] not in kinds:return
    count=len(kinds);current=kinds.index(context['kind'])+1
    text=str(count);position=str(current);font=ed._stack_count_font
    diameter=max(18,font.metrics('linespace')+5,font.measure(text)+8,font.measure(position)+8)
    between=max(8,font.measure('/')+4);width=2*diameter+between
    x,y,r,b=context['rect'];gap=4;cw,ch=ed.c.winfo_width(),ed.c.winfo_height()
    proposals=((r+gap,y),(x-gap-width,y),(r-width,y-gap-diameter),(r-width,b+gap),
               (r+gap,b-diameter),(x-gap-width,b-diameter),(x,y-gap-diameter),(x,b+gap))
    obstacles=ed.clean_geometry_boxes+ed.clean_label_boxes
    for left,top in proposals:
        group=(left,top,left+width,top+diameter)
        if left<2 or top<2 or group[2]>cw-2 or group[3]>ch-2:continue
        if any(intersects(group,obstacle) for obstacle in obstacles):continue
        active=(left,top,left+diameter,top+diameter);total=(left+diameter+between,top,left+width,top+diameter)
        common=(TAG,'stack_position_badges',context['oid'])
        tags=common+('stack_current_badge',)
        ed.c.create_oval(*active,fill='#FFFFFF',outline='#145A7A',width=1.5,tags=tags)
        ed.c.create_text(left+diameter/2,top+diameter/2,text=position,font=font,fill='#08263B',tags=tags+('stack_current_text',))
        middle=left+diameter+between/2
        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
            ed.c.create_text(middle+dx,top+diameter/2+dy,text='/',font=font,fill='#080808',tags=common+('stack_position_separator','stack_slash_halo'))
        ed.c.create_text(middle,top+diameter/2,text='/',font=font,fill='#FFFFFF',tags=common+('stack_position_separator',))
        tags=common+('stack_count_badge',)
        ed.c.create_oval(*total,fill='#145A7A',outline='#EAF5FC',width=1,tags=tags)
        ed.c.create_text(total[0]+diameter/2,top+diameter/2,text=text,font=font,fill='#FFFFFF',tags=tags+('stack_count_text',))
        ed.clean_label_boxes.append(group)
        ed._stack_counter=dict(count=count,current=current,rect=total,current_rect=active,group_rect=group,oid=context['oid'])
        return
