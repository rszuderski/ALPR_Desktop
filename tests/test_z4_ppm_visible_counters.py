"""Direct PPM, visible crossed inference frames, readable menus and counters."""
import copy
from tkinter import font as tkfont
from test_z4_clean_canvas import night,pair,pick,focus,frozen,unchanged
from test_z4_frame_stacks import ready
import test_mobile_p2b2_canvas_gui as native
from auto_annotation_tool.gui import mobile_canvas_clean_view as view
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as data

def center(q):return tuple(sum(p[i] for p in q)/4 for i in (0,1))
def crosses(h,oid):return [i for i in h.canvas.find_withtag('fp_diagonal') if oid in h.canvas.gettags(i)]

def test_C01_PPM_needs_no_selection_and_draws_two_diagonals(pair):
    root,h,ed,s,gid=pair;before=copy.deepcopy(core.scene(s,gid)['parents']);rev=s.review.review_revision
    assert not ed.selected
    q=data.snapshot(s,gid)['candidates'][0]['mt_quad'];native.click(h,ed,center(q),3);ready(root,h,ed)
    assert core._object(core.scene(s,gid),'GT:a0')['error'] and s.review.review_revision==rev+1
    assert 'GT:a0' in view.visible_ids(ed) and len(crosses(h,'GT:a0'))==2
    assert core.scene(s,gid)['parents']==before and len(data.snapshot(s,gid)['plates'])==1
    # A second right click reverses the same visible source candidate.
    native.click(h,ed,center(q),3);ready(root,h,ed)
    assert not core._object(core.scene(s,gid),'GT:a0')['error'] and not crosses(h,'GT:a0')
    ed.undo();ready(root,h,ed);assert len(crosses(h,'GT:a0'))==2

def test_C02_saved_FP_default_on_and_source_diagonals(night):
    root,h,ed,s,gid=night;before=frozen(s);fp=next(c for c in data.snapshot(s,gid)['candidates'] if c['association_status']=='UNASSOCIATED_NO_PLATE')
    assert ed.show_fp_var.get() and fp['object_id'] in view.visible_ids(ed)
    lines=crosses(h,fp['object_id']);assert len(lines)==2
    assert h.canvas.coords(lines[0])==[v for p in (fp['mt_quad'][0],fp['mt_quad'][2]) for v in ed.to_canvas(p)]
    view.toggle_fp(ed);ready(root,h,ed);assert not crosses(h,fp['object_id'])
    view.toggle_fp(ed);ready(root,h,ed);assert len(crosses(h,fp['object_id']))==2;unchanged(s,before)

def test_C03_focus_hides_other_FP_and_crosses(night):
    root,h,ed,s,gid=night;before=frozen(s);assert h.canvas.find_withtag('fp_diagonal')
    oid,(x,y)=focus(night)
    assert not h.canvas.find_withtag('fp_diagonal') and not h.canvas.find_withtag('mt_original')
    assert len(h.canvas.find_withtag('corner_handle'))==4 and view.right_hit(ed,x,y) is None
    h.canvas.event_generate('<ButtonRelease-1>',x=round(x),y=round(y));ready(root,h,ed)
    assert not h.canvas.find_withtag('fp_diagonal');ed.escape();ready(root,h,ed)
    assert h.canvas.find_withtag('fp_diagonal');unchanged(s,before)

def test_C04_PPM_hits_source_instead_of_moved_GT(pair):
    root,h,ed,s,gid=pair
    q=[[700,400],[850,400],[850,460],[700,460]];core.correct_plate(s,gid,'GT:a0',q);ready(root,h,ed)
    ed._choose_plate('GT:a0');ready(root,h,ed);before=frozen(s)
    native.click(h,ed,center(q),3);ready(root,h,ed);unchanged(s,before)
    source=data.snapshot(s,gid)['candidates'][0]['mt_quad'];native.click(h,ed,center(source),3);ready(root,h,ed)
    assert core._object(core.scene(s,gid),'GT:a0')['error'] and len(crosses(h,'GT:a0'))==2
    assert next(iter(data.snapshot(s,gid)['plates'].values()))['gt_quad']==q

def test_C05_RAW_cannot_be_targeted_as_final(night):
    root,h,ed,s,gid=night;raw=next(c for c in data.snapshot(s,gid)['candidates'] if c['role']=='SUPPRESSED');q=raw['mt_quad']
    assert raw['object_id'] not in view.visible_ids(ed)
    assert view.right_hit(ed,*ed.to_canvas(center(q)))!=raw['object_id']
    before=frozen(s);native.click(h,ed,(0,0),3);ready(root,h,ed);unchanged(s,before)

def contrast(widget,fg,bg):
    def lum(color):
        rgb=[v/65535 for v in widget.winfo_rgb(color)]
        rgb=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb]
        return .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
    a,b=sorted([lum(fg),lum(bg)]);return (b+.05)/(a+.05)

def test_C06_menu_contrast_and_no_disabled_help(pair):
    root,h,ed,s,gid=pair
    from auto_annotation_tool.gui.mobile_canvas_ui import palette
    for background in ('#F4F5EC','#18252E'):
        colors=view.menu_colors(palette(h.window,background))
        assert contrast(h.window,colors['foreground'],colors['background'])>=4.5
        assert contrast(h.window,colors['activeforeground'],colors['activebackground'])>=4.5
        assert contrast(h.window,colors['disabledforeground'],colors['background'])>=4.5
    for menu in (ed.clean_menu,ed.clean_help_menu):
        assert contrast(menu,menu.cget('foreground'),menu.cget('background'))>=4.5
    assert all(ed.clean_help_menu.entrycget(i,'state')=='normal' for i in range(ed.clean_help_menu.index('end')+1))
    assert ed.clean_menu.entrycget(0,'label')=='Pokaż błędne wskazania'

def test_C07_exclusion_is_a_panel_action_with_required_reason(pair):
    root,h,ed,s,gid=pair;before=frozen(s)
    labels=[ed.clean_menu.entrycget(i,'label') for i in range(ed.clean_menu.index('end')+1)]
    assert not any('ocen' in label for label in labels)
    assert ed.clean_exclude_button.winfo_ismapped();ed.clean_exclude_button.invoke();native.pump(root);unchanged(s,before)
    assert ed.clean_exclusion.winfo_ismapped() and not view.save_exclusion(ed);unchanged(s,before)
    ed.clean_reason.set('Test: zasłonięta klatka');assert view.save_exclusion(ed);ready(root,h,ed)
    assert core.scene(s,gid)['exclusion_reason']=='Test: zasłonięta klatka' and ed.clean_restore_button.winfo_ismapped()
    ed.clean_restore_button.invoke();ready(root,h,ed);assert not core.scene(s,gid)['exclusion_reason']

def test_C08_only_counter_numbers_grow_and_all_contexts_fit(night):
    from z4_sidebar_assertions import assert_sidebar_reaches
    root,h,ed,s,gid=night;pick(night);before=frozen(s)
    assert ed.contexts.counter_labels['decisions'].cget('text')=='Pozostałe decyzje'
    for size in ('1649x896','1180x760'):
        h.window.geometry(size+'+20+20');ready(root,h,ed)
        for label in ed.contexts.counters.values():
            assert tkfont.Font(root=root,font=label.cget('font')).actual('size')==20
        assert_sidebar_reaches(root,h,list(ed.contexts.counters.values()))
        for label in [*ed.contexts.fields.values(),*ed.contexts.counter_labels.values()]:
            assert tkfont.Font(root=root,font=label.cget('font')).actual('size')==10
        assert ed.clean_exclude_button.winfo_ismapped() and h.review_sidebar_scroll.winfo_ismapped()
    from auto_annotation_tool.ranking.mobile_mt_detection_evaluation import evaluate_unit
    result=evaluate_unit(s,gid);measured=result['status']=='MEASURED'
    for key,field in (('gt','n_gt'),('mt','tp'),('fp','fp'),('fn','fn')):
        assert ed.contexts.counters[key].cget('text')==(str(result[field]) if measured else '—')
    unchanged(s,before)

def test_C09_exclusion_draft_cannot_move_to_next_image(night):
    root,h,ed,s,gid=night;before=frozen(s);ed.clean_exclude_button.invoke();ed.clean_reason.set('Stary szkic')
    ed.navigate(1);native.pump(root,lambda:h.current_invocation!=gid and not h._saving);ready(root,h,ed)
    assert not ed.clean_exclusion.winfo_ismapped() and ed.clean_reason.get()=='' and not view.save_exclusion(ed)
    unchanged(s,before)
