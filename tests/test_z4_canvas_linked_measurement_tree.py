"""AKT67: operator navigates images only, not low-level detection records.

Synthetic fixtures. No source archive, saved GT, Android or experimental data is modified.
"""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

SOURCE=(Path(__file__).resolve().parent.parent / 'auto_annotation_tool' / 'gui' / 'mobile_canvas_editor.py')


def load_editor():
    prefix='akt67_image_list_test'
    for suffix in ('', '.gui', '.ranking'):
        name=prefix+suffix
        mod=types.ModuleType(name)
        mod.__path__=[]
        sys.modules[name]=mod
    core=types.ModuleType(prefix+'.ranking.mobile_canvas_review')
    core.merge_review_cells=lambda rows: tuple(
        '✓' if all(r[n]=='✓' for r in rows) else '○'
        for n in range(5)) if (rows:=list(rows)) else ('○',)*5
    sys.modules[core.__name__]=core
    for leaf in ('mobile_source_frame_roi','mobile_pose_metrics','mobile_canvas_presentation'):
        mod=types.ModuleType(prefix+'.ranking.'+leaf)
        sys.modules[mod.__name__]=mod
    sys.modules[prefix+'.ranking.mobile_source_frame_roi'].quad_contains_point=lambda *a,**kw:False
    sys.modules[prefix+'.ranking.mobile_pose_metrics'].validate_mt_quad=lambda *a,**kw:None
    sys.modules[prefix+'.ranking.mobile_canvas_presentation'].build_inventory=lambda session:session.inventory
    sys.modules[prefix+'.ranking.mobile_canvas_presentation'].subject_groups=lambda session,key:session.group_keys.get(key,[])
    # The overlay now shares the pure per-object provenance view-model. Keep
    # the existing event assertions and exercise that production helper too.
    from auto_annotation_tool.ranking.mobile_canvas_presentation import plate_review_state
    sys.modules[prefix+'.ranking.mobile_canvas_presentation'].plate_review_state=plate_review_state
    # This isolated tree fixture has no archive/evidence contract. Historical
    # inspection is covered by the native/real-source AKT89G tests.
    history=types.ModuleType(prefix+'.ranking.mobile_mt_history')
    history.history_evidence=lambda *a,**kw:None
    sys.modules[history.__name__]=history
    # Exercise the real fail-closed adapter: these legacy fixtures have no
    # validated scene registry, so they retain the existing navigation.
    from auto_annotation_tool.gui import mobile_canvas_observation_blocks
    sys.modules[prefix+'.gui.mobile_canvas_observation_blocks']=mobile_canvas_observation_blocks
    # This minimal legacy fixture intentionally has no session/source binding,
    # so it cannot construct candidate stacks. Stack behavior is exercised on
    # source-backed sessions by T01–T28; keep these event assertions unchanged.
    stacks=types.ModuleType(prefix+'.gui.mobile_canvas_stacks')
    stacks.view=lambda *a,**k:None
    stacks.enhanced=stacks.paint=stacks.ambiguous=stacks.associate_click=lambda *a,**k:False
    stacks.hit=stacks.cycle=lambda *a,**k:None
    stacks.members=lambda *a,**k:[]
    stacks.present=lambda editor,model:model
    stacks.write_reference=lambda editor,oid,quad=None,text=None,reason='':core.correct_plate(editor.h.session,editor.h.current_invocation,oid,quad) if quad is not None else core.set_text(editor.h.session,editor.h.current_invocation,oid,text,reason)
    sys.modules[stacks.__name__]=stacks
    ui=types.ModuleType(prefix+'.gui.mobile_canvas_ui')
    ui.palette=lambda *a,**k:{}
    ui.label=lambda *a,**k:None
    ui.MetricCard=ui.Disclosure=object
    sys.modules[ui.__name__]=ui
    panel=types.ModuleType(prefix+'.gui.mobile_canvas_results_panel')
    panel.CanvasResultsPanel=object
    sys.modules[panel.__name__]=panel
    spec=importlib.util.spec_from_file_location(prefix+'.gui.mobile_canvas_editor',SOURCE)
    obj=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=obj
    spec.loader.exec_module(obj)
    return obj


class Tree:
    def __init__(self):
        self.nodes={'':{'children':[], 'parent':None}}
        self.active=()
        self.focus_id=''
        self.seen=[]
    def exists(self,iid):return iid in self.nodes
    def parent(self,iid):return self.nodes[iid]['parent'] or ''
    def insert(self,parent,index,iid,text,values,tags,open=False):
        assert parent in self.nodes
        self.nodes[iid]=dict(parent=parent,children=[],text=text,values=values,tags=tags,open=open)
        self.nodes[parent]['children'].append(iid)
    def item(self,iid,option=None,**kwargs):
        if kwargs:self.nodes[iid].update(kwargs)
        return self.nodes[iid][option] if option is not None else self.nodes[iid]
    def get_children(self,parent=''):return tuple(self.nodes[parent]['children'])
    def delete(self,iid):
        for c in tuple(self.get_children(iid)):self.delete(c)
        self.nodes[self.parent(iid)]['children'].remove(iid)
        del self.nodes[iid]
        if iid in self.active:self.active=()
    def move(self,iid,parent,index):
        self.nodes[self.parent(iid)]['children'].remove(iid)
        self.nodes[parent]['children'].append(iid)
        self.nodes[iid]['parent']=parent
    def selection(self):return self.active
    def selection_set(self,iid):
        assert iid in self.nodes, iid
        self.active=(iid,)
    def selection_remove(self,*iid):
        if any(i in self.active for i in iid):self.active=()
    def focus(self,iid=''):
        if iid:self.focus_id=iid
    def see(self,iid):self.seen.append(iid)


def fixture(n=20, with_alias=True):
    mod=load_editor()
    names={f'subject-{i:02d}':f'key{i:02d}' for i in range(n)}
    group_keys={f'key{i:02d}':[f'G{i:02d}'] for i in range(n)}
    groups={f'G{i:02d}':SimpleNamespace(subject_keys=(f'key{i:02d}',),records=[{'id':f'a{i:02d}'}]) for i in range(n)}
    inv={f'G{i:02d}':{'status':'IN_PROGRESS','cells':('○','○','○','○','○'),
                      'scene':dict(checked=False,_post_nms_pending=False)} for i in range(n)}
    if with_alias:
        alias_index=n-1
        group_keys[f'key{alias_index:02d}']=[f'G{alias_index-1:02d}']
        groups.pop(f'G{alias_index:02d}')
        inv.pop(f'G{alias_index:02d}')
        groups[f'G{alias_index-1:02d}'].subject_keys=(f'key{alias_index-1:02d}',f'key{alias_index:02d}')
    session=SimpleNamespace(subjects={k:{} for k in names.values()},
        mt_invocations=groups,group_keys=group_keys, inventory={'groups':inv})
    tree=Tree()
    for i,iid in enumerate(names):
        tree.insert('','end',iid,text=f'Obraz {i+1:02d}',values=('','','','','',''),tags=('canvas_pending',),open=True)
    # Simulate confusing old subrows from AKT66 and previously selected alias.
    main=f'subject-{n-2:02d}'
    if with_alias:
        alias=f'subject-{n-1:02d}'
        tree.move(alias,main,'end')
        tree.insert(main,'end',f'canvas-object::{main}',text='Wynik 1',values=('','','','','',''),tags=('canvas_pending',))
        tree.insert(main,'end',f'canvas-raw::{main}',text='Wstępne 2',values=('','','','','',''),tags=('canvas_pending',))
        tree.selection_set(alias)
    h=SimpleNamespace(session=session,_subject_ids=names,subject_tree=tree,
        _schedule_status_pills=lambda:None,current_invocation=f'G{n-2:02d}',subject_key=f'key{n-2:02d}',
        _saving=False)
    navigation=[]
    def host_select():
        iid=tree.selection()[0]
        if editor.select_tree_row(iid):return
        h.subject_key=h._subject_ids[iid]
        h.current_invocation=group_keys[h.subject_key][0]
        navigation.append(iid)
    h._select_subject=host_select
    h._multi_focus_attempt=lambda aid: navigation.append('focus:'+aid)
    editor=mod.CanvasReviewEditor.__new__(mod.CanvasReviewEditor)
    editor.h=h
    editor.inventory=lambda:session.inventory
    editor.tree_refreshing=False
    editor.tree_branch=main
    editor.tree_targets={}
    editor.linked_tree_rows={}
    editor.selected='gtA';editor.hover='';editor.mode='edit'
    editor.prepare_navigation=lambda:True
    editor.update_header=lambda _inv:None
    editor.message=lambda msg:navigation.append('msg:'+msg)
    editor.pending_tree_target=None
    return mod,editor,tree,session,navigation


def test_twenty_source_rows_display_nineteen_images_and_no_children():
    mod,e,tree,s,events=fixture()
    raw=copy.deepcopy(s.group_keys)
    e.refresh_tree()
    roots=tree.get_children('')
    assert len(roots)==19
    assert roots[-1]=='subject-18'
    assert not tree.exists('subject-19')
    assert all(not tree.get_children(i) for i in roots)
    assert 'canvas-object::subject-18' not in tree.nodes
    assert 'canvas-raw::subject-18' not in tree.nodes
    assert tree.item('subject-18','text')=='Obraz 19  ·  Do sprawdzenia'
    assert all(all(word not in tree.item(i,'text') for word in ('Wynik','Wstępne','Zapis','NMS','RAW')) for i in roots)
    assert s.group_keys==raw
    assert tree.selection()==('subject-18',)
    # Refresh once again without multiplying or reshuffling tasks.
    e.refresh_tree()
    assert tree.get_children('')==roots


def test_human_status_explains_next_action_without_raw_labels():
    mod,e,tree,s,events=fixture(5,with_alias=False)
    s.inventory['groups']['G00']['status']='READY'
    s.inventory['groups']['G01']['status']='EXCLUDED'
    s.inventory['groups']['G02']['scene']['_post_nms_pending']=True
    s.inventory['groups']['G03']['scene']['checked']=True
    s.inventory['groups']['G03']['cells']=('✓','✓','○','✓','○')
    s.inventory['groups']['G04']['scene']=None
    e.refresh_tree()
    expected=('Gotowe','Poza oceną','Wyjaśnij rozbieżność','Uzupełnij dane','Brak pełnego obrazu')
    assert tuple(tree.item(i,'text').split('·')[-1].strip() for i in tree.get_children(''))==expected


def test_current_image_selection_event_never_clears_selected_gt():
    mod,e,tree,s,events=fixture()
    e.refresh_tree()
    current='subject-18'
    assert e.select_tree_row(current) is True
    assert e.selected=='gtA'
    assert e.mode=='edit'
    # Explicit alias subject currently active also maps to the visible owner.
    e.h.subject_key='key19'
    e.sync_tree_selection()
    assert tree.selection()==('subject-18',)
    assert e.selected=='gtA'


def test_keyboard_navigation_skips_hidden_duplicate():
    mod,e,tree,s,events=fixture()
    e.refresh_tree()
    e.h.subject_key='key17'
    e.tree_branch='subject-17'
    e.navigate(+1)
    assert tree.selection()==('subject-18',)
    assert e.h.subject_key=='key18'
    assert events==['subject-18']
    e.navigate(+1)
    assert tree.selection()==('subject-18',)
    assert len(events)==1
    e.navigate(-1)
    assert tree.selection()==('subject-17',)
    assert e.h.subject_key=='key17'


def test_next_pending_does_not_use_hidden_row_or_delete_evidence():
    mod,e,tree,s,events=fixture()
    e.refresh_tree()
    for i in range(18): s.inventory['groups'][f'G{i:02d}']['status']='READY'
    e.h.subject_key='key17'
    e.tree_branch='subject-17'
    tree.selection_set('subject-17')
    e.next_pending()
    assert tree.selection()==('subject-18',)
    assert e.h.subject_key=='key18'
    assert events==['subject-18']
    assert 'key19' in s.subjects


def test_ambiguous_group_mapping_fail_closed_not_hidden():
    mod,e,tree,s,events=fixture(4,with_alias=False)
    # Ambiguous related record has two groups. It must remain independently
    # visible rather than be silently merged with the owner image.
    s.group_keys['key03']=['G02','G03']
    s.mt_invocations['G02'].subject_keys=('key02','key03')
    e.refresh_tree()
    assert len(tree.get_children(''))==4
    assert tree.exists('subject-03')
    assert all(not tree.get_children(i) for i in tree.get_children(''))


def test_one_photo_with_two_true_rois_is_one_row_but_both_groups_preserved():
    mod,e,tree,s,events=fixture(3,with_alias=False)
    s.group_keys['key00']=['G00','G01']
    s.group_keys['key01']=['G01','G00']
    s.mt_invocations['G00'].subject_keys=('key00','key01')
    s.mt_invocations['G01'].subject_keys=('key00','key01')
    e.refresh_tree()
    assert len(tree.get_children(''))==3
    assert tree.get_children('subject-00')==()
    assert len(s.group_keys['key00'])==2


if __name__=='__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
