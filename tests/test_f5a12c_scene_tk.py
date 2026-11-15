"""Native Windows Tk scene scope in the same Z4 editor; producer copies only."""
import pytest
import test_mobile_p2b2_canvas_gui as native
from test_f5a12c_scene_evidence import private_session
from auto_annotation_tool.ranking import mobile_scene_frame_review as scenes


def test_T19_native_corner_edit_release_and_immediate_next_grip(tmp_path,monkeypatch):
    import hashlib
    s=private_session(tmp_path,'zero-roi-with-plate');key=next(iter(s.scene_frames))
    oid=scenes.add_plate(s,key,[(25,35),(65,35),(65,55),(25,55)])
    physical_id=scenes.scene(s,key)['objects'][0]['physical_gt_id']
    monkeypatch.setattr(native,'source_session',lambda _:s)
    generator=native.gui.__wrapped__(tmp_path,monkeypatch)
    root,h,ed,s=next(generator)
    try:
        monkeypatch.setattr(ed,'_pointer_for_selection',lambda event=None:ed.pointer_canvas)
        native.select(h,ed,(45,45))
        assert ed.selected==oid
        image=h._image
        for index in (0,1):
            point=scenes.scene(s,key)['objects'][0]['quad'][index]
            x,y=map(round,ed.to_canvas(point))
            revision=s.review.review_revision
            old_sha=hashlib.sha256(s.sidecar_path.read_bytes()).hexdigest()
            h.canvas.event_generate('<ButtonPress-1>',x=x,y=y)
            assert ed.drag and ed.drag['index']==index and not h._saving
            h.canvas.event_generate('<B1-Motion>',x=x+4,y=y+2)
            assert s.review.review_revision==revision
            assert hashlib.sha256(s.sidecar_path.read_bytes()).hexdigest()==old_sha
            h.canvas.event_generate('<ButtonRelease-1>',x=x+4,y=y+2)
            assert s.review.review_revision==revision+1
            assert h._image is image and not h._saving
        saved=scenes.scene(s,key)['objects'][0]
        assert saved['physical_gt_id']==physical_id
        ed.undo();ed.undo(redo=True)
        assert scenes.scene(s,key)['objects'][0]['quad']==saved['quad']
        assert len(scenes.scene(s,key)['objects'])==1
    finally:
        try:next(generator)
        except StopIteration:pass


def test_T21_native_presence_only_then_same_identity_corners(tmp_path,monkeypatch):
    from auto_annotation_tool.gui.mobile_canvas_scene_frames import show_presence_only
    from auto_annotation_tool.gui.mobile_canvas_clean_view import save_exclusion
    s=private_session(tmp_path,'zero-roi-with-plate')
    monkeypatch.setattr(native,'source_session',lambda _:s)
    generator=native.gui.__wrapped__(tmp_path,monkeypatch)
    root,h,ed,s=next(generator)
    try:
        key=next(iter(s.scene_frames))
        monkeypatch.setattr(ed,'_pointer_for_selection',lambda event=None:ed.pointer_canvas)
        show_presence_only(ed)
        ed.clean_reason.set('Rogi chwilowo zasloniete')
        assert save_exclusion(ed)
        original=scenes.scene(s,key)['objects'][0]
        assert original['quad'] is None and ed.entry is not None
        ed.entry.delete(0,'end');ed.entry.insert(0,'SYN002');ed.entry_enter()
        assert scenes.scene(s,key)['objects'][0]['gt_text']=='SYN002'
        native.key(h,'d')
        for point in [(20,30),(60,30),(60,50),(20,50)]:
            native.click(h,ed,point)
        current=scenes.scene(s,key)['objects']
        assert len(current)==1 and current[0]['physical_gt_id']==original['physical_gt_id']
        assert current[0]['quad'] is not None
    finally:
        try:next(generator)
        except StopIteration:pass

def test_T19_native_menu_two_final_links_and_source_rows(tmp_path,monkeypatch):
    from auto_annotation_tool.gui.mobile_canvas_stacks import context_menu
    s=private_session(tmp_path,'two-roi')
    monkeypatch.setattr(native,'source_session',lambda _:s)
    generator=native.gui.__wrapped__(tmp_path,monkeypatch)
    root,h,ed,s=next(generator)
    try:
        key=next(iter(s.scene_frames))
        monkeypatch.setattr(ed,'_pointer_for_selection',lambda event=None:ed.pointer_canvas)
        native.key(h,'d')
        for point in [(72,40),(78,40),(78,50),(72,50)]:
            native.click(h,ed,point)
        oid=ed.selected
        assert oid and len(ed.scene_mt_tree_ids)==2
        menu=context_menu(ed,show=False)
        labels=[menu.entrycget(i,'label') for i in range(menu.index('end')+1)]
        index=next(i for i,v in enumerate(labels) if v=='Ta GTR odpowiada FINAL MT 1')
        menu.invoke(index)
        menu=context_menu(ed,show=False)
        labels=[menu.entrycget(i,'label') for i in range(menu.index('end')+1)]
        menu.invoke(next(i for i,v in enumerate(labels) if v=='Ta GTR odpowiada FINAL MT 2'))
        obj=scenes.scene(s,key)['objects'][0]
        assert len(obj['mt_attempt_links'])==2 and len(scenes.scene(s,key)['objects'])==1
        for iid,(_,gid) in list(ed.scene_mt_tree_ids.items()):
            ed.select_tree_row(iid)
            assert h.current_record==('scene',key) and ed.scene_mt_focus==gid
            assert scenes.scene(s,key)['objects'][0]['object_id']==oid
        assert len(s.mt_invocations)==2 and not s.review.invocation_annotations
    finally:
        try:next(generator)
        except StopIteration:pass



@pytest.mark.parametrize('theme,size',[('light',(1649,896)),('dark',(1649,896)),('light',(1180,760)),('dark',(1180,760))])
def test_T28_scene_scope_existing_layout_sizes(tmp_path,monkeypatch,theme,size):
    from auto_annotation_tool.gui import mobile_canvas_editor as editor
    original=editor.palette
    monkeypatch.setattr(editor,'palette',lambda w,b:original(w,'#FFFFFF' if theme=='light' else '#17232D'))
    s=private_session(tmp_path,'zero-roi-with-plate')
    monkeypatch.setattr(native,'source_session',lambda _:s)
    generator=native.gui.__wrapped__(tmp_path,monkeypatch)
    root,h,ed,s=next(generator)
    try:
        h.window.geometry(f'{size[0]}x{size[1]}+20+20');h.window.update()
        assert ed.available() and h.canvas.winfo_ismapped()
        assert ed.contexts.winfo_rootx()+ed.contexts.winfo_width()<=h.window.winfo_rootx()+h.window.winfo_width()+2
        assert h.subject_tree.winfo_ismapped() and ed.clean_gtr_confirm_button.winfo_ismapped()
        assert len(ed.review_tabs.tabs())==4
        assert not s.review.scene_annotations
    finally:
        try:next(generator)
        except StopIteration:pass



@pytest.mark.parametrize('name',['zero-roi','zero-roi-with-plate','two-roi'])
def test_T19_native_gui_export_complete_roundtrip(tmp_path,monkeypatch,name,request):
    import json,hashlib
    from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
    from auto_annotation_tool.gui.mobile_comparison_core import load_session_export
    s=private_session(tmp_path,name);key=next(iter(s.scene_frames))
    if name=='zero-roi':scenes.confirm_empty(s,key)
    else:
        oid=scenes.add_plate(s,key,[(72,40),(78,40),(78,50),(72,50)])
        scenes.edit(s,key,oid,text='',reason='Synthetic unreadable text')
        if name=='two-roi':
            for role in s.mt_detection_roles.values():
                for aid in role['final_attempt_ids']:
                    scenes.decide_mt(s,key,aid,reference=oid)
    assert not s.completion_issues(), s.completion_issues()
    s.complete()
    original=json.loads(json.dumps(s.review.to_dict()))
    source_sha=hashlib.sha256(s.path.read_bytes()).hexdigest()
    monkeypatch.setattr(native,'source_session',lambda _:s)
    generator=native.gui.__wrapped__(tmp_path,monkeypatch)
    root,h,ed,s=next(generator)
    try:
        out=tmp_path/'03_wyniki';out.mkdir()
        errors=[]
        monkeypatch.setattr(native.module.messagebox,'showerror',lambda *a,**kwargs:errors.append(a))
        monkeypatch.setattr(native.module.filedialog,'askdirectory',lambda **kwargs:str(out))
        monkeypatch.setattr(native.module.messagebox,'askyesno',lambda *a,**kwargs:True)
        h.export()
        native.pump(root,lambda:not h._saving,seconds=90)
        assert not errors,errors
        manifests=list(out.rglob('export_manifest.json'))
        assert len(manifests)==1,h.status_var.get()
        folder=manifests[0].parent
        result=load_session_export(folder)
        assert result.scene_assessment['scene_annotations']==original['scene_annotations']
        assert result.scene_assessment['source_mt_roles']==list(s.mt_detection_roles.values())
        assert result.scene_assessment['n_scene_events']==1
        assert result.scene_assessment['tv_e2e_reason']=='SCENE_UNMAPPED'
        imported=MobileReviewSession.open(s.path,sidecar_path=folder/'review.json')
        assert imported.review.to_dict()==original
        assert imported.review.review_status=='COMPLETED'
        assert hashlib.sha256(s.path.read_bytes()).hexdigest()==source_sha
        manifest=json.loads(manifests[0].read_text(encoding='utf-8'))
        request.node.user_properties.append(('roundtrip',dict(
            fixture=name,source_sha256=source_sha,
            manifest_sha256=hashlib.sha256(manifests[0].read_bytes()).hexdigest(),
            review_sha256=hashlib.sha256((folder/'review.json').read_bytes()).hexdigest(),
            manifest_dependencies=len(manifest['files']),
            review_revision=imported.review.review_revision,review_status=imported.review.review_status,
            scene_events=len(imported.scene_frames),mt_invocations=len(imported.mt_invocations),
            attempts=len(imported.attempts),physical_gt=sum(len(v['objects']) for v in imported.review.scene_annotations.values()),
            explicit_final_links=sum(len(v['mt_decisions']) for v in imported.review.scene_annotations.values()),
            full_native_export=True,save_restart_export_import='PASS')))
    finally:
        try:next(generator)
        except StopIteration:pass


@pytest.mark.parametrize('name',['zero-roi','zero-roi-with-plate','two-roi'])
def test_T28_existing_canvas_scene_scope_gestures(tmp_path,monkeypatch,name):
    session=private_session(tmp_path,name)
    monkeypatch.setattr(native,'source_session',lambda _:session)
    generator=native.gui.__wrapped__(tmp_path,monkeypatch)
    root,h,ed,s=next(generator)
    try:
        # Keep the user's physical pointer untouched. Model the pointer from
        # private Canvas Motion events while exercising the actual key bindings.
        monkeypatch.setattr(ed, '_pointer_for_selection', lambda event=None: ed.pointer_canvas)
        key=next(iter(s.scene_frames))
        assert h.current_record==('scene',key) and ed.available()
        assert len(ed.scene_tree_ids)==1
        assert 'SCENE_FRAME' in ed._read()['scope']
        if name=='zero-roi':
            assert not ed._read()['checked']
            native.key(h,'space')
            assert ed._read()['presence']=='NO_PLATES'
            ed.undo()
            assert ed._read()['presence']=='UNRESOLVED'
            ed.undo(redo=True)
            assert ed._read()['presence']=='NO_PLATES'
        else:
            native.key(h,'d')
            assert ed.mode=='draw'
            for point in [(15,35),(65,35),(65,55),(15,55)]:
                native.click(h,ed,point)
            assert len(ed._read()['objects'])==1
            oid=ed._read()['objects'][0]['object_id']
            ed.escape()
            assert not ed.selected
            native.select(h,ed,(40,45))
            assert ed.selected==oid
            ed.start_entry(oid)
            assert ed.entry is not None
            ed.entry.delete(0,'end');ed.entry.insert(0,'SYN123')
            ed.entry_enter()
            assert scenes.scene(s,key)['objects'][0]['gt_text']=='SYN123'
            ed.undo();ed.undo(redo=True)
            assert scenes.scene(s,key)['objects'][0]['gt_text']=='SYN123'
            native.key(h,'space')
            assert not scenes.scene(s,key)['objects'][0]['scene_confirmed']
            native.key(h,'space')
            assert scenes.scene(s,key)['objects'][0]['scene_confirmed']
        assert len(s.mt_invocations)==(2 if name=='two-roi' else 0)
    finally:
        try:next(generator)
        except StopIteration:pass
