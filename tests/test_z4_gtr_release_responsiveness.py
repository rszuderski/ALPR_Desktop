"""Consecutive native GTR drags must not wait for statistics or ranking."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,cProfile,hashlib,json,os,pstats,shutil,threading,time
from io import StringIO
import pytest
from windows_dpi import enable_windows_dpi_awareness
DPI=enable_windows_dpi_awareness()
import test_mobile_p2b2_canvas_gui as native
from test_z4_frame_stacks import ready
from auto_annotation_tool.ranking.mobile_human_review import MobileReviewSession
from auto_annotation_tool.ranking import mobile_canvas_review as core,mobile_candidate_stacks as stacks
from auto_annotation_tool.ranking.mobile_session_metrics import current,result_key
from auto_annotation_tool.gui import mobile_canvas_navigator as nav,mobile_canvas_clean_view as clean
from auto_annotation_tool.gui import z4_mobile_sample_review as module

SID='exp-1792699088870-cc33764a'

@pytest.fixture
def gui(tmp_path,monkeypatch):
    baseline=os.environ.get('GTR_RELEASE_BASELINE')
    if not baseline:pytest.skip('Pinned private F4 source required')
    row=next(r for r in json.loads(Path(baseline).read_text(encoding='utf-8'))['sessions'] if r['session_id']==SID)
    source=tmp_path/'source.alprsession';side=tmp_path/'review.json'
    shutil.copyfile(row['private_source'],source);shutil.copyfile(row['private_sidecar'],side)
    session=MobileReviewSession.open(source,sidecar_path=side)
    monkeypatch.setattr(native,'source_session',lambda _:session)
    gen=native.gui.__wrapped__(tmp_path,monkeypatch);root,h,ed,s=next(gen)
    calls=[];store=NS(root_dir=tmp_path/'rank',apply_human_review=lambda *a,**k:calls.append('ranking') or False)
    monkeypatch.setattr(module,'MobilePackageExperimentStore',lambda *a,**k:store)
    h._test_ranking_calls=calls
    h.window.state('normal');h.window.geometry('1649x896+10+10');root.update();ready(root,h,ed)
    gid=h.current_invocation
    obj=next(o for o in core.scene(s,gid)['objects'] if o.get('quad'))
    oid=obj['object_id'];assert nav.select_layer(ed,'GTR',oid);ready(root,h,ed)
    h._test_oid=oid
    try:yield root,h,ed,s
    finally:
        if ed.drag:ed.drag=None
        # The generic fixture sends Escape during cleanup. This case already
        # closed/disposed the toplevel, so a second Escape has no live widgets.
        if h._closed:monkeypatch.setattr(ed,'escape',lambda *a,**k:None)
        try:next(gen)
        except StopIteration:pass

def position(ed,index):
    return tuple(map(round,ed.to_canvas(clean.selected(ed)['gt'][index])))

def grab(ed,index):
    x,y=position(ed,index)
    ed.c.event_generate('<ButtonPress-1>',x=x,y=y)
    return x,y

def move_release(ed,index,dx=5,dy=2):
    x,y=grab(ed,index);assert ed.drag is not None
    ed.c.event_generate('<B1-Motion>',x=x+dx,y=y+dy)
    ed.c.event_generate('<ButtonRelease-1>',x=x+dx,y=y+dy)

def wait_current(root,h,ed):
    native.pump(root,lambda:current(h.stats.get('live_session_metrics_v2'),result_key(h.session))
        and not h._saving and not ed.in_refresh and not ed.refresh_after,seconds=30)
    ready(root,h,ed)

def test_profile_real_corner_release_and_refresh(gui):
    root,h,ed,s=gui;original=s.review.review_revision
    x,y=grab(ed,0);ed.c.event_generate('<B1-Motion>',x=x+5,y=y+2)
    profile=cProfile.Profile();started=time.perf_counter();profile.enable()
    ed.c.event_generate('<ButtonRelease-1>',x=x+5,y=y+2)
    profile.disable();elapsed=time.perf_counter()-started
    assert s.review.review_revision==original+1
    out=os.environ.get('GTR_RELEASE_OUTPUT')
    if out:
        tag=os.environ.get('GTR_RELEASE_PHASE','profile')
        stream=StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(35)
        (Path(out)/(tag+'_release_profile.txt')).write_text(stream.getvalue(),encoding='utf-8')
        (Path(out)/(tag+'_release_profile.json')).write_text(json.dumps(dict(release_ms=1000*elapsed,source_revision=original)),encoding='utf-8')
    wait_current(root,h,ed)

def test_next_corner_can_be_saved_while_statistics_for_previous_release_runs(gui,monkeypatch):
    root,h,ed,s=gui;release_stats=threading.Event();started=threading.Event();observed=[]
    original=MobileReviewSession.statistics
    def delayed(snapshot):
        if threading.current_thread() is not threading.main_thread():
            frozen=hashlib.sha256(json.dumps(snapshot.review.to_dict(),sort_keys=True).encode()).hexdigest()
            observed.append(snapshot.review.review_revision);started.set()
            assert release_stats.wait(8),'Test did not release statistics'
            observed.append(snapshot.review.review_revision)
        result=original(snapshot)
        if threading.current_thread() is not threading.main_thread():
            assert hashlib.sha256(json.dumps(snapshot.review.to_dict(),sort_keys=True).encode()).hexdigest()==frozen
        return result
    monkeypatch.setattr(MobileReviewSession,'statistics',delayed)
    before=s.review.review_revision;parents=copy.deepcopy(core.scene(s,h.current_invocation)['parents'])
    try:
        move_release(ed,0)
        ed.refresh_results();native.pump(root,lambda:started.is_set())
        first_revision=s.review.review_revision
        assert not h._saving,'Statistics must not set the write lock'
        x,y=grab(ed,1)
        assert ed.drag and ed.drag['index']==1,'The next corner is blocked by the statistics worker'
        ed.c.event_generate('<B1-Motion>',x=x+4,y=y+2)
        ed.c.event_generate('<ButtonRelease-1>',x=x+4,y=y+2)
        assert s.review.review_revision==before+2
    finally:release_stats.set()
    wait_current(root,h,ed)
    assert observed[:2]==[first_revision,first_revision],'Statistics mixed two saved revisions'
    assert core.scene(s,h.current_invocation)['parents']==parents
    assert current(h.stats['live_session_metrics_v2'],result_key(s))


def test_eight_consecutive_corners_without_waiting_for_results(gui):
    root,h,ed,s=gui;revision=s.review.review_revision;rows=[]
    # On the full photograph this plate is only a few screen pixels high;
    # separate corner hit areas as an operator does when editing its outline.
    nav.show_geometry(ed,clean.selected(ed)['gt']);ready(root,h,ed)
    parents=copy.deepcopy(core.scene(s,h.current_invocation)['parents'])
    texts={pid:p['gt_text'] for pid,p in stacks.snapshot(s,h.current_invocation)['plates'].items()}
    for number in range(8):
        start=time.perf_counter();x,y=grab(ed,number%4);grab_ms=1000*(time.perf_counter()-start)
        assert ed.drag and ed.drag['index']==number%4
        ed.c.event_generate('<B1-Motion>',x=x+3,y=y+1)
        start=time.perf_counter();ed.c.event_generate('<ButtonRelease-1>',x=x+3,y=y+1)
        rows.append(dict(corner=number%4,grab_ms=grab_ms,release_ms=1000*(time.perf_counter()-start)))
        root.update()
    assert s.review.review_revision==revision+8 and not h._saving
    assert core.scene(s,h.current_invocation)['parents']==parents
    assert {pid:p['gt_text'] for pid,p in stacks.snapshot(s,h.current_invocation)['plates'].items()}==texts
    fresh=MobileReviewSession.open(s.path,sidecar_path=s.sidecar_path)
    assert stacks.snapshot(fresh,h.current_invocation)==stacks.snapshot(s,h.current_invocation)
    wait_current(root,h,ed)
    out=os.environ.get('GTR_RELEASE_OUTPUT')
    if out:(Path(out)/'eight_corners.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


def test_finished_worker_does_not_repaint_or_reload_an_active_corner(gui,monkeypatch):
    root,h,ed,s=gui;release_stats=threading.Event();started=threading.Event()
    original=MobileReviewSession.statistics
    def delayed(snapshot):
        if threading.current_thread() is not threading.main_thread():
            started.set();assert release_stats.wait(8)
        return original(snapshot)
    monkeypatch.setattr(MobileReviewSession,'statistics',delayed)
    monkeypatch.setattr(h,'_render_subject',lambda:pytest.fail('Statistics reloaded the image'))
    try:
        move_release(ed,0);ed.refresh_results();native.pump(root,lambda:started.is_set())
        x,y=grab(ed,1);assert ed.drag
        ed.c.event_generate('<B1-Motion>',x=x+4,y=y+1)
        preview={k:copy.deepcopy(ed.drag[k]) for k in ('oid','index','quad','before')}
        old_stats=h.stats;image=h._image
        view=(h._view_center,h._view_zoom,h._view_pixel_scale)
        release_stats.set()
        native.pump(root,lambda:not ed.in_refresh and getattr(h,'_canvas_statistics_job',None) is None,seconds=15)
        assert {k:ed.drag[k] for k in preview}==preview
        assert h.stats is old_stats and h._image is image
        assert (h._view_center,h._view_zoom,h._view_pixel_scale)==view
        assert h._canvas_statistics_pending is not None
        ed.c.event_generate('<ButtonRelease-1>',x=x+4,y=y+1)
    finally:release_stats.set()
    wait_current(root,h,ed)
    assert h._image is image
    assert h.stats['mt_object_evaluation_v2']==s.statistics()['mt_object_evaluation_v2']


def test_statistics_completion_preserves_GTZ_focus_and_draft(gui,monkeypatch):
    root,h,ed,s=gui;release_stats=threading.Event();started=threading.Event()
    original=MobileReviewSession.statistics
    def delayed(snapshot):
        if threading.current_thread() is not threading.main_thread():
            started.set();assert release_stats.wait(8)
        return original(snapshot)
    monkeypatch.setattr(MobileReviewSession,'statistics',delayed)
    try:
        move_release(ed,0);ed.refresh_results();native.pump(root,lambda:started.is_set())
        ed.start_entry(h._test_oid);entry=ed.entry;assert entry is not None
        ed.entry_var.set('PRIVATE_DRAFT');entry.icursor(4);entry.focus_force();root.update()
        saved=s.sidecar_path.read_bytes();release_stats.set()
        native.pump(root,lambda:not ed.in_refresh and getattr(h,'_canvas_statistics_job',None) is None,seconds=15)
        assert ed.entry is entry and ed.entry_var.get()=='PRIVATE_DRAFT' and entry.index('insert')==4
        assert h.window.focus_get() is entry and s.sidecar_path.read_bytes()==saved
        ed.escape()
    finally:release_stats.set()
    wait_current(root,h,ed)


def test_navigation_does_not_drop_a_drawing_during_read_only_refresh(gui,monkeypatch):
    root,h,ed,s=gui;release_stats=threading.Event();started=threading.Event()
    original=MobileReviewSession.statistics
    def delayed(snapshot):
        if threading.current_thread() is not threading.main_thread():started.set();assert release_stats.wait(8)
        return original(snapshot)
    monkeypatch.setattr(MobileReviewSession,'statistics',delayed)
    try:
        move_release(ed,0);ed.refresh_results();native.pump(root,lambda:started.is_set())
        ed.escape();ed.draw_toggle();assert ed.mode=='draw'
        ed.points=[[20.,20.]];saved=s.sidecar_path.read_bytes()
        assert not ed.prepare_navigation() and ed.points==[[20.,20.]]
        assert s.sidecar_path.read_bytes()==saved
        ed.points=[];ed.mode='select'
    finally:release_stats.set()
    wait_current(root,h,ed)


def test_worker_error_clears_refresh_state_and_keeps_corner_available(gui,monkeypatch):
    root,h,ed,s=gui
    def failed(snapshot):raise OSError('PRIVATE_QA_READ_FAILURE')
    monkeypatch.setattr(MobileReviewSession,'statistics',failed)
    move_release(ed,0);saved=s.sidecar_path.read_bytes();ed.refresh_results()
    native.pump(root,lambda:not ed.in_refresh and getattr(h,'_canvas_statistics_job',None) is None)
    assert not h._saving and 'PRIVATE_QA_READ_FAILURE' in h.status_var.get()
    x,y=grab(ed,1);assert ed.drag and ed.drag['index']==1
    ed.c.event_generate('<ButtonRelease-1>',x=x,y=y)
    assert s.sidecar_path.read_bytes()==saved


def test_repeated_refresh_requests_coalesce_while_a_worker_is_running(gui,monkeypatch):
    root,h,ed,s=gui;release_stats=threading.Event();started=threading.Event();jobs=[]
    original=MobileReviewSession.statistics
    def delayed(snapshot):
        if threading.current_thread() is not threading.main_thread():
            jobs.append(snapshot.review.review_revision);started.set();assert release_stats.wait(8)
        return original(snapshot)
    monkeypatch.setattr(MobileReviewSession,'statistics',delayed)
    try:
        move_release(ed,0);ed.refresh_results();native.pump(root,lambda:started.is_set())
        for _ in range(20):ed.refresh_results()
        assert len(jobs)==1 and not h._saving
    finally:release_stats.set()
    wait_current(root,h,ed)
    assert len(jobs)<=2 and current(h.stats['live_session_metrics_v2'],result_key(s))


def test_close_while_statistics_run_does_not_write_or_touch_destroyed_widgets(gui,monkeypatch):
    root,h,ed,s=gui;release_stats=threading.Event();started=threading.Event()
    original=MobileReviewSession.statistics
    def delayed(snapshot):
        if threading.current_thread() is not threading.main_thread():started.set();assert release_stats.wait(8)
        return original(snapshot)
    monkeypatch.setattr(MobileReviewSession,'statistics',delayed)
    try:
        move_release(ed,0);saved=s.sidecar_path.read_bytes();ed.refresh_results()
        native.pump(root,lambda:started.is_set());h.close()
        assert h._closed and s.sidecar_path.read_bytes()==saved
    finally:release_stats.set()
    h._executor.shutdown(wait=True)
    root.update()


@pytest.mark.parametrize('operation',['edit','undo','redo'])
def test_task_cache_rechecks_only_changed_images_and_equals_uncached_view(gui,monkeypatch,operation):
    from auto_annotation_tool.gui import mobile_canvas_review_tasks as tasks
    from auto_annotation_tool.ranking.mobile_canvas_presentation import build_inventory
    root,h,ed,s=gui;gid=h.current_invocation;ed.inventory()
    if operation in ('undo','redo'):
        move_release(ed,0)
        if operation=='redo':ed.undo()
    cached_gids={g for g,v in ed.task_inventory_cache['scenes'].items() if g!=gid}
    calls=[];original=tasks.for_scene
    def counted(session,key,*a,**k):calls.append(key);return original(session,key,*a,**k)
    monkeypatch.setattr(tasks,'for_scene',counted)
    if operation=='edit':move_release(ed,0)
    else:ed.undo(redo=operation=='redo')
    cached=ed.inventory()
    assert gid in calls and not (cached_gids&set(calls))
    monkeypatch.setattr(tasks,'for_scene',original)
    assert cached==tasks.inventory(s,build_inventory(s))
    wait_current(root,h,ed)


def test_task_cache_detects_changed_projection_even_with_same_annotation(gui):
    from auto_annotation_tool.gui import mobile_canvas_review_tasks as tasks
    from auto_annotation_tool.ranking.mobile_canvas_presentation import build_inventory
    root,h,ed,s=gui;cache={};canonical=build_inventory(s)
    tasks.inventory(s,canonical,cache=cache)
    changed=copy.deepcopy(build_inventory(s));gid=h.current_invocation
    changed['groups'][gid]['scene']['exclusion_reason']='PRIVATE_QA_CHANGED_PROJECTION'
    result=tasks.inventory(s,changed,cache=cache)
    assert result==tasks.inventory(s,changed)
    assert result['groups'][gid]['review_tasks']['count']==0
