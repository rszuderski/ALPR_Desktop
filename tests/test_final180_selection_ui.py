"""Only the five requested F2 selection/UI checks. No TV, run or QR scan."""
from pathlib import Path
from _private_test_data import private_test_path
from types import SimpleNamespace as NS
import json,os,time,traceback,runpy,sys
import tkinter as tk
import pytest
from auto_annotation_tool.gui import final180_tv as tv
from auto_annotation_tool.ranking import photo_reference as photo,final180_timeline as timeline
from auto_annotation_tool.session import SessionManager
from auto_annotation_tool import config

BASE=private_test_path('F6-FINAL180-A12-01')
OLD=BASE/'prepared_20261112T220901Z_2cf67887'
F2=BASE/'prepared_20261113T195755Z_c74eb669'
EVIDENCE=Path(os.environ['F6_SELECTION_EVIDENCE']) if os.environ.get('F6_SELECTION_EVIDENCE') else None

def evidence(name,value):
    if EVIDENCE:(EVIDENCE/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8')

def widgets(window):
    for w in window.winfo_children():yield w;yield from widgets(w)

def choose_button(app):
    return next(w for w in widgets(app.window) if w.winfo_class()=='TButton' and w.cget('text') in ('Wczytaj przygotowany','Wybierz zestaw…','Zapisane pokazy…'))

def pump(root,condition,seconds=15):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        root.update()
        if condition():return
        time.sleep(.01)
    raise AssertionError('Callback did not finish within the bounded UI wait')

@pytest.fixture(scope='module')
def root():
    if not F2.is_dir():pytest.skip('Read-only F2 preparation not available')
    tv.monitor_dpi.enable_windows_dpi_awareness();root=tk.Tk();root.withdraw()
    yield root
    root.destroy()

@pytest.fixture
def gui(root,tmp_path,monkeypatch):
    preferences=SessionManager(tmp_path/'preferences.json');preferences.set('final_presentation','prepared_path',str(OLD))
    monkeypatch.setattr(config,'SESSION',preferences)
    native_monitors=tv.monitor_dpi.final_monitors
    monkeypatch.setattr(tv.monitor_dpi,'final_monitors',lambda:[dict(name='PRIVATE_HOST',x=0,y=0,w=1920,h=1080,verified=True,geometry_source='ISOLATED_UI_TEST')])
    monkeypatch.setattr(tk.Toplevel,'deiconify',lambda self:None)
    monkeypatch.setattr(tk.Toplevel,'lift',lambda self:None)
    monkeypatch.setattr(tk.Toplevel,'focus_set',lambda self:None)
    monkeypatch.setattr(tv.messagebox,'showerror',lambda *a,**kw:pytest.fail(str(a)))
    # Actual Desktop browser action; only its owning window is a private host.
    from auto_annotation_tool.gui.z4_mobile_report_browser import MobileReportBrowser
    owner=NS(window=root);MobileReportBrowser.open_final_presentation(owner);app=owner._final_presentation;app.window.withdraw()
    errors=[];root.report_callback_exception=lambda *args:errors.append(''.join(traceback.format_exception(*args)))
    def hidden_viewer(screen):
        app.viewer=tk.Toplevel(root);app.viewer.withdraw();app.canvas=tk.Canvas(app.viewer,highlightthickness=0)
    monkeypatch.setattr(app,'_make_viewer',hidden_viewer)
    monkeypatch.setattr(tv,'PresentationRun',lambda *a,**kw:pytest.fail('These checks must not create a run'))
    technical=[];monkeypatch.setattr(timeline,'register_calibration',lambda path,identity:technical.append((str(path),identity)))
    draws=[];draw=app.draw
    def capture(image):draws.append(image.copy());draw(image)
    monkeypatch.setattr(app,'draw',capture)
    extra=[]
    yield NS(app=app,root=root,preferences=preferences,errors=errors,draws=draws,technical=technical,extra=extra,tmp=tmp_path,native_monitors=native_monitors,folder_requests=[])
    for owner in [*extra,app]:
        for token in root.tk.call('after','info'):
            command=root.tk.call('after','info',token)[0]
            if command in (owner.window._tclCommands or []):owner.window.after_cancel(token)
        owner._close()
    root.update_idletasks()

def select(gui,path,monkeypatch):
    def folder(**kw):gui.folder_requests.append(kw);return str(path)
    monkeypatch.setattr(tv.filedialog,'askdirectory',folder)
    choose_button(gui.app).invoke()

def test_t1_native_choose_switches_complete_state_to_existing_f2(gui,monkeypatch):
    app=gui.app;assert app.prepared==OLD
    select(gui,F2,monkeypatch);pump(gui.root,lambda:app.prepared==F2 and not getattr(app,'_selection_loading',False))
    assert app.manifest['layout']['layout_version']=='final_f2_v1'
    assert app.gt_preflight['counts']['GT_COMPLETE']==176 and Path(app.gt_preflight['reference_path']).parent==F2
    assert 'final_f2_v1' in app.plan.get() and 'FINAL176 · F2' in app.set_info.get() and F2.name in app.selected_folder.get()
    assert str(F2/'reference.json') in app.details.get('1.0','end')
    assert gui.preferences.get('final_presentation','prepared_path')==str(F2) and not gui.errors
    assert gui.folder_requests[0]['initialdir']==str(OLD)
    def cancel(**kw):gui.folder_requests.append(kw);return ''
    monkeypatch.setattr(tv.filedialog,'askdirectory',cancel);choose_button(app).invoke()
    assert gui.folder_requests[-1]['initialdir']==str(F2) and app.prepared==F2
    evidence('T1',dict(status='PASS',prepared=str(app.prepared),layout=app.manifest['layout']['layout_version'],gt=app.gt_preflight['counts'],footer=app.plan.get()))

def test_t2_open_old_board_is_closed_before_f2_and_renderer_follows(gui,monkeypatch):
    app=gui.app;app.show_calibration();old_window=app.viewer
    assert app.calibration_preview and app.calibration_preview['identity']['token'].startswith('F6v1|')
    select(gui,F2,monkeypatch);assert app.viewer is None and not old_window.winfo_exists()
    pump(gui.root,lambda:app.prepared==F2 and not getattr(app,'_selection_loading',False))
    app.calibration_btn.invoke();assert app.calibration_preview and app.run is None
    identity=app.calibration_preview['identity'];assert identity['photo_rect']==[210,100,1710,980]
    assert identity['token'].startswith('F2') and len(identity['token'])==20 and app.manifest['layout']['qr_positions']==['LEFT','RIGHT']
    assert app.manifest['layout']['qr_rects']==[[6,441,204,639],[1716,441,1914,639]]
    assert not(F2/'runs').exists() and not gui.errors
    if EVIDENCE:gui.draws[-1].save(EVIDENCE/'HOST_F2_TEST_ONLY.png')
    evidence('T2',dict(status='PASS_HOST_ONLY',old_closed=True,photo_rect=identity['photo_rect'],qr_positions=['LEFT','RIGHT'],no_run=True,no_tv=True))

def test_t3_bad_folder_manifest_and_callback_error_do_not_poison_next_choice(gui,monkeypatch,caplog):
    app=gui.app;bad=gui.tmp/'not_a_prepared';bad.mkdir()
    select(gui,bad,monkeypatch)
    pump(gui.root,lambda:bool(gui.errors) or 'Nie wczytano' in app.status.get())
    if gui.errors:
        select(gui,F2,monkeypatch)
        until=time.monotonic()+.2
        while time.monotonic()<until:gui.root.update();time.sleep(.01)
        evidence('T3_before_failure',dict(active=str(app.prepared),callback_errors=gui.errors,pending_results=app.queue.qsize()))
    assert not gui.errors,gui.errors
    assert app.prepared==OLD and str(app.start_btn.cget('state'))=='disabled' and str(app.calibration_btn.cget('state'))=='disabled'
    assert str(bad) in app.status.get()
    rejected=[r for r in caplog.records if r.message.startswith('FINAL selection rejected:')]
    assert rejected and all(r.exc_info is None for r in rejected)
    # A parent directory is never silently converted into a guessed preparation.
    child=bad/'prepared_private_invalid';child.mkdir();(child/'manifest.json').write_text('{"schema":"BROKEN"}','utf-8')
    select(gui,bad,monkeypatch);pump(gui.root,lambda:not app._selection_loading)
    assert app.prepared==OLD and 'podfolderów' in app.status.get()
    (bad/'manifest.json').write_text('{"schema":"BROKEN"}','utf-8')
    select(gui,bad,monkeypatch);pump(gui.root,lambda:not app._selection_loading)
    assert app.prepared==OLD and 'Nie wczytano' in app.status.get()
    assert any(r.exc_info for r in caplog.records if r.message=='FINAL background operation failed')
    def faulty_callback(_):raise ValueError('TEST_CALLBACK_FAILURE')
    app.queue.put((faulty_callback,None,None));pump(gui.root,lambda:'TEST_CALLBACK_FAILURE' in app.status.get())
    select(gui,F2,monkeypatch);pump(gui.root,lambda:app.prepared==F2 and not app._selection_loading)
    assert not gui.errors and not app._selection_error
    evidence('T3',dict(status='PASS',prior_retained_on_failure=True,start_and_board_blocked=True,queue_recovers=True,folder_error_has_path_and_no_traceback=True,parent_does_not_guess=True,unexpected_error_keeps_traceback=True))

def test_t4_desktop_and_real_wrapper_reopen_last_successful_f2(gui,monkeypatch):
    select(gui,F2,monkeypatch);pump(gui.root,lambda:gui.app.prepared==F2 and not gui.app._selection_loading)
    monkeypatch.setattr(config,'SESSION',SessionManager(gui.preferences.session_file))
    from auto_annotation_tool.gui.z4_mobile_report_browser import MobileReportBrowser
    owner=NS(window=gui.root);MobileReportBrowser.open_final_presentation(owner);desktop=owner._final_presentation;gui.extra.append(desktop);desktop.window.withdraw()
    original=tv.make_app;created=[]
    def create(legacy,args,parent=None):
        app=original(legacy,args,parent=gui.root);app.window.withdraw();created.append(app);return app
    monkeypatch.setattr(tv,'make_app',create);monkeypatch.setattr(tk.Misc,'mainloop',lambda self,*a,**kw:None)
    tv.main(tv.accepted_arguments());gui.extra.extend(created)
    assert desktop.prepared==created[0].prepared==F2
    assert desktop.manifest['layout']==created[0].manifest['layout'] and not gui.errors
    evidence('T4',dict(status='PASS_NATIVE_PANELS',desktop=str(desktop.prepared),wrapper=str(created[0].prepared),no_tv=True))

def test_t5_simple_controls_advanced_hidden_and_base_folder_cannot_select_plan(gui,monkeypatch):
    app=gui.app;select(gui,F2,monkeypatch);pump(gui.root,lambda:app.prepared==F2 and not app._selection_loading)
    assert not app.advanced_visible.get() and app.advanced.winfo_manager()==''
    for button in (app.choose_set_btn,app.calibration_btn,app.start_btn,app.stop_btn,app.import_btn):assert button.cget('command')
    assert str(app.start_btn.cget('state'))=='normal' and str(app.stop_btn.cget('state'))=='disabled'
    calls=[];monkeypatch.setattr(tv.filedialog,'askopenfilename',lambda **kw:calls.append(kw) or '')
    app.import_btn.invoke();assert len(calls)==1
    app.advanced_btn.invoke();assert app.advanced_visible.get() and app.advanced.winfo_manager()=='grid'
    output=app.paths['output'].get();monkeypatch.setattr(tv.filedialog,'askdirectory',lambda **kw:str(OLD));app.browse('output')
    assert app.prepared==F2 and app.paths['output'].get()==output and 'Wybierz zestaw' in app.status.get()
    app.advanced_btn.invoke();assert not app.advanced_visible.get()
    app.status.set('HOST_ONLY — kontrola panelu; TV nieużywany')
    # Only the control panel is briefly visible on the primary monitor.
    if EVIDENCE:
        from PIL import ImageGrab
        actual=gui.native_monitors();monkeypatch.setattr(tv.monitor_dpi,'final_monitors',lambda:actual)
        app._selected_device=actual[-1]['name'];app.refresh_monitors()
        api=tv.monitor_dpi.display_api()
        with api.pixels():primary=next(m for m in api.monitors() if m['primary'])
        assert primary['rect'][:2]==[0,0]
        app.window.geometry('1000x550+20+20');app.window.title('HOST_ONLY — panel F2; bez okna na TV')
        app.window.tk.call('wm','deiconify',app.window);app.window.attributes('-topmost',True);app.window.attributes('-alpha',1);gui.root.update();time.sleep(.7);gui.root.update()
        with api.pixels():state=api.window_state(api.hwnd(app.window))
        l,t,r,b=state['rect'];assert 0<=l<r<=primary['rect'][2] and 0<=t<b<=primary['rect'][3]
        ImageGrab.grab(bbox=(l,t,r,b),all_screens=True).save(EVIDENCE/'HOST_simple_panel.png');app.window.withdraw()
    assert not gui.errors
    evidence('T5',dict(status='PASS_NATIVE_CONTROLS',advanced_collapsed=True,base_does_not_activate=True,start_not_executed=True,no_tv=True))
