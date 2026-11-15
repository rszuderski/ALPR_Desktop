"""FINAL pixels must not inherit the main Tk process's virtual monitor rects."""
from contextlib import contextmanager
from copy import deepcopy
import pytest
import windows_dpi as dpi


class DisplayAPI:
    def __init__(self,size=(1920,1080),scale=100,system_scale=125,position=(1920,0)):
        self.context=1;self.restores=0;self.size=size;self.scale=scale;self.system_scale=system_scale;self.position=position
        self.signal=size;self.connected=True;self.error=None;self.extra=[]

    def awareness(self):return self.context

    @contextmanager
    def pixels(self):
        old=self.context;self.context=2
        try:yield
        finally:self.context=old;self.restores+=1

    def monitors(self):
        if not self.connected:return []
        x,y=self.position;w,h=self.size
        factor=1 if self.context==2 else self.system_scale/self.scale
        return [dict(name=r'\\.\DISPLAY4',hmonitor=42,rect=[round(v*factor) for v in (x,y,x+w,y+h)],primary=False),*deepcopy(self.extra)]

    def mode(self,name):
        x,y=self.position;w,h=self.size
        return dict(position=[x,y],size=[w,h],orientation=0,frequency=60)

    def paths(self):
        if self.error:raise OSError(self.error)
        x,y=self.position
        return [dict(name=r'\\.\DISPLAY4',source_size=list(self.size),source_position=[x,y],signal_size=list(self.signal),rotation=1,
                     target_available=True,adapter=[1,0],source_id=0,target_id=4,device_path='TEST_MONITOR_4')]

    def monitor_dpi(self,handle):return dict(effective=[round(96*self.scale/100)]*2,scale_percent=self.scale)


def test_reproduces_legacy_2400_but_final_uses_independent_1920():
    api=DisplayAPI();old=api.monitors()[0]['rect'];assert old==[2400,0,4800,1350]
    monitor=dpi.final_monitors(api=api)[0]
    assert (monitor['x'],monitor['y'],monitor['w'],monitor['h'])==(1920,0,1920,1080)
    assert monitor['window_manager_rect']==old and monitor['signal_size']==[1920,1080]
    assert monitor['verified'] and api.context==1 and api.restores==1
    dpi.validate_final_monitor(monitor,dict(width=1920,height=1080))


@pytest.mark.parametrize('scale',[100,125,150])
@pytest.mark.parametrize('position',[(0,0),(1920,0),(-1920,0),(-800,-1080),(300,1080)])
def test_dpi_and_negative_positions_are_physical_not_tk_offsets(scale,position):
    api=DisplayAPI(scale=scale,position=position);m=dpi.final_monitors(api=api)[0]
    assert [m[k] for k in ('x','y','w','h')]==[*position,1920,1080]
    assert m['dpi']['scale_percent']==scale and api.context==1


@pytest.mark.parametrize('size',[(1280,720),(1920,1080),(2400,1350),(2560,1440)])
def test_actual_resolution_never_divided_by_scale(size):
    m=dpi.final_monitors(api=DisplayAPI(size=size))[0]
    dpi.validate_final_monitor(m,dict(zip(('width','height'),size)))
    if size!=(1920,1080):
        with pytest.raises(ValueError,match='Plan: 1920×1080'):
            dpi.validate_final_monitor(m,dict(width=1920,height=1080))


@pytest.mark.parametrize('fault',['signal','missing_path','api_error','source_rect','clone'])
def test_unproven_or_conflicting_physical_modes_fail_closed(fault):
    api=DisplayAPI()
    if fault=='signal':api.signal=(2400,1350)
    if fault=='missing_path':api.paths=lambda:[]
    if fault=='api_error':api.error='QueryDisplayConfig failed'
    if fault=='source_rect':api.mode=lambda name:dict(position=[0,0],size=[1920,1080])
    if fault=='clone':
        original=api.paths;api.paths=lambda:original()*2
    m=dpi.final_monitors(api=api)[0]
    assert not m['verified'] and m['verification_error'] and api.context==1
    with pytest.raises(ValueError,match='NO_GO'):dpi.validate_final_monitor(m,dict(width=1920,height=1080))


def test_disconnection_and_mode_changes_are_requeried():
    api=DisplayAPI();first=dpi.final_monitors(api=api)[0]
    api.connected=False;assert dpi.final_monitors(api=api)==[]
    api.connected=True;api.size=api.signal=(1280,720)
    current=dpi.final_monitors(api=api)[0]
    assert first['w']==1920 and current['w']==1280
    with pytest.raises(ValueError):dpi.validate_final_monitor(current,dict(width=1920,height=1080))


def test_context_restored_even_if_monitor_enumeration_fails():
    api=DisplayAPI();original=api.monitors
    def failing():
        if api.context==2:raise OSError('unplugged')
        return original()
    api.monitors=failing
    with pytest.raises(OSError):dpi.final_monitors(api=api)
    assert api.context==1


def test_mixed_monitors_keep_independent_dpi_and_positions():
    api=DisplayAPI(scale=125,system_scale=100);single=api.monitors;path=api.paths;mode=api.mode
    def monitors():
        first=dict(name=r'\\.\DISPLAY1',hmonitor=1,rect=[0,0,1920,1080],primary=True)
        return [first,*single()]
    api.monitors=monitors
    api.paths=lambda:[dict(path()[0],name=r'\\.\DISPLAY1',source_position=[0,0],target_id=1,device_path='TEST_MONITOR_1'),*path()]
    api.mode=lambda name:dict(mode(name),position=[0,0]) if name.endswith('DISPLAY1') else mode(name)
    api.monitor_dpi=lambda handle:dict(effective=[96,96] if handle==1 else [120,120],scale_percent=100 if handle==1 else 125)
    first,second=dpi.final_monitors(api=api)
    assert first['verified'] and second['verified'] and (first['x'],second['x'])==(0,1920)
    assert (first['dpi']['scale_percent'],second['dpi']['scale_percent'])==(100,125)
    assert first['window_manager_rect']==first['physical_rect'] and second['window_manager_rect']!=second['physical_rect']


@pytest.mark.parametrize('fault',[None,'window_size','canvas_size','awareness','wrong_monitor','wrong_origin'])
def test_window_verification_uses_native_physical_pixels_and_rejects_misalignment(fault):
    from types import SimpleNamespace as NS
    api=DisplayAPI(position=(-1920,-200));monitor=dpi.final_monitors(api=api)[0];placed=[]
    api.hwnd=lambda widget:1;api.position=lambda hwnd,rect:placed.append(rect)
    state=dict(rect=[-1920,-200,0,880],client_size=[1920,1080],client_origin=[-1920,-200],hmonitor=42,awareness=2,dpi=96)
    def read(hwnd):
        row=deepcopy(state)
        if fault=='window_size' and hwnd==1:row['rect'][-1]=1080
        if fault=='canvas_size' and hwnd==2:row['client_size']=[2400,1350]
        if fault=='awareness':row['awareness']=1
        if fault=='wrong_monitor':row['hmonitor']=1
        if fault=='wrong_origin':row['client_origin']=[0,0]
        return row
    api.window_state=read
    window=NS(update_idletasks=lambda:None,tk=NS(dooneevent=lambda flags:0))
    canvas=NS(winfo_id=lambda:2,winfo_width=lambda:1920,winfo_height=lambda:1080)
    if fault:
        with pytest.raises(ValueError,match='NO_GO'):dpi.final_window_geometry(window,canvas,monitor,position=True,api=api)
    else:assert dpi.final_window_geometry(window,canvas,monitor,position=True,api=api)['canvas_verified']
    assert placed==[state['rect']] and api.context==1


def test_native_tk_hotplug_identity_mismatch_and_failed_window_create_no_run(tmp_path,monkeypatch):
    import os,sys,subprocess
    from pathlib import Path
    if os.environ.get('F6_MONITOR_GUI')!='1':
        child=subprocess.run([sys.executable,'-m','pytest',str(Path(__file__).resolve())+'::test_native_tk_hotplug_identity_mismatch_and_failed_window_create_no_run','-q'],
            env=dict(os.environ,F6_MONITOR_GUI='1'),capture_output=True,text=True,timeout=90)
        assert child.returncode==0,child.stdout+child.stderr;return
    import tkinter as tk
    from types import SimpleNamespace as NS
    from auto_annotation_tool.gui import final180_tv as tv
    from auto_annotation_tool.ranking import final180_reference as ref,photo_reference as photo
    from test_final180_desktop import frozen
    b=dpi.final_monitors(api=DisplayAPI())[0];a=dict(b,name=r'\\.\DISPLAY1',x=0,primary=True,hmonitor=1)
    available=[a,b];monkeypatch.setattr(dpi,'final_monitors',lambda:deepcopy(available))
    root=tk.Tk();root.withdraw();app=tv.make_app(tv.legacy_module(tv.LEGACY_DEFAULT),NS(xml=None,positives=None,negatives=None,output=tmp_path/'plans',dwell=5,negatives_declared=True,unreadable_ids=[]),parent=root);app.window.withdraw()
    prepared=frozen(tmp_path/'fixture',dwell_seconds=5);before={p:photo.digest(p) for p in prepared.rglob('*') if p.is_file()}
    calls=[]
    def hidden(screen):
        calls.append(screen);app.viewer=tk.Toplevel(root);app.viewer.withdraw();app.canvas=tk.Canvas(app.viewer)
    monkeypatch.setattr(app,'_make_viewer',hidden)
    monkeypatch.setattr(tv,'PresentationRun',lambda *a,**k:pytest.fail('No run when the monitor/window is invalid'))
    try:
        app.loaded(prepared);assert app._selected_device==b['name']
        available.reverse();app.refresh_monitors();assert app._select_monitor()['name']==b['name']
        app.show_calibration();assert app.mode=='CALIBRATION';app.close_calibration()
        available[:]=[a];app.refresh_monitors();app.show_calibration();app.begin_countdown()
        assert app.viewer is None and 'odłączony' in app.status.get() and str(app.start_btn.cget('state'))=='disabled'
        assert app._selected_device==b['name']  # Never fall back silently to the primary.
        available[:]=[a,dict(b,w=2400,h=1350,signal_size=[2400,1350])];app.refresh_monitors()
        app.show_calibration();assert 'Plan: 1920×1080' in app.status.get() and '2400×1350' in app.status.get()
        app.begin_countdown();assert not(prepared/'runs').exists()
        available[:]=[a,b];app.refresh_monitors()
        def failed(screen):raise ValueError('NO_GO: test post-position failure')
        monkeypatch.setattr(app,'_make_viewer',failed);app.begin_countdown()
        assert 'post-position' in app.status.get() and app.run is None and not(prepared/'runs').exists()
        assert all(photo.digest(p)==sha for p,sha in before.items())
    finally:app._close();root.destroy()


def test_native_primary_hidden_window_is_physical_and_restores_main_context(tmp_path,monkeypatch):
    import os,sys,subprocess
    from pathlib import Path
    if sys.platform!='win32':pytest.skip('Native Windows check')
    if os.environ.get('F6_DPI_NATIVE')!='1':
        child=subprocess.run([sys.executable,'-m','pytest',str(Path(__file__).resolve())+'::test_native_primary_hidden_window_is_physical_and_restores_main_context','-q'],
            env=dict(os.environ,F6_DPI_NATIVE='1'),capture_output=True,text=True,timeout=90)
        assert child.returncode==0,child.stdout+child.stderr;return
    import tkinter as tk
    from types import SimpleNamespace as NS
    from auto_annotation_tool.gui import final180_tv as tv
    from auto_annotation_tool.ranking import final180_reference as ref,photo_reference as photo
    from test_final180_desktop import material
    dpi.enable_windows_dpi_awareness();api=dpi.display_api();before_context=api.awareness()
    monitors=dpi.final_monitors();primary=next(m for m in monitors if m['primary']);dpi.validate_final_monitor(primary)
    # No native window, including a hidden one, may be positioned on the TV.
    monkeypatch.setattr(dpi,'final_monitors',lambda:[primary])
    original_position=api.position
    def primary_only(hwnd,rect):
        assert rect==primary['physical_rect'];return original_position(hwnd,rect)
    monkeypatch.setattr(api,'position',primary_only)
    states=[];native_state=api.window_state
    def record_state(hwnd):
        value=native_state(hwnd);states.append(value);return value
    monkeypatch.setattr(api,'window_state',record_state)
    deiconify=tk.Toplevel.deiconify
    def transparent_primary(self):
        self.attributes('-alpha',0);deiconify(self)
    monkeypatch.setattr(tk.Toplevel,'deiconify',transparent_primary)
    monkeypatch.setattr(tk.Toplevel,'lift',lambda self:None)
    monkeypatch.setattr(tk.Toplevel,'focus_set',lambda self:None)
    root=tk.Tk();root.withdraw();errors=[];root.report_callback_exception=lambda *args:errors.append(args)
    legacy=tv.legacy_module(tv.LEGACY_DEFAULT)
    args=NS(xml=None,positives=None,negatives=None,output=tmp_path/'plans',dwell=5,negatives_declared=True,unreadable_ids=[])
    app=tv.make_app(legacy,args,parent=root);app.window.withdraw()
    prepared=ref.freeze(material(tmp_path/'sources'),tmp_path/'plans',screen=(primary['w'],primary['h']),dwell_seconds=5,technical=True,layout_version='final_f2_v1',reference_aware=True)
    protected={p:photo.digest(p) for p in prepared.rglob('*') if p.is_file()}
    try:
        app.loaded(prepared);app.show_calibration()
        assert app.mode=='CALIBRATION',(app.status.get(),states)
        assert api.awareness()==before_context
        proof=app._viewer_geometry;assert proof['window']['awareness']==proof['canvas']['awareness']==2
        assert proof['window']['rect']==primary['physical_rect']
        assert proof['tk_canvas_size']==[primary['w'],primary['h']]
        assert app.photos[0].width()==primary['w'] and app.photos[0].height()==primary['h']
        assert float(app.viewer.attributes('-alpha'))==0 and not(prepared/'runs').exists()
        app.close_calibration();assert app.viewer is None and api.awareness()==before_context
        # The unextended F5A factory has no hook and retains its DPI/keyboard path.
        ordinary=legacy.TvApp.__new__(legacy.TvApp);ordinary.tk=tk;ordinary.window=root
        ordinary.abort=lambda:None;ordinary.begin_countdown=lambda:None
        assert not hasattr(ordinary,'_position_viewer')
        ordinary._make_viewer(dict(x=primary['x'],y=primary['y'],w=320,h=200))
        try:
            root.update()
            assert ordinary.viewer.bind('<Escape>') and ordinary.viewer.bind('<space>')
            assert [ordinary.canvas.winfo_width(),ordinary.canvas.winfo_height()]==[320,200]
            with api.pixels():ordinary_proof=api.window_state(api.hwnd(ordinary.viewer))
            assert ordinary_proof['awareness']==before_context and ordinary_proof['hmonitor']==primary['hmonitor']
        finally:ordinary._dispose_viewer()
        target=os.environ.get('F6_MONITOR_NATIVE_PROOF')
        if target:
            import json
            Path(target).write_text(json.dumps(dict(proof=proof,monitor=primary,thread_before=before_context,thread_after=api.awareness(),
                tk_scaling=float(root.tk.call('tk','scaling')),fpixels_1i=root.winfo_fpixels('1i'),transparent=True,no_tv_window=True,legacy_f5a=ordinary_proof),indent=2),'utf-8')
        assert all(photo.digest(p)==sha for p,sha in protected.items()) and not errors
    finally:app._close();root.destroy()
