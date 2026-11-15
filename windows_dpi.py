"""Select native pixel rendering before Windows or Tk creates any app window."""
from __future__ import annotations

import ctypes
import sys
from contextlib import contextmanager


def enable_windows_dpi_awareness() -> str:
    """Avoid Windows bitmap stretching; preserve an already configured process.

    Tk 8.6 uses a system DPI for its interpreter. System awareness matches that
    model without claiming per-monitor font/layout support. This changes only
    the calling process and must run before the dependency bootstrap's Tk UI.
    """
    if sys.platform != "win32":
        return "not_windows"
    try:
        result = ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    else:
        if result == 0:
            return "system"
        if result & 0xFFFFFFFF == 0x80070005:  # Already fixed by a manifest/API.
            return "already_configured"
    try:
        if ctypes.windll.user32.SetProcessDPIAware():
            return "system_legacy"
    except (AttributeError, OSError):
        pass
    return "unavailable"


class Win32Display:
    """Read active CCD modes; use scoped physical pixels, never change displays.

    The process-wide startup policy above stays unchanged. FINAL's top-level
    raster window alone is created per-monitor aware (Windows 10 mixed mode).
    """
    def __init__(self):
        from ctypes import wintypes as w
        c=ctypes
        class LUID(c.Structure):
            _fields_=[('low',w.DWORD),('high',w.LONG)]
        class Rational(c.Structure):
            _fields_=[('numerator',w.UINT),('denominator',w.UINT)]
        class Region(c.Structure):
            _fields_=[('width',w.UINT),('height',w.UINT)]
        class Signal(c.Structure):
            _fields_=[('pixelRate',c.c_uint64),('hSync',Rational),('vSync',Rational),('active',Region),('total',Region),('standard',w.UINT),('scanline',w.UINT)]
        class Source(c.Structure):
            _fields_=[('width',w.UINT),('height',w.UINT),('format',w.UINT),('position',w.POINT)]
        class ModeValue(c.Union):
            _fields_=[('target',Signal),('source',Source)]
        class Mode(c.Structure):
            _fields_=[('type',w.UINT),('id',w.UINT),('adapter',LUID),('value',ModeValue)]
        class PathSource(c.Structure):
            _fields_=[('adapter',LUID),('id',w.UINT),('mode',w.UINT),('flags',w.UINT)]
        class PathTarget(c.Structure):
            _fields_=[('adapter',LUID),('id',w.UINT),('mode',w.UINT),('technology',w.UINT),('rotation',w.UINT),('scaling',w.UINT),('refresh',Rational),('scanline',w.UINT),('available',w.BOOL),('flags',w.UINT)]
        class DisplayPath(c.Structure):
            _fields_=[('source',PathSource),('target',PathTarget),('flags',w.UINT)]
        class Header(c.Structure):
            _fields_=[('type',w.UINT),('size',w.UINT),('adapter',LUID),('id',w.UINT)]
        class SourceName(c.Structure):
            _fields_=[('header',Header),('name',w.WCHAR*32)]
        class TargetName(c.Structure):
            _fields_=[('header',Header),('flags',w.UINT),('technology',w.UINT),('manufacturer',w.WORD),('product',w.WORD),('connector',w.UINT),('friendly',w.WCHAR*64),('path',w.WCHAR*128)]
        class MonitorInfo(c.Structure):
            _fields_=[('size',w.DWORD),('rect',w.RECT),('work',w.RECT),('flags',w.DWORD),('name',w.WCHAR*32)]
        class DevMode(c.Structure):
            _fields_=[('name',w.WCHAR*32),('spec',w.WORD),('driver',w.WORD),('size',w.WORD),('extra',w.WORD),('fields',w.DWORD),
                ('x',w.LONG),('y',w.LONG),('orientation',w.DWORD),('fixed',w.DWORD),('color',c.c_short),('duplex',c.c_short),('yres',c.c_short),('tt',c.c_short),('collate',c.c_short),
                ('form',w.WCHAR*32),('logpixels',w.WORD),('bits',w.DWORD),('width',w.DWORD),('height',w.DWORD),('flags',w.DWORD),('frequency',w.DWORD),
                ('icmmethod',w.DWORD),('icmintent',w.DWORD),('media',w.DWORD),('dither',w.DWORD),('reserved1',w.DWORD),('reserved2',w.DWORD),('panwidth',w.DWORD),('panheight',w.DWORD)]
        assert (c.sizeof(DevMode),c.sizeof(Mode),c.sizeof(DisplayPath))==(220,64,72)
        self.DevMode,self.MonitorInfo,self.Mode,self.DisplayPath=DevMode,MonitorInfo,Mode,DisplayPath
        self.SourceName,self.TargetName=SourceName,TargetName
        self.callback=c.WINFUNCTYPE(w.BOOL,w.HMONITOR,w.HDC,c.POINTER(w.RECT),w.LPARAM)
        # Private function prototypes: the legacy viewer also binds user32 via
        # ctypes.windll, with its own RECT/callback types.
        self.u=c.WinDLL('user32',use_last_error=True);self.s=c.WinDLL('shcore',use_last_error=True)
        signatures={
            'SetThreadDpiAwarenessContext':([c.c_void_p],c.c_void_p),
            'GetThreadDpiAwarenessContext':([],c.c_void_p),
            'GetAwarenessFromDpiAwarenessContext':([c.c_void_p],c.c_int),
            'GetWindowDpiAwarenessContext':([w.HWND],c.c_void_p),
            'GetDpiForWindow':([w.HWND],w.UINT),
            'EnumDisplayMonitors':([w.HDC,c.POINTER(w.RECT),self.callback,w.LPARAM],w.BOOL),
            'GetMonitorInfoW':([w.HMONITOR,c.c_void_p],w.BOOL),
            'EnumDisplaySettingsExW':([w.LPCWSTR,w.DWORD,c.POINTER(DevMode),w.DWORD],w.BOOL),
            'GetDisplayConfigBufferSizes':([w.UINT,c.POINTER(w.UINT),c.POINTER(w.UINT)],w.LONG),
            'QueryDisplayConfig':([w.UINT,c.POINTER(w.UINT),c.POINTER(DisplayPath),c.POINTER(w.UINT),c.POINTER(Mode),c.c_void_p],w.LONG),
            'DisplayConfigGetDeviceInfo':([c.c_void_p],w.LONG),
            'GetAncestor':([w.HWND,w.UINT],w.HWND),
            'SetWindowPos':([w.HWND,w.HWND,c.c_int,c.c_int,c.c_int,c.c_int,w.UINT],w.BOOL),
            'GetWindowRect':([w.HWND,c.POINTER(w.RECT)],w.BOOL),
            'GetClientRect':([w.HWND,c.POINTER(w.RECT)],w.BOOL),
            'ClientToScreen':([w.HWND,c.POINTER(w.POINT)],w.BOOL),
            'MonitorFromWindow':([w.HWND,w.DWORD],w.HMONITOR),
        }
        for name,(args,result) in signatures.items():
            fn=getattr(self.u,name);fn.argtypes=args;fn.restype=result
        self.s.GetDpiForMonitor.argtypes=[w.HMONITOR,c.c_int,c.POINTER(w.UINT),c.POINTER(w.UINT)];self.s.GetDpiForMonitor.restype=w.LONG
        self.s.GetScaleFactorForMonitor.argtypes=[w.HMONITOR,c.POINTER(c.c_int)];self.s.GetScaleFactorForMonitor.restype=w.LONG

    def awareness(self):
        return self.u.GetAwarenessFromDpiAwarenessContext(self.u.GetThreadDpiAwarenessContext())

    @contextmanager
    def pixels(self):
        previous=self.u.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
        if not previous:raise OSError('Nie można potwierdzić kontekstu per-monitor DPI.')
        try:yield
        finally:
            if not self.u.SetThreadDpiAwarenessContext(previous):raise OSError('Nie przywrócono kontekstu DPI wątku.')

    def monitors(self):
        found=[];errors=[]
        def visit(handle,hdc,rect,param):
            info=self.MonitorInfo();info.size=ctypes.sizeof(info)
            if not self.u.GetMonitorInfoW(handle,ctypes.byref(info)):errors.append('GetMonitorInfoW failed');return True
            r=info.rect
            found.append(dict(name=info.name,hmonitor=int(handle),rect=[r.left,r.top,r.right,r.bottom],primary=bool(info.flags&1)))
            return True
        callback=self.callback(visit)
        if not self.u.EnumDisplayMonitors(None,None,callback,0) or errors:raise OSError('EnumDisplayMonitors/GetMonitorInfoW failed')
        return found

    def mode(self,name):
        d=self.DevMode();d.size=ctypes.sizeof(d)
        if not self.u.EnumDisplaySettingsExW(name,0xffffffff,ctypes.byref(d),0):raise OSError('EnumDisplaySettingsExW failed: '+name)
        return dict(position=[d.x,d.y],size=[d.width,d.height],orientation=d.orientation,frequency=d.frequency)

    def paths(self):
        from ctypes import wintypes as w
        c=ctypes
        for _ in range(3):  # Topology may change between the two calls.
            np,nm=w.UINT(),w.UINT()
            code=self.u.GetDisplayConfigBufferSizes(2,c.byref(np),c.byref(nm))
            if code:raise OSError(f'GetDisplayConfigBufferSizes: {code}')
            paths=(self.DisplayPath*np.value)();modes=(self.Mode*nm.value)()
            code=self.u.QueryDisplayConfig(2,c.byref(np),paths,c.byref(nm),modes,None)
            if code==122:continue
            if code:raise OSError(f'QueryDisplayConfig: {code}')
            result=[]
            for path in paths[:np.value]:
                source,target=path.source,path.target
                if source.mode>=nm.value or target.mode>=nm.value:raise OSError('CCD active mode index missing')
                sm,tm=modes[source.mode],modes[target.mode]
                if sm.type!=1 or tm.type!=2 or sm.id!=source.id or tm.id!=target.id:raise OSError('CCD mode identity mismatch')
                for mode,record in [(sm,source),(tm,target)]:
                    if (mode.adapter.low,mode.adapter.high)!=(record.adapter.low,record.adapter.high):raise OSError('CCD adapter mismatch')
                name=self.SourceName();name.header.type=1;name.header.size=c.sizeof(name);name.header.adapter=source.adapter;name.header.id=source.id
                code=self.u.DisplayConfigGetDeviceInfo(c.byref(name))
                if code:raise OSError(f'CCD source name: {code}')
                monitor=self.TargetName();monitor.header.type=2;monitor.header.size=c.sizeof(monitor);monitor.header.adapter=target.adapter;monitor.header.id=target.id
                code=self.u.DisplayConfigGetDeviceInfo(c.byref(monitor))
                if code:raise OSError(f'CCD target name: {code}')
                src=sm.value.source;sig=tm.value.target
                result.append(dict(name=name.name,source_size=[src.width,src.height],source_position=[src.position.x,src.position.y],
                    signal_size=[sig.active.width,sig.active.height],signal_total=[sig.total.width,sig.total.height],rotation=target.rotation,
                    refresh=[target.refresh.numerator,target.refresh.denominator],target_available=bool(target.available),
                    adapter=[source.adapter.low,source.adapter.high],source_id=source.id,target_id=target.id,device_path=monitor.path,friendly_name=monitor.friendly))
            return result
        raise OSError('QueryDisplayConfig: topology kept changing')

    def monitor_dpi(self,handle):
        from ctypes import wintypes as w
        x,y=w.UINT(),w.UINT();scale=ctypes.c_int()
        d=self.s.GetDpiForMonitor(handle,0,ctypes.byref(x),ctypes.byref(y))
        s=self.s.GetScaleFactorForMonitor(handle,ctypes.byref(scale))
        return dict(effective=[x.value,y.value] if d==0 else None,scale_percent=scale.value if s==0 else None)

    def hwnd(self,widget):return self.u.GetAncestor(widget.winfo_id(),2)

    def position(self,handle,rect):
        x,y,r,b=rect
        if not self.u.SetWindowPos(handle,None,x,y,r-x,b-y,0x14):raise ctypes.WinError()  # NOZORDER | NOACTIVATE; no SHOW

    def window_state(self,handle):
        from ctypes import wintypes as w
        rect,client,origin=w.RECT(),w.RECT(),w.POINT(0,0)
        if not(self.u.GetWindowRect(handle,ctypes.byref(rect)) and self.u.GetClientRect(handle,ctypes.byref(client)) and self.u.ClientToScreen(handle,ctypes.byref(origin))):raise ctypes.WinError()
        return dict(rect=[rect.left,rect.top,rect.right,rect.bottom],client_size=[client.right-client.left,client.bottom-client.top],
            client_origin=[origin.x,origin.y],hmonitor=int(self.u.MonitorFromWindow(handle,0) or 0),
            awareness=self.u.GetAwarenessFromDpiAwarenessContext(self.u.GetWindowDpiAwarenessContext(handle)),dpi=self.u.GetDpiForWindow(handle))


_display=None
def display_api():
    global _display
    if _display is None:_display=Win32Display()
    return _display


@contextmanager
def physical_pixels(*,api=None):
    with (api or display_api()).pixels():yield


def final_monitors(*,api=None):
    """Keep caller/WM rects as evidence; only verified physical rects drive FINAL."""
    api=api or display_api();awareness=api.awareness();wm={m['name']:m['rect'] for m in api.monitors()}
    result=[]
    with api.pixels():
        monitors=api.monitors()
        try:paths=api.paths();path_error=''
        except OSError as exc:paths=[];path_error=str(exc)
        for item in monitors:
            l,t,r,b=item['rect'];row=dict(item,x=l,y=t,w=r-l,h=b-t,physical_rect=item['rect'],window_manager_rect=wm.get(item['name']),
                caller_awareness=awareness,probe_awareness=api.awareness(),verified=False,verification_error='',
                geometry_source='EnumDisplaySettingsExW + QueryDisplayConfig + per-monitor GetMonitorInfoW')
            try:
                row['dpi']=api.monitor_dpi(item['hmonitor']);mode=api.mode(item['name']);row['desktop_mode']=mode
                matches=[p for p in paths if p['name']==item['name']]
                if path_error:raise ValueError(path_error)
                if len(matches)!=1:raise ValueError('CCD: brak jednoznacznego aktywnego monitora (lub klonowanie).')
                path=matches[0];row['display_path']=path;row['signal_size']=path['signal_size']
                signal=path['signal_size'][::-1] if path['rotation'] in (2,4) else path['signal_size']
                if not path['target_available']:raise ValueError('Monitor odłączony.')
                if mode['size']!=[r-l,b-t] or mode['position']!=[l,t] or path['source_size']!=mode['size'] or path['source_position']!=[l,t]:raise ValueError('Niezgodne wymiary/pozycje z niezależnych API Windows.')
                if signal!=mode['size']:raise ValueError('Raster pulpitu różni się od aktywnego sygnału TV.')
                if row['probe_awareness']!=2:raise ValueError('Nie potwierdzono fizycznej domeny współrzędnych.')
                row['verified']=True
            except (ValueError,OSError) as exc:row['verification_error']=str(exc)
            result.append(row)
    return sorted(result,key=lambda m:(not m['primary'],m['name']))


def monitor_description(monitor):
    scale=monitor.get('dpi',{}).get('scale_percent');signal=monitor.get('signal_size')
    return (f"{monitor['name']} · {monitor['w']}×{monitor['h']} px"
        +(f' · DPI {scale}%' if scale else '')
        +(f" · sygnał {signal[0]}×{signal[1]}" if signal and signal!=[monitor['w'],monitor['h']] else '')
        +(' · NO_GO' if not monitor.get('verified') else ''))


def validate_final_monitor(monitor,layout=None):
    prefix=(f"Plan: {layout['width']}×{layout['height']}; " if layout else '')+monitor_description(monitor)
    if not monitor.get('verified'):raise ValueError('NO_GO: '+prefix+'; '+monitor.get('verification_error','niezweryfikowany monitor'))
    if layout and [monitor['w'],monitor['h']]!=[layout['width'],layout['height']]:
        raise ValueError('NO_GO: '+prefix+'; źródło: '+monitor['geometry_source']+'. Plansza nie została przeskalowana. Wybierz zgodny plan/monitor lub świadomie przygotuj nowy plan.')
    return monitor


def final_window_geometry(window,canvas,monitor,*,position=False,check_canvas=True,api=None):
    """Verify HWND/client/Canvas pixels and monitor BEFORE showing, and afterwards."""
    api=api or display_api();validate_final_monitor(monitor)
    with api.pixels():
        window.update_idletasks();handle=api.hwnd(window)
        expected=[monitor['x'],monitor['y'],monitor['x']+monitor['w'],monitor['y']+monitor['h']]
        if position:api.position(handle,expected);window.update_idletasks()
        if check_canvas:
            # SetWindowPos queues Tk Configure/Map events; idletasks alone does
            # not dispatch them. Drain only window/layout events, not timers or
            # worker callbacks, with a finite bound before verifying the pixels.
            import _tkinter
            for _ in range(64):
                if [canvas.winfo_width(),canvas.winfo_height()]==[monitor['w'],monitor['h']]:break
                if not window.tk.dooneevent(_tkinter.WINDOW_EVENTS|_tkinter.IDLE_EVENTS|_tkinter.DONT_WAIT):break
            window.update_idletasks()
        state=api.window_state(handle);child=api.window_state(canvas.winfo_id())
        if (state['rect']!=expected or state['client_origin']!=expected[:2] or state['client_size']!=[monitor['w'],monitor['h']]
            or state['hmonitor']!=monitor['hmonitor'] or state['awareness']!=2
            or (check_canvas and (child['client_size']!=state['client_size'] or child['client_origin']!=state['client_origin']
                or child['awareness']!=2 or [canvas.winfo_width(),canvas.winfo_height()]!=state['client_size']))):
            raise ValueError('NO_GO: nie potwierdzono okna/Canvas 1:1 na '+monitor['name']+'. Nie wyświetlono przeskalowanej planszy.')
        return dict(window=state,canvas=child,tk_canvas_size=[canvas.winfo_width(),canvas.winfo_height()],canvas_verified=check_canvas,domain='PHYSICAL_PIXELS')
