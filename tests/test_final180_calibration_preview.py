"""Static calibration is technical T; it must never create a research run."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_final180_desktop import frozen,material
from auto_annotation_tool.gui import final180_tv as tv
from auto_annotation_tool.ranking import final180_markers as markers, final180_reference as ref, photo_reference as photo


@pytest.mark.parametrize('size',[(1280,720),(1920,1080),(2560,1440)])
def test_full_photo_field_and_unchanged_marker_renderer(size):
    manifest=dict(layout=markers.layout(*size),geometry_policy=markers.GEOMETRY_POLICY,exposures=[dict(exposure_id='e0001')])
    before=deepcopy(manifest);frame,identity=tv.calibration_frame(manifest,'a'*64,'b'*32)
    reference,_=markers.render(None,manifest['layout'],identity['token'])
    assert frame.size==size and manifest==before
    l,t,r,b=manifest['layout']['photo_rect']
    mask=np.ones((size[1],size[0]),dtype=bool);mask[t:b,l:r]=False
    assert np.array_equal(np.asarray(frame)[mask],np.asarray(reference)[mask])
    assert identity['photo_rect']==[l,t,r,b]
    assert frame.getpixel((l,t))!=(0,0,0) and frame.getpixel((r-1,b-1))!=(0,0,0)
    assert identity['purpose']=='TEST_ONLY' and identity['exposure_id']=='TEST_ONLY' and identity['phase']=='T'
    decoded=markers.decode(frame,manifest,'a'*64,'b'*32)
    assert decoded['status']=='N/A' and decoded['reason']=='TECHNICAL_TRANSITION_FRAME'
    assert decoded['decoded_tokens']==[identity['token']]*2 and not decoded.get('exposure_id')
    assert markers.decode(frame,manifest,'a'*64,'c'*32)['reason']=='FOREIGN_MANIFEST_OR_RUN'


def test_preview_identity_is_independent_and_real_photo_has_no_guides():
    from PIL import Image
    m=dict(layout=markers.layout(1920,1080));first,a=tv.calibration_frame(m,'a'*64);second,b=tv.calibration_frame(m,'a'*64)
    assert a['technical_id']!=b['technical_id']
    image=Image.new('RGB',(100,100),'#123456')
    frame,fit=markers.render(image,m['layout'],markers.token('a'*64,'d'*32,'e0001','P'))
    assert frame.getpixel((960,540))==(18,52,86)
    assert fit['rect']!=m['layout']['photo_rect']  # Portrait/square fit does not redefine the common calibration field.


def test_native_preview_lifecycle_and_both_panel_entrypoints(tmp_path,monkeypatch):
    if os.environ.get('F6_CALIBRATION_ISOLATED_TK')!='1':
        process=subprocess.run([sys.executable,'-m','pytest',str(Path(__file__).resolve())+'::test_native_preview_lifecycle_and_both_panel_entrypoints','-q'],
            env=dict(os.environ,F6_CALIBRATION_ISOLATED_TK='1'),capture_output=True,text=True,timeout=120)
        assert process.returncode==0,process.stdout+process.stderr
        return
    import tkinter as tk
    from auto_annotation_tool import config
    from auto_annotation_tool.session import SessionManager
    from auto_annotation_tool.gui.z4_mobile_report_browser import MobileReportBrowser
    monkeypatch.setattr(config,'SESSION',SessionManager(tmp_path/'preferences.json'))
    d=ref.freeze(material(tmp_path/'fixture'),tmp_path/'prepared',dwell_seconds=5);m,_,_=ref.load(d)
    assert m['status']=='NO_GO' and m['acceptance']['a12_tv']=='NOT_VERIFIED'
    protected={p:photo.digest(p) for p in d.rglob('*') if p.is_file()}
    legacy=tv.legacy_module(tv.LEGACY_DEFAULT)
    monitors=[dict(name='TEST_ONLY',w=1920,h=1080,x=0,y=0,verified=True,geometry_source='ISOLATED_TEST')]
    monkeypatch.setattr(tv.monitor_dpi,'final_monitors',lambda:monitors)
    root=tk.Tk();root.withdraw()
    args=NS(xml=None,positives=None,negatives=None,output=tmp_path/'plans',dwell=5,negatives_declared=True,unreadable_ids=[])
    app=tv.make_app(legacy,args,parent=root);app.window.withdraw()
    def viewer(_):
        app.viewer=tk.Toplevel(root);app.viewer.withdraw();app.canvas=tk.Canvas(app.viewer)
        app.viewer.bind('<space>',lambda _:pytest.fail('Inherited START must be removed'))
    monkeypatch.setattr(app,'_make_viewer',viewer)
    monkeypatch.setattr(ref.FrozenMaterial,'image',lambda *a:pytest.fail('Calibration must not read a FINAL photo'))
    try:
        app.loaded(d);app.dwell.set('7')  # Unsaved operator choice must survive preview.
        original_run=tv.PresentationRun
        monkeypatch.setattr(tv,'PresentationRun',lambda *a,**kw:pytest.fail('Preview must not create a run'))
        original_status=app.status.get();original_mode=app.mode
        app.show_calibration()
        assert app.mode=='CALIBRATION' and app.run is None and app.cache is None and app.pending_id is None
        assert str(app.calibration_btn.cget('text'))=='Zamknij planszę'
        assert app.viewer.bind('<space>')=='' and app.viewer.bind('<Escape>')
        identity=deepcopy(app.calibration_preview['identity']);root.update()
        assert app.calibration_preview['identity']==identity and not (d/'runs').exists()
        assert app.manifest['status']==m['status'] and app.manifest['acceptance']==m['acceptance']
        # Exercise the actual Tcl Escape binding without global keyboard input.
        app.viewer.tk.eval(app.viewer.bind('<Escape>').replace('%#','0').replace('%b','0').replace('%f','0').replace('%h','0').replace('%k','0').replace('%s','0').replace('%t','0').replace('%w','0').replace('%x','0').replace('%y','0').replace('%A','{}').replace('%E','0').replace('%K','Escape').replace('%N','0').replace('%W',str(app.viewer)).replace('%T','2').replace('%X','0').replace('%Y','0').replace('%D','0'))
        assert app.viewer is None and app.calibration_preview is None and app.mode==original_mode
        assert app.status.get()==original_status and app.dwell.get()=='7'
        app.show_calibration();app.calibration_btn.invoke()
        assert app.viewer is None and app.calibration_preview is None
        for mode in tv.ACTIVE_MODES:
            app.mode=mode;app.show_calibration();assert app.viewer is None
        app.mode='IDLE';app.monitors[0]['w']=1280;app.show_calibration()
        assert app.viewer is None and 'przeskalowana' in app.status.get()
        app.monitors[0]['w']=1920
        app.show_calibration();app.loaded(d);assert app.calibration_preview is None and app.viewer is None
        app.show_calibration();app.screen_choice.set(app.monitor_names[0]);assert app.calibration_preview is None
        app.show_calibration();preview_id=app.calibration_preview['identity']['technical_id']
        # Inspect the START boundary and use the real controller with fake time in a private fixture.
        def controlled_run(*args,**kwargs):
            # Calibration is closed; the new pixel-verified viewer precedes run creation.
            assert app.calibration_preview is None and app.viewer is not None
            return original_run(*args,**kwargs)
        monkeypatch.setattr(tv,'PresentationRun',controlled_run)
        monkeypatch.setattr(tv,'PreparedFrames',lambda *args:NS(request=lambda _:None,close=lambda:None))
        monkeypatch.setattr(app,'_tick',lambda:None)
        app.begin_countdown();assert app.mode=='COUNTDOWN' and app.run.state['run_id']!=preview_id
        assert all(preview_id not in json.dumps(e) and 'TEST_ONLY' not in json.dumps(e) for e in app.run.events)
        app.abort()
        assert all(photo.digest(p)==sha for p,sha in protected.items())
        # Existing app action delegates to the same function used by the launcher.
        calls=[];monkeypatch.setattr(tv,'main',lambda argv,**kwargs:calls.append((argv,kwargs)) or NS(window=NS(winfo_exists=lambda:True)))
        tv.open_from_desktop(root);assert calls[0]==(tv.accepted_arguments(),dict(parent=root))
        browser=NS(window=root);MobileReportBrowser.open_final_presentation(browser)
        assert calls[1]==calls[0]
    finally:app._close();root.destroy()
