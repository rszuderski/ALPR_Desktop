"""Usable native layout and a read-only preview, including Windows DPI startup."""
import sys,time
import pytest
import windows_dpi

windows_dpi.enable_windows_dpi_awareness()

from test_photo_material_workflow import selected
from test_final180_final_blockers import app_fixture,close_app
from auto_annotation_tool.gui import photo_material_ui


def pump(root,predicate,timeout=15):
    end=time.monotonic()+timeout
    while not predicate() and time.monotonic()<end:
        root.update();time.sleep(.01)
    root.update()
    assert predicate()


@pytest.fixture
def gui(selected,tmp_path,monkeypatch):
    root,app,errors=app_fixture(tmp_path,monkeypatch)
    app.window.title('TEST UKŁADU — materiały syntetyczne')
    app.window.deiconify();app.use_material(selected)
    pump(root,lambda:app.material_preview.image is not None)
    yield root,app,errors
    close_app(root,app)


def inside(window,widget):
    x,y=widget.winfo_rootx()-window.winfo_rootx(),widget.winfo_rooty()-window.winfo_rooty()
    assert widget.winfo_ismapped(),str(widget)
    assert x>=-1 and y>=-1,(widget,x,y)
    assert x+widget.winfo_width()<=window.winfo_width()+1,(widget,x,widget.winfo_width(),window.winfo_width())
    assert y+widget.winfo_height()<=window.winfo_height()+1,(widget,y,widget.winfo_height(),window.winfo_height())


def test_primary_controls_fit_small_normal_and_large_windows(gui):
    root,app,errors=gui
    for width,height in ((960,620),(1240,780),(1800,960)):
        app.window.geometry(f'{width}x{height}+0+0');root.update()
        for _ in range(8):root.update();time.sleep(.015)
        for widget in (app.start_btn,app.choose_material_btn,app.choose_set_btn,app.advanced_btn,
                app.monitor_combo,app.dwell_entry,app.edit_reference_btn,app.reference_menu_btn,
                app.material_preview.canvas,app.material_preview.previous,app.material_preview.next):
            inside(app.window,widget)
        for widget in (app.start_btn,app.choose_material_btn,app.choose_set_btn,app.edit_reference_btn,app.reference_menu_btn):
            assert widget.winfo_width()>=widget.winfo_reqwidth(),widget.cget('text')
        assert app.material_preview.canvas.winfo_width()>app.window.winfo_width()*.4
        assert app.material_preview.canvas.winfo_height()>app.window.winfo_height()*.3
        assert not app.stop_btn.winfo_ismapped() and not app.advanced.winfo_ismapped()
    assert not errors


def test_preview_navigation_resume_and_advanced_preserve_material_without_gt(gui,selected):
    root,app,errors=gui
    app.material_preview.next.invoke()
    pump(root,lambda:app.material_preview.image is not None)
    assert app.material_preview.index==1 and not selected.default_output.exists()
    before=app.window.geometry()
    app.toggle_advanced();root.update()
    assert app.advanced.winfo_ismapped() and not app.material_body.winfo_ismapped()
    inside(app.window,app.advanced_btn)
    app.toggle_advanced();root.update()
    assert app.window.geometry()==before and app.material_preview.index==1
    assert not selected.default_output.exists()
    app.material_preview.material=None
    app.sync_presentation_ui()
    assert app.material_preview.index==1 and not errors


def test_stop_returns_to_editable_material_actions(gui,selected,monkeypatch):
    root,app,errors=gui
    app.mode='STOPPED';app.refresh_plan()
    assert app.material_actions_available() and str(app.edit_reference_btn.cget('state'))=='normal'
    monkeypatch.setattr(photo_material_ui.filedialog,'askdirectory',lambda **kwargs:str(selected.root))
    app.choose_photographs()
    pump(root,lambda:not app._selection_loading)
    assert app.mode=='IDLE' and app.source_material.material_id==selected.material_id
    assert not errors


def test_calibration_prepares_material_without_starting_a_run(gui,selected,monkeypatch):
    root,app,errors=gui;called=[]
    monkeypatch.setattr(app,'toggle_calibration',lambda:called.append(True))
    app.preview_calibration()
    pump(root,lambda:bool(called))
    assert app.prepared is not None and app.run is None
    assert not selected.default_output.exists() and not list(selected.directory.glob('presentations/*/runs/*/run.json'))
    assert not errors


@pytest.mark.skipif(sys.platform!='win32',reason='Native Windows DPI contract')
def test_window_dpi_matches_tk_font_scale(gui):
    root,app,_=gui;api=windows_dpi.display_api()
    window_dpi=api.u.GetDpiForWindow(app.window.winfo_id())
    awareness=api.u.GetAwarenessFromDpiAwarenessContext(api.u.GetWindowDpiAwarenessContext(app.window.winfo_id()))
    assert awareness in (1,2)
    assert abs(root.winfo_fpixels('1i')-window_dpi)<2
