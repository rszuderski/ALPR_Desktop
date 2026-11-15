"""UX01: actual configurator widgets, conservative readiness and no fake export.

No trained weights, network, conversion or Android calls. Backend checks are
explicit doubles; window and callback logic comes from the shipped source.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
import pytest

from z4_export_configurator_support import create, pump
from auto_annotation_tool.gui.z4_export_configurator_view import role_plan_text, ExportConfiguratorView

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'auto_annotation_tool/gui/z4_model_export.py'

@pytest.fixture(scope='module')
def tkroot():
    try: root=tk.Tk()
    except tk.TclError as exc: pytest.skip('Tk display unavailable: '+str(exc))
    root.geometry('120x80+0+0');pump(root)
    yield root
    root.destroy()

@pytest.fixture
def gui(tkroot,tmp_path):
    errors=[];old=tkroot.report_callback_exception
    tkroot.report_callback_exception=lambda *e:errors.append(e)
    ctx,env=create(tkroot,SOURCE,tmp_path/'packages')
    yield tkroot,ctx,env,errors
    if ctx['executor_modal'].winfo_exists():ctx['executor_modal'].destroy()
    pump(tkroot)
    tkroot.report_callback_exception=old
    assert not errors, errors


def replace_cell(fn,name,value):
    i=fn.__code__.co_freevars.index(name);fn.__closure__[i].cell_contents=value


def preflight_double(ctx,env,problems=()):
    fn=ctx['run_executor_preflight']
    replace_cell(fn,'_executor_model_check_rows',lambda state:[('Jest',state['marker'],'Źródło','test source')])
    replace_cell(fn,'_executor_local_validation_problems',lambda:[])
    replace_cell(fn,'_build_executor_package_request',lambda *a,**k:SimpleNamespace())
    replace_cell(fn,'_executor_missing_install_requirements',lambda:[])
    replace_cell(fn,'_executor_problem_is_reflected',lambda text:False)
    env['exporter'].preflight=lambda request:list(problems)
    return fn


def widgets(ctx,cls):return [w for w in ExportConfiguratorView.walk(ctx['exec_root']) if isinstance(w,cls)]


def test_model_tabs_and_checks_tab_not_two_competing_columns(gui):
    r,c,e,_=gui;v=c['presentation']
    assert [v.notebook.tab(t,'text') for t in v.notebook.tabs()]==['MT · Tablice','MZ · Znaki','Kontrola eksportu']
    assert v.pages['MT'].shell.winfo_ismapped() and not v.pages['MZ'].shell.winfo_ismapped()
    assert v.pages['MT'].canvas.winfo_width()>c['executor_modal'].winfo_width()*.85


def test_switching_model_keeps_independent_inputs(gui):
    r,c,e,_=gui;v=c['presentation'];a,b=c['model_states']
    a['conf'].set(.32);v.notebook.select(v.pages['MZ'].shell);pump(r)
    assert a['conf'].get()==.32 and b['conf'].get()==.25
    assert b['imgsz'].get()==320 and a['imgsz'].get()==512


def test_no_export_or_readiness_on_creation(gui):
    r,c,e,_=gui
    assert not c['exec_preflight_state']['ready']
    assert c['exec_progress_var'].get()==0
    assert not c['exec_progress_row'].winfo_ismapped()
    assert not c['exec_buttons']['pause'].winfo_ismapped()
    assert 'eksport nieuruchomiony' in c['presentation'].phase.get()
    assert 'Sprawdź gotowość' in c['exec_buttons']['primary'].cget('text')


def test_successful_preflight_is_not_export_progress(gui):
    r,c,e,_=gui
    assert preflight_double(c,e)();pump(r)
    assert c['exec_preflight_state']['ready']
    assert c['exec_progress_var'].get()==0 and c['exec_progress_percent_var'].get()=='0%'
    assert not c['exec_progress_row'].winfo_ismapped()
    assert c['presentation'].notebook.select()==str(c['presentation'].checks)
    assert 'GOTOWOŚĆ SPRAWDZONA' in c['presentation'].phase.get()
    assert 'eksport nieuruchomiony' in c['presentation'].phase.get()


def test_pending_redraw_does_not_erase_failure_diagnostics(gui):
    r,c,e,_=gui
    assert preflight_double(c,e,['MZ: test brak pliku'])() is False
    original=list(c['executor_requirements_table']['rows'])
    c['_schedule_pending_executor_rows']();pump(r)
    assert c['executor_requirements_table']['rows']==original
    assert any('test brak pliku' in str(row) for row in original)
    assert not c['exec_preflight_state']['ready']


@pytest.mark.parametrize('value',['','abc','1e','-'])
def test_editing_imgsz_does_not_raise_tcl_callback_exception(gui,value):
    r,c,e,_=gui;s=c['model_states'][0]
    preflight_double(c,e)()
    r.setvar(str(s['imgsz']),value);pump(r)
    assert not c['exec_preflight_state']['ready']
    assert 'wejście ? px' in c['export_plan_var'].get()
    assert any('Do korekty' in str(row) for row in c['executor_requirements_table']['rows'])
    s['imgsz'].set(512);pump(r)
    assert 'wejście 512 px' in c['export_plan_var'].get()


@pytest.mark.parametrize('role',['MT','MZ'])
def test_reuse_hides_new_conversion_and_calibration_fields(gui,role):
    r,c,e,_=gui;v=c['presentation'];s=next(s for s in c['model_states'] if s['marker']==role)
    v.notebook.select(v.pages[role].shell)
    s['reuse_'+role.lower()].set(True);s['int8'].set(True);pump(r)
    controls=s['_reuse_'+role.lower()+'_source_controls']
    assert controls.winfo_ismapped()
    status=s['_manifest_status_label'];assert not status.winfo_ismapped()
    assert 'z biblioteki' in c['export_plan_var'].get()
    s['int8'].set(False)


@pytest.mark.parametrize('role',['MT','MZ'])
def test_zero_threshold_remains_selectable_and_preserved(gui,role):
    r,c,e,_=gui;s=next(s for s in c['model_states'] if s['marker']==role)
    s['conf'].set(0.0);s['iou'].set(0.0);pump(r)
    assert s['conf'].get()==0 and s['iou'].get()==0
    spins=[w for w in ExportConfiguratorView.walk(c['presentation'].pages[role].body) if isinstance(w,ttk.Spinbox)]
    assert len([w for w in spins if float(w.cget('from'))==0])==2


@pytest.mark.parametrize('op',['export','install'])
def test_busy_configuration_is_locked_and_close_x_is_guarded(gui,op):
    r,c,e,_=gui
    if op=='export':e['worker_state']['running']=True
    else:c['exec_install_control']['running']=True
    c['_sync_executor_buttons']();pump(r)
    assert c['exec_buttons']['close'].instate(['disabled'])
    for page in c['presentation'].pages.values():
        for w in ExportConfiguratorView.walk(page.body):
            if isinstance(w,(ttk.Entry,ttk.Button,ttk.Checkbutton,ttk.Radiobutton)):
                assert w.instate(['disabled'])
    assert c['_request_executor_close']()=='break' and c['executor_modal'].winfo_exists()
    before=dict(c['exec_preflight_state']);c['_mark_executor_dirty']()
    assert c['exec_preflight_state']==before
    e['worker_state']['running']=False;c['exec_install_control']['running']=False
    c['_sync_executor_buttons']();pump(r)
    assert not c['exec_buttons']['close'].instate(['disabled'])


def test_pause_exists_only_for_dependency_installation(gui):
    r,c,e,_=gui
    e['worker_state']['running']=True;c['_sync_executor_buttons']();pump(r)
    assert not c['exec_buttons']['pause'].winfo_ismapped()
    e['worker_state']['running']=False;c['exec_install_control']['running']=True
    c['_sync_executor_buttons']();pump(r)
    assert c['exec_buttons']['pause'].winfo_ismapped()
    assert 'instalację' in c['exec_buttons']['pause'].cget('text')
    c['exec_install_control']['running']=False;c['_sync_executor_buttons']();pump(r)
    assert not c['exec_buttons']['pause'].winfo_ismapped()


@pytest.mark.parametrize('completed,failed,text',[(True,False,'EKSPORT ZAKOŃCZONY'),(False,True,'BŁĄD EKSPORTU')])
def test_export_outcome_has_distinct_phase(gui,completed,failed,text):
    r,c,e,_=gui;c['exec_operation'].update(completed=completed,failed=failed)
    c['_sync_executor_buttons']();pump(r)
    assert text in c['presentation'].phase.get()
    assert 'nieuruchomiony' not in c['presentation'].phase.get()


def test_changed_settings_invalidate_preflight_not_model_values(gui):
    r,c,e,_=gui
    preflight_double(c,e)();c['model_states'][1]['conf'].set(.35);pump(r)
    assert not c['exec_preflight_state']['ready'] and c['model_states'][1]['conf'].get()==.35
    assert 'Sprawdź gotowość' in c['exec_buttons']['primary'].cget('text')


@pytest.mark.parametrize('size,scale',[('1040x660',1.333),('1280x720',1.667),('1649x870',2.0)])
def test_visible_actions_readable_fonts_and_scrollable_role_at_dpi(gui,size,scale):
    r,c,e,_=gui;r.tk.call('tk','scaling',scale)
    c['executor_modal'].geometry(size+'+0+0');pump(r,10)
    w=c['executor_modal'];v=c['presentation']
    for b in (c['exec_buttons']['primary'],c['exec_buttons']['close'],v.phase_label,v.status_text):
        assert b.winfo_ismapped()
        x=b.winfo_rootx()-w.winfo_rootx();y=b.winfo_rooty()-w.winfo_rooty()
        assert x>=0 and y>=0 and x+b.winfo_width()<=w.winfo_width()+1 and y+b.winfo_height()<=w.winfo_height()+1
    assert v.pages['MT'].canvas.winfo_height()>=100
    for label in widgets(c,tk.Label):
        size_actual=abs(int(tkfont.Font(root=r,font=label.cget('font')).actual('size')))
        assert size_actual>=10
    assert v.font_body.actual('size')==11


def test_wheel_scrolls_active_card_without_changing_spinbox(gui):
    r,c,e,_=gui;v=c['presentation'];s=c['model_states'][0]
    c['executor_modal'].geometry('1040x660+0+0')
    buttons=[b for b in widgets(c,ttk.Button) if 'Inne formaty' in str(b.cget('text'))]
    buttons[0].invoke();pump(r)
    spin=next(w for w in ExportConfiguratorView.walk(v.pages['MT'].body) if isinstance(w,ttk.Spinbox))
    before=s['imgsz'].get();other=v.pages['MZ'].canvas.yview()
    spin.event_generate('<MouseWheel>',delta=-120);pump(r)
    assert s['imgsz'].get()==before
    assert v.pages['MT'].canvas.yview()[0]>0
    assert v.pages['MZ'].canvas.yview()==other


def test_fractional_wheel_deltas_accumulate(gui):
    r,c,e,_=gui;p=c['presentation'].pages['MT']
    c['executor_modal'].geometry('1040x660');pump(r)
    buttons=[b for b in widgets(c,ttk.Button) if 'Inne formaty' in str(b.cget('text'))];buttons[0].invoke();pump(r)
    p.wheel(SimpleNamespace(delta=-40));p.wheel(SimpleNamespace(delta=-40));p.wheel(SimpleNamespace(delta=-40))
    pump(r);assert p.canvas.yview()[0]>0


def test_hidden_extra_formats_keep_values_and_visible_count(gui):
    r,c,e,_=gui;s=c['model_states'][0]
    s['onnx'].set(True);pump(r)
    b=next(b for b in widgets(c,ttk.Button) if 'Inne formaty' in str(b.cget('text')))
    assert 'wybrane: 1' in b.cget('text') and 'ONNX FP32' in c['export_plan_var'].get()
    b.invoke();b.invoke();pump(r)
    assert s['onnx'].get()
    assert e['_selected_export_format_quantizations'](s)['onnx']==('fp32',)


def test_default_path_is_a_plan_not_successful_write(gui):
    r,c,e,_=gui;preflight_double(c,e)()
    last=c['executor_requirements_table']['rows'][-1]['values']
    assert last['status']=='Plan' and 'wybór przy eksporcie' in last['requirement']
    assert not Path(last['description']).exists()


def test_all_source_values_remain_in_wrapped_table(gui):
    r,c,e,_=gui;table=c['executor_requirements_table']
    raw='SHA '+'a'*64+' ścieżka '*30
    table['set_rows']([{'values':{'status':'Brak','model':'MZ','requirement':'test','description':raw}}])
    c['presentation'].show_checks();pump(r)
    assert any(w.cget('text')==raw for w in table['labels'])
    assert table['canvas'].winfo_height()>100


@pytest.mark.parametrize('roles',[('MP',),('MT',),('MZ',),('MP','MT','MZ')])
def test_other_supported_selections_keep_role_pages(tkroot,tmp_path,roles):
    c,e=create(tkroot,SOURCE,tmp_path,roles)
    try:
        assert tuple(c['presentation'].pages)==roles
        assert len(c['presentation'].notebook.tabs())==len(roles)+1
        assert all(s['litert'].get() for s in c['model_states'])
    finally:c['executor_modal'].destroy();pump(tkroot)


def test_only_explicit_preflight_selects_checks_tab(gui):
    r,c,e,_=gui;v=c['presentation'];tab=v.notebook.select()
    c['model_states'][0]['conf'].set(.2);pump(r)
    assert v.notebook.select()==tab
    preflight_double(c,e)();assert v.notebook.select()==str(v.checks)


def test_f4_reuse_requires_both_explicit_selections(gui):
    r,c,e,_=gui;a,b=c['model_states']
    a['reuse_mt'].set(True);pump(r);assert not c['f4_no_conversion'].get()
    b['reuse_mz'].set(True);pump(r);assert c['f4_no_conversion'].get()
    a['reuse_mt'].set(False);pump(r);assert not c['f4_no_conversion'].get()


def test_production_guards_and_no_fallback_remain():
    source=SOURCE.read_text(encoding='utf-8-sig')
    for key in ['require_f4_selection(', 'require_f4_no_conversion_request(bundle_request)',
                'require_f4_no_conversion_request(request, require_fresh_destination=True)',
                'validate_existing_mt_package(', 'validate_existing_mz_package(',
                'export_f4_atomic_no_overwrite(', 'record_conversion_results(', 'record_conversion_starts(']:
        assert key in source
    modal=source.split('    def open_selected_mobile_export_executor()',1)[1].split('\n    def export_package()',1)[0]
    assert 'title="Android stabilny"' not in modal and 'title="Android lekki"' not in modal
    assert 'model_grid.grid_columnconfigure' not in modal


def test_helper_contains_no_model_io_or_exporter_imports():
    path=ROOT/'auto_annotation_tool/gui/z4_export_configurator_view.py'
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):assert not node.module or 'exporter' not in node.module
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name): assert node.func.id not in {'open','eval','exec'}


def test_window_styles_do_not_modify_global_label_style(gui):
    r,c,e,_=gui
    style=ttk.Style(r);before=(style.configure('TLabel') or {}).copy()
    c['presentation'].style_widgets(c['exec_root']);pump(r)
    assert (style.configure('TLabel') or {})==before


def scoped_callback(name, env):
    tree=ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(matches)==1
    node=matches[0]
    namespace=dict(env)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),namespace)
    return namespace[name]


def test_late_mz_calibration_lookup_is_not_scheduled_for_reused_mz(gui):
    r,c,e,_=gui;state=c['model_states'][1];state['int8'].set(True);state['reuse_mz'].set(True)
    calls=[]
    queue=scoped_callback('_queue_auto_manifest_lookup',dict(state=state,marker='MZ',
            worker_state={'running':False},exec_install_control={'running':False},
            _state_reuses_existing_mt=lambda s:False,_state_reuses_existing_mz=lambda s:True,
            manifest_auto_timer={'id':None},_manifest_status=lambda *a:None,
            executor_modal=SimpleNamespace(after=lambda *a:calls.append(a)),_search_manifest=lambda **k:None))
    queue();assert not calls
    state['int8'].set(False)


def test_queued_paint_cannot_replace_failed_preflight():
    callbacks=[];writes=[]
    fn=scoped_callback('_schedule_pending_executor_rows',dict(
        _pending_executor_rows_after={'id':None},exec_preflight_state={'ready':False,'problems':['missing']},
        worker_state={'running':False},exec_install_control={'running':False},
        executor_modal=SimpleNamespace(after_idle=lambda f:(callbacks.append(f) or 'job')),
        _set_executor_rows=lambda rows:writes.append(rows),_pending_executor_rows=lambda:['pending']))
    fn();callbacks[0]();assert not writes


def test_sync_plan_catches_incomplete_tk_integer():
    class Var:
        def __init__(self,value):self.value=value
        def get(self):return self.value
        def set(self,value):self.value=value
    class BadVar:
        def get(self):raise tk.TclError('expected integer')
    state={'marker':'MT','reuse_mt':Var(False),'reuse_mt_path':Var(''),'reuse_mt_precision':Var('int8'),'imgsz':BadVar()}
    calls=[]
    fn=scoped_callback('_sync_f4_reuse_plan',dict(model_states=[state],tk=tk,
           _refresh_plan_preview=lambda:calls.append('preview'),export_plan_var=Var(''),
           f4_no_conversion=Var(False),summarize_export_plan=lambda rows:('text',False)))
    fn();assert calls==['preview']


def test_preflight_success_never_sets_export_to_one_hundred():
    class Var:
        def __init__(self,value):self.value=value
        def get(self):return self.value
        def set(self,value):self.value=value
    progress=Var(0);percent=Var('0%')
    env=dict(tk=tk,worker_state={'running':False},exec_install_control={'running':False},
         exec_operation={'completed':False},exec_preflight_state={},model_states=[],
         exec_progress_var=progress,exec_progress_percent_var=percent,
         presentation=SimpleNamespace(show_checks=lambda:None),
         _set_executor_status=lambda *a,**k:None,executor_modal=SimpleNamespace(update_idletasks=lambda:None),
         _executor_model_check_rows=lambda s:[],_executor_local_validation_problems=lambda:[],
         _default_executor_destination=lambda:Path('plan-only.alprmodel'),
         _build_executor_package_request=lambda *a,**k:object(),MobileAlprPackageRequest=type('Unused',(),{}),
         exporter=SimpleNamespace(preflight=lambda req:[]),artifact=SimpleNamespace(label='MT+MZ'),
         _executor_missing_install_requirements=lambda:[],_set_executor_rows=lambda rows:None,
         f4_no_conversion=Var(False),exec_progress=SimpleNamespace(stop=lambda:None,configure=lambda **k:None),
         _sync_executor_buttons=lambda:None)
    assert scoped_callback('run_executor_preflight',env)()
    assert progress.value==0 and percent.value=='0%'


def test_shared_worker_during_install_is_not_labelled_as_export(gui):
    r,c,e,_=gui
    e['worker_state']['running']=True;c['exec_install_control']['running']=True
    c['_sync_executor_buttons']();pump(r)
    assert c['presentation'].phase.get()=='INSTALACJA ZALEŻNOŚCI'
    assert c['exec_buttons']['primary'].cget('text')=='Instaluję zależności...'
    e['worker_state']['running']=False;c['exec_install_control']['running']=False
    c['_sync_executor_buttons']();pump(r)


def test_dependency_progress_label_tracks_actual_progress_variable(gui):
    r,c,e,_=gui
    c['exec_install_control']['running']=True;c['_sync_executor_buttons']()
    c['exec_progress_var'].set(37.4);pump(r)
    assert c['exec_progress_percent_var'].get()=='37%'
    c['exec_install_control']['running']=False;c['_sync_executor_buttons']();pump(r)
    assert not c['exec_progress_row'].winfo_ismapped()
