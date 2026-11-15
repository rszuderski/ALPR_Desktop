"""Synthetic harness for the real nested configurator. No trained model is loaded.
Only metadata helpers and expensive backends are test doubles. The actual
UI/trace/preflight state callbacks are compiled from z4_model_export.py.
"""
import ast,copy,tkinter as tk, tkinter.font as tkfont
from tkinter import ttk, messagebox,filedialog
from pathlib import Path
from types import SimpleNamespace
import os,sys,queue,threading,time,datetime,math,json,subprocess,re
from dataclasses import replace
from auto_annotation_tool.gui.z4_mobile_export_contract import describe_mobile_export_selection
from auto_annotation_tool.exporters.mobile_model_exporter import MobileExportRequest,MobileAlprPackageRequest,MobileExportError
from auto_annotation_tool.gui.web_slim_scrollbar import blend_hex_colors
from auto_annotation_tool.gui.z4_int8_calibration import CalibrationModeControl,calibration_request_options
from auto_annotation_tool.gui.z4_export_configurator_view import ExportConfiguratorView

def pump(root,n=5):
 for _ in range(n):
  root.update_idletasks();root.update();time.sleep(.003)

def create(root, path, fixture_path, roles=('MT','MZ')):
 code=Path(path).read_text(encoding='utf-8-sig'); tree=ast.parse(code)
 def node(name):return copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name))
 runstate={'running':False}; exports=[]
 class FakeExporter:
  def preflight(self,request):return []
  def export(self,*a,**k):raise AssertionError('No export from UI setup/test')
 fixture_path=Path(fixture_path);fixture_path.mkdir(parents=True,exist_ok=True)
 candidates=[]
 for role in roles:
  f=fixture_path/(role+'.pt');f.write_bytes(b'synthetic only no model')
  candidates.append(dict(marker=role,model_label=f'{role}-SYNTHETIC-NOT-REAL',best_weights=f,
                         target={'MT':'plate','MZ':'char','MP':'vehicle'}[role],role={'MT':'plate','MZ':'character','MP':'vehicle'}[role],model_version='YOLO26n', img_size=320 if role=='MZ' else 512,
                         run=SimpleNamespace(id=role,img_size=320 if role=='MZ' else 512), training_provenance={}))
 app=SimpleNamespace(style_dialog_window=lambda w,**k:(w.title(k['title']),w.geometry(k['geometry'])))
 host=SimpleNamespace(app=app,_build_mobile_alpr_package_export_path=lambda *a,**k:fixture_path/'ALPR_OUT.alprmodel',
                      _build_mobile_model_export_path=lambda *a,**k:fixture_path/'ONE_OUT.alprmodel',
                      _safe_model_export_slug=lambda value,**k:value.replace(' ','-'),
                      _format_training_target_label=lambda value:value)
 env=dict(__name__='auto_annotation_tool.gui._export_ux_test',__package__='auto_annotation_tool.gui',__file__=str(path),
          tk=tk,tkfont=tkfont,ttk=ttk,Path=Path,os=os,sys=sys,queue=queue,threading=threading,time=time,datetime=datetime,json=json,
          subprocess=subprocess,re=re,math=math,replace=replace,messagebox=SimpleNamespace(showwarning=lambda *a,**k:None,showerror=lambda *a,**k:None),filedialog=filedialog,
          self=host,dialog=root,worker_state=runstate, export_marker_order=['MP','MT','MZ'],
          _selected_export_candidates=lambda:candidates,_export_selection_validation=lambda:(True,'pakiet ALPR MT+MZ',''),
          _mobile_export_target_marker=lambda c:c['marker'],describe_mobile_export_selection=describe_mobile_export_selection,
          _bind_mobile_export_child_window_motion=lambda *a,**k:None,_install_mobile_export_win32_move_guard=lambda *a,**k:None,
          _mobile_export_default_calibration_path=lambda c:'',prepared_calibrations={},
          context_from_candidate=lambda c,**k:SimpleNamespace(identity=lambda:'synthetic-'+c['marker']),
          export_image_size=lambda c:c['img_size'],_mobile_export_default_vehicle_class_labels=lambda c:[],
          _mobile_export_target_color=lambda p,c:p['accent'],blend_hex_colors=blend_hex_colors,
          _mobile_export_candidate_params_millions=lambda c:(3.0,'catalog'),_mobile_export_compact_yolo_label=lambda x:x,
          _candidate_metric_label=lambda c:'mAP50–95: 0.787 (test)',_mobile_export_candidate_epochs_label=lambda c:'190',
          _format_mobile_export_params=lambda *a,**k:'3.0 mln',CalibrationModeControl=CalibrationModeControl,
          calibration_request_options=calibration_request_options,_mobile_export_candidate_class_labels=lambda c:[],
          _mobile_export_window_is_moving=lambda:False,_formats_description=lambda r:', '.join(r.formats)+' '+', '.join(r.quantizations),
          MobileExportRequest=MobileExportRequest,MobileAlprPackageRequest=MobileAlprPackageRequest,MobileExportError=MobileExportError,
          exporter=FakeExporter(),package_exporter=FakeExporter(),set_running=lambda x:runstate.update(running=x),set_status=lambda *a,**k:None,
          CONFIG=SimpleNamespace(DEFAULT_IMG_SIZE=512,DEFAULT_CONFIDENCE=.25,DEFAULT_IOU=.45,DIR_6_MODELS_MOBILE_PACKAGES=fixture_path,get_mobile_model_packages_dir=lambda r:fixture_path),
          litert_var=tk.BooleanVar(root,True),int8_var=tk.BooleanVar(root,False),onnx_var=tk.BooleanVar(root,False),onnx_int8_var=tk.BooleanVar(root,False),ncnn_var=tk.BooleanVar(root,False),
          imgsz_var=tk.IntVar(root,512),conf_var=tk.DoubleVar(root,.25),iou_var=tk.DoubleVar(root,.45),calibration_var=tk.StringVar(root,''),destination_var=tk.StringVar(root,''))
 for name in ['_selected_export_format_quantizations','_selected_export_formats_and_quantizations']:
  fn=node(name);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),env)
 fn=node('open_selected_mobile_export_executor');fn.body.append(ast.Return(value=ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])))
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),env)
 ctx=env['open_selected_mobile_export_executor']();pump(root)
 return ctx,env
