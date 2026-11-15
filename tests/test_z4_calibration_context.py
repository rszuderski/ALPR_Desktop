"""SYNTHETIC context identities with real Tk widgets; no production conversion."""
from dataclasses import replace
import json
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from auto_annotation_tool.gui import z4_int8_calibration as ui
from auto_annotation_tool.gui import z4_model_export as export
from auto_annotation_tool.gui.z4_calibration_context import (
    context_from_candidate, verify_selection, run_label, assert_manifest_context)
from auto_annotation_tool.exporters import exploratory_calibration as exp
from int8_calibration_fixture import fixture, write_json
from test_z4_int8_calibration import root, pump, calibration_tk_runtime

RUNS = [("plate", "20261009_082038", "MT-DAY", 512), ("plate", "20261009_114220", "MT-NIGHT", 512),
        ("character", "20261007_215302", "MZ-DAY", 320), ("character", "20261008_100135", "MZ-NIGHT", 320)]


def candidate_fixture(tmp_path, role="plate", rid="20261009_082038", label="MT-DAY"):
    f = fixture(tmp_path, role)
    f["run"].update(id=rid, name="SYNTHETIC " + label, started_at="2026-10-09T08:20:49", img_size=256)
    f["checkpoint"].write_bytes(("SYNTHETIC " + rid).encode())
    f["config"]["run_id"] = rid
    write_json(f["history"], {"runs": {rid: f["run"]}})
    c = dict(role=role, target="plate" if role == "plate" else "char", run=SimpleNamespace(**f["run"]),
             history_dir=str(f["history"].parent), best_weights=f["checkpoint"], img_size=256,
             model_label="SYNTHETIC " + label, model_version="YOLO26", task="pose" if role == "plate" else "detect")
    return f, c


def options(root, size):
    return dict(imgsz=tk.IntVar(root, size), conf=tk.DoubleVar(root, .25), iou=tk.DoubleVar(root, .45),
                calibration=tk.StringVar(root, ""), calibration_mode=tk.StringVar(root, "exploratory"))


def bind_inputs(d, f):
    # Context auto-check is already running. A changed exclusion file invalidates
    # that result; a fresh explicit SHA check verifies the synthetic selection.
    d.values["eval_inventory"].set(str(f["eval_path"]))
    pump(d, lambda: not d._busy)
    d.read_checkpoint_sha(); pump(d, lambda: not d._busy)


@pytest.mark.parametrize("role,rid,label,size", RUNS)
def test_context_role_run_history_and_series_size_are_exact(root,tmp_path,role,rid,label,size):
    f,c = candidate_fixture(tmp_path,role,rid,label)
    ctx=context_from_candidate(c)
    assert (ctx.role,ctx.run_id,ctx.image_size,ctx.history_path)==(role,rid,size,str(f["history"].resolve()))
    d=ui.Int8CalibrationDialog(root,context=ctx,mode="exploratory")
    bind_inputs(d,f)
    assert d.values["run_id"].get()==rid and d.values["image_size"].get()==str(size)
    assert d._verified["sha256"]==exp.strict.file_sha256(f["checkpoint"])
    assert "SYNTHETIC "+label in d.card_text.get() and rid in d.card_text.get()
    assert f["checkpoint"].name in d.card_text.get() and d._verified["checkpoint"]==str(f["checkpoint"].resolve())
    assert "Ustaw maksymalną liczbę próbek" in d.next_text.get()
    assert d._source_audit["counts"]["eligible_unique"]==3 and "3 dopuszczalnych" in d.source_text.get()
    assert not d.approval.get() and not hasattr(d, "research_check")
    assert not d.field_widgets["final_inventory"][1].winfo_manager()
    assert d.final_notice.winfo_manager()=="grid"
    d.destroy()


@pytest.mark.parametrize("fault", ["no_model", "mp", "missing_checkpoint", "missing_history", "wrong_sha", "wrong_run_mapping"])
def test_context_fails_closed_without_guessing(tmp_path,fault):
    f,c=candidate_fixture(tmp_path)
    if fault=="no_model": c=None
    elif fault=="mp": c["role"]="vehicle"
    elif fault=="missing_checkpoint": c["best_weights"]=tmp_path/"missing.pt"
    elif fault=="missing_history": c.pop("history_dir")
    elif fault=="wrong_sha": c["verified_checkpoint_sha256"]="a"*64
    else:
        other=tmp_path/"different.pt"; other.write_bytes(b'different'); c["best_weights"]=other
    with pytest.raises((ValueError,OSError)):
        ctx=context_from_candidate(c)
        verify_selection(ctx.history_path,ctx.run_id,ctx.role,context=ctx)


def test_manual_labels_keep_raw_identity_and_safe_missing_metadata(root,tmp_path):
    f,c=candidate_fixture(tmp_path)
    d=ui.Int8CalibrationDialog(root)
    d.values["history_path"].set(str(f["history"]))
    d.load_runs()
    label=d.run_combo["values"][0]
    assert "MT · SYNTHETIC MT-DAY" in label and "2026-10-09 08:20" in label and "Run: 20261009_082038" in label
    d.run_display.set(label); d.run_combo.event_generate("<<ComboboxSelected>>"); root.update()
    assert d.values["run_id"].get()=="20261009_082038"
    fallback=run_label("20261009_082038",{},"")
    assert "nazwa nieustalona" in fallback and "data nieustalona" in fallback and "rola nieustalona" in fallback
    assert "DAY" not in fallback and "NIGHT" not in fallback
    d.destroy()


@pytest.mark.parametrize("role,rid,label,size", RUNS[:1]+RUNS[2:3])
def test_native_guidance_save_and_attach_only_own_slot(root,tmp_path,role,rid,label,size):
    f,c=candidate_fixture(tmp_path/"inputs",role,rid,label)
    own=options(root,size); other=options(root,320 if role=="plate" else 512)
    status=[]
    d=export._open_candidate_int8_calibration(root,c,options=own,source="Slot MT" if role=="plate" else "Slot MZ",status=status.append)
    bind_inputs(d,f)
    assert d._next_field=="limit"
    d.values["limit"].set("2")
    assert not d.operator_frame.winfo_manager() and not d.approval_check.winfo_manager()
    assert d.values["approved_by"].get()=="" and not d.approval.get()
    assert d._next_action=="prepare"
    d.next_button.invoke();pump(d,lambda:not d._busy)
    assert d.result["status"]=="EXPLORATORY_ONLY" and d._next_action=="save"
    path=tmp_path/"saved.json"
    with patch.object(ui.filedialog,"asksaveasfilename",return_value=str(path)):
        d.next_button.invoke();pump(d,lambda:not d._busy)
    assert own["calibration"].get()==str(path) and other["calibration"].get()==""
    assert own["calibration_fingerprint"].get()==d.result["input_fingerprint"]
    assert d._next_action=="return" and "nie uruchomiono konwersji" in status[-1]
    d.next_button.invoke()


@pytest.mark.parametrize("key,value", [("imgsz",544),("conf",.4),("iou",.6),("calibration","other.json")])
def test_stale_slot_never_receives_manifest(root,tmp_path,key,value):
    f,c=candidate_fixture(tmp_path/"inputs")
    state=options(root,512); status=[]
    d=export._open_candidate_int8_calibration(root,c,options=state,source="Slot MT",status=status.append)
    bind_inputs(d,f)
    d.values["limit"].set("2")
    d.prepare();pump(d,lambda:not d._busy)
    state[key].set(value)
    p=tmp_path/"saved.json"
    with patch.object(ui.filedialog,"asksaveasfilename",return_value=str(p)):
        d.save();pump(d,lambda:not d._busy)
    assert p.exists() and d._saved_path is None
    assert "nie przypisany" in d.status.get()
    assert state["calibration"].get()==("other.json" if key=="calibration" else "")
    d.destroy()


def test_context_parameter_conflict_is_visible_without_retuning(root,tmp_path):
    f,c=candidate_fixture(tmp_path)
    ctx=context_from_candidate(c,image_size=576,confidence_threshold=.31,iou_threshold=.51)
    d=ui.Int8CalibrationDialog(root,context=ctx)
    bind_inputs(d,f)
    d.mode.set(exp.LABEL)
    assert d.values["image_size"].get()=="576"
    d.values["image_size"].set("512");d.values["limit"].set("2");d.prepare()
    assert d.result is None and "Konflikt image_size" in d.status.get() and "kontekst=576" in d.status.get()
    assert not d.approval.get() and not hasattr(d, "research_check")
    d.destroy()


def test_sha_read_is_async_and_close_is_safe(root,tmp_path):
    f,c=candidate_fixture(tmp_path)
    started,release,done=threading.Event(),threading.Event(),threading.Event()
    original=ui.verify_selection; errors=[];root.report_callback_exception=lambda *a:errors.append(a)
    def slow(*a,**kw):
        started.set();release.wait(5)
        try:return original(*a,**kw)
        finally:done.set()
    with patch.object(ui,"verify_selection",side_effect=slow):
        d=ui.Int8CalibrationDialog(root,context=context_from_candidate(c))
        assert started.wait(2) and d._busy
        root.update(); assert d._next_action=="wait"
        d.destroy(); release.set(); assert done.wait(5); root.update()
    assert not errors and d._poll_id is None


def test_strict_missing_final_inventory_still_visible_and_blocked(root,tmp_path):
    f,c=candidate_fixture(tmp_path)
    d=ui.Int8CalibrationDialog(root,context=context_from_candidate(c),mode="exploratory")
    bind_inputs(d,f); d.mode.set("Ścisła kalibracja INT8 — wymaga final_test")
    assert d.field_widgets["final_inventory"][1].winfo_manager()=="grid"
    assert "FINAL_TEST_INVENTORY_NOT_VERIFIED" in d.next_text.get()
    d.values["limit"].set("2"); d.values["approved_by"].set("SYNTHETIC operator");d.approval.set(True)
    d.prepare();pump(d,lambda:not d._busy)
    assert d.result["status"]=="BLOCKED"
    d.destroy()


def test_contextual_modal_slots_preserve_original_candidate_identity(root,tmp_path):
    pairs=[candidate_fixture(tmp_path/label,role,rid,label) for role,rid,label,_ in RUNS]
    candidates=[c for f,c in pairs]
    host=SimpleNamespace(app=SimpleNamespace(root=root,palette={},show_assistant_message=Mock()),frame=root,
                         _build_mobile_model_export_path=lambda *_:tmp_path/'unused.alprmodel')
    def walk(w):
        yield w
        for child in w.winfo_children(): yield from walk(child)
    with patch.object(export,"_collect_mobile_export_candidates",return_value=candidates), \
         patch.object(export,"open_int8_calibration_dialog") as open_dialog, \
         patch.object(export.MobileModelExporter,"export",side_effect=AssertionError("export forbidden")):
        center=export._open_mobile_model_export_center(host);center.deiconify();root.update()
        tree=next(w for w in walk(center) if isinstance(w,ttk.Treeview) and len(w.get_children())==4)
        for index in (0,2):
            iid=candidates[index]["iid"];tree.see(iid);root.update()
            box=tree.bbox(iid,'#1');assert box
            tree.event_generate('<Button-1>',x=box[0]+5,y=box[1]+box[3]//2);root.update()
        # Different highlight must not change the export slots.
        tree.selection_set(candidates[1]['iid']);tree.event_generate('<<TreeviewSelect>>');root.update()
        next(w for w in walk(center) if isinstance(w,ttk.Button) and w['text']=='Eksportuj zaznaczone').invoke();root.update()
        controls=[w for w in walk(center) if isinstance(w,ui.CalibrationModeControl)]
        assert len(controls)==2
        for marker,index in [('MT',0),('MZ',2)]:
            slot=next(c for c in controls if c.variables['marker']==marker)
            slot.variables['litert'].set(False)
            slot.variables['int8'].set(True)
            button=next(w for w in walk(center) if isinstance(w,ttk.Button) and w['text']=='Przygotuj manifest INT8 — '+marker)
            button.invoke()
            ctx=open_dialog.call_args.kwargs['context']
            assert ctx.run_id==RUNS[index][1] and ctx.image_size==RUNS[index][3]
            assert ctx.role==RUNS[index][0] and ctx.source=='Slot eksportu '+marker
        center.destroy()


def test_no_highlight_does_not_open_a_random_calibrator(root):
    messages=[]
    with patch.object(export,'open_int8_calibration_dialog') as opened:
        assert export._open_candidate_int8_calibration(root,None,options=options(root,512),
            source='Wskazany model',status=messages.append) is None
    opened.assert_not_called()
    assert 'Najpierw wskaż model MT lub MZ' in messages[0]


def test_wrong_role_manifest_never_attaches_even_with_same_candidate_params(tmp_path):
    f,c=candidate_fixture(tmp_path/'mt')
    mz=fixture(tmp_path/'mz','character')
    config={k:v for k,v in mz['config'].items() if k!='final_inventory'}
    payload=exp.build_manifest_v1(**config,operator_confirmation=True,checkpoint_sha256=exp.strict.file_sha256(mz['checkpoint']))
    with pytest.raises(ValueError,match='innego modelu'):
        assert_manifest_context(payload,context_from_candidate(c,image_size=32))


def test_save_cancelled_by_input_change_before_write(root,tmp_path):
    f,c=candidate_fixture(tmp_path/'inputs')
    d=ui.Int8CalibrationDialog(root,context=context_from_candidate(c),mode='exploratory')
    bind_inputs(d,f)
    d.values['limit'].set('2')
    d.prepare();pump(d,lambda:not d._busy)
    started,release=threading.Event(),threading.Event();original=exp.build_manifest
    def slow(**kw):
        started.set();release.wait(5);return original(**kw)
    p=tmp_path/'must_not_exist.json'
    with patch.object(exp,'build_manifest',side_effect=slow),patch.object(ui.filedialog,'asksaveasfilename',return_value=str(p)):
        d.save();assert started.wait(2)
        d.values['limit'].set('1');release.set();pump(d,lambda:not d._busy)
    assert not p.exists() and d.result is None and not hasattr(d, "research_check")
    d.destroy()


def test_manifest_prepared_in_mt_slot_does_not_leak_to_mz(root,tmp_path):
    f,mt=candidate_fixture(tmp_path/'mt')
    _,mz=candidate_fixture(tmp_path/'mz','character','20261007_215302','MZ-DAY')
    host=SimpleNamespace(app=SimpleNamespace(root=root,palette={},show_assistant_message=Mock()),frame=root,
                         _build_mobile_model_export_path=lambda *_:tmp_path/'unused.alprmodel')
    def walk(w):
        yield w
        for child in w.winfo_children():yield from walk(child)
    with patch.object(export,'_collect_mobile_export_candidates',return_value=[mt,mz]), \
         patch.object(export.MobileModelExporter,'export',side_effect=AssertionError('export forbidden')):
        center=export._open_mobile_model_export_center(host);center.deiconify();root.update()
        tree=next(w for w in walk(center) if isinstance(w,ttk.Treeview) and len(w.get_children())==2)
        for c in (mt,mz):
            tree.see(c['iid']);root.update();box=tree.bbox(c['iid'],'#1');assert box
            tree.event_generate('<Button-1>',x=box[0]+5,y=box[1]+box[3]//2);root.update()
        next(w for w in walk(center) if isinstance(w,ttk.Button) and w['text']=='Eksportuj zaznaczone').invoke();root.update()
        controls=[w for w in walk(center) if isinstance(w,ui.CalibrationModeControl)]
        assert len(controls)==2
        mt_slot=next(c for c in controls if c.variables['marker']=='MT')
        mt_slot.variables['litert'].set(False);mt_slot.variables['int8'].set(True)
        next(w for w in walk(center) if isinstance(w,ttk.Button) and w['text']=='Przygotuj manifest INT8 — MT').invoke()
        d=next(w for w in walk(center) if isinstance(w,ui.Int8CalibrationDialog))
        d.mode.set(exp.LABEL);bind_inputs(d,f)
        d.values['limit'].set('2')
        d.prepare();pump(d,lambda:not d._busy)
        p=tmp_path/'bound_mt.json'
        with patch.object(ui.filedialog,'asksaveasfilename',return_value=str(p)):
            d.save();pump(d,lambda:not d._busy)
        fingerprint=d.result['input_fingerprint'];assert d._saved_path==p
        d.destroy();root.update()
        for control in controls:
            s=control.variables
            if s['marker']=='MT':
                assert s['calibration'].get()==str(p) and s['calibration_fingerprint'].get()==fingerprint
                assert s['calibration_mode'].get()=='exploratory'
            else: assert not s['calibration'].get()
            assert 'exploratory_confirmation' not in s and not hasattr(control, 'check')
        center.destroy()


def test_tab_skips_labels_and_scrolls_interactive_fields(root):
    root.deiconify()
    d=ui.Int8CalibrationDialog(root);d.geometry('800x600+0+0');root.update()
    assert d.entries['limit'].tk_focusNext()==d.entries['approved_by']
    d.entries['limit'].focus_force();root.update()
    d.entries['limit'].event_generate('<Tab>');root.update()
    assert d.focus_get()==d.entries['approved_by']
    d.destroy()


def test_destroy_releases_owned_tk_variables_on_ui_thread(root):
    import weakref
    d=ui.Int8CalibrationDialog(root)
    refs=[weakref.ref(v) for v in [*d.values.values(),*d.research_values.values(),d.status,d.next_text]]
    d.destroy()
    assert all(ref() is None for ref in refs)


def test_context_card_keeps_expected_sha_from_training_snapshot(tmp_path):
    f,c=candidate_fixture(tmp_path)
    digest=exp.strict.file_sha256(f['checkpoint'])
    f['run']['output_checkpoint_snapshot']={'best_checkpoint_sha256':digest}
    write_json(f['history'],{'runs':{f['config']['run_id']:f['run']}})
    ctx=context_from_candidate(c)
    binding=verify_selection(ctx.history_path,ctx.run_id,ctx.role,context=ctx)
    assert binding['expected_sha256']==binding['sha256']==digest
