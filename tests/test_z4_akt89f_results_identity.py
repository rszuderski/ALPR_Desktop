"""AKT89F: approved Canvas is distinct from history and model identity is recorded."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from test_z4_akt87_verified_mt_mz import make_export, canvas_example, SHA, SID
from test_z4_akt85_import_charts import make_report
from auto_annotation_tool.gui.mobile_report_canvas_view import load_canvas_snapshot, canvas_overview, export_stamp
from auto_annotation_tool.gui.mobile_report_science_dashboard import build_overview, ScientificDashboard
from auto_annotation_tool.gui.mobile_model_identity import recorded_model_identity, identity_from_bundle
from auto_annotation_tool.gui.mobile_comparison_core import load_session_export, figures, Latency
from auto_annotation_tool.gui.mobile_import_comparison import export_imported_charts


def provenance(mt="MT-261009-1053-4A2D", mz="MZ-261007-2336-390B", mz_variant="tflite-fp32"):
    return dict(experiment_session=dict(package_id="package-"+mt+"-"+mz),
        pipeline_manifests=dict(package=dict(package_id="package-"+mt+"-"+mz, name="Recorded package")),
        model_refs={role:dict(model_display_id=name, model_id="model-"+name, variant_id=variant,
            checkpoint_sha256=char*64, package_sha256=char.upper()*64)
            for role,name,variant,char in (("plate",mt,"tflite-fp32","a"),("character",mz,mz_variant,"b"))})


def reseal(directory):
    path = directory / "export_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for row in manifest["files"]:
        raw = (directory / row["path"]).read_bytes()
        row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    path.write_text(json.dumps(manifest), encoding="utf-8")


def approved_export(root, *, r02=False, sid=SID, sha=SHA, mt="MT-261009-1053-4A2D"):
    c = canvas_example()
    linked, missed = c["rows"]
    c["rows"] = []
    for i in range(18):
        gt = "ABC1234" if i < 17 else "ABCD1234"
        pred = gt if i < (16 if r02 else 15) else ("X"+gt[1:] if i < 17 or r02 else "XY"+gt[2:])
        errors = sum(a != b for a,b in zip(gt,pred))
        c["rows"].append(dict(linked, object_id=f"GT:{i}", gt_text=gt, mz_prediction=pred,
            cer=errors/len(gt), polygon_iou_pct=88.9088466403544 if r02 else 89.00641336228016,
            nmce_pct=2.088679302041393 if r02 else 2.1075489840662343))
    c["rows"].append(dict(missed, gt_text=""))
    c.update(scene_count=19, checked_scenes=19, expected_plates=19, linked_plates=18,
        geometry_n=18, evaluated_reads=18, exact_reads=16 if r02 else 15,
        gt_characters=127, text_errors=2 if r02 else 4, cer=(2 if r02 else 4)/127,
        scenes=[dict(status="CHECKED") for _ in range(19)],
        polygon_iou_mean_pct=c["rows"][0]["polygon_iou_pct"], nmce_mean_pct=c["rows"][0]["nmce_pct"])
    directory = make_export(root, canvas=c, rev=181 if r02 else 247, sid=sid, sha=sha)
    path = directory / "review.json"
    review = json.loads(path.read_text(encoding="utf-8"))
    review["provenance"].update(provenance(mt=mt))
    path.write_text(json.dumps(review), encoding="utf-8")
    reseal(directory)
    return directory


@pytest.mark.parametrize("r02,exact,cer,iou", [(False,"15/18","3,15%","89,01%"),(True,"16/18","1,57%","88,91%")])
def test_verified_own_revision_canvas_cards_preserve_fn_and_denominators(tmp_path,r02,exact,cer,iou):
    directory=approved_export(tmp_path,r02=r02)
    frozen={p.name:p.read_bytes() for p in directory.iterdir()}
    snap=load_canvas_snapshot(SID,SHA,tmp_path,tmp_path/'reviews')
    view=canvas_overview(snap,session_id=SID,source_sha=SHA)
    assert view['ready'] and view['status']=='ZATWIERDZONA'
    assert view['cards'][0]['value']=='18/19' and 'FN: 1' in view['cards'][0]['note']
    assert iou in view['cards'][1]['value'] and 'n=18' in view['cards'][1]['note']
    assert view['cards'][2]['value']==exact and view['cards'][3]['value']==cer
    assert f"{2 if r02 else 4}/127" in view['cards'][3]['note']
    assert frozen=={p.name:p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize("change", ['sha','sid','revision','canvas_sid'])
def test_unrelated_or_stale_snapshot_never_shows_approved_cards(tmp_path,change):
    approved_export(tmp_path)
    snap=load_canvas_snapshot(SID,SHA,tmp_path,None)
    if change=='sha':snap['source_sha256']='c'*64
    elif change=='sid':snap['session_id']='other'
    elif change=='revision':snap['summary']['review_revision']+=1
    else:snap['summary']['session_id']='other'
    view=canvas_overview(snap,session_id=SID,source_sha=SHA)
    assert not view['ready'] and all(x['value']=='—' for x in view['cards'])


def test_live_revision_and_newer_export_revoke_cached_quality(tmp_path):
    directory=approved_export(tmp_path)
    before=export_stamp(tmp_path,SID)
    sidecar=tmp_path/'reviews/human_reviews'/f'{SHA}.review.json'
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps(dict(session_id=SID,source_archive_sha256=SHA,
        review_revision=248,review_status='IN_PROGRESS')),encoding='utf-8')
    snap=load_canvas_snapshot(SID,SHA,tmp_path,tmp_path/'reviews')
    assert snap['summary'] is None and snap['verification']['state']=='STALE'
    assert canvas_overview(snap,session_id=SID,source_sha=SHA)['status']=='EKSPORT NIEAKTUALNY'
    make_export(tmp_path,rev=249)
    assert export_stamp(tmp_path,SID)!=before


def test_history_and_phone_times_remain_separate_from_canvas(tmp_path):
    approved_export(tmp_path)
    snap=load_canvas_snapshot(SID,SHA,tmp_path,None)
    r,b=make_report(sid=SID)
    b.model_refs=provenance()['model_refs'];b.pipeline_manifests=provenance()['pipeline_manifests']
    historical=dict(source_archive_sha256=SHA,mt=dict(single_plate_n=0,localized_n=0),
        pose={},mz={},alpr={},progress={},form_verification={},_ui_canvas_snapshot=snap)
    before=deepcopy(historical)
    view=build_overview(r,b,historical)
    assert view['canvas']['cards'][0]['value']=='18/19'
    assert view['cards'][0]['value']=='Brak wyniku'
    assert historical==before
    unreviewed=build_overview(r,b,dict(source_archive_sha256=SHA,_ui_canvas_progress=dict(
        session_id=SID,source_sha256=SHA,total=19,inspected=0,remaining=19)))
    # An unversioned operational inventory does not prove image population or
    # saved human decisions. Timing and MT invocation counts cannot supply it.
    assert not unreviewed['canvas']['ready']
    assert unreviewed['canvas']['progress'] == 'Nie ustalono liczby obrazów do oceny'
    assert unreviewed['canvas']['progress_ratio'] is None
    assert unreviewed['performance']==view['performance']


def test_model_identity_missing_conflicting_and_mixed_precision():
    p=provenance(mz_variant='tflite-int8')
    result=recorded_model_identity(p,verified=True)
    assert result['variant_caption']=='MT FP32 / MZ INT8'
    assert result['models']['plate']['model_display_id']=='MT-261009-1053-4A2D'
    assert result['models']['character']['package_sha256']=='B'*64
    unknown=recorded_model_identity(p,verified=False)
    assert 'brak danych źródłowych' in unknown['caption'] and not unknown['package_id']
    missing=recorded_model_identity(dict(p,model_refs={}),verified=True)
    assert 'brak danych źródłowych' in missing['caption']
    conflict=recorded_model_identity(p,verified=True,recorded_package_id='different')
    assert conflict['conflicts'] and 'brak danych źródłowych' in conflict['caption']


def test_two_fp32_model_pairs_have_distinct_plot_labels_and_export_metadata(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=tmp_path/'03_wyniki';root.mkdir();(root/'porownanie').mkdir()
    a=load_session_export(approved_export(root))
    b=load_session_export(approved_export(root,sid='exp-second',sha='b'*64,mt='MT-OTHER-FP32'))
    assert a.label!=b.label and 'MT-OTHER-FP32' in b.label
    output=export_imported_charts([a,b],[[],[]],root)
    meta=json.loads((output/'wykresy_manifest.json').read_text(encoding='utf-8'))
    models=[s['package_and_models']['models']['plate']['model_display_id'] for s in meta['sources']]
    assert models==['MT-261009-1053-4A2D','MT-OTHER-FP32']
    for source,record in zip(meta['sources'],[a,b]):
        assert source['package_and_models']==record.model_identity
        assert source['source_archive_sha256']==record.source_sha256
        assert source['review_revision']==record.review_revision
        assert source['review_export_manifest_sha256']==record.manifest_sha256
    for item in meta['files']:
        raw=(output/item['name']).read_bytes()
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    for name,fig in figures([a,b]):
        try:
            fig.canvas.draw()
            renderer=fig.canvas.get_renderer()
            for ax in fig.axes:
                labels=[ax.title,ax.xaxis.label,ax.yaxis.label,*ax.get_yticklabels()]
                if ax.get_legend():labels.append(ax.get_legend())
                for artist in labels:
                    box=artist.get_window_extent(renderer)
                    assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1, (name,box,fig.bbox)
        finally:plt.close(fig)


@pytest.mark.parametrize("width",[1649,1180])
def test_native_primary_cards_history_collapsed_and_two_column_readability(tmp_path,width):
    import tkinter as tk
    approved_export(tmp_path)
    snap=load_canvas_snapshot(SID,SHA,tmp_path,None)
    r,b=make_report(sid=SID)
    p=provenance();b.model_refs=p['model_refs'];b.pipeline_manifests=p['pipeline_manifests']
    root=tk.Tk();root.geometry(f'{width-620}x680+0+0')
    panel=ScientificDashboard(root,palette={},on_review=lambda:None,on_refresh=lambda:None,
                              on_details=lambda:None,show_identity=False)
    panel.pack(fill='both',expand=True)
    try:
        data=build_overview(r,b,dict(source_archive_sha256=SHA,_ui_canvas_snapshot=snap))
        panel.render(data);root.update()
        assert not panel.history.opened and not panel.history.body.winfo_ismapped()
        assert panel.card_vars[0][0].get()=='18/19'
        assert panel.card_vars[2][0].get()=='15/18'
        assert 'MT-261009-1053-4A2D' in panel.package_var.get()
        assert all(w.winfo_ismapped() for w in panel._card_frames)
        assert panel._card_columns==2
        assert not panel.package_label.winfo_ismapped()
        panel.history.toggle();root.update()
        assert panel.history.body.winfo_ismapped()
        assert 'Brak wyniku' in panel.history_vars[0].get()
    finally:root.destroy()


@pytest.mark.parametrize("size",["1649x896","1180x760"])
def test_comparison_preview_keeps_titles_axes_and_model_legend_inside(tmp_path,size):
    import tkinter as tk
    from auto_annotation_tool.gui.mobile_import_comparison_gui import ImportedMobileComparisonWindow
    root=tk.Tk();root.withdraw()
    a=load_session_export(approved_export(tmp_path))
    b=load_session_export(approved_export(tmp_path,sid='second',sha='b'*64,mt='MT-OTHER-FP32'))
    a=replace(a,latency={key:Latency(19,700,780) for key in ('mt','mz','pipeline')})
    b=replace(b,latency={key:Latency(19,710,790) for key in ('mt','mz','pipeline')})
    window=ImportedMobileComparisonWindow(root,[a,b],[[],[]])
    try:
        window.geometry(size+'+0+0');root.update()
        for index in (0,len(window.figure_items)-1,0):
            window.plot_choice.current(index);window.show_plot();root.update()
            fig=window.figure_items[index][1]
            fig.canvas.draw();renderer=fig.canvas.get_renderer()
            widget=window.drawn.get_tk_widget()
            assert abs(fig.bbox.width-widget.winfo_width())<=1
            assert abs(fig.bbox.height-widget.winfo_height())<=1
            for artist in fig.texts:
                box=artist.get_window_extent(renderer)
                assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,(size,index,box,fig.bbox)
            for ax in fig.axes:
                labels=[ax.title,ax.xaxis.label,ax.yaxis.label,*ax.get_yticklabels(),*ax.texts]
                if ax.get_legend():labels.append(ax.get_legend())
                for artist in labels:
                    box=artist.get_window_extent(renderer)
                    assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,(size,index,box,fig.bbox)
                for note in fig.texts:
                    if note is fig._suptitle:
                        continue
                    footer=note.get_window_extent(renderer)
                    for artist in (ax,ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()):
                        assert not footer.overlaps(artist.get_window_extent(renderer)), (size,index,'footer overlap',artist)
    finally:
        window._close();root.destroy()
