"""Presentation-only regression: exact coordinates, provenance, picks and DPI."""
from copy import deepcopy
from dataclasses import replace
import io
import xml.etree.ElementTree as ET

import pytest

from test_z4_akt89f_results_identity import approved_export
from auto_annotation_tool.gui.mobile_comparison_core import load_session_export, _geometry_text_figure
from auto_annotation_tool.gui.mobile_compare_review_gate import paired_geometry_text
from auto_annotation_tool.gui.mobile_geometry_scatter import inspection_records, hit_indices, fit_geometry_scatter


@pytest.fixture
def records(tmp_path):
    import matplotlib
    matplotlib.use('Agg')  # GUI cases explicitly attach TkAgg; pure plots need no Tk root.
    return [replace(load_session_export(approved_export(tmp_path,sid=f'exp-repeat-{i}',sha=str(i)*64)),
        repeat_label=f'r{i:02d}') for i in (1,2,3)]


def figure(records):
    return _geometry_text_figure(records,scenario='PAIR_DAY_01',measure='nmce_pct')


def test_exact_existing_pairs_and_literal_text_provenance_are_preserved(records):
    frozen=deepcopy(records)
    raw=records[0].canvas['rows'][0]
    raw['gt_text']='aBc 1234';raw['mz_prediction']='aBc 1234'
    points=inspection_records(records)
    expected=[(s.session_id,p) for s in records for p in paired_geometry_text(s.canvas)]
    assert len(points)==len(expected)==54
    for actual,(sid,pair) in zip(points,expected):
        assert actual['session_id']==sid
        assert all(actual[k]==v for k,v in pair.items())
        s=next(s for s in records if s.session_id==sid)
        assert actual['source_sha256']==s.source_sha256
        assert actual['review_revision']==s.review_revision
        assert actual['review_export_manifest_sha256']==s.manifest_sha256
    assert points[0]['gt_text']==points[0]['mz_prediction']=='aBc 1234'
    raw.update(gt_text=frozen[0].canvas['rows'][0]['gt_text'],mz_prediction=frozen[0].canvas['rows'][0]['mz_prediction'])
    assert records==frozen


def test_original_gate_excludes_missed_suppressed_unincluded_and_unapproved(records):
    canvas=records[0].canvas
    original=deepcopy(canvas['rows'][0])
    canvas['rows'].append(dict(original,included=False,object_id='excluded',mz_status='NOT_RUN',cer=None))
    # The approved fixture already contains a manual MISSED_MT without CER.
    rows=inspection_records([records[0],replace(records[1],review_status='IN_PROGRESS')])
    assert len(rows)==18
    assert not any(r['object_id']=='excluded' or r['mz_status']=='NOT_RUN' for r in rows)


def test_series_markers_legend_and_all_scientific_offsets(records):
    import matplotlib.pyplot as plt
    fig=figure(records)
    try:
        model=fig._alpr_geometry_scatter
        labels=[t.get_text() for t in model['legend'].get_texts()]
        assert labels==['r01 · n=18','r02 · n=18','r03 · n=18']
        assert model['identity_text'].count('MT-261009-1053-4A2D')==1
        paths=[s.get_paths()[0].vertices.tolist() for s in model['series']]
        assert len({repr(p) for p in paths})==3
        for s,artist in zip(records,model['series']):
            expected=[[r['nmce_pct'],r['cer_pct']] for r in paired_geometry_text(s.canvas)]
            assert artist.get_offsets().tolist()==expected
        assert 'nie są niezależne' in model['footer_text']
        assert 'Współwystępowanie' in model['footer_text']
    finally:plt.close(fig)


def test_overlapping_and_nearby_points_all_pickable_without_jitter(records):
    import matplotlib.pyplot as plt
    fig=figure(records)
    try:
        fig.canvas.draw();model=fig._alpr_geometry_scatter;rows=model['records'];ax=model['axes']
        before=[a.get_offsets().tolist() for a in model['series']]
        p=rows[0];x,y=ax.transData.transform((p['nmce_pct'],p['cer_pct']))
        same=[i for i,r in enumerate(rows) if (r['nmce_pct'],r['cer_pct'])==(p['nmce_pct'],p['cer_pct'])]
        assert len(same)>1 and set(same)<=set(hit_indices(fig,x,y))
        assert set(same)<=set(hit_indices(fig,x+2,y+2))
        assert hit_indices(fig,-100,-100)==[]
        for i,r in enumerate(rows):
            px,py=ax.transData.transform((r['nmce_pct'],r['cer_pct']))
            assert i in hit_indices(fig,px,py)
        assert [a.get_offsets().tolist() for a in model['series']]==before
    finally:plt.close(fig)


def test_different_recorded_models_and_checkpoints_remain_distinguishable(records):
    import matplotlib.pyplot as plt
    other=replace(records[1],mt_checkpoint='d'*64)
    fig=figure([records[0],other])
    try:
        m=fig._alpr_geometry_scatter
        assert 'M1:' in m['identity_text'] and 'M2:' in m['identity_text']
        assert 'dddddddd' in m['identity_text']
        assert [t.get_text() for t in m['legend'].get_texts()]==['r01 · M1 · n=18','r02 · M2 · n=18']
    finally:plt.close(fig)


def test_png_svg_render_without_clipped_annotations_or_coordinate_changes(records):
    import matplotlib.pyplot as plt
    from PIL import Image
    fig=figure(records)
    try:
        model=fig._alpr_geometry_scatter
        offsets=[c.get_offsets().tolist() for c in model['series']]
        for dpi in (100,220):
            fig.set_dpi(dpi);fit_geometry_scatter(fig);fig.canvas.draw()
            renderer=fig.canvas.get_renderer()
            for artist in [*fig.texts,model['legend'],model['axes'].xaxis.label,model['axes'].yaxis.label]:
                box=artist.get_window_extent(renderer)
                assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
            assert [c.get_offsets().tolist() for c in model['series']]==offsets
        png=io.BytesIO();fig.savefig(png,format='png',dpi=220);png.seek(0)
        with Image.open(png) as im:im.load();assert im.width>2000
        svg=io.BytesIO();fig.savefig(svg,format='svg');svg.seek(0);ET.parse(svg)
        assert [c.get_offsets().tolist() for c in model['series']]==offsets
    finally:plt.close(fig)


@pytest.mark.parametrize('size',['1649x896','1180x760'])
def test_native_inspection_all_records_errors_resize_and_disconnect(records,size):
    import tkinter as tk
    from matplotlib.backend_bases import MouseEvent
    from auto_annotation_tool.gui.mobile_import_comparison_gui import ImportedMobileComparisonWindow
    root=tk.Tk();root.withdraw();w=ImportedMobileComparisonWindow(root,records,[[],[],[]])
    errors=[];root.report_callback_exception=lambda *a:errors.append(a)
    try:
        w.geometry(size);root.update()
        index=next(i for i,(name,_) in enumerate(w.figure_items) if name.endswith('nmce_vs_cer_pairs'))
        w.plot_choice.current(index);w.show_plot();root.update()
        inspector=w.point_inspector;fig=w.drawn.figure;model=fig._alpr_geometry_scatter;ax=model['axes']
        assert len(inspector.visible_indices)==54
        p=model['records'][0];x,y=ax.transData.transform((p['nmce_pct'],p['cer_pct']))
        fig.canvas.callbacks.process('button_press_event',MouseEvent('button_press_event',fig.canvas,x,y,button=1))
        assert len(inspector.visible_indices)>3
        for i in inspector.visible_indices:
            inspector.select_record(i)
            assert inspector.current_record is model['records'][i]
            text=inspector.detail.get('1.0','end')
            assert model['records'][i]['session_id'] in text
            assert model['records'][i]['gt_text'] in text
        inspector.show_errors();assert all(model['records'][i]['cer_pct']>0 for i in inspector.visible_indices)
        assert len(inspector.visible_indices)==sum(r['cer_pct']>0 for r in model['records'])
        assert sum(len(c.get_offsets()) for c in model['series'])==54
        assert ax.bbox.width*ax.bbox.height/(fig.bbox.width*fig.bbox.height)>.5
        dpi=fig.dpi;old_cid=inspector._cid
        w.plot_choice.current(0);w.show_plot();root.update()
        assert w.point_inspector is None and old_cid not in fig.canvas.callbacks.callbacks.get('button_press_event',{})
        assert not any(c.get_gid()=='inspection-highlight' for c in ax.collections)
        for _ in range(3):
            w.plot_choice.current(index);w.show_plot();root.update()
            assert fig.dpi==dpi
            w.plot_choice.current(0);w.show_plot();root.update()
        assert not errors
    finally:w._close();root.destroy()
