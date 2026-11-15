"""Selector semantics only: unchanged renderers/keys, readable labels, refresh."""
from dataclasses import replace

import pytest

from test_z4_akt89i_nmce_inspection import records
from auto_annotation_tool.gui.mobile_plot_choices import PlotChoice, plot_choices


def test_every_existing_summary_and_trace_key_maps_to_the_same_figure(records):
    import matplotlib.pyplot as plt
    from auto_annotation_tool.gui.mobile_comparison_core import figures
    from auto_annotation_tool.gui.mobile_import_trace_charts import trace_figures
    items=list(figures(records))+[('oś czasu · '+key,fig) for key,fig in trace_figures([],session_label='r01')]
    try:
        expected=[
            ('PAIR_DAY_01_mt_p50_ms','Czasy · MT — mediana (p50) [ms]'),
            ('PAIR_DAY_01_mt_p95_ms','Czasy · MT — p95 [ms]'),
            ('PAIR_DAY_01_mz_p50_ms','Czasy · MZ — mediana (p50) [ms]'),
            ('PAIR_DAY_01_mz_p95_ms','Czasy · MZ — p95 [ms]'),
            ('PAIR_DAY_01_pipeline_p50_ms','Czasy · ALPR — mediana (p50) [ms]'),
            ('PAIR_DAY_01_pipeline_p95_ms','Czasy · ALPR — p95 [ms]'),
            ('oś czasu · plate_inference_ms','Czasy · MT — przebieg sesji [ms]'),
            ('oś czasu · character_inference_ms','Czasy · MZ — suma w śladzie [ms]'),
            ('oś czasu · total_ms','Czasy · ALPR — przebieg sesji [ms]'),
            ('PAIR_DAY_01_assisted_exact','Jakość MZ · pełne zgodne numery [%]'),
            ('PAIR_DAY_01_assisted_cer','Jakość MZ · micro-CER [%]'),
            ('PAIR_DAY_01_assisted_iou','Geometria MT · średnie IoU [%]'),
            ('PAIR_DAY_01_assisted_iou_distribution','Geometria MT · rozkład IoU [%]'),
            ('PAIR_DAY_01_assisted_nmce_distribution','Geometria MT · rozkład NMCE [%]'),
            ('PAIR_DAY_01_assisted_iou_vs_cer_pairs','Zależność · IoU a CER (tablice)'),
            ('PAIR_DAY_01_assisted_nmce_vs_cer_pairs','Zależność · NMCE a CER (tablice)'),
        ]
        options=plot_choices(reversed(items))
        assert [(o.key,o.label) for o in options]==expected
        original=dict(items)
        assert len(options)==len(original)==16
        assert all(o.figure is original[o.key] for o in options)
        assert len({o.label for o in options})==16
    finally:
        for _,fig in items:plt.close(fig)


def test_duplicate_keys_do_not_duplicate_choices_and_first_renderer_wins():
    first,duplicate,other=object(),object(),object()
    items=[('DAY_assisted_cer',first),('DAY_mt_p50_ms',other),('DAY_assisted_cer',duplicate)]
    options=plot_choices(items)
    assert len(options)==2
    assert [(x.key,x.figure) for x in options]==[('DAY_mt_p50_ms',other),('DAY_assisted_cer',first)]
    assert items[2][1] is duplicate  # The helper never mutates input or renderers.


def test_scenarios_and_future_colliding_labels_are_not_conflated():
    keys=['PAIR_NIGHT_01_assisted_cer','PAIR_DAY_01_assisted_cer','future_plot','future plot']
    options=plot_choices((k,object()) for k in keys)
    labels={o.key:o.label for o in options}
    assert labels[keys[0]]=='Jakość MZ · micro-CER [%] · PAIR_NIGHT_01'
    assert labels[keys[1]]=='Jakość MZ · micro-CER [%] · PAIR_DAY_01'
    assert len(set(labels.values()))==len(keys)
    assert set(labels)==set(keys)


def _widgets(widget):
    yield widget
    for child in widget.winfo_children():yield from _widgets(child)


@pytest.mark.parametrize('size',['1649x896','1180x760'])
def test_native_refresh_preserves_key_callbacks_and_inspection(records,monkeypatch,size):
    import tkinter as tk
    import matplotlib.pyplot as plt
    from auto_annotation_tool.gui import mobile_import_comparison_gui as module
    from auto_annotation_tool.gui.mobile_geometry_scatter import nmce_cer_figure
    closed_duplicates=[]
    def plots(group):
        for suffix in ('assisted_cer','mt_p50_ms'):
            fig,ax=plt.subplots();ax.plot([1,2],[3,4])
            yield 'PAIR_DAY_01_'+suffix,fig
        duplicate,_=plt.subplots();closed_duplicates.append(duplicate.number)
        yield 'PAIR_DAY_01_mt_p50_ms',duplicate
        yield 'PAIR_DAY_01_assisted_nmce_vs_cer_pairs',nmce_cer_figure(group,scenario='PAIR_DAY_01')
    monkeypatch.setattr(module,'figures',plots)
    root=tk.Tk();root.withdraw()
    errors=[];root.report_callback_exception=lambda *args:errors.append(args)
    w=module.ImportedMobileComparisonWindow(root,records,[[],[],[]])
    try:
        w.geometry(size);root.update()
        combo=str(w.plot_choice);binding=w.plot_choice.bind('<<ComboboxSelected>>')
        key='PAIR_DAY_01_assisted_nmce_vs_cer_pairs'
        w.plot_choice.current(next(i for i,(k,_) in enumerate(w.figure_items) if k==key))
        w.plot_choice.event_generate('<<ComboboxSelected>>');root.update()
        for _ in range(5):
            w._make_charts();root.update()
            assert w.current_plot_key()==key and w.point_inspector is not None
            assert len(w.figure_items)==len(w.plot_choice['values'])==3
            assert len({k for k,_ in w.figure_items})==3
            assert str(w.plot_choice)==combo and w.plot_choice.bind('<<ComboboxSelected>>')==binding
            assert sum(c.winfo_class()=='TCombobox' for c in _widgets(w))==1
            assert not plt.fignum_exists(closed_duplicates[-1])
        assert int(w.plot_choice.cget('height'))==8
        assert any(getattr(c,'cget',lambda _:'')('text')=='Wybierz wykres'
                   for c in _widgets(w) if c.winfo_class()=='TLabel')
        actual_options=module.plot_choices
        monkeypatch.setattr(module,'plot_choices',lambda items:[PlotChoice(o.key,'Nowa etykieta · '+o.label,o.figure) for o in actual_options(items)])
        old_label=w.plot_choice.get();w._make_charts();root.update()
        assert w.current_plot_key()==key and w.plot_choice.get()!=old_label
        w.point_inspector.show_errors()
        assert all(w.point_inspector.records[i]['cer_pct']>0 for i in w.point_inspector.visible_indices)
        assert len(w.point_inspector.model['records'])==54
        calls=[];show=w.show_plot
        monkeypatch.setattr(w,'show_plot',lambda:(calls.append(True),show())[-1])
        w.plot_choice.event_generate('<<ComboboxSelected>>');root.update()
        assert calls==[True]
        assert not errors
    finally:w._close();root.destroy()


def test_native_changed_sessions_fallback_empty_and_mixed_scenario_gate(records,monkeypatch):
    import tkinter as tk
    import matplotlib.pyplot as plt
    from auto_annotation_tool.gui import mobile_import_comparison_gui as module
    def plots(group):
        for suffix in ('mt_p50_ms','assisted_cer'):
            fig,ax=plt.subplots();ax.plot([1,2],[3,4])
            yield 'PAIR_DAY_01_'+suffix,fig
    monkeypatch.setattr(module,'figures',plots)
    root=tk.Tk();root.withdraw();w=module.ImportedMobileComparisonWindow(root,records[:2],[[],[]])
    try:
        root.update();w.plot_choice.current(1);w.show_plot();root.update()
        key=w.current_plot_key();w.selected_list.selection_set(1);w.remove_selected();root.update()
        assert w.records==records[:1] and w.current_plot_key()==key
        assert len(w.figure_items)==5  # Two fixture summaries + three real timeline definitions.
        trace_key='oś czasu · total_ms'
        w.plot_choice.current(next(i for i,(k,_) in enumerate(w.figure_items) if k==trace_key));w.show_plot();root.update()
        w.records=records[:2];w.traces=[[],[]];w._make_charts();root.update()
        assert w.current_plot_key()=='PAIR_DAY_01_mt_p50_ms'
        assert len(w.figure_items)==2
        w.records=[records[0],replace(records[1],scenario='PAIR_NIGHT_01')];w._make_charts();root.update()
        assert not w.figure_items and not w.plot_choice['values'] and w.current_plot_key() is None
        assert 'PAIR_DAY_01' in w.heading.cget('text') and 'PAIR_NIGHT_01' in w.heading.cget('text')
        assert 'Zablokowano' in w.status.get()
        w.records=[];w.traces=[];w._make_charts();root.update()
        assert not w.figure_items and w.plot_choice.get()==''
        assert sum(c.winfo_class()=='TCombobox' for c in _widgets(w))==1
    finally:w._close();root.destroy()


def test_failed_refresh_does_not_leave_stale_options(records,monkeypatch):
    import tkinter as tk
    import matplotlib.pyplot as plt
    from auto_annotation_tool.gui import mobile_import_comparison_gui as module
    monkeypatch.setattr(module,'figures',lambda records:[('PAIR_DAY_01_mt_p50_ms',plt.subplots()[0])])
    root=tk.Tk();root.withdraw();w=module.ImportedMobileComparisonWindow(root,records,[[],[],[]])
    try:
        root.update()
        def fail(records):raise ValueError('Test unavailable renderer')
        monkeypatch.setattr(module,'figures',fail);w._make_charts();root.update()
        assert w.drawn is None and not w.figure_items and not w.plot_choice['values']
        assert w.current_plot_key() is None and 'Test unavailable renderer' in w.status.get()
    finally:w._close();root.destroy()
