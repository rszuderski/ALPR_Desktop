"""Z4MOBREP019 FIX03: readable non-color-only latency plots.

Presentation-only tests: no Android, archive, GT, or benchmark writes.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT / 'auto_annotation_tool/gui/mobile_report_chart_styles.py'
BROWSER = ROOT / 'auto_annotation_tool/gui/z4_mobile_report_browser.py'
spec = importlib.util.spec_from_file_location('alpr_chart_style_test', STYLE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CanvasRecorder:
    def __init__(self):
        self.calls=[]
    def create_oval(self,*p,**k):self.calls.append(('oval',p,k))
    def create_rectangle(self,*p,**k):self.calls.append(('rectangle',p,k))
    def create_polygon(self,*p,**k):self.calls.append(('polygon',p,k))


def _relative_luminance(hex_color):
    rgb=[int(hex_color[i:i+2],16)/255 for i in (1,3,5)]
    vals=[v/12.92 if v<=0.04045 else ((v+0.055)/1.055)**2.4 for v in rgb]
    return sum(v*k for v,k in zip(vals,(0.2126,0.7152,0.0722)))


def test_models_have_distinct_robust_colors_and_shapes():
    styles=m.LATENCY_SERIES_STYLES
    assert set(styles)=={'total_ms','plate_inference_ms','character_inference_ms','vehicle_inference_ms'}
    assert styles['plate_inference_ms'].marker=='circle'
    assert styles['character_inference_ms'].marker=='square'
    assert styles['total_ms'].marker=='diamond'
    assert styles['total_ms'].line_width>styles['plate_inference_ms'].line_width
    assert len({s.color for s in styles.values()})==4
    assert len({s.marker for s in styles.values()})==4
    assert styles['plate_inference_ms'].color != styles['character_inference_ms'].color


def test_series_contrast_against_light_report_background():
    # Data marks need sufficient contrast on typical pale Tk report panels.
    bg=_relative_luminance('#F1F3EC')
    for style in m.LATENCY_SERIES_STYLES.values():
        fg=_relative_luminance(style.color)
        assert (max(bg,fg)+0.05)/(min(bg,fg)+0.05)>=3.0, style


def test_chart_markers_are_shapes_not_just_different_colors():
    seen=set()
    for key,style in m.LATENCY_SERIES_STYLES.items():
        canvas=CanvasRecorder()
        m.draw_chart_marker(canvas,100,75,style,background='#EEEEEE')
        assert len(canvas.calls)==1
        shape,coordinates,opts=canvas.calls[0]
        assert opts['fill']==style.color
        assert opts['outline']=='#EEEEEE'
        assert len(coordinates) in (4,6,8)
        seen.add(shape)
    assert seen=={'oval','rectangle','polygon'}


def test_legend_and_data_marker_have_same_shape_and_color():
    for style in m.LATENCY_SERIES_STYLES.values():
        canvas=CanvasRecorder()
        m.draw_chart_marker(canvas,20,11,style,radius=4)
        m.draw_chart_marker(canvas,120,80,style,radius=3.3)
        legend,point=canvas.calls
        assert legend[0]==point[0]
        assert legend[2]['fill']==point[2]['fill']==style.color


def test_browser_uses_pinned_plot_styles_not_theme_semantic_colors():
    text=BROWSER.read_text(encoding='utf-8')
    tree=ast.parse(text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MobileReportBrowser')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_draw_latency_chart')
    segment=ast.get_source_segment(text,fn)
    assert 'LATENCY_SERIES_STYLES' in segment
    assert segment.count('draw_chart_marker(')>=3
    assert 'self.accent)' not in segment
    assert 'self.warning)' not in segment
    assert 'self.success)' not in segment
    assert 'smooth=False' in segment
    assert 'axis_ticks(xs, plot_w, timeline["axis_label"])' in segment
    assert 'pixel_column_extrema(xs, values, min_x, max_x, plot_w)' in segment
    assert 'continuous_segments(values, xs' in segment
    assert 'dense = len(xs) > 350' in segment
    assert 'No sampling at 100 frames' in segment
