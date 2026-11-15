"""Publication semantics and physical layout using the frozen real F4 cohort."""
from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import fields,replace
import copy,json,hashlib
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text
import pytest
from auto_annotation_tool.gui.mobile_analysis import METRICS,MetricEligibility
from auto_annotation_tool.gui.mobile_analysis_export import make_figure,figure_plan,DEFAULT_FIGURES
from auto_annotation_tool.gui.mobile_analysis_presentation import LABELS,BOUNDED,bar_label,format_value,report_sections
from _private_test_data import private_test_path
OUT=private_test_path('output/Z4_READABLE_CHARTS_20261102_001')

@pytest.fixture(autouse=True)
def require_frozen_chart_evidence():
    if not all((OUT/name).is_file() for name in ('analysis_before.json','baseline.json')):
        pytest.skip('Frozen chart evidence not supplied; set ALPR_TEST_DATA_DIR.')

def real_cohort():
    p=json.loads((OUT/'analysis_before.json').read_text(encoding='utf-8'));sessions=[]
    allowed={f.name for f in fields(MetricEligibility)}
    for s in p['sessions']:
        sessions.append(NS(input=NS(session_id=s['session_id'],source_sha256=s['source_archive_sha256'],report_id=s['report_id']),
            metadata=s['metadata'],metrics={k:MetricEligibility(**{n:v for n,v in row.items() if n in allowed}) for k,row in s['metrics'].items()}))
    return NS(sessions=sessions,question=p['question'],protocol_files=p['protocol_files'],payload=lambda:p)

@pytest.fixture
def cohort():return real_cohort()

def joined(fig):return ' '.join(t.get_text().replace('\n',' ') for t in fig.findobj(Text) if t.get_visible())

def assert_layout(fig):
    canvas=FigureCanvasAgg(fig);canvas.draw();renderer=canvas.get_renderer();frame=fig.bbox
    items=[*fig.texts,*fig.axes[0].texts,*fig.axes[0].get_xticklabels(),*fig.axes[0].get_yticklabels(),fig.axes[0].xaxis.label]
    for t in items:
        if not t.get_visible() or not t.get_text():continue
        box=t.get_window_extent(renderer)
        assert box.x0>=-1 and box.x1<=frame.width+1 and box.y0>=-1 and box.y1<=frame.height+1,(t.get_text(),box.bounds,frame.bounds)
    labels=[t.get_window_extent(renderer) for t in fig.axes[0].texts]
    for a,b in zip(labels,labels[1:]):assert not a.overlaps(b),'Adjacent value labels overlap'
    head=fig._alpr_layout
    if head['legend_inline']:
        assert not head['context'].get_window_extent(renderer).overlaps(head['legend'].get_window_extent(renderer))
        assert head['legend'].get_window_extent(renderer).y0>head['title'].get_window_extent(renderer).y1
    else:assert head['subtitle'].get_window_extent(renderer).y0>=head['legend'].get_window_extent(renderer).y1-1
    assert head['note'].get_window_extent(renderer).y1<fig.axes[0].xaxis.label.get_window_extent(renderer).y0
    assert len(head['note'].get_text().splitlines())<=2

@pytest.mark.parametrize('key',list(METRICS))
@pytest.mark.parametrize('scenario',['PAIR_DAY_01','PAIR_NIGHT_01'])
def test_L21_L22_all31_metrics_fit_at_14cm_without_changing_values(cohort,key,scenario):
    before=copy.deepcopy([(s.metadata,s.metrics) for s in cohort.sessions])
    fig=make_figure(cohort,key,scenario);assert_layout(fig)
    assert fig.get_figwidth()*2.54==pytest.approx(14)
    assert [(s.metadata,s.metrics) for s in cohort.sessions]==before
    assert all(v is None or v>=0 for v in fig._alpr_analysis['values'])

def test_L01_L04_L09_L20_exact_title_axis_counts_and_stable_order(cohort):
    cohort.sessions.reverse();fig=make_figure(cohort,'exact_mz_pct','PAIR_DAY_01');assert_layout(fig)
    text=joined(fig)
    assert LABELS['exact_mz_pct']['title'] in text
    assert 'Odsetek poprawnych odczytów [%]' in text
    assert '83,3% · 20 z 24 numerów' in text and '55,6% · 10 z 18 numerów' in text
    assert fig.axes[0].get_xlim()==(0,100)
    assert [r['label'] for r in fig._alpr_analysis['rows']]==[f'{v} · powtórzenie {i}' for v in ('FP32','INT8') for i in (1,2,3)]
    assert not any(word in text for word in ('DAY_','PAIR_DAY','braki=','FULL','DESCRIPTIVE'))

def test_L05_L08_L10_L31_night_subset_is_not_a_third_variant(cohort):
    day=make_figure(cohort,'exact_mz_pct','PAIR_DAY_01');night=make_figure(cohort,'exact_mz_pct','PAIR_NIGHT_01');assert_layout(night)
    text=joined(night);assert 'Zdjęcia nocne na TV' in text and '50,0% · 8 z 16 numerów' in text
    assert 'dane częściowe: 16 z 18 odczytów' in text
    assert [p.get_facecolor() for p in day.axes[0].patches]==[p.get_facecolor() for p in night.axes[0].patches]
    assert len(night.legends[0].get_texts())==2 and all(p.get_hatch() for p in night.axes[0].patches[3:])

@pytest.mark.parametrize('key',['exact_e2e_pct','no_read_e2e_pct','mz_call_p95_ms'])
def test_L06_L36_missing_values_are_explained_and_not_zero_bars(cohort,key):
    fig=make_figure(cohort,key,'PAIR_DAY_01');assert_layout(fig)
    assert not fig.axes[0].patches and all(v is None for v in fig._alpr_analysis['values'])
    assert 'Brak danych do wiarygodnej oceny' in joined(fig)
    if 'e2e' in key:assert 'Nie obliczono — brak mapy scen' in joined(fig)

@pytest.mark.parametrize('key',['micro_cer_pct','nmce_pct'])
def test_L19_unbounded_error_percent_above100_is_not_clamped(cohort,key):
    s=cohort.sessions[0];s.metrics[key]=replace(s.metrics[key],value=145.5,numerator=291,denominator=200)
    fig=make_figure(cohort,key,'PAIR_DAY_01');assert_layout(fig)
    assert fig.axes[0].get_xlim()[1]>145.5 and fig.axes[0].patches[0].get_width()==145.5
    assert '145,5%' in joined(fig)

@pytest.mark.parametrize('value',[0,100])
def test_L20_L21_exact_endpoint_label_stays_outside_plot_without_expanding_scale(cohort,value):
    cohort.sessions[0].metrics['exact_mz_pct']=replace(cohort.sessions[0].metrics['exact_mz_pct'],value=value)
    fig=make_figure(cohort,'exact_mz_pct','PAIR_DAY_01');assert_layout(fig)
    assert fig.axes[0].get_xlim()==(0,100) and fig.axes[0].patches[0].get_width()==value
    assert fig.axes[0].texts[0].get_window_extent(fig.canvas.get_renderer()).x0>fig.axes[0].bbox.x1

def test_L07_L23_png_svg_metadata_and_visible_copy_match(cohort,tmp_path):
    from PIL import Image
    import xml.etree.ElementTree as ET
    fig=make_figure(cohort,'exact_mz_pct','PAIR_DAY_01');meta=json.dumps(fig._alpr_analysis,ensure_ascii=False)
    for ext in ('png','svg'):fig.savefig(tmp_path/('chart.'+ext),dpi=300,metadata={'Description':meta})
    with Image.open(tmp_path/'chart.png') as im:assert json.loads(im.info['Description'])==fig._alpr_analysis and im.info['dpi'][0]==pytest.approx(300,abs=.01)
    root=ET.parse(tmp_path/'chart.svg').getroot();desc=root.find('.//{http://purl.org/dc/elements/1.1/}description')
    assert json.loads(desc.text)==fig._alpr_analysis
    assert all(r['session_id'] and len(r['source_archive_sha256'])==64 for r in fig._alpr_analysis['rows'])

def test_L11_L18_L33_units_and_interpretations_have_distinct_meaning(cohort):
    for key in [k for k in METRICS if k.endswith('_ms')]:
        fig=make_figure(cohort,key,'PAIR_DAY_01');assert '1000 ms = 1 s' in joined(fig)
        assert 'Czas' in LABELS[key]['x_axis'] and fig._alpr_analysis['source_unit']=='ms'
    assert 'ślad' in LABELS['mz_trace_p95_ms']['subtitle'] and 'nie czas pojedynczego' in LABELS['mz_trace_p95_ms']['subtitle']
    assert 'kamera–ekran' in LABELS['pipeline_p95_ms']['footnote']
    assert 'F1 łączy' in LABELS['mt_f1_pct']['subtitle']
    assert 'wszystkich ocenionych wskazań' in LABELS['mt_precision_pct']['subtitle']
    assert 'referencji zapisanych' in LABELS['mt_recall_pct']['subtitle']
    assert 'część wspólna / suma' in LABELS['polygon_iou_pct']['subtitle']
    assert 'przekątna' in LABELS['nmce_pct']['subtitle']
    assert 'Wstawienia + usunięcia + podstawienia / znaki referencji' in LABELS['micro_cer_pct']['footnote']
    assert all('Historyczna' in LABELS[k]['title'] for k in METRICS if k.startswith('bbox_v1_'))

def test_L24_L25_L26_L34_dictionary_markdown_and_filenames(cohort):
    assert set(LABELS)==set(METRICS) and len(LABELS)==31
    report='\n'.join(report_sections(cohort));assert all(v['title'] in report for v in LABELS.values())
    assert 'precyzja OCR' not in report and 'Pochodzenie i dane techniczne' in report
    assert all(s.input.session_id in report and s.input.source_sha256 in report for s in cohort.sessions)
    assert list(figure_plan(cohort))==[(scenario,key) for scenario in ('PAIR_DAY_01','PAIR_NIGHT_01') for key in DEFAULT_FIGURES]

def test_L28_no_source_units_or_metric_formula_changes():
    base=json.loads((OUT/'baseline.json').read_text());root=OUT.parents[1]
    for p in ['auto_annotation_tool/gui/mobile_analysis.py','auto_annotation_tool/gui/mobile_analysis_object_quality.py','auto_annotation_tool/ranking/mobile_mt_object_evaluation.py','auto_annotation_tool/ranking/mobile_object_mz_quality.py']:
        assert hashlib.sha256((root/p).read_bytes()).hexdigest()==base['sources'][p]
