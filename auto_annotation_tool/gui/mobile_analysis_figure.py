"""Native publication figures shared by Tk preview and the analysis exporter."""
import math
from .mobile_analysis import METRICS,PROTOCOL_ID
from .mobile_analysis_presentation import (LABELS,STYLES,BOUNDED,ordered_sessions,
    variant,session_label,context_title,bar_label,number)


def _wrapped(text,width_pt,font_size,*,bold=False):
    from matplotlib.textpath import TextToPath
    from matplotlib.font_manager import FontProperties
    font=FontProperties(family='DejaVu Sans',size=font_size,weight='bold' if bold else 'normal')
    measure=TextToPath();lines=[]
    for paragraph in text.split('\n'):
        current=''
        for word in paragraph.split():
            candidate=(current+' '+word).strip()
            if current and measure.get_text_width_height_descent(candidate,font,False)[0]>width_pt:
                lines.append(current);current=word
            else:current=candidate
        lines.append(current)
    return '\n'.join(lines)


def fit_figure(fig,*,preview=False):
    """Wrap within measured physical widths; preserve all labels at 14 cm."""
    spec=fig._alpr_layout;ax=fig.axes[0]
    width,height=fig.get_size_inches()*72
    scale=1.05 if preview else 1
    body=8.2*scale;title_size=11.2*scale;small=8.0*scale
    y=height-10
    for name,size,bold in (('context',8.2*scale,False),('title',title_size,True),('subtitle',body,False)):
        artist=spec[name];text=_wrapped(artist._alpr_full_text,width*(.62 if name=='context' and spec['legend_inline'] else .96),size,bold=bold)
        artist.set(text=text,fontsize=size,position=(.02,y/height))
        y-=len(text.splitlines())*size*1.23+5
    legend=spec['legend']
    legend.set_bbox_to_anchor((.98,(height-10)/height) if spec['legend_inline'] else (.25,(y-2)/height),transform=fig.transFigure)
    for text in legend.get_texts():text.set_fontsize(small)
    top=(y-(6 if spec['legend_inline'] else 24))/height
    note=spec['note'];note_text=_wrapped(note._alpr_full_text,width*.96,small)
    note.set(text=note_text,fontsize=small,position=(.02,8/height))
    axis_title=_wrapped(spec['x_label'],width*.38,body)
    bottom=(8+len(note_text.splitlines())*small*1.23+7+len(axis_title.splitlines())*body*1.23+18)/height
    line_counts=[len(_wrapped(full,width*(.70 if missing else .33),body).splitlines()) for _,full,missing in spec['annotations']]
    required=(len(spec['rows'])+.1)*(max(line_counts,default=1)*body*1.23+8)
    available=(top-bottom)*height
    if not preview and available+1<required:
        fig.set_size_inches(width/72,(height+required-available+1)/72)
        return fit_figure(fig,preview=False)
    ax.set_xlabel(axis_title,fontsize=body,labelpad=6)
    ax.set_position([.25,bottom,.38,max(.12,top-bottom)])
    ax.tick_params(axis='both',labelsize=small)
    ax.set_yticklabels([_wrapped(full,width*.22,body) for full in spec['rows']],fontsize=body)
    # Values use their own column, never extra percentages on the plot axis.
    for text,full,missing in spec['annotations']:
        text.set_text(_wrapped(full,width*(.70 if missing else .33),body))
        text.set_fontsize(body)
    fig._alpr_preview=preview


def make_figure(cohort,metric_id,scenario):
    if cohort.question == 'F5A':
        return _f5a_figure(cohort, metric_id, scenario)
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter,MaxNLocator
    from matplotlib.transforms import blended_transform_factory
    if metric_id not in METRICS:raise ValueError('Nieznana miara.')
    sessions=ordered_sessions(cohort,scenario)
    if not sessions:raise ValueError('Brak sesji w tym scenariuszu.')
    copy=LABELS[metric_id];values=[s.metrics[metric_id].value for s in sessions]
    if any(v is not None and (not math.isfinite(v) or v<0) for v in values):raise ValueError('Nieprawidłowa wartość miary do wykresu.')
    if metric_id in BOUNDED and any(v is not None and v>100 for v in values):raise ValueError('Wartość udziału wykracza poza 0–100%.')
    fig=Figure(figsize=(14/2.54,max(4.7,2.25+.5*len(sessions))),facecolor='white')
    ax=fig.subplots();ax.set_facecolor('white');ax.set_ylim(len(sessions)-.5,-.6)
    maximum=max((v for v in values if v is not None),default=1)
    ax.set_xlim(0,100 if metric_id in BOUNDED else max(1,maximum*1.06))
    if metric_id in BOUNDED:ax.set_xticks([0,25,50,75,100])
    else:ax.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=copy['display_type']=='count'))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:number(x,0 if maximum>=10 else 1,trim=True)))
    rows=[session_label(s) for s in sessions];ax.set_yticks(range(len(sessions)),rows)
    ax.tick_params(axis='y',length=0,pad=6)
    ax.grid(axis='x',color='#d3d8dd',linewidth=.55);ax.set_axisbelow(True)
    for name in ('top','right','left'):ax.spines[name].set_visible(False)
    ax.spines['bottom'].set_color('#72808e')
    annotations=[]
    transform=blended_transform_factory(fig.transFigure,ax.transData)
    for i,s in enumerate(sessions):
        m=s.metrics[metric_id];style=STYLES.get(variant(s),dict(color='#65717c',hatch='..'))
        missing=m.value is None
        if not missing:
            ax.barh(i,m.value,height=.44,color=style['color'],hatch=style['hatch'],edgecolor='#303b44',linewidth=.45)
        text=bar_label(m)
        artist=ax.text(.26 if missing else .655,i,text,transform=transform,va='center',ha='left',
            fontsize=8.2,color='#72551e' if missing else '#1f2933',clip_on=False)
        annotations.append((artist,text,missing))
    def text(full,**options):
        artist=fig.text(.02,1,full,ha='left',va='top',fontfamily='DejaVu Sans',color='#1f2933',**options)
        artist._alpr_full_text=full;return artist
    context=text(context_title(cohort,scenario,sessions))
    title=text(copy['title'],fontweight='bold')
    subtitle=text(copy['subtitle'])
    note=text(copy['footnote']);note.set_verticalalignment('bottom')
    variants=list(dict.fromkeys(variant(s) for s in sessions))
    legend=fig.legend(handles=[Patch(label=v,facecolor=STYLES.get(v,dict(color='#65717c'))['color'],
        hatch=STYLES.get(v,dict(hatch='..'))['hatch'],edgecolor='#303b44',linewidth=.45) for v in variants],
        loc='upper right' if len(variants)<=2 else 'upper left',ncols=min(3,len(variants)),frameon=False,borderaxespad=0,borderpad=0,handlelength=1.5,columnspacing=1.2)
    fig._alpr_layout=dict(context=context,title=title,subtitle=subtitle,note=note,legend=legend,
        rows=rows,annotations=annotations,x_label=copy['x_axis'],legend_inline=len(variants)<=2)
    from ..ranking.mobile_mt_object_evaluation import POLICY_ID,POLICY_SHA256
    fig._alpr_analysis=dict(metric_id=metric_id,scenario=scenario,protocol=PROTOCOL_ID,
        sources=[s.input.source_sha256 for s in sessions],session_ids=[s.input.session_id for s in sessions],
        values=values,protocol_files=cohort.protocol_files,matching_policy_id=POLICY_ID,
        matching_policy_sha256=POLICY_SHA256,gt_mode='ASSISTED',chronology='RETROSPECTIVE_FOR_F4_NOT_PREREGISTERED',
        source_unit=METRICS[metric_id][1],render_unit=METRICS[metric_id][1],publication_width_cm=14,
        title=copy['title'],subtitle=copy['subtitle'],axis_label=copy['x_axis'],footnote=copy['footnote'],
        rows=[dict(session_id=s.input.session_id,source_archive_sha256=s.input.source_sha256,
            label=session_label(s),annotation=bar_label(s.metrics[metric_id]),metric=s.metrics[metric_id].payload()) for s in sessions])
    fit_figure(fig)
    return fig


def _f5a_figure(cohort, metric_id, scenario):
    """Three variant rows; unavailable measurements stay in diagnostics."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from matplotlib.transforms import blended_transform_factory
    from .mobile_analysis import metric_definition
    from .mobile_analysis_f5a import ALGORITHM_ID, aggregate
    from .mobile_analysis_presentation import metric_labels
    rows = aggregate(cohort, metric_id, scenario)
    copy = metric_labels(metric_id, 'F5A')
    fig = Figure(figsize=(14 / 2.54, 5.7), facecolor='white')
    ax = fig.subplots()
    values = [r['value'] for r in rows]
    ax.set_ylim(2.6, -.6)
    ax.set_xlim(0, max(1, max((v for v in values if v is not None), default=1) * 1.06))
    ax.set_yticks(range(3), ['R0', 'R1', 'R2'])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=copy['display_type'] == 'count'))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: number(x, 0, trim=True)))
    ax.grid(axis='x', color='#d3d8dd', linewidth=.55)
    ax.set_axisbelow(True)
    for name in ('top', 'right', 'left'):
        ax.spines[name].set_visible(False)
    annotations = []
    transform = blended_transform_factory(fig.transFigure, ax.transData)
    missing = []
    for i, row in enumerate(rows):
        if row['value'] is None:
            reasons = row['missing_reasons']
            missing.append(row['variant'] + ': ' + ('MP wyłączony' if 'MP_DISABLED' in reasons else 'N/A; powód w tabeli źródeł'))
            continue
        style = STYLES[row['variant']]
        ax.barh(i, row['value'], height=.44, color=style['color'], hatch=style['hatch'], edgecolor='#303b44', linewidth=.45)
        unit = ' ms' if metric_definition(metric_id)[1] == 'ms' else ''
        label = number(row['value'], 1 if unit else 0, trim=bool(unit)) + unit
        label += f"\n{row['n_sessions']}/{row['n_sessions_selected']} sesji · n={row['n_source_observations']}/{row['n_source_observations_total']} śladów"
        if row['status'] == 'SUBSET' and metric_id.endswith('_calls'):
            label += '\nsuma częściowa'
        artist = ax.text(.655, i, label, transform=transform, va='center', ha='left', color='#1f2933', clip_on=False)
        annotations.append((artist, label, False))
    def text(full):
        artist = fig.text(.02, 1, full, ha='left', va='top', fontfamily='DejaVu Sans', color='#1f2933')
        artist._alpr_full_text = full
        return artist
    note = copy['footnote'] + ('\n' + '; '.join(missing) if missing else '')
    note += '\nŹródło: traces.csv z archiwów Android; SHA i mianowniki w CSV/JSON.'
    context = text('F5A · MP / ROI · ' + scenario)
    title = text(copy['title'])
    subtitle = text(copy['subtitle'])
    foot = text(note)
    foot.set_verticalalignment('bottom')
    legend = fig.legend(handles=[Patch(label=v, facecolor=STYLES[v]['color'], hatch=STYLES[v]['hatch'], edgecolor='#303b44')
                                for v in ('R0', 'R1', 'R2')], loc='upper left', ncols=3, frameon=False)
    fig._alpr_layout = dict(context=context, title=title, subtitle=subtitle, note=foot, legend=legend,
        rows=['R0', 'R1', 'R2'], annotations=annotations, x_label=copy['x_axis'], legend_inline=False)
    fig._alpr_analysis = dict(metric_id=metric_id, scenario=scenario, protocol=ALGORITHM_ID,
        gt_mode='NOT_APPLICABLE_TIMING', chronology='RETROSPECTIVE_DESCRIPTIVE_PERFORMANCE',
        qualification=cohort.performance.get('status'), values=values, rows=rows,
        sources=[s.input.source_sha256 for s in cohort.sessions], session_ids=[s.input.session_id for s in cohort.sessions],
        source_unit=metric_definition(metric_id)[1], render_unit=metric_definition(metric_id)[1],
        publication_width_cm=14, title=copy['title'], subtitle=copy['subtitle'], footnote=note)
    fit_figure(fig)
    return fig
