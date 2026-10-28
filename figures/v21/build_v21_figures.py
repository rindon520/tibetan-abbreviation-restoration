"""Reproduce V21 figures 2-6 from public aggregate statistics only.

Figure 1 is supplied with the author manuscript and is not redistributed.
This source-only entry point needs neither private experiment folders nor models.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
from PIL import Image

W=80/25.4
NAVY='#22304A';BLUE='#2F6F9F';TEAL='#2A9D8F';ORANGE='#C76621'
INK='#1D2730';GRAY='#8B969E';MID='#65727C';LIGHT='#E7ECEF';GRID='#D9E0E4'
BLUE_LT='#EAF2F7';TEAL_LT='#E8F3F2';ORANGE_LT='#F8EEE8'
TEAL_TEXT='#1D746A'
REPORT=[];SOURCES=set()


def configure(out):
    fonts={f.name for f in font_manager.fontManager.ttflist}
    serif=next((f for f in ['Noto Serif SC','SimSun','Microsoft YaHei'] if f in fonts),None)
    sans=next((f for f in ['Noto Sans SC','Microsoft YaHei','SimHei'] if f in fonts),None)
    if serif is None or sans is None:
        raise RuntimeError('Install a Chinese font: Noto Serif SC and Noto Sans SC, or Microsoft YaHei.')
    latin='Times New Roman' if 'Times New Roman' in fonts else 'DejaVu Serif'
    # FreeType may select a variable CJK font's thin default instance. Pin
    # weight 400 in a private work directory, preserving the original family.
    def pinned_family(family):
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont
        source=Path(font_manager.findfont(family));font=TTFont(source)
        if 'fvar' not in font:return family
        axes={axis.axisTag for axis in font['fvar'].axes}
        if 'wght' not in axes:return family
        root=out/'work'/'figure_fonts';root.mkdir(parents=True,exist_ok=True)
        target=root/(family.replace(' ','_')+'_Regular.ttf')
        name='V21 '+family
        if not target.exists():
            font=instantiateVariableFont(font,{'wght':400},inplace=True)
            for record in font['name'].names:
                if record.nameID in [1,16]:value=name
                elif record.nameID in [2,17]:value='Regular'
                elif record.nameID==4:value=name+' Regular'
                elif record.nameID==6:value=name.replace(' ','')+'-Regular'
                else:continue
                record.string=value.encode(record.getEncoding())
            font.save(target)
        font_manager.fontManager.addfont(str(target));return name
    serif=pinned_family(serif);sans=pinned_family(sans)
    plt.rcParams.update({'font.family':[latin,serif], 'font.size':7.5,
        'axes.titlesize':8,'axes.labelsize':7.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'legend.fontsize':7.5,'axes.linewidth':.65,'axes.unicode_minus':False,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','figure.facecolor':'white',
        'savefig.facecolor':'white','hatch.linewidth':.35,'text.color':INK,'axes.labelcolor':INK,
        'mathtext.fontset':'custom','mathtext.rm':latin,
        'mathtext.it':latin+':italic','mathtext.bf':latin+':bold'})
    return serif,sans,latin


def read_json(path):
    SOURCES.add(path.resolve());return json.loads(path.read_text(encoding='utf-8'))


def write_csv(out,name,rows):
    with (out/'source_data'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def text(ax,x,y,value,**kw):
    return ax.text(x,y,value,transform=ax.transAxes,**kw)


def box(ax,x,y,w,h,title,subtitle='',edge=GRAY,face='white',dark=False):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.003,rounding_size=0.015',
        transform=ax.transAxes,fc=face,ec=edge,lw=.8)
    ax.add_patch(patch)
    text(ax,x+w/2,y+h*(.73 if dark and subtitle else .68 if subtitle else .5),title,ha='center',va='center',
         fontsize=7.8,color='white' if dark else edge if edge in [BLUE,TEAL,ORANGE] else NAVY,weight='bold',fontfamily=getattr(ax,'diagram_font',None))
    if subtitle:
        text(ax,x+w/2,y+h*.28,subtitle,ha='center',va='center',fontsize=7.5,
             color='#EFF3F8' if dark else MID,linespacing=1.12,fontfamily=getattr(ax,'diagram_font',None))
    return patch


def arrow(ax,start,end,color=INK,dashed=False,curve=0):
    ax.add_patch(FancyArrowPatch(start,end,transform=ax.transAxes,arrowstyle='-|>',
        mutation_scale=7.0,color=color,lw=.8,linestyle='--' if dashed else '-',
        connectionstyle=f'arc3,rad={curve}',shrinkA=1,shrinkB=1))


def clean(ax,grid=None):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2,width=.6,pad=2)
    if grid:
        ax.set_axisbelow(True);ax.grid(axis=grid,color=GRID,lw=.45,linestyle=(0,(2,2)))


def heading(fig,y,label,title):
    fig.text(.035,y,label,fontsize=8.2,weight='bold',va='bottom')
    fig.text(.10,y,title,fontsize=7.8,va='bottom')


def save(fig,out,name,min_font=7.5,preview_only=False):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');fig.canvas.draw()
    issues=[str(w.message) for w in caught if 'missing from font' in str(w.message)]
    renderer=fig.canvas.get_renderer();bounds=fig.bbox;sizes=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():continue
        sizes.append(t.get_fontsize());b=t.get_window_extent(renderer)
        if b.x0<bounds.x0-1 or b.y0<bounds.y0-1 or b.x1>bounds.x1+1 or b.y1>bounds.y1+1:
            issues.append(f'Outside canvas: {t.get_text()!r}')
    if min(sizes)<min_font-.001:issues.append('Font smaller than permitted minimum')
    # Include secondary axes: their complementary tick labels need the same QA.
    for ax in list(fig.axes)+[child for parent in fig.axes for child in parent.child_axes]:
        for axis in [ax.xaxis,ax.yaxis]:
            ts=[t for t in axis.get_ticklabels()if t.get_visible()and t.get_text()]
            bs=[t.get_window_extent(renderer)for t in ts]
            for a,b in zip(bs,bs[1:]):
                if a.overlaps(b):issues.append('Adjacent tick labels overlap')
    REPORT.append(dict(figure=name,width_mm=fig.get_figwidth()*25.4,height_mm=fig.get_figheight()*25.4,
        minimum_font_pt=min(sizes),issues=sorted(set(issues))))
    if issues:raise RuntimeError(name+': '+'; '.join(sorted(set(issues))))
    fig.savefig(out/'previews'/f'{name}.png',dpi=220)
    with Image.open(out/'previews'/f'{name}.png')as im:
        im.convert('L').save(out/'previews'/f'{name}_grayscale.png',dpi=(220,220))
    if not preview_only:
        for ext in ['pdf','svg']:fig.savefig(out/f'{name}.{ext}')
        fig.savefig(out/f'{name}.png',dpi=600)
    plt.close(fig)


def data_figure(out,manifest,sans,preview_only):
    fig=plt.figure(figsize=(W,70.75/25.4))
    ax=fig.add_axes([.02,.02,.96,.96]);ax.axis('off');ax.diagram_font=sans
    text(ax,.015,.965,'(a) 平行行对、片段与联结分组',fontsize=8,weight='bold',fontfamily=sans,va='top',color=NAVY)
    entries=[('原始行对',f"{manifest['raw_line_pairs']:,}",.795,GRAY,BLUE_LT),
             ('数量匹配筛选',f"{manifest['raw_line_pairs']-manifest['rejected_line_pairs']:,}",.665,TEAL,'white'),
             ('片段化',f"{manifest['trusted_segments']:,}",.535,BLUE,'white')]
    for title,count,y,edge,face in entries:
        ax.add_patch(FancyBboxPatch((.015,y),.58,.087,boxstyle='round,pad=0.003,rounding_size=0.015',
            transform=ax.transAxes,fc=face,ec=edge,lw=.7))
        text(ax,.038,y+.0435,title,ha='left',va='center',fontsize=7.5,fontfamily=sans,color=TEAL if edge==TEAL else BLUE if edge==BLUE else NAVY)
        text(ax,.57,y+.0435,count,ha='right',va='center',fontsize=8,weight='bold',color=NAVY)
    for y in [.783,.653]:arrow(ax,(.065,y),(.065,y-.022),color=GRAY)
    text(ax,.105,.774,f"排除 {manifest['rejected_line_pairs']:,} 行对",fontsize=7.5,color=ORANGE,fontfamily=sans,va='center')
    text(ax,.635,.695,'按句末符号切分；\n片段数量一致。\n同父行、相同\nNFC 源形式联结。',fontsize=7.5,fontfamily=sans,color=MID,va='top',linespacing=1.2)
    arrow(ax,(.065,.525),(.065,.502),color=GRAY)
    box(ax,.015,.365,.96,.132,f"联结组冻结划分：{manifest['connected_groups']:,} 组",
        '训练 / 开发 / 校准 / 测试',edge=NAVY,face=NAVY,dark=True)
    text(ax,.015,.325,'(b) 四个划分的规模与用途',fontsize=8,weight='bold',fontfamily=sans,va='center',color=NAVY)
    widths=[.17,.21,.22,.36];xstarts=np.cumsum([.015]+widths[:-1]).tolist()
    ax.add_patch(Rectangle((.015,.250),.96,.048,transform=ax.transAxes,fc=NAVY,ec='none'))
    for x,w,label in zip(xstarts,widths,['划分','片段数','组数','用途']):
        text(ax,x+w/2,.274,label,ha='center',va='center',fontsize=7.5,color='white',fontfamily=sans)
    rows=[('train','训练','资源与权重',TEAL),('dev','开发','分支选参',BLUE),('calibration','校准','重排序拟合',ORANGE),('test','测试','冻结评估',NAVY)]
    csv_rows=[]
    for i,(key,label,role,color) in enumerate(rows):
        y=.221-i*.044;spec=manifest['splits'][key]
        vals=[label,f"{spec['samples']:,}",f"{spec['groups']:,}",role]
        for x,w,val in zip(xstarts,widths,vals):
            text(ax,x+w/2,y,val,ha='center',va='center',fontsize=7.5,color=color if val==label else NAVY,fontfamily=sans if val in [label,role]else plt.rcParams['font.family'][0])
        ax.plot([.015,.975],[y-.023,y-.023],transform=ax.transAxes,color=GRID,lw=.45)
        csv_rows.append(dict(split=key,samples=spec['samples'],groups=spec['groups'],purpose=role))
    text(ax,.015,.010,'跨划分父行 / NFC 源文本重叠均为 0。',fontsize=7.5,color=MID,fontfamily=sans,va='bottom')
    write_csv(out,'fig2_splits.csv',csv_rows)
    save(fig,out,'fig2_data_protocol',preview_only=preview_only)


def method_figure(out,sans,preview_only):
    fig=plt.figure(figsize=(W,86.58/25.4));ax=fig.add_axes([.02,.02,.96,.96]);ax.axis('off');ax.diagram_font=sans
    text(ax,.015,.985,'(a) 证据学习与独立校准',fontsize=8,weight='bold',fontfamily=sans,va='top',color=NAVY)
    steps=[('语料划分','分组冻结',GRAY,BLUE_LT),('对齐监督','训练集',GRAY,BLUE_LT),('双路学习','开发选参',TEAL,TEAL_LT),('OOF 校准','独立校准',ORANGE,ORANGE_LT)]
    xs=[.015,.26,.505,.75]
    for x,(title,sub,edge,face)in zip(xs,steps):box(ax,x,.838,.215,.09,title,sub,edge,face)
    for i in range(3):arrow(ax,(xs[i]+.22,.883),(xs[i+1]-.003,.883),color=GRAY)
    ax.plot([.015,.97],[.788,.788],transform=ax.transAxes,color=GRID,lw=.6)
    text(ax,.015,.748,'(b) 冻结推理：生成、处理与重排序',fontsize=8,weight='bold',fontfamily=sans,va='center',color=NAVY)
    box(ax,.25,.638,.49,.073,'缩写输入 X',edge=GRAY,face='#F4F6F8')
    arrow(ax,(.495,.635),(.245,.570),color=NAVY)
    arrow(ax,(.495,.635),(.74,.570),color=NAVY)
    box(ax,.015,.412,.455,.155,'上下文感知转写','三元 → 左邻 → 右邻\n证据不足 → 全局回退',TEAL,TEAL_LT)
    box(ax,.51,.412,.455,.155,'非自回归编辑标注','COPY / DELETE / 展开\n低置信位置 → 规则回退',BLUE,BLUE_LT)
    arrow(ax,(.245,.410),(.245,.377),color=TEAL)
    arrow(ax,(.74,.410),(.74,.377),color=BLUE)
    box(ax,.015,.309,.455,.065,'句末＋词表重切分',edge=TEAL,face='white')
    box(ax,.51,.309,.455,.065,'句末＋词表重切分',edge=BLUE,face='white')
    arrow(ax,(.245,.306),(.40,.258),color=TEAL)
    arrow(ax,(.74,.306),(.59,.258),color=BLUE)
    box(ax,.17,.160,.65,.095,'候选重排序：完整候选选择','28 维特征 · 标准化逻辑回归',ORANGE,ORANGE_LT)
    ax.plot([.86,.988,.988,.86],[.835,.805,.211,.211],transform=ax.transAxes,color=ORANGE,lw=.8,ls='--')
    arrow(ax,(.86,.211),(.823,.211),color=ORANGE,dashed=True)
    text(ax,.885,.705,'仅传参数',ha='center',fontsize=7.5,color=ORANGE,fontfamily=sans)
    arrow(ax,(.495,.158),(.495,.124),color=ORANGE)
    box(ax,.11,.018,.77,.102,'规范还原输出','输出片段 · 候选来源与选择分数',NAVY,NAVY,dark=True)
    save(fig,out,'fig3_method_framework',preview_only=preview_only)


def comparison_figure(out,result,external,preview_only):
    m=result['test_metrics'];effect=result['selector_effect'];fig=plt.figure(figsize=(W,117.32/25.4))
    heading(fig,.958,'a','方法比较（CA = 100 − CER）')
    # Match Fig. 3's branch colors. Distinct marker shapes survive grayscale.
    methods=[('直接复制',m['identity'],GRAY,'x'),('字符转写',m['global_transducer'],GRAY,'o'),
        ('＋句末修正',m['global_plus_shad'],GRAY,'s'),('＋词表处理',m['global_plus_shad_vocab'],GRAY,'v'),
        ('上下文转写',m['contextual'],TEAL,'o'),('编辑标注器',m['edit_tagger'],BLUE,'^'),
        ('候选重排序',m['hybrid'],ORANGE,'D'),('TibNorm†',external['test_metrics'],GRAY,'^')]
    ax=fig.add_axes([.345,.680,.60,.205]);ys=np.arange(len(methods))[::-1]
    for y,(label,metric,color,marker)in zip(ys,methods):
        value=metric['micro_cer']*100
        if marker=='x':
            ax.scatter(value,y,s=17,color=MID,marker=marker,linewidths=.8)
        else:
            hollow=color==GRAY
            ax.scatter(value,y,s=19 if color==ORANGE else 18,marker=marker,
                facecolors='white' if hollow else color,
                edgecolors=MID if hollow else 'white',linewidths=.65 if hollow else .35)
        ax.text(value+.6,y,f'{value:.2f}',va='center',fontsize=7.0)
    ax.set_yticks(ys,[x[0]for x in methods]);ax.set_ylim(-.5,len(methods)-.5);ax.set_xlim(0,42);ax.set_xticks([0,20,40]);ax.tick_params(axis='y',length=0);clean(ax,'x')
    ax.set_xlabel('字符错误率 CER（%）↓',labelpad=1.5,fontsize=7.0)
    # This is a complementary scale of the same points, not a second outcome.
    acc=ax.secondary_xaxis('top',functions=(lambda cer:100-cer,lambda ca:100-ca))
    acc.set_xticks([100,80,60]);acc.set_xlabel('字符编辑准确率 CA（%）↑',labelpad=1.5,fontsize=7.0)
    acc.tick_params(length=2,width=.6,pad=2,labelsize=7.0)
    heading(fig,.591,'b','配对结局：改善 / 变差 / 不变')
    axb=fig.add_axes([.095,.508,.85,.048]);vals=[effect['selected_better'],effect['selected_worse'],effect['test_samples']-effect['selected_better']-effect['selected_worse']]
    left=0
    for i,(v,color)in enumerate(zip(vals,[TEAL,ORANGE,LIGHT])):
        w=v/effect['test_samples']*100
        axb.barh(0,w,left=left,height=.6,color=color,
            edgecolor=INK if i==1 else 'white',lw=.35 if i==1 else .5,
            hatch='//' if i==1 else None)
        if i!=1:axb.text(left+w/2,0,str(v),ha='center',va='center',fontsize=7.5,color=INK)
        else:
            axb.annotate(str(v),xy=(left+w/2,.3),xytext=(left+w/2,.85),ha='center',va='bottom',fontsize=7.5,color=ORANGE,arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.6),annotation_clip=False)
        left+=w
    axb.set_xlim(0,100);axb.set_ylim(-.6,.6);axb.set_xticks([0,50,100]);axb.set_yticks([]);clean(axb)
    heading(fig,.410,'c','最终候选来源（%）')
    axc=fig.add_axes([.095,.316,.85,.068]);primary=effect['test_samples']-effect['selected_auxiliary'];aux=effect['selected_auxiliary']
    a=primary/effect['test_samples']*100
    axc.barh(0,a,height=.75,color=TEAL,edgecolor='white',lw=.5)
    axc.barh(0,100-a,left=a,height=.75,color=BLUE_LT,edgecolor=BLUE,lw=.4,hatch='//')
    axc.text(a/2,0,f'上下文 {primary}',fontsize=7.5,ha='center',va='center',color=INK)
    axc.text(a+(100-a)/2,0,f'编辑\n{aux}',fontsize=7.5,ha='center',va='center',linespacing=1.0,
        bbox=dict(facecolor=BLUE_LT,edgecolor='none',boxstyle='square,pad=0.05'))
    axc.set_xlim(0,100);axc.set_ylim(-.5,.5);axc.set_xticks([0,50,100]);axc.set_yticks([]);clean(axc)
    heading(fig,.232,'d','CER 绝对降幅（百分点）')
    axd=fig.add_axes([.19,.119,.755,.080]);value=effect['absolute_cer_reduction']*100;low,high=np.array(effect['absolute_cer_reduction_ci95'])*100
    axd.axvline(0,color=INK,lw=.6);axd.errorbar(value,0,xerr=[[value-low],[high-value]],fmt='D',color=ORANGE,elinewidth=1,markersize=4,capsize=2.5)
    axd.text(value,.75,f'{value:.2f} [{low:.2f}, {high:.2f}]',transform=axd.get_xaxis_transform(),ha='center',fontsize=7.5)
    axd.set_xlim(-.05,1.0);axd.set_xticks([0,.5,1]);axd.set_yticks([]);clean(axd,'x')
    fig.text(.04,.071,'(a) 点旁为CER；精选8种，完整见表3。',fontsize=7.0,color=INK)
    fig.text(.04,.042,f"95% 区间：{effect['bootstrap_groups']} 联结组配对自助 5,000 次。",fontsize=7.0,color=INK)
    fig.text(.04,.014,f"† 外部规则库；区间不含训练随机性；{effect['test_samples']} 段。",fontsize=7.0,color=INK)
    write_csv(out,'fig4_comparison.csv',[dict(method=name,cer=v['micro_cer'],
        character_accuracy=1-v['micro_cer'])for name,v,c,marker in methods])
    write_csv(out,'fig4_outcomes.csv',[dict(outcome=k,count=v)for k,v in zip(['改善','变差','不变'],vals)])
    write_csv(out,'fig4_sources.csv',[dict(source='contextual',count=primary),dict(source='edit',count=aux)])
    write_csv(out,'fig4_effect.csv',[dict(effect_pp=value,ci_low_pp=low,ci_high_pp=high,bootstrap_groups=effect['bootstrap_groups'],iterations=effect['bootstrap_iterations'])])
    save(fig,out,'fig4_main_results',min_font=7,preview_only=preview_only)


def calibration_figure(out,summary,preview_only):
    rows=[r for r in summary['tuning']if r['C']==summary['best_C']]
    rows.sort(key=lambda r:r['threshold']);x=np.array([r['threshold']for r in rows]);chosen=summary['best_threshold'];selected=next(r for r in rows if r['threshold']==chosen)
    fig=plt.figure(figsize=(W,99.82/25.4))
    for bottom,label,title,field,color,ylabel in [(.575,'a','折外字符错误率 CER','calibration_oof_cer',BLUE,'CER（%）'),(.220,'b','折外整段还原准确率 EM','calibration_oof_exact',TEAL,'EM（%）')]:
        heading(fig,bottom+.277,label,title)
        ax=fig.add_axes([.20,bottom,.745,.247]);v=np.array([r[field]for r in rows])*100
        ax.plot(x,v,color=color,marker='o'if label=='a'else's',ms=3,ls='-'if label=='a'else'--',lw=.9)
        ax.axvline(chosen,color=ORANGE,ls='--',lw=.7)
        ax.scatter([chosen],[selected[field]*100],s=30,facecolors='white',edgecolors=ORANGE,lw=.9,zorder=3)
        ax.annotate(f"{selected[field]*100:.2f}%",xy=(chosen,selected[field]*100),xytext=(3,5 if label=='a'else 5),textcoords='offset points',fontsize=7.5,color=ORANGE)
        padding=.12 if label=='a'else .16;ax.set_ylim(v.min()-padding,v.max()+padding);ax.set_xlim(.335,.715)
        ax.set_xticks(x,[f'{val:.2f}'for val in x]);ax.tick_params(axis='x',labelsize=7.5)
        ax.set_ylabel(ylabel);clean(ax,'y')
        # Explicit axis-break markers: local y range, not a baseline-zero magnitude chart.
        ax.plot([-.013,.013],[-.015,.015],transform=ax.transAxes,color=INK,lw=.7,clip_on=False)
        ax.plot([-.013,.013],[.012,.042],transform=ax.transAxes,color=INK,lw=.7,clip_on=False)
        if label=='b':ax.set_xlabel('候选选择阈值 τ',labelpad=2)
    fig.text(.04,.107,f"独立校准 {summary['calibration_samples']} 段 / {summary['calibration_group_count']} 组；分组五折。",fontsize=7.5,color=INK)
    fig.text(.04,.072,f"固定 C = {summary['best_C']}；橙色虚线为校准集选定 τ。",fontsize=7.5,color=INK)
    fig.text(.04,.037,'纵轴为局部范围；连线连接已评估操作点。',fontsize=7.5,color=INK)
    write_csv(out,'fig5_calibration_thresholds.csv',rows)
    save(fig,out,'fig5_calibration_sensitivity',preview_only=preview_only)


def error_figure(out,result,preview_only,qualitative):
    fig=plt.figure(figsize=(W,82.56/25.4));heading(fig,.931,'a','编辑操作：预测 → 参考')
    ax=fig.add_axes([.265,.575,.66,.30]);a=result['test_metrics']['contextual'];b=result['test_metrics']['hybrid']
    fields=[('替换 S','edit_substitutions'),('插入 I','edit_insertions'),('删除 D','edit_deletions')];rows=[]
    for y,(name,key)in zip([2,1,0],fields):
        v1,v2=a[key],b[key];ax.plot([v1,v2],[y,y],color=GRID,lw=2)
        ax.scatter(v1,y,color=TEAL,s=22,edgecolor='white',linewidth=.35,zorder=2)
        ax.scatter(v2,y,marker='D',facecolor='white',edgecolor=ORANGE,s=27,lw=.9,zorder=3)
        ax.text(v1,y+.15,str(v1),ha='center',va='bottom',fontsize=7.5,color=TEAL_TEXT)
        ax.text(v2,y-.15,str(v2),ha='center',va='top',fontsize=7.5,color=ORANGE)
        ax.text(2200,y,f'{v2-v1:+d}',ha='right',va='center',fontsize=7.0,color=ORANGE if v2>v1 else TEAL_TEXT)
        rows.append(dict(operation=name,contextual=v1,reranked=v2,delta=v2-v1,direction='prediction_to_reference'))
    ax.set_xlim(0,2300);ax.set_xticks([0,1000,2000]);ax.set_ylim(-.6,2.6);ax.set_yticks([2,1,0],[r[0]for r in fields]);ax.tick_params(axis='y',length=0);clean(ax,'x')
    fig.legend(handles=[Line2D([0],[0],marker='o',ls='',color=TEAL,label='上下文候选'),Line2D([0],[0],marker='D',ls='',markerfacecolor='white',markeredgecolor=ORANGE,label='候选重排序')],loc='upper center',bbox_to_anchor=(.57,.926),frameon=False,ncol=2,columnspacing=.7,handletextpad=.3,fontsize=7.0,markerscale=.7)
    heading(fig,.490,'b','缩写机制的定性结构关联')
    labels=qualitative['rows']
    cols=[label[:2]+'\n'+label[2:] for label in qualitative['columns']]
    matrix=np.array(qualitative['matrix'],dtype=int)
    if matrix.shape!=(len(labels),len(cols)) or not np.isin(matrix,[0,1]).all():
        raise ValueError('Figure6 qualitative relation matrix must be binary and rectangular.')
    axb=fig.add_axes([.31,.15,.63,.245])
    for row in range(len(labels)):
        for col in range(len(cols)):
            axb.add_patch(Rectangle((col-.46,row-.42),.92,.84,fc='#F7F9FA',ec=LIGHT,lw=.4))
            axb.scatter(col,row,s=20 if matrix[row,col]else 9,marker='s'if matrix[row,col]else'o',color=BLUE if matrix[row,col]else'white',edgecolor='white'if matrix[row,col]else GRID,lw=.5)
    axb.set_xlim(-.5,len(cols)-.5);axb.set_ylim(len(labels)-.5,-.5)
    axb.set_xticks(range(len(cols)),cols);axb.xaxis.tick_top();axb.set_yticks(range(len(labels)),labels)
    axb.tick_params(length=0,pad=3,labelsize=7.5)
    for tick in axb.get_xticklabels()+axb.get_yticklabels():tick.set_linespacing(1.0)
    for spine in axb.spines.values():spine.set_visible(False)
    fig.text(.04,.069,'插入＝补回缺失；删除＝去掉冗余。',fontsize=7.0,color=INK)
    fig.text(.04,.032,'方块仅示结构关联，不代表标注频数。',fontsize=7.0,color=INK)
    write_csv(out,'fig6_edit_operations.csv',rows)
    write_csv(out,'fig6_qualitative_matrix.csv',[dict(mechanism=labels[r],aspect=qualitative['columns'][c],association=int(matrix[r,c]),evidence_type=qualitative['evidence_type'])for r in range(len(labels))for c in range(len(cols))])
    save(fig,out,'fig6_error_analysis',min_font=7,preview_only=preview_only)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'source_data')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--figures',nargs='+',type=int,choices=[2,3,4,5,6],default=[2,3,4,5,6])
    parser.add_argument('--method-only',action='store_true',help='Generate figure 3 without aggregate inputs.')
    parser.add_argument('--preview-only',action='store_true')
    args=parser.parse_args();out=args.output_dir.resolve();data=args.data_dir.resolve()
    out.mkdir(parents=True,exist_ok=True);(out/'source_data').mkdir(exist_ok=True);(out/'previews').mkdir(exist_ok=True)
    serif,sans,latin=configure(out)
    wanted=[3] if args.method_only else args.figures
    aggregates=None if args.method_only else read_json(data/'v21_aggregate.json')
    if 2 in wanted:data_figure(out,aggregates['split'],sans,args.preview_only)
    if 3 in wanted:method_figure(out,sans,args.preview_only)
    if 4 in wanted:comparison_figure(out,aggregates['result'],aggregates['external'],args.preview_only)
    if 5 in wanted:calibration_figure(out,aggregates['selector'],args.preview_only)
    if 6 in wanted:error_figure(out,aggregates['result'],args.preview_only,aggregates['qualitative_fig6'])
    report=dict(chinese_serif=serif,chinese_sans=sans,latin=latin,figure_checks=REPORT,
        updated_raster_dpi=600,figure1_not_redistributed=True,
        sources=[dict(path=f.relative_to(data).as_posix(),sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in sorted(SOURCES)])
    (out/'figure_quality_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
