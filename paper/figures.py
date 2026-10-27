"""Regenerate the two figures whose paper panels required numerical corrections.

Fig1: contextual information improves target-domain ranking, with a strong
single-learner control. Fig4: performance depends on the held-out identity;
matched EasyDesign comparison uses its own 5,995-record cohort.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import ScaledTranslation
from revision_statistics.resampling import bootstrap_metric_draws, percentile_interval

matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
    'legend.fontsize':6.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.6,'savefig.dpi':600})
BLUE='#0072B2';GREY='#858585';PALE='#D1D6DB';ORANGE='#D55E00'
COLORS=[BLUE,'#009E73',ORANGE,'#687680']
PROTOCOLS=['target','guide','atomic_template','dual_atomic']
SHORT=['Target','Guide','Template','Joint']
MODELS=['ridge','svr','lightgbm','D','HISTORY_PRIOR','SEQ_MAPPING_XGB','TriCasRank']
DISPLAY=['Ridge','SVR','LightGBM','D','History prior','Sequence + mapping\nXGBoost','TriCasRank']

def title(ax,letter,text):
    ax.set_label(letter)
    ax.text(0,1,letter,transform=ax.transAxes+ScaledTranslation(-28/72,16/72,ax.figure.dpi_scale_trans),fontsize=9,fontweight='bold',va='baseline')
    ax.set_title(text,loc='left',pad=16,fontweight='bold')

def save(fig,path,axes,row_groups,column_groups=()):
    fig.canvas.draw()
    # Public, dependency-free physical geometry check.
    positions={ax.get_label():[float(v)*72 for v in ax.get_position().bounds*np.array([*fig.get_size_inches(),*fig.get_size_inches()])] for ax in axes}
    for names in row_groups:
        box=np.array([positions[x] for x in names]);assert np.ptp(box[:,1])<1.5 and np.ptp(box[:,3])<1.5
        if len(names)>=3: assert np.ptp(box[:,2])<1.5
    path.with_suffix('.layout.json').write_text(json.dumps({'axes_xywh_pt':positions,'row_groups':row_groups,'column_groups':column_groups,'tolerance_pt':1.5},indent=2))
    # This optional QA dependency is used for the publication audit. It is not
    # needed to run the public reproduction workflow.
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        pass
    else:
        require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=[x.get_label() for x in axes],row_groups=row_groups,column_groups=column_groups,
            require_panel_labels=False,json_out=str(path)+'.alignment.json',strict=True)
    fig.savefig(path.with_suffix('.svg'),facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),facecolor='white')
    fig.savefig(path.with_suffix('.png'),dpi=600,facecolor='white')
    plt.close(fig)

def intervals(data,out,pred):
    path=out/'figure_statistics.csv';drawpath=out/'figure_bootstrap.npz'
    if path.exists() and drawpath.exists(): return pd.read_csv(path),dict(np.load(drawpath))
    rows=[];arrays={}
    for cohort in ('core_dev','common_dev'):
        block=pred[(pred.cohort_id==cohort)&(pred.protocol_id=='target')].copy()
        if cohort=='core_dev':
            for f in sorted((data/'results/classics').glob('pred__*.csv')):
                t=pd.read_csv(f).rename(columns={'true_y':'y_true','pred_y':'y_pred','cluster_id':'group_id','fold':'outer_fold','model':'model_id'})
                block=pd.concat([block,t],ignore_index=True)
        wide=block.pivot(index='record_id',columns='model_id',values='y_pred')
        meta=block.drop_duplicates('record_id').set_index('record_id').reindex(wide.index)
        draws=bootstrap_metric_draws(meta.y_true,{name:wide[name].to_numpy(float) for name in wide},meta.group_id,seed=20261012,draws=2000)
        for model,metrics in draws.items():
            for metric,v in metrics.items():
                rows.append({'cohort_id':cohort,'model_id':model,'metric':metric,**percentile_interval(v),'seed':20261012,'draws':2000})
        if cohort=='core_dev': arrays['delta_scc']=draws['TriCasRank']['SCC']-draws['D']['SCC']
    result=pd.DataFrame(rows);result.to_csv(path,index=False);np.savez_compressed(drawpath,**arrays)
    return result,arrays

def draw(data,output):
    destination=output/'figures';destination.mkdir(parents=True,exist_ok=True)
    source=destination/'source_data';source.mkdir(exist_ok=True)
    p=pd.read_csv(data/'results/fixed/oof_predictions.csv',dtype={'group_id':str})
    m=pd.concat([pd.read_csv(output/'fixed_metrics.csv'),pd.read_csv(output/'classic_metrics.csv')],ignore_index=True)
    ci,draws=intervals(data,destination,p)
    pooled=m[m.evaluation_level.eq('pooled_oof')]
    target=pooled[(pooled.cohort_id=='core_dev')&(pooled.protocol_id=='target')].set_index('model_id').reindex(MODELS)
    target.to_csv(source/'Fig1_model_metrics.csv')
    primary=target.reset_index().rename(columns={'index':'model_id'})
    scc_ci=ci[(ci.cohort_id=='core_dev')&(ci.metric=='SCC')][['model_id','lower','upper']]
    primary.merge(scc_ci,on='model_id',validate='one_to_one').to_csv(output/'tables/Table3_primary_models_with_intervals.csv',index=False)
    fig=plt.figure(figsize=(7.2,8.2))
    upper=fig.add_gridspec(2,2,left=.17,right=.97,bottom=.40,top=.93,wspace=.72,hspace=.82)
    bottom=fig.add_gridspec(1,3,left=.17,right=.97,bottom=.085,top=.285,wspace=.86)
    axes=[fig.add_subplot(upper[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]+[fig.add_subplot(bottom[0,i]) for i in range(3)]
    a,b,c,d,e,f,g=axes
    title(a,'a','Target-fold evaluation');a.axis('off')
    for y,heading,detail in [(.72,'8,417 records','6,825 distinct target groups'),(.40,'Five grouped folds','One out-of-fold prediction per record'),(.08,'Matched comparisons','Same records, labels and group allocation')]:
        a.text(0,y,heading,color=BLUE,fontweight='bold',transform=a.transAxes)
        a.text(0,y-.13,detail,fontsize=6.5,transform=a.transAxes)
    title(b,'b','Ranking with contextual controls')
    for k,model in enumerate(MODELS):
        row=ci[(ci.cohort_id=='core_dev')&(ci.model_id==model)&(ci.metric=='SCC')].iloc[0]
        val=target.loc[model,'SCC'];color=BLUE if model=='TriCasRank' else ORANGE if model=='SEQ_MAPPING_XGB' else GREY
        b.errorbar(val,k,xerr=[[val-row.lower],[row.upper-val]],fmt='o',ms=3,color=color,elinewidth=.8,capsize=2)
    b.set_yticks(range(7),DISPLAY);b.invert_yaxis();b.set_xlim(.48,.87);b.set_xlabel('Spearman correlation (SCC)');b.grid(axis='x',alpha=.15)
    title(c,'c','Five-metric profile')
    values=target[['SCC','PCC','RMSE','MAE','R2']].to_numpy(float)
    from scipy.stats import rankdata
    colors=np.column_stack([(rankdata(values[:,j] if j not in (2,3) else -values[:,j])-1)/6 for j in range(5)])
    im=c.imshow(colors,cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='nearest')
    c.set_yticks(range(7),DISPLAY);c.set_xticks(range(5),['SCC','PCC','RMSE ↓','MAE ↓','R²']);c.tick_params(length=0)
    for i in range(7):
        for j in range(5): c.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=5.7,color='white' if colors[i,j]>.58 else '#222222')
    c.text(0,-.16,'Colour: within-metric rank, light → dark = better',transform=c.transAxes,fontsize=5.7)
    title(d,'d','Consistency across outer folds')
    for model in MODELS:
        row=m[(m.cohort_id=='core_dev')&(m.protocol_id=='target')&(m.model_id==model)&m.outer_fold.notna()].sort_values('outer_fold')
        color=BLUE if model=='TriCasRank' else ORANGE if model=='SEQ_MAPPING_XGB' else GREY
        d.plot(row.outer_fold+1,row.SCC,'o-',color=color,lw=1.2 if model in ('TriCasRank','SEQ_MAPPING_XGB') else .6,ms=2,alpha=1 if model in ('TriCasRank','SEQ_MAPPING_XGB') else .5,label=model)
    d.set_xticks(range(1,6));d.set_xlabel('Outer fold');d.set_ylabel('SCC');d.set_ylim(.45,.89)
    title(e,'e','Paired gain over D')
    delta=draws['delta_scc'];lo,hi=np.quantile(delta,[.025,.975]);point=target.loc['TriCasRank','SCC']-target.loc['D','SCC']
    e.hist(delta,bins=30,color='#ADD8EE',edgecolor='white',lw=.2)
    e.axvline(point,color=BLUE,lw=1);e.axvline(lo,color=BLUE,ls='--',lw=.7);e.axvline(hi,color=BLUE,ls='--',lw=.7)
    e.set_xlabel('ΔSCC: TriCasRank − D');e.set_ylabel('Bootstrap frequency');e.set_xlim(.06,.10);e.set_xticks([.06,.08,.10])
    e.text(0,1.07,f'{point:+.3f} [{lo:.3f}, {hi:.3f}]',transform=e.transAxes,fontsize=6,color=BLUE)
    title(f,'f','Prediction density')
    tri=p[(p.cohort_id=='core_dev')&(p.protocol_id=='target')&(p.model_id=='TriCasRank')]
    f.hexbin(tri.y_true,tri.y_pred,gridsize=45,mincnt=1,cmap='Blues',norm=LogNorm(),linewidths=0,rasterized=True)
    f.plot([-4.2,.3],[-4.2,.3],ls='--',color=GREY,lw=.6);f.set_xlim(-4.2,.3);f.set_ylim(-4.2,.3)
    f.set_xlabel('Observed activity');f.set_ylabel('Predicted activity');f.set_xticks([-4,-2,0]);f.set_yticks([-4,-2,0])
    f.text(0,1.07,'Darker cells = more records',transform=f.transAxes,fontsize=5.7)
    title(g,'g','Activity calibration')
    bins=pd.read_csv(output/'tables/Activity_bins.csv');cal=bins[bins.profile.eq('calibration')]
    g.plot(cal.mean_observed,cal.mean_predicted,'o-',ms=3,color=BLUE,lw=.8)
    g.plot([-3.8,-.8],[-3.8,-.8],ls='--',color=GREY,lw=.6);g.set_xlim(-3.8,-.8);g.set_ylim(-3.8,-.8)
    g.set_xlabel('Mean observed activity');g.set_ylabel('Mean predicted activity');g.set_xticks([-3.5,-2.5,-1.5]);g.set_yticks([-3.5,-2.5,-1.5])
    save(fig,destination/'Fig1_corrected',axes,[['a','b'],['c','d'],['e','f','g']],[['a','c'],['b','d']])
    # Fig4 retains the original a-j reading order. Its paired differences come
    # from the corrected fixed evaluation, not the older template/joint runs.
    core=pooled[pooled.cohort_id.eq('core_dev')]
    cp=(output/'fixed_paired_comparisons.csv')
    paired=pd.read_csv(cp if cp.exists() else data/'results/accepted/fixed/paired_comparisons.csv')
    pair=paired[(paired.cohort_id=='core_dev')&(paired.model_b=='D')].set_index('protocol_id').reindex(PROTOCOLS)
    pair.to_csv(source/'Fig4_paired_changes.csv');core.to_csv(source/'Fig4_protocol_metrics.csv',index=False)
    fig=plt.figure(figsize=(7.2,9.4))
    grid=fig.add_gridspec(4,3,left=.115,right=.975,bottom=.07,top=.93,wspace=.78,hspace=1.02)
    a=fig.add_subplot(grid[0,:2]);b=fig.add_subplot(grid[0,2])
    c,d,e=[fig.add_subplot(grid[1,i]) for i in range(3)]
    f,g,h=[fig.add_subplot(grid[2,i]) for i in range(3)]
    i1,i2,j=[fig.add_subplot(grid[3,i]) for i in range(3)]
    title(a,'a','Separate identity-isolation protocols');a.axis('off')
    for idx,(name,count) in enumerate(zip(SHORT,[6825,1341,21,20])):
        x=.015+idx*.25
        a.add_patch(FancyBboxPatch((x,.35),.21,.34,boxstyle='round,pad=.012',transform=a.transAxes,fc='#F5F8FA',ec=COLORS[idx],lw=.8))
        a.text(x+.105,.52,name,ha='center',va='center',transform=a.transAxes,fontsize=6.8)
        a.text(x+.105,.19,f'{count:,} groups',ha='center',transform=a.transAxes,fontsize=6)
    a.text(.0,.91,'The same 8,417 records; a different held-out identity',transform=a.transAxes,fontsize=6.5,color=GREY)
    counts=np.array([6825,1341,21,20]);assert np.all(counts>0)
    title(b,'b','Available isolation groups');b.barh(range(4),np.log10(counts),color=COLORS,height=.55)
    b.set_yticks(range(4),SHORT);b.invert_yaxis();b.set_xlim(0,4.9);b.set_xticks([0,2,4]);b.set_xlabel('log10(group count)')
    for y,n in enumerate([6825,1341,21,20]): b.text(np.log10(n)+.12,y,f'{n:,}',va='center',fontsize=6)
    for ax,letter,metric,heading in [(c,'c','SCC','Rank correlation'),(d,'d','RMSE','Numerical error')]:
        title(ax,letter,heading)
        for model,color,marker,offset in [('D',GREY,'s',-.08),('TriCasRank',BLUE,'o',.08)]:
            values=core[core.model_id.eq(model)].set_index('protocol_id').reindex(PROTOCOLS)[metric]
            ax.plot(values,np.arange(4)+offset,marker,ls='none',color=color,ms=3,label=model)
        ax.set_yticks(range(4),SHORT);ax.invert_yaxis();ax.set_xlabel(metric);ax.grid(axis='x',alpha=.15)
    c.set_xlim(.5,.9);d.set_xlim(.3,.6);d.legend(loc='upper left',bbox_to_anchor=(-.05,-.36),ncol=2,handletextpad=.3,columnspacing=.9,frameon=False)
    for ax,letter,metric,heading in [(e,'e','SCC','Paired ranking change'),(f,'f','MAE','Paired error change')]:
        title(ax,letter,heading)
        val=pair[f'delta_{metric}'].to_numpy();lo=pair[f'delta_{metric}_ci_low'].to_numpy();hi=pair[f'delta_{metric}_ci_high'].to_numpy()
        ax.errorbar(val,range(4),xerr=[val-lo,hi-val],fmt='o',color=BLUE,ms=3,capsize=2,lw=.8)
        ax.axvline(0,color=GREY,lw=.7);ax.set_yticks(range(4),SHORT);ax.invert_yaxis();ax.set_xlabel('TriCasRank − D · '+metric);ax.grid(axis='x',alpha=.15)
    title(g,'g','Calibration by protocol')
    for protocol,name,color in zip(PROTOCOLS,SHORT,COLORS):
        cal=bins[(bins.profile=='identity_protocol_calibration')&(bins.protocol==protocol)]
        g.plot(cal.mean_predicted,cal.mean_observed,'o-',ms=2,lw=.7,color=color,label=name)
    g.plot([-3.7,-1.7],[-3.7,-1.7],ls='--',color=GREY,lw=.6);g.set_xlabel('Mean predicted activity');g.set_ylabel('Mean observed activity')
    g.legend(loc='upper left',bbox_to_anchor=(-.02,-.36),ncol=2,handlelength=1,columnspacing=.8,frameon=False)
    title(h,'h','Within-guide ranking')
    guides=pd.read_csv(output/'tables/within_guide_values.csv');v=[guides[(guides.model==model)&guides.SCC.notna()].SCC.to_numpy() for model in ['D','TriCasRank']]
    bp=h.boxplot(v,labels=['D','TriCasRank'],patch_artist=True,widths=.55,showfliers=True,flierprops={'markersize':1,'markeredgewidth':.3},medianprops={'color':'black','linewidth':.8})
    for box,color in zip(bp['boxes'],[GREY,BLUE]): box.set_facecolor(color);box.set_alpha(.65)
    h.set_ylim(-1.08,1.08);h.set_yticks([-1,0,1]);h.set_ylabel('Within-guide SCC')
    h.text(0,1.08,f'{len(v[0]):,} eligible guides; ≥5 records',transform=h.transAxes,fontsize=5.8)
    title(i1,'i','Matched cohort: ranking');title(i2,'i2','Matched cohort: error')
    for ax,metric in [(i1,'SCC'),(i2,'RMSE')]:
        common=pooled[(pooled.cohort_id=='common_dev')&(pooled.protocol_id=='target')].set_index('model_id')
        for y,model,color in [(0,'D',GREY),(1,'TriCasRank',BLUE),(2,'EasyDesign-adapted',PALE)]:
            row=ci[(ci.cohort_id=='common_dev')&(ci.model_id==model)&(ci.metric==metric)].iloc[0];val=common.loc[model,metric]
            ax.errorbar(val,y,xerr=[[val-row.lower],[row.upper-val]],fmt='o',color=color,ms=3,capsize=2,lw=.8)
        ax.set_yticks(range(3),['D','TriCasRank','EasyDesign-\nadapted'] if ax==i1 else [])
        ax.invert_yaxis();ax.set_xlabel(metric);ax.grid(axis='x',alpha=.15)
    # i2 is a continuation of panel i, not an additional manuscript panel.
    for text in i2.texts: text.set_visible(False)
    i2.set_label('i2')
    title(j,'j','Matched activity strata')
    matched=bins[bins.profile=='matched_context_quintile_error']
    for model,color,label in [('TriCasRank',BLUE,'TriCasRank'),('EasyDesign-adapted',GREY,'EasyDesign-adapted')]:
        sub=matched[matched.model.eq(model)];j.plot(sub.bin+1,sub.mean_abs_error,'o-',ms=3,lw=.8,color=color,label=label)
    j.set_xticks([1,3,5]);j.set_xlabel('Observed-activity quintile');j.set_ylabel('Mean absolute error')
    common_p=p[(p.cohort_id=='common_dev')&(p.protocol_id=='target')].pivot(index='record_id',columns='model_id',values='y_pred')
    true=p[(p.cohort_id=='common_dev')&(p.protocol_id=='target')].drop_duplicates('record_id').set_index('record_id').reindex(common_p.index).y_true
    win=float((abs(common_p.TriCasRank-true)<abs(common_p['EasyDesign-adapted']-true)).mean())
    j.text(0,1.08,f'TriCasRank lower error: {win:.1%}',transform=j.transAxes,fontsize=5.8,color=BLUE)
    fig.text(.115,.018,'Matched panels i–j: 5,995 records, 5,923 target groups. Points/intervals: estimate and 95% group bootstrap CI.',fontsize=6,color=GREY)
    axes=[a,b,c,d,e,f,g,h,i1,i2,j]
    save(fig,destination/'Fig4_corrected',axes,[['c','d','e'],['f','g','h'],['i','i2','j']],[['c','f','i'],['d','g','i2'],['b','e','h','j']])
