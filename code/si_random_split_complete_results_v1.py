# -*- coding: utf-8 -*-
"""论文1：随机划分完整结果、多种子稳定性及 Figure 4 一致性核验。
放入项目 src/ 运行。依赖 Python>=3.9、numpy、pandas、matplotlib。
读取现有真实预测，不训练模型，不修改原数据、原结果或 Figure 4。
输出沿用 outputs/jece_final_package/{tables,figures,reports,logs}/。
共9种方法、5个种子；1015为每种方法合并预测事件数，并非独立样本数。
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import time
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

STEM='SI_random_split_complete_results_v1'
SEEDS=[7,21,42,66,88]
METRICS=['R2','RMSE','MAE','MSE']
REFERENCE='Selected_Weighted_4Model'
LABELS={
 'Selected_Weighted_4Model':'Weighted 4-model',
 'Selected_HistGradientBoosting':'Selected HGB',
 'Selected_CatBoost':'Selected CatBoost',
 'Selected_RF_CatBoost_Weighted':'Weighted RF-CatBoost',
 'RawCompact_HistGradientBoosting':'Raw compact HGB',
 'RawCompact_ExtraTrees':'Raw compact ET',
 'RawCompact_RandomForest':'Raw compact RF',
 'Selected_RandomForest':'Selected RF',
 'Selected_ExtraTrees':'Selected ET'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def resolve(candidates):
    candidates=list(dict.fromkeys(candidates));found=[p for p in candidates if p.is_file()]
    if not found:raise FileNotFoundError('Required evidence is missing; checked:\n'+'\n'.join(map(str,candidates)))
    if len({sha(p) for p in found})>1:raise ValueError('Conflicting versions; resolve the current file before running:\n'+'\n'.join(map(str,found)))
    return found[0]


def paths():
    root=Path(__file__).resolve().parents[1]
    cp=resolve([root/'src/config.py'])
    spec=importlib.util.spec_from_file_location('_si_config',cp)
    cfg=importlib.util.module_from_spec(spec);spec.loader.exec_module(cfg)
    if Path(cfg.PROJECT_ROOT).resolve()!=root:raise ValueError('config.py root differs from script location; place this file in src/.')
    legacy=Path(cfg.TABLE_DIR)
    if not legacy.is_absolute():legacy=root/legacy
    package=root/'outputs/jece_final_package'  # Same paths as formal Figure 4 v9.
    out={k:package/k for k in ['tables','figures','reports','logs']}
    inputs={k:resolve([legacy/f'19_selected_fusion_{k}.csv',out['tables']/f'19_selected_fusion_{k}.csv'])
            for k in ['summary','test_results','predictions','candidate_results']}
    for key in ['ranking_data','taylor_metrics']:
        name=f'Figure_4_random_split_benchmark_and_model_selection_v9_{key}.csv'
        inputs[key]=resolve([out['tables']/name,legacy/name])
    return root,cp,inputs,out


def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);e=p-y
    mse=float(np.mean(e**2));sst=float(np.sum((y-y.mean())**2))
    if sst<=0:raise ValueError('R2 is undefined for constant observations.')
    return {'R2':1-float(np.sum(e**2))/sst,'RMSE':np.sqrt(mse),'MAE':float(np.mean(abs(e))),'MSE':mse}


def check_equal(records,kind,method,metric,recorded,recomputed,seed=''):
    recorded=float(recorded);recomputed=float(recomputed)
    ok=np.isclose(recorded,recomputed,rtol=1e-8,atol=1e-8)
    records.append(dict(Check=kind,Method=method,Seed=seed,Metric=metric,Recorded=recorded,
                        Recomputed=recomputed,Abs_difference=abs(recorded-recomputed),Passed=bool(ok)))
    if not ok:raise ValueError(f'Metric discrepancy: {kind}, {method}, {seed}, {metric}: {recorded} vs {recomputed}')


def audit(t):
    pred=t['predictions'];test=t['test_results'];summary=t['summary'];candidates=t['candidate_results']
    for key,cols in [('predictions',['Seed','Method','Observed','Predicted']),
                     ('test_results',['Seed','Method','Selected_K',*METRICS]),
                     ('summary',['Method',*[m+'_'+s for m in METRICS for s in ['mean','std']]])]:
        if not set(cols).issubset(t[key]):raise ValueError(f'Missing columns in {key}: {set(cols)-set(t[key])}')
    methods=summary.Method.tolist()
    if set(methods)!=set(LABELS) or len(methods)!=9:raise ValueError('Expected the nine locked Step 19 methods; review new evidence separately.')
    if len(pred)!=9135 or len(test)!=45 or test.duplicated(['Seed','Method']).any():raise ValueError('Unexpected record counts or duplicate seed-method result rows.')
    if set(pred.Method)!=set(methods) or set(test.Method)!=set(methods):raise ValueError('Method sets differ across tables.')
    for df in [pred,test]:
        if set(df.Seed)!=set(SEEDS):raise ValueError('Unexpected seed set.')
    if not np.isfinite(pred[['Observed','Predicted']].to_numpy(float)).all():raise ValueError('Nonfinite prediction values.')
    if not (pred.groupby(['Seed','Method']).size()==203).all():raise ValueError('Expected 203 test events in each seed-method cell.')
    # This compares recorded observations, not inferred row identities.
    for seed in SEEDS:
        reference=pred[(pred.Seed==seed)&(pred.Method==REFERENCE)].Observed.to_numpy()
        for method in methods:
            values=pred[(pred.Seed==seed)&(pred.Method==method)].Observed.to_numpy()
            if not np.array_equal(values,reference):raise ValueError('Recorded observed values/order differ across methods within seed.')
    checks=[];seed_rows=[];summary_rows=[];pooled_rows=[];residual_rows=[];selection=[]
    for method in methods:
        per_seed=[]
        for seed in SEEDS:
            g=pred[(pred.Method==method)&(pred.Seed==seed)]
            m=metrics(g.Observed,g.Predicted)
            original=test[(test.Method==method)&(test.Seed==seed)].iloc[0]
            for metric in METRICS:check_equal(checks,'seed_metrics',method,metric,original[metric],m[metric],seed)
            row=dict(Method=method,Label=LABELS[method],Seed=seed,N_prediction_events=len(g),
                     Selected_K=original.Selected_K,Validation_RMSE=original.get('Validation_RMSE',np.nan),**m)
            seed_rows.append(row);per_seed.append(row)
            if method.startswith('Selected_'):
                c=candidates[(candidates.Kind==method)&(candidates.Seed==seed)]
                chosen=c[c.Selected_K.astype(str)==str(original.Selected_K)]
                if len(chosen)!=1:raise ValueError(f'Cannot identify selected K in candidate table: {method}/{seed}')
                recorded=float(original.Validation_RMSE);best=float(c.Validation_RMSE.min())
                check_equal(checks,'selected_validation',method,'Validation_RMSE',recorded,chosen.iloc[0].Validation_RMSE,seed)
                check_equal(checks,'validation_minimum',method,'Validation_RMSE',recorded,best,seed)
                selection.append(dict(Method=method,Seed=seed,Selected_K=original.Selected_K,
                    N_candidates=len(c),Validation_RMSE=recorded,Minimum_candidate_validation_RMSE=best,
                    Selection_basis='Existing validation RMSE; test scores not used by this audit'))
        block=pd.DataFrame(per_seed);row=dict(Method=method,Label=LABELS[method],N_seeds=5)
        for metric in METRICS:
            for stat in ['mean','std']:
                value=float(block[metric].mean() if stat=='mean' else block[metric].std(ddof=1))
                row[metric+'_'+stat]=value
                old=summary.loc[summary.Method==method,metric+'_'+stat].iloc[0]
                check_equal(checks,'summary_'+stat,method,metric,old,value)
            row[metric+'_min']=float(block[metric].min());row[metric+'_max']=float(block[metric].max())
        summary_rows.append(row)
        g=pred[pred.Method==method];m=metrics(g.Observed,g.Predicted)
        y=g.Observed.to_numpy();p=g.Predicted.to_numpy();e=p-y
        corr=float(np.corrcoef(y,p)[0,1]);ratio=float(np.std(p,ddof=1)/np.std(y,ddof=1))
        # Match the original Taylor definition; validated against the exported table.
        crmse=float(np.sqrt(np.mean(((p-p.mean())-(y-y.mean()))**2))/np.std(y,ddof=1))
        pooled_rows.append(dict(Method=method,Label=LABELS[method],N_prediction_events=len(g),
            N_seeds=5,Unique_experimental_rows='Not identifiable from this prediction table',
            **m,Correlation=corr,Std_ratio=ratio,CRMSE_normalized=crmse))
        residual_rows.append(dict(Method=method,Residual_definition='Predicted - Observed',
            Mean_bias=float(e.mean()),Residual_min=float(e.min()),Residual_max=float(e.max()),
            Residual_Q05=float(np.quantile(e,.05)),Residual_Q50=float(np.median(e)),Residual_Q95=float(np.quantile(e,.95)),
            Absolute_error_Q50=float(np.median(abs(e))),Absolute_error_Q90=float(np.quantile(abs(e),.9)),
            Absolute_error_Q95=float(np.quantile(abs(e),.95)),N_abs_error_gt_50=int((abs(e)>50).sum())))
    s=pd.DataFrame(summary_rows);pool=pd.DataFrame(pooled_rows)
    for r in t['ranking_data'].to_dict('records'):
        recomputed=s[s.Method==r['Method']].iloc[0]
        for col in [m+'_'+a for m in METRICS for a in ['mean','std']]:
            check_equal(checks,'Figure4_v9_ranking',r['Method'],col,r[col],recomputed[col])
    for r in t['taylor_metrics'].to_dict('records'):
        recomputed=pool[pool.Method==r['Method']].iloc[0]
        for old,new in [('r2','R2'),('rmse','RMSE'),('mae','MAE'),('n','N_prediction_events'),
                        ('corr','Correlation'),('std_ratio','Std_ratio'),('crmse_norm','CRMSE_normalized')]:
            check_equal(checks,'Figure4_v9_Taylor',r['Method'],old,r[old],recomputed[new])
    return pd.DataFrame(seed_rows),s,pool,pd.DataFrame(residual_rows),pd.DataFrame(selection),pd.DataFrame(checks)


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})


def save_figure(fig,out,suffix,products):
    fig.canvas.draw()
    for ext in ['png','pdf']:
        p=out['figures']/f'{STEM}_{suffix}.{ext}';fig.savefig(p,dpi=600);products.append(p)
    plt.close(fig)


def figures(seed,summary,pool,pred,out,products):
    methods=summary.Method.tolist()
    fig,axes=plt.subplots(1,3,figsize=(12.4,5.8))
    fig.subplots_adjust(left=.205,right=.95,bottom=.15,top=.86,wspace=.30)
    for ax,metric,cmap,letter in zip(axes,['R2','RMSE','MAE'],['YlGnBu','YlOrRd','YlOrRd'],'abc'):
        data=seed.pivot(index='Method',columns='Seed',values=metric).loc[methods,SEEDS].to_numpy()
        im=ax.imshow(data,cmap=cmap,aspect='auto')
        ax.set_xticks(range(5),SEEDS);ax.set_xlabel('Random seed')
        ax.set_yticks(range(9),[LABELS[m] for m in methods] if ax is axes[0] else ['']*9)
        ax.set_title(f'({letter}) '+{'R2':'Test R²','RMSE':'Test RMSE (pp)','MAE':'Test MAE (pp)'}[metric],loc='left',pad=10)
        for i in range(9):
            for j in range(5):
                rgb=im.cmap(im.norm(data[i,j]))[:3];lum=np.dot(rgb,[.2126,.7152,.0722])
                ax.text(j,i,f'{data[i,j]:.3f}' if metric=='R2' else f'{data[i,j]:.2f}',ha='center',va='center',fontsize=8.3,color='white' if lum<.48 else '#161616')
        ax.set_xticks(np.arange(-.5,5,1),minor=True);ax.set_yticks(np.arange(-.5,9,1),minor=True)
        ax.grid(which='minor',color='white',lw=.7);ax.tick_params(which='minor',length=0)
        fig.colorbar(im,ax=ax,fraction=.045,pad=.035)
    fig.suptitle('Random-split test performance across five seeds',fontsize=13,y=.97)
    fig.text(.205,.035,'203 test prediction events per cell. SD across seeds describes split variability; it is not a confidence interval.\npp: percentage points. Higher R² and lower RMSE/MAE indicate better performance.',fontsize=8.2)
    save_figure(fig,out,'seed_stability',products)
    fig,axes=plt.subplots(3,3,figsize=(10.8,10.8))
    fig.subplots_adjust(left=.08,right=.98,bottom=.09,top=.93,hspace=.38,wspace=.30)
    low=min(0,float(pred[['Observed','Predicted']].min().min()));high=max(100,float(pred[['Observed','Predicted']].max().max()))
    limits=(np.floor(low/10)*10-3,np.ceil(high/10)*10+3)
    for i,(ax,method) in enumerate(zip(axes.flat,methods)):
        g=pred[pred.Method==method];m=pool[pool.Method==method].iloc[0]
        ax.scatter(g.Observed,g.Predicted,s=7,alpha=.24,c='#347C9E',edgecolors='none',rasterized=True)
        ax.plot(limits,limits,color='#303A40',ls='--',lw=.8)
        ax.set(xlim=limits,ylim=limits,xlabel='Observed removal (%)',ylabel='Predicted removal (%)')
        ax.set_aspect('equal',adjustable='box');ax.set_xticks([0,25,50,75,100]);ax.set_yticks([0,25,50,75,100])
        ax.set_title(f'({chr(97+i)}) {LABELS[method]}',loc='left',fontsize=9.3)
        ax.text(.035,.97,f'Pooled R² = {m.R2:.3f}\nRMSE = {m.RMSE:.2f} pp\nMAE = {m.MAE:.2f} pp',transform=ax.transAxes,va='top',fontsize=8,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
        ax.grid(alpha=.18,lw=.5)
    fig.suptitle('Five-seed pooled test predictions: all nine benchmark methods',fontsize=12.5,y=.976)
    fig.text(.08,.027,'Each panel contains 1,015 recorded test prediction events (5 × 203), not 1,015 independent experimental samples.\nRepeated events are retained as recorded. The dashed line is y = x; all panels use the same limits.',fontsize=8.5)
    save_figure(fig,out,'all_method_pooled_predictions',products)


def pdf_tables(seed,summary,pool,out,products):
    pdfpath=out['reports']/f'{STEM}_tables.pdf';products.append(pdfpath)
    headers=['Method','R²','RMSE (pp)','MAE (pp)','MSE (pp²)']
    rows=[[LABELS[r.Method],*[f'{getattr(r,m+"_mean"):.3f} ± {getattr(r,m+"_std"):.3f}' for m in METRICS]] for r in summary.itertuples()]
    sections=[('Five-seed test summary: mean ± sample SD',headers,rows,'SD uses ddof=1 across five seeds. Splits can share training and test records; SD is not an independent-sample confidence interval.')]
    sections.append(('Pooled test metrics: five seeds combined',headers,
                     [[r.Label,*[f'{getattr(r,m):.6f}' for m in METRICS]] for r in pool.itertuples()],
                     'Each method has 1,015 test prediction events. Pooled metrics are recomputed from all recorded observations and predictions; they are not averages of seed-level R² or RMSE.'))
    for start in range(0,len(seed),15):
        block=seed.iloc[start:start+15]
        sections.append((f'Complete seed-level test results: part {start//15+1}/3',
            ['Method / seed / K',*headers[1:]],
            [[f'{r.Label} / {r.Seed} / {r.Selected_K}',*[f'{getattr(r,m):.6f}' for m in METRICS]] for r in block.itertuples()],
            'N=203 test events for every row. K is the recorded selection; raw denotes the raw compact baseline. Full-precision values and validation scores are retained in CSV.'))
    with PdfPages(pdfpath) as pdf:
        for page,(title,head,rows,note) in enumerate(sections,1):
            fig=plt.figure(figsize=(11.7,8.3));ax=fig.add_axes([0,0,1,1]);ax.set_axis_off();ax.set(xlim=(0,1),ylim=(0,1))
            fig.text(.04,.95,title,weight='bold',fontsize=13,va='top')
            xs=[.04,.36,.51,.67,.83];y=.87
            ax.plot([.04,.97],[y,y],lw=1.2,c='#253746')
            for x,h in zip(xs,head):fig.text(x+.004,y-.014,h,weight='bold',fontsize=9,va='top')
            y-=.05;ax.plot([.04,.97],[y,y],lw=.6,c='#253746')
            for row in rows:
                wrapped=['\n'.join(textwrap.wrap(str(v),width=37 if i==0 else 22)) for i,v in enumerate(row)]
                height=.041 if len(rows)>10 else .06
                for x,v in zip(xs,wrapped):fig.text(x+.004,y-.008,v,fontsize=8.5,va='top',linespacing=1.08)
                y-=height;ax.plot([.04,.97],[y,y],lw=.4,c='#CDD5DA')
            fig.text(.04,.10,'\n'.join(textwrap.wrap(note,142)),fontsize=8.3,va='top')
            fig.text(.96,.035,str(page),ha='right',fontsize=9)
            pdf.savefig(fig)
            p=out['figures']/f'{STEM}_table_page_{page:02d}.png';fig.savefig(p,dpi=600);products.append(p);plt.close(fig)


def main():
    root,config,inputs,out=paths()
    hashes={str(p.relative_to(root)):sha(p) for p in [config,*inputs.values()]}
    t={k:pd.read_csv(p) for k,p in inputs.items()}
    seed,summary,pool,residual,selection,checks=audit(t)
    for p in out.values():p.mkdir(parents=True,exist_ok=True)
    products=[]
    exports={'seed_metrics':seed,'seed_summary':summary,'pooled_metrics':pool,
        'residual_statistics':residual,'validation_selection_audit':selection,'metric_consistency_checks':checks,
        'complete_validation_candidates':t['candidate_results'],'recorded_prediction_events':t['predictions']}
    for name,df in exports.items():
        p=out['tables']/f'{STEM}_{name}.csv';df.to_csv(p,index=False,encoding='utf-8-sig');products.append(p)
    style();figures(seed,summary,pool,t['predictions'],out,products);pdf_tables(seed,summary,pool,out,products)
    s=summary[summary.Method==REFERENCE].iloc[0];p=pool[pool.Method==REFERENCE].iloc[0]
    report=f'''# 随机划分完整结果与多种子稳定性：补充材料草稿

本项沿用Figure 4正式v9中锁定的Step 19真实预测与结果表。新Figure 4中原六个子图顺延为c–h，未改变这一取数基准。本次保留全部9种方法、5个种子和45个方法—种子组合，不重训模型、不重新选择全局最优方法。

## 汇总口径与结果

每个方法—种子组合含203次测试预测，每种方法合并后为1015次预测事件，9种方法合计9135条预测记录。不同随机划分可能反复测试同一实验记录，不能把1015解释为独立样本数；预测表未提供原始行号，因此不按相同目标值去重，也不反推实验身份。

逐种子R²、RMSE（均方根误差）、MAE（平均绝对误差）和MSE（均方误差）从真实预测重算；均值和样本标准差按5个种子计算，标准差采用ddof=1。合并指标直接从每种方法的1015次预测计算。尤其R²和RMSE的合并值通常不等于逐种子的算术均值。Taylor辅助核验严格沿用v9：中心化误差先取平方均值的平方根，再除以实测值样本标准差（ddof=1）；不更改这一既有定义。

四模型候选融合的逐种子均值±标准差为R²={s.R2_mean:.6f}±{s.R2_std:.6f}、RMSE={s.RMSE_mean:.6f}±{s.RMSE_std:.6f}个百分点、MAE={s.MAE_mean:.6f}±{s.MAE_std:.6f}个百分点。合并预测指标为R²={p.R2:.6f}、RMSE={p.RMSE:.6f}个百分点、MAE={p.MAE:.6f}个百分点。上述指标与Figure 4正式排序表及Taylor（泰勒）图导出数据核验一致。

对已选方法逐种子核对所选K、候选验证RMSE和最小验证RMSE，保留原始选择。全部180条候选验证记录单独导出；不依据本次测试结果反向修改K或权重。Raw compact（原始紧凑特征）方法无对应特征筛选验证分数的空值按原记录保留。

标准差描述5次划分的波动，不作为总体泛化误差的置信区间。本项不进行基于独立样本假设的显著性检验，也不把随机划分成绩解释为跨文献外推能力。后续来源分组、LOLO（留一文献外验证）和AD（适用域）仍需单独评价。

## 补充图图注

Random-split performance across five seeds. The heatmaps show test R², RMSE and MAE for all nine benchmark methods at seeds 7, 21, 42, 66 and 88, with 203 test prediction events per cell. All cells are annotated. RMSE and MAE are expressed in percentage points. Colors are scaled separately for each metric. RF: random forest; ET: extra trees; HGB: histogram-based gradient boosting. Selected denotes the recorded validation-selected feature setting, including the all-feature option.

Five-seed pooled test predictions for all nine benchmark methods. Each panel displays 1,015 recorded prediction events, not 1,015 independent experimental samples. The dashed line represents agreement between observations and predictions. Identical axis limits cover all recorded values. Pooled metrics are calculated from the concatenated prediction records. RF: random forest; ET: extra trees; HGB: histogram-based gradient boosting.

## 文件与后续排版

8个CSV分别保留逐种子指标、均值±标准差、合并指标、完整残差统计、验证选择核验、数值一致性核验、完整候选验证记录和真实预测事件。残差定义为预测值减实测值，统计涵盖全部误差，不截断大残差。2张补充图同时输出600 dpi PNG及PDF；5页参数式结果表提供PDF与逐页PNG。最终补充图表编号和版面在合并时统一。
'''
    fp=out['reports']/f'{STEM}_supplementary_text.md';fp.write_text(report,encoding='utf-8');products.append(fp)
    for rel,h in hashes.items():
        if sha(root/rel)!=h:raise RuntimeError('Input changed during audit: '+rel)
    log={'script':Path(__file__).name,'script_sha256':sha(Path(__file__)), 'input_sha256':hashes,
         'inputs_unchanged':True,'n_methods':9,'n_seeds':5,'n_seed_method_cells':45,
         'n_total_prediction_events':len(t['predictions']),'n_numeric_checks':len(checks),
         'all_numeric_checks_passed':bool(checks.Passed.all()),
         'output_sha256':{str(p.relative_to(root)):sha(p) for p in products}}
    (out['logs']/f'{STEM}_audit.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'PASS: {len(checks)} numeric consistency checks; 9 methods x 5 seeds x 203 prediction events.')
    print(summary[['Label','R2_mean','R2_std','RMSE_mean','MAE_mean']].to_string(index=False))
    print(f'Weighted 4-model pooled: R2={p.R2:.9f}, RMSE={p.RMSE:.9f}, MAE={p.MAE:.9f}')
    print('Outputs: '+str(out['tables'].parent))
    print('Original data, results and main figures unchanged.')


if __name__=='__main__':
    started=time.perf_counter()
    try:main()
    finally:print(f'Total runtime: {time.perf_counter()-started:.2f} s')
