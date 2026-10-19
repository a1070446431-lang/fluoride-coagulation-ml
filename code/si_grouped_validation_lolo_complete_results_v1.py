# -*- coding: utf-8 -*-
"""论文1：Step 31n 分组验证与LOLO完整补充结果。放入项目src/运行。
只读取既有结果，不训练、不调整模型或AD规则、不修改正文图。
依赖Python>=3.9、numpy、pandas、matplotlib。输出沿用正式jece_final_package。
本文件审计Figure 5所用Step31n，不把后续游离氟修正敏感性结果混入。
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
STEM='SI_grouped_validation_lolo_complete_results_v1'
PREFIX='31n_grouped_validation_'
SUFFIXES=['predictions','summary','test_results','split_summary','paper_metrics','paper_macro_summary','fusion_weights','validation_candidates','residual_correlation_by_strategy']
STRATEGIES=['Balanced_5Seed_Current31F','RepeatedBalanced_Extended','RepeatedOrdinary_GroupedSplit','GroupKFold_FullyGrouped','LOLO_LeaveOneLiteratureOut']
SHORT=['Balanced (5)','Balanced (15)','Ordinary (15)','GroupKFold (5)','LOLO (27)']
METHODS=['KRR_Poly','CatBoost','XGBoost','CB-XGB','KRR_CatBoost_Weighted_2Model','KRR_XGBoost_Weighted_2Model','GPR_Matern','PLS']
LABELS=['KRR-Poly','CatBoost','XGBoost','CB-XGB','KRR + CatBoost','KRR + XGBoost','GPR-Matern','PLS']
L=dict(zip(METHODS,LABELS)); S=dict(zip(STRATEGIES,SHORT))

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(d):
 y=d.Observed.to_numpy(float); p=d.Predicted.to_numpy(float); e=p-y
 den=np.sum((y-y.mean())**2)
 return dict(R2=1-np.sum(e**2)/den if len(y)>1 and den>0 else np.nan,RMSE=np.sqrt(np.mean(e**2)),MAE=np.mean(abs(e)),MSE=np.mean(e**2),Bias=-np.mean(e))
def main():
 start=time.perf_counter(); root=Path(__file__).resolve().parents[1]
 cp=root/'src/config.py'
 if not cp.exists(): raise FileNotFoundError(f'Missing project config: {cp}. Place this script in src/.')
 spec=importlib.util.spec_from_file_location('project_config_si',cp); cfg=importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
 if Path(cfg.PROJECT_ROOT).resolve()!=root: raise ValueError('config.PROJECT_ROOT differs from script project root.')
 base=root/'outputs/jece_final_package'; out={k:base/k for k in ['tables','figures','reports','logs']}
 files={}; data={}
 for name in SUFFIXES:
  candidates=[Path(cfg.TABLE_DIR)/(PREFIX+name+'.csv'),base/'tables'/(PREFIX+name+'.csv')]
  existing=[p for p in candidates if p.exists()]
  if not existing: raise FileNotFoundError('Required original result missing. Candidates:\n'+'\n'.join(map(str,candidates)))
  if len({sha(p) for p in existing})>1: raise ValueError(f'Conflicting input versions: {existing}')
  files[name]=existing[0]; data[name]=pd.read_csv(existing[0])
 inputs={str(p.relative_to(root)):sha(p) for p in [cp,*files.values()]}
 pred=data['predictions']; summary=data['summary']; tests=data['test_results']; splits=data['split_summary']
 if pred.shape[0]!=66224 or len(tests)!=442 or len(splits)!=67 or len(summary)!=34: raise ValueError('Input dimensions differ from the audited Figure 5 Step31n package; do not silently substitute another run.')
 if set(pred.Strategy)!=set(STRATEGIES) or pred['文献编号'].nunique()!=27: raise ValueError('Unexpected strategies/source count.')
 if not np.isfinite(pred[['Observed','Predicted']].to_numpy()).all(): raise ValueError('Nonfinite predictions.')
 if pred.duplicated(['Strategy','Split_ID','Method','Data_Index']).any(): raise ValueError('Duplicate prediction keys.')
 identity=pred.groupby('Data_Index').agg(n_y=('Observed','nunique'),n_source=('文献编号','nunique'))
 if len(identity)!=1011 or (identity!=1).any().any(): raise ValueError('Row identity/target/source mismatch.')
 checks=[]
 def check(label,a,b):
  ok=bool(np.isclose(float(a),float(b),rtol=1e-7,atol=1e-8,equal_nan=True))
  checks.append(dict(Check=label,Recomputed=a,Recorded=b,Passed=ok))
  if not ok: raise ValueError(f'Numeric mismatch {label}: {a} != {b}')
 iso=[]
 for _,r in splits.iterrows():
  sets=[set(str(r[k]).split(';')) for k in ['train_groups','valid_groups','test_groups']]
  overlap=len(sets[0]&sets[1])+len(sets[0]&sets[2])+len(sets[1]&sets[2])
  if overlap or len(set.union(*sets))!=27: raise ValueError(f'Source isolation failure: {r.Split_ID}')
  q=pred[(pred.Strategy==r.Strategy)&(pred.Split_ID==r.Split_ID)]
  if set(q['文献编号'])!=sets[2]: raise ValueError('Recorded test sources disagree with predictions.')
  for k,g in zip(['train','valid','test'],sets): check(r.Split_ID+' '+k+' sources',len(g),r['n_'+k+'_groups'])
  check(r.Split_ID+' total rows',r.n_train+r.n_valid+r.n_test,1011)
  iso.append(dict(Strategy=r.Strategy,Split_ID=r.Split_ID,Train_sources=len(sets[0]),Validation_sources=len(sets[1]),Test_sources=len(sets[2]),Source_overlap=overlap,Total_sources=27))
 recomputed=[]
 for key,d in pred.groupby(['Strategy','Split_ID','Method'],sort=False):
  rr=tests[(tests.Strategy==key[0])&(tests.Split_ID==key[1])&(tests.Method==key[2])]
  if len(rr)!=1: raise ValueError('Missing/duplicate test summary key.')
  m=metric(d)
  for k,v in m.items(): check(str(key)+' '+k,v,rr.iloc[0]['Test_'+k])
  check(str(key)+' N',len(d),rr.iloc[0].n_test)
  recomputed.append(dict(zip(['Strategy','Split_ID','Method'],key),**m,N_events=len(d),N_unique=d.Data_Index.nunique()))
 rec=pd.DataFrame(recomputed); pooled=[]; papers=[]
 for key,d in pred.groupby(['Strategy','Method'],sort=False):
  rr=summary[(summary.Strategy==key[0])&(summary.Method==key[1])].iloc[0]; m=metric(d)
  for k,v in m.items(): check(str(key)+' pooled '+k,v,rr['Pooled_'+k])
  z=rec[(rec.Strategy==key[0])&(rec.Method==key[1])]
  for k in ['R2','RMSE','MAE']:
   check(str(key)+' split mean '+k,z[k].mean(),rr['Split_'+k+'_mean'])
   check(str(key)+' split std '+k,z[k].std(ddof=1),rr['Split_'+k+'_std'])
  check(str(key)+' events',len(d),rr.n_pooled)
  pooled.append(dict(Strategy=key[0],Method=key[1],**m,N_events=len(d),N_unique=d.Data_Index.nunique(),N_sources=d['文献编号'].nunique(),N_splits=d.Split_ID.nunique()))
  if key[0] in STRATEGIES[3:]:
   if len(d)!=1011 or d.Data_Index.nunique()!=1011: raise ValueError('GroupKFold/LOLO incomplete or repeated test rows.')
  for paper,g in d.groupby('文献编号'):
   mm=metric(g); old=data['paper_metrics']; old=old[(old.Strategy==key[0])&(old.Method==key[1])&(old['文献编号']==paper)]
   if len(old)!=1: raise ValueError('Missing paper summary.')
   for k in ['R2','RMSE','MAE','Bias']: check(str(key)+paper+k,mm[k],old.iloc[0]['Paper_'+k])
   papers.append(dict(Strategy=key[0],Method=key[1],Source=paper,**mm,N_events=len(g),N_unique=g.Data_Index.nunique(),Observed_SD=g.Observed.std(ddof=1)))
 pooled=pd.DataFrame(pooled); paper=pd.DataFrame(papers)
 for key,g in paper.groupby(['Strategy','Method']):
  old=data['paper_macro_summary']; rr=old[(old.Strategy==key[0])&(old.Method==key[1])].iloc[0]
  for k in ['RMSE','MAE']:
   for agg in ['mean','median']: check(str(key)+' macro '+k+agg,getattr(g[k],agg)(),rr['PaperMacro_'+k+'_'+agg])
 for p in out.values(): p.mkdir(parents=True,exist_ok=True)
 products=[]
 def csv(d,name):
  p=out['tables']/(STEM+'_'+name+'.csv');d.to_csv(p,index=False,encoding='utf-8-sig');products.append(p)
 for name,d in data.items(): csv(d,'original_'+name)
 for name,d in [('recomputed_split_metrics',rec),('recomputed_pooled_metrics',pooled),('recomputed_paper_metrics',paper),('source_isolation_audit',pd.DataFrame(iso)),('numeric_checks',pd.DataFrame(checks))]: csv(d,name)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 def save(fig,name):
  for ext in ['png','pdf']:
   p=out['figures']/(STEM+'_'+name+'.'+ext); fig.savefig(p,dpi=400,facecolor='white');products.append(p)
  plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(14,5.6));fig.subplots_adjust(left=.15,right=.98,bottom=.25,top=.9,wspace=.37)
 for ax,k,cm in zip(axs,['R2','RMSE','MAE'],['RdBu','YlOrRd','YlOrRd']):
  mat=pooled.pivot(index='Method',columns='Strategy',values=k).reindex(index=METHODS,columns=STRATEGIES).to_numpy()
  cmap=plt.get_cmap(cm).copy();cmap.set_bad('#eeeeee');im=ax.imshow(np.ma.masked_invalid(mat),cmap=cmap,aspect='auto',**({'vmin':-.3,'vmax':.3} if k=='R2' else {}))
  ax.set_xticks(range(5),SHORT,rotation=45,ha='right');ax.set_yticks(range(8),LABELS if ax==axs[0] else ['']*8);ax.set_title('Pooled '+k)
  for i in range(8):
   for j in range(5):
    v=mat[i,j];ax.text(j,i,'NA' if np.isnan(v) else f'{v:.3f}' if k=='R2' else f'{v:.2f}',ha='center',va='center',fontsize=8,color='white' if np.isfinite(v) and im.norm(v)>.72 else 'black')
  fig.colorbar(im,ax=ax,fraction=.046,pad=.03)
 fig.text(.15,.035,'NA: model not evaluated under this strategy. Repeated-split events are not independent samples.',fontsize=9)
 save(fig,'strategy_model_metrics')
 lo=paper[paper.Strategy==STRATEGIES[-1]]; lm=[m for m in METHODS if m in set(lo.Method)]; sources=sorted(lo.Source.unique())
 fig,ax=plt.subplots(figsize=(9.5,10));fig.subplots_adjust(left=.15,right=.9,bottom=.16,top=.92)
 mat=lo.pivot(index='Source',columns='Method',values='RMSE').reindex(index=sources,columns=lm).to_numpy();im=ax.imshow(mat,cmap='YlOrRd',aspect='auto')
 ax.set_xticks(range(len(lm)),[L[m] for m in lm],rotation=35,ha='right');ax.set_yticks(range(27),sources);ax.set_title('LOLO: source-level RMSE (removal percentage points)',pad=14)
 for i in range(27):
  for j in range(len(lm)): ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if im.norm(mat[i,j])>.65 else 'black')
 fig.colorbar(im,ax=ax,fraction=.035,pad=.035);fig.text(.15,.025,'Each source is held out once. Source IDs are diagnostic labels, not model predictors.',fontsize=9);save(fig,'lolo_source_rmse')
 fig,axs=plt.subplots(2,3,figsize=(12,8));fig.subplots_adjust(left=.08,right=.97,bottom=.1,top=.91,wspace=.27,hspace=.32)
 d=pred[pred.Strategy==STRATEGIES[-1]];v=d[['Observed','Predicted']].to_numpy();low=min(-5,float(v.min())-3);high=max(105,float(v.max())+3)
 for ax,m in zip(axs.flat,lm):
  g=d[d.Method==m];mm=metric(g);ax.scatter(g.Observed,g.Predicted,s=7,alpha=.3,c='#287a8c',rasterized=True);ax.plot([low,high],[low,high],color='black',lw=.8,ls='--');ax.set(xlim=(low,high),ylim=(low,high),title=L[m]);ax.text(.04,.96,f"R² = {mm['R2']:.3f}\nRMSE = {mm['RMSE']:.2f}\nMAE = {mm['MAE']:.2f}",transform=ax.transAxes,va='top',fontsize=8)
 fig.supxlabel('Observed removal (%)');fig.supylabel('Predicted removal (%)');fig.suptitle('LOLO pooled predictions: 1011 distinct rows per method');save(fig,'lolo_predictions')
 tablefile=out['reports']/(STEM+'_summary_tables.pdf')
 with PdfPages(tablefile) as pdf:
  for strategy in STRATEGIES:
   g=pooled[pooled.Strategy==strategy].copy();g['Method']=g.Method.map(L);g=g[['Method','R2','RMSE','MAE','N_events','N_unique','N_sources','N_splits']]
   cells=[[f'{v:.3f}' if isinstance(v,float) else str(v) for v in row] for row in g.itertuples(index=False,name=None)]
   fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off');ax.set_title('Step 31n: '+S[strategy],fontsize=16,pad=25)
   t=ax.table(cellText=cells,colLabels=['Method','R²','RMSE','MAE','Events','Unique rows','Sources','Splits'],cellLoc='center',colWidths=[.24,.10,.11,.11,.11,.12,.10,.10],bbox=[0,.28,1,.57]);t.auto_set_font_size(False);t.set_fontsize(10)
   for (i,j),c in t.get_celld().items():
    c.set_edgecolor('#ccd3d7');c.set_linewidth(.5)
    if i==0:c.set_facecolor('#244957');c.get_text().set_color('white')
    elif i%2:c.set_facecolor('#edf3f5')
   fig.text(.13,.16,'Pooled metrics use all recorded test events within this strategy and method.\nRMSE and MAE: removal percentage points. Methods absent from the source run are not imputed.\nComplete split-level, source-level, configuration and audit records are supplied as CSV tables.',fontsize=10,linespacing=1.6)
   pdf.savefig(fig);plt.close(fig)
 products.append(tablefile)
 report='''# 分组验证与LOLO补充材料（Step 31n）

## 统计范围与边界
本模块完整保留Figure 5取数所用Step 31n历史结果：5种策略、67次划分、442组模型测试结果、66224条预测事件、1011个原始行标识及27个有效来源。各策略实际评估的方法数不同；缺测组合不补零、不推算。后续游离氟占比修正的配对敏感性结果不属于此结果版本，应另表报告。

分组隔离审计确认每次训练、验证和测试来源集合互不重叠。此检查仅证明来源集合隔离，不能单独证明整个建模流程不存在任何信息泄露。原31F/31n的balanced split（平衡划分）使用了目标及输入变量的分布统计来选择划分，因此应作为目标分布参与平衡的分组敏感性设计，不宜称为完全不使用测试目标分布的外部验证。GroupKFold（分组交叉验证）与LOLO（留一文献外验证）的内部验证来源选择使用非测试池目标分布，测试来源保持排除。

## 结果口径
Pooled metrics（合并指标）由同一策略、同一方法的全部测试预测计算。Repeated split（重复划分）中同一原始行可多次受测，预测事件数与独立行数分别列出，不能把事件当作新增独立实验。Split mean ± SD（划分均值±标准差）为描述性汇总，标准差采用ddof=1，不作为独立重复试验的置信区间。Paper macro metrics（来源宏平均指标）对各来源指标等权汇总，与逐条预测加权的合并指标不同。来源内目标变异小会使R²不稳定，应结合样本量、目标标准差、RMSE和MAE解读。

历史Balanced-5的最好合并R²约0.249558；RepeatedBalanced-15的KRR与CatBoost加权融合合并R²约0.193744。两者不是同一评估集合。LOLO的XGBoost合并R²约0.183378；这不是适用域筛选后的可靠窗口指标。所有策略结果均完整呈现，不根据测试结果逐来源切换最佳模型，不重新筛选AD（适用域）。

## English captions
Strategy-level performance. Pooled R², RMSE and MAE for every strategy-method combination recorded in Step 31n. NA denotes an unevaluated combination. Repeated-split events may include the same experimental row more than once.

Source-level LOLO errors. RMSE is reported for each held-out source and each evaluated method. Source identifiers are used solely for validation and diagnostics. Errors are in removal percentage points.

LOLO predictions. Observed and predicted removal for 1011 distinct experimental rows per method, each tested in the fold holding out its source. All recorded predictions are retained, including predictions outside 0–100%; the dashed line indicates perfect agreement.

KRR: kernel ridge regression（核岭回归）；GPR: Gaussian process regression（高斯过程回归）；PLS: partial least squares（偏最小二乘）；CB-XGB: CatBoost–XGBoost融合。KRR + CatBoost及KRR + XGBoost为原结果中记录的加权融合。

完整原始验证候选表直接保留；其候选记录没有Strategy与Split_ID字段，因此本模块不虚构逐候选与分组划分的一一映射。所有核验仅对已有记录执行，不重训模型。最终Table S/Figure S编号在补充材料总编排时统一分配。
'''
 rp=out['reports']/(STEM+'_supplementary_text.md');rp.write_text(report,encoding='utf-8');products.append(rp)
 unchanged=all(sha(root/p)==h for p,h in inputs.items())
 if not unchanged: raise RuntimeError('Input file hash changed.')
 audit=dict(input_sha256=inputs,inputs_unchanged=unchanged,n_numeric_checks=len(checks),all_checks_passed=True,n_source_isolation_checks=len(iso),n_prediction_events=len(pred),n_unique_rows=len(identity),n_sources=27,script_sha256=sha(Path(__file__)),output_sha256={str(p.relative_to(root)):sha(p) for p in products})
 (out['logs']/(STEM+'_audit.json')).write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
 print(f'PASS: {len(checks)} numeric checks; {len(iso)} source-isolated splits; 27 sources.')
 print(pooled[pooled.Strategy==STRATEGIES[-1]][['Method','R2','RMSE','MAE','N_unique']].to_string(index=False))
 print('Outputs:',base);print('Original predictions/results unchanged; no models trained.')
 print(f'Total runtime: {time.perf_counter()-start:.2f} s')
if __name__=='__main__': main()
