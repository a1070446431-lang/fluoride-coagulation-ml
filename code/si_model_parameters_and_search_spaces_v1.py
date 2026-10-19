# -*- coding: utf-8 -*-
"""论文1：模型参数、实际搜索空间及逐种子入选配置的补充材料。

将完整文件放入项目 src/ 运行。依赖：Python >=3.9、pandas、numpy、matplotlib。
只解析原脚本的 AST（抽象语法树）并读取既有结果；不导入或运行建模脚本。
输出沿用 outputs/jece_final_package/{tables,figures,reports,logs}/。
表格不预占 Table S 编号。未显式设置的库默认参数不以当前环境补写。
逐种子结果保留原值；早期搜索与最终 Step 19、分组探索分别标明。
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import importlib.metadata
import json
import re
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

STEM = 'SI_model_parameters_and_search_spaces_v1'
SCRIPTS = {
    '12': '12_rf_tuning.py',
    '13': '13_catboost_tuning.py',
    '16': '16_catboost_bayes_multiseed.py',
    '19': '19_selected_feature_fusion.py',
    '27': '27_oof_stacking_benchmark.py',
    '31g': '31g_expanded_model_complementarity_screening.py',
    '31h': '31h_group_aware_catboost_xgboost_tuning.py',
    '31h2': '31h2_tuning_grid_adequacy_check.py',
    '31l': '31l_non_tree_model_screening.py',
}
TABLES = {
    'random_weights': '19_selected_fusion_weights.csv',
    'random_selection': '19_selected_fusion_test_results.csv',
    'h_selected': '31h_tuned_catboost_xgboost_selected_params.csv',
    'h2_selected': '31h2_tuning_grid_adequacy_selected_params.csv',
    'h2_catalog': '31h2_tuning_grid_adequacy_config_catalog.csv',
    'non_tree': '31l_non_tree_model_config_audit.csv',
}
MODEL_CALLS = {'RandomForestRegressor','ExtraTreesRegressor','HistGradientBoostingRegressor',
    'CatBoostRegressor','XGBRegressor','LGBMRegressor','GradientBoostingRegressor',
    'AdaBoostRegressor','KernelRidge','PLSRegression','GaussianProcessRegressor',
    'Ridge','RidgeCV','LinearRegression','ElasticNet','SVR','BayesianRidge','MLPRegressor',
    'StackingRegressor'}
PREPROCESS_CALLS = {'SimpleImputer','OneHotEncoder','StandardScaler','RobustScaler','MinMaxScaler',
    'KFold','GroupKFold','GroupShuffleSplit','train_test_split','RandomizedSearchCV','GridSearchCV',
    'TPESampler','RBF','Matern','WhiteKernel','ConstantKernel'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jdump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def literal(node):
    """Only literals and literal dict(...) calls; never eval/exec source code."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
        if node.args or any(k.arg is None for k in node.keywords):
            raise ValueError('Dynamic dict is not a literal configuration')
        return {k.arg: literal(k.value) for k in node.keywords}
    if isinstance(node, ast.Dict):
        return {literal(k):literal(v) for k,v in zip(node.keys,node.values)}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [literal(x) for x in node.elts]
    return ast.literal_eval(node)


def value_text(node):
    try:
        return literal(node)
    except (ValueError, TypeError, SyntaxError):
        return '<expression> ' + ast.unparse(node)


def locate(candidates, required=True):
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        if required:
            raise FileNotFoundError('Missing required project evidence; checked:\n'+'\n'.join(str(p) for p in candidates))
        return None
    if len({sha(p) for p in existing}) > 1:
        raise ValueError('Conflicting copies exist. Resolve which version is current before running:\n'+'\n'.join(str(p) for p in existing))
    return existing[0]


def paths():
    root = Path(__file__).resolve().parents[1]
    config_path = root/'src'/'config.py'
    if not config_path.is_file():
        raise FileNotFoundError(f'Put this script in project src/. Required: {config_path}')
    spec=importlib.util.spec_from_file_location('_si_config',config_path)
    cfg=importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
    if Path(cfg.PROJECT_ROOT).resolve()!=root:
        raise ValueError('Script location and config.py PROJECT_ROOT disagree.')
    table_root=Path(cfg.TABLE_DIR)
    if not table_root.is_absolute():table_root=root/table_root
    package=root/'outputs'/'jece_final_package'  # Same formal package as Figure 4 v9.
    outputs={k:package/k for k in ['tables','figures','reports','logs']}
    source_paths={k:locate([root/'src'/n,root/n]) for k,n in SCRIPTS.items()}
    table_paths={k:locate(list(dict.fromkeys([table_root/n,outputs['tables']/n]))) for k,n in TABLES.items()}
    return root, config_path, source_paths, table_paths, outputs


def assignment(tree, name):
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    if len(matches)!=1:raise ValueError(f'Expected exactly one assignment to {name}; found {len(matches)}')
    return literal(matches[0].value), matches[0].lineno


def function(tree, name):
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(matches)!=1:raise ValueError(f'Expected one function {name}')
    return matches[0]


def constructor_rows(tree, stage, source, scope=None):
    rows=[]
    start=function(tree,scope) if scope else tree
    for n in ast.walk(start):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in MODEL_CALLS:
            params={k.arg if k.arg is not None else '**expanded':value_text(k.value) for k in n.keywords}
            if n.args:params['positional_arguments']=[value_text(x) for x in n.args]
            rows.append(dict(stage=stage,model=n.func.id,parameters_json=jdump(params),source_file=source,line=n.lineno))
    return rows


def get_catalog(trees, source_paths, core):
    records=[]
    baseline=json.loads(next(x['parameters_json'] for x in core if x['model']=='CatBoostRegressor'))
    for stage in ['31h','31h2']:
        for model,name in [('CatBoost','CATBOOST_CONFIGS'),('XGBoost','XGBOOST_CONFIGS')]:
            configs,line=assignment(trees[stage],name)
            factory=function(trees[stage], 'make_catboost_model' if model=='CatBoost' else 'make_xgboost_model')
            calls=constructor_rows(trees[stage],stage,source_paths[stage].name,factory.name)
            fixed=json.loads(calls[0]['parameters_json']);fixed.pop('**expanded',None)
            for c in configs:
                declared=c['Params']
                effective=baseline.copy() if declared is None else {**fixed,**declared}
                records.append(dict(stage=stage,model=model,config_id=c['Config_ID'],
                    block=c.get('Grid_Block',c.get('Source','')),
                    declared_parameters_json=jdump(declared),explicit_effective_parameters_json=jdump(effective),
                    source_file=source_paths[stage].name,line=line,
                    interpretation='Exact listed combination; not a Cartesian product'))
    return pd.DataFrame(records)


def check_recorded_selections(catalog, tables):
    checks=[]
    for stage,key in [('31h','h_selected'),('31h2','h2_selected')]:
        for r in tables[key].to_dict('records'):
            for family in ['CatBoost','XGBoost']:
                cid=r.get(family+'_Config_ID','')
                if not cid:continue
                candidates=catalog[(catalog.stage==stage)&(catalog.model==family)&(catalog.config_id==cid)]
                if len(candidates)!=1:raise ValueError(f'Unknown recorded selected configuration: {stage}/{cid}')
                expected=json.loads(candidates.iloc[0].declared_parameters_json)
                recorded=r.get(family+'_Params','')
                if expected is None:
                    if recorded not in ('','script19_factory_default','None'):
                        raise ValueError(f'Unexpected baseline parameter record: {recorded}')
                elif ast.literal_eval(recorded)!=expected:
                    raise ValueError(f'Recorded/source parameter mismatch: {stage}/{cid}')
                checks.append(dict(stage=stage,seed=r['Seed'],method=r['Method'],model=family,config_id=cid,status='matched_source_config'))
    for r in tables['h2_catalog'].to_dict('records'):
        c=catalog[(catalog.stage=='31h2')&(catalog.model==r['Model_Family'])&(catalog.config_id==r['Config_ID'])]
        if len(c)!=1:raise ValueError('31H2 archived catalog does not match code IDs')
        expected=json.loads(c.iloc[0].declared_parameters_json)
        if expected is not None and ast.literal_eval(r['Params'])!=expected:raise ValueError('31H2 catalog parameter mismatch')
    if len(tables['h2_catalog'])!=len(catalog[catalog.stage=='31h2']):raise ValueError('31H2 catalog count mismatch')
    weights=tables['random_weights'].copy()
    weights['Weight']=pd.to_numeric(weights.Weight)
    if not weights.Weight.between(0,1).all():raise ValueError('Invalid ensemble weights')
    if not np.allclose(weights.groupby(['Seed','Method']).Weight.sum(),1,atol=1e-7):raise ValueError('Ensemble weights do not sum to 1')
    if weights.duplicated(['Seed','Method','Model']).any():raise ValueError('Duplicate seed-method-model weights')
    if set(weights.Seed.astype(int))!={7,21,42,66,88}:raise ValueError('Unexpected seed set in Step 19 weights')
    return pd.DataFrame(checks)


def search_rows(trees,source_paths):
    rows=[]
    for stage,var,model in [('12','param_distributions','RandomForest'),('13','param_space','CatBoost')]:
        space,line=assignment(trees[stage],var)
        for p,values in space.items():
            rows.append(dict(stage=stage,model=model,parameter=p,range_or_expression=jdump(values),
                search_type='Historical random sampling; not final Step 19',source_file=source_paths[stage].name,line=line))
    for n in ast.walk(function(trees['16'],'suggest_catboost_params')):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr.startswith('suggest_'):
            rows.append(dict(stage='16',model='CatBoost',parameter=literal(n.args[0]),range_or_expression=ast.unparse(n),
                search_type='Historical Optuna TPE; not final Step 19',source_file=source_paths['16'].name,line=n.lineno))
    return pd.DataFrame(rows)


def source_inventory(root,source_paths):
    """Capture explicit constructors, preprocessing, fit options and controls in project stages.

    Symbolic expressions retain their source form. Static presence is not proof that
    a branch executed; only recorded-result tables establish observed selections.
    """
    files={p.name:p for p in source_paths.values()}
    for base in [root,root/'src']:
        for p in base.glob('*.py'):
            if re.match(r'^(0[4-9]|1[0-9]|2[2-8]|30_|31[a-p])',p.name):
                if p.name in files and sha(files[p.name])!=sha(p):raise ValueError(f'Conflicting script copies: {p.name}')
                files[p.name]=p
    inventory=[];manifest=[]
    for name,p in sorted(files.items()):
        text=p.read_text(encoding='utf-8-sig'); tree=ast.parse(text,filename=str(p))
        manifest.append(dict(source_file=str(p.relative_to(root)),sha256=sha(p),n_lines=len(text.splitlines())))
        parents={c:n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        for n in ast.walk(tree):
            kind=None
            if isinstance(n,ast.Call):
                end=ast.unparse(n.func).rsplit('.',1)[-1]
                if end in MODEL_CALLS:kind='model_constructor'
                elif end in PREPROCESS_CALLS:kind='preprocessing_split_or_search'
                elif end in ['fit','optimize'] and n.keywords:kind='fit_or_optimizer_options'
                elif end.startswith('suggest_'):kind='trial_search_range'
            elif isinstance(n,ast.Assign):
                targets=ast.unparse(n.targets)
                if re.search(r'SEED|SPLIT|K_OPTIONS|N_TRIAL|N_RANDOM_WEIGHTS|ALLOWED_K|param_space|param_distributions|CONFIGS|TEST_SIZE|VALID_SIZE',targets):kind='search_or_evaluation_control'
            if kind:
                parent=n
                while parent in parents and not isinstance(parent,(ast.FunctionDef,ast.AsyncFunctionDef)):parent=parents[parent]
                scope=parent.name if isinstance(parent,(ast.FunctionDef,ast.AsyncFunctionDef)) else '<module>'
                inventory.append(dict(source_file=str(p.relative_to(root)),line=n.lineno,scope=scope,
                    evidence_type=kind,source_expression=ast.get_source_segment(text,n) or ast.unparse(n),
                    status='Source declaration only; not proof of execution'))
    return pd.DataFrame(inventory),pd.DataFrame(manifest),list(files.values())


def non_tree_table(table, tree, source):
    # Compare recorded parameter families/ranges with the explicit loop candidates
    # in make_model_configs, without importing that script or its dependencies.
    fn=function(tree,'make_model_configs')
    loop_candidates=[]
    for n in fn.body:
        if isinstance(n,ast.For):
            loop_candidates.append(literal(n.iter))
    expected_counts=[6,8,5,3,3]
    if [len(x) for x in loop_candidates]!=expected_counts:
        raise ValueError('31L candidate loop structure changed; review non-tree catalog extraction.')
    families=['PLS','KRR_RBF','KRR_Poly','GPR_RBF','GPR_Matern']
    for family,values in zip(families,loop_candidates):
        actual=[ast.literal_eval(s) for s in table.loc[table.Model_Family==family,'Params']]
        if family=='PLS':expected=[dict(n_components=x) for x in values]
        elif family=='KRR_RBF':expected=[dict(kernel='rbf',alpha=x,gamma=y) for x,y in values]
        elif family=='KRR_Poly':expected=[dict(kernel='polynomial',alpha=x,degree=y,coef0=1.0,gamma=None) for x,y in values]
        elif family=='GPR_RBF':expected=[dict(kernel_type='rbf',length_scale=x,noise=y) for x,y in values]
        else:expected=[dict(kernel_type='matern',nu=x,length_scale=y,noise=z) for x,y,z in values]
        if sorted(map(jdump,actual))!=sorted(map(jdump,expected)):
            raise ValueError(f'31L recorded/source candidates mismatch: {family}')
    result=table.copy();result.insert(0,'Stage','31l')
    result['Evidence_Source']=source;result['Evidence_Line']=fn.lineno
    return result


def render_tables(sections, outputs):
    """Paginated vector tables, with a 600-dpi PNG for every PDF page."""
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    pdfpath=outputs['reports']/(STEM+'_parameter_tables.pdf')
    products=[pdfpath];page_no=0
    with PdfPages(pdfpath) as pdf:
        for title,headers,rows,note,widths in sections:
            # Three columns, with explicit wrapping and row heights derived from text.
            capacities=[max(10,int(w*120)) for w in widths]
            prepared=[]
            for row in rows:
                cells=['\n'.join(textwrap.wrap(str(v),width=c,break_long_words=True,break_on_hyphens=False)) for v,c in zip(row,capacities)]
                nlines=max(len(c.splitlines()) for c in cells)
                prepared.append((cells,max(.036,.0195*nlines+.014)))
            pages=[];batch=[];used=0
            for item in prepared:
                if item[1]>.69:raise ValueError('A parameter row is too long for one page; review table structure.')
                if batch and used+item[1]>.69:pages.append(batch);batch=[];used=0
                batch.append(item);used+=item[1]
            if batch:pages.append(batch)
            # Avoid a single orphan row on the last continuation page.
            if len(pages)>1 and len(pages[-1])==1 and len(pages[-2])>=4:
                moved=pages[-2][-2:]
                if sum(item[1] for item in moved+pages[-1])<=.69:
                    pages[-2]=pages[-2][:-2]
                    pages[-1]=moved+pages[-1]
            for part,batch in enumerate(pages,1):
                page_no+=1;fig=plt.figure(figsize=(11.7,8.3));ax=fig.add_axes([0,0,1,1]);ax.set_axis_off()
                fig.text(.045,.955,title,fontsize=13,weight='bold',va='top')
                fig.text(.045,.913,f'Part {part}/{len(pages)} | Parameters and search evidence',fontsize=9,color='#455866')
                x=np.r_[.045,.045+np.cumsum(np.array(widths)*.91)]
                y=.875
                ax.plot([.045,.955],[y,y],color='#253746',lw=1.3)
                for i,h in enumerate(headers):fig.text(x[i]+.005,y-.01,h,fontsize=9,weight='bold',va='top')
                y-=.041;ax.plot([.045,.955],[y,y],color='#253746',lw=.65)
                for cells,height in batch:
                    for i,c in enumerate(cells):fig.text(x[i]+.005,y-.008,c,fontsize=8.7,va='top',linespacing=1.13)
                    y-=height;ax.plot([.045,.955],[y,y],color='#CBD3D8',lw=.45)
                fig.text(.045,.09,'\n'.join(textwrap.wrap(note,145)),fontsize=8,va='top',color='#35444E')
                fig.text(.955,.032,str(page_no),ha='right',fontsize=9)
                ax.set_xlim(0,1);ax.set_ylim(0,1)
                fig.canvas.draw()
                renderer=fig.canvas.get_renderer()
                for item in fig.texts:
                    box=item.get_window_extent(renderer)
                    if box.x0<0 or box.x1>fig.bbox.width or box.y0<0 or box.y1>fig.bbox.height:
                        raise RuntimeError('Table text exceeds page bounds: '+item.get_text()[:70])
                pdf.savefig(fig)
                png=outputs['figures']/f'{STEM}_table_page_{page_no:02d}.png'
                fig.savefig(png,dpi=600);products.append(png);plt.close(fig)
    return products,page_no


def parameter_string(value):
    p=json.loads(value) if isinstance(value,str) else value
    if not p:
        return 'No explicit keyword arguments; historical library defaults not reconstructed.'
    return '; '.join(f'{k}={v}' for k,v in p.items())


def main():
    root,config,source_paths,table_paths,outputs=paths()
    trees={k:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p)) for k,p in source_paths.items()}
    tables={k:pd.read_csv(p,keep_default_na=False) for k,p in table_paths.items()}
    evidence_paths=[config,*source_paths.values(),*table_paths.values()]
    original_hashes={str(p.relative_to(root)):sha(p) for p in evidence_paths}
    core=constructor_rows(trees['19'],'19',source_paths['19'].name,'get_model_factories')
    if {r['model'] for r in core}!={'RandomForestRegressor','ExtraTreesRegressor','HistGradientBoostingRegressor','CatBoostRegressor'}:
        raise ValueError('Step 19 factory structure changed; inspect before publication.')
    catalog=get_catalog(trees,source_paths,core)
    checks=check_recorded_selections(catalog,tables)
    nt=non_tree_table(tables['non_tree'],trees['31l'],source_paths['31l'].name)
    search=search_rows(trees,source_paths)
    inventory,manifest,extra_paths=source_inventory(root,source_paths)
    original_hashes.update({str(p.relative_to(root)):sha(p) for p in extra_paths})
    factories=pd.DataFrame(core+constructor_rows(trees['31g'],'31g',source_paths['31g'].name)+
        constructor_rows(trees['27'],'27',source_paths['27'].name,'fit_meta_and_predict')+
        constructor_rows(trees['31l'],'31l',source_paths['31l'].name,'fit_predict_non_tree'))
    weights=tables['random_weights']; selected=tables['random_selection']
    # Preserve recorded selections: do not choose a new global winner from test scores.
    selected_exports=[]
    for stage,key in [('31h','h_selected'),('31h2','h2_selected')]:
        frame=tables[key].copy();frame.insert(0,'Stage',stage);frame['Evidence_Source']=table_paths[key].name
        selected_exports.append(frame)
    grouped_selected=pd.concat(selected_exports,ignore_index=True).fillna('')
    # Expanded values are a summary of listed combinations, not a new search grid.
    range_rows=[]
    for (stage,model),sub in catalog.groupby(['stage','model'],sort=False):
        values={}
        for cell in sub.explicit_effective_parameters_json:
            for p,v in json.loads(cell).items():values.setdefault(p,set()).add(jdump(v))
        for p,vs in values.items():range_rows.append(dict(stage=stage,model=model,parameter=p,
            listed_values=' | '.join(sorted(vs)),n_exact_configs=len(sub),
            interpretation='Union of values across listed combinations; not independently crossed'))
    ranges=pd.DataFrame(range_rows)
    for p in outputs.values():p.mkdir(parents=True,exist_ok=True)
    products=[]
    exports={'explicit_model_parameters':factories,'grouped_exact_configurations':catalog,
        'grouped_listed_value_ranges':ranges,'non_tree_exact_configurations':nt,
        'historical_search_spaces':search,'random_selected_settings':selected,
        'random_selected_weights':weights,'grouped_selected_settings':grouped_selected,
        'source_code_evidence':inventory,'source_manifest':manifest,'selection_crosscheck':checks}
    for name,df in exports.items():
        p=outputs['tables']/f'{STEM}_{name}.csv';df.to_csv(p,index=False,encoding='utf-8-sig');products.append(p)
    sections=[]
    sections.append(('Random-split benchmark: fixed base-model parameters',
        ['Stage / model','Explicit parameters','Evidence'],
        [[r['stage']+' / '+r['model'].replace('Regressor',''),parameter_string(r['parameters_json']),f'Step 19 factory; line {r["line"]}'] for r in core],
        'These are the Step 19 factory settings. seed varies by split. Unspecified library defaults are not reconstructed from the current environment.',[.22,.61,.17]))
    add=factories[(factories.stage=='31g')|(factories.stage=='27')|(factories.stage=='31l')]
    sections.append(('Additional model families: explicit constructor settings',
        ['Stage / model','Explicit parameters / expressions','Evidence'],
        [[r.stage+' / '+r.model.replace('Regressor',''),parameter_string(r.parameters_json),f'Step {r.stage}; line {r.line}'] for r in add.itertuples()],
        'Expressions retain their code meaning; expanded configuration dictionaries are documented in the exact candidate tables. Constructor presence alone does not prove a branch ran.',[.22,.61,.17]))
    for stage in ['31h','31h2']:
        sub=catalog[catalog.stage==stage]
        sections.append((f'Grouped tuning {stage.upper()}: exact candidate combinations',
            ['Model / configuration ID','Explicit effective parameters','Block'],
            [[r.model+' / '+r.config_id,parameter_string(r.explicit_effective_parameters_json),r.block.replace('_',' ')] for r in sub.itertuples()],
            'Each row is one listed candidate. This is not a Cartesian grid. Seed-dependent selections are retained separately; the baseline CatBoost entry resolves to the Step 19 factory.',[.25,.58,.17]))
    sections.append(('Non-tree screening: exact candidate configurations',
        ['Family / configuration','Parameters','Feature K / weighting'],
        [[r.Model_Family+' / '+r.Config_Name,parameter_string(ast.literal_eval(r.Params)),str(r.Allowed_K)+' / '+r.Weighting_Mode.replace('_',' ')] for r in nt.itertuples()],
        'Allowed_K=all in the catalog means no additional restriction within the stage K options, not only the full feature set. GP length scales are fixed; optimizer=None. PLS effective components are capped by available dimensions.',[.27,.47,.26]))
    sections.append(('Historical tuning: declared search spaces',
        ['Stage / model / parameter','Declared values or trial expression','Status'],
        [[r.stage+' / '+r.model+' / '+r.parameter,r.range_or_expression,'Historical exploration'] for r in search.itertuples()],
        'These early search declarations are not substituted for final Step 19 settings. Step 12: 40 random draws, 5-fold CV; Step 13: 35 random draws; Step 16: 45 TPE trials. Execution history and data versions must be distinguished.',[.27,.53,.20]))
    weighted=weights[weights.Method=='Selected_Weighted_4Model']
    weight_rows=[]
    for seed,sub in weighted.groupby('Seed',sort=False):
        weight_rows.append([str(seed),' ; '.join(f'{r.Model}={float(r.Weight):.6f}' for r in sub.itertuples()),str(sub.Selected_K.iloc[0])])
    sections.append(('Final random-split candidate ensemble: recorded seed-level weights',
        ['Seed','Recorded base-model weights','Selected K'],weight_rows,
        'Weights are not averaged into a fabricated final configuration. A zero weight is retained as observed. Four-model refers to the candidate pool; some splits may assign all weight to one model.',[.10,.75,.15]))
    pdfproducts,npages=render_tables(sections,outputs);products.extend(pdfproducts)
    stage_counts=catalog.groupby(['stage','model']).size().to_string()
    zero_seed=weighted.assign(nonzero=pd.to_numeric(weighted.Weight)>0).groupby('Seed').nonzero.sum().astype(int)
    report='\n'.join([
        '# 模型参数、搜索范围与逐种子入选配置：补充材料草稿','',
        '本次由项目现有脚本和结果表提取参数证据。固定模型参数、历史随机搜索、分组候选组合和逐种子入选配置分别报告，不重新训练模型，不依据测试误差重新选择参数。',
        'Step 19为当前随机划分基准的参数来源；Step 12、13、16属于前期探索，其结果不能替代Step 19。31H和31H2是不同轮次的分组调参，分别保留。',
        '', '## 搜索与选择口径','',
        'Step 19的种子为7、21、42、66、88；测试集比例为0.20，其余训练部分再划出0.25作为验证集。'
        '候选特征数K为12、18、25、35、45和all（全部特征）。基模型参数在该轮中固定，K与融合权重根据验证集结果选择。'
        '权重搜索包含1800组Dirichlet（狄利克雷）随机权重和单模型端点。最终入选K和权重直接读取已有逐种子结果。',
        'Step 27采用5折OOF（折外）预测构建元学习输入；线性元模型与RidgeCV（岭回归交叉验证）分开报告，后者的alpha候选为np.logspace(-4,4,25)。',
        '31H与31H2采用预先列出的参数组合。参数取值范围表仅为这些组合中各取值的并集，不能据此宣称做过完整笛卡尔积搜索。'
        '逐种子入选配置与代码候选ID和参数逐项交叉核对；既有测试指标原样保留，仅用作记录，不用于本次选择。',
        '31L的25组非树模型候选包含PLS（偏最小二乘）、KRR（核岭回归）和GPR（高斯过程回归）。'
        'PLS实际分量数可能受样本量和编码后维度限制；GPR核长度尺度固定且optimizer=None，不能写成进行了核超参数似然优化。'
        'GP允许K=12、18、25，多项式KRR允许K=12、18、25、35；其他候选仍受该轮全局K选项限制。',
        '', '## 参数表的使用边界','',
        '模型构造器未明确设置的参数不补写为当前软件版本的默认值。审计环境版本不等于历史训练版本。'
        '含seed、cfg.params等表达式的参数保留表达式，同时提供原脚本与行号；这些声明本身不证明相应分支实际运行。'
        'source_code_evidence保留其他已有建模、预处理、搜索和拟合选项，便于复核；不能把静态清单中的全部构造器都称为最终采用模型。',
        '随机划分四模型候选集的逐种子非零权重模型数为：'+str(zero_seed.to_dict())+'。零权重按原结果保留。',
        '文献编号仅用于分组和来源诊断，不是化学预测特征。本项不变更数据矩阵、模型性能、适用域规则、正文和主图。'
        '适用域阈值、敏感性及LOLO（留一文献外验证）结果将在后续对应补充材料中单列。',
        '', '## 本轮精确候选数量','',stage_counts,'',
        '完整文件包含：显式模型参数、两轮精确组合与取值并集、非树精确组合、历史搜索空间、随机入选设置与权重、分组入选设置、来源代码证据、源文件校验清单和入选配置交叉核验。',
        'PDF提供排版参数表，CSV保留完整机器可读记录；最终Table S编号在补充材料合并时统一。',''])
    p=outputs['reports']/(STEM+'_supplementary_text.md');p.write_text(report,encoding='utf-8');products.append(p)
    for rel,h in original_hashes.items():
        if sha(root/rel)!=h:raise RuntimeError(f'Input evidence changed during audit: {rel}')
    versions={}
    for package in ['numpy','pandas','matplotlib','scikit-learn','catboost','xgboost','lightgbm','optuna']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='not installed in audit runtime'
    log={'script':Path(__file__).name,'script_sha256':sha(Path(__file__)),'input_sha256':original_hashes,
        'inputs_unchanged':True,'audit_runtime_versions_not_historical_training_versions':versions,
        'n_source_scripts':len(manifest),'n_source_evidence_records':len(inventory),
        'n_exact_grouped_configs':len(catalog),'n_non_tree_configs':len(nt),'n_selection_crosschecks':len(checks),
        'pdf_pages':npages,'output_sha256':{str(p.relative_to(root)):sha(p) for p in products}}
    p=outputs['logs']/(STEM+'_audit.json');p.write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Exact grouped configurations:\n'+stage_counts)
    print(f'Non-tree configurations: {len(nt)}; selected configuration checks: {len(checks)}; PDF pages: {npages}')
    print(f'Source scripts: {len(manifest)}; evidence records: {len(inventory)}')
    print('Outputs: '+str(outputs['tables'].parent))
    print('Input hashes unchanged. No models were trained; no performance results were rewritten.')


if __name__=='__main__':
    started=time.perf_counter()
    try:main()
    finally:print(f'Total runtime: {time.perf_counter()-started:.2f} s')
