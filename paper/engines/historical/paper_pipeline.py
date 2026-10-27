"""Frozen-architecture paper evaluations. No release files are modified.

Run with final_repo_audit/test_env/bin/python3 and BLAS threads limited to one.
The original training helpers are imported read-only; every source is hashed.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
REPO = PROJECT / 'ranker_v1_5_worktree'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
sys.path.append(str(PROJECT/'architecture_search/lightgbm_packages'))
for p in [PROJECT/'last_week_experiments/scripts',
          PROJECT/'architecture_search/scripts', PROJECT/'parallel_model_optimization_audit/work']:
    sys.path.insert(0, str(p))
import numpy as np
import pandas as pd
import joblib, scipy, sklearn, xgboost, lightgbm
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from run_last_week_experiments import prepare, XGB_PARAMS, FOLD, LABEL
from run_guide_architecture import make_guide_target_encoding
from run_mapping_metadata_architecture import (crossfitted_mapping_te, mapping_onehot, numeric_mapping,
                                               CORE_CATEGORICAL, TEMPLATE_CATEGORICAL)
from run_sequence_pair_architecture import pair_onehot
from run_clean_heterogeneous import make_models

PROTOCOLS = ['target', 'guide', 'atomic_template', 'dual_atomic']
SEEDS = [7,21,42,84,2024]
MAP_COLUMNS = CORE_CATEGORICAL + TEMPLATE_CATEGORICAL
DATA = REPO/'data/processed/v2_2/EasyDesign_2024_V2-2_core_context_feature_table.csv'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def dump(p, d):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str)+'\n')

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    variable=np.std(p)>1e-14 and np.std(y)>1e-14
    return dict(spearman=float(spearmanr(y,p).statistic) if variable else np.nan,
                pearson=float(pearsonr(y,p).statistic) if variable else np.nan,
                rmse=float(np.mean((y-p)**2)**.5),mae=float(np.mean(abs(y-p))),
                r2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)) if np.std(y)>1e-14 else np.nan)

def load():
    if sha(DATA)!='39cda8368c216784507ac002df687b28a4f9cc6f81e2b0e84043e45eddb4c1c0':
        raise RuntimeError('Canonical data changed')
    data=pd.read_csv(DATA,low_memory=False)
    folds=pd.read_csv(REPO/'data/processed/v2_2/frozen_target_grouped_folds.csv')
    train=data[data.baseline_split.eq('baseline_train')].merge(
        folds[['record_id','target_group_id',FOLD]],on='record_id',validate='one_to_one')
    val=data[data.baseline_split.eq('baseline_validation')].copy()
    features=pd.read_csv(REPO/'data/processed/v2_2/feature_manifest.csv').feature_name.tolist()
    assert (len(train),len(val),len(features))==(8417,2217,188)
    assert not train.record_id.duplicated().any()
    return train.reset_index(drop=True),val.reset_index(drop=True),features

def atoms(value):
    if pd.isna(value) or str(value) in ('','missing'): return []
    result=[x.strip() for x in str(value).split(';') if x.strip()]
    if not all(x.isdigit() for x in result):
        raise ValueError('Invalid atomic template identifiers: '+str(value))
    return result

def components(frame, include_guide=False):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    for r in frame.itertuples():
        t='target:'+str(r.target_sequence)
        find(t)
        for a in atoms(r.mapping_candidate_template_nos):union(t,'template:'+a)
        if include_guide:union(t,'guide:'+str(r.crRNA_sequence))
    roots=[find('target:'+str(t)) for t in frame.target_sequence]
    names={r:f'component_{i:03d}' for i,r in enumerate(sorted(set(roots)))}
    return np.array([names[r] for r in roots])

def initialize():
    train,val,features=load()
    a=train[['record_id','target_sequence','crRNA_sequence','mapping_candidate_template_nos',FOLD]].copy()
    a['target_group']=train.target_group_id
    a['guide_group']=train.crRNA_sequence
    a['atomic_template_group']=components(train)
    a['dual_atomic_group']=components(train,True)
    assert a['atomic_template_group'].nunique()==21
    assert a['dual_atomic_group'].nunique()==20
    a['target_fold']=train[FOLD]
    for protocol in PROTOCOLS[1:]:
        a[protocol+'_fold']=-1
        for fold,(_,ix) in enumerate(GroupKFold(5).split(a,groups=a[protocol+'_group'])):
            a.loc[ix,protocol+'_fold']=fold
    audit=[]
    for protocol in PROTOCOLS:
        for fold in range(5):
            tr=a[a[protocol+'_fold'].ne(fold)];te=a[a[protocol+'_fold'].eq(fold)]
            row=dict(protocol_id=protocol,fold=fold,n_train=len(tr),n_test=len(te),
                     n_groups=te[protocol+'_group'].nunique())
            for k in ['record_id','target_sequence','crRNA_sequence']:
                row[k+'_overlap']=len(set(tr[k])&set(te[k]))
            atr=set(sum((atoms(v) for v in tr.mapping_candidate_template_nos),[]))
            ate=set(sum((atoms(v) for v in te.mapping_candidate_template_nos),[]))
            row['atomic_template_overlap']=len(atr&ate)
            assert row['record_id_overlap']==row['target_sequence_overlap']==0
            if protocol in ['guide','dual_atomic']:assert row['crRNA_sequence_overlap']==0
            if protocol in ['atomic_template','dual_atomic']:assert row['atomic_template_overlap']==0
            audit.append(row)
    out=ROOT/'02_protocols/splits'
    p=out/'assignments.csv'
    if p.exists():
        pd.testing.assert_frame_equal(pd.read_csv(p).astype(str),a.astype(str),check_dtype=False)
    else:a.to_csv(p,index=False)
    pd.DataFrame(audit).to_csv(out/'overlap_audit.csv',index=False)
    sources=[DATA,REPO/'data/processed/v2_2/frozen_target_grouped_folds.csv',
             REPO/'data/processed/v2_2/feature_manifest.csv',REPO/'results/v2_dbc/run_manifest.json',
             REPO/'results/v2_dbc/meta_oof_predictions.csv',REPO/'results/v2_dbc/fixed_validation_predictions.csv',
             REPO/'results/v2_dbc/v2_vs_v1_target_cluster_bootstrap.csv',Path(__file__)]
    for folder in ['last_week_experiments/scripts','architecture_search/scripts','parallel_model_optimization_audit/work']:
        sources.extend((PROJECT/folder).glob('*.py'))
    pd.DataFrame([dict(source_id=f'SRC-{i:03d}',path=str(p.relative_to(PROJECT)),sha256=sha(p),
                       size_bytes=p.stat().st_size,role='frozen_input_or_implementation') for i,p in enumerate(sources,1)]).to_csv(
                           ROOT/'01_sources/source_manifest.csv',index=False)
    config=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),stage='locked_before_new_evaluation',
                model_version='2.0.0',weights=dict(D=.2,B=.47,C=.33),D_weights=dict(xgb=.35,lightgbm=.59,mlp=.06),
                B=dict(XGB_PARAMS, max_depth=7,min_child_weight=8,seeds=SEEDS),
                C=dict(XGB_PARAMS,max_depth=7,min_child_weight=2,seed=42),
                protocols=PROTOCOLS,seed=42,bootstrap_seed=20260906,bootstrap_resamples=2000,
                generic_baselines=dict(ridge={'alpha':1.},svr={'C':1.,'epsilon':.1,'kernel':'rbf','gamma':'scale'},
                    random_forest={'n_estimators':500,'min_samples_leaf':2,'max_features':1.,'random_state':42}),
                D_parameter_source='parallel_model_optimization_audit/work/run_clean_heterogeneous.py::make_models',
                fitting_policy='No new model/weight search. Baseline defaults and frozen D component settings; unequal historical development budgets disclosed.',
                inner_history_encoding='Original target-group folds, restricted to outer train; guide histories leave-one-row-out within outer train.',
                preprocessing='Fit medians, active columns and categorical vocabulary on outer train only.',
                feature_policy='Frozen 188 candidate numeric features; active columns learned per fit; B/C use frozen feature-generation recipe.',
                inference='Internal validation conditional on developed architecture; no selection-nested or external claim.',
                family='target: full_v2 minus six generic baselines and D; SCC/PCC paired cluster permutation, Holm over 14 contrasts; CIs descriptive.',
                missing='Unsupported baseline inputs retained with explicit prediction_status. Compare common supported subsets.',
                versions={n:getattr(m,'__version__','') for n,m in [('numpy',np),('pandas',pd),('scipy',scipy),('sklearn',sklearn),('xgboost',xgboost),('lightgbm',lightgbm),('joblib',joblib)]},
                python=sys.version)
    cp=ROOT/'02_protocols/configs/frozen_evaluation.json'
    if not cp.exists():dump(cp,config)
    summary=dict(n_train=len(train),n_historical_validation=len(val),n_excluded_external=1358,
                 groups={p:int(a[p+'_group'].nunique()) for p in PROTOCOLS},
                 normalized_equals_supplied_raw=bool(np.allclose(train[LABEL],train.label_raw_value)),
                 label_range=[float(train[LABEL].min()),float(train[LABEL].max())],
                 exact_target_multi_guide=int((train.groupby('target_sequence').crRNA_sequence.nunique()>1).sum()))
    dump(ROOT/'01_sources/data_audit.json',summary)
    print(json.dumps(summary),flush=True)

def matrices(tr,te,features):
    prepared=prepare(tr[features].apply(pd.to_numeric,errors='coerce'),te[features].apply(pd.to_numeric,errors='coerce'))
    pairtr=pair_onehot(tr);pairte=pair_onehot(te)
    pairtr=pairtr.loc[:,pairtr.nunique(dropna=False).gt(1)]
    pairte=pairte.reindex(columns=pairtr.columns,fill_value=0)
    gt,ge=make_guide_target_encoding(tr,te)
    mt,me=crossfitted_mapping_te(tr,te,MAP_COLUMNS,False)
    ot,oe=mapping_onehot(tr,te,MAP_COLUMNS)
    nt,ne=numeric_mapping(tr),numeric_mapping(te)
    seqtr=pd.concat([prepared.train,pairtr],axis=1);seqte=pd.concat([prepared.validation,pairte],axis=1)
    fulltr=pd.concat([prepared.train,gt,pairtr,nt,mt,ot],axis=1)
    fullte=pd.concat([prepared.validation,ge,pairte,ne,me,oe],axis=1)
    assert fulltr.columns.equals(fullte.columns)
    assert np.isfinite(fulltr.to_numpy()).all() and np.isfinite(fullte.to_numpy()).all()
    return dict(dtr=prepared.train,dte=prepared.validation,fulltr=fulltr,fullte=fullte,
                seqtr=seqtr,seqte=seqte,gt=gt,ge=ge,mt=mt,me=me,ot=ot,oe=oe,nt=nt,ne=ne)

def evaluate(protocol,run_id=None):
    train,val,features=load()
    assignments=pd.read_csv(ROOT/'02_protocols/splits/assignments.csv')
    assert assignments.record_id.tolist()==train.record_id.tolist()
    run_id=run_id or time.strftime('%Y%m%dT%H%M%S')+'_'+protocol
    out=ROOT/'03_experiments'/f'EXP-001_{protocol}'/run_id
    if out.exists():raise FileExistsError('Runs are append-only: '+str(out))
    out.mkdir(parents=True)
    start=time.time()
    dump(out/'status.json',dict(status='running',started=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    dump(out/'manifest.json',dict(run_id=run_id,protocol_id=protocol,data_sha256=sha(DATA),
         config_sha256=sha(ROOT/'02_protocols/configs/frozen_evaluation.json'),script_sha256=sha(__file__),
         assignments_sha256=sha(ROOT/'02_protocols/splits/assignments.csv'),python=sys.executable))
    def log(s):
        line=time.strftime('%H:%M:%S')+' '+s
        print(line,flush=True)
        with (out/'run.log').open('a') as f:f.write(line+'\n')
    allpred=[];foldmetrics=[];details=[]
    try:
        for fold in range(5):
            ix=assignments[protocol+'_fold'].eq(fold).to_numpy()
            tr=train.loc[~ix].copy();te=train.loc[ix].copy()
            # Holdout labels are removed from the feature path and used for metrics only.
            te_features=te.copy();te_features[LABEL]=np.nan
            m=matrices(tr,te_features,features);y=tr[LABEL].to_numpy(float)
            pred={};t=time.time()
            for name,model in make_models().items():
                model.fit(m['dtr'],y)
                pred[name]=np.asarray(model.predict(m['dte']),float)
                log(f'{protocol} fold {fold}: {name} fitted')
            pred['D']=.35*pred['xgb']+.59*pred['lightgbm']+.06*pred['mlp']
            guide_tr=m['gt'].te_guide_all.to_numpy(float);guide_te=m['ge'].te_guide_all.to_numpy(float)
            temptr=m['mt'].mapping_te_mapping_candidate_template_nos.to_numpy(float)
            tempte=m['me'].mapping_te_mapping_candidate_template_nos.to_numpy(float)
            b=[]
            for seed in SEEDS:
                model=XGBRegressor(**dict(XGB_PARAMS,max_depth=7,min_child_weight=8),random_state=seed)
                model.fit(m['fulltr'],y-guide_tr)
                b.append(guide_te+model.predict(m['fullte']))
                log(f'{protocol} fold {fold}: B seed {seed} fitted')
            pred['B']=np.mean(b,axis=0)
            model=XGBRegressor(**dict(XGB_PARAMS,max_depth=7,min_child_weight=2),random_state=42)
            model.fit(m['fulltr'],y-(guide_tr+temptr)/2)
            pred['C']=(guide_te+tempte)/2+model.predict(m['fullte'])
            pred['DBC_unrouted']=.2*pred['D']+.47*pred['B']+.33*pred['C']
            miss=te.mapping_hit_count.fillna(0).eq(0).to_numpy()
            pred['full_v2']=np.where(miss,pred['D'],pred['DBC_unrouted'])
            pred['without_D']=(.47*pred['B']+.33*pred['C'])/.8
            pred['without_B']=(.2*pred['D']+.33*pred['C'])/.53
            pred['without_C']=(.2*pred['D']+.47*pred['B'])/.67
            pred['global_mean']=np.full(len(te),y.mean())
            pred['guide_mean']=guide_te;pred['template_mean']=tempte
            pred['guide_template_mean']=(guide_te+tempte)/2
            if protocol=='target':
                generic=dict(ridge=make_pipeline(StandardScaler(),Ridge(alpha=1.)),
                    svr=make_pipeline(StandardScaler(),SVR(C=1.,epsilon=.1,kernel='rbf',gamma='scale')),
                    random_forest=RandomForestRegressor(n_estimators=500,min_samples_leaf=2,max_features=1.,random_state=42,n_jobs=4))
                for name,model in generic.items():
                    model.fit(m['dtr'],y);pred[name]=model.predict(m['dte'])
                    log(f'{protocol} fold {fold}: {name} fitted')
                # 2x2 information-block ablation: same tree capacity and fitting target in all arms.
                for name,usemap,usehist in [('block_sequence',False,False),('block_mapping',True,False),
                                          ('block_history',False,True),('block_both',True,True)]:
                    atr=[m['seqtr']];ate=[m['seqte']]
                    if usehist:atr.extend([m['gt'],m['mt']]);ate.extend([m['ge'],m['me']])
                    if usemap:atr.extend([m['nt'],m['ot']]);ate.extend([m['ne'],m['oe']])
                    xtr=pd.concat(atr,axis=1);xte=pd.concat(ate,axis=1)
                    model=XGBRegressor(**dict(XGB_PARAMS,max_depth=7,min_child_weight=8),random_state=42)
                    model.fit(xtr,y-y.mean());pred[name]=y.mean()+model.predict(xte)
                    log(f'{protocol} fold {fold}: {name} fitted')
            base=pd.DataFrame(dict(record_id=te.record_id.to_numpy(),dataset_id='baseline_train',protocol_id=protocol,
                 fold=fold,y_true=te[LABEL].to_numpy(float),isolation_group=assignments.loc[ix,protocol+'_group'].to_numpy(),
                 guide_reference_count=np.expm1(m['ge'].te_guide_all_log_count.to_numpy()),
                 template_reference_count=np.expm1(m['me'].mapping_te_mapping_candidate_template_nos_log_count.to_numpy()),
                 prediction_route=np.where(miss,'sequence_fallback','mapping_aware')))
            for name,p in pred.items():
                assert np.isfinite(p).all(),name
                f=base.copy();f['model_id']=name;f['y_pred']=p;f['prediction_status']='ok'
                allpred.append(f);foldmetrics.append(dict(protocol_id=protocol,fold=str(fold),model_id=name,n=len(te),**metrics(f.y_true,p)))
            details.append(dict(fold=fold,n_train=len(tr),n_test=len(te),d_features=m['dtr'].shape[1],
                                mapping_features=m['fulltr'].shape[1],seconds=time.time()-t))
            pd.concat(allpred).to_csv(out/'predictions.partial.csv',index=False)
            log(f'{protocol} fold {fold} complete; full_v2 SCC {metrics(te[LABEL],pred["full_v2"])["spearman"]:.6f}')
        result=pd.concat(allpred,ignore_index=True)
        assert not result.duplicated(['record_id','model_id']).any()
        result.to_csv(out/'predictions.csv',index=False)
        (out/'predictions.partial.csv').unlink()
        for name,f in result.groupby('model_id'):
            assert len(f)==8417
            foldmetrics.append(dict(protocol_id=protocol,fold='pooled',model_id=name,n=len(f),**metrics(f.y_true,f.y_pred)))
        pd.DataFrame(foldmetrics).to_csv(out/'metrics.csv',index=False)
        dump(out/'fit_details.json',details)
        dump(out/'status.json',dict(status='complete',seconds=time.time()-start,n_predictions=len(result),
                                   prediction_sha256=sha(out/'predictions.csv')))
        (out/'experiment_log.md').write_text(f'---\nexperiment_id: EXP-001_{protocol}\nrun_id: {run_id}\nstatus: complete\nsample_batch: EasyDesign_baseline_train\n---\n\nFrozen architecture evaluation. See manifest.json, fit_details.json and metrics.csv.\n')
        log(f'COMPLETE {protocol}: {time.time()-start:.1f}s')
    except Exception as e:
        dump(out/'status.json',dict(status='failed',error=str(e),traceback=traceback.format_exc()))
        raise

def verify_features():
    tr,_,features=load();a=pd.read_csv(ROOT/'02_protocols/splits/assignments.csv')
    fit=tr.loc[a.target_fold.ne(0)].copy();test=tr.loc[a.target_fold.eq(0)].copy()
    m=matrices(fit,test,features)
    perturbed=test.copy();perturbed[LABEL]=1e6+np.arange(len(test))
    m2=matrices(fit,perturbed,features)
    for key in m:pd.testing.assert_frame_equal(m[key],m2[key])
    dump(ROOT/'08_review/statistics/holdout_label_perturbation.json',dict(status='pass',
         check='Changing held-out labels leaves every train/holdout feature and anchor unchanged',matrices=list(m)))
    print('Feature holdout-label perturbation check PASS',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['initialize','verify','run']);ap.add_argument('--protocol',choices=PROTOCOLS)
    args=ap.parse_args()
    if args.action=='initialize':initialize()
    elif args.action=='verify':verify_features()
    else:evaluate(args.protocol)
