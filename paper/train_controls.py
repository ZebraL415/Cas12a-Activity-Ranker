"""Replay the accepted Target-fold classic baselines or same-capacity 2x2 controls."""
import argparse
import json
import os
import sys
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'1')
from settings import PAPER
sys.path.insert(0,str(PAPER/'engines/historical'))
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from paper_pipeline import matrices, make_models, XGB_PARAMS

def run(data,output,task,outer_fold):
    raw=pd.read_csv(data/'inputs/data/inputs/EasyDesign_2024_V2-2_core_context_feature_table.csv',low_memory=False)
    assignments=pd.read_csv(data/'inputs/data/group_assignments.csv')
    train=raw[raw.baseline_split.eq('baseline_train')].merge(assignments[['record_id','target_group','target_fold']],on='record_id',validate='one_to_one',sort=False)
    train['context_cv_fold']=train.target_fold
    features=pd.read_csv(data/'inputs/data/inputs/feature_manifest.csv').feature_name.tolist()
    accepted=pd.read_csv(data/'results/development/target_components_and_blocks.csv').set_index(['model_id','record_id'])
    output.mkdir(parents=True,exist_ok=True);rows=[];checks=[]
    for fold in range(5) if outer_fold is None else [outer_fold]:
        fit=train[train.target_fold.ne(fold)].reset_index(drop=True);query=train[train.target_fold.eq(fold)].reset_index(drop=True)
        query_features=query.copy();query_features['label_normalized']=np.nan
        m=matrices(fit,query_features,features);y=fit.label_normalized.to_numpy(float)
        if task=='classic':
            candidates={'ridge':make_pipeline(StandardScaler(),Ridge(alpha=1.)),
                        'svr':make_pipeline(StandardScaler(),SVR(C=1.,epsilon=.1,kernel='rbf',gamma='scale')),
                        'lightgbm':make_models()['lightgbm']}
            work=[(name,model,m['dtr'],m['dte'],y,0.) for name,model in candidates.items()]
        else:
            work=[]
            for name,use_mapping,use_history in [('block_sequence',False,False),('block_mapping',True,False),('block_history',False,True),('block_both',True,True)]:
                tr=[m['seqtr']];te=[m['seqte']]
                if use_history:tr.extend([m['gt'],m['mt']]);te.extend([m['ge'],m['me']])
                if use_mapping:tr.extend([m['nt'],m['ot']]);te.extend([m['ne'],m['oe']])
                model=XGBRegressor(**dict(XGB_PARAMS,max_depth=7,min_child_weight=8),random_state=42)
                work.append((name,model,pd.concat(tr,axis=1),pd.concat(te,axis=1),y-y.mean(),y.mean()))
        for name,model,xfit,xquery,target,anchor in work:
            print(f'START {task}/{fold}/{name}',flush=True)
            model.fit(xfit,target);prediction=anchor+model.predict(xquery)
            ref=accepted.loc[name].reindex(query.record_id).y_pred.to_numpy(float)
            error=float(np.max(abs(prediction.astype(float)-ref)))
            if error>1e-8:raise ValueError(f'{name} differs from accepted predictions: {error}')
            rows.append(pd.DataFrame({'record_id':query.record_id,'outer_fold':fold,'group_id':query.target_group,'model_id':name,'y_true':query.label_normalized,'y_pred':prediction}))
            checks.append({'model_id':name,'outer_fold':fold,'n':len(query),'max_absolute_delta':error,'passed':True})
    pd.concat(rows,ignore_index=True).to_csv(output/'predictions.csv',index=False)
    (output/'training_regression.json').write_text(json.dumps(checks,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--task',choices=['classic','blocks'],required=True);p.add_argument('--outer-fold',type=int,choices=range(5))
    a=p.parse_args();run(a.data_dir.resolve(),a.output_dir.resolve(),a.task,a.outer_fold)
