"""Portable adapter for the original L02 training code and its original intact controls."""
from pathlib import Path
import json
import sys
import importlib.util
import numpy as np
import pandas as pd
from settings import PAPER

def run(data, output, kind, seed, outer_fold):
    spec=importlib.util.spec_from_file_location('l02',PAPER/'engines/controls.py')
    l02=importlib.util.module_from_spec(spec);spec.loader.exec_module(l02)
    source=pd.read_csv(data/'inputs/data/inputs/EasyDesign_2024_V2-2_core_context_feature_table.csv',low_memory=False)
    assignments=pd.read_csv(data/'inputs/data/group_assignments.csv')
    assignments=assignments[['record_id','target_group','target_fold']].rename(columns={'target_group':'target_group_id','target_fold':'context_cv_fold'})
    train=source[source.baseline_split.eq('baseline_train')].merge(assignments,on='record_id',validate='one_to_one',sort=False)
    features=pd.read_csv(data/'inputs/data/inputs/feature_manifest.csv').feature_name.tolist()
    accepted=pd.read_csv(data/'results/development/target_components_and_blocks.csv')
    baseline={name:accepted[accepted.model_id.eq(name)].set_index('record_id').reindex(train.record_id).y_pred.to_numpy(float) for name in ('D','B','C','full_v2','block_mapping')}
    assert all(np.isfinite(x).all() for x in baseline.values())
    l02.PERMUTATION_SEEDS=(seed,)
    l02.OUTER_FOLDS=(outer_fold,) if outer_fold is not None else tuple(range(5))
    # Original post-run checks refer to the full run size. A fold check uses the
    # exact fitted fold size without changing any features, RNG or learner.
    original_n=l02.N_RECORDS
    if outer_fold is not None: l02.N_RECORDS=int(train.context_cv_fold.eq(outer_fold).sum())
    fn=l02.fit_history_shuffle if kind=='history_shuffle' else l02.fit_mapping_permutation
    result,metrics,checks=fn(train,features,baseline)
    output.mkdir(parents=True,exist_ok=True)
    result.to_csv(output/'predictions.csv',index=False);metrics.to_csv(output/'metrics.csv',index=False)
    model='tricast_full' if kind=='history_shuffle' else 'sequence_mapping_xgb'
    expected=pd.read_csv(data/f'results/perturbations/predictions/pred__{model}__{kind}__seed{seed}.csv').set_index('record_id')
    name='full_v2_shuffled' if kind=='history_shuffle' else 'block_mapping_permuted'
    current=result[result.model_id.eq(name)].set_index('record_id')
    left=current.y_pred.to_numpy(float)
    right=expected.reindex(current.index).pred_y.to_numpy(float)
    if kind=='mapping_permutation':
        left=left.astype(np.float32).astype(float);right=right.astype(np.float32).astype(float)
    error=float(np.max(np.abs(left-right)))
    checks.update(max_absolute_prediction_error=error,passed=error<=1e-8,seed=seed,outer_fold=outer_fold,perturbation=kind)
    (output/'regression.json').write_text(json.dumps(checks,indent=2)+'\n')
    if error>1e-8: raise ValueError(f'L02 prediction mismatch {error}')
    print(json.dumps(checks,indent=2),flush=True)
