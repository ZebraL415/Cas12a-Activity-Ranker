"""Public training adapter; the accepted features, learners and selector are unchanged."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')
from settings import PAPER, settings, read_json
sys.path.insert(0, str(PAPER/'engines/fixed/src'))
sys.path.insert(0, str(PAPER/'engines/nested/src'))
import numpy as np
import pandas as pd
from models import resolved_recipe
from selection import select_policy, predictions_for_outer
from splits import make_nested_splits

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','output-dir','cohort','protocol','kind'): p.add_argument('--'+name, required=True)
    p.add_argument('--outer-fold', type=int)
    p.add_argument('--tf-python', type=Path)
    p.add_argument('--seed', type=int, default=137)
    p.add_argument('--perturbation', default='history_shuffle')
    a = p.parse_args()
    data, output = Path(a.data_dir).resolve(), Path(a.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if a.kind == 'perturbation':
        if (a.cohort,a.protocol)!=('core_dev','target'):
            p.error('The accepted L02 perturbations use core_dev / target')
        from perturbations import run
        run(data, output, a.perturbation, a.seed, a.outer_fold)
        return
    config = settings(data)
    expected = read_json(PAPER/'config/frozen_recipe_resolved.json')
    if resolved_recipe() != expected: raise ValueError('Use the pinned core environment: resolved learner parameters differ')
    runner = module('nested_training', PAPER/'engines/nested/run_nested.py')
    frame, assignments, manifest = runner._load_cohort(a.cohort, data/'inputs', config)
    plan = read_json(PAPER/'config/selection.json')
    fixed = runner._fixed_recipe(resolved_recipe())
    predictions = []
    folds = range(5) if a.outer_fold is None else [a.outer_fold]
    for fold in folds:
        destination = output / a.cohort / a.protocol / f'outer_{fold}'
        destination.mkdir(parents=True, exist_ok=True)
        fit, query, splits = make_nested_splits(frame, a.protocol, fold)
        inner = []
        if a.kind == 'nested':
            for split in splits:
                inner_fold = int(split['inner_fold'])
                folder = destination / f'inner_{inner_fold}'
                folder.mkdir(exist_ok=True)
                split['fit'][['record_id']].to_csv(folder/'fit_ids.csv', index=False)
                split['validation'][['record_id']].to_csv(folder/'validation_ids.csv', index=False)
                print(f'START {a.cohort}/{a.protocol}/{fold}/inner_{inner_fold}', flush=True)
                component, seconds = runner._fit_one_component_table(split['fit'], split['validation'], assignments, manifest, a.protocol, a.cohort, fold, inner_fold, config)
                component.to_csv(folder/'components.csv', index=False)
                inner.append(component)
                print(f'COMPLETE inner_{inner_fold}: {seconds:.1f}s', flush=True)
            selected, scores = select_policy(inner, plan)
            scores.to_csv(destination/'candidate_scores.csv', index=False)
        else:
            selected = {'d_weights':plan['d_original'], 'db_weights':plan['db_original'], 'tri_weights':plan['tri_original'], 'selected_family':'TriCasRank'}
        (destination/'selection.json').write_text(json.dumps(selected, indent=2)+'\n')
        fit[['record_id']].to_csv(destination/'outer_fit_ids.csv', index=False)
        query[['record_id']].to_csv(destination/'outer_query_ids.csv', index=False)
        print(f'START {a.cohort}/{a.protocol}/{fold}/outer', flush=True)
        components, seconds = runner._fit_one_component_table(fit, query, assignments, manifest, a.protocol, a.cohort, fold, None, config)
        components.to_csv(destination/'outer_components.csv', index=False)
        easy_rows = None
        if a.cohort == 'common_dev':
            if not a.tf_python: p.error('common_dev training requires --tf-python for stopping and reset/refit')
            fit.to_csv(destination/'ed_fit.csv', index=False)
            runner._query_without_labels(query).to_csv(destination/'ed_query.csv', index=False)
            (destination/'ed_config.json').write_text(json.dumps(config, indent=2))
            command = [str(a.tf_python), str(PAPER/'train_easydesign.py'), '--fit', str(destination/'ed_fit.csv'),
                       '--query', str(destination/'ed_query.csv'), '--config', str(destination/'ed_config.json'),
                       '--output', str(destination/'easydesign'), '--protocol', a.protocol, '--outer-fold', str(fold)]
            subprocess.run(command, check=True)
            easy_rows = query[['record_id', 'label_normalized', f'{a.protocol}_group']].rename(columns={'label_normalized':'y_true', f'{a.protocol}_group':'group_id'}).merge(pd.read_csv(destination/'easydesign/predictions.csv'), on='record_id', validate='one_to_one')
        rows = runner._prediction_rows(components, query, selected, fixed, a.cohort, a.protocol, fold, easy_rows)
        rows['evaluation_kind'] = 'nested_grouped_selection' if a.kind=='nested' else 'fixed_recipe_grouped_cv'
        if a.kind=='fixed':
            aliases={'D_fixed':'D','TriCasRank_fixed':'TriCasRank','EasyDesign-adapted':'EasyDesign-adapted'}
            if a.cohort=='core_dev':
                aliases.update({'D_B_nested':'D_B_FIXED','HISTORY_PRIOR':'HISTORY_PRIOR','SEQ_MAPPING_XGB':'SEQ_MAPPING_XGB'})
            rows=rows[rows.model_id.isin(aliases)].copy()
            rows['model_id']=rows.model_id.map(aliases)
            rows['selection_id']='fixed_DBC_020_047_033'
            rows['selected_family']=''
        rows.to_csv(destination/'predictions.csv', index=False)
        predictions.append(rows)
        # Check all component predictions against the corresponding accepted outer fit.
        reference_path = data/'nested/runs'/a.cohort/a.protocol/f'outer_{fold}/outer_components.csv'
        if reference_path.exists():
            reference = pd.read_csv(reference_path).set_index('record_id')
            current = components.set_index('record_id').reindex(reference.index)
            errors = {name:float(np.max(np.abs(current[name].to_numpy(float)-reference[name].to_numpy(float)))) for name in runner.COMPONENT_COLUMNS[:7]}
            (destination/'training_regression.json').write_text(json.dumps({'max_absolute_deltas':errors, 'tolerance':1e-8, 'passed':all(x<=1e-8 for x in errors.values())}, indent=2))
        print(f'COMPLETE outer {fold}: {seconds:.1f}s', flush=True)
    pd.concat(predictions, ignore_index=True).to_csv(output/'predictions.csv', index=False)

if __name__ == '__main__': main()
