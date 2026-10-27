"""Recalculate the paper from saved predictions, using the original random streams."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name,'1')
from settings import PAPER, PROTOCOLS, COHORTS, read_json
sys.path[:0] = [str(PAPER/'engines/fixed/src'), str(PAPER/'engines/nested/src')]
import numpy as np
import pandas as pd
from revision_statistics.metrics import basic_metrics
from revision_statistics.analyzer import _comparison_rows_for_complete, _calibrate_holm
from selection import select_policy, predictions_for_outer

def load(path): return pd.read_csv(path, dtype={'record_id':str, 'group_id':str, 'cluster_id':str})

def validate(frame, data):
    labels=load(data/'inputs/data/development_labels.csv').set_index('record_id').y_true
    assignments=load(data/'inputs/data/group_assignments.csv').set_index('record_id')
    common=set(load(data/'inputs/data/common_record_ids.csv').record_id)
    checks=[]
    for (cohort, protocol, model), part in frame.groupby(['cohort_id','protocol_id','model_id'], sort=False):
        expected=set(labels.index) if cohort=='core_dev' else common
        assert len(part)==len(expected) and set(part.record_id)==expected, (cohort,protocol,model,'coverage')
        assert not part.record_id.duplicated().any()
        assert set(part.outer_fold.astype(int))==set(range(5))
        assert np.array_equal(part.record_id.map(assignments[f'{protocol}_fold']).to_numpy(int), part.outer_fold.to_numpy(int))
        assert np.array_equal(part.record_id.map(assignments[f'{protocol}_group']).astype(str).to_numpy(),part.group_id.to_numpy(str))
        assert np.allclose(part.record_id.map(labels),part.y_true,atol=1e-12,rtol=0)
        assert np.isfinite(part.y_pred).all()
        checks.append({'cohort_id':cohort,'protocol_id':protocol,'model_id':model,'n':len(part),'groups':part.group_id.nunique(),'passed':True})
    return checks

def points(frame, kind):
    rows=[]
    for (cohort,protocol,model), part in frame.groupby(['cohort_id','protocol_id','model_id'],sort=False):
        for fold in [None,0,1,2,3,4]:
            sub=part if fold is None else part[part.outer_fold.eq(fold)]
            rows.append({'evaluation_kind':kind,'cohort_id':cohort,'protocol_id':protocol,'model_id':model,
                         'evaluation_level':'pooled_oof' if fold is None else 'outer_fold','outer_fold':fold,
                         'record_n':len(sub),'group_n':sub.group_id.nunique(),**basic_metrics(sub.y_true,sub.y_pred)})
    return rows

def reconstruct(data, output):
    config=read_json(PAPER/'config/selection.json')
    recipe={'xgb':.35,'lightgbm':.59,'mlp':.06,'fusion':{'D':.20,'B':.47,'C':.33},'db_fusion':{'D':20/67,'B':47/67}}
    accepted=load(data/'results/nested/predictions_oof.csv')
    checks=[]
    for folder in sorted((data/'nested/runs').glob('*/*/outer_*')):
        cohort,protocol,outer=folder.parts[-3:]
        # Recover the exact in-memory doubles used when the CSVs were saved.
        # The default fast parser can split a prediction tie in an unselected
        # grid candidate, changing its Spearman by a few 1e-7.
        inner=[pd.read_csv(f,float_precision='round_trip',dtype={'record_id':str,'group_id':str}) for f in sorted(folder.glob('inner_*/components.csv'))]
        assert len(inner)==4
        outer_ids=set(load(folder/'outer_query_ids.csv').record_id)
        fit_ids=set(load(folder/'outer_fit_ids.csv').record_id)
        assert outer_ids.isdisjoint(fit_ids)
        assignment=load(data/'inputs/data/group_assignments.csv').set_index('record_id')
        groups=assignment[f'{protocol}_group'].astype(str)
        assert set(groups.reindex(list(outer_ids))).isdisjoint(groups.reindex(list(fit_ids)))
        for f,t in zip(sorted(folder.glob('inner_*/components.csv')),inner):
            inner_fit=set(load(f.parent/'fit_ids.csv').record_id)
            inner_query=set(t.record_id)
            assert inner_query<=fit_ids and inner_fit<=fit_ids and inner_fit.isdisjoint(inner_query)
            assert set(groups.reindex(list(inner_fit))).isdisjoint(groups.reindex(list(inner_query)))
        selected,scores=select_policy(inner,config)
        old=read_json(folder/'selection.json')
        for name in ('d_weights','db_weights','tri_weights','selected_family'):
            assert selected[name]==old[name], (cohort,protocol,outer,name)
        old_scores=pd.read_csv(folder/'candidate_scores.csv',float_precision='round_trip')
        assert len(scores)==len(old_scores)
        for name in ('SCC','MAE','mean_SCC','mean_MAE','ranking_value'):
            assert np.allclose(pd.to_numeric(scores[name]),pd.to_numeric(old_scores[name]),atol=1e-10,rtol=0,equal_nan=True)
        components=pd.read_csv(folder/'outer_components.csv',float_precision='round_trip',dtype={'record_id':str,'group_id':str})
        prediction=predictions_for_outer(components,selected,recipe)
        errors=[]
        for name,(values,routes) in prediction.items():
            ref=accepted[(accepted.cohort_id==cohort)&(accepted.protocol_id==protocol)&(accepted.outer_fold==int(outer.split('_')[1]))&(accepted.model_id==name)].set_index('record_id')
            assert set(ref.index)==set(components.record_id)
            error=float(np.max(np.abs(np.asarray(values)-ref.reindex(components.record_id).y_pred.to_numpy(float))))
            assert error<=1e-10
            errors.append(error)
        checks.append({'cohort_id':cohort,'protocol_id':protocol,'outer_fold':int(outer.split('_')[1]),
                       'selected_family':selected['selected_family'],'d_weights':json.dumps(selected['d_weights']),
                       'db_weights':json.dumps(selected['db_weights']),'tri_weights':json.dumps(selected['tri_weights']),
                       'inner_fits':4,'candidate_rows':len(scores),'max_prediction_error':max(errors),'passed':True})
        print(f'Reconstructed {cohort}/{protocol}/{outer}',flush=True)
    assert len(checks)==40
    pd.DataFrame(checks).to_csv(output/'selection_reconstruction.csv',index=False)
    pd.DataFrame(checks).groupby(['cohort_id','protocol_id','selected_family']).size().rename('outer_folds').reset_index().to_csv(output/'selection_frequency.csv',index=False)

def perturbations(data, output):
    from scipy.stats import spearmanr, pearsonr
    rows=[]
    for family,model in [('history_shuffle','tricast_full'),('mapping_permutation','sequence_mapping_xgb')]:
        base=pd.read_csv(data/f'results/perturbations/predictions/pred__{model}__intact.csv',float_precision='round_trip').set_index('record_id')
        for seed in (137,271,409):
            frame=pd.read_csv(data/f'results/perturbations/predictions/pred__{model}__{family}__seed{seed}.csv',float_precision='round_trip').sort_values(['fold','record_id'],kind='stable')
            y=frame.true_y.to_numpy(float); pred=frame.pred_y.to_numpy(float); intact=base.reindex(frame.record_id).pred_y.to_numpy(float)
            # XGBoost's original mapping-permutation output was float32.
            # Its CSV uses the shortest float32 representation; recovering that
            # dtype restores the original values before float64 statistics.
            if family=='mapping_permutation': pred=pred.astype(np.float32).astype(float)
            assert np.allclose(y,base.reindex(frame.record_id).true_y,atol=1e-12,rtol=0)
            codes,groups=pd.factorize(frame.cluster_id,sort=True)
            rng=np.random.default_rng(20260906)
            draws=np.zeros((2000,5)); indices=np.arange(len(frame))
            # Keep the L02 int64 RNG stream and explicit expanded-row ranking.
            for i in range(2000):
                sampled=rng.integers(0,len(groups),size=len(groups))
                idx=np.repeat(indices,np.bincount(sampled,minlength=len(groups))[codes])
                pert=basic_metrics(y[idx],pred[idx]); original=basic_metrics(y[idx],intact[idx])
                draws[i]=[pert[k]-original[k] for k in ('SCC','PCC','RMSE','MAE','R2')]
            pa,pb=basic_metrics(y,pred),basic_metrics(y,intact)
            for j,metric in enumerate(('SCC','PCC','RMSE','MAE','R2')):
                lo,hi=np.quantile(draws[:,j],[.025,.975])
                rows.append({'perturbation':family,'seed':seed,'metric':metric,'estimate':pa[metric],'baseline_estimate':pb[metric],
                             'delta':pa[metric]-pb[metric],'ci_low':lo,'ci_high':hi,'n':len(frame),'group_n':len(groups),
                             'bootstrap_draws':2000,'bootstrap_seed':20260906,'direction':'perturbed minus own intact'})
            print(f'Resampled {family}/{seed}',flush=True)
    pd.DataFrame(rows).to_csv(output/'perturbation_statistics.csv',index=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True); parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--points-only',action='store_true');parser.add_argument('--skip-figures',action='store_true')
    args=parser.parse_args(); data=args.data_dir.resolve(); output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    fixed=load(data/'results/fixed/oof_predictions.csv'); nested=load(data/'results/nested/predictions_oof.csv')
    pd.DataFrame(validate(fixed,data)+validate(nested,data)).to_csv(output/'coverage.csv',index=False)
    fixed_rows=points(fixed,'fixed_recipe_grouped_cv');nested_rows=points(nested,'nested_grouped_selection')
    pd.DataFrame(fixed_rows).to_csv(output/'fixed_metrics.csv',index=False);pd.DataFrame(nested_rows).to_csv(output/'nested_metrics.csv',index=False)
    extra=[]
    for f in sorted((data/'results/classics').glob('pred__*.csv')):
        t=load(f); t=t.rename(columns={'fold':'outer_fold','true_y':'y_true','pred_y':'y_pred','cluster_id':'group_id'})
        t['cohort_id']='core_dev';t['protocol_id']='target';t['model_id']=t['model']
        extra.extend(points(t,'fixed_recipe_grouped_cv'))
    pd.DataFrame(extra).to_csv(output/'classic_metrics.csv',index=False)
    development=load(data/'results/development/target_components_and_blocks.csv')
    descriptive=[]
    for model,part in development.groupby('model_id',sort=False):
        for fold in [None,0,1,2,3,4]:
            sub=part if fold is None else part[part.fold.eq(fold)]
            descriptive.append({'model_id':model,'outer_fold':fold,'n':len(sub),'evaluation_kind':'development_target_oof',**basic_metrics(sub.y_true,sub.y_pred)})
    pd.DataFrame(descriptive).to_csv(output/'development_components_and_blocks.csv',index=False)
    historical=[]
    for f in sorted((data/'results/historical').glob('pred__*.csv')):
        part=pd.read_csv(f)
        assert len(part)==2217 and part.record_id.is_unique
        historical.append({'model':part.model.iloc[0],'evaluation_kind':'non_independent_historical_fixed_validation','n':len(part),**basic_metrics(part.true_y,part.pred_y)})
    if historical:pd.DataFrame(historical).to_csv(output/'historical_metrics.csv',index=False)
    accepted=load(data/'results/accepted/fixed/metrics_pooled_and_fold.csv')
    current=pd.DataFrame(fixed_rows)
    joined=current.merge(accepted,on=['cohort_id','protocol_id','model_id','evaluation_level','outer_fold'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(joined)==len(accepted)
    errors={m:float(np.nanmax(np.abs(joined[m+'_new']-joined[m+'_old']))) for m in ('SCC','PCC','RMSE','MAE','R2')}
    assert max(errors.values())<=1e-10, errors
    (output/'metric_regression.json').write_text(json.dumps({'fixed_metric_rows':len(joined),'max_absolute_deltas':errors,'passed':True},indent=2))
    if not args.points_only:
        reconstruct(data,output)
        comparisons=[]
        for cohort in COHORTS:
            models=('D','HISTORY_PRIOR','SEQ_MAPPING_XGB','D_B_FIXED','TriCasRank') if cohort=='core_dev' else ('D','TriCasRank','EasyDesign-adapted')
            for protocol in PROTOCOLS:
                print(f'Fixed inference {cohort}/{protocol}',flush=True)
                comparisons.extend(_comparison_rows_for_complete(fixed[(fixed.cohort_id==cohort)&(fixed.protocol_id==protocol)],models,draws=2000,seed=20261012,minimum_valid_fraction=.95,cohort_id=cohort,protocol_id=protocol))
        _calibrate_holm(comparisons,16,4)
        pd.DataFrame(comparisons).to_csv(output/'fixed_paired_comparisons.csv',index=False)
        spec=importlib.util.spec_from_file_location('nested_analysis',PAPER/'engines/analyze_nested.py')
        analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
        analysis.score_and_compare(nested,output/'nested_statistics')
        perturbations(data,output)
    # Tables and figures use the same point estimates; resampling uses its original lanes.
    from tables import build
    build(data,output)
    if not args.skip_figures:
        from figures import draw
        draw(data,output)
    print(json.dumps({'output':str(output),'fixed_rows':len(fixed),'nested_rows':len(nested),'metric_regression':errors},indent=2))

if __name__=='__main__': main()
