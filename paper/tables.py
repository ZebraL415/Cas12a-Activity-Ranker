"""Build small source tables and the six-sheet S6 workbook with public dependencies."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from revision_statistics.metrics import basic_metrics

NAMES={'ridge':'Ridge','svr':'SVR','lightgbm':'LightGBM','HISTORY_PRIOR':'History prior','SEQ_MAPPING_XGB':'Sequence + mapping XGBoost','TriCasRank':'TriCasRank','D':'D'}

def build(data, output):
    tables=output/'tables';tables.mkdir(parents=True,exist_ok=True)
    fixed=pd.read_csv(output/'fixed_metrics.csv');classic=pd.read_csv(output/'classic_metrics.csv')
    all_models=pd.concat([fixed,classic],ignore_index=True)
    seven=('ridge','svr','lightgbm','D','HISTORY_PRIOR','SEQ_MAPPING_XGB','TriCasRank')
    target=all_models[(all_models.cohort_id=='core_dev')&(all_models.protocol_id=='target')&all_models.model_id.isin(seven)]
    main=target[target.evaluation_level=='pooled_oof'].copy()
    main['model']=main.model_id.map(NAMES)
    main.to_csv(tables/'Table_3_primary_models.csv',index=False)
    fixed[fixed.evaluation_level=='pooled_oof'].to_csv(tables/'Table_5_identity_and_matched_results.csv',index=False)
    folds=target[target.evaluation_level=='outer_fold'].copy()
    assert len(folds)==35
    folds['model']=folds.model_id.map(NAMES)
    source=data/'results/figure_sources/s6'
    support=pd.read_csv(source/'History_support.csv').drop(columns=['source_file'],errors='ignore')
    templates=pd.read_csv(source/'Template_support.csv').drop(columns=['source_file'],errors='ignore')
    predictions=pd.read_csv(data/'results/fixed/oof_predictions.csv')
    guide_ids=pd.read_csv(data/'inputs/data/inputs/EasyDesign_2024_V2-2_core_context_feature_table.csv',usecols=['record_id','crRNA_sequence']).set_index('record_id').crRNA_sequence
    guides=[]
    raw=[]
    gblock=predictions[(predictions.cohort_id=='core_dev')&(predictions.protocol_id=='guide')&predictions.model_id.isin(['D','TriCasRank'])].copy()
    gblock['guide']=gblock.record_id.map(guide_ids)
    for (model,guide), frame in gblock.groupby(['model_id','guide'],sort=True):
        if len(frame)<5: continue
        m=basic_metrics(frame.y_true,frame.y_pred)
        raw.append({'model':model,'guide_id':guide,'n':len(frame),'SCC':m['SCC'],'status':'defined' if m['SCC'] is not None else 'undefined_constant'})
    raw=pd.DataFrame(raw); raw.to_csv(tables/'within_guide_values.csv',index=False)
    for model,group in raw.groupby('model'):
        v=group.SCC.dropna()
        guides.extend([{'model':model,'status':'defined','guides':len(v),'minimum_records':5,'median_SCC':v.median(),'q1':v.quantile(.25),'q3':v.quantile(.75)},
                       {'model':model,'status':'undefined_constant','guides':int(group.SCC.isna().sum()),'minimum_records':5}])
    activity=[]
    # The original Fig1 calibration bins order by observed activity, whereas
    # Fig4 protocol calibration bins order by predictions. Keep both definitions.
    for protocol in ('target','guide','atomic_template','dual_atomic'):
        frame=predictions[(predictions.cohort_id=='core_dev')&(predictions.protocol_id==protocol)&(predictions.model_id=='TriCasRank')]
        for b,ids in enumerate(np.array_split(frame.sort_values(['y_pred','record_id'],kind='stable').index.to_numpy(),10)):
            part=frame.loc[ids];activity.append(bin_row(part,'identity_protocol_calibration',protocol,'TriCasRank',b,'predicted activity'))
    target_tri=predictions[(predictions.cohort_id=='core_dev')&(predictions.protocol_id=='target')&(predictions.model_id=='TriCasRank')]
    for b,ids in enumerate(np.array_split(target_tri.sort_values(['y_true','record_id'],kind='stable').index.to_numpy(),10)):
        activity.append(bin_row(target_tri.loc[ids],'calibration','target','TriCasRank',b,'observed activity'))
    legacy=pd.read_csv(data/'results/development/target_components_and_blocks.csv')
    order=legacy[legacy.model_id=='D'].sort_values(['y_true','record_id'],kind='stable')
    bins=[set(ids) for ids in np.array_split(order.record_id.to_numpy(),5)]
    for b,ids in enumerate(bins):
        for model,display in [('D','D'),('B','B'),('C','C'),('full_v2','TriCasRank')]:
            frame=legacy[(legacy.model_id==model)&legacy.record_id.isin(ids)]
            activity.append(bin_row(frame,'activity_quintile_error','target',display,b,'observed activity'))
    common=predictions[(predictions.cohort_id=='common_dev')&(predictions.protocol_id=='target')]
    order=common[common.model_id=='TriCasRank'].sort_values(['y_true','record_id'],kind='stable')
    for b,ids in enumerate(np.array_split(order.record_id.to_numpy(),5)):
        for model in ('EasyDesign-adapted','TriCasRank'):
            frame=common[(common.model_id==model)&common.record_id.isin(ids)]
            activity.append(bin_row(frame,'matched_context_quintile_error','target',model,b,'observed activity'))
    activity=pd.DataFrame(activity);assert len(activity)==80
    provenance=pd.DataFrame([
        {'sheet':'Fold_metrics','source':'results/fixed/oof_predictions.csv; results/classics/pred__*.csv','derivation':'Five outer folds for seven primary models; five metrics recomputed','evaluation':'fixed_recipe_grouped_cv'},
        {'sheet':'History_support','source':'results/figure_sources/s6/History_support.csv','derivation':'Target-fold guide/template-key history availability','evaluation':'fixed target OOF'},
        {'sheet':'Template_support','source':'results/figure_sources/s6/Template_support.csv','derivation':'Number of candidate templates, not historical support count','evaluation':'fixed target OOF'},
        {'sheet':'Activity_bins','source':'results/fixed/oof_predictions.csv; results/development/target_components_and_blocks.csv','derivation':'Observed-order deciles/quintiles and prediction-order deciles as labelled','evaluation':'fixed grouped OOF'},
        {'sheet':'Guide_ranking','source':'results/fixed/oof_predictions.csv + inputs canonical guide IDs','derivation':'Guide-isolated OOF; >=5 records; constant correlations remain missing','evaluation':'descriptive within-guide'},
        {'sheet':'Provenance','source':'paper/tables.py','derivation':'Registry for this workbook','evaluation':'metadata'},
    ])
    sheets={'Fold_metrics':folds,'History_support':support,'Template_support':templates,'Activity_bins':activity,'Guide_ranking':pd.DataFrame(guides),'Provenance':provenance}
    book=Workbook();book.remove(book.active)
    guide=[]
    for name,frame in sheets.items():
        frame.to_csv(tables/(name+'.csv'),index=False)
        sheet=book.create_sheet(name);sheet.sheet_view.showGridLines=False
        sheet.append([name.replace('_',' ')]);sheet.append(['Source data and definitions are listed on Provenance.']);sheet.append([])
        sheet.append(list(frame.columns))
        for row in frame.itertuples(index=False,name=None):
            sheet.append([None if pd.isna(v) else v.item() if isinstance(v,np.generic) else v for v in row])
        for row in sheet.iter_rows():
            for cell in row:
                cell.font=Font(name='Arial',size=10)
                cell.alignment=Alignment(vertical='center',horizontal='right' if isinstance(cell.value,(int,float)) else 'left')
                if isinstance(cell.value,float): cell.number_format='0.000000'
        sheet['A1'].font=Font(name='Arial',size=14,bold=True)
        for cell in sheet[4]:
            cell.fill=PatternFill('solid',fgColor='243C55');cell.font=Font(name='Arial',size=10,color='FFFFFF',bold=True)
            cell.alignment=Alignment(vertical='center',horizontal='center',wrap_text=True)
        sheet.row_dimensions[4].height=42
        for c,col in enumerate(frame.columns,1):
            width=min(56,max(14,len(col)+2, max((len(str(x)) for x in frame[col].head(100) if not pd.isna(x)),default=0)+2))
            sheet.column_dimensions[get_column_letter(c)].width=width
            for row in range(5,len(frame)+5):
                cell=sheet.cell(row,c)
                if isinstance(cell.value,str) and len(cell.value)>width:
                    cell.alignment=Alignment(vertical='center',horizontal='left',wrap_text=True)
                    sheet.row_dimensions[row].height=max(sheet.row_dimensions[row].height or 15,14*int(np.ceil(len(cell.value)/max(1,width-2))))
        sheet.freeze_panes='D5' if name=='Fold_metrics' else 'B5';sheet.auto_filter.ref=f'A4:{get_column_letter(len(frame.columns))}{len(frame)+4}'
        guide.append({'sheet':name,'data_rows':len(frame),'header_row':4,'contents':provenance.loc[provenance.sheet.eq(name),'derivation'].iloc[0]})
    destination=tables/'Table_S6_fold_subgroup_audit.xlsx';book.save(destination)
    reopened=load_workbook(destination,read_only=True,data_only=True)
    assert reopened.sheetnames==list(sheets)
    assert all(reopened[name].max_row==len(frame)+4 for name,frame in sheets.items())
    pd.DataFrame(guide).to_csv(tables/'Table_S6_contents.csv',index=False)

def bin_row(frame,profile,protocol,model,b,ordering):
    y=frame.y_true.to_numpy(float);pred=frame.y_pred.to_numpy(float)
    return {'profile':profile,'protocol':protocol,'model':model,'bin':b,'n':len(frame),'ordering':ordering,
            'mean_observed':float(y.mean()),'mean_predicted':float(pred.mean()),'mean_abs_error':float(np.abs(pred-y).mean()),
            'mean_signed_error':float((pred-y).mean()),'rmse':float(np.sqrt(np.mean((pred-y)**2)))}
