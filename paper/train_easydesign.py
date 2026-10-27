"""Run the unchanged EasyDesign stopping and reset/refit procedure in its own environment."""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parent/'engines/fixed/src'))
import pandas as pd
from easydesign import fit_predict_fold

def main():
    p=argparse.ArgumentParser()
    for name in ('fit','query','config','output','protocol'): p.add_argument('--'+name, required=True)
    p.add_argument('--outer-fold', type=int, required=True)
    a=p.parse_args()
    fit=pd.read_csv(a.fit); query=pd.read_csv(a.query)
    stop=fit.loc[fit[f'{a.protocol}_fold'].eq((a.outer_fold+1)%5),'record_id'].tolist()
    prediction, metadata=fit_predict_fold(fit, stop, query, outer_fold=a.outer_fold, protocol_id=a.protocol, output_dir=a.output, config=json.loads(Path(a.config).read_text()))
    pd.DataFrame({'record_id':query.record_id, 'y_pred':prediction}).to_csv(Path(a.output)/'predictions.csv', index=False)
    print(f'EasyDesign completed stopping and reset/refit: epoch {metadata["selected_epoch"]}', flush=True)

if __name__=='__main__': main()
