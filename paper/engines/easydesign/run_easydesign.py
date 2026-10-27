"""Use only the separate TensorFlow environment for EasyDesign.

Restore the author's layers, reset every variable, and vectorize local connections.
The graph topology and all layer configurations are preserved and parity audited.
"""
import os
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
from pathlib import Path
import json,time,hashlib,traceback
import numpy as np
import pandas as pd
import setuptools  # Enable vendored distutils for TensorFlow on Python 3.12.
import tensorflow as tf
from tf_keras.src import backend as legacy_backend
ROOT=Path(__file__).resolve().parents[2]
MODEL=ROOT/'01_sources/third_party/EasyDesign/easyDesign/models/regress/cas12a'
CONFIG=json.loads((ROOT/'02_protocols/configs/easydesign_vectorized_retraining.json').read_text())

def encode(f):
    codes={'A':'A','C':'C','G':'G','T':'T','-':'-','K':'GT','M':'AC','R':'AG','Y':'CT','S':'CG','W':'AT','B':'CGT','V':'ACG','H':'ACT','D':'AGT','N':'ACGT'}
    alphabet={b:i for i,b in enumerate('ACGT-')}
    x=np.zeros((len(f),45,10),np.float32)
    for j,r in enumerate(f.itertuples()):
        assert len(r.target_context_45)==45 and len(r.guide)==25
        for pos,b in enumerate(r.target_context_45):
            for c in codes[b]:x[j,pos,alphabet[c]]=1/len(codes[b])
        for pos,b in enumerate(r.guide,10):
            for c in codes[b]:x[j,pos,5+alphabet[c]]=1/len(codes[b])
    return x

def reset_model(seed,use_vectorized=True):
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
    # tf-keras 2.18 passes float 1e9 to Random.randint in its seeded branch;
    # Python 3.12 rejects this. Its fallback uses int(1e9) and the same seeded
    # Python RNG. Every saved variable is explicitly reset below regardless.
    legacy_backend._SEED_GENERATOR.generator=None
    m=tf.keras.models.load_model(str(MODEL),compile=False)
    original=[v.numpy().copy() for v in m.weights]
    for i,v in enumerate(m.weights):
        name=v.name.split('/')[-1].split(':')[0]
        if name=='kernel':value=tf.keras.initializers.GlorotUniform(seed=seed+i)(v.shape,v.dtype)
        elif name in ['gamma','moving_variance']:value=tf.ones(v.shape,v.dtype)
        elif name in ['bias','beta','moving_mean']:value=tf.zeros(v.shape,v.dtype)
        else:raise ValueError('Unrecognized saved variable '+v.name)
        v.assign(value)
    assert all(not np.array_equal(a,v.numpy()) for a,v in zip(original,m.weights) if 'kernel' in v.name)
    if use_vectorized:
        from easydesign_vectorized import vectorize
        m=vectorize(m)
    m.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=CONFIG['learning_rate']),loss='mse')
    return m

def main():
    run_id=time.strftime('%Y%m%dT%H%M%S')+'_vectorized_regressor_retrain'
    out=ROOT/'03_experiments/EXP-002_easydesign'/run_id;out.mkdir(parents=True)
    def dump(name,d):(out/name).write_text(json.dumps(d,indent=2,default=str)+'\n')
    def log(s):
        print(s,flush=True)
        with (out/'run.log').open('a') as f:f.write(s+'\n')
    dump('status.json',dict(status='running'))
    dump('manifest.json',dict(run_id=run_id,config=CONFIG,tensorflow=tf.__version__,
         model_sha256=hashlib.sha256((MODEL/'saved_model.pb').read_bytes()).hexdigest(),
         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    start=time.time()
    try:
        allf=pd.read_csv(ROOT/'02_protocols/splits/easydesign_contexts.csv')
        allf=allf[allf.baseline_split.eq('baseline_train')].copy()
        f=allf[allf.context_status.eq('ok')].copy().reset_index(drop=True)
        x=encode(f);y=f.y_true.to_numpy(np.float32);pred=np.full(len(f),np.nan);records=[]
        for fold in range(5):
            test=f.fold.eq(fold).to_numpy();inner=f.fold.eq((fold+1)%5).to_numpy();fit=~test&~inner
            m=reset_model(42+fold)
            callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=CONFIG['early_stopping_patience'],min_delta=CONFIG['min_delta'])
            class Progress(tf.keras.callbacks.Callback):
                def on_epoch_end(self,epoch,logs=None):
                    if (epoch+1)%25==0:log(f'Fold {fold}: inner epoch {epoch+1}, validation objective {logs["val_loss"]:.5f}')
            log(f'Fold {fold}: inner fit {fit.sum()}, inner validation {inner.sum()}, outer test {test.sum()}')
            h=m.fit(x[fit],y[fit],validation_data=(x[inner],y[inner]),batch_size=CONFIG['batch_size'],
                    epochs=CONFIG['max_epochs'],callbacks=[callback,Progress()],verbose=0)
            best=int(np.argmin(h.history['val_loss']))+1
            log(f'Fold {fold}: selected {best} epochs by inner validation (ran {len(h.history["loss"])})')
            m=reset_model(42+fold)
            m.fit(x[~test],y[~test],epochs=best,batch_size=CONFIG['batch_size'],verbose=0)
            pred[test]=m.predict(x[test],batch_size=CONFIG['batch_size'],verbose=0).reshape(-1)
            records.append(dict(fold=fold,best_epoch=best,inner_epochs_run=len(h.history['loss']),
                                inner_val_mse_min=float(min(h.history['val_loss'])),n_train=int((~test).sum()),n_test=int(test.sum())))
            log(f'Fold {fold}: predictions complete')
            m.save_weights(str(out/f'fold_{fold}.weights.h5'))
            f.assign(y_pred=pred).to_csv(out/'predictions.partial.csv',index=False)
        assert np.isfinite(pred).all()
        f['y_pred']=pred;f['model_id']='EasyDesign_retrained';f['protocol_id']='target_common_context';f['dataset_id']='baseline_train';f['prediction_status']='ok'
        f.to_csv(out/'predictions.csv',index=False);(out/'predictions.partial.csv').unlink()
        excluded=allf[~allf.context_status.eq('ok')].copy();excluded['prediction_status']=excluded.context_status
        excluded.to_csv(out/'unsupported_records.csv',index=False)
        dump('epoch_selection.json',records);dump('status.json',dict(status='complete',n=len(f),seconds=time.time()-start))
        log('EasyDesign retraining COMPLETE')
    except Exception as e:
        dump('status.json',dict(status='failed',error=str(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
