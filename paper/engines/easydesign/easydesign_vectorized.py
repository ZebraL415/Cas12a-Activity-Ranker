"""Vectorize the released LocallyConnected1D operation without changing topology.

The layer-call edge order was read from the SavedModel FunctionDef. In particular,
dropout_2 is reused after fc_1 AND fc_2. Configurations, regularizers and weights
are copied from restored author layers, not guessed from the paper description.
"""
import numpy as np
import tensorflow as tf

class VectorLocal(tf.keras.layers.Layer):
    def __init__(self,source,**kwargs):
        super().__init__(name=source.name,**kwargs)
        self.width=source.kernel_size[0];self.filters=source.filters
        self.kshape=source.kernel.shape
        self.reg=source.kernel_regularizer;self.breg=source.bias_regularizer
        self.act=source.activation
    def build(self,shape):
        self.kernel=self.add_weight('kernel',shape=self.kshape,initializer='zeros',regularizer=self.reg)
        self.bias=self.add_weight('bias',shape=(self.kshape[0],self.filters),initializer='zeros',regularizer=self.breg)
    def call(self,x):
        frames=tf.signal.frame(x,self.width,1,axis=1)
        xw=tf.reshape(frames,[-1,self.kshape[0],self.kshape[1]])
        return self.act(tf.einsum('bpi,pio->bpo',xw,self.kernel)+self.bias)

def vectorize(source):
    old={l.name:l for l in source.layers};new={}
    for name,l in old.items():
        new[name]=VectorLocal(l) if isinstance(l,tf.keras.layers.LocallyConnected1D) else type(l).from_config(l.get_config())
    x=tf.keras.Input((45,10));z=new['group_w1_conv'](x);z=new['group_w1_batchnorm'](z)
    z=new['merge_pool']([new['group_w1_maxpool'](z),new['group_w1_avgpool'](z)])
    z=new['merge_lc']([new['group_w1_lc_w1'](z),new['group_w1_lc_w2'](z)])
    z=new['flatten'](z);z=new['dropout_1'](z);z=new['fc_1'](z)
    z=new['dropout_2'](z);z=new['fc_2'](z);z=new['dropout_2'](z);z=new['fc_final'](z)
    model=tf.keras.Model(x,z,name='EasyDesign_released_vectorized')
    for name,l in old.items():new[name].set_weights(l.get_weights())
    assert source.count_params()==model.count_params()
    return model

def parity(source,fast,x,y):
    def evaluate(m):
        with tf.GradientTape() as tape:
            p=m(x,training=False);loss=tf.reduce_mean(tf.square(p-tf.reshape(y,[-1,1])))+tf.add_n(m.losses)
        grads=tape.gradient(loss,m.trainable_weights)
        result={w.name:g.numpy() for w,g in zip(m.trainable_weights,grads)}
        return p.numpy(),float(loss),result
    a,la,ga=evaluate(source);b,lb,gb=evaluate(fast)
    assert ga.keys()==gb.keys()
    differences={k:float(np.max(abs(ga[k]-gb[k]))) for k in ga}
    rel={k:float(np.linalg.norm(ga[k]-gb[k])/(np.linalg.norm(ga[k])+1e-7)) for k in ga}
    assert np.allclose(a,b,rtol=2e-5,atol=2e-5)
    assert abs(la-lb)<2e-5
    assert all(np.allclose(ga[k],gb[k],rtol=2e-4,atol=2e-5) for k in ga)
    return dict(prediction_max_abs_error=float(np.max(abs(a-b))),objective_abs_error=abs(la-lb),
                gradient_max_abs_errors=differences,gradient_relative_l2_errors=rel,n=len(x),
                all_14_gradient_tensors_checked=len(ga)==14,status='pass')
