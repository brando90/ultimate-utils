"""
Multi-head attention with JAX

Att(Q, K, V) = softmax(QK^T/sqrt(d_k))V

Note:
    - sequences are represented as row vectors, so K^T the "normal" colum vector
    - todo:
        test in colab gpu, tpu & make tensors bigger
        attention timeit experiment,surprising that the manual looped one wasnt too slow after jit, wonder if it holds in tpu, gpu with much larger matrices

ref:
    - https://jax.readthedocs.io/en/latest/jax-101/index.html
    - jit ref: https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html
"""
# %%
# create a jax random matrix of size Tx x d
from typing import Callable

import jax
import jax.numpy as jnp

from jax import Array
from jax.random import KeyArray

Tx: int = 3
D: int = 2

key: KeyArray = jax.random.PRNGKey(0)
x: Array = jax.random.normal(key, (Tx, D))


def att(Q: Array, K: Array, V: Array) -> Array:
    # att(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    att: Array = jax.nn.softmax(Q @ K.T / jnp.sqrt(K.shape[1])) @ V
    return att


att_jit: Callable = jax.jit(att)

att_x: Array = att(x, x, x)
print(f'{att_x = }')
print(f'{att_jit(x, x, x) = }')
% timeit
att(x, x, x)
% timeit
att_jit(x, x, x)
"""
cpu
134 µs ± 831 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
4.51 µs ± 20.8 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
gpu

tpu
"""

B = 4
key: KeyArray = jax.random.PRNGKey(0)
x: Array = jax.random.normal(key, (B, Tx, D))


# %%

def att_batch_manual(Q: Array, K: Array, V: Array) -> Array:
    # loop through first dim of Q, K, V and do att and create jax arrary
    out = []
    for i in range(Q.shape[0]):
        att_x: Array = att(Q[i], K[i], V[i])
        out.append(att_x)
    return jnp.array(out)


# vmp att over batch dim of x
att_x: Array = jax.vmap(att)(x, x, x)
print(f'{jnp.sum(att_x)=}')
print(f'{jnp.sum(att_batch_manual(x, x, x))=}')

# jit ref: https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html
att_jit: Callable = jax.jit(jax.vmap(att))
att_batch_manual_jit: Callable = jax.jit(att_batch_manual)

% timeit
jax.vmap(att)(x, x, x)
% timeit
att_batch_manual(x, x, x)
% timeit
att_jit(x, x, x)
% timeit
att_batch_manual_jit(x, x, x)
"""
# cpu
1.76 ms ± 247 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
4.75 ms ± 45.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
4.77 µs ± 31.4 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
6.44 µs ± 20.6 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
# gpu

# tpu
"""

# %%
"""
Multi-head attention with JAX (Encoder)
    h_i = head(Wq @ x, Wk @ x, Wv @ x) = head_i(Q, K, V) = softmax(Q_iK_i^T/sqrt(d_k))V_i
    MHAtt(Q, K, V) = Concat(head_1, ..., head_h)W^O
Multi-head attention with JAX (Decoder)
    Decoder: [Tx, Dm] x [Ty, Dm'] -> [Ty, Dm']
    Decoder_att: [Ty, Dm] x [Dm, Tx] x [Tx, Dm'] -> [Ty, Dm']  (note besides the mask, att can be same as encoder)

x.shape = [Tx, Dm]
Wq.shape = [Dm, Dq] = [Dm, Dk]
Wk.shape = [Dm, Dk]
Wv.shape = [Dm, Dv] = [Dm, Dk]
Wo.shape = [num_heads * Dv, Dm]
Dq = Dk = Dv = Dm / num_heads

Goals
1. single vector first
2. then batches
"""


def compute_head_simple(Q: Array, K: Array, V: Array,  # [Tx, Dm]
                        WQ_i: Array, WK_i: Array, WV_i: Array,  # [Dm, Dk] [Dm, Dv]
                        ):
    Tx: int = K.shape[0]
    # compute each head todo vectorize computing these linear projections
    Q_i: Array = Q @ WQ_i  # [Tx, Dq] x [Dq, Dq'] -> [Tx, Dq']
    K_i: Array = K @ WK_i
    V_i: Array = V @ WV_i
    assert K_i.shape == [Tx, WV_i.shape[1]], f'Error: {K_i.shape} != {WV_i.shape}'
    head_i: Array = att(Q_i, K_i, V_i)
    assert head_i.shape == [Tx, WV_i.shape[1]]
    return head_i


def compute_multi_heads_simple(Q: Array, K: Array, V: Array,  # [Tx, Dm]
                               Ws: list[tuple[Array]],  # [WQi, WKi, WVi] i \in [num_heads] # [Dm, Dk] [Dm, Dv]
                               ) -> list[Array]:
    """
    Note:
        - eventual goal would be to return an "iter" of heads h_i but had compute the hi's in a vectorized/"parallel: way
    """
    Dv: int = Ws[0][2].shape[1]
    heads: list[Array] = []
    for WQ_i, WK_i, WV_i in Ws:  # todo paralelize/vectorize head hi computation for i \in [num_heads]
        head_i: Array = compute_head_simple(Q, K, V, WQ_i, WK_i, WV_i)
        Dv_: int = WV.shape[1]
        assert Dv_ == Dv
        assert head_i.shape == [Tx, Dv]
        heads.append(head_i)
    return heads


def mha_simple(Q: Array, K: Array, V: Array,  # [Tx, Dm]
               Ws: list[tuple[Array, Array, Array]],  # [WQi, WKi, WVi] i \in [num_heads] # [Dm, Dk] [Dm, Dv]
               Wo: Array,  # [num_heads * Dv, Dm]
               num_heads: int = 8,  # len(Ws)
               ):
    Dm: int = Q.shape[1]
    assert len(Ws) == num_heads
    # - compute heads hi = Att(QWQi, KWKi, VWVi)  [Tx, Dv]
    heads: list[Array] = compute_multi_heads_simple(Q, K, V, Ws)
    # - concatenate array heads in jax, shape should be [Tx, num_heads * Dv]
    heads_concat: Array = jnp.concatenate(heads, axis=1)
    Dv = Ws[0][2].shape[1]
    assert heads_concat.shape == [Tx, num_heads * Dv]
    # - compute final mha output
    mha_out: Array = heads_concat @ Wo
    assert mha_out.shape == [Tx, Dm]
    return mha_out

mha_simple_jit: Callable = jax.jit(mha_simple)

#%%
"""
Observe! Now that you've defined it like this it should be easy to share the weights accross the layers!
Let's give an example why not! :) [oh, but we need PosEmbeddings + Transformer Layer Norm to do that!]
Share Ws, Wo across layers!
"""

def generate_random_project_matrices(key: jax.random.PRNGKey, Dm: int, Dk: int, num_heads: int,
                                     ) -> list[tuple[Array, Array, Array]]:
    Ws: list[tuple[Array, Array, Array]] = []
    for _ in range(num_heads):
        Wq, Wk, Wv = jax.random.normal(key, (Dm, Dk)), jax.random.normal(key, (Dm, Dk)), jax.random.normal(key,
                                                                                                           (Dm, Dk))
        Ws.append((Wq, Wk, Wv))
    return Ws

Tx: int = 2
Dm: int = 3
num_heads: int = 4
Dk: int = Dm // num_heads
# create num_heads random linear projections [Wq, Wk, Wv] of shape [Dm, Dk]
Ws: list[tuple[Array, Array, Array]] = [
    (jax.random.normal(key, (Dm, Dk)), jax.random.normal(key, (Dm, Dk)), jax.random.normal(key, (Dm, Dk))) for _ in
    range(num_heads)]
Wo: Array = jax.random.normal(key, (num_heads * Dk, Dm))
x: Array = jax.random.normal(key, (Tx, Dm))

mha_out: Array = mha_simple(x, x, x, Ws, Wo, num_heads)
print(f'{mha_out.shape = }')
%timeit mha_out: Array = mha_simple(x, x, x, Ws, Wo, num_heads)
# mha_out_jit:Array = mha_simple_jit(x, x, x, Ws, Wo, num_heads)
