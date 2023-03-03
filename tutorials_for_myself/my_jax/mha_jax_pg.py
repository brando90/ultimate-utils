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
%timeit att(x, x, x)
%timeit att_jit(x, x, x)
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

#%%

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

att_jit: Callable = jax.jit(jax.vmap(att))
att_batch_manual_jit: Callable = jax.jit(att_batch_manual)

%timeit jax.vmap(att)(x, x, x)
%timeit att_batch_manual(x, x, x)
%timeit att_jit(x, x, x)
%timeit att_batch_manual_jit(x, x, x)
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
