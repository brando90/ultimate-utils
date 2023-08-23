# -*- coding: utf-8 -*-
"""jax-stanford-demo-11-2022.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/brando90/e2d3c8c5a1a3516430252adf6330027c/jax-stanford-demo-11-2022.ipynb
"""

import jax
jax.config.update('jax_array', True)  # required for jax<=0.4.0

"""### `jax.numpy` on TPU (or GPU, or CPU)"""

import jax.numpy as jnp
from jax import random

x = random.normal(random.PRNGKey(0), (8192, 8192))
x

print(x.shape)
print(x.dtype)

y = jnp.dot(x, jnp.cos(x.T))
z = y[[0, 2, 1, 0], ..., ::-1, None]
print(z[:3, :3])

# Commented out IPython magic to ensure Python compatibility.
# %timeit -n 5 -r 5 jnp.dot(x, x).block_until_ready()

import numpy as np
x_cpu = np.array(x)

# Commented out IPython magic to ensure Python compatibility.
# %timeit -n 1 -r 2 np.dot(x_cpu[:2048, :2048], x_cpu[:2048, :2048])

"""### Automatic differentiation"""

from jax import grad

def f(x):
    if x > 0:
        return 2 * x ** 3
    else:
        return 3 * x

x = -3.14

print(grad(f)(x))
print(grad(f)(-x))

print(grad(grad(grad(f)))(-x))

def predict(params, inputs):
    for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.maximum(outputs, 0)
    return outputs

def loss(params, batch):
    inputs, targets = batch
    predictions = predict(params, inputs)
    return jnp.sum((predictions - targets)**2)

def init_layer(key, n_in, n_out):
    k1, k2 = random.split(key)
    W = random.normal(k1, (n_in, n_out)) / jnp.sqrt(n_in)
    b = random.normal(k2, (n_out,))
    return W, b

def init_model(key, layer_sizes, batch_size):
    key, *keys = random.split(key, len(layer_sizes))
    params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))

    key, *keys = random.split(key, 3)
    inputs = random.normal(keys[0], (batch_size, layer_sizes[0]))
    targets = random.normal(keys[1], (batch_size, layer_sizes[-1]))

    return params, (inputs, targets)

layer_sizes = [784, 2048, 2048, 2048, 10]
batch_size = 128

params, batch = init_model(random.PRNGKey(0), layer_sizes, batch_size)

print(loss(params, batch))

step_size = 1e-5

for _ in range(30):
    grads = grad(loss)(params, batch)
    params = [(W - step_size * dW, b - step_size * db)
              for (W, b), (dW, db) in zip(params, grads)]

print(loss(params, batch))

"""Lots more autodiff...
* forward- and reverse-mode, totally composable
* fast Jacobians and Hessians
* complex number support (holomorphic and non-holomorphic)
* exponentially-faster very-high-order autodiff
* precise control over stored intermediate values

### End-to-end optimized compilation with `jit`
"""

from jax import jit

loss_jit = jit(loss)

print(loss_jit(params, batch))

# Commented out IPython magic to ensure Python compatibility.
# %timeit -n 5 -r 5 loss(params, batch).block_until_ready()

# Commented out IPython magic to ensure Python compatibility.
# %timeit -n 5 -r 5 loss_jit(params, batch).block_until_ready()

gradfun = jit(grad(loss))

for _ in range(30):
    grads = gradfun(params, batch)
    params = [(W - step_size * dW, b - step_size * db)
              for (W, b), (dW, db) in zip(params, grads)]
    
print(loss_jit(params, batch))

"""Limitations with jit:
* value-dependent Python control flow disallowed, use e.g. `lax.cond`, `lax.scan` instead
* must be functionally pure, **like all JAX code**

### Batching with `vmap`
"""

from jax import vmap

def l1_distance(x, y):
    assert x.ndim == y.ndim == 1
    return jnp.sum(jnp.abs(x - y))

xs = random.normal(random.PRNGKey(0), (20, 3))
ys = random.normal(random.PRNGKey(1), (20, 3))

dists = jnp.stack([l1_distance(x, y) for x, y in zip(xs, ys)])
print(dists)

dists = vmap(l1_distance)(xs, ys)
print(dists)

from jax import make_jaxpr
make_jaxpr(l1_distance)(xs[0], ys[0])

make_jaxpr(vmap(l1_distance))(xs, ys)

def pairwise_distances(xs, ys):
    return vmap(vmap(l1_distance, (0, None)), (None, 0))(xs, ys)

make_jaxpr(pairwise_distances)(xs, ys)

perexample_grads = jit(vmap(grad(loss), in_axes=(None, 0)))

(dW, db), *_ = perexample_grads(params, batch)
dW.shape

"""Use `vmap` to plumb batch dimensions through anything: vectorize your code, library code, autodiff-generated code...

### Explicit SPMD parallelism with `pmap`
"""

from jax import pmap

jax.devices()

keys = random.split(random.PRNGKey(0), 8)
mats = pmap(lambda key: random.normal(key, (8192, 8192)))(keys)
mats.shape

result = pmap(jnp.dot)(mats, mats)
print(result.shape)

# timeit -n 5 -r 5 pmap(jnp.dot)(mats, mats).block_until_ready()

from functools import partial
from jax import lax

@partial(pmap, axis_name='i')
def allreduce_sum(x):
    return lax.psum(x, 'i')

allreduce_sum(jnp.ones(8))

"""### **NEW**: Implicit parallelism with `jit`!"""

import jax

x = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))

jax.debug.visualize_array_sharding(x)

"""Sharding an array across multiple devices:"""

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
devices = mesh_utils.create_device_mesh((8,))
sharding = PositionalSharding(devices)

x = jax.device_put(x, sharding.reshape(8, 1))
jax.debug.visualize_array_sharding(x)

"""A sharding is an array of sets of devices:"""

sharding

"""Shardings can be reshaped, just like arrays:"""

sharding.shape

sharding.reshape(8, 1)

sharding.reshape(4, 2)

"""An array `x` can be sharded with a sharding if the sharding is _congruent_ with `x.shape`, meaning the sharding has the same length as `x.shape` and each element evenly divides the corresponding element of `x.shape`.

For example:
"""

sharding = sharding.reshape(4, 2)
print(sharding)

y = jax.device_put(x, sharding)
jax.debug.visualize_array_sharding(y)

"""Different `sharding`s result in different distributed layouts:"""

sharding = sharding.reshape(1, 8)
print(sharding)

y = jax.device_put(x, sharding)
jax.debug.visualize_array_sharding(y)

"""Sometimes we might want to _replicate_ some slices:

We can express replication by calling the sharding reducer method `replicate`:
"""

sharding = sharding.reshape(4, 2)
print(sharding.replicate(axis=0, keepdims=True))

y = jax.device_put(x, sharding.replicate(axis=0, keepdims=True))
jax.debug.visualize_array_sharding(y)

"""The `replicate` method acts similar to the familiar NumPy array reduction methods like `.sum()` and `.prod()`."""

print(sharding.replicate(0).shape)
print(sharding.replicate(1).shape)

y = jax.device_put(x, sharding.replicate(1))
jax.debug.visualize_array_sharding(y)

"""## Computation follows sharding

JAX uses a computation-follows-data layout policy, which extends to shardings:
"""

sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))

x = jax.device_put(x, sharding.reshape(4, 2))
print('Input sharding:')
jax.debug.visualize_array_sharding(x)

y = jnp.sin(x)

print('Output sharding:')
jax.debug.visualize_array_sharding(y)

"""For an elementwise operation like `jnp.sin` the compiler avoids communication and chooses the output sharding to be the same as the input.

A richer example:
"""

y = jax.device_put(x, sharding.reshape(4, 2).replicate(1))
z = jax.device_put(x, sharding.reshape(4, 2).replicate(0))
print('LHS sharding:')
jax.debug.visualize_array_sharding(y)
print('RHS sharding:')
jax.debug.visualize_array_sharding(z)

w = jnp.dot(y, z)

print('Output sharding:')
jax.debug.visualize_array_sharding(w)

"""The compiler chose an output sharding that maximally parallelizes the computation and avoids communication.

How can we be sure it's actually running in parallel? We can do a simple timing experiment:
"""

x_single = jax.device_put(x, jax.devices()[0])
jax.debug.visualize_array_sharding(x_single)

np.allclose(jnp.dot(x_single, x_single),
            jnp.dot(y, z))

# Commented out IPython magic to ensure Python compatibility.
# %timeit -n 5 -r 5 jnp.dot(x_single, x_single).block_until_ready()

# Commented out IPython magic to ensure Python compatibility.
# %timeit -n 5 -r 5 jnp.dot(y, z).block_until_ready()

"""## Examples: neural networks

We can use `jax.device_put` and `jax.jit`'s computation-follows-sharding features to parallelize computation in neural networks. Here are some simple examples, based on this basic neural network:
"""

def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.maximum(outputs, 0)
  return outputs

def loss(params, batch):
  inputs, targets = batch
  predictions = predict(params, inputs)
  return jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))

loss_jit = jax.jit(loss)
gradfun = jax.jit(jax.grad(loss))

def init_layer(key, n_in, n_out):
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (n_in, n_out)) / jnp.sqrt(n_in)
    b = jax.random.normal(k2, (n_out,))
    return W, b

def init_model(key, layer_sizes, batch_size):
    key, *keys = jax.random.split(key, len(layer_sizes))
    params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))

    key, *keys = jax.random.split(key, 3)
    inputs = jax.random.normal(keys[0], (batch_size, layer_sizes[0]))
    targets = jax.random.normal(keys[1], (batch_size, layer_sizes[-1]))

    return params, (inputs, targets)

layer_sizes = [784, 8192, 8192, 8192, 10]
batch_size = 8192

params, batch = init_model(jax.random.PRNGKey(0), layer_sizes, batch_size)

"""### 8-way batch data parallelism"""

sharding = PositionalSharding(jax.devices()).reshape(8, 1)

batch = jax.device_put(batch, sharding)
params = jax.device_put(params, sharding.replicate())

jax.debug.visualize_array_sharding(batch[0])
jax.debug.visualize_array_sharding(params[0][0])

loss_jit(params, batch)

step_size = 1e-5

for _ in range(30):
  grads = gradfun(params, batch)
  params = [(W - step_size * dW, b - step_size * db)
            for (W, b), (dW, db) in zip(params, grads)]

print(loss_jit(params, batch))

jax.debug.visualize_array_sharding(params[0][0])

# Commented out IPython magic to ensure Python compatibility.
# %timeit -n 5 -r 5 gradfun(params, batch)[0][0].block_until_ready()

batch_single = jax.device_put(batch, jax.devices()[0])
params_single = jax.device_put(params, jax.devices()[0])

# Commented out IPython magic to ensure Python compatibility.
# %timeit -n 5 -r 5 gradfun(params_single, batch_single)[0][0].block_until_ready()

"""### 4-way batch (data) parallelism and 2-way model (weight) parallelism"""

sharding = sharding.reshape(4, 2)

batch = jax.device_put(batch, sharding.replicate(1))
jax.debug.visualize_array_sharding(batch[0])
jax.debug.visualize_array_sharding(batch[1])

params, batch = init_model(jax.random.PRNGKey(0), layer_sizes, batch_size)

(W1, b1), (W2, b2), (W3, b3), (W4, b4) = params

W1 = jax.device_put(W1, sharding.replicate())
b1 = jax.device_put(b1, sharding.replicate())

W2 = jax.device_put(W2, sharding.replicate(0))
b2 = jax.device_put(b2, sharding.replicate(0))

W3 = jax.device_put(W3, sharding.replicate(0).T)
b3 = jax.device_put(b3, sharding.replicate())

W4 = jax.device_put(W4, sharding.replicate())
b4 = jax.device_put(b4, sharding.replicate())

params = (W1, b1), (W2, b2), (W3, b3), (W4, b4)

jax.debug.visualize_array_sharding(W2)

jax.debug.visualize_array_sharding(W3)

print(loss_jit(params, batch))

step_size = 1e-5

for _ in range(30):
    grads = gradfun(params, batch)
    params = [(W - step_size * dW, b - step_size * db)
              for (W, b), (dW, db) in zip(params, grads)]

print(loss_jit(params, batch))

(W1, b1), (W2, b2), (W3, b3), (W4, b4) = params
jax.debug.visualize_array_sharding(W2)
jax.debug.visualize_array_sharding(W3)

# Commented out IPython magic to ensure Python compatibility.
# %timeit -n 10 -r 10 gradfun(params, batch)[0][0].block_until_ready()

"""We didn't change our model code at all! Write your code for one device, run it on _N_...

Compose with `grad`, `vmap`, `jit`, ...
"""