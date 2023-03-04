"""
Parallel Evaluation in JAX

Single-program, multiple-data (SPMD) (refers to a parallelism technique where the same computation
(e.g., the forward pass of a neural net) is run on different input data (e.g., different inputs in a batch) in parallel
on different devices (e.g., several TPUs).

JAX supports device parallelism analogously, using jax.pmap to transform a function written for one device into a
function that runs in parallel on multiple devices. This colab will teach you all about it.

Main concept of pmap:
    tldr; jax.pmap(f)(x, w) will apply f by looping through the first dimension of x & w (according to how f works for a
    single element of x & w, instead of a batch) for multiple devices. More details see pmap (bellow).

    - pmap is a transformation that takes a function f and maps it over a specified axis of an input array.
    - e.g. if you have f and want to apply f to x [N, D], w [N, D] then
        jax.pmap(f)(x, w) will return [N, D] where each element is the result of f(x[i], w[i]) for i in range(N)
    i.e. "loops through the specified axis of the inputs array and applies the function to each element of the array."
    vmap & map conceptually do the same except vmap parallelizes in a single device and pmap over multiple devices.
    More concretely vmap vectorizes and pmap parallelizes.

vectorize ~ parallelize in one tpu/gpu.
parallelize ~ parallelize in multiple tpu/gpus.

ref:
    - pmap: https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html
"""
#%%
colab: bool = False
if colab:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
    import jax
    jax.devices()
#%%
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.basearray import Array

x = np.arange(5)
w = np.array([2., 3., 4.])
print(f'{x=}')
print(f'{x[0:3]=}')

def my_conv(x: Array, w: Array) -> Array:
    assert len(x) >= len(w), "x must be longer than w"
    # convoluve w with x with a loop
    conv: list = []
    for i in range(len(x) - len(w) + 1):
        c = x[i:i+len(w)] @ w
        conv.append(c)
    return jnp.array(conv)

def convolve(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  return jnp.array(output)


print(f'{my_conv(x, w)=}')
print(f'{convolve(x, w)=}')

#%%
if colab:
    n_devices = jax.device_count()
else:
    n_devices = 1
    # n_devices = 2
    # n_devices = 8  # to make it look the same as the tutorial
xs = np.arange(5 * n_devices).reshape(-1, 5)
print(f'{xs=}')
print(f'{xs.shape=}')
#  Join a sequence of arrays along a new axis.
ws = np.stack([w] * n_devices)
print(f'{w.shape=}')
print(f'{ws=}')
print(f'{ws.shape=}')
assert xs.shape == (n_devices, 5)
assert ws.shape == (n_devices, 3)

# do convolution on first dimension using vectorization (on one device)
from jax._src.array import DeviceArray
conv_vmap_val: DeviceArray = jax.vmap(convolve)(xs, ws)
print(f'{conv_vmap_val=}')

# do convolution on first dimension using parallelization (on multiple devices)
from jax.interpreters.pxla import ShardedDeviceArray
conv_pmap_val: ShardedDeviceArray = jax.pmap(convolve)(xs, ws)
print(f'{conv_pmap_val=}')
# That is because the elements of this array are sharded across all of the devices used in the parallelism.
# If we were to run another parallel computation, the elements would stay on their respective devices,
# without incurring cross-device communication costs.
conv_pmap_val: ShardedDeviceArray = jax.pmap(convolve)(xs, jax.pmap(convolve)(xs, ws))
print(f'{conv_pmap_val=}')

#%%
conv_pmap_val: ShardedDeviceArray = jax.pmap(convolve, in_axes=(0, None))(xs, w)
