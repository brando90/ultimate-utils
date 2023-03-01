# %%

import jax
import jax.numpy as jnp
from jaxlib.xla_extension import DeviceArray

# x: DeviceArray = jnp.arange(5)
# w = jnp.arange(3)
x = jnp.arange(5)
w = jnp.array([2., 3., 4.])

jnp.convolve(x, w, mode="same")

print(x)
print(type(w))


def my_conv_uu(x: DeviceArray, w: DeviceArray) -> DeviceArray:
    # return jnp.convolve(x, w, mode="same")
    assert len(x) >= len(w), "x must be longer than w"
    # convoluve w with x with a loop
    conv = []
    for i in range(len(x) - len(w) + 1):
        conv.append(jnp.dot(x[i:i + len(w)], w))
    return jnp.array(conv)


def convolve(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i - 1:i + 2], w))
    return jnp.array(output)


print(my_conv_uu(x, w))
print(convolve(x, w))
print()
print(jnp.convolve(x, w))
print(jnp.convolve(x, w, mode="same"))
print(jnp.convolve(x, w, mode="valid"))
print(jnp.convolve(x, w, mode="full"))

# %%
xs = jnp.stack([x, x])
ws = jnp.stack([w, w])

print(xs)
print(xs.shape)
print(ws)
print(ws.shape)
print(type(xs))


# %%

def manually_convolve_over_batch_dim(xs, ws) -> DeviceArray:
    output = []
    for i in range(xs.shape[0]):
        output.append(convolve(xs[i], ws[i]))
    return jnp.stack(output)


out: DeviceArray = manually_convolve_over_batch_dim(xs, ws)
print(out)

# In order to batch the computation efficiently, you would normally have to rewrite the function manually to ensure it is done in vectorized form. This is not particularly difficult to implement, but does involve changing how the function treats indices, axes, and other parts of the input.

def manually_vectorized_convolve(xs, ws) -> DeviceArray:
  output = []
  for i in range(1, xs.shape[-1] -1):
    output.append(jnp.sum(xs[:, i-1:i+2] * ws, axis=1))
  return jnp.stack(output, axis=1)

out: DeviceArray = manually_vectorized_convolve(xs, ws)
print(out)

# %%
"""
Automatica vectorization.
Applies your function over the batch dimension.
"""
from typing import Callable
from jax import Array

print(type(manually_vectorized_convolve))
auto_batch_convolve: Callable = jax.vmap(convolve)
assert isinstance(auto_batch_convolve, Callable), "auto_batch_convolve is not a Callable"
print(type(auto_batch_convolve))

out: DeviceArray = manually_vectorized_convolve(xs, ws)
print(out)
out: Array = auto_batch_convolve(xs, ws)
print(out)

# # If the batch dimension is not the first, you may use the in_axes and out_axes arguments to specify the location of the batch dimension in inputs and outputs. These may be an integer if the batch axis is the same for all inputs and outputs, or lists, otherwise.
#
# auto_batch_convolve_v2 = jax.vmap(convolve, in_axes=1, out_axes=1)
#
# xst = jnp.transpose(xs)
# wst = jnp.transpose(ws)
#
# auto_batch_convolve_v2(xst, wst)
#
# # jax.vmap also supports the case where only one of the arguments is batched: for example, if you would like to convolve to a single set of weights w with a batch of vectors x; in this case the in_axes argument can be set to None:
#
# batch_convolve_v3 = jax.vmap(convolve, in_axes=[0, None])
#
# batch_convolve_v3(xs, w)
# %
# jitted_batch_convolve = jax.jit(auto_batch_convolve)
#
# jitted_batch_convolve(xs, ws)
#
# todo %timeit jit auto vs not jit