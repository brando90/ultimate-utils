# %%

import jax
import jax.numpy as jnp
from jaxlib.xla_extension import DeviceArray

x: DeviceArray = jnp.arange(5)
w = jnp.arange(3)

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
print(jnp.convolve(x, w))
print()
print(jnp.convolve(x, w, mode="same"))
print(jnp.convolve(x, w, mode="valid"))
print(jnp.convolve(x, w, mode="full"))
