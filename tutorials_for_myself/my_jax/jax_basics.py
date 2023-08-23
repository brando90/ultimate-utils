# %%
import jax
import jax.numpy as jnp

x = jnp.arange(10)
print(x)
print(type(x))

# (Technical detail: when a JAX function is called (including jnp.array creation), the corresponding operation is dispatched to an accelerator to be computed asynchronously when possible. The returned array is therefore not necessarily ‘filled in’ as soon as the function returns. Thus, if we don’t require the result immediately, the computation won’t block Python execution. Therefore, unless we block_until_ready or convert the array to a regular Python type, we will only time the dispatch, not the actual computation

#%%
long_vector = jnp.arange(int(1e7))

%timeit jnp.dot(long_vector, long_vector).block_until_ready()

#%%

