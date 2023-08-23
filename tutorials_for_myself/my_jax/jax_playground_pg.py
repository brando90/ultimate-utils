# %%
# -- Goal: concatenate [hi]_i=1^num_heads, list [Tx, Dv] -> [Tx, Dv*num_heads]
import jax

num_heads: int = 4
Tx, Dv = 3, 2
# Generate 8 jax random matrices of shape [Tx, Dv]
key = jax.random.PRNGKey(0)
heads = [jax.random.normal(key, (Tx, Dv)) for _ in range(num_heads)]
print(f'{heads = }')
# concatenate heads so that we get [Tx, Dv*num_heads]
concat_heads = jax.numpy.concatenate(heads, axis=1)
print(f'{concat_heads.shape = }')
assert concat_heads.shape == (Tx, Dv * num_heads), f'Error: {concat_heads.shape = }'