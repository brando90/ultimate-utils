"""
Demo of how to load pt models to jax and in reverse.

Idea:
- have the two share the same name of dictionary and convert them to each others tensors.


"""
#%%
"""
pt -> jax

idea:
- get the params of the pt in a maned dictionary
- pass that dictionary (e.g. f(x, **params) byt def is f(x, w1, w2, ..., wn)) or f(x, w) ) and to the definition of
your pure jax function (NN).
- Done. Just make sure your forward pass in Jax is correctly implemented, the pt ws are convert to jax weights/tensors
before using.
"""

#%%
"""
jax  -> pt

idea:
- should be the same just make sure the names of your weights in jax match the nn.Modules param names + that the
jax tensors are converted to pt tensors/params correctly. 
"""