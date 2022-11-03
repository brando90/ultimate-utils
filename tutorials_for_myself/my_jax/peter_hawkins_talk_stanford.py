"""

jax treats functions as blackboxes.
Traces ~ Drops in symbolic/abstract values.

My Q:
- jax performance gpu vs tpu

- jax <-> tf <-> pt
    - A: seems yes, write pure function in jax then load models
    - (ask HF...)
- feedback aligment
    -
- non-diff e.g. discrete, proof process
    -
- can I combine pytorch & jax?
    - yes-ish, write the backward pass for the pytorch stuff, e.g. use the pt backward or something
- colab code
    -
- community discuss: ...

other Q:
- Q: does jax preserve py type?
A:you can't type jax, jit, grad.
- Q: pmap vs vmap
    - A:
    - both push batch dimensions, but in different wants
    - pmap =
    - vmap =

colab link: 
"""
#%%


#%%
"""
grads
"""


#%%
"""
jit compiles python code to faster computation.
"""


#%%
"""
vmap vectorizes for you! e.g. if you give a list shuves it into a gpu

vmaps, you can apply it to code you didn't write.

"""

#%%
"""
parallelism

pmap, similar to vmap but parallelisize instead of vectorizes.


%times yout code
%timeit -n 5 -r 5 pythoncode

Q: for DDP?
"""


#%%
"""
How does Jax work?

Trace it with abstract values
"""

