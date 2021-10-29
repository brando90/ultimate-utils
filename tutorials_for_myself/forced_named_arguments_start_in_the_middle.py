#%%
"""
https://stackoverflow.com/questions/2965271/forced-naming-of-parameters-in-python/14298976#14298976
"""

def foo(pos, *, forcenamed = None):
    print(pos, forcenamed)

foo(pos=10, forcenamed=20)
foo(10, forcenamed=20)
foo(10)
foo(10, 20)  # error

def foo2(pos, *, forcenamed):
    print(pos, forcenamed)

foo2(pos=10, forcenamed=20)
foo2(10, forcenamed=20)
# basically you always have to give the (named) value!
foo2(10)