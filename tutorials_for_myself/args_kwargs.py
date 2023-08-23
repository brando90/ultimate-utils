#%%
"""
Q: what happens for f(arg, hp1, hp2) being called as f(arg, **hps)?
"""

def f(arg, hp1=None, hp2=None):
    print('-- in f')
    print(arg)
    print(hp1)
    print(hp2)


f('arg', 1, 2)
f('arg', **{'hp1': 1, 'hp2': 2})
f(arg='arg', **{'hp1': 1, 'hp2': 2})
# f('arg', **{'hp1': 1, 'hp2': 2, 'hp3': 3})
f('arg', **{})

print('------------')

def f2(arg, hp1, hp2=None):
    print('-- in f')
    print(arg)
    print(hp1)
    print(hp2)


f2('arg', 1, 2)
f2('arg', **{'hp1': 1, 'hp2': 2})
# f2('arg', **{})

print('------------')

def f3(arg, hp1, hp2, hp3, hp4):
    print('-- in f')
    print(arg)
    print(hp1)
    print(hp2)
    print(hp3)
    print(hp4)


f3('arg', 1, 2, 3, 4)
f3('arg', **{'hp1': 1, 'hp2': 2}, **{'hp3': 3, 'hp4': 4})

# -

def f(arg, hp1=None, hp2=None, hp3='default'):
    print('-- in f')
    print(arg)
    print(hp1)
    print(hp2)
    print(hp3)

f('arg', **{})

# %%
"""
what happens if some of the values in the args dict are missing? will they be filled with defaults?
"""

def f(req_arg, arg0=None, arg1=None, arg2=None):
    print('-- in f')
    print(req_arg)
    print(arg0)
    print(arg1)
    print(arg2)

f('req_arg', **{'arg0': 0, 'arg1': 1})