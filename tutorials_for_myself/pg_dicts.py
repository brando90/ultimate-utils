# %%
'''
using tuples as indices in python means we don't need parens! :-)
'''

d: dict = {(1, 2, 3, 4): 'a'}

print(d)
print(d[1, 2, 3, 4])
print(d[(1, 2, 3, 4)])
