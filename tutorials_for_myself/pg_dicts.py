# %%
""""
using tuples as indices in python means we don't need parens! :-)
"""

# d: dict = {(1, 2, 3, 4): 'a'}
#
# print(d)
# print(d[1, 2, 3, 4])
# print(d[(1, 2, 3, 4)])

# %%
"""
delete  key from a dict in while looping through it
"""
from pprint import pprint

d: dict = {1: 'a', 2: 'b', 3: 'c', 4: 'd'}
print(d)
pprint(d)

# illegal
# for k in d:
#     if k % 2 == 0:
#         del d[k]

for k in [2, 4]:
    del d[k]

print(d)
pprint(d)
