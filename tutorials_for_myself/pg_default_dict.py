#%%

from collections import defaultdict

# Defining the dict
d = defaultdict(lambda: {})
d["a"] = {'aa': 11}
d["b"] = {'bb': 22}

print(d["a"])
print(d["b"])
print(d["c"])
print(d)

#%%

d = defaultdict(lambda: {})

d['file1']['']