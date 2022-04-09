"""
Zip:
The zip() function takes iterables (can be zero or more), aggregates them in a tuple, and returns it.

ref:
    - https://www.programiz.com/python-programming/methods/built-in/zip

Unzip:
Zip is its own inverse! Provided you use the special * operator.

ref:
    - https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip


Zip as transpose:
"""

# %%
# - Zip
languages = ['Java', 'Python', 'JavaScript']
versions = [14, 3, 6]

result = zip(languages, versions)
print(list(result))

# Output: [('Java', 14), ('Python', 3), ('JavaScript', 6)]

# %%
# - Unzip

zip(*[('a', 1), ('b', 2), ('c', 3), ('d', 4)])
# [('a', 'b', 'c', 'd'), (1, 2, 3, 4)]


