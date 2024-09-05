#%%
"""
https://stackoverflow.com/questions/41641449/how-do-i-annotate-types-in-a-for-loop
"""

i: int
for i in range(5):
    print(f'{i=}')

s: str
for s in ['a', 'b', 'c']:
    print(f'{s=}')