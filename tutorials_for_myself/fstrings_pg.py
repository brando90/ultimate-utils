# %%
"""
https://docs.python.org/3/library/stdtypes.html#str.format
https://stackoverflow.com/questions/54351740/how-can-i-use-f-string-with-a-variable-not-with-a-string-literal
"""
template_fstring: str = 'folder_{split}'
print(template_fstring.format(split='train'))
print(template_fstring.format(split='test'))
x = template_fstring.format(split='train')
print(f'{x=}')
y = template_fstring.format(split='test')
print(f'{y=}')
print(f'{template_fstring=}')
