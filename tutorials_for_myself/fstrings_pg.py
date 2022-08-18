# %%

template_fstring: str = 'folder_{split}'
print(template_fstring.format(split='train'))
print(template_fstring.format(split='test'))
x = template_fstring.format(split='train')
print(f'{x=}')
y = template_fstring.format(split='test')
print(f'{y=}')
print(f'{template_fstring=}')
