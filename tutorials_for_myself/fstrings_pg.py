#%%

template_fstring: str = 'folder_{split}'
print(template_fstring.format(split='train'))
print(template_fstring.format(split='test'))
