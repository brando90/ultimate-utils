#%%

from tqdm import tqdm

iters = 5
with tqdm(range(iters)) as pbar:
    for i in pbar:
        print(i)

print('Done!\a')

#%%

from tqdm import tqdm

iters = 5
with tqdm(range(iters)) as pbar:
    i = 0
    while i < iters:
        print(i)
        i += 1
        pbar.update()
        if i >= iters:
            break

print('Done!\a')