# Changes to original SVCCA will be listed here

- added code from https://github.com/google/svcca/pull/9 as `pwcca2` from 
pull request that has not been merged at the time of this writing from https://github.com/google/svcca/pull/9
- added a few print statements for debugging pwcca vs anatome's pwcca
- noticed taking the square root of the `sigma_xx, sigma_yy` sometimes results in NANs. Thus,
we were going to add code:
```python
  # import scipy
  # # print('npy\'s sqrt')
  # invsqrt_xx = scipy.linalg.sqrtm(inv_xx)
  # invsqrt_yy = scipy.linalg.sqrtm(inv_yy)
```
but decided it to left it uncomented.
- nothing more...