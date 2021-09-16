#%%

import time
import progressbar

bar = progressbar.ProgressBar(max_value=5)
for i in range(5):
    time.sleep(1)
    bar.update(i)

"""
DOES NOT COMPLETE PROPERLY

 80% (4 of 5) |####################      | Elapsed Time: 0:00:04 ETA:   0:00:01
"""

#%%

"""

USE THIS ONE! (but it's also fine if it doesn't end perfectly, not worth the effort or adding the extra ugly update
at the end of the loop - let it go!).

Use this one to make sure the "end" is shown properly 100% etc

https://gist.github.com/brando90/3304119120841b1ebf892fe93a2cc3c9 

the key is to wrap the iterator (of fixed length) using bar e.g. bar(range(100))
"""

import time
import progressbar

widgets = [
    progressbar.Percentage(),
    progressbar.Bar(),
    ' ', progressbar.SimpleProgress(),
    ' ', progressbar.ETA(),
    ' ', progressbar.AdaptiveTransferSpeed(unit='it'),
]

bar = progressbar.ProgressBar(widgets=widgets)
for i in bar(range(100)):
    time.sleep(0.1)
    bar.update(i)
"""
19%|##########                                           | 19 of 100 ETA:   0:00:17   4.9 it/s

when done:

100%|####################################| 100 of 100 Time:  0:00:20   4.9 it/s
"""

#%%

"""
from default
 99% (9998 of 10000) |########## | Elapsed Time: 1 day, 16:35:09 ETA:   0:00:26
"""

import time
import progressbar

widgets = [
    progressbar.Percentage(),
    ' ', progressbar.SimpleProgress(format=f'({progressbar.SimpleProgress.DEFAULT_FORMAT})'),
    ' ', progressbar.Bar(),
    ' ', progressbar.Timer(), ' |',
    ' ', progressbar.ETA(), ' |',
    ' ', progressbar.AdaptiveTransferSpeed(unit='it'),
]

bar = progressbar.ProgressBar(widgets=widgets)
for i in bar(range(100)):
    time.sleep(0.1)
    bar.update(i)

#%%

import uutils

def test_good_progressbar():
    import time
    bar = uutils.get_good_progressbar()
    for i in bar(range(100)):
        time.sleep(0.1)
        bar.update(i)

    print('---- start context manager test ---')
    max_value = 10
    with uutils.get_good_progressbar(max_value=max_value) as bar:
        for i in range(max_value):
            time.sleep(1)
            bar.update(i)

test_good_progressbar()

#%%

import time
import progressbar

# bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
bar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
for i in range(20):
    time.sleep(0.1)
    bar.update(i)

#%%


import time
import progressbar

bar = progressbar.ProgressBar(max_value=5)
for i in range(5):
    time.sleep(1)
    bar.update(i)
    print(f'\n{i=}')

"""
N/A% (0 of 5) |                          | Elapsed Time: 0:00:00 ETA:  --:--:--
i=0
 20% (1 of 5) |#####                     | Elapsed Time: 0:00:01 ETA:   0:00:04
i=1
 40% (2 of 5) |##########                | Elapsed Time: 0:00:02 ETA:   0:00:03
i=2
 60% (3 of 5) |###############           | Elapsed Time: 0:00:03 ETA:   0:00:02
i=3
 80% (4 of 5) |####################      | Elapsed Time: 0:00:04 ETA:   0:00:01
i=4
"""

#%%

"""
do update
do new line \n
the do print
"""

import uutils
import time

bar = uutils.get_good_progressbar(max_value=5)
for i in range(5):
    time.sleep(1)
    bar.update(i)
    print(f'\n{i=}')

"""
N/A% (0 of 5) |           | Elapsed Time: 0:00:00 | ETA:  --:--:-- |   0.0 s/it
i=0
 20% (1 of 5) |##         | Elapsed Time: 0:00:01 | ETA:   0:00:04 |   1.0 it/s
i=1
 40% (2 of 5) |####       | Elapsed Time: 0:00:02 | ETA:   0:00:03 |   1.0 it/s
i=2
 60% (3 of 5) |######     | Elapsed Time: 0:00:03 | ETA:   0:00:02 |   1.0 it/s
i=3
 80% (4 of 5) |########   | Elapsed Time: 0:00:04 | ETA:   0:00:01 |   1.0 it/s
i=4
"""