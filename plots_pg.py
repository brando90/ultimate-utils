# %%

print('start')

# f_avg: PLinReg vs MAML

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

datas_std = [0.1, 0.125, 0.1875, 0.2]

pl = np.array([2.3078539778125768e-07,
               1.9997889411762922e-07,
               2.729681222011256e-07,
               3.2532371115080884e-07])
pl_stds = [1.4852212316567463e-08,
           5.090588920661132e-09,
           1.1424832554909115e-08,
           5.058656213138166e-08]

maml = np.array([3.309504692539563e-07,
                 4.1058904888091606e-06,
                 6.8326703386053605e-06,
                 7.4616147721799645e-06])
maml_stds = [4.039131189060566e-08,
             3.66839089258494e-08,
             9.20683484136399e-08,
             9.789292209743077e-08]

# fig = plt.figure()
fig, ax = plt.subplots(nrows=1, ncols=1)

ax.set_title('MAML vs Pre-Trained embedding with Linear Regression')

x = datas_std
diff = pl - maml

ax.errorbar(x, pl, yerr=pl_stds, label='PLinReg', marker='o')
ax.errorbar(x, maml, yerr=maml_stds, label='MAML', marker='o')
ax.plot(x, diff, label='Difference (PLinReg - MAML)')
ax.legend()

ax.set_xlabel('std (of FNN Data set) of the embedding layer (layer 1)')
ax.set_ylabel('meta-test loss (MSE)')

plt.show()

# path = Path('~/ultimate-utils/plot').expanduser()
# fig.savefig(path)

print('done \a')

# %%

print('start')

# f_avg: PLinReg vs MAML

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

datas_std = [0.1, 0.125, 0.1875, 0.2]

pl = np.array([2.3078539778125768e-07,
               1.9997889411762922e-07,
               2.729681222011256e-07,
               3.2532371115080884e-07])
pl_stds = [1.4852212316567463e-08,
           5.090588920661132e-09,
           1.1424832554909115e-08,
           5.058656213138166e-08]

maml = np.array([3.309504692539563e-07,
                 4.1058904888091606e-06,
                 6.8326703386053605e-06,
                 7.4616147721799645e-06])
maml_stds = [4.039131189060566e-08,
             3.66839089258494e-08,
             9.20683484136399e-08,
             9.789292209743077e-08]

# fig = plt.figure()
fig, ax = plt.subplots(nrows=1, ncols=1)

ax.set_title('MAML vs Pre-Trained embedding with Linear Regression')

x = datas_std
diff = pl - maml

ax.errorbar(x, pl, yerr=pl_stds, label='PLinReg', marker='o')
ax.errorbar(x, maml, yerr=maml_stds, label='MAML', marker='o')
ax.plot(x, diff, label='Difference (PLinReg - MAML)')
ax.legend()

ax.set_xlabel('std (of FNN Data set) of the embedding layer (layer 1)')
ax.set_ylabel('meta-test loss (MSE)')

plt.show()

# path = Path('~/Desktop/').expanduser()
# fig.savefig(path)

print('done \a')