#%%
"""
Get the weights & biases to set them to a nn.Linear layer in pytorch
"""
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from torch import nn


X, y = load_iris(return_X_y=True)
print(f'{X.shape=}')
print(f'{y.shape=}')
Din: int = X.shape[1]
total_data_set_size: int = X.shape[0]
assert y.shape[0] == total_data_set_size

clf = LogisticRegression(random_state=0).fit(X, y)
out = clf.predict(X[:2, :])
# print(f'{out=}')

out = clf.predict_proba(X[:2, :])
print(f'{out=}')


clf.score(X, y)

# - coef_ndarray of shape (1, n_features) or (n_classes, n_features)
print(f'{clf.coef_.shape=}')
print(f'{clf.intercept_.shape=}')
assert (clf.coef_.shape[1] == Din)
Dout: int = clf.coef_.shape[0]
print(f'{Dout=} which is the number of classes too in classification')
assert (Dout == clf.intercept_.shape[0])

print()
num_classes: int = Dout
mdl = nn.Linear(in_features=Din, out_features=num_classes)
mdl.weight = torch.nn.Parameter(torch.from_numpy(clf.coef_))
mdl.bias = torch.nn.Parameter(torch.from_numpy(clf.intercept_))

out2 = torch.softmax(mdl(torch.from_numpy(X[:2, :])), dim=1)
print(f'{out2=}')

assert np.isclose(out2.detach().cpu().numpy(), out).all()

# -
# module: nn.Module = getattr(base_model, layer_to_replace)
# num_classes: int = clf.coef_[0]  # out_features=Dout
# num_features: int = clf.coef_[1]  # in_features
# assert module.weight.Size() == torch.Size([num_classes, num_features])
# assert module.bias.Size() == torch.Size([num_classes])
# module.weight = torch.nn.Parameter(torch.from_numpy(clf.coef_))
# module.bias = torch.nn.Parameter(torch.from_numpy(clf.intercept_))