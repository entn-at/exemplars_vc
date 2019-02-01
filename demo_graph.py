import numpy as np
import pdb

from sklearn.decomposition import NMF
from sklearn import decomposition


X = np.array([[1, 51], [2, 1], [3,5], [4,58], [5.5, 92], [1, 6]])
# model = NMF(n_components=2, init='random', random_state=0)
_W = np.random.randn(6,2)
model = decomposition.SparseCoder(dictionary=_W)

H = model.fit_transform(X=X)
W = model.components_

# print(W)
# print(H)
# print(_W)
print(model.n_iter)
print(np.matmul(H, W))
print(np.sum(np.square(np.matmul(H, W) - X)))