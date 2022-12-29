#정규화
from sklearn import preprocessing
import numpy as np
X = np.array([[ 1., -2.,  2.],
              [ 3.,  0.,  0.],
              [ 0.,  1., -1.]])

X_normalized_l1 = preprocessing.normalize(X, norm='l1') #L1노름
print("[L1]")
print(X_normalized_l1)

X_normalized_l2 = preprocessing.normalize(X, norm='l2') #L2노름
print("[L2]")
print(X_normalized_l2)