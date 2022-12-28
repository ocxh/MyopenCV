#이진화
from sklearn import preprocessing
import numpy as np
X = np.array([[ 1., -2.,  2.],
              [ 3.,  0.,  0.],
              [ 0.,  1., -1.]])

binarizer = preprocessing.Binarizer(threshold=0.5) #임계값 0.5
X_binarized = binarizer.transform(X)
print(X_binarized)