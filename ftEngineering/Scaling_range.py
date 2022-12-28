#특징의 범위 확장
from sklearn import preprocessing
import numpy as np
X = np.array([[ 1., -2.,  2.],
              [ 3.,  0.,  0.],
              [ 0.,  1., -1.]])

min_max_scaler = preprocessing.MinMaxScaler() #기본(0~1)
X_min_max = min_max_scaler.fit_transform(X)
print(X_min_max)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-10, 10)) #지정(-10, 10)
X_min_max2 = min_max_scaler.fit_transform(X)
print(X_min_max2)