#표준화
from sklearn import preprocessing
import numpy as np
X = np.array([[ 1., -2.,  2.],
              [ 3.,  0.,  0.],
              [ 0.,  1., -1.]])

X_scaled = preprocessing.scale(X) #표준화
print(X_scaled)

print(X_scaled.mean(axis=0)) #평균값 확인(0에 가까워야함)
print(X_scaled.std(axis=0)) #분산 확인(표준화된 특징행렬의 모든행은 분산 1을 가짐)