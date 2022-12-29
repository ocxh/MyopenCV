#누락 데이터 처리
import numpy as np
from numpy import nan
from sklearn.impute import SimpleImputer
X = np.array([[ nan, 0,   3  ],
              [ 2,   9,  -8  ],
              [ 1,   nan, 1  ],
              [ 5,   2,   4  ],
              [ 7,   6,  -3  ]])
print("[누락 데이터 처리]")

print("mean : 평균값")
imp = SimpleImputer(strategy='mean')
X2 = imp.fit_transform(X)
print(X2)
print(np.mean(X[1:, 0]), X2[0, 0])

print("median : 중앙값")
imp = SimpleImputer(strategy='median')
X3 = imp.fit_transform(X)
print(X3)
print(np.median(X[1:, 0]), X3[0, 0])