import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
from sklearn import linear_model

boston = datasets.load_boston() #boston 데이터 불러오기

linreg = linear_model.LinearRegression() #선형회귀모델 생성

X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, test_size=0.1, random_state=42
)

linreg.fit(X_train, y_train)

train_mean_squared_error = metrics.mean_squared_error(y_train, linreg.predict(X_train)) #훈련세트의 평균제곱오차
print("[Train]mean squared error: ", train_mean_squared_error)
R_squared = linreg.score(X_train, y_train) #결정계수(R^2)
print("R squared: ",R_squared)

#모델 테스트
y_pred = linreg.predict(X_test)
test_mean_squared_error  = metrics.mean_squared_error(y_test, y_pred) #테스트세트의 평균제곱오차
print("[Test]mean squared error: ", test_mean_squared_error)

plt.figure(figsize=(10, 6))
plt.plot(y_test, linewidth=3, label='ground truth')
plt.plot(y_pred, linewidth=3, label='predicted')
plt.legend(loc='best')
plt.xlabel('test data points')
plt.ylabel('target value')

plt.figure(figsize=(10, 6))
plt.plot(y_test, y_pred, 'o')
plt.plot([-10, 60], [-10, 60], 'k--')
plt.axis([-10, 60, -10, 60])
plt.xlabel('ground truth')
plt.ylabel('predicted')

scorestr = r'R$^2$ = %.3f' % linreg.score(X_test, y_test)
errstr = 'MSE = %.3f' % metrics.mean_squared_error(y_test, y_pred)
plt.text(-5, 50, scorestr, fontsize=12)
plt.text(-5, 45, errstr, fontsize=12);