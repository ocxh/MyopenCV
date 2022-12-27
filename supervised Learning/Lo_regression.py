import numpy as np
import cv2

from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics

import matplotlib.pyplot as plt

plt.style.use('ggplot')
iris = datasets.load_iris()

#이진분류 문제로 만들기(클래스2에 속하지 않으면 삭제)
idx = iris.target != 2
data = iris.data[idx].astype(np.float32)
target = iris.target[idx].astype(np.float32)

plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=target, cmap=plt.cm.Paired, s=100)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1]);

#훈련세트와 테스트세트 나누기
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    data, target, test_size=0.1, random_state=42
)

#로지스틱 회귀
lr = cv2.ml.LogisticRegression_create()
#훈련방법 지정
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(1)
lr.setIterations(100) #반복횟수

lr.train(X_train, cv2.ml.ROW_SAMPLE, y_train);

#분류기 테스트
ret, y_pred = lr.predict(X_train) #훈련세트 정확도
train_score = metrics.accuracy_score(y_train, y_pred)
print("[Train]: ",train_score)
ret, y_pred = lr.predict(X_test) #테스트세트 정확도
test_score = metrics.accuracy_score(y_test, y_pred)
print("[Test]: ",test_score)