import numpy as np
import cv2
import matplotlib.pyplot as plt

#데이터 생성
def generate_data(num_samples, num_features=2):
  np.random.seed(42)

  data_size = (num_samples, num_features)
  train_data = np.random.randint(0, 100, size=data_size)

  labels_size = (num_samples, 1)
  labels = np.random.randint(0, 2, size=labels_size)

  return train_data.astype(np.float32), labels
#파란집 빨간집 시각화
def plot_data(all_blue, all_red):
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.scatter(all_blue[:, 0], all_blue[:, 1], c='b', marker='s', s=180)
    plt.scatter(all_red[:, 0], all_red[:, 1], c='r', marker='^', s=180)
    plt.xlabel('x coordinate (feature 1)')
    plt.ylabel('y coordinate (feature 2)')

#11개의 데이터 생성
train_data, labels = generate_data(11)
#생성된 데이터의 라벨에 따라 파란집/빨간집 분류
blue = train_data[labels.ravel() == 0]
red = train_data[labels.ravel() == 1]
#시각화
plot_data(blue, red)

#분류기
#모델 생성 후 학습
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, labels)

#새로운 값 생성(label은 무시_)
newcomer, _ = generate_data(1)
plot_data(blue, red)
plt.plot(newcomer[0, 0], newcomer[0, 1], 'go', markersize=14)

#예측(findNearest 함수를 사용하는 방법)
print("[findNearest]")
#K=1일 때
print("K=1")
ret, results, neighbor, dist = knn.findNearest(newcomer, 1)
print("Predicted label:\t", results)
print("Neighbor's label:\t", neighbor)
print("Distance to neighbor:\t", dist)
#K=5
print("K=5")
ret, results, neighbor, dist = knn.findNearest(newcomer, 5)
print("Predicted label:\t", results)
print("Neighbor's label:\t", neighbor)
print("Distance to neighbor:\t", dist)

#예측(predict 함수를 사용하는 방법)
print("[predict]")
#K=1일 때
print("K=1")
knn.setDefaultK(1)
print(knn.predict(newcomer))
print("K=5")
knn.setDefaultK(7)
print(knn.predict(newcomer))