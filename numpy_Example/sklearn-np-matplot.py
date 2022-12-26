import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()

print(digits.data.shape) #(1797, 64)
print(digits.images.shape) #(1797, 8, 8)

img = digits.images[0, :, :] #datasets에서의 하나의 이미지
plt.imshow(img, cmap='gray')
plt.savefig('figures/one8x8.png')

