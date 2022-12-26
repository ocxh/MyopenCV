import numpy as np

int_arr = np.array(list(range(10)))
print(int_arr) #[0 1 2 3 4 5 6 7 8 9]
#ndim : 차원 수
print(int_arr.ndim) #1
#shape : 각 차원의 크기
print(int_arr.shape) #(10, )
#size : 배열의 총 요소 수
print(int_arr.size) #10
#dtype : 배열의 데이터 유형
print(int_arr.dtype) #int64