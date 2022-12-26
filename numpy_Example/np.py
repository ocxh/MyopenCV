#리스트의 곱셈과 numpy의 곱셈
import numpy as np

int_list = list(range(10))
print(int_list * 2) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

int_arr = np.array(int_list)
print(int_arr * 2) # [ 0  2  4  6  8 10 12 14 16 18]

