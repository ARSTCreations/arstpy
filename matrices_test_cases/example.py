import sys
sys.path.append('../')
from arstpy import matrices

print(matrices.isMatrix([[1, 2], [3, 4]])) # True
print(matrices.isMatrix([[1, 2], [3, 4, 5]])) # False
print(matrices.transpose([[1, 2], [3, 4]])) # [[1, 3], [2, 4]]
print(matrices.transpose([[1, 2, 3], [4, 5, 6]])) # [[1, 4], [2, 5], [3, 6]]
print(matrices.multiply([[1, 2], [3, 4]], [[1, 2], [3, 4]])) # [[7, 10], [15, 22]]
print(matrices.inverse([[1, 2], [3, 4]])) # [[-2.0, 1.0], [1.5, -0.5]]
print(matrices.divide([[1, 2], [3, 4]], [[1, 2], [3, 4]])) # [[1.0, 1.0], [1.0, 1.0]]
print(matrices.add([[1, 2], [3, 4]], [[1, 2], [3, 4]])) # [[2, 4], [6, 8]]
print(matrices.subtract([[1, 2], [3, 4]], [[1, 2], [3, 4]])) # [[0, 0], [0, 0]]