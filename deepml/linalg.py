
"""Matrix-Vector Dot Product"""
a = [[1,2],[2,4]]
b = [1,2]
output = [5,10]

def matrix_dot_vector(a:list[list[float|int]], b:list[int|float])->list[int|float]:
    if len(a[0]) != len(b):
        return -1

    output = []
    for row in a:
        vals = 0
        for i in range(len(b)):
            vals += row[i] * b[i]
        output.append(vals)
    return output

print(matrix_dot_vector(a,b))


"""Transpose of a Matrix"""
a = [[1,2,3],[4,5,6]]
output = [[1,4],[2,5],[3,6]]

def transpose_matrix(a:list[list[int|float]]) -> list[list[int|float]]:
    return [list(i) for i in zip(*a)]

transpose_matrix(a)




"""Calculate Mean by Row or Col"""
matrix = [[1,2,3], [4,5,6], [7,8,9]]
mode = 'column'
output = [4.0, 5.0, 6.0]

def calculate_matrix_mean(matrix: list[list[int|float]], mode: str) -> list[float]:
    if mode=='column':
        return [sum(i)/len(i) for i in zip(*matrix)]
    if mode=='row':
        return [sum(i)/len(i) for i in matrix]

print(calculate_matrix_mean(matrix, mode))



"""Scalar Multiplication"""
matrix = [[1,2], [3,4]]
scalar = 2
output = [[2,4], [6,8]]

def scalar_multiply(matrix: list[list[int|float]], scalar: int) -> list[list[int|float]]:
   return [[val*scalar for val in row] for row in matrix]



"""Eigenvalues of Matrix"""
matrix = [[2,1], [1,2]]
output = [3.0, 1.0]

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    a,b,c,d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    trace = a + d
    det = a*d - b*c

    discriminant = trace**2 - 4 * det
    lambda_1 = (trace + discriminant**.5)/2
    lambda_2 = (trace - discriminant**.5)/2
    return [lambda_1, lambda_2]



"""Convert Vector to Diagonal Matrix"""
x = np.array([1,2,3])

output = [[1,0,0],
    [0,2,0],
    [0,0,3]]

def make_diagonal(x):
    return x * np.identity(np.size(x))

print(make_diagonal(x))






"""Correlation Matrix

corr(X, Y) = cov(X,Y) / std(X) * std(Y)

cov(X,Y) = E[(X-E[X])*(Y-E[Y])]

"""
X = np.array([[1,2],[3,4],[5,6]])
output = [[1,1],[1,1]]
Y = None

def calculate_std_dev(A):
    return np.sqrt(np.mean((A - A.mean(0))**2, axis=0))

if Y is None:
    Y = X

n_samples = np.shape(X)[0]

# Cov(X,Y) = E[(X-E[X])(Y-E[Y])]
a = X-X.mean(0)
b = Y-Y.mean(0)
cov = a.T @ b * (1/n_samples)

print(f"X - E[X]: \n {a}")
print(f"Y - E[Y]: \n {b}")
print(cov)

stddev_X = np.expand_dims(calculate_std_dev(a), 1)
stddev_Y = np.expand_dims(calculate_std_dev(b), 1)
# stddev matrix
stddev_matrix = stddev_X @ stddev_Y.T

# correlation matrix
correl_matrix = cov / stddev_matrix
print(correl_matrix)


"""2D Translation Matrix

1. Homogeneous Rotation + translation
    * Translation Matrix is just identity, with the translation column
    [1 0 tx],
    [0 1 ty],
    [0 0 1]

    [x y 1]

2. Homogeneous Matrix @ Vector for each points
"""
import numpy as np
points = [[0, 0], [1, 0], [0.5, 1]]
tx, ty = 2, 3
output = [[2, 3], [3, 3], [2.5, 4]]


transformation_matrix = np.eye(3)
transformation_matrix[0][2] = tx
transformation_matrix[1][2] = ty
print(transformation_matrix)

homogeneous_points = np.hstack([points, np.ones((len(points), 1))])
print(homogeneous_points)

result = transformation_matrix @ homogeneous_points.T
print(result[:,:2])





""" Cosine Similarity Between Vectors

doesnt consider magnitude so (1,1) and (1000,1000) will have cosine similarity of = 1
    * project A onto B
    * then scale A by ||B||
    * this is the length of the projection of A onto B
    * this value is the dot product

hence formula is:
    1. ||A|| * cos(theta) to project A onto B
    2. then ||A|| * cos(theta) * ||B|| to get scaled length of A
    3. full formula is:
        A dot B = ||A||x||B||cos(theta)

    therefore:
        cos(theta) = A dot B / (||A||x||B||)
"""

v1 = np.array([1,2,3])
v2 = np.array([2,4,6])

def cosine_similarity(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    print("norm_v1: ", norm_v1)
    norm_v2 = np.linalg.norm(v2)
    print("norm_v2: ", norm_v2)

    dot_product = v1.dot(v2)
    print("dot product: ", dot_product)

    result = dot_product / (norm_v1 * norm_v2)
    print("result: ", result)
    return result
cosine_similarity(v1, v2)

"""Dot Product Calculator"""

vec1 = np.array([1,2,3])
vec2 = np.array([2,4,6])
dot_product = np.dot(vec1, vec2)
print(dot_product)




"""
Cross Product of Two 3D Vectors
"""

a = [1,0,0]
b = [0,1,0]


a = np.array(a)
b = np.array(b)

c = np.array([
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0]
])

print(a, b, c)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0,0,0, a[0], a[1], a[2], color='r', label='Vector A')
ax.quiver(0,0,0, b[0], b[1], b[2], color='g', label='Vector B')
ax.quiver(0,0,0, c[0], c[1], c[2], color='b', label='Vector C')
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
