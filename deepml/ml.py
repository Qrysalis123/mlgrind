""" PCA
https://www.youtube.com/watch?v=FD4DeN81ODY&t=230s
"""

import numpy as np
data = np.array([[1,2],[3,4],[5,6]])
k = 1

# standardize
data_standardized = (data - data.mean(0)) / np.std(data,0)

# covariance matrix
cov_matrix = np.cov(data_standardized, rowvar=False)

# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(eigenvalues, eigenvectors)

# Sort by eigenvalues
idx = np.argsort(eigenvalues)[::-1] # default np.argsort is ascending
eigenvectors_sorted = eigenvectors[idx]
eigenvalues_sorted = eigenvalues[idx]
print(eigenvectors_sorted)

# top K pca
pca = eigenvectors_sorted[:, :k]
pca1 = np.round(pca, 4)
print("pca1: ", pca1)


"""Random Shuffle of Dataset

X, Y shuffled randomly maintaining corresponding order between them
"""
X = np.array([[1,2],
    [3,4],
    [5,6],
    [7,8]])

y = np.array([1,2,3,4])

seed = None

if seed:
    np.random.seed(seed)
idx = np.arange(X.shape[0])
print("idx: ", idx)
np.random.shuffle(idx)
print("shuffled idx: ", idx)




"""Batch Iterator for Dataset

turns numpy array into batches loader of specified size

if y is provided, iterates batches of (X, y)
"""
X = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
y = np.array([1,2,3,4,5])
batch_size = 2


n_samples = X.shape[0]
batches = []

for i in range(0, n_samples, batch_size):
    begin, end = i, min(i+batch_size, n_samples) # i+interval, but min() if interval > n_samples
    print(begin, end)
    if y is not None:
        batches.append([X[begin:end], y[begin:end]])
    else:
        batches.append(X[begin:end])

print(batches)





"""
R-Squared for Regression Analysis

variance explained by prediction out of total variance of true
ssr / sst
"""

y_true = np.array([1,2,3,4,5])
y_pred = np.array([1.1,2.1,2.9,4.2,4.8])

if np.array_equal(y_true, y_pred):
    r2 = 1.0

# mean of true
y_mean = np.mean(y_true)

# sum of squares
ssr = np.sum((y_pred - y_true)**2)

# total sum of squares
sst = np.sum((y_true - y_mean)**2)

# variance explained by prediction
r2 = 1 - (ssr/sst)
print(r2)

"""Implement Layer Normalization for Sequence Data

given [B, N, D] normalize on D

* batch, sequence, feature dim -> normalize across feature dim for each sequence
* apply scaling and shifting params
"""
np.random.seed(42)
X = np.random.randn(2, 2, 3)
gamma = np.ones(3).reshape(1, 1, -1)
beta = np.zeros(3).reshape(1, 1, -1)
epsilon = 1e-5
print("X: ", X)
# [B, N, D] mean on D, so axis = -1
# keepdims maintains dimension
no_keepdims = np.mean(X, axis=-1)
print("X mean no dims: ", no_keepdims, "X shape: ", no_keepdims.shape)

# keep 3D, but D becomes 1
mean = np.mean(X, axis=-1, keepdims=True)
print("X mean dims: ", mean, "X shape: ", mean.shape)

# std of D
var = np.var(X, axis=-1, keepdims=True)
print("X var dims: ", var, "X shape: ", var.shape)

# Numpy element wise operations:
# Looks at broadcasting rules:
#   starts rightmost dimensions and works left
#   (2,2,3) - (2,2,1)
#   3 -> 1 compatible
#   so (2,2,1) expands to (2,2,3) by replicating values on 1
#   then elementwise operation is done
X_norm = (X - mean) / np.sqrt(var + epsilon)
norm_X = gamma * X_norm + beta
print(norm_X)
"""Implement Early Stopping Based on Validation Loss

early stopping criterion:
    * if validation loss hasnt "improved" for specified number of epochs
    * "improved" defined as when loss decreased by (min_delta)
"""

val_losses = [0.9, 0.8, 0.75, 0.77, 0.76, 0.77, 0.78]
patience, min_delta = 2, 0.01


best_loss = float('inf')
best_epoch = 0
epochs_without_improvement = 0

for epoch,loss in enumerate(val_losses):
    print(f"epoch: {epoch}, loss: {loss}")
    if loss < best_loss - min_delta:
        best_loss = loss
        best_epoch = epoch
        epochs_without_improvement = 0
    else:
        epochs_without_improvement +=1
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}, best epoch {best_epoch}")
        break
