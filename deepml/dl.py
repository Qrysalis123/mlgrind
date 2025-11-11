
"""Sigmoid Activation Function Understanding
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (3, 2)

z = 0

xs = np.linspace(-5,5, 100)
zs = np.exp(-xs)
plt.title("e^(-x)")
plt.plot(xs, zs)
plt.show()

sigmoid = 1/(1+zs)
plt.title("1 / (1+e**-z)")
plt.plot(xs, sigmoid)
plt.show()


"""Softmax Activation Function Implementation
"""
scores = [1,2,3]
exp_scores = np.exp(scores)
prob = exp_scores / np.sum(exp_scores)
plt.title("exp^z/np.sum(exp^z)")
plt.plot(prob)
plt.show()






"""Single Neuron w/o backprop

X.shape = (3,2)
w.shape = (2,1)
bias.shape = (1,)

z = X @ w + bias # (3,1)

y_pred = softmax(z) # (3,1)

mse = np.mean(y_pred - labels.reshape(-1,1)) #


"""

features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]]
labels = [0, 1, 0]
weights = [0.7, -0.4]
bias = -0.1


features = np.array(features)
labels = np.array(labels).reshape(-1,1)
weights = np.array(weights).reshape(1,-1)
bias = np.array(bias).reshape(1,-1)

print(f"features shape: {features.shape}")
print(f"weights shape: {weights.shape}")
print(f"bias shape: {bias.shape}")

z = features @ weights.T + bias
print(f"z shape: {z.shape}")
probs = 1 / (1+np.exp(-z))
print(f"probs shape: {probs}")

mse = np.mean((probs - labels)**2)
print(f"mse: {mse}")



"""
Single Neuron with Backpropagation

gradient descent:
    * mse dw = 2/n * sum(error * predictions * (1 - predictions)) * features
    * mse db = 2/n * sum(errors * predictions * (1 - predictions))
"""

features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]
labels = [1,0,0]
initial_weights = [0.1, -0.2]
initial_bias = 0.0
learning_rate = 0.1
epochs = 2

features = np.array(features) # (3, 2)
labels = np.array(labels) # (3,)
weights = np.array(initial_weights) # (2,)
bias = np.array(initial_bias) # (1,)
print(f"features shape: {features.shape}")
print(f"labels shape: {labels.shape}")
print(f"weights shape: {weights.shape}")
print(f"bias shape: {bias.shape}")

n = features.shape[0]
sigmoid = lambda x: 1/(1+np.exp(-x))
mse_values = []

for epoch in range(epochs):
    z = features @ weights.T + bias
    prob = sigmoid(z)
    mse = np.mean((prob - labels)**2)
    mse_values.append(mse)

    # dw db
    error = prob - labels
    dw = (2/n) * features.T @ (error * prob * (1-prob))
    db = (2/n) * np.sum(error * prob * (1-prob))

    # update
    weights -= learning_rate * dw
    bias -= learning_rate * db

    print(f"epoch {epoch}: mse = {mse:.4f}")
print(f"weights: {weights}")
print(f"bias: {bias}")


"""Implementing a Custom Dense Layer in Python

* create fully connected layer
need:

    __init__(self, n_units, input_shape=None):

    initialize weights using uniform dist 1/(sqrt(input_shape[0]))
    bias w0 set 0

    param count: return total trianable params which includes w and w0 params

    forward pass: np.dot(x, w) + bias

    backward pass: compute adn return gradient wrt x
    if layers are trainable, then update the weights and biases

    output shape: return shape of output produced by forward pass. shape is (self.n_units,)

"""


import copy
np.random.seed(42)

n_units = 3
input_shape = (2,)
X = np.array([[1,2]])
accum_grad = np.array([[0.1, 0.2, 0.3]]) # sample gradient

class MockOptimizer:
    def update(self, weights, grad):
        return weights - 0.01 * grad

optimizer = MockOptimizer()

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):

		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape=input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        limit = 1 / np.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units)) # low high size
        self.w0 = np.zeros((self.n_units))
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self): # returns number of W + bias params
        return np.prod(self.W.shape) + np.prod(self.w0.shape)


    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X @ self.W + self.w0

    def backward_pass(self, accum_grad):
        W = self.W
        if self.trainable:
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)
        accum_grad = accum_grad.dot(W.T)
        return accum_grad

        def output_shape(self):
            return (self.n_units,)


dense_layer = Dense(n_units=3, input_shape=(2,))
dense_layer.initialize(optimizer)

output = dense_layer.forward_pass(X)
print(f"Forward pass: {output}")
back_output = dense_layer.backward_pass(accum_grad)
print(f"Backward pass: {back_output}")

"""Simple Convolutional 2D Layer"""

input_matrix = np.array([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

# simple conv2d
input_height, input_width = input_matrix.shape
kernel_height, kernel_width = kernel.shape

# np.pad modes=['constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median']
# pad_width: number of values to pad for each exis. ((pad for axis=0), (pad for axis=1))
# (array, pad_width, mode, constant_values) the values to use as pad
#
padded_input = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant')
print(f"padded input: {padded_input}")
input_height_padded, input_width_padded = padded_input.shape
print(f"padding HxW: {input_height_padded}, {input_width_padded}")


output_height = (input_height_padded - kernel_height) // stride + 1
output_width = (input_width_padded - kernel_width) // stride + 1
output_matrix = np.zeros((output_height, output_width))
print(output_matrix)


for i in range(output_height):
    for j in range(output_width):
        region = padded_input[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
        output_matrix[i,j] = np.sum(region * kernel)

print(output_matrix)





"""Implement ReLU Activation Function"""
relu = lambda x: max(0,x)
relu(-1)



"""Implement Self-Attention Mechanism

Q = XWq
K = XWk
V = XWv

S = attention score = Q @ K.T
S = norm attention score = Q @ K.T / sqrt(dim.K)
A = softmax(S)
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(dim.K)) @ V

"""

X = np.array([[1,0], [0,1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q = X @ W_q # (2,2)
K = X @ W_k # (2,2)
V = X @ W_v # (2,2)

softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


S = (Q @ K.T) / np.sqrt(K.shape[1]) # (2,2)
A = softmax(S) # (2,2)
attention = A @ V
print(attention)



"""
KL Divergence Between Two Normal Distributions


entropy diff

= p(x)log(p(x)) - log(q(x))
= p(x) * log(p(x) / q(x))
Divergence between 2
0 = no divergence


for dist params: mu, sigma:
    KL(P || Q) = log(sigma q / sigma p) + sigma_p**2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2) - 0.5
"""
mu_p, sigma_p = 0.0, 1.0
mu_q, sigma_q = 1.0, 1.0

term1 = np.log(sigma_q / sigma_p)
term2 = (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)
kl_div = term1 + term2 - 0.5
print(kl_div)




"""Positional Encoding Calculator

using sine and cosine functions for positional encodings
"""
# total positions or length of sequence, dimensionality of models output
position = 2; d_model = 8

pos = np.arange(position, dtype=np.float32).reshape(position, 1)
ind = np.arange(d_model, dtype=np.float32).reshape(1, d_model)
print(pos, ind)
# generate base matrix. rows = positions, columns = feature dims
angle_rads = pos / np.power(10000, (2*(ind//2)) / d_model)
print(f"angle rads: {angle_rads}")

# apply sine and cosine
# for even apply sine, for odd apply cosine
angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) # 0,2,4...
angle_rads[:, 1::2] = np.sin(angle_rads[:, 1::2]) # 1,3,5...

print(f"angle rads: {angle_rads}")


"""Implement Multi-Head Attention"""

X = np.array([[1, 2], [3, 4]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 0], [0, 1]])

# compute qkv
Q = X @ W_q # (2,2)
K = X @ W_k # (2,2)
V = X @ W_v # (2,2)

# self attention
scores = Q @ K.T / np.sqrt(Q.shape[1]) # (2,2)
score_max = np.max(scores, axis=1, keepdims=True) # (2,1)
attention_weights = np.exp(scores-score_max)/np.sum(np.exp(scores-score_max),axis=1,keepdims=True) # (2,2)
attention_output = attention_weights @ V # (2,2)

# mha
def self_attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(Q.shape[1]) # (2,2)
    score_max = np.max(scores, axis=1, keepdims=True) # (2,1)
    attention_weights = np.exp(scores-score_max)/np.sum(np.exp(scores-score_max),axis=1,keepdims=True) # (2,2)
    attention_output = attention_weights @ V # (2,2)
    return attention_output

n_heads = 2

d_model = Q.shape[1]
assert d_model % n_heads == 0 # ensure col dim can be split by n heads
d_k = d_model // n_heads # Dim for each head

# Reshape QKV into batches. (2,2) -> (n_heads, seq_len, d_k)
Q_reshaped = Q.reshape(Q.shape[0], n_heads, d_k).transpose(1,0,2)
K_reshaped = K.reshape(K.shape[0], n_heads, d_k).transpose(1,0,2)
V_reshaped = V.reshape(V.shape[0], n_heads, d_k).transpose(1,0,2)
print(Q_reshaped.shape)

attentions = []

for i in range(n_heads):
    attn = self_attention(Q_reshaped[i], K_reshaped[i], V_reshaped[i]) # (2, 1)
    attentions.append(attn)

attention_output = np.concatenate(attentions, axis=-1)
attention_output





"""Implement a Simple Residual Block with Shortcut Connection"""

# helps solve vanisihing gradient problem
x = np.array([1.0, 2.0])
w1 = np.array([[1.0, 0.0], [0.0, 1.0]])
w2 = np.array([[0.5, 0.0], [0.0, 0.5]])

# 1st layer
y = w1 @ x
# 1st ReLu
y = np.maximum(0, y)
# 2nd layer
y = w2 @ y
# add shortcut conn ( x + F(x))
y = y + x
# Final ReLu
y = np.maximum(0, y)
print(y)




"""Implement Batch Normalization for BCHW Input"""

B,C,H,W = 2,2,2,2
X = np.random.randn(B,C,H,W)
gamma = np.ones(C).reshape(1, C, 1, 1)
beta = np.zeros(C).reshape(1, C, 1, 1)
epsilon = 1e-5

# normalize across batch and spatial dims (H W) for each channel
mean = np.mean(X, axis=(0, 2,3), keepdims=True)
var = np.var(X, axis=(0,2,3), keepdims=True)
X_norm = (X - mean) / np.sqrt(var + epsilon)
norm_X = gamma * X_norm + beta
