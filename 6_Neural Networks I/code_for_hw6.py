import numpy as np

z = np.array([[-1, 0, 1]]).T
exp = np.exp(z)
soft_max = exp / np.sum(exp)
#print(soft_max)

# Weight matrices of all layers
W_1 = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
W0_1 = np.full((4, 1), -1)
W_2 = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]]).T
W0_2 = np.array([[0, 2]]).T

# Input vector
X = np.array([[0.5, 0, -3], [0.5, 2, 0.5]])

# First layer dot pre activation i.e. product output
z_1 = np.dot(W_1.T, X) + W0_1

# f1 is ReLU activation function
def ReLU(x):
    return x * (x > 0)

# Output of the first layer
a1 = ReLU(z_1)
print(a1)


# 2nd layer pre activation i.e. dot product output
z_2 = np.dot(W_2.T, a1) + W0_2
print(z_2)


def SoftMax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)

a2 = SoftMax(z_2)
print(a2)