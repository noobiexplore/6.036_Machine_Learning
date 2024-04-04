import numpy as np

class Module(object):
    def sgd_step(self, lrate):
        pass

class Linear(Module):
    def __init__(self, m, n):
        self.m, self.n = (m, n)   # (in size, out size) 
        self.W0 = np.zeros((self.n, 1))   # (n x 1)
        self.W = np.random.normal(0, 1.0 * m ** (-.5), (m, n))  # (m x n)
        
    def forward(self, A):
        self.A = A  # (m x b)
        return np.dot(self.W.T, self.A) + self.W0   # (n x b)

    def backward(self, dLdZ):    # dLdZ is (n x b)
        self.dLdW = np.dot(self.A, dLdZ.T)
        self.dLdw0 = dLdZ
        return np.dot(self.W, dLdZ)   # (m x b) dLdA

    def sgd_step(self, lrate):
        self.W = self.W - lrate * self.dLdW
        self.W0 = self.W0 - lrate* self.dLdW0

class ReLU(Module):    # Layer activation
    def forward(self, Z):
        self.A = Z * (Z > 0)
        return self.A
    
    def backward(self, dLdA):
        v = 1 * (self.A > 0)[:, 0]  # Rank 1 view, done to make a valid input for np.diag
        dAdZ = np.diag(v)
        dLdZ = np.dot(dAdZ, dLdA)
        return dLdZ

class Tanh(Module):
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A
    
    def backward(self, dLdA):
        v = 1 - np.square(self.A)[:, 0]   # Rank 1 view
        dAdZ = np.diag(v)
        dLdZ = np.dot(dAdZ, dLdA)
        return dLdZ


class SoftMax(Module):
    def forward(self, Z):
        exp = np.exp(Z) 
        self.A = exp / np.sum(exp, axis=0)
        return self.A
    
    def backward(self, dLdZ):
        return dLdZ
    
    def class_fun(self, Ypred):
        return np.argmax(Ypred, axis=0)


class NLL(Module):        # Loss
    def forward(self, Ypred, Y):
        self.Ypred = Ypred
        self.Y = Y
        nll = -((self.Y * np.log(self.Ypred)) + ((1 - self.Y) * np.log(1 - self.Ypred)))
        return nll

    def backward(self):
        d_nll = -((self.Y / self.Ypred) - ((1 - self.Y) / (1 - self.Ypred)))
        return d_nll
        
    
class Sequential:
    def __init__(self, modules, loss):           
        self.modules = modules
        self.loss = loss
    
    def sgd(self, X, Y, iters=1000, lrate=0.005):
        D, N = X.shape
        for it in range(iters):
            # Picking up a random Xt and Yt
            i = np.random.randint(N)
            Xt = X[:, i:i+1]
            Yt = Y[:, i:i+1]

            # Forward pass computing Ypred and loss
            Ypred = self.forward(Xt)
            loss = self.loss.forward(Ypred, Yt)

            # backward backpropagation
            delta = self.loss.backward()
            self.backward(delta)
            self.sgd_step(lrate)
    
    
    def forward(self, Xt):
        for m in self.modules:
            Xt = m.forward(Xt)
        return Xt
        
    def backward(self, delta):
        for m in self.modules[:, :, -1]:
            delta = m.backward(delta)

    def sgd_step(self, lrate):
        for m in self.modules:
            m.sgd_step(lrate)