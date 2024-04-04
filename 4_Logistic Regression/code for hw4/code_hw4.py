import numpy as np

# Function 1 f1(x) and its derivative   
def f1(x):
    return float((2 * x + 3) ** 2)

def df1(x):
    return float(2 * 2 * (2 * x + 3))


# Function 2 f(x, y) and its gradient vector
def f2(v):
    x = float(v[0, 0]); y = float(v[1, 0])
    return ((x - 1)**2) + (y**2)

def df2(v):
    x = float(v[0, 0]); y = float(v[1, 0])
    return np.array([[(2 * (x - 1)), (2 * y)]]).T

def step_size_fn(i):
    return 0.1

'''Analytic Gradient Descent'''
# Gradient descent algorithm using analytic gradient
def gd(f, df, x0, step_size_fn, max_iter):
    xs = [x0]
    fs = [f(x0)] 
    for i in range(max_iter):
        x_old = xs[-1]
        x_new = x_old - step_size_fn(i) * df(x_old)
        xs.append(x_new)
        fs.append(f(x_new))
        if abs(fs[-1] - fs[-2]) < 0.0001:
            break
    return (xs[-1], xs, fs)

def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), [fs[0], fs[-1]], [xs[0].tolist(), xs[-1].tolist()]]

#ans=package_ans(gd(f2, df2, np.array([[4., 4.]]).T, lambda i: 0.01, 1000))

'''Numerical Gradient Descent'''
# Calculating numerical gradient
def num_grad(f, delta=0.001):
    def df(x):
        d, _ = x.shape
        # Initializing Del vector [0 0 0 ....(ith)delta .....0]
        delv = np.zeros((d, 1))
        gradv = np.zeros((d, 1))
        for i in range(d):
            delv[i, 0] = delta
            gradv[i, 0] = (f(x + delv) - f(x - delv)) / (2 * delta)
            delv[i, 0] = 0
        return gradv
    return df

a = np.array([[2.0, 2.0]]).T
#print(num_grad(f2)(a))


# Gradient descent algorithm using numerical gradient
def minimize(f, x0, step_size_fn, max_iter):
    xs = [x0]
    fs = [f(x0)]
    for i in range(max_iter):
        x_old = xs[-1]
        x_new = x_old - step_size_fn(i) * num_grad(f)(x_old)
        xs.append(x_new)
        fs.append(f(x_new))
        if abs(fs[-1] - fs[-2]) < 0.00001:
            break
    return (xs[-1], xs, fs)


x_min, _, _ = minimize(f2, np.array([[4, 4]]).T, lambda i: 0.1, 1000)
#print(x_min)

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])

'''SVM objective function'''

'''Basically implements the max(0, 1-v) piece-wise function without using loops
utilizes numpy boolean array to achieve it'''
def hinge(v):
    c = 1 - v
    idx = c <= 0
    c[idx] = 0
    return c

'''Calculates hinge loss. First find the 1xn margin array and then sends
it to hinge function for max(0, 1-margin)'''
# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    margin = y * (np.dot(th.T, x) + th0)
    return hinge(margin)

'''Returns svm objective function J(th, th0) = avg_hinge_loss + regularizer(th)'''
# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    _, n = x.shape
    avg_hinge_loss = np.sum(hinge_loss(x, y, th, th0)) / n
    # SVM objective
    J = avg_hinge_loss + (lam * ((np.linalg.norm(th)) ** 2))
    return J

# Test case 1
x_1, y_1 = super_simple_separable()
th1, th1_0 = sep_e_separator
ans = svm_obj(x_1, y_1, th1, th1_0, .1)
print(ans)


'''Gradient of SVM Objective function'''

'''Return gradient of max(0, 1-v) where v is a numpy array. Uses ananlytic 
gradient of hinge function which is also piece-wise'''
# Returns the gradient of hinge(v) with respect to v.
def d_hinge(v):
    dv = np.copy(v)
    idx = v < 1
    dv[idx] = -1 
    # inverting the boolean array to assign values to the False indexes of idx
    not_idx = np.invert(idx)
    dv[not_idx] = 0
    return dv

'''Uses analytical gradient of hinge loss which is either 0 or -yx. 
First send the margin to the d_hinge function to get the coefficient
0 or -1(to get the piece-wise behavior). Then multiply it with yx to get the final gradient'''
# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
    margin = y * (np.dot(th.T, x) + th0)
    d_hinge_coeff = d_hinge(margin)
    grad_hinge_loss_th = d_hinge_coeff * (y * x)
    return grad_hinge_loss_th


'''Analytical partial derivative of hinge_loss wrt th0 is 0 or -y.
First send the margin to the d_hinge function to get the coefficient
0 or -1(to get the piece-wise behavior). Then multiply it with y to get the derivative'''
# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
    margin = y * (np.dot(th.T, x) + th0)
    d_hinge_coeff = d_hinge(margin)
    partial_hinge_loss_th0 = d_hinge_coeff * y
    return partial_hinge_loss_th0


# Returns the gradient of svm_obj(x, y, th, th0) with respect to th
def d_svm_obj_th(x, y, th, th0, lam):
    _, n = x.shape
    ''' Adding up all the columns of gradient of hinge loss wrt th into a single column and 
    dividing by n to get the average'''
    avg_grad_hinge_loss_th = np.array([np.sum(d_hinge_loss_th(x, y, th, th0), axis=1)]).T / n
    grad_J_th = avg_grad_hinge_loss_th + (lam * th)
    return grad_J_th

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
    _, n = x.shape
    avg_partial_hinge_loss_th0 = np.array([np.sum(d_hinge_loss_th0(x, y, th, th0), axis=1)]).T / n
    return avg_partial_hinge_loss_th0

# Returns the full gradient as a single vector (which includes both th, th0)
def svm_obj_grad(X, y, th, th0, lam):
    svm_grad_th = d_svm_obj_th(X, y, th, th0, lam)
    svm_grad_th0 = d_svm_obj_th0(X, y, th, th0, lam)
    svm_grad = np.vstack((svm_grad_th, svm_grad_th0))
    return svm_grad


def batch_svm_min(data, labels, lam):
    def svm_min_step_size_fn(i):
       return 2/(i+1)**0.5
    # Initializing th i.e. a single vector containing [th, th0].T 
    d, n = data.shape
    th_init = np.zeros((d+1, 1))
    # Applying gradient descent
    ths = [th_init]
    svms = [svm_obj(data, labels, th_init[:d], th_init[d:d+1], lam)]
    for i in range(10):
        th_old = ths[-1]
        th_new = th_old - svm_min_step_size_fn(i) * svm_obj_grad(data, labels, th_old[:d], th_old[d:d+1], lam)
        ths.append(th_new)
        svms.append(svm_obj(data, labels, th_new[:d], th_new[d:d+1], lam))
        if abs(svms[-1] - svms[-2]) < 0.00001:
            break
    return ths[-1], ths, svms
        

'''Numerical SVM gradient'''
def num_svm_grad_th(svm_obj_func, delta=0.001):
    def grad_th_svm(data, labels, th, th0, lam):
        d, n = data.shape
        del_th = np.zeros((d, 1))
        grad_th = np.zeros((d, 1))
        for i in range(d):
            del_th[i, 0] = delta
            grad_th[i, 0] = (svm_obj_func(data, labels, (th + del_th), th0, lam) - \
                svm_obj_func(data, labels, (th - del_th), th0, lam)) / (2 * del_th)
            del_th[i, 0] = 0
        return grad_th
    return grad_th_svm

def num_svm_grad_th0(svm_obj_func, delta=0.001):
    def grad_th0_svm(data, labels, th, th0, lam):
        del_th0 = np.full((1, 1), delta)
        partial_th0 = np.zeros((1, 1))
        partial_th0[0, 0] = (svm_obj_func(data, labels, th, th0 + del_th0, lam) - \
                svm_obj_func(data, labels, th, th0 - del_th0, lam)) / (2 * del_th0)
        return partial_th0
    return grad_th0_svm
