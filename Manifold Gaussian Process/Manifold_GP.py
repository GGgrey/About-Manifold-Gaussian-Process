#！/usr/bin/env python
#-*-coding:utf-8-*-
#user:xieyuhui

import numpy as np
import pylab
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from gp_kernels.kernels.Manifold_Kernel import ManifoldKernel

np.random.seed(1)

# Train dataset
N = 100
X = []
Y = []

for i in range(N):
    X_sample = np.random.normal(0, 1)
    X.append(X_sample)
X.sort()

for i in range (N):
    if X[i] < 0:
        Y_sample = 0 #+ np.random.normal(0, 0.01)
        Y.append(Y_sample)
    else:
        Y_sample = 1 #+ np.random.normal(0, 0.01)
        Y.append(Y_sample)
X = np.array(X)
Y = np.array(Y)

"""
# sin
for i in range(N):
    X_sample = np.random.uniform(-3, 3)
    X.append(X_sample)
X.sort()
Y = np.sin(X)
X = np.array(X)
Y = np.array(Y)
"""

# Observation noise level
alpha = []
for i in range(N):
    alpha_ = 0.01
    alpha.append(alpha_)
alpha = np.array(alpha)

X_plot = X
Y_plot = Y + np.random.normal(0 ,alpha)

pylab.figure(0, figsize=(10, 8))
# pylab.subplot(2, 1, 1)
# pylab.scatter(X_plot, Y_plot)
# pylab.xlim(-5, 5)
# pylab.title("Train dataset")

# Specify Gaussian Process
kernel = C(1.0, (0.01, 100)) \
    * ManifoldKernel.construct(base_kernel=RBF(0.1), architecture=((1, 6, 2),),
                               transfer_fct="tanh", max_nn_weight=1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2,
                              n_restarts_optimizer=1)

gp.fit(X[:, None], Y)

X1 = np.linspace(-5, 5, 500)
Y1 = np.linspace(0, 0, 500)
for i in range (500):
    if X1[i] < 0:
        Y1[i] = 0
    else:
        Y1[i] = 1

pylab.subplot(2, 1, 1)
y_mean, y_std = gp.predict(X1[:, None], return_std=True)
pylab.plot(X1, y_mean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X1, y_mean - 2 * y_std, y_mean + 2 * y_std,
                   alpha=0.5, color='b')
# y_samples = gp.sample_y(X1[:, None], 10)
# pylab.plot(X1, y_samples, color='b', lw=1)
# pylab.plot(X1, y_samples[:, 0], color='b', lw=1, label="samples")
pylab.scatter(X_plot, Y_plot, c='r', s=50, zorder=10)
pylab.plot(X1, Y1, 'r', lw=1, zorder=9, label="true function")
pylab.xlim(-5, 5)
pylab.ylim(-2.5, 2.5)
pylab.legend(loc="best")
pylab.title("Manifold GP")
# pylab.savefig('jieyue_mgp.png')

kernel0 = C(1.0) * RBF(length_scale=0.1)
gp0 = GaussianProcessRegressor(kernel=kernel0, alpha=alpha ** 2, n_restarts_optimizer=10)

gp0.fit(X[:, None], Y)

pylab.subplot(2, 1, 2)
y_mean, y_std =gp0.predict(X1[:, None], return_std=True)
pylab.plot(X1, y_mean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X1, y_mean - 2 * y_std, y_mean + 2 * y_std, alpha=0.5, color='b')
pylab.scatter(X_plot, Y_plot, c='r', s=50, zorder=10)
pylab.plot(X1, Y1, 'r', lw=1, zorder=9, label="true function")
pylab.xlim(-5, 5)
pylab.ylim(-2.5, 2.5)
pylab.legend(loc="best")
pylab.title("Simple GP")
pylab.savefig('step_function.png')
pylab.show()

############################################


