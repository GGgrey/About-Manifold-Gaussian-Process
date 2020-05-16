#！/usr/bin/env python
#-*-coding:utf-8-*-
#user:xieyuhui

import numpy as np
import pylab
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from gp_kernels.kernels.Manifold_Kernel import ManifoldKernel

np.random.seed(1)

X1 = list(np.linspace(0.1, 0.25, 2))
X2 = list(np.linspace(0.65, 0.85, 5))
X1.extend(X2)
X = X1
X = np.array(X)
Y = np.sin(2 * np.pi * X)
N = X.shape[0]

alpha = []
for i in range(N):
    alpha_ = 0.01
    alpha.append(alpha_)
alpha = np.array(alpha)

X_plot = X
Y_plot = Y + np.random.normal(0, alpha)
pylab.scatter(X_plot, Y_plot)

kernel = C(1.0, (0.01, 100)) \
    * ManifoldKernel.construct(base_kernel=RBF(length_scale=10), architecture=((1, 6, 2),),
                               transfer_fct="tanh", max_nn_weight=1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2,
                              n_restarts_optimizer=1)
'''
kernel = C(1.0) * RBF(length_scale=0.1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, n_restarts_optimizer=10)
'''

gp.fit(X[:, None], Y)

XX = np.linspace(-1.5, 1.5, 100)
YY = np.sin(2 * np.pi * XX)

pylab.figure(0, figsize=(10, 8))
pylab.subplot(2, 1, 1)
pylab.figure(0, figsize=(10, 8))
y_mean, y_std = gp.predict(XX[:, None], return_std=True)
pylab.plot(XX, y_mean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(XX, y_mean - 2 * y_std, y_mean + 2 * y_std,
                   alpha=0.5, color='m')
pylab.scatter(X_plot, Y_plot, c='r', s=50, zorder=10)
pylab.plot(XX, YY, 'r', lw=1, zorder=9, label="true function")
pylab.xlim(0, 1)
pylab.ylim(-1.5, 1.5)
pylab.legend(loc="best")
pylab.title("Manifold GP")


kernel = C(1.0) * RBF(length_scale=0.1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, n_restarts_optimizer=10)

gp.fit(X[:, None], Y)

XX = np.linspace(-1.5, 1.5, 100)
YY = np.sin(2 * np.pi * XX)

pylab.subplot(2, 1, 2)
pylab.figure(0, figsize=(10, 8))
y_mean, y_std = gp.predict(XX[:, None], return_std=True)
pylab.plot(XX, y_mean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(XX, y_mean - 2 * y_std, y_mean + 2 * y_std,
                   alpha=0.5, color='m')
pylab.scatter(X_plot, Y_plot, c='r', s=50, zorder=10)
pylab.plot(XX, YY, 'r', lw=1, zorder=9, label="true function")
pylab.xlim(0, 1)
pylab.ylim(-1.5, 1.5)
pylab.legend(loc="best")
pylab.title("Simple GP")
pylab.savefig('sin_function.png')
pylab.show()