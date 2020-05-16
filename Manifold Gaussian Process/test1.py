#！/usr/bin/env python
#-*-coding:utf-8-*-
#user:xieyuhui

import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel as C
import pylab
from sklearn.gaussian_process import GaussianProcessRegressor

#X_ = np.array([[0], [1.1], [1.3], [2.2], [2.8], [3.6], [3.7], [4.6], [4.7], [4.8]])
#Y_ = np.array([[-0.1], [0.9], [1], [0.1], [0], [0.7], [0.8], [-1.1], [-1], [-0.8]])
X = np.linspace(0, 5, 100)[:, None]

# RBF
kernel = C(1.0) * RBF(length_scale=1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10)

pylab.figure(0, figsize=(14, 12))
pylab.subplot(3, 2, 1)
ymean,y_std =gp.predict(X, return_std=True)
pylab.plot(X, ymean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X[:, 0], ymean - y_std, ymean + y_std, alpha=0.5, color='k')
y_samples =gp.sample_y(X, 10)
pylab.plot(X, y_samples, color='b', lw=2)
pylab.plot(X, y_samples[:, 0], color='b', lw=2, label="sample")
pylab.legend(loc="best")
pylab.xlim(0, 5)
pylab.ylim(-3, 3)
pylab.title("Prior Samples")

#Matern
kernel = C(1.0) * Matern(length_scale=1, nu=1.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10)

pylab.subplot(3, 2, 2)
ymean,y_std =gp.predict(X, return_std=True)
pylab.plot(X, ymean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X[:, 0], ymean - y_std, ymean + y_std, alpha=0.5, color='k')
y_samples =gp.sample_y(X, 10)
pylab.plot(X, y_samples, color='r', lw=2)
pylab.plot(X, y_samples[:, 0], color='r', lw=2, label="sample")
pylab.legend(loc="best")
pylab.xlim(0, 5)
pylab.ylim(-3, 3)
pylab.title("Prior Samples")

# RationalQuadratic
kernel = C(1.0) * RationalQuadratic(alpha=0.1, length_scale=1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10)

pylab.subplot(3, 2, 3)
ymean,y_std =gp.predict(X, return_std=True)
pylab.plot(X, ymean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X[:, 0], ymean - y_std, ymean + y_std, alpha=0.5, color='k')
y_samples =gp.sample_y(X, 10)
pylab.plot(X, y_samples, color='g', lw=2)
pylab.plot(X, y_samples[:, 0], color='g', lw=2, label="sample")
pylab.legend(loc="best")
pylab.xlim(0, 5)
pylab.ylim(-3, 3)
pylab.title("Prior Samples")

# ExpSineSquared
kernel = C(1.0) * ExpSineSquared(length_scale=1, periodicity=3)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10)

pylab.subplot(3, 2, 4)
ymean,y_std =gp.predict(X, return_std=True)
pylab.plot(X, ymean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X[:, 0], ymean - y_std, ymean + y_std, alpha=0.5, color='k')
y_samples =gp.sample_y(X, 10)
pylab.plot(X, y_samples, color='k', lw=2)
pylab.plot(X, y_samples[:, 0], color='k', lw=2, label="sample")
pylab.legend(loc="best")
pylab.xlim(0, 5)
pylab.ylim(-3, 3)
pylab.title("Prior Samples")

# DotProduct
kernel = C(0.316) * DotProduct(sigma_0=1) ** 2
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10)

pylab.subplot(3, 2, 5)
ymean,y_std =gp.predict(X, return_std=True)
pylab.plot(X, ymean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X[:, 0], ymean - y_std, ymean + y_std, alpha=0.5, color='k')
y_samples =gp.sample_y(X, 10)
pylab.plot(X, y_samples, color='y', lw=2)
pylab.plot(X, y_samples[:, 0], color='y', lw=2, label="sample")
pylab.legend(loc="best")
pylab.xlim(0, 5)
pylab.ylim(-3, 3)
pylab.title("Prior Samples")
pylab.show()