#！/usr/bin/env python
#-*-coding:utf-8-*-
#user:xieyuhui

import numpy as np

from sklearn.gaussian_process.kernels import Kernel, _approx_fprime, Hyperparameter, RBF

class ManifoldKernel(Kernel):

    def __init__(self, w, w_bounds, base_kernel, architecture, theta_nn_size,
                 transfer_fct="tanh", max_nn_weight=5.0):
        self.w = w
        self.w_bounds = w_bounds
        self.base_kernel = base_kernel
        self.architecture = architecture
        self.theta_nn_size = theta_nn_size
        self.transfer_fct = transfer_fct
        self.max_nn_weight = max_nn_weight
        self.hyperparameter_w = Hyperparameter("w", "numeric", self.w_bounds, self.w.shape[0])

    @classmethod
    def construct(cls, base_kernel, architecture, transfer_fct="tanh", max_nn_weight=5.0):
        n_outputs, theta_nn_size = determine_network_layout(architecture)
        w = np.array(list(np.random.uniform(-max_nn_weight, max_nn_weight, theta_nn_size)) + list(base_kernel.theta))
        wL = [-max_nn_weight] * theta_nn_size \
             + list(base_kernel.bounds[:, 0])
        wU = [max_nn_weight] * theta_nn_size \
             + list(base_kernel.bounds[:, 1])
        w_bounds = np.vstack((wL, wU)).T
        return cls(w, w_bounds, base_kernel=base_kernel,
                   architecture=architecture, theta_nn_size=theta_nn_size,
                   transfer_fct=transfer_fct, max_nn_weight=max_nn_weight)

    @property
    def theta(self):
        return self.w

    @theta.setter
    def theta(self, theta):
        self.w = np.asarray(theta, dtype=np.float)
        self.base_kernel.theta = theta[self.theta_nn_size:]

    @property
    def bounds(self):
        return self.w_bounds

    @bounds.setter
    def bounds(self, bounds):
        self.w_bounds = bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        X_nn = self._project_manifold(X)
        if Y is None:
            K = self.base_kernel(X_nn)
            if not eval_gradient:
                return K
            else:
                # approximate gradient numerically
                # XXX: Analytic expression for gradient based on chain rule and
                #      backpropagation?
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(X, Y)

                return K, _approx_fprime(self.theta, f, 1e-5)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            Y_nn = self._project_manifold(Y)
            return self.base_kernel(X_nn, Y_nn)

    def diag(self, X):
        return np.diag(self(X))  # XXX

    def is_stationary(self):

        return False

    def _project_manifold(self, X, w=None):

        if self.transfer_fct == "tanh":
            transfer_fct = np.tanh
        elif self.transfer_fct == "sin":
            transfer_fct = np.sin
        elif self.transfer_fct == "relu":
            transfer_fct = lambda x: np.maximum(0, x)
        elif self.transfer_fct == "linear":
            transfer_fct = lambda x: x
        elif self.transfer_fct == "sigmoid":
            transfer_fct = lambda x: 1 / (1 + np.exp(-x))
        elif hasattr(self.transfer_fct, "__call__"):
            transfer_fct = self.transfer_fct

        if w is None:
            w = self.w

        y = []
        for subnet in self.architecture:
            y.append(X[:, :subnet[0]])
            for layer in range(len(subnet) - 1):
                W = w[:subnet[layer] * subnet[layer + 1]]
                W = W.reshape((subnet[layer], subnet[layer + 1]))
                b = w[subnet[layer] * subnet[layer + 1]:
                      (subnet[layer] + 1) * subnet[layer + 1]]
                a = y[-1].dot(W) + b
                y[-1] = transfer_fct(a)


                w = w[(subnet[layer] + 1) * subnet[layer + 1]:]

            X = X[:, subnet[0]:]

        return np.hstack(y)


def determine_network_layout(architecture):
    n_outputs = 0
    n_params = 0
    for subnet in architecture:
        for layer in range(len(subnet) - 1):
            n_params += (subnet[layer] + 1) * subnet[layer + 1]

        n_outputs += subnet[-1]

    return n_outputs, n_params






