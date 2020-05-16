#！/usr/bin/env python
#-*-coding:utf-8-*-
#user:xieyuhui

import numpy as np
import pylab
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from gp_kernels.kernels.Manifold_Kernel import ManifoldKernel
from sklearn import datasets
from sklearn.model_selection import train_test_split
import xlrd


np.random.seed(1)

# Train dataset(normal)
X = np.array([[2, 2.6], [2.1, 3], [2.2, 3.1], [2.4, 3.3], [2.3, 2.2], [2.4, 2.9], [2.5, 2.5], [2.3, 2.6], [2.3, 2.7], [2.2, 2.4], [2.2, 2.7],
              [4.1, 1.2], [4.3, 1.1], [4.5, 1.5], [4.4, 2.0], [4.3, 0.9], [4.2, 1.9], [4.1, 1.5], [4.0, 1.7], [4.2, 1.6], [4.3, 1.8]])
Y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
N = len(X)
# Observation noise level
alpha = []
for i in range(N):
    alpha_ = 0.01
    alpha.append(alpha_)
alpha = np.array(alpha)

X_plot = X
Y_plot = Y + np.random.normal(0 ,alpha)

pylab.figure(0, figsize=(10, 8))

# Specify Gaussian Process
kernel = C(1.0, (0.01, 100)) \
    * ManifoldKernel.construct(base_kernel=RBF(0.1), architecture=((2, 6, 2),),
                               transfer_fct="tanh", max_nn_weight=1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2,
                              n_restarts_optimizer=10)

gp.fit(X, Y)

X1 = np.array([[2.2, 2.5], [4.3, 1.5]])

y_mean, y_std = gp.predict(X1, return_std=True)
pylab.scatter(X[0:10, 0], X[0:10, 1], c='r', s=50, zorder=10)
pylab.scatter(X[11:20, 0], X[11:20, 1], c='b', s=50, zorder=10)
pylab.show()
print("Result:")
print(y_mean)
print(y_std)

#############################################################
# Iris dataset #1
np.random.seed(1)

iris = datasets.load_iris()
X = iris['data']
y = iris['target']
X = X[y != 2]
y = y[y != 2]
X_train, X_test, y_train, y_test = train_test_split(X, y)
N = len(X_train)

alpha = []
for i in range(N):
    alpha_ = 0.01
    alpha.append(alpha_)
alpha = np.array(alpha)

pylab.figure(0, figsize=(10, 8))

# Specify Gaussian Process
kernel = C(1.0, (0.01, 100)) \
    * ManifoldKernel.construct(base_kernel=RBF(0.1), architecture=((4, 6, 5),),
                               transfer_fct="tanh", max_nn_weight=1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2,
                              n_restarts_optimizer=10)

gp.fit(X_train, y_train)

y_mean, y_std = gp.predict(X_test, return_std=True)

pylab.show()
print("Iris Result:")
print(y_mean)
print(y_std)

M = len(y_mean)
for i in range(M):
    if y_mean[i] <= 0.5:
        y_mean[i] = 0
    else:
        y_mean[i] = 1

re = (y_mean == y_test)
count = 0
for i in range(M):
    if re[i] == True:
        count = count + 1
a = count / M
print("Accuracy:", a)

##############################################################################
# image data #0.71
np.random.seed(1)

def excel_to_matrix(path):
    table = xlrd.open_workbook(path).sheets()[0]
    row = table.nrows
    col = table.ncols
    datamatrix = np.zeros((row, col))
    for x in range(col):
        cols = np.mat(table.col_values(x))
        datamatrix[:, x] = cols
    return datamatrix

datafile = u'E:\\PyCharm_projects\\Manifold Gaussian Process'
image_train_data_mat = excel_to_matrix(datafile + '\\image_train_data.xlsx')
image_train_label_mat = excel_to_matrix(datafile + '\\image_train_label.xlsx')
image_test_data_mat = excel_to_matrix(datafile + '\\image_test_data.xlsx')
image_test_label_mat = excel_to_matrix(datafile + '\\image_test_label.xlsx')
N = len(image_train_label_mat)
M = len(image_test_label_mat)
image_train_label_mat = image_train_label_mat.reshape(N)
image_test_label_mat = image_test_label_mat.reshape(M)

image_train_data_mat = image_train_data_mat.astype(np.float32)
image_train_label_mat = image_train_label_mat.astype(np.float32)
image_test_data_mat = image_test_data_mat.astype(np.float32)
image_test_label_mat = image_test_label_mat.astype(np.float32)

image_train_data_mat = image_train_data_mat[0:50, :]
image_train_label_mat = image_train_label_mat[0:50]
image_train_label_mat[image_train_label_mat == -1] = 0

image_test_data_mat = image_test_data_mat[0:100, :]
image_test_label_mat = image_test_label_mat[0:100]
image_test_label_mat[image_test_label_mat == -1] = 0

N = len(image_train_label_mat)
M = len(image_test_label_mat)

alpha = []
for i in range(N):
    alpha_ = 0.01
    alpha.append(alpha_)
alpha = np.array(alpha)

pylab.figure(0, figsize=(10, 8))

# Specify Gaussian Process
kernel = C(1.0, (0.01, 100)) \
    * ManifoldKernel.construct(base_kernel=RBF(0.1), architecture=((18, 6, 20),),
                               transfer_fct="tanh", max_nn_weight=1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2,
                              n_restarts_optimizer=1)

gp.fit(image_train_data_mat, image_train_label_mat)

y_mean, y_std = gp.predict(image_test_data_mat, return_std=True)

for i in range(M):
    if y_mean[i] <= 0.5:
        y_mean[i] = 0
    else:
        y_mean[i] = 1

re = (y_mean == image_test_label_mat)
count = 0
for i in range(M):
    if re[i] == True:
        count = count + 1
a = count / M
print("Accuracy:", a)

###########################################################
# diabetis data #0.7
np.random.seed(15)
datafile = u'E:\\PyCharm_projects\\Manifold Gaussian Process'
diabetis_train_data_mat = excel_to_matrix(datafile + '\\diabetis_train_data.xlsx')
diabetis_train_label_mat = excel_to_matrix(datafile + '\\diabetis_train_label.xlsx')
diabetis_test_data_mat = excel_to_matrix(datafile + '\\diabetis_test_data.xlsx')
diabetis_test_label_mat = excel_to_matrix(datafile + '\\diabetis_test_label.xlsx')
N = len(diabetis_train_label_mat)
M = len(diabetis_test_label_mat)
diabetis_train_label_mat = diabetis_train_label_mat.reshape(N)
diabetis_test_label_mat = diabetis_test_label_mat.reshape(M)

diabetis_train_data_mat = diabetis_train_data_mat.astype(np.float32)
diabetis_train_label_mat = diabetis_train_label_mat.astype(np.float32)
diabetis_test_data_mat = diabetis_test_data_mat.astype(np.float32)
diabetis_test_label_mat = diabetis_test_label_mat.astype(np.float32)

diabetis_train_data_mat = diabetis_train_data_mat[0:50, :]
diabetis_train_label_mat = diabetis_train_label_mat[0:50]
diabetis_train_label_mat[diabetis_train_label_mat == -1] = 0

diabetis_test_data_mat = diabetis_test_data_mat[0:100, :]
diabetis_test_label_mat = diabetis_test_label_mat[0:100]
diabetis_test_label_mat[diabetis_test_label_mat == -1] = 0

N = len(diabetis_train_label_mat)
M = len(diabetis_test_label_mat)

alpha = []
for i in range(N):
    alpha_ = 0.01
    alpha.append(alpha_)
alpha = np.array(alpha)

pylab.figure(0, figsize=(10, 8))

# Specify Gaussian Process
kernel = C(1.0, (0.01, 100)) \
    * ManifoldKernel.construct(base_kernel=RBF(0.1), architecture=((8, 6, 9),),
                               transfer_fct="tanh", max_nn_weight=1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                              n_restarts_optimizer=10)

gp.fit(diabetis_train_data_mat, diabetis_train_label_mat)

y_mean, y_std = gp.predict(diabetis_test_data_mat, return_std=True)

for i in range(M):
    if y_mean[i] <= 0.5:
        y_mean[i] = 0
    else:
        y_mean[i] = 1

re = (y_mean == diabetis_test_label_mat)
count = 0
for i in range(M):
    if re[i] == True:
        count = count + 1
a = count / M
print("Accuracy:", a)

###############################################################
# thyroid(甲状腺疾病数据集) # 0.97
np.random.seed(3)
datafile = u'E:\\PyCharm_projects\\Manifold Gaussian Process'
thyroid_train_data_mat = excel_to_matrix(datafile + '\\thyroid_train_data.xlsx')
thyroid_train_label_mat = excel_to_matrix(datafile + '\\thyroid_train_label.xlsx')
thyroid_test_data_mat = excel_to_matrix(datafile + '\\thyroid_test_data.xlsx')
thyroid_test_label_mat = excel_to_matrix(datafile + '\\thyroid_test_label.xlsx')
N = len(thyroid_train_label_mat)
M = len(thyroid_test_label_mat)
thyroid_train_label_mat = thyroid_train_label_mat.reshape(N)
thyroid_test_label_mat = thyroid_test_label_mat.reshape(M)

thyroid_train_data_mat = thyroid_train_data_mat.astype(np.float32)
thyroid_train_label_mat = thyroid_train_label_mat.astype(np.float32)
thyroid_test_data_mat = thyroid_test_data_mat.astype(np.float32)
thyroid_test_label_mat = thyroid_test_label_mat.astype(np.float32)

thyroid_train_data_mat = thyroid_train_data_mat[0:50, :]
thyroid_train_label_mat = thyroid_train_label_mat[0:50]
thyroid_train_label_mat[thyroid_train_label_mat == -1] = 0

thyroid_test_data_mat = thyroid_test_data_mat[0:100, :]
thyroid_test_label_mat = thyroid_test_label_mat[0:100]
thyroid_test_label_mat[thyroid_test_label_mat == -1] = 0

N = len(thyroid_train_label_mat)
M = len(thyroid_test_label_mat)

alpha = []
for i in range(N):
    alpha_ = 0.01
    alpha.append(alpha_)
alpha = np.array(alpha)

pylab.figure(0, figsize=(10, 8))

# Specify Gaussian Process
kernel = C(1.0, (0.01, 100)) \
    * ManifoldKernel.construct(base_kernel=RBF(0.1), architecture=((5, 6, 6),),
                               transfer_fct="tanh", max_nn_weight=1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2,
                              n_restarts_optimizer=10)

gp.fit(thyroid_train_data_mat, thyroid_train_label_mat)

y_mean, y_std = gp.predict(thyroid_test_data_mat, return_std=True)

for i in range(M):
    if y_mean[i] <= 0.5:
        y_mean[i] = 0
    else:
        y_mean[i] = 1

re = (y_mean == thyroid_test_label_mat)
count = 0
for i in range(M):
    if re[i] == True:
        count = count + 1
a = count / M
print("Accuracy:", a)
