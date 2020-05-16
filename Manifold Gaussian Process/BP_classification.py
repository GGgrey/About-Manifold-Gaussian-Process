#！/usr/bin/env python
#-*-coding:utf-8-*-
#user:xieyuhui

from sklearn import datasets
import numpy as np
import xlrd
from sklearn.model_selection import train_test_split

##############################################################
# Iris data # 1
np.random.seed(0)

iris = datasets.load_iris()
X = iris['data']
y = iris['target']
X = X[y != 2]
y = y[y != 2]
X_train, X_test, y_train, y_test = train_test_split(X, y)

num = X_train.shape[0]
input = X_train.shape[1] #4
output = 2
hid = 5
w1 = np.random.randn(input, hid) #(4 5)
w2 = np.random.randn(hid, output) #(5 2)
bias1 = np.zeros([hid, 1])
bias2 = np.zeros([output, 1])
rate1 = 0.25
rate2 = 0.25
temp1 = np.zeros([hid, 1])
net = temp1
temp2 = np.zeros([output, 1])
z = temp2

for i in range(100):
    for j in range(num):
        label = np.zeros([output, 1]) #(2 1)
        label[y_train[j]][0] = 1
        temp1 = np.dot(X_train[j][:], w1)[:, None] + bias1 #(5 1)
        net = 1 / (1 + np.exp(-temp1)) #(5 1)
        temp2 = (np.dot(net.transpose(), w2)).transpose() + bias2 #(2 1)
        z = 1 / (1 + np.exp(-temp2)) #(2 1)
        error = label - z #(2 1)
        deltaZ = error * z * (1-z) #(2 1)
        deltaNet = net * (1-net) * (np.dot(w2, deltaZ)) #(5 1)
        for k in range(output):
            w2[:, k][:, None] = w2[:, k][:, None] + rate2 * deltaZ[k] * net
        for k in range(hid):
            w1[:, k][:, None] = w1[:, k][:, None] + rate1 * deltaNet[k] * X_train[j][:].transpose()[:, None]
        bias2 = bias2 + rate2 * deltaZ
        bias1 = bias1 + rate1 * deltaNet

test_num = X_test.shape[0] # 25
count = 0
for i in range(test_num):
    temp1 = np.dot(X_test[i][:], w1)[:, None] + bias1
    net = 1 / (1 + np.exp(-temp1))
    temp2 = (np.dot(net.transpose(), w2)).transpose() + bias2
    z = 1 / (1 + np.exp(-temp2))
    index = np.nanargmax(z)
    if index == y_test[i]:
        count = count + 1

accuracy = count / test_num
print(accuracy)

##########################################################################
# image data # 0.77
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

image_train_data_mat = image_train_data_mat.astype(np.int)
image_train_label_mat = image_train_label_mat.astype(np.int)
image_test_data_mat = image_test_data_mat.astype(np.int)
image_test_label_mat = image_test_label_mat.astype(np.int)

image_train_data_mat = image_train_data_mat[0:50, :]
image_train_label_mat = image_train_label_mat[0:50]
image_train_label_mat[image_train_label_mat == -1] = 0

image_test_data_mat = image_test_data_mat[0:100, :]
image_test_label_mat = image_test_label_mat[0:100]
image_test_label_mat[image_test_label_mat == -1] = 0

N = len(image_train_label_mat)
M = len(image_test_label_mat)

np.random.seed(0)

num = image_train_data_mat.shape[0]
input = image_train_data_mat.shape[1]
output = 2
hid = 5
w1 = np.random.randn(input, hid)
w2 = np.random.randn(hid, output)
bias1 = np.zeros([hid, 1])
bias2 = np.zeros([output, 1])
rate1 = 0.25
rate2 = 0.25
temp1 = np.zeros([hid, 1])
net = temp1
temp2 = np.zeros([output, 1])
z = temp2

for i in range(100):
    for j in range(num):
        label = np.zeros([output, 1]) #(2 1)
        label[image_train_label_mat[j]][0] = 1
        temp1 = np.dot(image_train_data_mat[j][:], w1)[:, None] + bias1 #(5 1)
        net = 1 / (1 + np.exp(-temp1)) #(5 1)
        temp2 = (np.dot(net.transpose(), w2)).transpose() + bias2 #(2 1)
        z = 1 / (1 + np.exp(-temp2)) #(2 1)
        error = label - z #(2 1)
        deltaZ = error * z * (1-z) #(2 1)
        deltaNet = net * (1-net) * (np.dot(w2, deltaZ)) #(5 1)
        for k in range(output):
            w2[:, k][:, None] = w2[:, k][:, None] + rate2 * deltaZ[k] * net
        for k in range(hid):
            w1[:, k][:, None] = w1[:, k][:, None] + rate1 * deltaNet[k] * image_train_data_mat[j][:].transpose()[:, None]
        bias2 = bias2 + rate2 * deltaZ
        bias1 = bias1 + rate1 * deltaNet

test_num = image_test_data_mat.shape[0] # 25
count = 0
for i in range(test_num):
    temp1 = np.dot(image_test_data_mat[i][:], w1)[:, None] + bias1
    net = 1 / (1 + np.exp(-temp1))
    temp2 = (np.dot(net.transpose(), w2)).transpose() + bias2
    z = 1 / (1 + np.exp(-temp2))
    index = np.nanargmax(z)
    if index == image_test_label_mat[i]:
        count = count + 1

accuracy = count / test_num
print(accuracy)

########################################################################
# diabetis data # 0.69
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
diabetis_train_data_mat = excel_to_matrix(datafile + '\\diabetis_train_data.xlsx')
diabetis_train_label_mat = excel_to_matrix(datafile + '\\diabetis_train_label.xlsx')
diabetis_test_data_mat = excel_to_matrix(datafile + '\\diabetis_test_data.xlsx')
diabetis_test_label_mat = excel_to_matrix(datafile + '\\diabetis_test_label.xlsx')
N = len(diabetis_train_label_mat)
M = len(diabetis_test_label_mat)
diabetis_train_label_mat = diabetis_train_label_mat.reshape(N)
diabetis_test_label_mat = diabetis_test_label_mat.reshape(M)

diabetis_train_data_mat = diabetis_train_data_mat.astype(np.int)
diabetis_train_label_mat = diabetis_train_label_mat.astype(np.int)
diabetis_test_data_mat = diabetis_test_data_mat.astype(np.int)
diabetis_test_label_mat = diabetis_test_label_mat.astype(np.int)

diabetis_train_data_mat = diabetis_train_data_mat[0:50, :]
diabetis_train_label_mat = diabetis_train_label_mat[0:50]
diabetis_train_label_mat[diabetis_train_label_mat == -1] = 0

diabetis_test_data_mat = diabetis_test_data_mat[0:100, :]
diabetis_test_label_mat = diabetis_test_label_mat[0:100]
diabetis_test_label_mat[diabetis_test_label_mat == -1] = 0

N = len(diabetis_train_label_mat)
M = len(diabetis_test_label_mat)

np.random.seed(0)

num = diabetis_train_data_mat.shape[0]
input = diabetis_train_data_mat.shape[1]
output = 2
hid = 7
w1 = np.random.randn(input, hid)
w2 = np.random.randn(hid, output)
bias1 = np.zeros([hid, 1])
bias2 = np.zeros([output, 1])
rate1 = 0.25
rate2 = 0.25
temp1 = np.zeros([hid, 1])
net = temp1
temp2 = np.zeros([output, 1])
z = temp2

for i in range(100):
    for j in range(num):
        label = np.zeros([output, 1]) #(2 1)
        label[diabetis_train_label_mat[j]][0] = 1
        temp1 = np.dot(diabetis_train_data_mat[j][:], w1)[:, None] + bias1 #(5 1)
        net = 1 / (1 + np.exp(-temp1)) #(5 1)
        temp2 = (np.dot(net.transpose(), w2)).transpose() + bias2 #(2 1)
        z = 1 / (1 + np.exp(-temp2)) #(2 1)
        error = label - z #(2 1)
        deltaZ = error * z * (1-z) #(2 1)
        deltaNet = net * (1-net) * (np.dot(w2, deltaZ)) #(5 1)
        for k in range(output):
            w2[:, k][:, None] = w2[:, k][:, None] + rate2 * deltaZ[k] * net
        for k in range(hid):
            w1[:, k][:, None] = w1[:, k][:, None] + rate1 * deltaNet[k] * diabetis_train_data_mat[j][:].transpose()[:, None]
        bias2 = bias2 + rate2 * deltaZ
        bias1 = bias1 + rate1 * deltaNet

test_num = diabetis_test_data_mat.shape[0] # 25
count = 0
for i in range(test_num):
    temp1 = np.dot(diabetis_test_data_mat[i][:], w1)[:, None] + bias1
    net = 1 / (1 + np.exp(-temp1))
    temp2 = (np.dot(net.transpose(), w2)).transpose() + bias2
    z = 1 / (1 + np.exp(-temp2))
    index = np.nanargmax(z)
    if index == diabetis_test_label_mat[i]:
        count = count + 1

accuracy = count / test_num
print(accuracy)

############################################################
# thyroid data # 0.95
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
thyroid_train_data_mat = excel_to_matrix(datafile + '\\thyroid_train_data.xlsx')
thyroid_train_label_mat = excel_to_matrix(datafile + '\\thyroid_train_label.xlsx')
thyroid_test_data_mat = excel_to_matrix(datafile + '\\thyroid_test_data.xlsx')
thyroid_test_label_mat = excel_to_matrix(datafile + '\\thyroid_test_label.xlsx')
N = len(thyroid_train_label_mat)
M = len(thyroid_test_label_mat)
thyroid_train_label_mat = thyroid_train_label_mat.reshape(N)
thyroid_test_label_mat = thyroid_test_label_mat.reshape(M)

thyroid_train_data_mat = thyroid_train_data_mat.astype(np.int)
thyroid_train_label_mat = thyroid_train_label_mat.astype(np.int)
thyroid_test_data_mat = thyroid_test_data_mat.astype(np.int)
thyroid_test_label_mat = thyroid_test_label_mat.astype(np.int)

thyroid_train_data_mat = thyroid_train_data_mat[0:50, :]
thyroid_train_label_mat = thyroid_train_label_mat[0:50]
thyroid_train_label_mat[thyroid_train_label_mat == -1] = 0

thyroid_test_data_mat = thyroid_test_data_mat[0:100, :]
thyroid_test_label_mat = thyroid_test_label_mat[0:100]
thyroid_test_label_mat[thyroid_test_label_mat == -1] = 0

N = len(thyroid_train_label_mat)
M = len(thyroid_test_label_mat)

np.random.seed(0)

num = thyroid_train_data_mat.shape[0]
input = thyroid_train_data_mat.shape[1]
output = 2
hid = 14
w1 = np.random.randn(input, hid)
w2 = np.random.randn(hid, output)
bias1 = np.zeros([hid, 1])
bias2 = np.zeros([output, 1])
rate1 = 0.25
rate2 = 0.25
temp1 = np.zeros([hid, 1])
net = temp1
temp2 = np.zeros([output, 1])
z = temp2

for i in range(100):
    for j in range(num):
        label = np.zeros([output, 1]) #(2 1)
        label[thyroid_train_label_mat[j]][0] = 1
        temp1 = np.dot(thyroid_train_data_mat[j][:], w1)[:, None] + bias1 #(5 1)
        net = 1 / (1 + np.exp(-temp1)) #(5 1)
        temp2 = (np.dot(net.transpose(), w2)).transpose() + bias2 #(2 1)
        z = 1 / (1 + np.exp(-temp2)) #(2 1)
        error = label - z #(2 1)
        deltaZ = error * z * (1-z) #(2 1)
        deltaNet = net * (1-net) * (np.dot(w2, deltaZ)) #(5 1)
        for k in range(output):
            w2[:, k][:, None] = w2[:, k][:, None] + rate2 * deltaZ[k] * net
        for k in range(hid):
            w1[:, k][:, None] = w1[:, k][:, None] + rate1 * deltaNet[k] * thyroid_train_data_mat[j][:].transpose()[:, None]
        bias2 = bias2 + rate2 * deltaZ
        bias1 = bias1 + rate1 * deltaNet

test_num = thyroid_test_data_mat.shape[0] # 25
count = 0
for i in range(test_num):
    temp1 = np.dot(thyroid_test_data_mat[i][:], w1)[:, None] + bias1
    net = 1 / (1 + np.exp(-temp1))
    temp2 = (np.dot(net.transpose(), w2)).transpose() + bias2
    z = 1 / (1 + np.exp(-temp2))
    index = np.nanargmax(z)
    if index == thyroid_test_label_mat[i]:
        count = count + 1

accuracy = count / test_num
print(accuracy)