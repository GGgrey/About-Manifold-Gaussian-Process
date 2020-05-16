#！/usr/bin/env python
#-*-coding:utf-8-*-
#user:xieyuhui

# coding: utf-8

import numpy as np
import math
from sklearn import datasets
from collections import Counter
import xlrd

infinity = float(-2 ** 31)

def sigmodFormatrix(Xb, thetas):
    params = - Xb.dot(thetas)
    r = np.zeros(params.shape[0])
    for i in range(len(r)):
        r[i] = 1 / (1 + math.exp(params[i]))
    return r


def sigmodFormatrix2(Xb, thetas):
    params = - Xb.dot(thetas)
    r = np.zeros(params.shape[0])
    for i in range(len(r)):
        r[i] = 1 / (1 + math.exp(params[i]))
        if r[i] >= 0.5:
            r[i] = 1
        else:
            r[i] = 0
    return r


def sigmod(Xi, thetas):
    params = - np.sum(Xi * thetas)
    r = 1 / (1 + math.exp(params))
    return r


class LinearLogsiticRegression(object):
    thetas = None
    m = 0

    def fit(self, X, y, alpha=0.01, accuracy=0.00001):
        self.thetas = np.full(X.shape[1] + 1, 0.5)
        self.m = X.shape[0]
        a = np.full((self.m, 1), 1)
        Xb = np.column_stack((a, X))
        dimension = X.shape[1] + 1
        count = 1
        while True:
            oldJ = self.costFunc(Xb, y)
            c = sigmodFormatrix(Xb, self.thetas) - y
            for j in range(dimension):
                self.thetas[j] = self.thetas[j] - alpha * np.sum(c * Xb[:, j])
            newJ = self.costFunc(Xb, y)
            if newJ == oldJ or math.fabs(newJ - oldJ) < accuracy:
                print("代价函数迭代到最小值，退出！")
                print("收敛到:", newJ)
                break
            print("迭代第", count, "次!")
            print("代价函数上一次的差:", (newJ - oldJ))
            count += 1

    def costFunc(self, Xb, y):
        sum = 0.0
        for i in range(self.m):
            yPre = sigmod(Xb[i,], self.thetas)
            # print("yPre:",yPre)
            if yPre == 1 or yPre == 0:
                return infinity
            sum += y[i] * math.log(yPre) + (1 - y[i]) * math.log(1 - yPre)
        return -1 / self.m * sum

    def predict(self, X):
        a = np.full((len(X), 1), 1)
        Xb = np.column_stack((a, X))
        return sigmodFormatrix2(Xb, self.thetas)

    def score(self, X_test, y_test):
        y_predict = myLogstic.predict(X_test)
        re = (y_test == y_predict)
        re1 = Counter(re)
        a = re1[True] / (re1[True] + re1[False])
        return a


from sklearn.model_selection import train_test_split

# Iris data #1
iris = datasets.load_iris()
X = iris['data']
y = iris['target']
X = X[y != 2]
y = y[y != 2]
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""
myLogstic = LinearLogsiticRegression()
myLogstic.fit(X_train, y_train)
y_predict = myLogstic.predict(X_test)
print("parameter:", myLogstic.thetas)

print("test data accuracy:", myLogstic.score(X_test, y_test))
print("train data accuracy:", myLogstic.score(X_train, y_train))
"""

from sklearn.linear_model import LogisticRegression

print("Iris data result:")
logr = LogisticRegression()
logr.fit(X_train, y_train)
print("Accuracy:", logr.score(X_test, y_test))

#########################
# image data #0.67

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

print("image data result:")
logr = LogisticRegression()
logr.fit(image_train_data_mat, image_train_label_mat)
print("Accuracy:", logr.score(image_test_data_mat, image_test_label_mat))

#####################################################
# diabetis data #0.71
datafile = u'E:\\PyCharm_projects\\Manifold Gaussian Process'
image_train_data_mat = excel_to_matrix(datafile + '\\diabetis_train_data.xlsx')
image_train_label_mat = excel_to_matrix(datafile + '\\diabetis_train_label.xlsx')
image_test_data_mat = excel_to_matrix(datafile + '\\diabetis_test_data.xlsx')
image_test_label_mat = excel_to_matrix(datafile + '\\diabetis_test_label.xlsx')
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

print("diabetis data result:")
logr = LogisticRegression()
logr.fit(image_train_data_mat, image_train_label_mat)
print("Accuracy:", logr.score(image_test_data_mat, image_test_label_mat))

################################################################################
# thyroid data # 0.87
datafile = u'E:\\PyCharm_projects\\Manifold Gaussian Process'
image_train_data_mat = excel_to_matrix(datafile + '\\thyroid_train_data.xlsx')
image_train_label_mat = excel_to_matrix(datafile + '\\thyroid_train_label.xlsx')
image_test_data_mat = excel_to_matrix(datafile + '\\thyroid_test_data.xlsx')
image_test_label_mat = excel_to_matrix(datafile + '\\thyroid_test_label.xlsx')
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

print("thyroid data result:")
logr = LogisticRegression()
logr.fit(image_train_data_mat, image_train_label_mat)
print("Accuracy:", logr.score(image_test_data_mat, image_test_label_mat))