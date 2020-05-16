#！/usr/bin/env python
#-*-coding:utf-8-*-
#user:xieyuhui

import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
import xlrd

################################################################
# Iris data # 1
iris = datasets.load_iris()
X = iris['data']
y = iris['target']
X = X[y != 2]
y = y[y != 2]
X_train, X_test, y_train, y_test = train_test_split(X, y)
y_train[y_train == 0] = 0
y_test[y_test == 0] = 0
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

pred_y = clf.predict(X_test)
print("Accuracy:", clf.score(X_test, y_test))

##############################################################
# image data # 0.7
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

clf = SVC(kernel='rbf')
clf.fit(image_train_data_mat, image_train_label_mat)

pred_y = clf.predict(image_test_data_mat)
print("Accuracy:", clf.score(image_test_data_mat, image_test_label_mat))

####################################################
# diabetis data # 0.67
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

clf = SVC(kernel='rbf')
clf.fit(image_train_data_mat, image_train_label_mat)

pred_y = clf.predict(image_test_data_mat)
print("Accuracy:", clf.score(image_test_data_mat, image_test_label_mat))

#########################################################################
# thyroid data # 0.97
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

clf = SVC(kernel='rbf')
clf.fit(image_train_data_mat, image_train_label_mat)

pred_y = clf.predict(image_test_data_mat)
print("Accuracy:", clf.score(image_test_data_mat, image_test_label_mat))



