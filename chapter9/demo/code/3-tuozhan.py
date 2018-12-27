#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @Author  :  Hou

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot_c_m
import pickle


inputfile = '../../thinking/thinking.xls'
outputfile1 = '../../thinking/cm_train.xls'
outputfile2 = '../../thinking/cm_test.xls'
data = pd.read_excel(inputfile, encoding='gbk')
data_copy = data.copy()
data = data.values

from numpy.random import shuffle
shuffle(data)
data_train = data[:int(0.8*len(data)), :]
data_test = data[int(0.8*len(data)):, :]

x_train = data_train[:, :-1]*30
y_train = data_train[:, -1]
x_test = data_test[:, :-1]*30
y_test = data_test[:, -1]

# 支持向量机
# from sklearn import svm
# model = svm.SVC()
# model.fit(x_train, y_train)
# # pickle.dump(model, open('../../thinking/svm.model', 'wb'))
# model = pickle.load(open('../../thinking/svm.model', 'rb'))

# 决策树
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='gini')
model.fit(x_train, y_train)
pickle.dump(model, open('../../thinking/gini.model', 'wb'))
# model = pickle.load(open('../../thinking/entropy.model', 'rb'))

from sklearn import metrics
cm_train = metrics.confusion_matrix(y_train, model.predict(x_train))
cm_test = metrics.confusion_matrix(y_test, model.predict(x_test))

print('train：', model.score(x_train, y_train))
print('test：',model.score(x_test, y_test))

# 绘制混淆矩阵
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(121)
plot_c_m.plot_confusion_matrix(cm_train,classes=range(len(cm_train)),title='Confusion matrix on train-set')
ax = fig.add_subplot(122)
plot_c_m.plot_confusion_matrix(cm_test,classes=range(len(cm_test)),title='Confusion matrix on test-set')
plt.savefig('../../thinking/gini.jpg')
plt.show()

pd.DataFrame(cm_train, index=range(len(cm_train)), columns = range(len(cm_train))).to_excel(outputfile1)
pd.DataFrame(cm_test, index=range(len(cm_test)), columns = range(len(cm_test))).to_excel(outputfile2)

# 决策树可视化
from sklearn.tree import export_graphviz
with open('../../thinking/gini.dot', 'w') as f:
    # f = export_graphviz(model, feature_names=data_copy.columns, out_file=f)
    f = export_graphviz(model, out_file=f)

