#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @Author  :  Hou

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
# from sklearn.metrics import confusion_matrix
from sklearn import metrics
from plot_c_m import *

# 由于此单元的中间数据处理原书中有问题，所以此处采用书中给的训练数据，和测试数据，旨在测试模型在此数据上的运行
inputfile1 = '../data/train_neural_network_data.xls' # 训练数据
inputfile2 = '../data/test_neural_network_data.xls' # 测试数据
testoutputfile = '../my_data/test_output_data.xls' #测试数据模型输出文件

data_train = pd.read_excel(inputfile1) # 读入训练数据
data_test = pd.read_excel(inputfile2) # 读入测试数据


x_train = data_train.iloc[:,5:17].values # 训练样本特征
y_train = data_train.iloc[:,4].values # 训练样本标签列
x_test = data_test.iloc[:,5:17].values # 测试样本特征
y_test = data_test.iloc[:,4].values # 训练样本标签列

from sklearn import svm
model = svm.SVC()
model.fit(x_train, y_train)
import pickle
# pickle.dump(model, open('../tmp/svm.model', 'wb'))
#model = pickle.load(open('../tmp/svm.model', 'rb'))

predict_result_test = model.predict(x_test).reshape(len(data_test))

cm_train = metrics.confusion_matrix(y_train, model.predict(x_train))
cm_test = metrics.confusion_matrix(y_test, model.predict(x_test))

correctRate_train = (cm_train[1,1] + cm_train[0,0]) / cm_train.sum()
correctRate_test = (cm_test[1,1] + cm_test[0,0]) / cm_test.sum()

print('训练集正确率：',correctRate_train)
print('测试集正确率：',correctRate_test)
# 测试集正确率： 0.8571428571428571

fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(121)
plot_confusion_matrix(cm_train, classes=range(2), title='Confusion matrix on train-set')  #显示混淆矩阵可视化结果 看训练结果正确率
ax=fig.add_subplot(122)
plot_confusion_matrix(cm_test, classes=range(2), title='Confusion matrix on test-set') #显示混淆矩阵可视化结果 看训练结果正确率
plt.show()

r = DataFrame(predict_result_test, columns = [u'预测结果']) # 给出预测类别测试集
# predict_rate = DataFrame(model.predict(x_test), columns = [u'预测正确率']) # 给出预测类别测试集
res = pd.concat([data_test.iloc[:,:5],r], axis=1)#测试集
res.to_excel(testoutputfile)