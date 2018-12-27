#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


inputfile = '../data/type.csv'
outputfile1 = '../tmp/cm_train.xls'
outputfile2 = '../tmp/cm_test.xls'
data = pd.read_csv(inputfile, encoding = 'gbk')
data = data.as_matrix()

from numpy.random import shuffle
shuffle(data)
data_train = data[:int(0.8*len(data)), :]
data_test = data[int(0.8*len(data)):, :]


x_train = data_train[:, 2:]*30
y_train = data_train[:, 0].astype(int)
x_test = data_test[:, 2:]*30
y_test = data_test[:, 0].astype(int)


from sklearn import svm
model = svm.SVC()
model.fit(x_train, y_train)
import pickle
pickle.dump(model, open('../tmp/svm.model', 'wb'))
#model = pickle.load(open('../tmp/svm.model', 'rb'))


from sklearn import metrics
cm_train = metrics.confusion_matrix(y_train, model.predict(x_train))
cm_test = metrics.confusion_matrix(y_test, model.predict(x_test))

pd.DataFrame(cm_train, index = range(1, 6), columns = range(1, 6)).to_excel(outputfile1)
pd.DataFrame(cm_test, index = range(1, 6), columns = range(1, 6)).to_excel(outputfile2)

# 绘制混淆矩阵
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(121)
plot_confusion_matrix(cm_train,classes=range(5),title='Confusion matrix on train-set')
ax=fig.add_subplot(122)
plot_confusion_matrix(cm_test,classes=range(5),title='Confusion matrix on test-set')
plt.show()


