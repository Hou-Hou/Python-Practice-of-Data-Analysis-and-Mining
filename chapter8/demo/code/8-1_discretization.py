#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Hou-Hou

'''
聚类离散化，最后的result的格式为：
      1           2           3           4
A     0    0.178698    0.257724    0.351843
An  240  356.000000  281.000000   53.000000
即(0, 0.178698]有240个，(0.178698, 0.257724]有356个，依此类推。
'''


# 1>  数据预处理

# 1数据清洗
# 2属性规约
# 3数据变换
# （1）属性构造
# （2）数据离散化

from __future__ import print_function
import pandas as pd
from sklearn.cluster import KMeans #导入K均值聚类算法

datafile = '../data/data.xls' #待聚类的数据文件
processedfile = '../tmp/data_processed.xls' #数据处理后文件
typelabel ={u'肝气郁结证型系数':'A', u'热毒蕴结证型系数':'B', u'冲任失调证型系数':'C', u'气血两虚证型系数':'D', u'脾胃虚弱证型系数':'E', u'肝肾阴虚证型系数':'F'}
k = 4 #需要进行的聚类类别数

#读取数据并进行聚类分析
data = pd.read_excel(datafile) #读取数据
keys = list(typelabel.keys())

result = pd.DataFrame()

if __name__ == '__main__': #判断是否主窗口运行，如果是将代码保存为.py后运行，则需要这句，如果直接复制到命令窗口运行，则不需要这句。
  for i in range(len(keys)):
    #调用k-means算法，进行聚类离散化
    print(u'正在进行“%s”的聚类...' % keys[i])
    kmodel = KMeans(n_clusters = k, n_jobs=4) #n_jobs是并行数，一般等于CPU数较好
    # print('data[[keys[i]]]:', keys[i])
    kmodel.fit(data[[keys[i]]].values) #训练模型
    
    r1 = pd.DataFrame(kmodel.cluster_centers_, columns = [typelabel[keys[i]]]) #聚类中心
    r2 = pd.Series(kmodel.labels_).value_counts() #分类统计
    r2 = pd.DataFrame(r2, columns = [typelabel[keys[i]]+'n']) #转为DataFrame，记录各个类别的数目
    r = pd.concat([r1, r2], axis=1).sort_values(typelabel[keys[i]]) #匹配聚类中心和类别数目
    r.index = [1, 2, 3, 4]
    
    r[typelabel[keys[i]]] = r[typelabel[keys[i]]].rolling(2).mean() #rolling_mean()用来计算相邻2列的均值，以此作为边界点。

    # print('r[typelabel[keys[i]]]:', r[typelabel[keys[i]]])
    r[typelabel[keys[i]]][1] = 0.0 #这两句代码将原来的聚类中心改为边界点。
    result = result.append(r.T)

  result = result.sort_index() #以Index排序，即以A,B,C,D,E,F顺序排
  result.to_excel(processedfile)

  # 2>划分原始数据中的类别

  # 将分类后数据进行处理
  data_cut = pd.DataFrame(columns= data.columns[:6])
  print('data_cut:', data_cut)
  types = ['A','B','C','D','E','F']
  num = ['1','2','3','4']
  for i in range(len(data_cut.columns)):
      value = list(data.iloc[:,i])
      # result[(2*i):(2*i+1)].values                 # [[0 0.17869759 0.25772406 0.35184318]]
      bins = list(result[(2*i):(2*i+1)].values[0])   # [0.0, 0.17869, 0.25772, 0.3518]
      bins.append(1)                                 # [0.0, 0.17869, 0.25772, 0.3518, 1]
      names = [str(x)+str(y) for x in types for y in num]

      # names = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4',
      #          'D1', 'D2', 'D3', 'D4', 'E1', 'E2', 'E3', 'E4', 'F1', 'F2', 'F3', 'F4']

      group_names = names[4*i:4*(i+1)]
      print('group_names：', group_names)
      cats = pd.cut(value, bins, labels=group_names, right=False)
      data_cut.iloc[:, i] = cats
  data_cut.to_excel('../tmp/apriori.xlsx')
  data_cut.head()