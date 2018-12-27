#!/usr/bin/env python3 
# -*- coding:utf-8 -*-
# Author: Hou-Hou

import pandas as pd
import numpy as np

# 数据预处理
def datalossanalysis():
    cleanedfile = '../tmp/data_cleaned.csv'
    decresedfile = '../tmp/datalossanalysis.xlsx'

    data = pd.read_csv(cleanedfile)
    data = data[data['FLIGHT_COUNT']>6]

    data2 = pd.DataFrame()
    data2['FFP_TIER'] = data['FFP_TIER']
    data2['FLIGHT_COUNT'] = data['FLIGHT_COUNT']
    data2['AVG_INTERVAL'] = data['AVG_INTERVAL']
    data2['avg_discount'] = data['avg_discount']
    data2['EXCHANGE_COUNT'] = data['EXCHANGE_COUNT']
    data2['Eli_Add_Point_Sum'] = data['Eli_Add_Point_Sum']
    data2['SUM_per_KM'] = (data['SUM_YR_1'] + data['SUM_YR_1']) / data['SEG_KM_SUM']
    data2['Points_per_KM'] = data['Points_Sum'] / data['SEG_KM_SUM']
    data2['MEMBER_TYPE'] = data['L1Y_Flight_Count'] / data['P1Y_Flight_Count']
    data2['MEMBER_TYPE'] = pd.cut(data2['MEMBER_TYPE'], [0, 0.5, 0.9, 5], labels=[0, 1, 2])
    # data2[np.isinf(data2['MEMBER_TYPE'])] = 2
    data2['MEMBER_TYPE'].fillna(2, inplace=True)   #  L1Y_Flight_Count / P1Y_Flight_Count = inf 时，数据为空值，用类别2替换
    print(data2['MEMBER_TYPE'])

    data2.to_excel(decresedfile)

def bzh():
    from sklearn.preprocessing import StandardScaler
    d1 = pd.read_excel('../tmp/datalossanalysis.xlsx')
    d1 = d1.drop(['FLIGHT_COUNT'], axis=1)
    data = StandardScaler().fit_transform(d1)
    # d2 = (d1 - d1.min()) / (d1.max()-d1.min())
    data = pd.DataFrame(data, columns=d1.columns)
    data.to_excel('../tmp/lossanalysis_bzh.xlsx')

# bzh()

# 模型构建
from sklearn.cluster import KMeans #导入K均值聚类算法

if __name__=='__main__':
    inputfile = '../tmp/lossanalysis_bzh.xlsx' #待聚类的数据文件
    outputfile = '../tmp/lossanalysis_results1.xlsx'
    outputfile2 = '../tmp/lossanalysis_results2.xlsx'

    k = 3
    data = pd.read_excel(inputfile)
    #调用k-means算法，进行聚类分析
    kmodel = KMeans(n_clusters=k, n_jobs=4) #n_jobs是并行数，一般等于CPU数较好
    kmodel.fit(data) #训练模型

    center = pd.DataFrame(kmodel.cluster_centers_)  # 查看聚类中心
    label = pd.Series(kmodel.labels_).value_counts()   # 查看各样本对应的类别
    r = pd.concat([center, label], axis=1)
    r.columns = list(data.columns) + ['类别数目']
    r.to_excel(outputfile)

    r2 = pd.concat([data, pd.Series(kmodel.labels_, index=data.index)], axis=1)
    r2.columns = list(data.columns) + ['聚类类别']
    r2.to_excel(outputfile2)



