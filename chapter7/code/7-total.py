#!/usr/bin/env python3 
# -*- coding:utf-8 -*-
# Author: Hou-Hou

import pandas as pd
import numpy as np

# 数据质量分析
def explore():
    datafile = '../data/air_data.csv'  # 航空原始数据,第一行为属性标签
    resultfile = '../tmp/explore.xls'  # 数据探索结果表

    data = pd.read_csv(datafile, encoding='utf-8')  # 读取原始数据，指定UTF-8编码（需要用文本编辑器将数据装换为UTF-8编码）

    explore = data.describe(percentiles=[],
                            include='all').T  # 包括对数据的基本描述，percentiles参数是指定计算多少的分位数表（如1/4分位数、中位数等）；T是转置，转置后更方便查阅
    explore['null'] = len(data) - explore['count']  # describe()函数自动计算非空值数，需要手动计算空值数

    explore = explore[['null', 'max', 'min']]
    explore.columns = [u'空值数', u'最大值', u'最小值']  # 表头重命名
    '''这里只选取部分探索结果。
    describe()函数自动计算的字段有count（非空值数）、unique（唯一值数）、top（频数最高者）、freq（最高频数）、mean（平均值）、std（方差）、min（最小值）、50%（中位数）、max（最大值）'''

    explore.to_excel(resultfile)  # 导出结果

# 数据清洗
def datacleane():
    datafile = '../data/air_data.csv'  # 航空原始数据,第一行为属性标签
    cleanedfile = '../tmp/data_cleaned.csv'  # 数据清洗后保存的文件

    data = pd.read_csv(datafile, encoding='utf-8')  # 读取原始数据，指定UTF-8编码（需要用文本编辑器将数据装换为UTF-8编码）

    data = data[data['SUM_YR_1'].notnull() * data['SUM_YR_2'].notnull()]  # 票价非空值才保留

    # 只保留票价非零的，或者平均折扣率与总飞行公里数同时为0的记录。
    index1 = data['SUM_YR_1'] != 0
    index2 = data['SUM_YR_2'] != 0
    index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)  # 该规则是“与”
    data = data[index1 | index2 | index3]  # 该规则是“或”

    data.to_csv(cleanedfile)  # 导出结果

# 属性规约
def decresedata():
    cleanedfile = '../tmp/data_cleaned.csv'
    decresedfile = '../tmp/datadecrese.xlsx'

    data = pd.read_csv(cleanedfile)
    data1 = data[['LOAD_TIME','FFP_DATE','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]
    data1.to_excel(decresedfile, index=False)

# 数据变换
def transformdata():
    decresedfile = '../tmp/datadecrese.xlsx'
    lrfmcfile = '../tmp/datalrfmc.xlsx'
    summary_data = '../tmp/summary_data.xlsx'

    data = pd.read_excel(decresedfile)

    data['L1'] = pd.to_datetime(data['LOAD_TIME']) - pd.to_datetime(data['FFP_DATE'])
    # print('data[L1]：', data['L1'])
    data['L2'] = (data['L1'] / np.timedelta64(1, 'M')).round(2)
    data['LAST_TO_END'] = (data['LAST_TO_END']/30).round(2)
    data['avg_discount'] = data['avg_discount'].round(2)

    data.drop(['L1', 'LOAD_TIME', 'FFP_DATE'], axis=1, inplace=True)
    data.rename(columns={'L2':'L', 'LAST_TO_END':'R', 'FLIGHT_COUNT':'F', 'SEG_KM_SUM':'M', 'avg_discount':'C'}, inplace=True)
    data.to_excel(lrfmcfile, index=False)

    d = data.apply(lambda x: pd.Series([x.min(), x.max()], index=['min', 'max']))
    d.to_excel(summary_data)

# transformdata()

# 标准化
def bzh():
    from sklearn.preprocessing import StandardScaler
    d1 = pd.read_excel('../tmp/datalrfmc.xlsx')
    d2 = (d1 - d1.mean(axis=0)) / d1.std(axis=0)  # 等价于d2= StandardScaler().fit_transform(d1.values)
    # d2 = StandardScaler().fit_transform(d1.values)
    d1 = d2.iloc[:, [4, 0, 1, 2, 3]]
    d1.columns = ['Z' + i for i in d1.columns]  # 表头重命名
    d1.to_excel('../tmp/lrfmc_bzh.xlsx', index=False)

    # data = pd.read_excel('../tmp/datalrfmc.xlsx')
    # data = (data - data.mean(axis=0)) / data.std(axis=0)
    # data.columns = ['Z' + i for i in data.columns]
    # data.to_excel('../tmp/lrfmc_bzh.xlsx', index=False)

# bzh()



# 模型构建
from sklearn.cluster import KMeans #导入K均值聚类算法

if __name__=='__main__':
    inputfile = '../tmp/lrfmc_bzh.xlsx' #待聚类的数据文件
    outputfile = '../tmp/kmeansresults1.xlsx'
    outputfile2 = '../tmp/kmeansresults2.xlsx'

    k = 5
    data = pd.read_excel(inputfile)
    #调用k-means算法，进行聚类分析
    kmodel = KMeans(n_clusters=k, n_jobs=4, max_iter=200) #n_jobs是并行数，一般等于CPU数较好
    kmodel.fit(data) #训练模型

    center = pd.DataFrame(kmodel.cluster_centers_)  # 查看聚类中心
    label = pd.Series(kmodel.labels_).value_counts()   # 查看各样本对应的类别
    r = pd.concat([center, label], axis=1)
    r.columns = list(data.columns) + ['类别数目']
    r.to_excel(outputfile)

    r2 = pd.concat([data, pd.Series(kmodel.labels_, index=data.index)], axis=1)
    r2.columns = list(data.columns) + ['聚类类别']
    r2.to_excel(outputfile2)

    # 画雷达图
    subset = center.copy()
    subset.columns = data.columns
    subset = subset.round(3)
    data2 = subset.values

    from radar import drawRader
    title = 'RadarPicture'
    rgrids = [0.5, 1, 1.5, 2, 2.5]
    itemnames = ['ZL', 'ZR', 'ZF', 'ZM', 'ZC']
    labels = list('abcde')
    drawRader(itemnames=itemnames, data=data2, title=title, labels=labels, saveas='../tmp/radar.jpg', rgrids=rgrids)








