#!/usr/bin/env python3 
# -*- coding:utf-8 -*-
# Author: Hou-Hou

'''
数据变换----用水事件阈值寻优模型
'''

import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

# 第 1 步：确定阈值的变化与划分得到的事件个数关系
def analyse():
    data = pd.read_excel('../my_data/dataExchange_divideEvent.xlsx')
    data.drop(['事件编号'], inplace=True, axis=1)
    data.to_excel('../my_data/thresholdOptimization.xlsx')

    timedeltalist  = np.arange(2.25, 8, 0.25)
    counts = []
    for i in range(len(timedeltalist )):
        threshold = pd.Timedelta(minutes=timedeltalist[i])
        d = data['发生时间'].diff() > threshold
        data['事件编号'] = d.cumsum() + 1
        temp = data['事件编号'].max()
        counts.append(temp)

    coun = pd.Series(counts, index=timedeltalist)

    # 画频率分布直方图
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']= False
    plt.rc('figure', figsize=(8,6))
    np.set_printoptions(precision=4)  #作用：确定浮点数字、数组、和numpy对象的显示形式。
    fig, ax = plt.subplots(1,1)
    fig.set(alpha=0.2)#设置图标透明度
    coun.plot(style='-rs')
    ax.locator_params('x',nbins=int(len(coun)/2)+1)   # 调整坐标轴的刻度, 通过nbins设置坐标轴一共平均分为几份
    plt.xlabel('阈值（分钟）')
    plt.ylabel('事件个数')
    plt.grid(axis='y', linestyle='--')
    plt.savefig('../my_data/threshold_numofCase.jpg')
    plt.show()

'''
由上图可知，图像趋势平缓说明用户的停顿习惯趋于稳定，所以取该段时间开始作为阈值，
既不会将短的用水时间合并，也不会将长的用水时间拆开
因此，最后选取一次用水时间间隔阈值为4分钟
利用阈值的斜率指标来作为某点的斜率指标
'''


# 第 2 步：阈值优化
#****************************
#@2  目标：确定阈值的变化与划分得到的事件个数关系
#    方法：通过图像中斜率指标
#****************************

# 当存在阈值的斜率指标 k<KS :
#     取阈值最小的点A（可能存在多个阈值的斜率指标小于1）的横坐标x作为用水事件划分的阈值（该值是经过实验数据验证的专家阈值）
# 当不存在阈值的斜率指标 k<KS：
#     找所有阈值中“斜率指标最小”的阈值t1：
#     若：该阈值t1对应的斜率指标小于KS2：
#         则取该阈值作为用水事件划分的阈值
#     若：该阈值t1对应的斜率指标不小于KS2
#         则阈值取默认值——4分钟
# 备注：
# KS是评价斜率指标用的专家阈值1
# KS是评价斜率指标用的专家阈值2

data = pd.read_excel('../my_data/thresholdOptimization.xlsx')
n=4  # 使用以后四个点的平均斜率
KS = 1  # 专家阈值1
KS2 = 5  # 专家阈值2

def event_num(ts):
    d = data['发生时间'].diff() > ts
    return d.sum() + 1

dt = [pd.Timedelta(minutes=i) for i in np.arange(1, 9, 0.25)]
h = pd.DataFrame(dt, columns=['阈值'])
h['事件数'] = h['阈值'].apply(event_num)
h['斜率'] = h['事件数'].diff() / 0.25
h['斜率指标偏移前'] = h['斜率'].abs().rolling(n).mean()
h['斜率指标'] = np.nan
h['斜率指标'][:-4] = h['斜率指标偏移前'][4:]

mink = h['斜率指标'][h['斜率指标'] < KS]  # 斜率指标小于1的值的集合
mink1 = h['斜率指标'][h['斜率指标'] < KS2]  # 斜率指标小于5的值的集合

if list(mink):  # 斜率指标值小于1不为空时，即，存在斜率指标值小于1时
    minky = [h[u'阈值'][i] for i in mink.index]  # 取“阈值最小”的点A所对应的间隔时间作为ts
    ts = min(minky)  # 取最小时间为ts
elif list(mink1):  # 当不存在斜率指标值小于1时,找所有阈值中“斜率指标最小”的阈值
    t1 = h['阈值'][h['斜率指标'].idxmin()]  # “斜率指标最小”的阈值t1
    # ts = h[u'阈值'][h[u'斜率指标偏移前'].idxmin() - n] #等价于前一行作用（*****）
    # 备注：用idxmin返回最小值的Index，由于rolling_mean自动计算的是前n个斜率的绝对值的平均，所以结果要平移-n，得到偏移后的各个值的斜率指标，注意：最后四个值没有斜率指标因为找不出在它以后的四个更长的值
    if h['斜率指标'].min() < 5:
        ts = t1  # 当该阈值的斜率指标小于5，则取该阈值作为用水事件划分的阈值
    else:
        ts = pd.Timedelta(minutes=4)# 当该阈值的斜率指标不小于5，则阈值取默认值——4分钟

tm = ts/np.timedelta64(1, 'm')

print("当前时间最优时间间隔为%s分钟" % tm)




