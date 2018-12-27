#!/usr/bin/env python3 
# -*- coding:utf-8 -*-
# Author: Hou-Hou

'''
数据变换----一次完整用水事件的划分模型
方法：判断流量大于0的记录的时间差是否超过阈值，是，则是两次事件，否，则是一次事件
'''

# 第 1 步：探索划分一次完整用水事件的时间间隔阈值
import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

plt.rc('figure',figsize=(9,6))
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

data = pd.read_excel('../my_data/data_guiyue.xlsx')
data = data[data['水流量'] > 0]
data['用水停顿时间间隔'] = (data['发生时间'].diff() / np.timedelta64(1, 'm'))
data.fillna(0, inplace=True)  # 填补’用水停顿时间间隔’的第一个空值

# print(data.head())

# 第 2 步：数据探索， 查看各数值列的最大最小和空值情况
data_explore = data.describe().T
data_explore['null'] = len(data) - data_explore['count']
data_explore = data_explore[['min', 'max', 'null']]
print(data_explore)

'''
                  min      max      null
水流量            8.0    77.00      0.0
用水停顿时间间隔  0.0   2093.37     0.0
'''

# 第 3 步：离散化与面元划分，绘制‘用水停顿时间间隔’的频率分布直方图
# 将时间间隔列数据划分为0~0.1，0.1~0.2，0.2~0.3....13以上，由数据描述可知，
# data[u'用水停顿时间间隔']的最大值约为2094，因此取上限2100

Ti = list(data['用水停顿时间间隔'])  # 将要面元化的数据转成一维的列表
timegaplist = [0.0,0.1,0.2,0.3,0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,2100]  # 确定划分区间
cats = pd.cut(Ti, timegaplist, right=False).value_counts().sort_index()  # <class 'pandas.core.series.Series'>
dx = pd.DataFrame(cats, columns=['num'])
dx['fn'] = dx['num'] / sum(dx['num'])
dx['cumfn'] = dx['num'].cumsum() / sum(dx['num'])
# dx['f'] = dx['fn'].applymap(lambda x: '%.2f%%' % (x*100))  # AttributeError: 'Series' object has no attribute 'applymap'
dx[['f']] = dx[['fn']].applymap(lambda x: '%.2f%%' % (x*100))

# 绘制直方图
fig, ax = plt.subplots(1,1)
dx['fn'].plot(kind='bar')
plt.xlabel('时间间隔（分钟）')
plt.ylabel('频率/组距')
p = 1.0 * dx['fn'].cumsum() / dx['fn'].sum()  # 相当于 p=list(dx['cumfn'])
dx['cumfn'].plot(color='r', secondary_y=True, style='-o', linewidth=2)
plt.annotate(format((p[4]), '.4%'), xy=(7, p[4]), xytext=(7*0.9, p[4]*0.95),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.2"))   # 97%处添加注释
plt.ylabel('累计频率')
plt.title('用水停顿时间间隔频率分布直方图')
plt.grid(axis='y', linestyle='--')

# fig.autofmt_xdate() #自动根据标签长度进行旋转
for label in ax.xaxis.get_ticklabels():   #此语句完成功能同上,但是可以自定义旋转角度
       label.set_rotation(60)

plt.savefig('../my_data/Water-pause-times.jpg')
plt.show()

'''
结果分析：停顿时间间隔为0~0.3分钟的频率很高，根据经验可以判定其为一次用水时间中的停顿；
          停顿时间间隔为6~13分钟的频率较低，分析其为两次用水事件之间的停顿
'''

# 第 4 步：确定一次用水停顿阈值后，开始划分一次完整事件
threshold = pd.Timedelta(minutes=4)  # 根据2-2确定最优阈值
d = data['发生时间'].diff() > threshold
data['事件编号'] = d.cumsum() + 1  # 第一条记录为0，其为第一个事件，所以+1

data.to_excel('../my_data/dataExchange_divideEvent.xlsx')

