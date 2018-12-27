#!/usr/bin/env python3 
# -*- coding:utf-8 -*-
# Author: Hou-Hou

'''
1. 数据探索分析：  时序图---数据平稳性分析
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = pd.read_excel('../data/discdata.xls')

def matplot(d, name):
    plt.rc('figure', figsize=(9,7))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1,1)
    ax.plot(d['COLLECTTIME'], d['VALUE'], 'ro-')
    ax.set(xlabel='日期', ylabel='磁盘使用大小')
    # fig.autofmt_xdate()
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")

    ax.set_title('时序图----%s盘已使用空间' % name)
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(1,32), interval=10))
    # ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=range(1,32), interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    # ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.subplots_adjust(bottom=0.13, top=0.95)
    plt.savefig('../my_data/%s.jpg' % name)
    plt.show()

# 时序图
d1 = data[(data['ENTITY'] == 'C:\\') & (data['TARGET_ID'] == 184)]
d2 = data[(data['ENTITY'] == 'D:\\') & (data['TARGET_ID'] == 184)]
matplot(d1, 'C')
matplot(d2, 'D')

# 自相关图
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(d1[ 'VALUE']).show()

# ADF检验
from statsmodels.tsa.stattools import adfuller as ADF
adf = ADF(data['VALUE'])
print('序列的ADF检验结果为:', adf)

'''结果
时序图：通过图中可以发现，磁盘的使用情况都不具有周期性，表现出缓慢性增长，呈现趋势性。因此，可以初步确认数据是非平稳的

序列的ADF检验结果为: (-0.4163121490204017, 0.907339400297141, 15, 172, 
{'1%': -3.468952197801766, '5%': -2.878495056473015, '10%': -2.57580913601947}, 5089.183875737658)

p = 0.907339400297141 > 0.05  所以，接受原假设---非平稳序列
'''




