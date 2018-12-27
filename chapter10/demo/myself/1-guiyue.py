#!/usr/bin/env python3 
# -*- coding:utf-8 -*-
# Author: Hou-Hou

'''
1 数据抽取
2 数据探索分析
  通过频率分布直方图分析用户用水停顿时间间隔的规律性--->探究划分一次完整用水事件的时间间隔阈值

3 数据预处理
（1）数据规约 data_guiyue.py
-*- utf-8 -*-
规约掉"热水器编号"、"有无水流"、"节能模式"三个属性
注意：
书中提到：规约掉热水器"开关机状态"=="关"且”水流量”==0的数据，说明热水器不处于工作状态，数据记录可以规约掉。
但由后文知，此条件不能进行规约
因为，"开关机状态"=="关"且”水流量”==0可能是一次用水中的停顿部分，删掉后则无法准确计算关于停顿的数据。
'''

import pandas as pd
import numpy as np

data = pd.read_excel('../data/original_data.xls', encoding='gbk')
data.drop(['热水器编号', '有无水流', '节能模式'], inplace=True, axis=1)
# data1 = data[(data[u'开关机状态']==u'开')|(data[u'水流量']!=0)]  # 此条件不适用
data['发生时间'] = pd.to_datetime(data['发生时间'], format='%Y%m%d%H%M%S')

data.to_excel('../my_data/data_guiyue.xlsx')
