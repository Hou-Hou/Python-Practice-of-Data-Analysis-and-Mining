#!/usr/bin/env python3 
# -*- coding:utf-8 -*-
# Author: Hou-Hou

'''
2. 数据预处理：数据清洗 + 属性构造
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 第一步：数据清洗
# 一般情况下默认磁盘容量是定值，所以需要剔除磁盘容量重复的数据
def step1():
    data = pd.read_excel('../data/discdata.xls')
    data.drop_duplicates(data.columns[:-1], inplace=True)
    data.to_excel('../my_data/dataCleaned.xlsx')


# 第二步：属性构造
# 思路：由于每台服务器上的这三个属性值一直不变：NAME、TARGET_ID、ENTITY，将这三个属性值合并
# 方法一：
def step2_1():
    data = pd.read_excel('../my_data/dataCleaned.xlsx')

    df = data[data['TARGET_ID']==184].copy()
    df_group = df.groupby(data['COLLECTTIME'])

    # 定义属性变化函数
    def attr_trans(x):
        print('x:\n',x)
        result = pd.Series(index=['SYS_NAME','CWXT_DB:184:C:\\','CWXT_DB:184:D:\\','COLLECTTIME'])
        result['SYS_NAME'] = x['SYS_NAME'].iloc[0]
        result['CWXT_DB:184:C:\\'] = x['VALUE'].iloc[0]
        result['CWXT_DB:184:D:\\'] = x['VALUE'].iloc[1]
        result['COLLECTTIME'] = x['COLLECTTIME'].iloc[0]
        # print('result:\n', result)

        return result

    data_processed = df_group.apply(attr_trans)
    data_processed.to_excel('../my_data/attrsConstruction.xlsx', index=False)

def step2_2():
    data = pd.read_excel('../my_data/dataCleaned.xlsx')

    df = data[data['TARGET_ID']==184].copy()
    df_group = df.groupby(data['COLLECTTIME']).size()   # 每组的size

    indexpre = df_group.index
    data_processed = pd.DataFrame([], index=indexpre, columns=['SYS_NAME', 'CWXT_DB:184:C:\\', 'CWXT_DB:184:D:\\'])
    data_processed['SYS_NAME'] = '财务管理系统'
    data_processed['CWXT_DB:184:C:\\'] = df['VALUE'][df['ENTITY']=='C:\\'].values
    data_processed['CWXT_DB:184:D:\\'] = df['VALUE'][df['ENTITY']=='D:\\'].values
    print(data_processed)
    data_processed.to_excel('../my_data/attrsConstruction.xlsx')

step2_1()
