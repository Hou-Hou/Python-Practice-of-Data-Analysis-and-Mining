#!/usr/bin/env python3 
# -*- coding:utf-8 -*-
# Author: Hou-Hou

# 模型构建 C盘
# 详解AIC方式定信息准则 ＋　ARMA

import pandas as pd
import numpy as np

data = pd.read_excel('../my_data/attrsConstruction.xlsx')
df = data.iloc[:len(data)-5]

# 第   1   步--C盘---------平稳性检测
#1)平稳性检测 ：判断是否平稳，若不平稳，对其进行差分处理直至平稳
# 方法：采用单位根检验（ADF）的方法或者时序图的方法（见数据探索模块）
# 注意：其他平稳性检验方法见steadyCheck.py文件

from statsmodels.tsa.stattools import adfuller as ADF
adf = ADF(df['CWXT_DB:184:C:\\'])
print('原始序列的ADF值为：', adf)
diff = 0
while adf[1] >= 0.05:
    diff += 1
    adf = ADF(df['CWXT_DB:184:C:\\'].diff(diff).dropna())
    print('原始序列经过%s阶差分后的ADF：' % diff, adf[1])

print('原始序列经过%s阶差分后归于平稳，p值=%s' % (diff, adf[1]))
df['CWXT_DB:184:C:\\_adf'] = df['CWXT_DB:184:C:\\'].diff(1)

# 第   2   步--C盘---------白噪声检验
# 目的：验证序列中有用信息是否已经被提取完毕，需要进行白噪声检验。若序列是白噪声序列，说明序列中有用信息已经被提取完，只剩随机扰动
# 方法：采用LB统计量的方法进行白噪声检验
# 若没有通过白噪声检验，则需要进行模型识别，识别其模型属于AR、MA还是ARMA。

from statsmodels.stats.diagnostic import acorr_ljungbox
[[lb], [p]] = acorr_ljungbox(df['CWXT_DB:184:C:\\'], lags=1)
if p < 0.05:
    print('原始序列为非白噪音序列， 对应的p值为：%s' % p)
else:
    print('原始序列为白噪音序列， 对应的p值为：%s' % p)

[[lb], [p]] = acorr_ljungbox(df['CWXT_DB:184:C:\\_adf'].dropna(), lags=1)
if p < 0.05:
    print('一阶差分序列为非白噪音序列， 对应的p值为：%s' % p)
else:
    print('一阶差分序列为白噪音序列， 对应的p值为：%s' % p)

# 第   3   步----------模型识别
# 方法：采用极大似然比方法进行模型的参数估计，估计各个参数的值。
# 然后针对各个不同模型，采用信息准则方法（有三种：BIC/AIC/HQ)对模型进行定阶，确定p,q参数，从而选择最优模型。
# 注意，进行此步时，index需要为时间序列类型
# 确定最佳p、d、q的值
data.index = pd.Series(data['COLLECTTIME'])
xtest_value = data['CWXT_DB:184:C:\\'][-5:]
data2 = data.iloc[:len(data)-5]
xdata2 = data2['CWXT_DB:184:C:\\']

# ARIMA（p,d,q）中,AR是自回归,p为自回归项数；MA为滑动平均,q为滑动平均项数,d为使之成为平稳序列所做的差分次数(阶数)，由前一步骤知d=1

# from statsmodels.tsa.arima_model import ARIMA   # 建立ARIMA（p,d，q）模型
from statsmodels.tsa.arima_model import ARMA      # 建立ARMA（p,q）模型

# 定阶
# 目前选择模型常用如下准则!!!!!
# 增加自由参数的数目提高了拟合的优良性，
# AIC/BIC/HQ鼓励数据拟合的优良性但是尽量避免出现过度拟合(Overfitting)的情况。所以优先考虑的模型应是AIC/BIC/HQ值最小的那一个
#* AIC=-2 ln(L) + 2 k 中文名字：赤池信息量 akaike information criterion (AIC)
# * BIC=-2 ln(L) + ln(n)*k 中文名字：贝叶斯信息量 bayesian information criterion (BIC)
# * HQ=-2 ln(L) + ln(ln(n))*k hannan-quinn criterion (HQ)

# AIC方式定信息准则　＋　ARMA --------！！！模型检验中也要对应修改！！！------------------------------
pmax = int(len(xdata2)/10)
qmax = int(len(xdata2)/10)

matrix = []
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
        try:
            # tmp.append(ARMA(xdata2, (p,q)).fit().aic)
            tmp.append(ARMA(xdata2, (p,q)).fit().bic)
            # tmp.append(ARMA(xdata2, (p,q)).fit().hq)
        except:
            tmp.append(None)
    matrix.append(tmp)

matrix = pd.DataFrame(matrix)
print('matrix---------\n', matrix)
print('matrix---------\n', matrix.stack())


'''
              0            1            2            3            4
0  1254.084030  1229.484140  1213.260357          NaN  1214.282270
1  1197.862668  1201.518205  1204.591406          NaN  1208.042667
2  1201.540913  1203.785729  1207.997558  1208.143231  1211.687458
3  1204.822657  1209.032756  1210.746067  1211.813088  1215.288872
4  1204.570230  1207.153552  1210.775594  1211.662444  1216.329316

最小的p值和q值为：p = 1, q = 0
'''

# 第   4   步--C盘---------模型检验
# 确定模型后，需要检验其残差序列是否是白噪声，若不是，说明，残差中还存在有用的信息，需要修改模型或者进一步提取。
# 若其残差不是白噪声，重新更换p,q的值，重新确定

while 1:
    p, q = matrix.stack().idxmin()
    print('最小的p值和q值为：p = %s, q = %s' % (p, q))

    lagnum = 12
    arma = ARMA(xdata2, (p,q)).fit()
    xdata_pred = arma.predict()
    pred_error = (xdata_pred - xdata2).dropna()
    print('pred_error:\n', pred_error)

    # 白噪音检验
    lbx, px = acorr_ljungbox(pred_error, lags=lagnum)
    print('pred_error的p值为:', px)
    h = (px < 0.05).sum()
    print('h=', h)
    if h > 0:
        print('模型ARMA(%s, %s)不符合白噪音检验' % (p, q))
        print('在AIC矩阵中去掉[%s,%s]组合，重新进行计算' % (p, q))
        matrix.iloc[p, q] = np.nan
        arimafail = arma
        continue
    else:
        # print(p,q)
        print('模型ARMA(%s,%s)符合白噪声检验' % (p,q))
        break

'''
pred_error的p值为: [0.85213186 0.87622837 0.5091596  0.17549072 0.23165608 0.32636535
 0.23061123 0.21730751 0.26456821 0.34228534 0.32163025 0.23001753]
h= 0
模型ARMA(1,0)符合白噪声检验
 '''

# 第   5   步--C盘---------模型预测
print('模型报告：summary():\n', arma.summary())
forecast_values, forecasts_standard_error, forecast_confidence_interval = arma.forecast(5)

pre_data = pd.DataFrame(xtest_value)
pre_data.insert(1, 'CWXT_DB:184:C:\\_predict', forecast_values)
pre_data.rename(columns={'CWXT_DB:184:C:\\' : '实际值', 'CWXT_DB:184:C:\\_predict': '预测值'}, inplace=True)
result_d = pre_data.applymap(lambda x: '%.2f' % x)
result_d.to_excel('../my_data/pedictdata_C_AIC_ARMA.xlsx')

# 第   5   步--D盘---------模型评价
# 为了评价时序预测模型效果的好坏，本章采用3个衡量模型预测精度的统计量指标：平均绝对误差、均方根误差、平均绝对百分误差
result = pd.read_excel('../my_data/pedictdata_C_AIC_ARMA.xlsx', index_col='COLLECTTIME')
result = result.applymap(lambda x: x/10**6)
print('模型结果：\n', result)

abs_ = (result['预测值'] - result['实际值']).abs()
mae_ = abs_.mean()
rmse_ = ((abs_**2).mean())**0.5
mape_ = (abs_/result['实际值']).mean()

print(u'平均绝对误差为：%.4f, \n均方根误差为：%.4f, \n平均绝对百分误差为：%.4f' % (mae_, rmse_, mape_))
errors = 1.5
print('误差阈值为%s' % errors)
if (mae_ < errors) & (rmse_ < errors) & (mape_ < errors):
    print('误差检验通过！')
else:
    print('误差检验不通过！')

'''
模型结果：
                    实际值        预测值
COLLECTTIME                      
2014-11-12   35.704313  35.581706
2014-11-13   35.704981  35.488223
2014-11-14   34.570385  35.405986
2014-11-15   34.673821  35.333641
2014-11-16   34.793245  35.270000
平均绝对误差为：0.4623, 
均方根误差为：0.5335, 
平均绝对百分误差为：0.0133
误差阈值为1.5
误差检验通过！
'''
