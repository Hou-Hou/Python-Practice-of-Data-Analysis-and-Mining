#!/usr/bin/env python3 
# -*- coding:utf-8 -*-
# Author: Hou-Hou

import pandas as pd
from statsmodels.tsa.arima_model import ARIMAResults

model = ARIMAResults.load('arima_BIC.pkl')
forecast_values, forecasts_standard_error, forecast_confidence_interval = model.forecast(10)

data_index = pd.date_range(start='20141112', end='20141121')
data = pd.DataFrame(index=data_index)
data.index.name = '时间'
data['磁盘使用量预测值'] = forecast_values
result_d = data.applymap(lambda x: '%.2f' % x)
print(result_d)

'''
               
时间        磁盘使用量预测值       
2014-11-12  35722538.09
2014-11-13  35757103.59
2014-11-14  35791669.08
2014-11-15  35826234.58
2014-11-16  35860800.07
2014-11-17  35895365.57
2014-11-18  35929931.06
2014-11-19  35964496.56
2014-11-20  35999062.05
2014-11-21  36033627.54
'''