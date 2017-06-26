#-------------------------
# Libs
#-------------------------

# External libs
# %matplotlib qt         Fuck this line right here.
import pymysql.cursors
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
sys.path.append("/home/omalleyian/Documents/energy_market_project/scripts")
from ercot_data_interface import ercot_data_interface
from ARIMA import ARIMA
import csv

ercot = ercot_data_interface(password="Is79t5Is79t5")
nodes_crr = ercot.get_CRR_nodes()
nodes_all = ercot.all_nodes
nodes_source = ercot.get_sources_sinks()
df_2011 = ercot.query_prices(nodes_all[0], "2011-01-01","2011-12-31")
df_2012 = ercot.query_prices(nodes_all[0], "2012-01-01","2012-12-31")
matrix_2011 = df_2011.as_matrix()
matrix_2012 = df_2012.as_matrix()

arima = ARIMA(p = 2, d = 0, q = 0, seasonal = 24)
arima.fit(matrix_2011)
# arima.plot_predicted_vs_actual(matrix_2012)
# print arima.mae(matrix_2012)

#For focast of f=24
#wavenet
hours_2011 = []
hours_2012 = []
arima_models = []

for i in range(24):
    ind_2011 = np.arange(i,matrix_2011.shape[0],24)
    ind_2012 = np.arange(i,matrix_2012.shape[0],24)

    hours_2011.append(matrix_2011[ind_2011])
    hours_2012.append(matrix_2012[ind_2012])

    arima = ARIMA(p = 2, d = 0, q = 1, seasonal = 1)
    arima.fit(matrix_2011[ind_2011])
    arima_models.append(arima)

## Plot every prediction model on its own graph - BAD
# for i in range(24):
#     arima_models[i].plot_predicted_vs_actual(hours_2012[i])

## Plot each prediction model on the same graph, but unmerged - BAD
# predictions = []
# for i in range(24):
#     prediction, actual = arima_models[i].predict(hours_2012[i])
#     predictions.append(prediction)
#     plt.plot(prediction)
#
# plt.show()

## Merge each prediction model into the same list to be
## ploted against the actual values
merged_prediction = np.zeros(8617)
for i in range(24):
    prediction, actual = arima_models[i].predict(hours_2012[i])

    prediction = prediction.squeeze().tolist()
    for j in range(len(prediction)-1):
        front = [0] * i
        back = [0] * (23-i)
        index = j * 24
        prediction[index:index] = front
        prediction[index+i+1:index+i+1] = back

    if len(prediction) < 8617:
        prediction.extend(np.zeros(24))

    merged_prediction += prediction

    ## DEBUGGING - Output each prediction to a row of a csv file
    # with open("output.csv",'ab') as resutlFile:
    #     wr = csv.writer(resutlFile, dialect='excel')
    #     wr.writerow(prediction)

plt.plot(matrix_2012, linestyle='-', label='actual')
plt.plot(merged_prediction, linestyle=':', label='prediction', alpha=1.0)
plt.legend()
plt.show()
