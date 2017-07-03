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
sys.path.append("/home/omalleyian/Documents/Energy-Market/python_scripts")
import performance_metrics as perf
import csv

ercot = ercot_data_interface(password="Is79t5Is79t5")
nodes_all = ercot.all_nodes
df_2011 = ercot.query_prices(nodes_all[0], "2011-01-01","2011-12-31")
df_2012 = ercot.query_prices(nodes_all[0], "2012-01-01","2012-12-31")
matrix_2011 = df_2011.as_matrix()
matrix_2012 = df_2012.as_matrix()
#
# arima = ARIMA(p = 2, d = 0, q = 0, seasonal = 24)
# arima.fit(matrix_2011)
# arima.plot_predicted_vs_actual(matrix_2012)
# print arima.mae(matrix_2012)

# Arima Forcasting with variable horizon

def arima_forcast(train, test, horizon, p, d, q, seasonal):
    # First split data into subsets offset by 24, and fit ARIMA models to these subsets.
    hours_train = []
    hours_test = []
    arima_models = []
    loss = p + q + seasonal - 1

    for i in range(horizon):
        ind_train = np.arange(i,train.shape[0],horizon)
        ind_test = np.arange(i,test.shape[0],horizon)

        hours_train.append(train[ind_train])
        hours_test.append(test[ind_test])

        arima = ARIMA(p, d, q, seasonal)
        arima.fit(train[ind_train])
        arima_models.append(arima)

    # Make predictions with the ARIMA models
    # Merge the predictions into a single array to be ploted against actual data
    merged_prediction = np.zeros(horizon * (test.shape[0] / horizon - loss))
    merged_actual = np.zeros(horizon * (test.shape[0] / horizon - loss))

    for i in range(len(arima_models)):
        prediction, actual = arima_models[i].predict(hours_test[i])
        prediction = prediction.squeeze().tolist()
        actual = actual.squeeze().tolist()

        if len(prediction) < (len(test) / horizon) - loss:
            prediction.extend(np.zeros((len(test) / horizon - loss) - len(prediction)))
            actual.extend(np.zeros((len(test) / horizon - loss) - len(actual)))

        for j in range(len(prediction)):
            front = [0] * i
            back = [0] * (horizon-1-i)
            index = j * horizon

            prediction[index:index] = front
            actual[index:index] = front

            prediction[index+i+1:index+i+1] = back
            actual[index+i+1:index+i+1] = back

        merged_prediction += prediction
        merged_actual += actual

        ## DEBUGGING - Output each prediction to a row of a csv file
        # with open("output.csv",'ab') as resutlFile:
        #     wr = csv.writer(resutlFile, dialect='excel')
        #     wr.writerow(prediction)

    return merged_prediction, merged_actual

merged_prediction, merged_actual = \
    arima_forcast(matrix_2011, matrix_2012, 24, 2, 0, 1, 1)
plt.plot(merged_actual, label='actual')
plt.plot(merged_prediction, label='prediction', alpha=0.5)
plt.legend()
plt.show()

print "Merged MAE: " + str(perf.mae(merged_actual, merged_prediction))
