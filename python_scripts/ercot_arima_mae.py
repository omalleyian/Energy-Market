import os, sys
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
sys.path.append("/home/omalleyian/Documents/energy_market_project/scripts")
from ercot_data_interface import ercot_data_interface
sys.path.append("/home/omalleyian/Documents/Energy-Market/python_scripts")
import performance_metrics as perf
from ARIMA import ARIMA

# Get list of nodes from SQL database
ercot = ercot_data_interface(password="Is79t5Is79t5")
nodes_all = ercot.all_nodes

# Assign performance function to use
function = perf.mae

# Calculate performance on the first 5 nodes
for i in range(5):
    print str(function.__name__) + " Calculation on Node" + str(i) + ": " + str(nodes_all[i])
    # Load training and testing data from a node
    df_train = ercot.query_prices(nodes_all[i], "2011-01-01","2015-05-23")
    df_test = ercot.query_prices(nodes_all[i], "2015-05-23","2016-05-23")
    matrix_train = df_train.as_matrix()
    matrix_test = df_test.as_matrix()

    # Preform mean model calculations on data
    m = np.mean(matrix_train)
    m_matrix = [m] * len(matrix_test)
    print function(matrix_test, m_matrix)

    # Perform random walk model calculations on data
    walk = matrix_test[:-1]
    walk_test = matrix_test[1:]
    print function(walk_test, walk)

    # ARIMA(2,0,1,1)
    arima = ARIMA(p = 2, d = 0, q = 1, seasonal = 1)
    arima.fit(matrix_train)
    actual, forcast = arima.predict(matrix_test)
    print function(actual, forcast)

    # ARIMA(2,0,2,1)
    arima = ARIMA(p = 2, d = 0, q = 2, seasonal = 1)
    arima.fit(matrix_train)
    actual, forcast = arima.predict(matrix_test)
    print function(actual, forcast)

    # ARIMA(2,0,3,1)
    arima = ARIMA(p = 2, d = 0, q = 3, seasonal = 1)
    arima.fit(matrix_train)
    actual, forcast = arima.predict(matrix_test)
    print function(actual, forcast)

    # ARIMA(2,0,1,24)
    arima = ARIMA(p = 2, d = 0, q = 1, seasonal = 24)
    arima.fit(matrix_train)
    actual, forcast = arima.predict(matrix_test)
    print function(actual, forcast)

    # ARIMA(2,0,2,24)
    arima = ARIMA(p = 2, d = 0, q = 2, seasonal = 24)
    arima.fit(matrix_train)
    actual, forcast = arima.predict(matrix_test)
    print function(actual, forcast)

    # ARIMA(2,0,3,24)
    arima = ARIMA(p = 2, d = 0, q = 3, seasonal = 24)
    arima.fit(matrix_train)
    actual, forcast = arima.predict(matrix_test)
    print function(actual, forcast)

    print "\n"
