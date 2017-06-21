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
arima.plot_predicted_vs_actual(matrix_2012)
print arima.mae(matrix_2012)

#For focast of f=5
#wavenet
f = 5
x0 = df_2011[np.arange(0,matrix_2011.shape[0],f)]
arima0 = ARIMA(p = 2, d = 0, q = 0, seasonal = 1)
arima0.fit(x0)

x1 = df_2011[np.arange(1,matrix_2011.shape[0],f)]
arima1 = ARIMA(p = 2, d = 0, q = 0, seasonal = 1)
arima1.fit(x1)

x2 = df_2011[np.arange(2,matrix_2011.shape[0],f)]
arima2 = ARIMA(p = 2, d = 0, q = 0, seasonal = 1)
arima2.fit(x2)

x3 = df_2011[np.arange(3,matrix_2011.shape[0],f)]
arima3 = ARIMA(p = 2, d = 0, q = 0, seasonal = 1)
arima3.fit(x3)

x4 = df_2011[np.arange(4,matrix_2011.shape[0],f)]
arima4 = ARIMA(p = 2, d = 0, q = 0, seasonal = 1)
arima.fit(x4)
