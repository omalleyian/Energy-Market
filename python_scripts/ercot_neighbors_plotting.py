import sys
import matplotlib.pyplot as plt
import numpy as np
import csv
import ast
sys.path.append("/home/omalleyian/Documents/energy_market_project/scripts")
from ercot_data_interface import ercot_data_interface

raw = []
with open('neighbor_zeroes.csv', 'rb') as csvFile:
    reader = csv.reader(csvFile, dialect='excel')
    for row in reader:
        raw.append(row)

table = []
for row in raw:
    table.append([row[0], ast.literal_eval(row[1])])

possible = []
for group in table:
    over = False
    for i in group[1]:
        if i > 10000:
            over = True
    if not over:
        row = [group[0], group[1]]
        possible.append(row)

ercot = ercot_data_interface(password='Is79t5Is79t5')


center = possible[3][0]
nn = ercot.get_nearest_CRR_neighbors('SIL_SILAS_6')
for n in nn:
    plt.plot(ercot.query_prices(n, '2011-01-01', '2016-5-23'), label=n)
plt.legend()
plt.show()
