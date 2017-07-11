import sys
import csv
sys.path.append("/home/omalleyian/Documents/energy_market_project/scripts")
import numpy as np # noqa
from ercot_data_interface import ercot_data_interface # noqa


# Compares floating numbers
def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


# Find zeroes in the given list using is_close function
def find_zeroes(list):
    x = np.asarray(list).flatten()
    count = 0
    for e in x:
        if is_close(e, 0):
            count += 1
    return count


ercot = ercot_data_interface()
sources_sinks = ercot.get_sources_sinks()

# Finds all the nodes with at least 5 neighbors
# Saves nodes to list fiver
fiver = []
count = 0
for ss in sources_sinks:
    try:
        nn = ercot.get_nearest_CRR_neighbors(ss)
        if len(nn) >= 5:
            fiver.append(nn)

    except Exception as error:
        count += 1
        # print(repr(error))

# Queries database for prices of neighbor groups in fiver
# Outputs the number of zeroes in each node in each group to csv file
zeroes = []
index = 0
open('neighbor_zeroes.csv', 'w').close()
for g in fiver:
    group = []
    for n in g:
        q = ercot.query_prices(n, '2011-01-01', '2016-5-23').as_matrix()
        group.append(find_zeroes(q))
    zeroes.append(group)
    temp = [fiver[index][0], zeroes[index]]
    with open('neighbor_zeroes.csv', 'ab') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerow(temp)
    print str(index) + ' ' + str(temp)
    index += 1
