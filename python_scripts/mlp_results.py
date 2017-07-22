################################################################
# This script looks for a file 'test_nodes.txt' for input.
# It creates a folder in the parent directory for its output.
# In this folder will be a csv named after, and containing
# data for, each node. The first row in each file describes
# the data that follows.
################################################################

# Imports
import os
import sys
sys.path.append("/home/omalleyian/Documents/energy_market_project/scripts")
from ercot_data_interface import ercot_data_interface
from ARIMA import ARIMA
from MLP import MLP
import errno
import csv
import numpy as np

# Connect to ercot database
ercot = ercot_data_interface(password="Is79t5Is79t5")

# Read nodes chosen for testing from file
raw = []
try:
    with open('test_nodes.txt', 'rb') as txtFile:
        raw = [line.strip() for line in txtFile]
except IOError:
    print "File 'test_nodes.txt' was not found."

# Initialize consolidated ouput matrix
out_all = []

# Loop over all the test nodes
for node in raw:

    # Get data from ercot database
    nn = ercot.get_nearest_CRR_neighbors(node)
    train, test = ercot.get_train_test(nn, include_seasonal_vectors=False)

    # Find center node from train and test set
    center_node_train = np.expand_dims(train[:, 0], 1)
    center_node_test = np.expand_dims(test[:, 0], 1)

    # Create output file and directory, erases previous output files
    filename = os.path.join(os.pardir,
                            "results/mlp_results/" + str(nn[0]) + ".csv")
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:                   # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    open(filename, 'w').close()

    # Initialize node ouput matrix
    out_node = [[nn[0], len(nn)],
                ["log_diff"], ["MAE"], ["MASE"], ["HITS"], ['\n']]

    # Setup dynamic parameters for MLP object
    seeds = np.random.randint(0, 100000, 10)
    params = [False, True]

    # For each set of parameters, calculate predictions 10 times
    print "\t\t\t\t\t" + nn[0]
    for k in params:
        mae_list = []
        mase_list = []
        hits_list = []

        for i in range(10):
            mlp = MLP(random_seed=seeds[i],
                      log_difference=k)

            mlp.train(center_node_train, look_back=48)
            predicted, actual = mlp.predict(center_node_test)
            mae, mase, hits = mlp.print_statistics(predicted, actual)

            mae_list.append(mae)
            mase_list.append(mase)
            hits_list.append(hits)

        # Add results to node output matrix
        out_node[1].append(k)
        out_node[2].append(np.mean(mae_list) + np.std(mae_list))
        out_node[3].append(np.mean(mase_list) + np.std(mase_list))
        out_node[4].append(np.mean(hits_list) + np.std(hits_list))

    # Append node output to consolidated output
    for row in out_node:
        out_all.append(row)

    # Send node output matrix to csv file
    for row in out_node:
        with open(filename, 'ab') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerow(row)

# Output consolidated results to csv
filename = os.path.join(os.pardir,
                        "results/mlp_results/all_results.csv")
open(filename, 'w').close()
for row in out_all:
    with open(filename, "ab") as resultFile:
        wr = csv.writer(resultFile, dialect="excel")
        wr.writerow(row)
