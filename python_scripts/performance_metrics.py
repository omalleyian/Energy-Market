import numpy as np

def mape(actual, forcast):
    x = (np.asarray(actual)).flatten()
    y = (np.asarray(forcast)).flatten()
    return np.mean(np.square(np.abs((x - y)) / x))

def mae(actual, forcast):
    x = (np.asarray(actual)).flatten()
    y = (np.asarray(forcast)).flatten()
    return np.mean(np.abs((x - y)))

def mase(actual, forcast):
    x = (np.asarray(actual)).flatten()
    y = (np.asarray(forcast)).flatten()
    return np.mean(np.abs(x - y) / np.mean(np.abs(x[1:]-x[:-1])))

def hit_rate(actual, forcast):
    x = (np.asarray(actual)).flatten()
    y = (np.asarray(forcast)).flatten()
    f_t = (x[1:] - x[:-1]) * (y[1:] - y[:-1])
    # k is the subset of f_t where k[i] = f_t[i] iff f_t[i] > 0
    k = [i for i in f_t if i > 0]
    return np.mean(np.abs(k))

def hit_rate2(actual, forcast):
    x = (np.asarray(actual)).flatten()
    y = (np.asarray(forcast)).flatten()
    f_t = (x[1:] - x[:-1]) * (y[1:] - y[:-1])
    # k is the subset of y where k[i] = y[i] iff f_t[i] > 0
    k = [y[i+1] for i in range(f_t.shape[0]) if f_t[i] > 0]
    return np.mean(np.abs(k))
