import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy import stats

def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data
def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)
def segment_signal(data, window_size=200):
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    for (start, end) in windows(data["timestamp"], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if len(dataset["timestamp"][start:end]) == window_size:
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data["activity"][start:end])[0][0])
    return segments, labels

dataset = read_data('./preprocess/wisdm/WISDM_ar_v1.1_raw.txt')
dataset.dropna(axis=0, how='any', inplace=True)
dataset['z-axis'] = pd.to_numeric(dataset['z-axis'].str.replace(';', ''))  
