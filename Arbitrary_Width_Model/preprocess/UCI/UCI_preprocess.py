import numpy as np
import pandas as pd
from collections import Counter
import os
def load_x(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        with open(signal_type_path, "r") as f:
            uci_array=[np.array(serie, dtype=np.float32) for serie in [row.replace('  ', ' ').strip().split(' ') for row in f]]
            X_signals.append(uci_array)
    return np.transpose(X_signals, (1, 2, 0))
def load_y(y_path):
    with open(y_path, "r") as f:
        y = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in f]],dtype=np.int8)
    y = y.reshape(-1, )
    return y - 1
    
data_path = "./preprocess/UCI/UCI HAR Dataset/"
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]
x_train_paths_list = [data_path + "train/Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]
x_test_paths_list = [data_path + "test/Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES]
y_train_path = data_path + "train/y_train.txt"
y_test_path = data_path + "test/y_test.txt"

