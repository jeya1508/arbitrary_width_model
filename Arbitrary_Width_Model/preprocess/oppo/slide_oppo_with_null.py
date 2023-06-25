from preprocess.oppo.utils import *
import numpy as np
n_channels = 113 
window_size = 40 
stride = 15  


def load_data2array(name,len_seq,stride):
    Xs = np.empty(shape=[0, window_size, 113])
    ys = np.empty(shape=[0])
    path = "preprocess/oppo/data/{}_data_*".format(name)
    data = glob.glob(path)

    for file in data:
        X, y = load_dataset(file)  
        X, y = slide(X, y, len_seq, stride, save=False)
        print("X.shape", X.shape)
        print("y.shape", y.shape)
        Xs = np.vstack((Xs, X))
        print("Xs.shape", Xs.shape)
        ys = np.hstack((ys, y))
        print("ys.shape", ys.shape)
    return Xs, ys





