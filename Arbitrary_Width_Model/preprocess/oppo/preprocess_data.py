__author__ = 'fjordonez'

import os
import zipfile
import argparse
import numpy as np
import _pickle as cp

from io import BytesIO
from pandas import Series
NB_SENSOR_CHANNELS = 113


def select_subject(dataset_name, test):

    if dataset_name == 'Opportunity':
      
        if test == 'challenge':
            train_runs = ['S1-Drill', 'S1-ADL1', 'S1-ADL2', 'S1-ADL3', 'S1-ADL4', 'S2-Drill', 'S2-ADL1',
                          'S2-ADL2', 'S2-ADL3', 'S3-Drill', 'S3-ADL1',
                          'S3-ADL2', 'S3-ADL3']
            val_runs = ['S1-ADL5',]
            test_runs = ['S2-ADL4', 'S2-ADL5', 'S3-ADL3', 'S3-ADL4']

            train_files = ['OpportunityUCIDataset/dataset/{}.dat'.format(run) for run in train_runs]
            val_files = ['OpportunityUCIDataset/dataset/{}.dat'.format(run) for run in val_runs]
            test_files = ['OpportunityUCIDataset/dataset/{}.dat'.format(run) for run in test_runs]

        else: 

            train = ['1','2','3','4']
            runs = ['Drill','ADL1','ADL2','ADL3','ADL4','ADL5']
            val_runs = ['ADL5']

            test_files = ['OpportunityUCIDataset/dataset/S{}-{}.dat'.format(test,run) for run in runs]

            train.remove(test)
            runs.remove(val_runs[0])

            train_files = ['OpportunityUCIDataset/dataset/S{}-{}.dat'.format(sub,run) for sub in train for run in runs]
            val_files = ['OpportunityUCIDataset/dataset/S{}-{}.dat'.format(sub,run) for sub in train for run in val_runs]

    
    return train_files, test_files, val_files


NORM_MAX_THRESHOLDS = [3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       250,    25,     200,    5000,   5000,   5000,   5000,   5000,   5000,
                       10000,  10000,  10000,  10000,  10000,  10000,  250,    250,    25,
                       200,    5000,   5000,   5000,   5000,   5000,   5000,   
                       10000,  10000,
                       10000,  10000,  10000,  10000 ,  250 ]

NORM_MIN_THRESHOLDS = [-3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -250,   -100,   -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                       -10000, -10000, -10000, -10000, -10000, -10000, -250,   -250,   -100,
                       -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                        -10000, -10000,
                       -10000, -10000, -10000, -10000 ,-250 ]


def select_columns_opp(data):
    features_delete = np.arange(46, 50)  # [46 47 48 49]
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])  # [46 47 48 49 59 60 61 62]
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])
    return np.delete(data, features_delete, 1)


def normalize(data, max_list, min_list):
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        data[:, i] = (data[:, i]-min_list[i])/diffs[i]
    
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data


def divide_x_y(data, label):

    if label in ['locomotion','gestures']:
        # print(data[0])
        data_x = data[:, 1:NB_SENSOR_CHANNELS+1]                
        if label == 'locomotion':
            data_y = data[:, NB_SENSOR_CHANNELS+1]  # Locomotion label
        elif label == 'gestures':
            data_y = data[:, NB_SENSOR_CHANNELS+2]  # Gestures label
    
    elif label == -1:

        data_x = data[:,1:-1]
        data_y = data[:,-1]

    else:
        raise RuntimeError("Invalid label: '%s'" % label)

    return data_x, data_y


def adjust_idx_labels(data_y, label):

    if label == 'locomotion':  # Labels for locomotion are adjusted
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif label == 'gestures':  # Labels for gestures are adjusted  #将label切换为1-17
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17
    return data_y


def check_data(data_set):
    
    print('Checking dataset {0}'.format(data_set))
    data_dir, data_file = os.path.split(data_set)
    # When a directory is not provided, check if dataset is in the data directory
    if data_dir == "" and not os.path.isfile(data_set):
        new_path = os.path.join(os.path.split(__file__)[0], "data", data_set)
        if os.path.isfile(new_path) or data_file == 'OpportunityUCIDataset.zip':
            data_set = new_path

    # When dataset not found, try to download it from UCI repository
    if (not os.path.isfile(data_set)) and data_file == 'OpportunityUCIDataset.zip':
       print("File not found")
       exit

    return data_dir


def process_dataset_file(dataset_name, data, label):

    if dataset_name == 'Opportunity':
        # Select correct columns
        # print(data.shape)
        data = select_columns_opp(data)
        # print(data.shape)

        # Colums are segmentd into features and labels
        data_x, data_y = divide_x_y(data, label)
        data_y = adjust_idx_labels(data_y, label)
        data_y = data_y.astype(int)

        # Perform linear interpolation
        data_x = np.array([Series(i).interpolate() for i in data_x.T]).T  # (54966, 113)

        # Remaining missing data are converted to zero
        data_x[np.isnan(data_x)] = 0

        # All sensor channels are normalized
        data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)

    return data_x, data_y


def generate_data(dataset_name, dataset, test_sub, label):
    
    data_dir = check_data(dataset)

    train_files, test_files, val_files = select_subject(dataset_name, test_sub)

    zf = zipfile.ZipFile(dataset)
    print('Processing dataset files ...')

    try:
        os.mkdir('data')
    except FileExistsError: # Remove data if already there.
        for file in os.scandir('data'):
            if 'data' in file.name:
                os.remove(file.path)

    # Generate training files
    print('Generating training files')
    for i,filename in enumerate(train_files):
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))  # (54966, 250)
            print('... file {} -> train_data_{}'.format(filename,i))
            x, y = process_dataset_file(dataset_name, data, label)
            with open('preprocess/oppo/data/train_data_{}'.format(i),'wb') as f:
                cp.dump((x,y),f)
        except KeyError:
            print('ERROR: Did not find {} in zip file'.format(filename))

    # Generate validation files
    print('Generating validation files')
    for i,filename in enumerate(val_files):
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print('... file {} -> val_data_{}'.format(filename,i))
            x, y = process_dataset_file(dataset_name, data, label)
            with open('preprocess/oppo/data/val_data_{}'.format(i),'wb') as f:
                cp.dump((x,y),f)
        except KeyError:
            print('ERROR: Did not find {} in zip file'.format(filename))

    # Generate testing files
    print('Generating testing files')
    for i,filename in enumerate(test_files):
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print('... file {} -> test_data_{}'.format(filename,i))
            x, y = process_dataset_file(dataset_name, data, label)
            with open('preprocess/oppo/data/test_data_{}'.format(i),'wb') as f:
                cp.dump((x,y),f)
        except KeyError:
            print('ERROR: Did not find {} in zip file'.format(filename))


def find_data(name):
    dataset_dir = 'preprocess/oppo/data/raw/'
    dataset_names = {'Opportunity':'OpportunityUCIDataset.zip'}
    dataset = dataset_dir + dataset_names[name]

    return dataset


def get_args():
    '''This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(
        description='Preprocess OPPORTUNITY dataset')
    # Add arguments
    # parser.add_argument(
    #     '-i', '--input', type=str, help='OPPORTUNITY zip file', required=True)
    parser.add_argument(
        '-s','--subject', type=str, help='Subject to leave out for testing', default="challenge")
    parser.add_argument(
        '-t', '--task', type=str.lower, help='Type of activities to be recognized (for opportunity)', default="gestures", choices = ["gestures", "locomotion"], required=False)
    parser.add_argument(
        '-d', '--dataset', type=str, help='Name of dataset.', default="Opportunity", choices = "Opportunity", required=False)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    dataset = args.dataset
    subject = args.subject
    label = args.task
    # Return all variable values
    return dataset, subject, label

"""
if __name__ == '__main__':
    dataset_name, sub, l = get_args()
    dataset = find_data(dataset_name)
    generate_data(dataset_name, dataset, sub, l)
"""