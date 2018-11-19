import numpy as np
import cPickle as pickle
from sklearn.model_selection import train_test_split
import os


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s


def one_hot(y_):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def load_motorway_dataset(data_path='data'):
    # Function to load the motorway dataset only 

    with open(os.path.join(data_path, 'motorway_dataset_window_64_proc_veh_DtA.pkl'), 'rb') as f:
        save = pickle.load(f)
        motorway_dataset = save['dataset']
        motorway_labels = save['labels']
        del save
        print('Motorway set', motorway_dataset.shape, motorway_labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(motorway_dataset, motorway_labels, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test


def load_secondary_dataset(data_path='data'):
    # Function to load the secondary dataset only 

    with open(os.path.join(data_path,'secondary_dataset_window_64_proc_veh_DtA.pkl'), 'rb') as f:
        save = pickle.load(f)
        secondary_dataset = save['dataset']
        secondary_labels = save['labels']
        del save
        print('Secondary set', secondary_dataset.shape, secondary_labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(secondary_dataset, secondary_labels, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test


def load_full_dataset(data_path='data'):
    # Function to load the full dataset (motorway+secondary roads)

    with open(os.path.join(data_path, 'motorway_dataset_window_64_proc_veh_DtA.pkl'), 'rb') as f:
        save = pickle.load(f)
        motorway_dataset = save['dataset']
        motorway_labels = save['labels']
        del save
        print('Motorway set', motorway_dataset.shape, motorway_labels.shape)

    with open(os.path.join(data_path,'secondary_dataset_window_64_proc_veh_DtA.pkl'), 'rb') as f:
        save = pickle.load(f)
        secondary_dataset = save['dataset']
        secondary_labels = save['labels']
        del save
        print('Secondary set', secondary_dataset.shape, secondary_labels.shape)

    dataset = np.concatenate((motorway_dataset,secondary_dataset), axis=0)
    labels = np.concatenate((motorway_labels,secondary_labels), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test