"""
last update: 2023-09-18
Preprocessing for model1\n
0. REQUIRED: interim data is collected from `data/interim`
run script `data_preprocess.py` first
1. Get the necessary files from data/interim and read as DataFrames
2. Group the DataFrames into train set, cv set, and test set
3. From each set, build a tensorflow Dataset (consists of both inputs and labels)
4. Save the Datasets to data/processed
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from ..constants import ROOT_FOLDER, INPUT_GROUP_LAYER_NAME, INPUT_TECHNIQUE_LAYER_NAME, TRAIN_DATASET_FILENAME, CV_DATASET_FILENAME, TEST_DATASET_FILENAME

SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
SOURCE_FILENAME = 'FINAL.txt'
SOURCE_LIST_FILE = os.path.join (SOURCE_PATH, SOURCE_FILENAME)

TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/processed')

PROCESS_RUNNING_MSG = "--runing {}".format(__name__)
### MAIN FUNCTION ###
def model_preprocess(train_set_split: float= None):
    print (PROCESS_RUNNING_MSG)
    # get the data set as DataFrames
    train_df, cv_df, test_df = _get_data()
    
    if train_set_split is None:
        train_dataset = _build_dataset(df_set = train_df)
        _save_dataset (dataset = train_dataset, file_name= TRAIN_DATASET_FILENAME)
    
    else: 
        train_dataset, train_cv_dataset = _build_dataset (df_set = train_df, train_set_split= train_set_split)
        _save_dataset (dataset = train_dataset, file_name= TRAIN_DATASET_FILENAME)
        _save_dataset (dataset = train_cv_dataset, file_name = 'train_cv_dataset')

    
    cv_dataset = _build_dataset(cv_df)
    test_dataset = _build_dataset(test_df)
    
    _save_dataset (dataset = cv_dataset, file_name= CV_DATASET_FILENAME)
    _save_dataset (dataset = test_dataset, file_name= TEST_DATASET_FILENAME)
    

def _get_data ():
    """ Get the necessary files from data/interim\n
    The list of files to get is stored in `SOURCE_LIST_FILE`('data/interim/FINAL.txt')\n
    Returns 3 `dict`s that stores the data train set, cv set, and test set as DataFrame 
    """
    #1 get files
    print ('Collecting data')
    with open (SOURCE_LIST_FILE, 'r') as file:
        csv_file_names = file.read().splitlines()
    
    train_files = [file_name for file_name in csv_file_names if 'train' in file_name]
    cv_files = [file_name for file_name in csv_file_names if 'cv' in file_name]
    test_files = [file_name for file_name in csv_file_names if 'test' in file_name]
    
    #2 read as DataFrame
    train_set = _get_Xy_dfs (train_files)
    cv_set = _get_Xy_dfs (cv_files)
    test_set = _get_Xy_dfs (test_files)
    return train_set, cv_set, test_set

def _get_Xy_dfs (file_list: list) -> dict:
    """
    From the list of file in file_lists, find each file that store either input or output. \n
    Read the inputs and output as DataFrame\n
    The list of files belongs to either train, cv, or test set\n
    Returns a dictionary mapping the DataFrames as inputs or output\n
    """
    X_group_file_name =     [file_name for file_name in file_list if 'X_group' in file_name][0]
    X_technique_file_name = [file_name for file_name in file_list if 'X_technique' in file_name][0]
    y_file_name =           [file_name for file_name in file_list if '_y' in file_name][0]
    print (X_group_file_name)
    print (X_technique_file_name)
    print (y_file_name)

    
    X_group_df = pd.read_csv (os.path.join (SOURCE_PATH, X_group_file_name))
    X_technique_df = pd.read_csv (os.path.join (SOURCE_PATH, X_technique_file_name))
    y_df = pd.read_csv (os.path.join (SOURCE_PATH, y_file_name))
        
    return {
        'X_group' : X_group_df,
        'X_technique': X_technique_df,
        'y': y_df
    }

def _build_dataset(df_set, train_set_split: float = None):
    """
    From a set(train, cv, test) stored in a `dict` containing X and y values:\n
    Create and return a tensorflow Dataset
    """
    X_group = df_set['X_group'].drop(columns = 'group_ID')
    X_technique = df_set['X_technique'].drop(columns = 'technique_ID')
    y = df_set['y']
    
    
    X_group_tf = tf.convert_to_tensor(X_group.values, dtype = tf.float32)
    X_technique_tf = tf.convert_to_tensor(X_technique.values, dtype = tf.float32)
    y_tf = tf.convert_to_tensor(y.values, dtype = tf.float32)
    res_dataset = tf.data.Dataset.from_tensor_slices ((
        {
            INPUT_GROUP_LAYER_NAME: X_group_tf, 
            INPUT_TECHNIQUE_LAYER_NAME: X_technique_tf
            },
        y_tf))
    if train_set_split is None:
        return res_dataset
    else: 
        res_dataset_size = res_dataset.cardinality().numpy()
        res_dataset = res_dataset.shuffle (buffer_size= res_dataset_size, seed=13)
        train_size = int (res_dataset_size * train_set_split)
        
        train_dataset = res_dataset.take (train_size)
        train_cv_dataset = res_dataset.skip (train_size)
        return train_dataset, train_cv_dataset


def _save_dataset (dataset, file_name):
    file_path = os.path.join (TARGET_PATH, file_name)
    tf.data.Dataset.save (dataset, file_path)
    print ('Dataset saved to', file_path)
