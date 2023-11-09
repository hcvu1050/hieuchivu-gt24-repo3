"""
last update: 2023-09-25
data preprocessing modules specifically for Model1
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from ...constants import INPUT_GROUP_LAYER_NAME, INPUT_TECHNIQUE_LAYER_NAME
from ...constants import GROUP_ID_NAME, TECHNIQUE_ID_NAME, LABEL_NAME
from ...constants import RANDOM_STATE
from ...constants import *
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
SOURCE_FILENAME = 'PREPROCESSED.txt'
SOURCE_LIST_FILE = os.path.join (SOURCE_PATH, SOURCE_FILENAME)
PROCESS_RUNNING_MSG = "--runing {}".format(__name__)

def get_data (data_type = 'csv'):
    """
    Read the csv files (filenames stored in PREPROCESSED.txt) and return as DataFrames
    """
    print ('Collecting data')
    with open (SOURCE_LIST_FILE, 'r') as file:
        csv_file_names = file.read().splitlines()
    
    group_features_df = [name for name in csv_file_names if 'X_group' in name]
    technique_features_df = [name for name in csv_file_names if 'X_technique' in name]
    labels_df = [name for name in csv_file_names if 'y' in name]
        
    if data_type == 'csv':
        group_features_df = pd.read_csv (os.path.join (SOURCE_PATH, group_features_df[0]))
        technique_features_df = pd.read_csv (os.path.join (SOURCE_PATH, technique_features_df[0]))
        labels_df = pd.read_csv (os.path.join (SOURCE_PATH, labels_df[0]))
    elif data_type == 'pkl':
        group_features_df = pd.read_pickle (os.path.join (SOURCE_PATH, group_features_df[0]))
        technique_features_df = pd.read_pickle (os.path.join (SOURCE_PATH, technique_features_df[0]))
        labels_df = pd.read_pickle (os.path.join (SOURCE_PATH, labels_df[0]))
        
    return group_features_df, technique_features_df, labels_df

def split_by_group(df: pd.DataFrame, ratio: float):
    """Splits data by Group randomly so that: data of a Group ONLY belong to a set. 
    Returns two tables, the first table's size is indicated by `ratio`.
    If ratio  == 0: return None, df
    Args:
        `labels_df` (pd.DataFrame): a table that contains the column `GROUP_ID_NAME`
        `ratio` : the ratio for the first output table.
        `save_as_csv` (list, optional): save the split table as csv files by the names indiacted.
    """
    group_IDs = df[GROUP_ID_NAME].unique()
    if ratio == 0:
        return pd.DataFrame(), df
    group_1_IDs, group_2_IDs = train_test_split (group_IDs, 
                                         train_size = ratio, 
                                         random_state= RANDOM_STATE, 
                                         shuffle= True)
    df_1 = df[df[GROUP_ID_NAME].isin(group_1_IDs)]
    df_2 = df[df[GROUP_ID_NAME].isin(group_2_IDs)]
    
    return df_1, df_2
    
def label_resample (df: pd.DataFrame, sampling_strategy: dict):
    """
    Resampling the labels by either oversampling or undersampling or both
    Balance the labels stored in `LABEL_NAME` column of a DataFrame 'df'
    Currently this only works if 'df' has EXACTLY 3 columns: `GROUP_ID_NAME`, `TECHNIQUE_ID_NAME`, and `LABEL_NAME`
    """
    X = df[[GROUP_ID_NAME, TECHNIQUE_ID_NAME]]
    y = df[[LABEL_NAME]]
    x_resampled,y_resampled = X, y
    if sampling_strategy['oversample'] is not None:
        over_sampler = RandomOverSampler(random_state= RANDOM_STATE, sampling_strategy= sampling_strategy['oversample'] )
        x_resampled,y_resampled = over_sampler.fit_resample(x_resampled,y_resampled)
    if sampling_strategy['undersample'] is not None:
        under_sampler = RandomUnderSampler(random_state= RANDOM_STATE, sampling_strategy= sampling_strategy['undersample'])
        x_resampled,y_resampled = under_sampler.fit_resample(x_resampled,y_resampled)
    res_df = pd.concat ([x_resampled,y_resampled], axis =1)
    return res_df


def align_input_to_labels(feature_df: pd.DataFrame, object: str, label_df: pd.DataFrame):
    """ Aligns the instances in feature_df so that they match with their corresponding labels in target_df.
    The main purpose of the function is for the input features (group features and technique features)\n
    Args:
        feature_df (pd.DataFrame): DataFrame containing input's features \n
        target_df (pd.DataFrame): DataFrame that feature_df will be aligned to \n
        (`from_set` and `object` arguments are only used for naming the output tables)
        from_set (str): ('train'|'cv'|'test') describes which set does target_df come from (train, cross-validation, or test)\n
        object (str): "group" or "technique"\n
        
    """
    id_name = '' # for merging

    if object == 'group':
        id_name = GROUP_ID_NAME
    elif object == 'technique':
        id_name = TECHNIQUE_ID_NAME
    df_aligned = pd.merge(left = feature_df, right= label_df, on = id_name, how = 'right')
    
    # remove unecessary columns after merging
    if object == 'group':
        df_aligned.drop (columns= [TECHNIQUE_ID_NAME, LABEL_NAME], inplace= True)
    elif object == 'technique':
        df_aligned.drop (columns= [GROUP_ID_NAME, LABEL_NAME], inplace= True)
        
    return df_aligned


def build_dataset (X_group_df: pd.DataFrame, X_technique_df:pd.DataFrame, y_df:pd.DataFrame, ragged_input: bool):
    """
    From the (aligned) feature tables and label table, build and return a tensorflow dataset.
    args `ragged`: if the tensor in each example is ragged (varies in length)
    """
    # removing the ID columns because they are not used for training
    X_group_df = X_group_df.drop (columns= GROUP_ID_NAME)
    X_technique_df = X_technique_df.drop (columns= TECHNIQUE_ID_NAME)
    y_df = y_df[LABEL_NAME]
    
    if ragged_input:
        X_group_tf = tf.ragged.constant(X_group_df.values, dtype = tf.string)
        X_technique_tf = tf.ragged.constant(X_technique_df.values, dtype = tf.string)
    else:
        X_group_tf = tf.convert_to_tensor(X_group_df.values, dtype = tf.float32)
        X_technique_tf = tf.convert_to_tensor(X_technique_df.values, dtype = tf.float32)
    
    y_tf = tf.convert_to_tensor(y_df.values, dtype = tf.float32)
    
    res_dataset = tf.data.Dataset.from_tensor_slices ((
        {
            INPUT_GROUP_LAYER_NAME: X_group_tf, 
            INPUT_TECHNIQUE_LAYER_NAME: X_technique_tf
            },
        y_tf))
    return res_dataset

GROUP_FEATURE_NAME_LIST = [INPUT_GROUP_SOFTWARE_ID]
TECHNIQUE_FEATURE_NAME_LIST = [
    INPUT_TECHNIQUE_DATA_SOURCES,
    INPUT_TECHNIQUE_DEFENSES_BYPASSED,
    INPUT_TECHNIQUE_DETECTION_NAME,
    INPUT_TECHNIQUE_MITIGATION_ID,
    INPUT_TECHNIQUE_PERMISSIONS_REQUIRED,
    INPUT_TECHNIQUE_PLATFORMS,
    INPUT_TECHNIQUE_SOFTWARE_ID,
    INPUT_TECHNIQUE_TACTICS
]

def build_dataset_2 (X_group_df: pd.DataFrame, X_technique_df:pd.DataFrame, y_df:pd.DataFrame, ragged_input: bool):
    """
    From the (aligned) feature tables and label table, build and return a tensorflow dataset.
    Difference from the previous version: each feature has its own key-value pair in the input dictionary
    NOTE 2023-10-29: the function is assuming ALL inputs are ragged vector (from the dataframe)
    """
    X_group_df = X_group_df.drop (columns= GROUP_ID_NAME)
    X_technique_df = X_technique_df.drop (columns= TECHNIQUE_ID_NAME)
    y_df = y_df[LABEL_NAME]
    
    input_dict = dict()
    if ragged_input: 
        for feature_name in GROUP_FEATURE_NAME_LIST:
            feature_tf = tf.ragged.constant (X_group_df[feature_name].values, dtype= tf.string)
            input_dict [feature_name] = feature_tf
        for feature_name in TECHNIQUE_FEATURE_NAME_LIST:
            feature_tf = tf.ragged.constant (X_technique_df[feature_name].values, dtype= tf.string)
            input_dict [feature_name] = feature_tf
    
    y_tf = tf.convert_to_tensor(y_df.values, dtype = tf.float32)
    
    res_dataset = tf.data.Dataset.from_tensor_slices ((input_dict,y_tf))    
    return res_dataset
    

def save_dataset (dataset, target_folder, file_name):
    file_path = os.path.join (target_folder, file_name)
    tf.data.Dataset.save (dataset, file_path)
    print ('Dataset saved to', file_path)
