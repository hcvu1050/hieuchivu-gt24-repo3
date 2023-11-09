"""
last update: 2023-09-18\n
Load the Datasets from data/processed and return those Datasets
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from ...constants import TRAIN_DATASET_FILENAME, TRAIN_CV_DATASET_FILENAME, CV_DATASET_FILENAME, TEST_DATASET_FILENAME
from ...constants import INPUT_GROUP_LAYER_NAME, INPUT_TECHNIQUE_LAYER_NAME
from ...constants import RANDOM_STATE

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/processed/model1')

def _single_load (file_name):
    print ("loading\t", file_name, '-----', end='')
    file_path = os.path.join (SOURCE_PATH, file_name) 
    print (file_path)
    dataset = tf.data.Dataset.load (file_path)
    print ('loaded')
    return dataset

def load_train_datasets (empty_train_cv: bool = False, sample_train: float = None, return_feature_info = True ):
    train_dataset = _single_load (TRAIN_DATASET_FILENAME)
    
    if not empty_train_cv: 
        train_cv_dataset = _single_load (TRAIN_CV_DATASET_FILENAME)
    
    if sample_train is not None:
        num_samples = int(len(train_dataset) * sample_train)
        train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset), seed=RANDOM_STATE).take(num_samples)
        
    print ('train_dataset: {} examples'.format(len(train_dataset)))
    if not empty_train_cv:
        print ('train_cv_dataset: {} examples'.format(len(train_cv_dataset)))
    
    if return_feature_info:
        feature_info = {
            'group_feature_size' : train_dataset.element_spec[0][INPUT_GROUP_LAYER_NAME].shape[0],
            'technique_feature_size' : train_dataset.element_spec[0][INPUT_TECHNIQUE_LAYER_NAME].shape[0]
        }
        if not empty_train_cv:
            return train_dataset, train_cv_dataset, feature_info
    else: 
        if not empty_train_cv:
            return train_dataset, train_cv_dataset 
        return train_dataset

def load_datasets (empty_train_cv: bool = False, sample_train: float = None,  return_feature_info = True):
    """
    sample_train: option to sample and train only a fraction of train_dataset
    """
    train_dataset = _single_load (TRAIN_DATASET_FILENAME)
    if not empty_train_cv: 
        train_cv_dataset = _single_load (TRAIN_CV_DATASET_FILENAME)
    cv_dataset = _single_load (CV_DATASET_FILENAME)
    test_dataset = _single_load (TEST_DATASET_FILENAME)
    
    if sample_train is not None:
        num_samples = int(len(train_dataset) * sample_train)
        train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset), seed=RANDOM_STATE).take(num_samples)
        # print (train_dataset.element_spec)
    
    print ('train_dataset: {} examples'.format(len(train_dataset)))
    if not empty_train_cv:
        print ('train_cv_dataset: {} examples'.format(len(train_cv_dataset)))
    print ('cv_dataset: {} examples'.format(len(cv_dataset)))
    print ('test_dataset: {} examples'.format(len(test_dataset)))
    
    # return feature sizes to configure the model
    if return_feature_info:
        feature_info = {
            'group_feature_size' : train_dataset.element_spec[0][INPUT_GROUP_LAYER_NAME].shape[0],
            'technique_feature_size' : train_dataset.element_spec[0][INPUT_TECHNIQUE_LAYER_NAME].shape[0]
        }
        if not empty_train_cv:
            return train_dataset, train_cv_dataset, cv_dataset, test_dataset, feature_info
        return train_dataset, cv_dataset, test_dataset, feature_info
    else: 
        if not empty_train_cv:
            return train_dataset, train_cv_dataset, cv_dataset, test_dataset
        return train_dataset, cv_dataset, test_dataset