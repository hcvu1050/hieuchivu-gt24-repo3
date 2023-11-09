"""
last update: 2023-09-21
- used to split the dataset into train, cross-validation, and test set. The ratio of the sets are defined by the users
- Currently there are two ways to split the dataset:
	1. Split the dataset randomly into train, cross-validation, and test set
	2. (Preferred method) Split the dataset by randomly split the Groups. Then, based on the split Groups, build the train, cross-validation, and test set. 
        With this process, data from a Group and be only from either train, or cross-validation, or test set.
"""
import os
import pandas as pd
from .. import utils
from ..constants import GROUP_ID_NAME
from sklearn.model_selection import train_test_split

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
# path to get cleaned data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
# path to save the built-feature data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')
TARGET_PREFIX = 'split_'

RANDOM_STATE = 13
PROCESS_RUNNING_MSG = "--runing {}".format(__name__)

### MAIN FUNCTION ###
def split_data_by_group (target_df: pd.DataFrame, ratio: list, save_as_csv = True):
    """ Splits data by group randomly so that: data of a Group ONLY belong to either train set, or cv set, or test set.
    values in `ratio` indicate the ratio size traing/cv/test, and should be added up to 1.
    """
    print (PROCESS_RUNNING_MSG)
    
    train_size, cv_size, test_size =  ratio [0], ratio[1], ratio[2]
    group_IDs = target_df[GROUP_ID_NAME].unique()
    # get the list of Group IDs, then split this list based on the given ratio
    train_IDs, test_IDs = train_test_split (group_IDs, train_size=train_size, random_state=RANDOM_STATE)
    # relative ratio for cv_size/ test_size
    rel_cv_size = cv_size / (cv_size + test_size)
    rel_test_size = 1 - rel_cv_size    
    # split cv and test ids
    cv_IDs, test_IDs = train_test_split (test_IDs, test_size= rel_test_size, random_state= RANDOM_STATE)
    # build train, cv, and test sets based on splitted ids
    train_df   = target_df[target_df[GROUP_ID_NAME].isin(train_IDs)]
    cv_df      = target_df[target_df[GROUP_ID_NAME].isin(cv_IDs)]
    test_df    = target_df[target_df[GROUP_ID_NAME].isin(test_IDs)]
    if save_as_csv:
        dfs = {
            'train_y': train_df,
            'cv_y': cv_df,
            'test_y': test_df
        }
        utils.batch_save_df_to_csv (dfs, target_path = TARGET_PATH , postfix= 'split_by_group', output_list_file= 'split')
    return train_df, cv_df, test_df

### OLD FUNCTION: SPLIT DATA RANDOMLY
def split_data (target_df: pd.DataFrame, ratio: list, save_as_csv = True):
    """ Splits data randomly based on ratio.
    # Future task: ratio should be added up to 1
    """
    train_size, cv_size, test_size =  ratio [0], ratio[1], ratio[2]
    # split train set
    train_df, test_df = train_test_split(target_df, train_size = train_size, random_state=RANDOM_STATE)
    # relative ratio for cv_size/ test_size
    rel_cv_size = cv_size / (cv_size + test_size)
    rel_test_size = 1 - rel_cv_size
    # split cv and test set
    cv_df, test_df = train_test_split(test_df, test_size= rel_test_size, random_state= RANDOM_STATE)
    if save_as_csv:
        dfs = {
            'train_df': train_df,
            'cv_df': cv_df,
            'test_df': test_df
        }
        utils.batch_save_df_to_csv (dfs, target_path = TARGET_PATH , prefix= 'random_splitted_')
    
    return train_df, cv_df, test_df