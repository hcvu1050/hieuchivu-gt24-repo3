"""
Last update 2023-09-21
Balance the labels in a pandas Dataframe.
"""
import os
import pandas as pd
from .. import utils
from ..constants import GROUP_ID_NAME, TECHNIQUE_ID_NAME, LABEL_NAME
from imblearn.over_sampling import RandomOverSampler

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))

# path to get cleaned data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')

# path to save the built-feature data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

PROCESS_RUNNING_MSG = "--runing {}".format(__name__)
RANDOM_STATE = 13

def naive_random_oversampling (df: pd.DataFrame, save_as_csv = True):
    """
    Balance the labels stored in `LABEL_NAME` column of a DataFrame 'df'
    Currently this only works if 'df' has EXACTLY 3 columns: `GROUP_ID_NAME`, `TECHNIQUE_ID_NAME`, and `LABEL_NAME`
    """
    print (PROCESS_RUNNING_MSG)
    postfix = 'naive_oversampled'
    X = df[[GROUP_ID_NAME, TECHNIQUE_ID_NAME]]
    y = df[[LABEL_NAME]]
    ros = RandomOverSampler(random_state= RANDOM_STATE)
    x_resampled,y_resampled = ros.fit_resample(X , y)
    res_df = pd.concat ([x_resampled,y_resampled], axis =1)
    if save_as_csv:
        utils.save_df_to_csv (res_df, TARGET_PATH, filename= 'train_y', postfix = postfix)
    return res_df
