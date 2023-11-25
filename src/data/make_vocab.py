"""
make vocab for each feature in a dataframe
"""

import os, numpy
import pandas as pd
from . import utils
# from ..constants import GROUP_ID_NAME, TECHNIQUE_ID_NAME, LABEL_NAME
from ..constants import *
# Get the root directory of the project
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
# path to save the cleaned data
PROCESS_RUNNING_MSG = "--runing {}".format(__name__)
### END OF CONFIGURATION ###

def make_vocab(df: pd.DataFrame, features: list, path: str):
    for feature_name in features:
        unique_vals = df[feature_name].explode().unique()
        # if "" in unique_vals: unique_vals.remove("")
        numpy.savetxt (fname =os.path.join(path, '{feature_name}_vocab.csv'.format(object = object, feature_name = feature_name)), 
                       X=unique_vals, delimiter= ",",fmt='%s')