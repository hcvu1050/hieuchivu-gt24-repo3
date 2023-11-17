"""
limit the cardinality of features
"""

import os
import pandas as pd
from . import utils
from ..constants import GROUP_ID_NAME, TECHNIQUE_ID_NAME
TECHNIQUE_TABLE_PREFIX = 'X_technique'
GROUP_TABLE_PREFIX = 'X_group'
RESULT_FILE_POSTFIX = 'selected_features'

def batch_reduce_vals_based_on_nth_most_frequent (df: pd.DataFrame, setting: dict):
    for col in setting.keys():
        df = reduce_vals_based_on_nth_most_frequent (df = df, feature_name=col, n = setting[col])
    return df

def reduce_vals_based_on_nth_most_frequent (df: pd.DataFrame(), feature_name: str, n: int):
    """
    Note: if the size of current feature vals is less than or equal to the limit, do not remove the vals
    """
    all_vals = df[feature_name].explode()
    selected_vals = list(all_vals.value_counts().sort_values(ascending=False).index)[:n]
    if "" in selected_vals: 
        # print ('empty string detected, before length: ', len(selected_vals) )
        selected_vals.remove("")
        # print ('after len:', len(selected_vals))
    def _filter_seltected_vals (lst):
        if len(lst) <= n: 
            if "" in lst: 
                if len(lst) == 1: return list()
                return lst.remove("")
            return lst
        return [item for item in lst if item in selected_vals]
    
    res_df = df.copy()
    res_df[feature_name] = res_df[feature_name].apply(_filter_seltected_vals)
    return res_df

def reduce_vals_based_on_percentage (df: pd.DataFrame(), feature_name: str, percentage: float):
    all_vals = df[feature_name].explode()
    value_counts = all_vals.value_counts().sort_values(ascending=False)
    threshold = len(all_vals) * percentage
    
    selected_vals = []
    cumulative_count = 0
    for value, count in value_counts.items():
        if cumulative_count + count <= threshold:
            selected_vals.append(value)
            cumulative_count += count
        else:
            break
    
    def _filter_seltected_vals (lst):
        return [item for item in lst if item in selected_vals]
    res_df = df.copy()
    res_df[feature_name] = res_df[feature_name].apply(_filter_seltected_vals)
    return res_df