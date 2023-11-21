"""
limit the cardinality of features
"""

import os
import pandas as pd
from . import utils
from ..constants import *
TECHNIQUE_TABLE_PREFIX = 'X_technique'
GROUP_TABLE_PREFIX = 'X_group'
RESULT_FILE_POSTFIX = 'selected_features'

def batch_reduce_vals_based_on_nth_most_frequent (df: pd.DataFrame(), setting: dict):
    for col in setting.keys():
        df = reduce_vals_based_on_nth_most_frequent (df = df, feature_name=col, n = setting[col])
    return df

def reduce_vals_based_on_nth_most_frequent (df: pd.DataFrame(), feature_name: str, n: int):
    """
    Reduce feature values to nth most frequent values. 
    If after reducing, the values for an instance is empty. Return a list containing an empty str
    """
    all_vals = df[feature_name].explode()
    selected_vals = list(all_vals.value_counts().sort_values(ascending=False).index)[:n]
    if "" in selected_vals: 
        selected_vals = list(all_vals.value_counts().sort_values(ascending=False).index)[:n+1]
    def _filter_seltected_vals (lst):
        res = [item for item in lst if item in selected_vals]
        if len(res) == 0: 
            return ['']
        return res
    
    res_df = df.copy()
    res_df[feature_name] = res_df[feature_name].apply(_filter_seltected_vals)
    return res_df

def batch_reduce_vals_based_on_percentage (df: pd.DataFrame(), setting: dict):
    for col in setting.keys():
        df = reduce_vals_based_on_percentage (df = df, feature_name= col, percentage= setting[col])
    return df

def reduce_vals_based_on_percentage (df: pd.DataFrame(), feature_name: str, percentage: float):
    all_vals = df[feature_name].explode()
    value_counts = all_vals.value_counts().sort_values(ascending=False)
    threshold = len(all_vals) * percentage
    selected_vals = []
    cumulative_count = 0
    for value, count in value_counts.items():
        if value == '':
            selected_vals.append(value)
            continue
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