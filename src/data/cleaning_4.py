"""
V4: separate feature value processing from _combine_feature
used to clean the data by reducing outliers/noise, handling missing values, etc.
1. Reads collected files from data/interim
2. Filters the columns needed for training, then rename the columns 
3. Make interaction matrix between Group and Technique with option to include unused Techniques in the matrix.
4. Main Function clean_data: From the cleaned data, create the following tables:
    (a). Group-Technique interaction matrix
    (b). Technique features: Containing all available features for Techniques. 
    (c). Group features: Containing all available features for Groups
5. Export to data/interim
"""
import os, re, numpy
import pandas as pd
from . import utils
# from ..constants import GROUP_ID_NAME, TECHNIQUE_ID_NAME, LABEL_NAME
from ..constants import *
# Get the root directory of the project
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
# path to get collected data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
# path to save the cleaned data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')
PROCESS_RUNNING_MSG = "--runing {}".format(__name__)
### END OF CONFIGURATION ###

### MAIN FUNCTION
def clean_data(include_unused_technique: bool, target_path = TARGET_PATH , save_as_csv = True):
    """Filters the columns needed for training, then combines all features of a object group into one table.\n
    Returns 3 tables:\n
    a. Technique features\n
    b. Group features\n
    c. Group-Technique interaction matrix\n
    """
    print (PROCESS_RUNNING_MSG)
    data_and_setting = _get_data()
    filtered_dfs = _filter_rename_columns(data_and_setting)
    
    ### get Group dfs from filtered_dfs
    group_full_IDs_df = filtered_dfs['groups_df']
    group_string_feature_df_names = ['groups_software_df']
    group_string_feature_dfs = [filtered_dfs[key] for key in group_string_feature_df_names]
    group_sentence_feature_df = filtered_dfs['groups_description']
    ## Process string vals for group features
    group_string_feature_dfs = [_process_string_vals (group_full_IDs_df,feature_df) for feature_df in group_string_feature_dfs]
    ## Process sentence for group features
    group_sentence_feature_df = _process_sentence_vals (group_sentence_feature_df)

    group_features_df = _combine_features (IDs_df= group_full_IDs_df, feature_dfs= group_string_feature_dfs + [group_sentence_feature_df])
    
    ### get Technique dfs from filtered_dfs
    technique_full_IDs_df = filtered_dfs['techniques_df']
    technique_string_feature_df_names = ['techniques_platforms_df',
                                        'techniques_tactics_df',
                                        'techniques_data_sources',
                                        'techniques_defenses_bypassed_df',
                                        'techniques_permissions_required_df',
                                        'techniques_mitigations_df',
                                        'techniques_detections_df',
                                        'techniques_software_df']
    
    technique_string_feature_dfs = [filtered_dfs[key] for key in technique_string_feature_df_names]
    technique_sentence_feature_df = filtered_dfs['techniques_description']
    ## Process string vals for technique features
    technique_string_feature_dfs = [_process_string_vals (technique_full_IDs_df,feature_df) for feature_df in technique_string_feature_dfs]
    ## Process sentence for group features
    technique_sentence_feature_df = _process_sentence_vals (technique_sentence_feature_df)

    technique_features_df = _combine_features (IDs_df= technique_full_IDs_df, feature_dfs= technique_string_feature_dfs + [technique_sentence_feature_df])
    ### Interaction Matrix    
    interaction_matrix = _make_interaction_matrix(
        include_unused = include_unused_technique,
        user_IDs_df= filtered_dfs['groups_df'],
        item_IDs_df= filtered_dfs['techniques_df'],
        positive_cases= filtered_dfs['labels_df']
    )
    if save_as_csv:
        res_dfs = {
            'X_technique' : technique_features_df,
            'X_group' : group_features_df ,
            'y' : interaction_matrix,
        }
        utils.batch_save_df_to_csv (res_dfs, target_path, postfix = 'cleaned', output_list_file= 'cleaned')
    return technique_features_df, group_features_df, interaction_matrix


def _get_data():
    """Get the collected tables and make settings for filtering and renaming columns.\n
    Returns a dictionary named `data_and_setting` used to filter the columns needed for training from the collected tables
    key = filename for a table, value = tuple
    each table is assigned with a tuple including:
        (1) the dataframe\n
        (2) a list of columns in the table that are used for training\n
        (3) a list of names for re-naming columns in (1) for clarity.\n
    The tables that have every group IDs and technique IDs are stored in `groups_df` and `techniques_df` respectively
    """
    groups_df = pd.read_csv(os.path.join (SOURCE_PATH, 'collected_groups_df.csv'))
    groups_software_df = pd.read_csv(os.path.join (SOURCE_PATH, 'collected_groups_software_df.csv'))
    techniques_df = pd.read_csv(os.path.join (SOURCE_PATH, 'collected_techniques_df.csv'))
    techniques_mitigations_df = pd.read_csv(os.path.join (SOURCE_PATH, 'collected_techniques_mitigations_df.csv'))
    techniques_detections_df = pd.read_csv(os.path.join (SOURCE_PATH, 'collected_techniques_detections_df.csv'))
    techniques_software_df = pd.read_csv(os.path.join (SOURCE_PATH, 'collected_techniques_software_df.csv'))
    
    labels_df = pd.read_csv(os.path.join (SOURCE_PATH, 'collected_labels_df.csv'))
    
    data_and_setting = {
        'groups_df' :                   (groups_df,         ['ID'],                     [GROUP_ID_NAME]),
        'groups_description':           (groups_df,         ['ID','description'],       [GROUP_ID_NAME,INPUT_GROUP_DESCRIPTION]),
        'groups_software_df' :          (groups_software_df,['source ID', 'target ID'], [GROUP_ID_NAME, INPUT_GROUP_SOFTWARE_ID ]),
        'techniques_df' :               (techniques_df,     ['ID'],                     [TECHNIQUE_ID_NAME]), 
        'techniques_platforms_df':      (techniques_df,     ['ID', 'platforms'],        [TECHNIQUE_ID_NAME, INPUT_TECHNIQUE_PLATFORMS]),
        'techniques_tactics_df':        (techniques_df,     ['ID', 'tactics'],          [TECHNIQUE_ID_NAME, INPUT_TECHNIQUE_TACTICS]),
        'techniques_data_sources':      (techniques_df,     ['ID', 'data sources'],     [TECHNIQUE_ID_NAME, INPUT_TECHNIQUE_DATA_SOURCES]),
        'techniques_defenses_bypassed_df':      (techniques_df, ['ID', 'defenses bypassed'],    [TECHNIQUE_ID_NAME, INPUT_TECHNIQUE_DEFENSES_BYPASSED]),
        'techniques_permissions_required_df':   (techniques_df, ['ID','permissions required'],  [TECHNIQUE_ID_NAME, INPUT_TECHNIQUE_PERMISSIONS_REQUIRED]),
        'techniques_mitigations_df':    (techniques_mitigations_df, ['source ID', 'target ID'],     [INPUT_TECHNIQUE_MITIGATION_ID, TECHNIQUE_ID_NAME]), 
        'techniques_detections_df' :    (techniques_detections_df,  ['source name','target ID'],    [INPUT_TECHNIQUE_DETECTION_NAME, TECHNIQUE_ID_NAME]),
        'techniques_software_df':       (techniques_software_df,    ['source ID', 'target ID'],     [INPUT_TECHNIQUE_SOFTWARE_ID, TECHNIQUE_ID_NAME]),
        'techniques_description':       (techniques_df,             ['ID','description'],           [TECHNIQUE_ID_NAME,INPUT_TECHNIQUE_DESCRIPTION]),
        'labels_df' :                   (labels_df,                 ['source ID', 'target ID'],     [GROUP_ID_NAME, TECHNIQUE_ID_NAME])
    }
    return data_and_setting

def _filter_rename_columns (data_and_setting):
    """
    Based on data_and_setting:\n
    Filters the selected columns for the collected data, then re-name them
    Returns a dictionary. [key: value] = ["table_name": dataframe]
    """
    res_dfs = {}
    for key in data_and_setting.keys():
        df = data_and_setting[key][0]        
        # 1- Filter the columns
        df = df[data_and_setting[key][1]]
        # 2- Rename the columns
        df.columns = data_and_setting[key][2]
        
        res_dfs[key] = df
    return res_dfs

def _make_interaction_matrix (user_IDs_df, 
                              item_IDs_df, 
                              positive_cases, include_unused: bool) -> pd.DataFrame():
    """Creates an interaction matrix (all possible combination) between Groups and Techniques based on the IDs.
    `include_unused`: option to include Techniques that are not used by any Group in interaction matrix.
    """
    if include_unused == False:
        item_IDs_df = item_IDs_df [item_IDs_df[TECHNIQUE_ID_NAME].isin (positive_cases[TECHNIQUE_ID_NAME])]
    #else:
    group_technique_interactions = pd.merge (user_IDs_df, item_IDs_df, how = 'cross')
    # positive_cases ['label'] = 1
    positive_cases = positive_cases.assign (label = 1)
    group_technique_interaction_matrix = pd.merge (
        left = group_technique_interactions,
        right = positive_cases, 
        on = [GROUP_ID_NAME, TECHNIQUE_ID_NAME], 
        how = 'left'
    )
    group_technique_interaction_matrix[LABEL_NAME].fillna (0, inplace= True)
    return group_technique_interaction_matrix

def _process_string_vals (full_id_df: pd.DataFrame(), feature_df: pd.DataFrame(), feature_name: str = None, feature_sep_char = ','):
    """process string values for feature
    ❗only work properly when the dataframe has 2 columns only: [ID name] and [feature name]
    """
    id_name = [col_name for col_name in list(feature_df.columns) if col_name in [GROUP_ID_NAME, TECHNIQUE_ID_NAME]][0]
    id_df = feature_df[[id_name]]
    if feature_name == None: feature_name = [col_name for col_name in list(feature_df.columns) if col_name != id_name][0]
    
    feature_processed = feature_df[feature_name].str.lower()
    feature_processed = feature_processed.str.replace (r'[-/:]\s*', r' ', regex = True)
    feature_processed = feature_processed.str.replace(r',\s*',',', regex = True)
    feature_processed = feature_processed.str.replace(r'(?<=[a-zA-Z0-9])\s(?=[a-zA-Z0-9])','_', regex = True)
    feature_df = pd.concat ([id_df, feature_processed], axis= 1)
    multivalued = feature_processed.str.contains(feature_sep_char, case=False).any()
    if multivalued:
        feature_df[feature_name] = feature_df[feature_name].str.split(feature_sep_char)
        feature_df = feature_df.explode(column= feature_name,ignore_index = False)

    feature_df = pd.merge (left = full_id_df, right = feature_df, on = id_name, how = 'left')
    
    feature_df[feature_name].fillna (value = '', inplace = True)
    feature_df = feature_df.groupby(id_name, as_index= False).agg(list)
    return feature_df

def _process_sentence_vals (feature_df: pd.DataFrame(), feature_name: str = None):
    """
    process sentences as feature values
    ❗only work properly when the dataframe has 2 columns only: [ID name] and [feature name]
    """
    def _process_sentence(sentence):
        # Remove square brackets, but keep the contents inside
        sentence = re.sub(r'\[|\]', '', sentence)
        # Remove round brackets with content inside
        sentence = re.sub(r'\([^)]*\)', '', sentence)
        # Remove links
        sentence = re.sub(r'https?://\S+', '', sentence)
        sentence = sentence.replace('\n', ' ')
        return sentence
    if feature_name == None: feature_name = [col_name for col_name in list(feature_df.columns) if col_name not in [GROUP_ID_NAME, TECHNIQUE_ID_NAME ]][0]
    feature_df[feature_name] = feature_df[feature_name].apply (_process_sentence)
    return feature_df

def _combine_features (IDs_df: pd.DataFrame(), feature_dfs: list):
    id_name = IDs_df.columns[0]
    res_df = IDs_df
    for feature_df in feature_dfs: 
        res_df = pd.merge (left = res_df, right= feature_df, on=id_name, how = 'left')
    return res_df

# def _combine_features (object: str, dfs: dict, feature_sep_char = ',') -> pd.DataFrame():
#     """Combines the features of the object (Group or Technique) based of the tables of features stored in dfs. 
#     In dfs, the key indicates if its value belongs to Group or Technique based on the key's prefix.
#     The features are always merged based on the list of ALL object IDs (group_ID or technique_ID).

#     Args:
#         object (str): "group" or "technique"
#         dfs (dict): dfs stores the filtered tables for the object

#     Returns:
#         pd.DataFrame: Return the merged table of the object
#     """
#     object_features = pd.DataFrame()
#     id_name = ''
#     if object == 'group':
#         all_id_df = dfs['groups_df']
#         object_features = all_id_df #initialize the result dataframe. starts with the list of IDs
#         id_name = GROUP_ID_NAME
#     elif object == 'technique':
#         all_id_df = dfs['techniques_df'] #initialize the result dataframe. starts with the list of IDs
#         object_features = all_id_df
#         id_name = TECHNIQUE_ID_NAME
    
#     # The features are merged with the list of object IDs (group_ID or technique_ID)
#     # unique_vals = numpy.empty(shape = (0,))
#     for key in [key for key in dfs.keys() if (key.startswith (object)) and key not in (['groups_df', 'techniques_df'])]:
#         feature_df = dfs[key]
#         id_df = feature_df[[id_name]]
#         feature_name = [col_name for col_name in feature_df.columns if col_name != id_name][0]
#         # string values preprocessing 
#         feature_processed = feature_df[feature_name].str.lower()
#         feature_processed = feature_processed.str.replace (r'[-/:]\s*', r' ', regex = True)
#         feature_processed = feature_processed.str.replace(r',\s*',',', regex = True)
#         feature_processed = feature_processed.str.replace(r'(?<=[a-zA-Z0-9])\s(?=[a-zA-Z0-9])','_', regex = True)
#         feature_df = pd.concat ([id_df, feature_processed], axis= 1)
#         multivalued = feature_processed.str.contains(feature_sep_char, case=False).any()
#         if multivalued:
#             feature_df[feature_name] = feature_df[feature_name].str.split(feature_sep_char)
#             feature_df = feature_df.explode(column= feature_name,ignore_index = False)
        

#         # unique_vals = feature_df[feature_name].dropna().unique()
#         # numpy.savetxt (fname =os.path.join(TARGET_PATH, '{feature_name}_vocab.csv'.format(object = object, feature_name = feature_name)), 
#         #                X=unique_vals, delimiter= ",",fmt='%s')
        
#         feature_df = pd.merge (
#             left = all_id_df, right=feature_df, on = id_name, how = 'left'
#         )
#         feature_df[feature_name].fillna (value = '', inplace = True)
        
#         feature_df = feature_df.groupby(id_name, as_index= False).agg(list)
#         object_features = pd.merge (
#             left = object_features,
#             right= feature_df,
#             on = id_name,
#             how = 'left'
#         )
    
#     return object_features