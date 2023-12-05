"""
last update: 2023-10-12
build_features version 3: Difference from version 2: work with lists in dataframe.
Build features for training the model.
Current functions inlude:
1. One-hot encoding feature
2. Frequency encoding feature
"""
import os
import pandas as pd
import category_encoders as ce
from . import utils
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))

# path to get cleaned data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
# path to the txt file that contains a list of filtered features (exported from selecfeatures.py)
SOURCE_LIST_FILE = os.path.join (SOURCE_PATH, 'selected_features.txt')
# path to save the built-feature data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

PROCESS_RUNNING_MSG = "--runing {}".format(__name__)
TECHNIQUE_TABLE_PREFIX = 'X_technique'
GROUP_TABLE_PREFIX = 'X_group'
from ..constants import GROUP_ID_NAME, TECHNIQUE_ID_NAME

def build_features_onehot(technique_features_df: pd.DataFrame|None, 
                   technique_feature_names: list|None,
                   group_features_df: pd.DataFrame|None,
                   group_features_names: list |None,
                   target_path = TARGET_PATH, 
                   save_as_csv = True):
    """One-hot encoding features for Techniques and Group. 
    The features to be one-hot encoded are defined in `technique_feature_names` and `group_feature_names`"""
    print (PROCESS_RUNNING_MSG)
    
    if (technique_features_df is None) or (group_features_df is None):
    ###  If don't receive the tables as args, get the table from files instead
        technique_features_df, group_features_df = _get_data()
    
    onehot_technique_features_df = _onehot_encode_features (technique_features_df, 
                                                         id = TECHNIQUE_ID_NAME, 
                                                         feature_names= technique_feature_names)
    onehot_group_features_df = _onehot_encode_features (group_features_df,
                                                     id = GROUP_ID_NAME, 
                                                     feature_names= group_features_names)
    if save_as_csv:
        dfs = {
            TECHNIQUE_TABLE_PREFIX : onehot_technique_features_df,
            GROUP_TABLE_PREFIX: onehot_group_features_df
        }
        utils.batch_save_df_to_csv (dfs, target_path, postfix = 'onehot', output_list_file= 'built_features')
    return onehot_technique_features_df, onehot_group_features_df

def build_features_freq_encode (technique_features_df: pd.DataFrame|None, 
                   technique_feature_names: list|None,
                   group_features_df: pd.DataFrame|None,
                   group_features_names: list |None,
                   target_path = TARGET_PATH, 
                   save_as_csv = True):
    """Frequency encoding features for Techniques and Group. Returns as (`Technique_features`,` Group_features`)
    The features to be frequency encoded are defined in `technique_feature_names` and `group_feature_names`"""
    print (PROCESS_RUNNING_MSG)
    if (technique_features_df is None) or (group_features_df is None):
    ###  If don't receive the tables as args, get the table from files instead
        technique_features_df, group_features_df = _get_data()
    
    freq_encode_technique_features_df = _frequency_encode_features (technique_features_df, 
                                                         id_name = TECHNIQUE_ID_NAME, 
                                                         feature_names= technique_feature_names)
    freq_encode_group_features_df = _frequency_encode_features (group_features_df,
                                                     id_name = GROUP_ID_NAME, 
                                                     feature_names= group_features_names)
    if save_as_csv:
        dfs = {
            TECHNIQUE_TABLE_PREFIX : freq_encode_technique_features_df,
            GROUP_TABLE_PREFIX: freq_encode_group_features_df
        }
        utils.batch_save_df_to_csv (dfs, target_path, postfix = 'freq_enc', output_list_file= 'built_features')
    return freq_encode_technique_features_df, freq_encode_group_features_df

def _get_data():
    with open (SOURCE_LIST_FILE, 'r') as file:
        csv_file_names = file.read().splitlines()
        
    technique_file_name = [file_name for file_name in csv_file_names if file_name.startswith(TECHNIQUE_TABLE_PREFIX)]
    technique_features_df = pd.read_csv (os.path.join (SOURCE_PATH, technique_file_name[0]))

    group_file_name = [file_name for file_name in csv_file_names if file_name.startswith(GROUP_TABLE_PREFIX)]
    group_features_df = pd.read_csv (os.path.join (SOURCE_PATH, group_file_name[0]))
    return technique_features_df, group_features_df


def _onehot_encode_features(df: pd.DataFrame, id: str, feature_names: list ) -> pd.DataFrame():
    """Build one-hot encoded features in table `df` for the columns indicated by `feature_names`.\n
    Returns the entire DataFrame with the specified feature one-hot encoded.\n
    The values stored in the specified columns MUST be lists of single-valued strings.
    """
    # get the columns that will not change
    constant_col_names = [col for col in df.columns if col not in (feature_names+[id])]
    constant_cols = df[constant_col_names]
    id_col = df[[id]]
    onehot_feature_dfs = []
    for feature_name in feature_names:
        feature_single_valued = df[[feature_name]].explode(feature_name)
        feature_onehot = pd.get_dummies (feature_single_valued[feature_name], dtype= float)
        feature_onehot = feature_onehot.groupby(level=0).max()
        onehot_feature_dfs.append (feature_onehot)
    # ❗AVOID DATA LOSS, only group as max value for one-hot encoded features
    onehot_feature_dfs = [id_col] + onehot_feature_dfs
    onehot_features_df = pd.concat (
        onehot_feature_dfs, 
        axis = 1)
    
    constant_cols = [id_col] + [constant_cols]
    constant_cols_df = pd.concat (
        constant_cols,
        axis = 1
    )

    res_df = pd.merge (left = constant_cols_df, right= onehot_features_df, on = id, how = 'left')
    return res_df

def _frequency_encode_features (df: pd.DataFrame(), id_name: str, feature_names: list) -> pd.DataFrame():
    """Build frequency encoded features in table `df` for the columns indicated by `feature_names`.\n
    Returns the entire DataFrame with the specified feature frequency encoded.\n
    Work for 2 cases\n
    (1): Single-valued strings (e.g.: "MacOS" , "Windows")\n
    (2): Multiple-valued strings (e.g.: "MacOS, Windows"). The default char that separates the values is `,`
    """
    # get the columns that will not change
    constant_col_names = [col for col in df.columns if col not in (feature_names+[id_name])]
    constant_cols = df[constant_col_names]
    id_col = df[[id_name]]

    freq_encoded_feature_dfs = []
    
    for feature_name in feature_names:
        feature_single_valued = df[[feature_name]].explode(feature_name)
        freq_enc = ce.CountEncoder(normalize=True, handle_missing= 'return_nan') #❗return nan
        feature_freq_encoded = freq_enc.fit_transform (feature_single_valued[feature_name], return_df = True)
        feature_freq_encoded_oh = pd.get_dummies(feature_freq_encoded[feature_name], dtype= float)
        feature_freq_encoded_oh = feature_freq_encoded_oh.groupby(level=0).max()
        feature_freq_encoded_oh_true_val = feature_freq_encoded_oh.multiply(feature_freq_encoded_oh.columns, axis= 1)
        
        freq_encoded_feature_dfs.append (feature_freq_encoded_oh_true_val)
        
    freq_encoded_feature_dfs = [id_col] + freq_encoded_feature_dfs
    freq_encoded_features_df= pd.concat(
        freq_encoded_feature_dfs, axis= 1
    )
    
    constant_cols = [id_col] + [constant_cols]
    constant_cols_df = pd.concat (
        constant_cols,
        axis = 1
    )
    res_df = pd.merge (left = constant_cols_df, right= freq_encoded_features_df, on = id_name, how = 'left')
    return res_df

def build_feature_sentence_embed (df: pd.DataFrame(), feature_name:str, tokenizer, embed_model):
    """
    embedd sentences with bert pretrained model. Define `tokenizer` and `embed_model` to use.
    For example:\n
    `tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')`\n
    `model = TFBertModel.from_pretrained('bert-base-uncased')`
    """
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = TFBertModel.from_pretrained('bert-base-uncased')
    def embed_sentence (sentence):
        tokens = tokenizer(sentence, padding=True, truncation=True, return_tensors='tf')
        outputs = embed_model(tokens)
        flat_embedding = tf.reshape(outputs.last_hidden_state[:, 0, :], [-1])
        return flat_embedding.numpy()
        # return flat_embedding
    df[feature_name] = df[feature_name].apply(embed_sentence)
    return df

def build_feature_interaction_frequency (label_df: pd.DataFrame(), 
                                         feature_df: pd.DataFrame(), 
                                         object_ID: str, 
                                         feature_name: str, normalize: bool = True,
                                         initialize_null_interaction: list = None) -> pd.DataFrame():
    """Add a feature created by the number of interactions each Technique or Group was involved\n
    CAUTION: be aware of data leakage.

    Args:
        label_df (pd.DataFrame): interation table
        feature_df (pd.DataFrame): feature table to which the new feature is added to
        object_ID (str): object ID
        feature_name (str): column name for the new feature that will be added
        initialize_null_interaction (str): initialize interactions for not-interacted objects, either by the average or minimum interaction rate of the interacted objects. 
        This option is used for only Techniques and is calculated based only on TRAIN data.
    Returns:
        pd.DataFrame: return the feature table with the added feature column
    """
    interaction_count = label_df[label_df['label'] == 1.0][object_ID].value_counts()
    res_df = pd.merge (left = feature_df, right = interaction_count, on = object_ID, how = 'left')
    res_df.rename (columns= {'count': feature_name}, inplace= True)
    initial_value = 0
    if initialize_null_interaction is not None:
        if initialize_null_interaction[0] == 'global': 
            if initialize_null_interaction[1] == "min": 
                initial_value = res_df[feature_name].dropna().astype(float).min()
            elif initialize_null_interaction[1] == "avg":
                initial_value = res_df[feature_name].dropna().astype(float).mean()
                
        elif initialize_null_interaction[0] == 'tactics':
            ### 1. Get the list of unused Techniques
            pos_y = label_df[label_df['label'] == 1]
            used_techniques=  pos_y['technique_ID'].unique()
            all_techniques = res_df['technique_ID'].unique()
            unused_techniques = [t for t in all_techniques if t not in used_techniques]
            
            ### 2. Get the min or average interaction rate of each tactic used
            tactic_interaction_rate = res_df[[ 'input_technique_tactics', 'input_technique_interaction_rate']]
            tactic_interaction_rate = tactic_interaction_rate.explode ('input_technique_tactics')
            if initialize_null_interaction[1] == 'avg':
                tactic_interaction_rate = tactic_interaction_rate.groupby (by= 'input_technique_tactics', as_index= False)['input_technique_interaction_rate'].mean()
            elif initialize_null_interaction[1] == 'min':
                tactic_interaction_rate = tactic_interaction_rate.groupby (by= 'input_technique_tactics', as_index= False)['input_technique_interaction_rate'].min()
                
            ### 3. Make a table that assign new values for the unused techniques
            unused_techniques = res_df[res_df['technique_ID'].isin(unused_techniques)]
            unused_techniques = unused_techniques[['technique_ID', 'input_technique_tactics']]
            unused_techniques = unused_techniques.explode ('input_technique_tactics')
            unused_techniques = pd.merge (left = unused_techniques, right = tactic_interaction_rate, how = 'left', on = 'input_technique_tactics')
            unused_techniques = unused_techniques[['technique_ID', 'input_technique_interaction_rate']]
            unused_techniques = unused_techniques.groupby (by = 'technique_ID', as_index= False).agg ('mean')
            ### 4. Update the values in the original table `technique_df`    
            for index, row in unused_techniques.iterrows():
                id_val = row['technique_ID']
                updated_val = row['input_technique_interaction_rate']
                
                # Locate the corresponding row in df_main and update the value
                res_df.loc[res_df['technique_ID'] == id_val, 'input_technique_interaction_rate'] = updated_val
        
    elif initialize_null_interaction is None: 
        initial_value = 0
    res_df[feature_name].fillna (initial_value, inplace= True)

    if normalize: 
        scaler = StandardScaler()
        res_df[feature_name] = scaler.fit_transform (res_df[[feature_name]])
    return res_df

def build_feature_used_tactics (label_df: pd.DataFrame(), 
                                group_df: pd.DataFrame(), 
                                technique_df: pd.DataFrame(),
                                feature_name: str,
                                group_ID: str = 'group_ID', 
                                ) -> pd.DataFrame():
    """
    Add feature for Group containing interacted Tactics of each Group.
    Each feature value is a list of strings (of tactic names)
    CAUTION: be aware of data leakage.

    """
    pos_y = label_df[label_df['label'] == 1]
    g_tactic = pd.merge (left = pos_y[[group_ID,'technique_ID']], right = technique_df[['technique_ID','input_technique_tactics']], 
                     how = 'left', on = 'technique_ID')
    g_tactic = pd.merge (left = group_df[[group_ID]], right= g_tactic, how = 'left', on = group_ID)
    g_tactic['input_technique_tactics'].fillna ('', inplace= True)
    g_tactic = g_tactic.explode('input_technique_tactics').groupby(group_ID, as_index=False).agg(list)[[group_ID,'input_technique_tactics']]
    res_df = pd.merge (left = group_df, right = g_tactic, on = group_ID , how = 'left' )
    res_df.rename (columns= {'input_technique_tactics': feature_name}, inplace= True)
    return res_df