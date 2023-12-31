from ...constants import *
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
import tensorflow as tf
from tensorflow import keras
import sys,os
sys.path.append("..")
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'trained_models/model1')

def extract_technique_branch (model_name: str):
    trained_model = keras.models.load_model(os.path.join(SOURCE_PATH, model_name))    
    input_technique_data_sources =         keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_data_sources')
    input_technique_detection_name =       keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_detection_name')
    input_technique_mitigation_id =        keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_mitigation_id')
    input_technique_platforms =            keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_platforms')
    input_technique_software_id =          keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_software_id')
    input_technique_tactics =              keras.Input(shape=(None,), ragged=True, dtype= tf.string, name = 'input_technique_tactics')
    input_technique_interaction_rate =     keras.Input(shape=(1,), dtype=tf.float32, name='input_technique_interaction_rate')
    # input_technique_description =          keras.Input(shape=(768,), dtype=tf.float32, name='input_technique_description')

    vectorize_software_id =              trained_model.vectorize_software_id
    vectorize_tactics =              trained_model.vectorize_tactics
    vectorize_technique_data_sources =         trained_model.vectorize_technique_data_sources
    vectorize_technique_detection_name =       trained_model.vectorize_technique_detection_name
    vectorize_technique_mitigation_id =        trained_model.vectorize_technique_mitigation_id
    vectorize_technique_platforms =            trained_model.vectorize_technique_platforms

    embed_software_id =              trained_model.embed_software_id
    embed_tactics =                trained_model.embed_tactics
    embed_technique_data_sources =         trained_model.embed_technique_data_sources
    embed_technique_detection_name =       trained_model.embed_technique_detection_name
    embed_technique_mitigation_id =        trained_model.embed_technique_mitigation_id
    embed_technique_platforms =            trained_model.embed_technique_platforms
    
    sub_model_data_sources = keras.models.Sequential(
        [
            input_technique_data_sources,
            vectorize_technique_data_sources,
            embed_technique_data_sources
        ]
    )
    sub_model_detection_name = keras.models.Sequential(
        [
            input_technique_detection_name,
            vectorize_technique_detection_name,
            embed_technique_detection_name
        ]
    )
    sub_model_mitigation_id = keras.models.Sequential(
        [
            input_technique_mitigation_id,
            vectorize_technique_mitigation_id,
            embed_technique_mitigation_id
        ]
    )
    sub_model_platforms = keras.models.Sequential(
        [
            input_technique_platforms,
            vectorize_technique_platforms,
            embed_technique_platforms
        ]
    )
    sub_model_software_id = keras.models.Sequential(
        [
            input_technique_software_id,
            vectorize_software_id,
            embed_software_id
        ]
    )
    sub_model_tactics = keras.models.Sequential(
        [
            input_technique_tactics,
            vectorize_tactics,
            embed_tactics
        ]
    )

    technique_data_sources = tf.reduce_mean (sub_model_data_sources.output, axis = 1)
    technique_detection_name = tf.reduce_mean (sub_model_detection_name.output, axis = 1)
    technique_mitigation_id = tf.reduce_mean (sub_model_mitigation_id.output, axis = 1)
    technique_platforms = tf.reduce_mean (sub_model_platforms.output, axis = 1)
    technique_software_id = tf.reduce_mean (sub_model_software_id.output, axis = 1)
    technique_tactics = tf.reduce_mean (sub_model_tactics.output, axis = 1)

    technique_concat = keras.layers.Concatenate(axis=-1)

    concatenated_features = technique_concat (
        [
        input_technique_interaction_rate,
        # input_technique_description,
        technique_data_sources,
        technique_detection_name,
        technique_mitigation_id,
        technique_platforms,
        technique_software_id,
        technique_tactics, 
        ]
    )
    sub_model_technique_nn = trained_model.Technique_NN
    norm_output_Technique = tf.linalg.l2_normalize (concatenated_features, axis = 1)
    learned_feature = sub_model_technique_nn(norm_output_Technique)

    sub_model = keras.models.Model (
        inputs = {
            'input_technique_interaction_rate' : input_technique_interaction_rate,
            # 'input_technique_description' : input_technique_description,
            'input_technique_data_sources' : input_technique_data_sources,
            'input_technique_detection_name' : input_technique_detection_name,
            'input_technique_mitigation_id' : input_technique_mitigation_id,
            'input_technique_platforms' : input_technique_platforms,
            'input_technique_software_id' : input_technique_software_id,
            'input_technique_tactics' : input_technique_tactics,
        },
        outputs = learned_feature
    )
    return sub_model

def build_technique_dataset (X_technique_df: pd.DataFrame()):
    X_technique_df = X_technique_df.drop (columns= TECHNIQUE_ID_NAME)
    input_dict = dict()
    for feature_name in [name for name in X_technique_df.columns if name in RAGGED_TECHNIQUE_FEATURES]:
    # for feature_name in [name for name in X_technique_df.columns]:
        feature_tf = tf.ragged.constant (X_technique_df[feature_name].values, dtype= tf.string)
        input_dict [feature_name] = feature_tf
    feature_tf = tf.constant (X_technique_df[[INPUT_TECHNIQUE_INTERACTION_RATE]].values, dtype=tf.float32)
    input_dict [INPUT_TECHNIQUE_INTERACTION_RATE] = feature_tf
    # feature_tf = tf.convert_to_tensor (list(X_technique_df[INPUT_TECHNIQUE_DESCRIPTION].values))
    # input_dict [INPUT_TECHNIQUE_DESCRIPTION] = feature_tf
    
    res_dataset = tf.data.Dataset.from_tensor_slices (input_dict)
    return res_dataset

def make_look_up_table (learned_features: np.ndarray, id_list: list):
    similarity_matrix = cosine_similarity (learned_features)
    m, _ = similarity_matrix.shape
    sorted_indices_desc_list = []
    # Loop through each row (1D vector) in the 2D similarity_matrix
    for i in range(m):
        sorted_indices_desc = np.argsort(similarity_matrix[i])[::-1]
        sorted_indices_desc_list.append(sorted_indices_desc)
    look_up_table = pd.DataFrame (
        {
            'technique_ID' : id_list,
            'sorted_indices' : sorted_indices_desc_list
        }
    )
    def _technique_index_id_map(lst):
        return [id_list[i] for i in lst]
    look_up_table['sorted_similar_techniques'] = look_up_table['sorted_indices'].apply (_technique_index_id_map)
    look_up_table.drop (columns= ['sorted_indices'], inplace= True)
    
    ### filtering: remove the technique itself from the similar techniques
    look_up_table['sorted_similar_techniques'] = look_up_table['sorted_similar_techniques'].apply(lambda x: x[1:])
    return look_up_table

def get_interacted_tactic_range (interacted_techniques: list, look_up_table: pd.DataFrame()):
    """From a list of interacted techniques: Returns a tuple containing the earliest and latest tactic stage
    """
    interacted_table = look_up_table[look_up_table['technique_ID'].isin(interacted_techniques)]
    earliest_stage = interacted_table['technique_earliest_stage'].min()
    latest_stage = interacted_table['technique_earliest_stage'].max()
    
    return (earliest_stage, latest_stage)
def get_cadidate_techniques (interacted_techniques: list,  look_up_table: pd.DataFrame(), n: int, mode: str = 'latest'):
    """From a list of interacted techniques: Returns a list of candidate techniques. \n
    Step 1: Takes n most similar techniques for each interacted techniques.\n
    Step 2: From the list of Step 1: filter some techniques based on the tactic stage of the interacted techniques\n
        If `mode == 'latest'`: remove candidate techniques if their latest tactic stage is before the latest interacted stage\n
        If `mode == 'earliest'`: remove candidate techniques if their latest tactic stage is before the earliest interacted stage
    """
    interacted_table = look_up_table[look_up_table['technique_ID'].isin(interacted_techniques)]
    # get the first n items in each list
    interacted_table.loc[:, 'sorted_similar_techniques'] = interacted_table['sorted_similar_techniques'].apply(lambda x: x[0:n])
    # filter duplicates by getting unique values
    candidate_techniques = list(interacted_table['sorted_similar_techniques'].explode().unique())
    
    earliest_interacted_stage, latest_interacted_stage = get_interacted_tactic_range (interacted_techniques, look_up_table)
    candidate_table = look_up_table[look_up_table['technique_ID'].isin(candidate_techniques)]
    if mode == 'latest':
        candidate_techniques = list (candidate_table[candidate_table['technique_latest_stage'] >= latest_interacted_stage]['technique_ID'].values)
    elif mode == 'earliest':
        candidate_techniques = list (candidate_table[candidate_table['technique_latest_stage'] >= earliest_interacted_stage]['technique_ID'].values)
    return candidate_techniques

def get_technique_tatic_stage (technique_tactics_df: pd.DataFrame(),tactics_order_df: pd.DataFrame()):
    
    technique_stage = pd.merge (
    left = technique_tactics_df.explode ('input_technique_tactics'),
    right = tactics_order_df,
    how = 'left', left_on= 'input_technique_tactics', right_on= 'tactic_name'
    )
    technique_earliest_stage = technique_stage.groupby ('technique_ID', as_index= False).agg(min)
    technique_earliest_stage.drop (columns= [col for col in technique_earliest_stage if col not in ['technique_ID', 'stage_order']], inplace= True)
    technique_earliest_stage.rename (columns= {'stage_order': 'technique_earliest_stage'}, inplace= True)

    technique_latest_stage = technique_stage.groupby ('technique_ID', as_index= False).agg(max)
    technique_latest_stage.drop (columns= [col for col in technique_latest_stage if col not in ['technique_ID', 'stage_order']], inplace= True)
    technique_latest_stage.rename (columns= {'stage_order': 'technique_latest_stage'}, inplace= True)

    # technique_earliest_stage.head(40)
    technique_stage = pd.merge(
        left = technique_earliest_stage,
        right = technique_latest_stage, 
        how = 'inner', on = 'technique_ID'
    )
    return technique_stage
def make_test_data (report_data: pd.DataFrame, look_up_table: pd.DataFrame(), n: int = 200, mode: str = 'latest'):

    """From the CISA report data, make data for testing. Method:\n
    1. For each report, iteratively take from interacted Techniques as "detected techniques" and the rest as "true subsequent techniques".\n
    2. For each list of "detected techniques", get the candidate Techniques from the provided look-up table

    Args:
        report_data (pd.DataFrame): CiSA report data
        look_up_table (pd.DataFrame): look-up table created from a model
        n (int, optional): number of most similar Technique for each detected techniques. Defaults to 200.
        mode (str, optional): filter mode for look-up table. Defaults to 'latest'.

    Returns:
        _type_: _description_
    """
    test_group_IDs = []
    test_detected_techniques = []
    test_true_subsequent_techniques = []
    test_candidate_techniques = []
    for _, row in report_data.iterrows():
        group_ID = row['group_ID']
        for i in range (len (row['active_techniques'])-1):
            detected_techniques = row['active_techniques'][0:i+1]
            true_subsequent_techniques_techniques = row['active_techniques'][i+1:]
            candidate_techniques = get_cadidate_techniques (interacted_techniques = detected_techniques, look_up_table=look_up_table, n = n, mode = mode)
            
            test_group_IDs.append (group_ID)
            test_detected_techniques.append (detected_techniques)
            test_true_subsequent_techniques.append (true_subsequent_techniques_techniques)
            test_candidate_techniques.append (candidate_techniques)
    data = {
        'group_ID': test_group_IDs,
        'detected_techniques': test_detected_techniques,
        'candidate_techniques': test_candidate_techniques,
        'true_subsequent_techniques': test_true_subsequent_techniques,
    }
    res_df = pd.DataFrame(data = data)
    return res_df
def build_detected_group_profile (processed_group_features: pd.DataFrame(),
                                  processed_technique_features: pd.DataFrame(), 
                                  detected_techniques: list , threshold: int,
                                  train_labels: pd.DataFrame(), 
                                  group_id: str, settings: dict):
    """ 
    Build features for a newly detected group, including:
    1. Interaction rate (float): the initial value equals to the avg or min interaction rate of the interacted groups\n
    2. Interacted tactics (list of tactics): initial value: for each tactic from the interacted groups, the number of tactic interaction is the average number of interactions for that tactic\n
    3. Used software (list of software): initial value: the N most commonly used software, where N is the number of average software used by interacted groups\n

    Args:
        settings (dict): `'initial_interaction'`: set the initial interaction rate of the new group to the avg or min of the rate of train set

    """

    group_interaction_count = len(detected_techniques)
    
    # make a standard scaler, fit on train set's distribution
    pos_y = train_labels[train_labels['label'] == 1]
    train_interaction_count = pos_y['group_ID'].value_counts()
    scaler = StandardScaler()
    scaler.fit (train_interaction_count.values.reshape (-1,1))
    
    interacted_groups = list(pos_y['group_ID'].unique())
    interacted_group_features = processed_group_features [processed_group_features['group_ID'].isin(interacted_groups)]    
    
    # get the list of most frequent software from train set
    avg_software_interaction_rate = interacted_group_features['input_group_software_id'].apply(len).mean().round().astype(int)
    most_frequent_software = interacted_group_features['input_group_software_id'].explode().value_counts().sort_values(ascending = False)
    most_frequent_software = list(most_frequent_software.index)
    most_frequent_software.remove('other')
    most_frequent_software.remove('')
    
    group_interaction_rate = 0
    group_interacted_tactics = [[]]
    group_software =  [[]]
    
    ### 👉 Assign initial values if group has interaction count less than threshold 
    if group_interaction_count < threshold:
        if settings['initial_interaction'] == 'min':
            group_interaction_rate = scaler.transform(np.array(train_interaction_count.min()).reshape(1, -1)).item()
        elif settings['initial_interaction'] == 'avg':
            group_interaction_rate = 0
        
        avg_tactic_rate = interacted_group_features['input_group_tactics'].explode().value_counts()/len(interacted_groups)
        rounded_avg_tactic_rate = avg_tactic_rate.round().astype(int)
        group_interacted_tactics = [[idx for idx, val in rounded_avg_tactic_rate.items() for _ in range(val)]]
    
        group_software = [most_frequent_software[0:avg_software_interaction_rate]]
    
    elif group_interaction_count >= threshold:
        group_interaction_rate = scaler.transform([[group_interaction_count]])
        group_interaction_rate = group_interaction_rate[0][0]
        
        detected_techniques_features = processed_technique_features[processed_technique_features['technique_ID'].isin (detected_techniques)]
        group_interacted_tactics = [list (detected_techniques_features['input_technique_tactics'].explode().values)]
        possible_software = [list(detected_techniques_features['input_technique_software_id'].explode().unique())]
        # possible_software = detected_techniques_features['input_technique_software_id'].explode().unique()
        # possible_software = [software for software in possible_software if software in most_frequent_software[0:avg_software_interaction_rate]]
        group_software = [list (detected_techniques_features['input_technique_software_id'].explode().unique())]
    
    values = {
        'group_ID': group_id,
        'input_group_software_id': group_software,
        'input_group_tactics': group_interacted_tactics,
        'input_group_interaction_rate': group_interaction_rate,
        
    }
    detected_group_features = pd.DataFrame(values, index=[0])
    return detected_group_features

def get_metrics (model, cv_dataset):
    """return the performance metrics of a model given validation data.\n

    Args:
        model (keras.Model): a model instance
        cv_dataset (tf.data.Dataset): crosvalidation dataset

    Returns:
        _type_: best threshold (for F1 score), best F1 score, AUC PR
    """
    batch_cv_dataset =cv_dataset.batch(32)
    val_true = np.array ([label for _, label in cv_dataset])
    val_pred_logit = model.predict (batch_cv_dataset).flatten()
    val_pred_prob = tf.keras.activations.sigmoid(val_pred_logit)
    
    m = tf.keras.metrics.AUC(num_thresholds=200, curve = 'PR', from_logits = True)
    m.update_state (val_true, val_pred_logit, sample_weight = None)
    auc_pr = m.result().numpy()
    
    precisions, recalls, thresholds = precision_recall_curve(val_true, val_pred_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_threshold_index = np.argmax(f1_scores)
    
    best_threshold = thresholds[best_threshold_index]
    best_f1_score = f1_scores[best_threshold_index]
    result = {
        'best_threshold': best_threshold,
        'best_f1_score' : best_f1_score,
        'auc_pr': auc_pr,
    }
    
    return result
