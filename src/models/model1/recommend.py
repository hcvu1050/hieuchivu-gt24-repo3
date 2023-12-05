from ...constants import *
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from sklearn.metrics.pairwise import cosine_similarity
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
    input_technique_description =          keras.Input(shape=(768,), dtype=tf.float32, name='input_technique_description')

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
        input_technique_description,
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
            'input_technique_description' : input_technique_description,
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
    feature_tf = tf.convert_to_tensor (list(X_technique_df[INPUT_TECHNIQUE_DESCRIPTION].values))
    input_dict [INPUT_TECHNIQUE_DESCRIPTION] = feature_tf
    
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

def get_interacted_tactic_range (interacted_techniques: list, look_up_table: pd.DataFrame()):
    """From a list of interacted techniques: Returns a tuple containing the earliest and latest tactic stage
    """
    interacted_table = look_up_table[look_up_table['technique_ID'].isin(interacted_techniques)]
    earliest_stage = interacted_table['technique_earliest_stage'].min()
    latest_stage = interacted_table['technique_earliest_stage'].max()
    
    return (earliest_stage, latest_stage)
        
def extract_cisa_techniques (url: str) -> list:
    """Extract the techniques used by an advesary in a CISA Report url by web scraping.\n
    The techniques are assumed to be stored in a `<table>` class and sorted by the report.

    Args:
        url (str): Url of the CISA report

    Returns:
        list: A list of technique collected from the CISA report url.
    """
    response = requests.get(url)
    filtered_strings = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        all_tables = soup.find_all('table')
        regex_pattern = re.compile(r'T\d{4}\.*\d*')
        for table in all_tables: 
            matched_elements = list (table.find_all(string=regex_pattern))
            if len(matched_elements) >0 and len(filtered_strings)>0: 
                print ('WARNING: Extracting Techniques from more than one table. Check the url.')
                return
            if len(matched_elements) >0: filtered_strings.extend(matched_elements)
    else:
        print('Failed to fetch the webpage.')
        return
    return filtered_strings

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

def get_report_data (report_codes: list):
    """get the a Table of interacted techniques in each CISA report given the list of report codes.
    Args:
        report_codes (list): the list of report codes. code example 'aa22-277a'

    """
    group_IDs = []
    interacted_techniques = []
    for report_code in report_codes:
        url = 'https://www.cisa.gov/news-events/cybersecurity-advisories/' + report_code
        group_IDs.append(report_code)
        interacted_techniques.append (extract_cisa_techniques (url))
    data = {
        'group_ID': group_IDs,
        'interacted_techniques': interacted_techniques
    }
    report_data = pd.DataFrame (data=data)
    return report_data

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
        for i in range (len (row['interacted_techniques'])-1):
            detected_techniques = row['interacted_techniques'][0:i+1]
            true_subsequent_techniques_techniques = row['interacted_techniques'][i+1:]
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

def build_new_group_profile (processed_group_features: pd.DataFrame(), label_df: pd.DataFrame(), new_group_id: str, settings: dict):
    """Build features for a new groups, including:
    1. Description embedding: equals to the avg pooling of the processed groups' embeddings\n
    2. Interaction rate: equals to the avg or min interaction rate of the interacted groups\n
    3. Interacted tactics: average tactic interaction rate for each tactic from the interacted groups\n
    4. Used software: the N most commonly used software, where N is the number of average software used by interacted groups\n
    Args:
        processed_group_features (pd.DataFrame): _description_
        label_df (pd.DataFrame): _description_
        new_group_id (str): _description_
        settings (dict): _description_

    Returns:
        _type_: _description_
    """
    
    pos_y = label_df[label_df['label'] == 1]
    interacted_groups = list(pos_y['group_ID'].unique())
    interacted_group_features = processed_group_features [processed_group_features['group_ID'].isin(interacted_groups)]    
    
    initial_interaction = 0
    initial_description = interacted_group_features['input_group_description'].apply(pd.Series).mean().tolist()
    initial_tactics = [[]]
    initial_software =  [[]]
    if settings['interaction'] == 'min':
        initial_interaction = (interacted_group_features['input_group_interaction_rate']).min()
    elif settings['interaction'] == 'avg':
        initial_interaction = (interacted_group_features['input_group_interaction_rate']).mean()
    
    avg_tactic_rate = interacted_group_features['input_group_tactics'].explode().value_counts()/len(interacted_groups)
    rounded_avg_tactic_rate = avg_tactic_rate.round().astype(int)
    initial_tactics = [[idx for idx, val in rounded_avg_tactic_rate.items() for _ in range(val)]]
    
    avg_software_interaction_rate = interacted_group_features['input_group_software_id'].apply(len).mean().round().astype(int)
    most_frequent_software = interacted_group_features['input_group_software_id'].explode().value_counts().sort_values(ascending = False)
    most_frequent_software = list(most_frequent_software.index)
    most_frequent_software.remove('other')
    most_frequent_software.remove('')
    initial_software = [most_frequent_software[0:avg_software_interaction_rate]]
    
    values = {
        'group_ID': new_group_id,
        'input_group_software_id': initial_software,
        'input_group_tactics': initial_tactics,
        'input_group_description': [initial_description],
        'input_group_interaction_rate': initial_interaction,
        
    }
    new_group_features = pd.DataFrame(values, index=[0])
    return new_group_features