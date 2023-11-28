from ...constants import *
import pandas as pd
import numpy as np
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
    filtered_table = look_up_table[look_up_table['technique_ID'].isin(interacted_techniques)]
    earliest_stage = filtered_table['technique_earliest_stage'].min()
    latest_stage = filtered_table['technique_earliest_stage'].max()
    
    return (earliest_stage, latest_stage)

def get_most_similar_techniques (interacted_techniques: list,  look_up_table: pd.DataFrame(), n: int):
    """From a list of interacted techniques: Returns a list of most similar techniques. \n
    Takes n most similar techniques for each interacted techniques.
    """
    filtered_table = look_up_table[look_up_table['technique_ID'].isin(interacted_techniques)]
    # get the first n items in each list
    filtered_table.loc[:, 'sorted_similar_techniques'] = filtered_table['sorted_similar_techniques'].apply(lambda x: x[0:n])
    # filter duplicates by getting unique values
    return (list(filtered_table['sorted_similar_techniques'].explode().unique()))
        
    