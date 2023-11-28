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