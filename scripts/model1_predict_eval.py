import sys, os, argparse, yaml, time
sys.path.append("..")
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset
from src.data.utils import batch_save_df_to_pkl
from src.models.model1.model_preprocess import build_dataset_3
from src.models.model1.predict import make_test_data, build_detected_group_profile, get_metrics

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
CONFIG_FOLDER = os.path.join (ROOT_FOLDER, 'configs')
PROCESSED_DATA_FOLDER = os.path.join (ROOT_FOLDER, 'data/processed/model1')
TRAINED_MODEL_FOLDER  = os.path.join (ROOT_FOLDER, 'trained_models/model1')
LOOKUP_TABLE_FOLDER = os.path.join (ROOT_FOLDER, 'data/lookup_tables')
CISA_REPORT_FOLDER = os.path.join (ROOT_FOLDER, 'data/test')
REPORT_FOLDER = os.path.join (ROOT_FOLDER, 'reports/model1/results_eval')
def main():
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))
    parser.add_argument ('-config', required = True,
                         type = str, 
                         help='name of config file to build and train the model')
    args = parser.parse_args()
    config_filename = args.config    
    
    #### 👉LOAD CONFIGS FROM CONFIG FILE
    if not config_filename.endswith ('.yaml'): config_filename += '.yaml'
    config_file_path = os.path.join (CONFIG_FOLDER,config_filename)
    with open (config_file_path, 'r') as config_file:
        config = yaml.safe_load (config_file)
    
    report_file_name = config['report_file_name']
    model_name = config['model_name']
    group_profile_update_threshold = config['group_profile_update_threshold']
    initial_group_profile_settings = config['initial_group_profile_settings']
    
    group_features = pd.read_pickle (os.path.join(PROCESSED_DATA_FOLDER,'processed_group_features.pkl'))
    processed_technique_features = pd.read_pickle (os.path.join(PROCESSED_DATA_FOLDER,'processed_technnique_features.pkl'))
    train_labels =pd.read_pickle (os.path.join(PROCESSED_DATA_FOLDER,'processed_train_labels.pkl'))
    cv_dataset = Dataset.load (os.path.join(PROCESSED_DATA_FOLDER,'cv_dataset'))
    
    report_data = pd.read_pickle (os.path.join (CISA_REPORT_FOLDER, r'{report_file_name}.pkl'.format(report_file_name = report_file_name)))
    look_up_table = pd.read_pickle (os.path.join (LOOKUP_TABLE_FOLDER, r'{model_name}.pkl'.format(model_name = model_name)))
    
    model = keras.models.load_model (os.path.join (TRAINED_MODEL_FOLDER, model_name))
    metrics = get_metrics (model, cv_dataset)
    prediction_threshold = metrics['best_threshold']
    
    test_data = make_test_data (report_data= report_data, look_up_table=look_up_table)
    test_data_with_preds = test_data.copy()
    test_data_with_preds['predicted_techniques'] = None
    
    # 👉 TRAINING LOOP
        
    for index, row in test_data_with_preds.iterrows():
        detected_techniques = row['detected_techniques']
        candidate_techniques = row['candidate_techniques']
        group_ID = row['group_ID']
        candidate_technique_features = processed_technique_features[processed_technique_features['technique_ID'].isin(candidate_techniques)]
        # build group profile based on detected tecniques
        detected_group_profile = build_detected_group_profile (processed_group_features= group_features,
                                        processed_technique_features = processed_technique_features ,
                                        detected_techniques= detected_techniques, threshold=group_profile_update_threshold,train_labels= train_labels,group_id= group_ID, settings=initial_group_profile_settings)
        aligned_group_profile = pd.concat ([detected_group_profile] * len(candidate_techniques), ignore_index= True)

        blank_labels = pd.DataFrame({'label': [-1]* len(candidate_techniques)})

        # make dataset for current group profile and candidate techniques
        test_dataset = build_dataset_3 (X_group_df= aligned_group_profile, X_technique_df= candidate_technique_features, y_df= blank_labels,
                                        selected_ragged_group_features = [f for f in detected_group_profile.columns if f not in ('group_ID', 'input_group_interaction_rate', 'input_group_description')],
                                        selected_ragged_technique_features = [f for f in candidate_technique_features if f not in ('technique_ID', 'input_technique_description', 'input_technique_interaction_rate')])
        test_dataset = test_dataset.batch(32)
        test_dataset.batch(32)
        results = []
        # model makes prediction
        # if the final prediction results in an empty list, keep decreasing the threshold
        current_prediction_threshold = prediction_threshold
        while len(results) == 0:
            results_logit = model.predict(test_dataset,verbose=0)
            results_prob = tf.keras.activations.sigmoid(results_logit)
            results_binary = np.where(results_prob >= current_prediction_threshold, 1, 0)
            results_binary = results_binary.flatten().tolist()
            # convert binary prediction to technique names
            results = [technique for binary_val, technique in zip (results_binary, candidate_techniques) if binary_val == 1.0]
            current_prediction_threshold *= 0.99
        
        test_data_with_preds.at[index, 'predicted_techniques'] = results
        
    test_data_with_preds['accuracy'] = None
    test_data_with_preds['precision'] = None
    test_data_with_preds['recall'] = None
    for index, row in test_data_with_preds.iterrows():
        true_values = row['true_subsequent_techniques']
        predicted_values = row['predicted_techniques']
        correct_predictions = [1 if val in true_values else 0 for val in predicted_values]
        
        accuracy = sum(correct_predictions) / len(predicted_values)
        true_positives = sum([1 for val in predicted_values if val in true_values])
        false_positives = len(predicted_values) - true_positives
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / len(true_values) if len(true_values) > 0 else 0  
        test_data_with_preds.at[index,'accuracy'] = accuracy
        test_data_with_preds.at[index,'precision'] = precision
        test_data_with_preds.at[index,'recall'] = recall
        
    dfs = {
        'test_results_eval': test_data_with_preds,
        'test_data': test_data
    }
    batch_save_df_to_pkl (dfs,REPORT_FOLDER,prefix= r'{model_name}_{report_file_name}'.format(model_name = model_name, report_file_name = report_file_name) )
    return 
if __name__ == '__main__':
    main()