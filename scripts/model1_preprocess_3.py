"""
- Model1 preprocessing steps (srcript version 3)
- Usage: data preprocess pipeline specific to model 1. ❗Only works after running `data_preprocess_4`. Steps: 
    1. Load the data exported by running `data_preprocess_4`, including (1)Group feature, (2)Technique feature, and (3)Interaction matrix
    2. Split the data into train and cv set with ratios defined by a yaml file in `configs/folder`
    3. Select features for Group and Technique.
        1. Engineer some additional features for Group and Technique
        2. (opt) Limit feature cardinality
    4. Make vocab for the string-valued features (to be used later in downstream tasks)
    5. (opt) Re-sampling train and train-cv data
    6. Align features to labels
    7. Create TensorFlow Datasets for train and train-cv sets.
    8. Save the preprocessed data to `data/preprocessed/model1`
- Args: 
    - `config`: name of the `yaml` file in `configs/` that will be used to define the ratios for splitting the data sets.
    - `-lo`  (means 'last only', defaul = `True`): optional argument to save the intermediary data while going through the preprocessing steps.
"""
import sys, os, yaml, argparse
import pandas as pd
sys.path.append("..")

from src.models.model1.model_preprocess import get_data, split_by_group, label_resample, align_input_to_labels, build_dataset_3, save_dataset
from src.data.select_features import select_features
from src.data.cleaning_4 import _limit_samples_based_on_earliest_stage, _limit_samples_based_on_group_interaction
from src.data.limit_cardinality import batch_reduce_vals_based_on_nth_most_frequent, batch_reduce_vals_based_on_percentage
from src.data.build_features_3 import build_feature_interaction_frequency, build_feature_used_tactics
from src.data.make_vocab import make_vocab
from src.constants import *
from src.data.utils import batch_save_df_to_csv, batch_save_df_to_pkl

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
CONFIG_FOLDER = os.path.join (ROOT_FOLDER, 'configs')
TARGET_PATH = os.path.join (ROOT_FOLDER, 'data/processed/model1')

def main():
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))
    parser.add_argument ('-config', required = True,
                         type = str, 
                         help='name of config file preprocess data for model1')
    
    parser.add_argument ('--last-only','-lo', required=True, choices= ['True', 'False'],
                         help='Option to not save the tables for intermediary steps, only save the last processed tables value is either "True" or "False"')
    
    args = parser.parse_args()
    config_file_name = args.config
    last_only = args.last_only
    
    if last_only == "True": last_only = True
    elif last_only == "False": last_only = False
    save_intermediary_table = not last_only
    
    if not os.path.exists(TARGET_PATH):
        os.makedirs(TARGET_PATH)
        
    #### 👉LOAD CONFIGS FROM CONFIG FILE
    if not config_file_name.endswith ('.yaml'): config_file_name += '.yaml'
    config_file_path = os.path.join (CONFIG_FOLDER,config_file_name)
    with open (config_file_path, 'r') as config_file:
        config = yaml.safe_load (config_file)
    formatted_text = yaml.dump(config, default_flow_style=False, indent=2, sort_keys=False)
    print ('---config for model 1 preprocessing:\n',formatted_text)
    
    ## READING CONFIG
    limit_samples_based_on_earliest_stage = config ['limit_samples_based_on_earliest_stage']
    limit_samples_based_on_group_interaction = config ['limit_samples_based_on_group_interaction']
    limit_cardinality = config['limit_cardinality']
    
    data_split = config['data_split']
    selected_group_features = config['selected_group_features']
    selected_technique_features = config['selected_technique_features']
    limit_technique_features = config['limit_technique_features']
    limit_group_features = config['limit_group_features']
    resampling = config['resampling']
    train_size, train_cv_size, cv_size, test_size = data_split
    
    #### 👉1- LOAD DATA
    group_features_df, technique_features_df, labels_df = get_data(data_type = 'pkl')
    tactics_order = pd.read_csv ('../data/raw/tactics_order.csv', index_col=0)
    labels_df.to_pickle ('tmp_m1_pp_label_org.pkl')
    #### 👉1b - (OPT) LIMIT SAMPLES
    if limit_samples_based_on_earliest_stage:
        labels_df = _limit_samples_based_on_earliest_stage (
            technique_tactics_df = technique_features_df[[TECHNIQUE_ID_NAME, INPUT_TECHNIQUE_TACTICS]],
            tactics_order_df= tactics_order,
            labels_df= labels_df
        )
    
    if limit_samples_based_on_group_interaction is not None: 
        labels_df = _limit_samples_based_on_group_interaction (
            labels_df= labels_df, min_instances= limit_samples_based_on_group_interaction
        )
    labels_df.to_pickle ('tmp_m1pp_label.pkl')
    
    #### 👉2- SPLIT LABELS
    print ('--splitting data')
    train_y_df, remain_y_df  = split_by_group (labels_df, ratio = train_size)
    train_cv_y_df, remain_y_df = split_by_group (remain_y_df, 
                                                 ratio = train_cv_size/ (train_cv_size + cv_size + test_size))
    cv_y_df, test_y_df = split_by_group (remain_y_df, 
                                          ratio = cv_size/(cv_size + test_size))
    if save_intermediary_table:
        dfs = {
        'train_y': train_y_df,
            'train_cv_y': train_cv_y_df,
            'cv_y': cv_y_df,
            'test_y': test_y_df
        }
        batch_save_df_to_csv (dfs, TARGET_PATH, postfix= 'split')
    
    #### 👉3- SELECT FEATURES
    technique_features_df, group_features_df = select_features(technique_features_df= technique_features_df,
                                                         technique_feature_names= selected_technique_features, 
                                                         group_features_df= group_features_df,
                                                         group_feature_names=selected_group_features,
                                                         save_as_csv= save_intermediary_table)        
    ### export for unit test
    technique_features_df.to_pickle ('tmp_m1pp_technique_org.pkl')
    group_features_df.to_pickle ('tmp_m1pp_group_org.pkl')
    
    #### 👉3b- limit feature cardinality
    if limit_cardinality is not None: 
        if limit_cardinality == 'n_most_frequent':
            if limit_technique_features is not None:
                technique_features_df = batch_reduce_vals_based_on_nth_most_frequent (technique_features_df, setting = limit_technique_features)
            if limit_group_features is not None:
                group_features_df = batch_reduce_vals_based_on_nth_most_frequent (group_features_df, setting = limit_group_features)
        elif limit_cardinality == 'percentage':
            if limit_technique_features is not None:
                technique_features_df = batch_reduce_vals_based_on_percentage (technique_features_df, setting = limit_technique_features)
            if limit_group_features is not None:
                group_features_df = batch_reduce_vals_based_on_percentage (group_features_df, setting = limit_group_features)
    
    #### 👉 3c- Build addtional features 
    technique_features_df = build_feature_interaction_frequency (label_df= train_y_df, feature_df= technique_features_df, object_ID= 'technique_ID', feature_name = 'input_technique_interaction_rate')
    group_features_df = build_feature_interaction_frequency (label_df= train_y_df, feature_df= group_features_df, object_ID= 'group_ID', feature_name = 'input_group_interaction_rate')
    #### ❗extra ragged feature, will be added to selected_group_features
    group_features_df = build_feature_used_tactics (label_df= train_y_df, group_df= group_features_df, technique_df= technique_features_df, feature_name= 'input_group_tactics')
    selected_group_features = selected_group_features + ['input_group_tactics']
    
    ### export for unit test
    technique_features_df.to_pickle ('tmp_m1pp_technique.pkl')
    group_features_df.to_pickle ('tmp_m1pp_group.pkl')
    
    
    ### export for recommendation phase
    dfs = {
        'technnique_features': technique_features_df,
        'group_features': group_features_df
    }
    batch_save_df_to_pkl (dfs, TARGET_PATH, prefix = 'processed')
    #### 👉Make vocab
    make_vocab(group_features_df, [feature for feature in selected_group_features if feature not in ['input_group_description']], path = TARGET_PATH)
    make_vocab (technique_features_df, [feature for feature in selected_technique_features if feature not in ['input_technique_description']], path = TARGET_PATH)
    
    #### - (OPTIONAL) OVERSAMPLING train and train_cv, if train_cv size is set to 0, return an empty dataframe
    if resampling is not None: 
        print ('--resampling data')
        
        train_y_df = label_resample (train_y_df, sampling_strategy= resampling)
        if train_cv_size != 0:
            train_cv_y_df = label_resample (train_cv_y_df, sampling_strategy= resampling)
    if save_intermediary_table:
        dfs = {
            'train_y': train_y_df,
            'train_cv_y': train_cv_y_df,
        }
        batch_save_df_to_csv (dfs, TARGET_PATH, postfix='resampled')
    
    #### 4- ALIGNING features to labels
    ## train set
    print ('--aligning data')
    train_X_group_df = align_input_to_labels (group_features_df, 
                                              object= 'group', 
                                              label_df= train_y_df)
    train_X_technique_df = align_input_to_labels (technique_features_df, 
                                                  object= 'technique', 
                                                  label_df= train_y_df)
    # train_cv set
    train_cv_X_group_df = pd.DataFrame()
    train_cv_X_technique_df = pd.DataFrame()
    if train_cv_size != 0:
        train_cv_X_group_df = align_input_to_labels (group_features_df, 
                                                object= 'group', 
                                                label_df= train_cv_y_df)
        train_cv_X_technique_df = align_input_to_labels (technique_features_df, 
                                                    object= 'technique', 
                                                    label_df= train_cv_y_df)
    # cv set
    cv_X_group_df = align_input_to_labels (group_features_df, 
                                           object= 'group', 
                                           label_df= cv_y_df)
    cv_X_technique_df = align_input_to_labels (technique_features_df, 
                                               object= 'technique', 
                                               label_df= cv_y_df)
    # test set
    if test_size != 0:
        test_X_group_df = align_input_to_labels (group_features_df, 
                                                object= 'group', 
                                                label_df= test_y_df)
        test_X_technique_df = align_input_to_labels (technique_features_df, 
                                                object= 'technique', 
                                                label_df= test_y_df)


    #### 5- Make tensor flow datasets
    ### ❗different from model1_preprocess: build_dataset_3
    print ('--building datasets')
    
    train_dataset = build_dataset_3(X_group_df =      train_X_group_df, 
                                    X_technique_df =  train_X_technique_df,
                                    selected_ragged_group_features= selected_group_features,
                                    selected_ragged_technique_features = selected_technique_features,
                                    y_df =            train_y_df)
    
    if train_cv_size != 0:
        train_cv_dataset = build_dataset_3(X_group_df =    train_cv_X_group_df, 
                                           X_technique_df = train_cv_X_technique_df,
                                           selected_ragged_group_features = selected_group_features,
                                           selected_ragged_technique_features = selected_technique_features,
                                           y_df=           train_cv_y_df)
        
    cv_dataset = build_dataset_3(X_group_df =         cv_X_group_df, 
                                 X_technique_df =  cv_X_technique_df,
                                 selected_ragged_group_features= selected_group_features,
                                 selected_ragged_technique_features = selected_technique_features,
                                 y_df =            cv_y_df)
    if test_size != 0:
        test_dataset = build_dataset_3(X_group_df =       test_X_group_df, 
                                    X_technique_df =  test_X_technique_df,
                                    selected_ragged_group_features= selected_group_features,
                                    selected_ragged_technique_features = selected_technique_features,
                                    y_df =            test_y_df)
    
    
    save_dataset (train_dataset, TARGET_PATH, TRAIN_DATASET_FILENAME)
    if train_cv_size !=0:
        save_dataset (train_cv_dataset, TARGET_PATH, TRAIN_CV_DATASET_FILENAME)
    save_dataset (cv_dataset, TARGET_PATH, CV_DATASET_FILENAME)
    if test_size != 0: 
        save_dataset (test_dataset, TARGET_PATH, TEST_DATASET_FILENAME)
if __name__ == '__main__':
    main()