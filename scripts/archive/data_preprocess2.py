import sys
import os
sys.path.append("..")
import argparse
import yaml
### MODULES
from src.data.ingestion2 import collect_data
from src.data.cleaning2 import clean_data
from src.data.select_features import select_features
from src.data.build_features2 import build_features_onehot
from src.data.archived.splitting import split_data_by_group
from src.data.archived.balancing import naive_random_oversampling
from src.data.archived.aligning import align_input_to_target
from src.data.utils import batch_save_df_to_csv
from src.data.constants import GROUP_ID_NAME, TECHNIQUE_ID_NAME, LABEL_NAME

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
CONFIG_FOLDER = os.path.join (ROOT_FOLDER, 'configs')
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

def main():
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))
    parser.add_argument ('-config', required= True,
                         type=str,
                         help = 'name of config file to preprocess the data')
    parser.add_argument ('--last-only','-lo', type = bool, default= True,help='Option: Do not save the tables for intermediary steps, only save the LAST processed tables')
    args = parser.parse_args()
    last_only = args.last_only
    config_file_name = args.config
    #### SETTING: option to save tables in intermediary steps
    save_intermediary_table = not last_only
    
    #### SETTING: load config file config_file_name
    if not config_file_name.endswith ('.yaml'): config_file_name += '.yaml'
    config_file_path = os.path.join (CONFIG_FOLDER, config_file_name)
    with open (config_file_path, 'r') as config_file:
        config = yaml.safe_load (config_file)
    
    train_cv_test_ratio = config['train_cv_test_ratio']
    selected_group_features = config['selected_group_features']
    selected_technique_features = config['selected_technique_features']
    
    #### COLLECT DATA
    collect_data ()
    
    
    #### CLEANING DATA / SELECTING FEATURES
    technique_features, group_features, interaction_matrix = clean_data(save_as_csv = save_intermediary_table)
    technique_features, group_features = select_features(technique_features_df= technique_features,
                                                         technique_feature_names= selected_technique_features, 
                                                         group_features_df= group_features,
                                                         group_feature_names=selected_group_features,
                                                         save_as_csv= save_intermediary_table)
    
    #### INPUT: BUILDING FEATURES 
    ## note: for now all features are one-hot encoded
    technique_features, group_features = build_features_onehot (
        technique_features_df = technique_features,
        technique_feature_names = selected_technique_features,
        group_features_df = group_features,
        group_features_names = selected_group_features,
        save_as_csv= save_intermediary_table
    )

    #### LABEL: SPLITTING
    train_y_df, cv_y_df, test_y_df = split_data_by_group (interaction_matrix, ratio= train_cv_test_ratio, save_as_csv = save_intermediary_table)
    
    #### LABEL: BALANCING TRAIN LABEL
    train_y_balanced_df = naive_random_oversampling (train_y_df, save_as_csv= save_intermediary_table)
    
    ### CREATING FINAL TABLES ###
    train_X_technique = align_input_to_target ( feature_df= technique_features,
                                                   object= 'technique',
                                                   target_df= train_y_balanced_df,
                                                   from_set = 'train', save_to_csv= save_intermediary_table)
    train_X_group = align_input_to_target ( feature_df= group_features,
                                                   object= 'group',
                                                   target_df= train_y_balanced_df,
                                                   from_set = 'train', save_to_csv= save_intermediary_table)
    
    # aligning: cv input
    cv_X_technique = align_input_to_target (feature_df= technique_features,
                                                  object= 'technique',
                                                  target_df = cv_y_df,
                                                  from_set= 'cv', save_to_csv= save_intermediary_table)
    cv_X_group = align_input_to_target (feature_df= group_features,
                                                  object= 'group',
                                                  target_df = cv_y_df,
                                                  from_set = 'cv', save_to_csv=save_intermediary_table)
    
    # aligning: test input
    test_X_technique = align_input_to_target (feature_df= technique_features,
                                                  object= 'technique',
                                                  target_df=test_y_df,
                                                  from_set= 'test', save_to_csv= save_intermediary_table)
    test_X_input = align_input_to_target (feature_df= group_features,
                                                  object= 'group',
                                                  target_df=test_y_df,
                                                  from_set= 'test', save_to_csv= save_intermediary_table)
    
    # remove ID columns in label tables after aligning, only keep label columns
    train_y_balanced_df = train_y_balanced_df[LABEL_NAME]
    cv_y_df = cv_y_df[LABEL_NAME]
    test_y_df = test_y_df[LABEL_NAME]
    
    #### SAVING FINAL TABLES
    dfs = {
        'train_y_balanced':             train_y_balanced_df,
        'train_X_technique_aligned':    train_X_technique,
        'train_X_group_aligned':        train_X_group,
        'cv_y':                         cv_y_df,
        'cv_X_technique_aligned':       cv_X_technique,
        'cv_X_group_aligned':           cv_X_group,
        'test_y':                       test_y_df,
        'test_X_technique_aligned':     test_X_technique,
        'test_X_group_aligned':         test_X_input,
    }
    batch_save_df_to_csv (file_name_dfs= dfs, target_path=TARGET_PATH, prefix = 'FINAL', output_list_file = 'FINAL')
    
if __name__ == '__main__':
    main()