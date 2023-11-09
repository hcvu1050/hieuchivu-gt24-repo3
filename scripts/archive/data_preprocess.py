import sys
import os
sys.path.append("..")
import argparse
from src.data.archived.ingestion import collect_data
from src.data.archived.cleaning import clean_data
from src.data.archived.build_features import build_features
from src.data.archived.splitting import split_data_by_group
from src.data.archived.balancing import naive_random_oversampling
from src.data.archived.aligning import align_input_to_target
from src.data.utils import batch_save_df_to_csv

TRAIN_CV_TEST_RATIO = [.7,.15, .15]

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

def main():
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))
    parser.add_argument ('--last-only','-lo', type = bool, default= True,help='Option: Do not save the tables for intermediary steps, only save the last processed tables')
    args = parser.parse_args()
    last_only = args.last_only
    # option to save tables in intermediary steps
    save_intermediary_table = not last_only
    
    collect_data()
    technique_features, group_features, interaction_matrix = clean_data(save_as_csv = save_intermediary_table)
    technique_features, group_features = build_features(technique_features_df= technique_features,
                                                        group_features_df= group_features,
                                                        save_as_csv = save_intermediary_table)
    
    train_y_df, cv_y_df, test_y_df = split_data_by_group (interaction_matrix, ratio= TRAIN_CV_TEST_RATIO, save_as_csv = save_intermediary_table)
    
    train_y_balanced_df = naive_random_oversampling (train_y_df, save_as_csv= save_intermediary_table)
    
    ### CREATING FINAL TABLES ###
    # aligining: train input
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
    
    # remove ID columns in target tables after aligning
    train_y_balanced_df = train_y_balanced_df['target']
    cv_y_df = cv_y_df['target']
    test_y_df = test_y_df['target']
    
    ### SAVING FINAL TABLES 
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
