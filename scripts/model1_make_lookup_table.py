import sys, os, argparse
import pandas as pd
import tensorflow as tf
from tensorflow import keras
sys.path.append("..")
from src.models.model1.recommend import extract_technique_branch, build_technique_dataset, make_look_up_table, get_technique_earliest_tatic_stage
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
TRAINED_MODELS_FOLDER = os.path.join (ROOT_FOLDER, 'trained_models', 'model1')
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/lookup_tables')

DEFAULT_TECHNIQUE_TABLE_NAME = 'processed_technnique_features.pkl'

def main():
    #### PARSING ARGS
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))
    parser.add_argument ('-model_name', required = True,
                         type = str, 
                         help='name of the trained model')
    
    parser.add_argument ('-technique_feature_table', required = False,
                         type = str, 
                         help='(opt) name of the table (pkl file) that stores the processed technique features. Default: processed_technnique_features.pkl')
    
    args = parser.parse_args()
    technique_table_name = DEFAULT_TECHNIQUE_TABLE_NAME
    model_name = args.model_name
    if args.technique_feature_table is not None: technique_table_name = args.technique_feature_table
    
    ### load then extract technique branch
    submodel = extract_technique_branch (model_name=model_name)
    ### load then build preprocessed technique feature dataset
    technique_features_df = pd.read_pickle (os.path.join (
        ROOT_FOLDER,
        'data/processed/model1',
        technique_table_name
    ))
    id_list = list (technique_features_df['technique_ID'])
    technique_features_dataset = build_technique_dataset (technique_features_df)

    ### get learned technique features, then create lookup table
    technique_features_dataset = technique_features_dataset.batch(32)
    learned_features = submodel.predict (technique_features_dataset)
    look_up_table = make_look_up_table (learned_features, id_list)
    
    ### add earliest tactic stage to look-up table
    tactics_order= pd.read_csv (os.path.join (ROOT_FOLDER, 'data/raw/tactics_order.csv'), index_col= 0)
    technique_earliest_tatic_stage = get_technique_earliest_tatic_stage (technique_features_df[['technique_ID', 'input_technique_tactics']], tactics_order_df=tactics_order)
    look_up_table = pd.merge(
        left = look_up_table, 
        right = technique_earliest_tatic_stage,
        on = 'technique_ID', how = 'left'
    )
    
    look_up_table.to_pickle (os.path.join (
        TARGET_PATH, model_name + '.pkl'
    ))
    return

if __name__ == '__main__':
    main()