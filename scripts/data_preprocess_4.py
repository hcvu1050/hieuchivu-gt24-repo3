"""
- `data_preprocess_4`
	- Config(s): `data_pp5` 
		- include unused techniques
	1. Collect the data
	2. Clean the data with all candidate features
	3. Embed description sentences in Group and Technique feature
	4. Exports: (1)Group features, (2)Technique features, (3)Interaction matrix
"""

import sys, os, argparse, yaml
from transformers import BertTokenizer, TFBertModel
sys.path.append("..")
### MODULES
from src.data.utils import  batch_save_df_to_pkl
from src.data.ingestion_2 import collect_data
from src.data.cleaning_4 import clean_data
from src.data.build_features_3 import build_feature_sentence_embed

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
CONFIG_FOLDER = os.path.join (ROOT_FOLDER, 'configs')
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

def main():
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))
    parser.add_argument ('-config', required= True,
                         type=str,
                         help = 'name of config file to preprocess the data')
    parser.add_argument ('--last-only','-lo', choices= ['True', 'False'], required= True,
                         help='Option: Do not save the tables for intermediary steps, only save the LAST processed tables')
    args = parser.parse_args()
    last_only = args.last_only
    if last_only == "True": last_only = True
    elif last_only == "False": last_only = False
    config_file_name = args.config
    #### SETTING: option to save tables in intermediary steps
    save_intermediary_table = not last_only
    
    #### SETTING: load config file 
    if not config_file_name.endswith ('.yaml'): config_file_name += '.yaml'
    config_file_path = os.path.join (CONFIG_FOLDER, config_file_name)
    with open (config_file_path, 'r') as config_file:
        config = yaml.safe_load (config_file)
        
    formatted_text = yaml.dump(config, default_flow_style=False, indent=2, sort_keys=False)
    print ('---Config for data preprocessing\n',formatted_text)
    include_unused_techniques = config['include_unused_techniques']
    
    #### CLEANING DATA / SELECTING FEATURES    
    collect_data ()
    technique_features, group_features, interaction_matrix = clean_data(
        include_unused_techniques = include_unused_techniques, 
        save_as_csv = save_intermediary_table)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    embed_model = TFBertModel.from_pretrained('bert-base-uncased')
    group_features = build_feature_sentence_embed (group_features, 'input_group_description', tokenizer, embed_model)
    technique_features = build_feature_sentence_embed (technique_features, 'input_technique_description', tokenizer, embed_model)
    
    # #### LAST STEPS (save the output tables as pkl)
    
    dfs ={
        'y_cleaned': interaction_matrix,
        'X_group': group_features,
        'X_technique': technique_features,
    }
    batch_save_df_to_pkl (file_name_dfs= dfs, target_path=TARGET_PATH, output_list_file = 'PREPROCESSED')
    print ('---Shapes:')
    for df in dfs.keys():
        print ('{df}: {shape}'.format(df = df, shape = dfs[df].shape))
    
if __name__ == '__main__':
    main()