"""
Given the CiSA report codes within a yaml file, return a list of found techniques for each report
"""

import pandas as pd
import sys, yaml, argparse, os
sys.path.append("..")
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
CONFIG_FOLDER = os.path.join (ROOT_FOLDER, 'configs')
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/test')
from src.models.model1.recommend import get_report_data #, get_candidate_techniques
from src.data.utils import batch_save_df_to_pkl
def main():
    #### PARSING ARGS
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))
    parser.add_argument ('-config', required = True,
                         type = str, 
                         help='name of config file containing the report codes and other settings')
    
    args = parser.parse_args()
    config_filename = args.config
    if not config_filename.endswith ('.yaml'): config_filename += '.yaml'
    config_file_path = os.path.join (CONFIG_FOLDER, config_filename)
    with open (config_file_path, 'r') as config_file:
        config = yaml.safe_load (config_file)
        
    formatted_text = yaml.dump(config, default_flow_style=False, indent=2, sort_keys=False)
    print ('---Config for test data\n',formatted_text)
    # n = config ['n']
    reports = config['reports']
    # look_up_table_name = config['look_up_table']
    # look_up_table_name += '.pkl'
    ### get the look-up table
    # look_up_table = pd.read_pickle (os.path.join(ROOT_FOLDER, 'data/lookup_tables', look_up_table_name))
    
    report_data = get_report_data (reports = reports)
    # test_data = make_test_data (n = n, report_data = report_data, look_up_table= look_up_table)
    
    data = {
        'report_data': report_data,
        # 'test_data': test_data,
    }
    batch_save_df_to_pkl (file_name_dfs = data, target_path= TARGET_PATH)
if __name__ == '__main__':
    main()