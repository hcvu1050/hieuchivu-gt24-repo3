"""
last update: 2023-09-20
V2.2 Previous working version: ingestion.py.
Used to gather the data from MITRE ATT&CK. The idea is to collect all data that can potentially be used for training.
The specific data for training will be decided in downtreamm tasks. 
Extracts the following pandas DataFrames and save as csv files in "data/interim". New: also collect tactic table

1. `groups_df`
2. `groups_software_df`
3. `techniques_df`
4. `techniques_mitigations_df`
5. `techniques_detections_df`
6. `techniques_software_df`
7. `labels_df`

8. `tactics_df`
"""
from stix2 import MemoryStore
import mitreattack.attackToExcel.stixToDf as stixToDf
import pandas as pd
from . import utils
import os   

# Get the root directory of the project
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
MITRE_ATTCK_FILE_PATH = os.path.join(ROOT_FOLDER, 'data/raw', 'enterprise-attack_v14.json')
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')
PROCESS_RUNNING_MSG = "--runing {}".format(__name__)

def read_data_local(file_path = MITRE_ATTCK_FILE_PATH):
    """
    v2.2
    Reads local file 'enterprise-attack.json' from `path` (default = "data/raw").
    Returns the following DataFrames:\n
        1. `groups_df`
        2. `groups_software_df`
        3. `techniques_df`
        4. `techniques_mitigations_df`
        5. `techniques_detections_df`
        6. `techniques_software_df`
        7. `labels_df`
        8. `tactics_df`
    """
    
    attackdata = MemoryStore ()
    attackdata.load_from_file (file_path)
    
    groups_data = stixToDf.groupsToDf (attackdata)
    techniques_data = stixToDf.techniquesToDf(attackdata, "enterprise-attack")
    software_data = stixToDf.softwareToDf(attackdata)
    relationships_data = stixToDf.relationshipsToDf(attackdata)
    
    groups_df = groups_data['groups']
    groups_software_df = groups_data['associated software']
    
    techniques_df = techniques_data["techniques"]
    techniques_mitigations_df = techniques_data['associated mitigations']
    relationships_df = relationships_data['relationships']
    techniques_detections_df = relationships_df[relationships_df['mapping type'] == 'detects']
    techniques_software_df = software_data['techniques used']
    
    labels_df = groups_data['techniques used']
    
    tactics_data = stixToDf.tacticsToDf(attackdata)
    tactics_df = tactics_data['tactics']
    return groups_df, groups_software_df, techniques_df, techniques_mitigations_df, techniques_detections_df, techniques_software_df, labels_df,tactics_df

### MAIN FUNCTION ###
def collect_data(target_path = TARGET_PATH):
    """
    v2.1
    save the following DataFrames as csv in specifed path (default = "data/interim"):\n
    (file prefix: `collected_`)\n
        1.`groups_df`\n
        2.`groups_software_df`\n
        3.`techniques_df`\n
        4.`techniques_mitigations_df`\n
        5.`techniques_detections_df`\n
        6.`techniques_software_df`\n
        7.`labels_df`\n
        8.`tactics_df`\n
    """
    print (PROCESS_RUNNING_MSG)
    groups_df, groups_software_df, techniques_df, techniques_mitigations_df, techniques_detections_df, techniques_software_df, labels_df, tactics_df = read_data_local()
    
    dfs = {
        'groups_df':groups_df,
        'groups_software_df':groups_software_df,
        'techniques_df':techniques_df,
        'techniques_mitigations_df':techniques_mitigations_df,
        'techniques_detections_df':techniques_detections_df,
        'techniques_software_df':techniques_software_df,
        'labels_df':labels_df,
        'tactics_df': tactics_df
    }
    utils.batch_save_df_to_csv_with_index (dfs, target_path, prefix = 'collected_')
    return dfs
    