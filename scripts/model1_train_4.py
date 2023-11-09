"""
last update: 2023-10-29
- Usage: Train a single instance of model1 (script version 4). Difference from version 3: 
    - Dataset containin ragged tensors
    - Model version 0.5
- Args: 
    - `-config`:  name of the `yaml` file in `configs/model1/single_train` that will be used to define the hyperparameters for model1
"""

import sys, os, argparse, yaml, time
import pandas as pd
import tensorflow as tf
from tensorflow import keras

sys.path.append("..")
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
CONFIG_FOLDER = os.path.join (ROOT_FOLDER, 'configs')
### folder for config of single model train
SINGLE_TRAIN_FOLDER_NAME = 'model1_single_train'
REPORT_FOLDER = os.path.join (ROOT_FOLDER, 'reports', 'model1')
TRAINED_MODELS_FOLDER = os.path.join (ROOT_FOLDER, 'trained_models', 'model1')

from src.models.model1.dataloader import load_datasets
from src.models.model1.model_v0_5 import Model1

from src.models.model1.model_preprocess import  align_input_to_labels, build_dataset_2
def main():
    #### PARSING ARGS
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))
    parser.add_argument ('-config', required = True,
                         type = str, 
                         help='name of config file to build and train the model')
    
    args = parser.parse_args()
    config_filename = args.config

    #### LOAD CONFIGS FROM CONFIG FILE
    if not config_filename.endswith ('.yaml'): config_filename += '.yaml'
    config_file_path = os.path.join (CONFIG_FOLDER, SINGLE_TRAIN_FOLDER_NAME,config_filename)
    with open (config_file_path, 'r') as config_file:
        config = yaml.safe_load (config_file)
        
    model_architecture_config = config['model_architecture']
    train_config = config['train']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    learning_rate = float (train_config['learning_rate'])
    class_weights = train_config['class_weights']
    formatted_text = yaml.dump(config, default_flow_style=False, indent=2, sort_keys=False)
    print ('---config for Model1\n',formatted_text)

    #### 👉LOAD VOCABS
    input_group_software_id = pd.read_csv('../data/interim/input_group_software_id_vocab.csv', header = None)
    input_technique_data_sources = pd.read_csv('../data/interim/input_technique_data_sources_vocab.csv', header= None)
    input_technique_defenses_bypassed = pd.read_csv('../data/interim/input_technique_defenses_bypassed_vocab.csv', header= None)
    input_technique_detection_name = pd.read_csv('../data/interim/input_technique_detection_name_vocab.csv', header= None)
    input_technique_mitigation_id = pd.read_csv('../data/interim/input_technique_mitigation_id_vocab.csv', header= None)
    input_technique_permissions_required = pd.read_csv('../data/interim/input_technique_permissions_required_vocab.csv', header= None)
    input_technique_platforms = pd.read_csv('../data/interim/input_technique_platforms_vocab.csv', header= None)
    input_technique_software_id = pd.read_csv('../data/interim/input_technique_software_id_vocab.csv', header= None)
    input_technique_tactics = pd.read_csv('../data/interim/input_technique_tactics_vocab.csv', header= None)
    
    vocabs = {
    'input_group_software_id' : input_group_software_id[0].dropna().values,
    'input_technique_data_sources' : input_technique_data_sources[0].dropna().values,
    'input_technique_defenses_bypassed' : input_technique_defenses_bypassed[0].dropna().values,
    'input_technique_detection_name' : input_technique_detection_name[0].dropna().values,
    'input_technique_mitigation_id' : input_technique_mitigation_id[0].dropna().values,
    'input_technique_permissions_required' : input_technique_permissions_required[0].dropna().values,
    'input_technique_platforms' : input_technique_platforms[0].dropna().values,
    'input_technique_software_id' : input_technique_software_id[0].dropna().values,
    'input_technique_tactics' : input_technique_tactics[0].dropna().values    
    }
    #### 👉LOAD DATASETS, THEN CONFIG DATASETS
    train_dataset, cv_dataset, test_dataset  = load_datasets(empty_train_cv= True, return_feature_info=False)
    
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    cv_dataset = cv_dataset.batch(32)
    
    #### 👉 LOAD/COMPILE MODEL THEN TRAIN MODEL
    model = Model1 (config=model_architecture_config, vocabs=vocabs)    
    
    optimizer = keras.optimizers.Adam (learning_rate= learning_rate)    
    # # ❗
    # loss = keras.losses.BinaryCrossentropy (from_logits= True)
    loss = keras.losses.BinaryFocalCrossentropy (from_logits= True, 
                                                 apply_class_balancing= False,
                                                 alpha = 0.15, 
                                                 gamma = 2.5 )
    model.compile (optimizer, 
                   loss = loss, 
                   metrics = [tf.keras.metrics.AUC(curve = 'PR', from_logits= True, name = 'auc-pr')],
                   )
    
    ### TRAIN MODEL 
    start_time = time.time()
    history = model.fit (
        train_dataset,
        validation_data= cv_dataset,
        epochs=epochs,
        class_weight=class_weights
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int (elapsed_time % 60)
    print(f"Training completed in {elapsed_minutes} minutes and {elapsed_seconds} seconds")
    
    #### SAVE HISTORY
    history_df = pd.DataFrame(history.history)
    file_name = '{config_file}.csv'.format(config_file = args.config)
    if not os.path.exists(os.path.join (REPORT_FOLDER, 'train_loss', SINGLE_TRAIN_FOLDER_NAME)):
        os.makedirs(os.path.join (REPORT_FOLDER, 'train_loss', SINGLE_TRAIN_FOLDER_NAME))
    
    file_path = os.path.join (REPORT_FOLDER, 'train_loss', SINGLE_TRAIN_FOLDER_NAME,file_name)
    history_df.to_csv(file_path, index=False)
    
    ### SAVE TRAINED MODEL
    
    base_config_filename = config_filename.split(".")[0]
    model_file_name = base_config_filename
    if not os.path.exists(TRAINED_MODELS_FOLDER):
        os.makedirs(TRAINED_MODELS_FOLDER)
    model_file_path = os.path.join (TRAINED_MODELS_FOLDER, model_file_name)
    model.save (model_file_path)
    
if __name__ == '__main__':
    main()
    
    
