"""
last update: 2023-10-05
- Usage: Train multiple instances of model1 with different hyperparameters
- Args: 
    - `configs`: name of the subfolder in `configs/` that contains the `yaml` files for different instances (each instance is defined by a set of hyperparameters) of model1
"""
import time
import sys
import os
import yaml
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import argparse

sys.path.append("..")

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
CONFIG_FOLDER = os.path.join (ROOT_FOLDER, 'configs')
REPORT_FOLDER = os.path.join (ROOT_FOLDER, 'reports', 'model1')
TRAINED_MODELS_FOLDER = os.path.join (ROOT_FOLDER, 'trained_models')

from src.models.model1.dataloader import load_train_datasets
from src.models.model1.model_v0_4 import Model1

def main():
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))

    parser.add_argument ('-configs', type = str,
                         required= True,
                         help = 'name of the folder in `configs/` that stores the config files')
    args = parser.parse_args()
    config_sub_folder = args.configs
    
    filenames = os.listdir(os.path.join (CONFIG_FOLDER, config_sub_folder))
    print (filenames)

    ### BATCH CONFIG AND TRAIN MODELS
    for config_file_name in filenames:
        train_from_config (config_file_name, target_folder_name =config_sub_folder)

def train_from_config (config_filename: str, target_folder_name: str):
    
    config_file_path = os.path.join (CONFIG_FOLDER, target_folder_name, config_filename)
    with open (config_file_path, 'r') as config_file:
        config = yaml.safe_load (config_file)
    
    formatted_text = yaml.dump(config, default_flow_style=False, indent=2, sort_keys=False)
    print ('---config for Model1\n',formatted_text)
    
    model_architecture_config = config['model_architecture']
    
    train_config = config['train']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    learning_rate = float (train_config['learning_rate'])

    #### LOAD DATASETS, THEN CONFIG DATASETS
    train_dataset, train_cv_dataset, feature_info  = load_train_datasets ()
    ## Config Datasets
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    train_cv_dataset = train_cv_dataset.batch(32)
    
    #### LOAD/COMPILE MODEL THEN TRAIN MODEL
    model = Model1 (input_sizes= feature_info,
                config=model_architecture_config)    
    
    optimizer = keras.optimizers.Adam (learning_rate= learning_rate)    
    loss = keras.losses.BinaryCrossentropy (from_logits= True)
    model.compile (optimizer, 
                   loss = loss, 
                   metrics = [tf.keras.metrics.AUC(curve = 'PR', from_logits= True, name = 'auc-pr')])
    
    #### TRAIN MODEL
    start_time = time.time()
    history = model.fit (
        train_dataset,
        validation_data= train_cv_dataset,
        epochs=epochs
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int (elapsed_time  % 60)
    print(f"Training completed in {elapsed_minutes} minutes and {elapsed_seconds} seconds")
    
    #### SAVE HISTORY
    base_config_filename = config_filename.split(".")[0]
    
    history_df = pd.DataFrame(history.history)
    report_file_name = 'train_loss_{config_file}.csv'.format(config_file = base_config_filename)
    
    # CREATE THE FOLDER
    folder_path = os.path.join (REPORT_FOLDER, 'train_loss', target_folder_name)
    if not os.path.exists (folder_path):
        os.makedirs (folder_path)
    file_path = os.path.join (folder_path, report_file_name)
    history_df.to_csv(file_path, index=False)
    
    #### SAVE TRAINED MODEL
    model_file_name = base_config_filename
    if not os.path.exists (os.path.join (TRAINED_MODELS_FOLDER, target_folder_name)):
        os.makedirs (os.path.join (TRAINED_MODELS_FOLDER, target_folder_name))
    model_file_path = os.path.join (TRAINED_MODELS_FOLDER, target_folder_name, model_file_name)
    model.save (model_file_path)
        
if __name__ == '__main__':
    main()