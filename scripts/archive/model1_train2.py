"""
last update: 2023-10-05
- Usage: Train a single instance of model1 (script version 2). 
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

from src.models.model1.dataloader import load_train_datasets
from src.models.model1.model_v0_4 import Model1
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
    formatted_text = yaml.dump(config, default_flow_style=False, indent=2, sort_keys=False)
    print ('---config for Model1\n',formatted_text)

    #### LOAD DATASETS, THEN CONFIG DATASETS
    train_dataset, train_cv_dataset, feature_info = load_train_datasets()
    
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    train_cv_dataset = train_cv_dataset.batch(32)

    #### LOAD/COMPILE MODEL THEN TRAIN MODEL
    model = Model1 (input_sizes= feature_info,
                config=model_architecture_config)    
    
    optimizer = keras.optimizers.Adam (learning_rate= learning_rate)    
    loss = keras.losses.BinaryCrossentropy (from_logits= True)
    model.compile (optimizer, loss = loss, metrics = [tf.keras.metrics.AUC(curve = 'PR', from_logits= True)])
    
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
    
    
