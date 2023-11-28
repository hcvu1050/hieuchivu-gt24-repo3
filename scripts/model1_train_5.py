"""
last update: 2023-10-29
- Usage: Train a single instance of model1 (script version 5).
    - Model version 0.6.c
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
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/processed/model1')
from src.models.model1.dataloader import load_datasets_2
from src.models.model1.model_v0_6c import Model1

def main():
    #### PARSING ARGS
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))
    parser.add_argument ('-config', required = True,
                         type = str, 
                         help='name of config file to build and train the model')
    
    args = parser.parse_args()
    config_filename = args.config

    #### 👉LOAD CONFIGS FROM CONFIG FILE
    if not config_filename.endswith ('.yaml'): config_filename += '.yaml'
    config_file_path = os.path.join (CONFIG_FOLDER, SINGLE_TRAIN_FOLDER_NAME,config_filename)
    with open (config_file_path, 'r') as config_file:
        config = yaml.safe_load (config_file)
    
    ## Model config
    model_architecture_config = config['model_architecture']
    
    if model_architecture_config['limit_technique_features'] is None:
        model_architecture_config['limit_technique_features'] = dict()
        model_architecture_config['limit_technique_features']['input_technique_data_sources'] = None
        model_architecture_config['limit_technique_features']['input_technique_defenses_bypassed'] = None
        model_architecture_config['limit_technique_features']['input_technique_detection_name'] = None
        model_architecture_config['limit_technique_features']['input_technique_mitigation_id'] = None
        model_architecture_config['limit_technique_features']['input_technique_permissions_required'] = None
        model_architecture_config['limit_technique_features']['input_technique_platforms'] = None
        model_architecture_config['limit_technique_features']['input_tactics'] = None
        model_architecture_config['limit_technique_features']['input_software_id'] = None
    
    if model_architecture_config['limit_group_features'] is None:
        model_architecture_config['limit_group_features'] = dict()
        model_architecture_config['limit_group_features']['input_software_id'] = None
        model_architecture_config['limit_group_features']['input_tactics'] = None
    
    ## Train config
    train_config = config['train']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    patience = train_config['patience']
    start_from_epoch = train_config['start_from_epoch']
    learning_rate = float (train_config['learning_rate'])
    class_weights = train_config['class_weights']
    
    formatted_text = yaml.dump(config, default_flow_style=False, indent=2, sort_keys=False)
    print ('---config for Model1\n',formatted_text)

    #### 👉LOAD VOCABS
    group_software_id_vocab = pd.read_csv('../data/processed/model1/input_group_software_id_vocab.csv', header = None)
    technique_tactics_vocab = pd.read_csv('../data/processed/model1/input_technique_tactics_vocab.csv', header= None)
    technique_data_sources_vocab = pd.read_csv('../data/processed/model1/input_technique_data_sources_vocab.csv', header= None)
    # technique_defenses_bypassed_vocab = pd.read_csv('../data/processed/model1/input_technique_defenses_bypassed_vocab.csv', header= None)
    technique_detection_name_vocab = pd.read_csv('../data/processed/model1/input_technique_detection_name_vocab.csv', header= None)
    technique_mitigation_id_vocab = pd.read_csv('../data/processed/model1/input_technique_mitigation_id_vocab.csv', header= None)
    # technique_permissions_required_vocab = pd.read_csv('../data/processed/model1/input_technique_permissions_required_vocab.csv', header= None)
    technique_platforms_vocab = pd.read_csv('../data/processed/model1/input_technique_platforms_vocab.csv', header= None)
    technique_software_id_vocab = pd.read_csv('../data/processed/model1/input_technique_software_id_vocab.csv', header= None)
    
    vocabs = {
    'input_software_id' : pd.concat ([group_software_id_vocab, technique_software_id_vocab])[0].dropna().unique(),
    'input_tactics' : technique_tactics_vocab[0].dropna().values,   
    'input_technique_data_sources' : technique_data_sources_vocab[0].dropna().values,
    # 'input_technique_defenses_bypassed' : technique_defenses_bypassed_vocab[0].dropna().values,
    'input_technique_detection_name' : technique_detection_name_vocab[0].dropna().values,
    'input_technique_mitigation_id' : technique_mitigation_id_vocab[0].dropna().values,
    # 'input_technique_permissions_required' : technique_permissions_required_vocab[0].dropna().values,
    'input_technique_platforms' : technique_platforms_vocab[0].dropna().values,
    'input_technique_software_id' : technique_software_id_vocab[0].dropna().values,
    }
    #### 👉LOAD DATASETS, THEN CONFIG DATASETS
    train_dataset, cv_dataset = load_datasets_2(empty_train_cv= True, return_feature_info=False)
    
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    cv_dataset = cv_dataset.batch(batch_size)
    cv_dataset = cv_dataset.shuffle (buffer_size=len(cv_dataset))
    
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
    
    
    early_stopping = tf.keras.callbacks.EarlyStopping (
        verbose = 1,
        monitor = 'val_auc-pr',
        mode = 'max',
        patience = patience,
        start_from_epoch= start_from_epoch,
        restore_best_weights= True
    )
    
    #### 👉 TRAIN MODEL 
    start_time = time.time()
    history = model.fit (
        train_dataset,
        validation_data= cv_dataset,
        epochs=epochs,
        class_weight=class_weights,
        callbacks= [early_stopping]
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int (elapsed_time % 60)
    print(f"Training completed in {elapsed_minutes} minutes and {elapsed_seconds} seconds")
    
    #### 👉 SAVE HISTORY
    history_df = pd.DataFrame(history.history)
    file_name = '{config_file}.csv'.format(config_file = args.config)
    if not os.path.exists(os.path.join (REPORT_FOLDER, 'train_loss', SINGLE_TRAIN_FOLDER_NAME)):
        os.makedirs(os.path.join (REPORT_FOLDER, 'train_loss', SINGLE_TRAIN_FOLDER_NAME))
    
    file_path = os.path.join (REPORT_FOLDER, 'train_loss', SINGLE_TRAIN_FOLDER_NAME,file_name)
    history_df.to_csv(file_path, index=False)
    
    #### 👉 SAVE TRAINED MODEL
    
    base_config_filename = config_filename.split(".")[0]
    model_file_name = base_config_filename
    if not os.path.exists(TRAINED_MODELS_FOLDER):
        os.makedirs(TRAINED_MODELS_FOLDER)
    model_file_path = os.path.join (TRAINED_MODELS_FOLDER, model_file_name)
    model.save (model_file_path)
    
if __name__ == '__main__':
    main()
    
    
