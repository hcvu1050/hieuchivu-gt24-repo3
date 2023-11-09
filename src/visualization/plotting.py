import sys, os, yaml, re
sys.path.append("..")

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
SOURCE_REPORT_FOLDER = os.path.join (ROOT_FOLDER, 'reports')
SOURCE_CONFIG_FOLDER = os.path.join (ROOT_FOLDER, 'configs')

import pandas as pd
import matplotlib.pyplot as plt

def grid_plot_history_with_config (model_name:str, folder_name: str,labels: list, file_name_filter: str = None, ylims: list = None, xlims: list = None):
    # get list of configs
    config_folder_path = os.path.join (SOURCE_CONFIG_FOLDER, folder_name)
    config_file_list = os.listdir (config_folder_path)
    if file_name_filter != None:
        config_file_list = [f for f in config_file_list if re.fullmatch (file_name_filter, f) is not None]
    # config_file_list = [f for f in config_file_list if f.startswith(model_name)]
    config_file_list = [os.path.join(config_folder_path, f) for f in config_file_list]
        
    # get list of train loss files
    history_folder_path = os.path.join (SOURCE_REPORT_FOLDER, model_name, 'train_loss', folder_name)
    history_file_list = os.listdir (history_folder_path)
    if file_name_filter != None:
        history_file_list = [f for f in history_file_list if re.fullmatch (file_name_filter, f) is not None]
    history_file_list = [os.path.join (history_folder_path, f) for f in history_file_list]

    # PLOTTING
    
    num_grid_rows = len (history_file_list)
    plt.figure(figsize=(12, 5 * num_grid_rows)) 
    
    for grid in range (1, len(history_file_list) + len(config_file_list)+1):
        plt.subplot (num_grid_rows, 2, grid)
        
        if grid % 2 == 1: 
            if ylims is not None: plt.ylim(ylims) 
            if xlims is not None: plt.ylim(xlims) 
            for label in labels:
                _plot_history (
                    file_name = history_file_list[int((grid-1)/2)],
                    title = ' / '.join (labels), label= label
                )
        else:
            _plot_config(
                config_file_list[int(grid/2-1)],
                title=  config_file_list[int(grid/2-1)].split(sep = "\\")[-1]
                )
def multi_line_plot_history (model_name: str, folder_name: str, label: str,  hyperparameter: list, file_name_filter: str = None, ylims: list = None, xlims: list = None):
    # get list of configs
    config_folder_path = os.path.join (SOURCE_CONFIG_FOLDER, folder_name)
    config_file_list = os.listdir (config_folder_path)
    if file_name_filter != None:
        config_file_list = [f for f in config_file_list if re.fullmatch (file_name_filter, f) is not None]
    # config_file_list = [f for f in config_file_list if f.startswith(model_name)]
    print (config_file_list)
    config_file_list = [os.path.join(config_folder_path, f) for f in config_file_list]
    
    # get list of train loss files
    history_folder_path = os.path.join (SOURCE_REPORT_FOLDER, model_name, 'train_loss', folder_name)
    history_file_list = os.listdir (history_folder_path)
    if file_name_filter != None:
        history_file_list = [f for f in history_file_list if re.fullmatch (file_name_filter, f) is not None]
    history_file_list = [os.path.join (history_folder_path, f) for f in history_file_list]
    
    # plot the lines with the line labels being the value of the chosen hyper parameter
    plt.figure (figsize= (24,10))
    if ylims is not None: plt.ylim(ylims) 
    if xlims is not None: plt.xlim(xlims)
    for i in range (len (history_file_list)):
        # get the hyperparameter value from the config file
        with open (config_file_list[i], 'r') as config_file:
            config = yaml.safe_load(config_file)
        
        line_name = config[hyperparameter[0]]
        line_name = line_name[hyperparameter[1]]
        if line_name is None: line_name = 'None'
        
        _plot_history (
            file_name = history_file_list[i],
            title = hyperparameter[-1],
            label = label, 
            line_name = line_name
        )
    
    
def _plot_history (file_name: str, title: str, label:str, line_name: str = None):
    history_df = pd.read_csv(file_name)
    epochs = range(1, len(history_df) + 1)
    values = history_df[label]
    # plt.figure(figsize=(6, 5))
    # plt.ylim(.50, .65) 
    if line_name is not None: 
        plt.plot(epochs, values, label= line_name)
    else: 
        plt.plot(epochs, values, label = label)
    plt.title(title)
    plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    plt.legend()

def _plot_config (filename: str, title: str): 
    with open (filename, 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    formatted_text = yaml.dump(config, default_flow_style=False, indent=4, sort_keys=False)
    # Display the formatted text
    plt.text(0.1, 0.5, formatted_text, fontsize=9, va='center', ha='left')
    # Turn off axis for this subplot
    plt.axis('off')
    # Add a title
    plt.title(title)
    # Save or display the figure
    plt.tight_layout()
    