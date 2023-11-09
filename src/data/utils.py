import os
import pandas as pd
# Get the root directory of the project
FILE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
# path to get collected data
TARGET_PATH = os.path.join (FILE_DIR, 'data/interim')

def _make_file_list (filename: str, target_path, content: list):
    """
    create a .txt file containing a list of file names stored in `content`
    """
    # file_path = os.path.join(data_folder, filename)
    os.makedirs (target_path, exist_ok= True)
    if not filename.endswith ('.txt'): filename += '.txt'
    output_file = os.path.join (target_path, filename)
    
    with open (output_file, 'w') as file:
        for line in content:
            file.write (line)
            file.write ('\n')
    print ('List of exported files saved at: ', output_file)
    

def save_df_to_csv (df: pd.DataFrame, target_path, filename, prefix = '', postfix = '', output_list_file = None):
    """
    save a pandas DataFrame as csv file. The name of the file is customized by `filename`, `prefix`, and `postfix`
    """
    
    os.makedirs (target_path, exist_ok = True)
    if not (prefix == '') and (not prefix.endswith ('_')): prefix += '_'
    if not (postfix == '') and (not prefix.startswith ('_')): postfix = '_' + postfix
    filename = prefix + filename + postfix
    
    if not filename.endswith (".csv"): filename+= ".csv"
    output_file = os.path.join(target_path, filename)
    print ('Saving:\t', filename, end = '-----')
    df.to_csv (output_file, index = False)
    print ("Saved")
    print ("Finished: file saved to", target_path)
    
    if output_list_file is not None:
        _make_file_list (filename= filename, target_path=target_path, content= [filename])

def batch_save_df_to_csv (file_name_dfs: dict, target_path, prefix ='', postfix ='',output_list_file = None):
    """
    Saves the DataFrames stored in a dict as csv file. \n
    Arg `file_name_dfs` stores the filenames as keys and the corresponding DataFrame as values\n
        key: filename\n
        value = DataFrame\n
    """
    if not (prefix == '') and (not prefix.endswith ('_')): prefix += '_'
    if not (postfix == '') and (not prefix.startswith ('_')): postfix = '_' + postfix
    content = []
    
    for key in file_name_dfs.keys():
        os.makedirs (target_path, exist_ok = True)
        
        filename = prefix + key + postfix
        if not filename.endswith (".csv"): filename+= ".csv"
        output_file = os.path.join(target_path, filename)
        
        print ('Saving:\t', filename, end = '-----')
        df = file_name_dfs[key]
        df.to_csv (output_file, index = False)
        content.append (filename)
        print ('Saved')
    
    print ("Finished: {} files saved to {}".format (len(file_name_dfs.keys()),target_path))
    
    if output_list_file is not None:
        # make a txt file containing the names of exported file
        _make_file_list (filename = output_list_file, target_path=target_path, content=content)
def batch_save_df_to_pkl (file_name_dfs: dict, target_path, prefix ='', postfix ='',output_list_file = None):
    """
    Saves the DataFrames stored in a dict as csv file. \n
    Arg `file_name_dfs` stores the filenames as keys and the corresponding DataFrame as values\n
        key: filename\n
        value = DataFrame\n
    """
    if not (prefix == '') and (not prefix.endswith ('_')): prefix += '_'
    if not (postfix == '') and (not prefix.startswith ('_')): postfix = '_' + postfix
    content = []
    
    for key in file_name_dfs.keys():
        os.makedirs (target_path, exist_ok = True)
        
        filename = prefix + key + postfix
        if not filename.endswith (".pkl"): filename+= ".pkl"
        output_file = os.path.join(target_path, filename)
        
        print ('Saving:\t', filename, end = '-----')
        df = file_name_dfs[key]
        df.to_pickle (output_file)
        content.append (filename)
        print ('Saved')
    
    print ("Finished: {} files saved to {}".format (len(file_name_dfs.keys()),target_path))
    
    if output_list_file is not None:
        # make a txt file containing the names of exported file
        _make_file_list (filename = output_list_file, target_path=target_path, content=content)
        
def batch_save_df_to_csv_with_index (file_name_dfs: dict, target_path, prefix ='', postfix ='',output_list_file = None):
    """
    Saves the DataFrames stored in a dict as csv file. \n
    Arg `file_name_dfs` stores the filenames as keys and the corresponding DataFrame as values\n
        key: filename\n
        value = DataFrame\n
    """
    if not (prefix == '') and (not prefix.endswith ('_')): prefix += '_'
    if not (postfix == '') and (not prefix.startswith ('_')): postfix = '_' + postfix
    content = []
    
    for key in file_name_dfs.keys():
        os.makedirs (target_path, exist_ok = True)
        
        filename = prefix + key + postfix
        if not filename.endswith (".csv"): filename+= ".csv"
        output_file = os.path.join(target_path, filename)
        
        print ('Saving:\t', filename, end = '-----')
        df = file_name_dfs[key]
        df.to_csv (output_file, index = True)
        content.append (filename)
        print ('Saved')
    
    print ("Finished: {} files saved to {}".format (len(file_name_dfs.keys()),target_path))
    
    if output_list_file is not None:
        # make a txt file containing the names of exported file
        _make_file_list (filename = output_list_file, target_path=target_path, content=content)