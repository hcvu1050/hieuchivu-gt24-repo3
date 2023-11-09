"""
delete all files in a folder in `data/`. Folder name is specified as an arg
Except for files in `data/raw`, which can NOT be deleted.
"""
import sys, os, argparse, shutil
sys.path.append("..")
# path to save the built-feature data
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')

def main ():
    parser = argparse.ArgumentParser (description= 'command-line arguments when running current script')
    parser.add_argument ('-dir', required= True,  choices = ['interim', 'processed'],
                         help = 'name of the directory in `data/` whose contents will be deleted. Can ONLY delete from folder "interim" or "processed"')
    args = parser.parse_args()
    dir_name = args.dir
    
    target_path = os.path.join (DATA_FOLDER, dir_name)
    clear_folder (target_path)

def clear_folder(folder_path):
    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Remove all files and subdirectories within the folder
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"Contents of {folder_path} have been cleared.")
        else:
            print(f"{folder_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main ()