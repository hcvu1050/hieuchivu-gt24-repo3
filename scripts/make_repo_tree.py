import os
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
import os

def create_md_file(directory, output_file):
    with open(output_file, 'w') as f:
        f.write(f"# Folder Structure\n\n")
        generate_md(directory, f)

def generate_md(directory, f, indent=0):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            f.write(f"{'  ' * indent}- **{item}/**\n")
            generate_md(item_path, f, indent + 1)

if __name__ == "__main__":
    create_md_file(ROOT_FOLDER, "folder_structure.md")