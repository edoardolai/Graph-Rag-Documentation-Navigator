import os
import shutil
from collections import defaultdict

def get_file_types(source_dir):
    """
    Returns a dictionary with file extensions as keys and file paths as values.
    """
    file_types = defaultdict(list)

    for root, _, files in os.walk(source_dir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            file_types[ext].append(os.path.join(root, file))

    return file_types

def copy_excluding_files(file_types, excluded_extensions, destination_dir):
    """
    Copy all files except those with specified extensions 
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for ext, files in file_types.items():
        if ext not in excluded_extensions:
            for file in files:
                dest_path = os.path.join(destination_dir, os.path.basename(file))
                shutil.copy2(file, dest_path)
                print(f"Copied: {file} -> {dest_path}")

if __name__ == "__main__":
    source_directory = r"C:\Users\hp\Imad\Orange\codecarbon"
    destination_directory = r"C:\Users\hp\Imad\Orange\codecarbon_cleaned"
    excluded_file_types = {".md", ".rst", ".drawio", ".svg", ".png", ".jpg", ".gif", ".ico", ".rst",".example" }

    # Get all file types in the repo
    file_types_dict = get_file_types(source_directory)

    # Print summary of file types
    print("Found file types:")
    for ext, files in file_types_dict.items():
        print(f"{ext}: {len(files)} files")

    # Copy all except excluded files
    copy_excluding_files(file_types_dict, excluded_file_types, destination_directory)

    print("File cleaning completed.")
