import os
import shutil
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

source_dir_path = os.getenv("RAW_REPO_DIRECTORY_PATH")
destination_dir_path = os.getenv("CLEANED_REPO_DIRECTORY_PATH")


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


def copy_excluding_files(file_types, included_extensions, destination_dir):
    """
    Copy all files except those with specified extensions
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for ext, files in file_types.items():
        if ext in included_extensions:
            for file in files:
                dest_path = os.path.join(destination_dir, os.path.basename(file))
                shutil.copy2(file, dest_path)
                print(f"Copied: {file} -> {dest_path}")


if __name__ == "__main__":
    # update to macos path
    source_directory = source_dir_path

    destination_directory = destination_dir_path
    included_file_types = {".py", ".js", ".html", ".css", ".ts", ".tsx"}

    # Get all file types in the repo
    file_types_dict = get_file_types(source_directory)

    # Print summary of file types
    print("Found file types:")
    for ext, files in file_types_dict.items():
        print(f"{ext}: {len(files)} files")

    # Copy all except excluded files
    copy_excluding_files(file_types_dict, included_file_types, destination_directory)

    print("File cleaning completed.")
