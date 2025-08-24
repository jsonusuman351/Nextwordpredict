import os
from pathlib import Path
import logging

# Configure basic logging to show INFO level messages
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# The name of the project (can be changed for other projects)
project_name = "CampusX_FAQ_Bot"

# List of all files and directories to be created
list_of_files = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "data/faqs.txt",
    "saved_models/.gitkeep",
    "src/__init__.py",
    "src/data_preprocessing.py",
    "src/model.py",
    "src/train.py",
    "src/predict.py"
]

# Loop through each file path in the list
for filepath in list_of_files:
    # Convert the string path to a Path object for OS compatibility
    filepath = Path(filepath)
    
    # Split the path into its directory and filename components
    filedir, filename = os.path.split(filepath)

    # If the directory part is not empty, create the directory
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    # Check if the file does not exist or if it is an empty file
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        # Create an empty file
        with open(filepath, 'w') as f:
            pass  # 'pass' does nothing, just ensures the file is created
            logging.info(f"Creating empty file: {filepath}")
    
    # If the file already exists and is not empty, log it
    else:
        logging.info(f"{filename} already exists")
