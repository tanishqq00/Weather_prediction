import os
import pandas as pd

def load_raw_data(filename="weatherAUS.csv"):
    """
    Load a CSV from data/raw/ folder.
    """
    # Get project root (one level up from src/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Full path to the CSV
    file_path = os.path.join(project_root, "data", "raw", filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load CSV
    df = pd.read_csv(file_path)
    return df
