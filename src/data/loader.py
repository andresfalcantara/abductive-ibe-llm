import pandas as pd
from src.config import PROCESSED_DATASET_PATH

def load_data():
    return pd.read_parquet(PROCESSED_DATASET_PATH)

