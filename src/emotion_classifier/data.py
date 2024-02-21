import pandas as pd
from typing import Dict
from datasets import load_dataset

DATASET = load_dataset('emotion', trust_remote_code=True)

EMOTION_DICT: Dict[int, str] = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise',
}

def set_display_options() -> None:
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.width', None)  # Adjust display width to terminal size
    pd.set_option('display.max_colwidth', None)  # Show full content of each column