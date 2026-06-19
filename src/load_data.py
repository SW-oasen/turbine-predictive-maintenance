import os
from typing import Literal
import pandas as pd
import numpy as np
#import yaml
from sqlalchemy import create_engine
import requests

from src.config import *

SENSOR_COUNT = 21
COLS = ["unit_nr","time_cycles","setting1","setting2","setting3"] + [f"sensor{i}" for i in range(1,SENSOR_COUNT+1)]

# Read CMAPSS txt file into DataFrame
def read_cmapss_txt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :len(COLS)]
    df.columns = COLS
    df["unit_nr"] = df["unit_nr"].astype(int)
    df["time_cycles"] = df["time_cycles"].astype(int)
    for c in ["setting1","setting2","setting3"] + [f"sensor{i}" for i in range(1,SENSOR_COUNT+1)]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    #print(f"\nLoaded data from {path.split('/')[-1]} shape: {df.shape}")
    return df 

# Compute Remaining Useful Life (RUL)
def compute_rul(df: pd.DataFrame) -> pd.Series:
    max_cycles = df.groupby("unit_nr")["time_cycles"].transform("max")
    return (max_cycles - df["time_cycles"]).astype(int)


# load complete dataset
def load_complete_dataset(data_purpose: Literal["train", "test"] = "train", datasets_info: list[dict] = DATASETS_META) -> pd.DataFrame:
    '''
    Loads the training data for all datasets, computes RUL, and concatenates them into a single DataFrame.      
    Returns:
        pd.DataFrame: Concatenated training data with RUL.
    '''
    if data_purpose == "train":
        file_key = "train_file"
    elif data_purpose == "test":
        file_key = "test_file"
    else:
        raise ValueError("data_purpose must be 'train' or 'test'")
    df_list = []
    for dataset in datasets_info:
        df_tmp = read_cmapss_txt(f"{RAW_DATA_PATH}{dataset[file_key]}")
        # check single value columns
        #single_value_cols = [col for col in df_train_tmp.columns if df_train_tmp[col].nunique() == 1]
        #print(f"Single value columns in {dataset['code']}:\n", single_value_cols)
        
        df_tmp['dataset_id'] = dataset['dataset_id']
        df_tmp['split'] = data_purpose
        df_rul = compute_rul(df_tmp)
        df_tmp['RUL'] = df_rul
        #display(df_tmp.head())
        df_list.append(df_tmp)
        df_concat = pd.concat(df_list, ignore_index=True)
        print(f"\n{data_purpose} data {dataset['dataset_id']}: shape: {df_concat.shape}")
    return df_concat

# load complete RUL test dataset
def load_rul_test(datasets_info: list[dict] = DATASETS_META):
    '''
    Loads the RUL test data for all datasets, adds dataset_id and unit_nr columns, and concatenates them into a single DataFrame.      
    Returns:
        pd.DataFrame: Concatenated RUL test data with dataset_id and unit_nr.
    '''
    rul_test_list = []
    for dataset in datasets_info:
        rul_test_tmp = pd.read_csv(f"{RAW_DATA_PATH}{dataset['rul_file']}", 
                                   header=None, 
                                   names=['true_RUL'])
        rul_test_tmp['dataset_id'] = dataset['dataset_id']
        rul_test_tmp['unit_nr'] = rul_test_tmp.index + 1
        rul_test_list.append(rul_test_tmp)
    return pd.concat(rul_test_list, ignore_index=True)
