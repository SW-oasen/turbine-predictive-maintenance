import os
from typing import Literal
import pandas as pd
import numpy as np
#import yaml
from sqlalchemy import create_engine

from src.config import *

def engineer_features(df, important_sensors: list[int]):
    important_sensors = [f"sensor{i}" for i in important_sensors]
    df_out = df.copy()
    # Create difference features for important sensors
    for sensor in important_sensors:
        df_out[f"{sensor}_diff1"] = (
            df_out.groupby(["dataset_id", "unit_nr"])[sensor].diff()
        )
        df_out[f"{sensor}_diff5"] = (
            df_out.groupby(["dataset_id", "unit_nr"])[sensor].diff(5)
        )
    # Clip RUL to a maximum of 125 to reduce weight of early cycles. 
    # Purpose is to identify RUL at the end of the cycle, which is more critical for maintenance decisions.
    if "RUL" in df_out.columns:
        df_out["RUL"] = df_out["RUL"].clip(upper=125)

    return df_out

# Prepare features and target for modeling
def prepare_features_and_target(df_train, df_test, df_test_rul):
    feature_train = df_train.drop(columns=['dataset_id', 'condition', 'fault_mode', 'unit_nr','RUL'])
    target_train = df_train['RUL']

    feature_test = df_test.drop(columns=['dataset_id', 'condition', 'fault_mode', 'unit_nr'])
    target_test = df_test_rul['true_RUL']
    return feature_train, target_train, feature_test, target_test

# Rolling mean and std for sensor readings
def create_rolling_features(df, rolling_cols, window_sizes=[5,10]):
    for col in rolling_cols:
        for window_size in window_sizes:
            df[f'{col}_rolling_mean_{window_size}'] = df.groupby(['dataset_id', 'unit_nr'])[col].transform(lambda x: x.rolling(window_size, min_periods=1).mean())
            df[f'{col}_rolling_std_{window_size}'] = df.groupby(['dataset_id', 'unit_nr'])[col].transform(lambda x: x.rolling(window_size, min_periods=1).std())
    return df