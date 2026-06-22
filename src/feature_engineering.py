import os
from typing import Literal
import pandas as pd
import numpy as np
#import yaml
from sqlalchemy import create_engine

from config import *
from database import *

def create_sensor_delta(df, selected_sensor_index: list[int] = None):
    if selected_sensor_index is None:
        selected_sensor_index = range(1,22)
    important_sensors = [f"sensor{i}" for i in selected_sensor_index]
    df_out = df.copy()
    # Create difference features for important sensors
    for sensor in important_sensors:
        df_out[f"{sensor}_diff1"] = (
            df_out.groupby(["dataset_id", "unit_nr"])[sensor].diff()
        )
        df_out[f"{sensor}_diff5"] = (
            df_out.groupby(["dataset_id", "unit_nr"])[sensor].diff(5)
        )

    return df_out


# Clip RUL to a maximum of 125 to reduce weight of early cycles. 
# Purpose is to identify RUL at the end of the cycle, which is more critical for maintenance decisions.
def clip_rul(df, max_rul=125):
    if max_rul is not None and "RUL" in df.columns:
        df["RUL"] = df["RUL"].clip(upper=max_rul)
    return df

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

#
def prepare_data_for_modeling(selected_sensor_index, max_rul=150):
    df_train, df_test, df_summary, df_test_rul, _ = load_data_from_database()

    #selected_sensor_index = [4, 5, 6, 7, 9, 11, 12, 13, 14, 14, 15]
    df_train_engineered = create_sensor_delta(df_train, selected_sensor_index)
    df_train_engineered = clip_rul(df_train_engineered, max_rul=150)
    df_test_engineered = create_sensor_delta(df_test, selected_sensor_index)
    df_test_engineered = create_sensor_delta(df_test, selected_sensor_index)

    feature_train, target_train, feature_test, target_test = prepare_features_and_target(df_train_engineered, df_test_engineered, df_test_rul)

    return feature_train, target_train, feature_test, target_test, df_test, df_test_rul, df_summary