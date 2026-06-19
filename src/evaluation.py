import os
from typing import Literal
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import *

# create evaluation DataFrame by merging predictions with true RUL values
# in order to compute evaluation metrics and create plots of true vs predicted RUL
def create_eval_dataframe_from_predictions(predictions, df_test, df_test_rul):
    pred_last = (
        pd.DataFrame(predictions, columns=["predicted_RUL"])
        .assign(dataset_id=df_test["dataset_id"], unit_nr=df_test["unit_nr"], time_cycles=df_test["time_cycles"])
        .sort_values("time_cycles")
        .groupby(["dataset_id", "unit_nr"])
        .tail(1)
    )
    eval_df = pred_last.merge(
        df_test_rul,
        on=["dataset_id", "unit_nr"],
        how="inner"
    )
    return eval_df

def print_scores(model_name, target_test, predictions):
    mae = mean_absolute_error(target_test, predictions)
    rmse = np.sqrt(mean_squared_error(target_test, predictions))
    r2 = r2_score(target_test, predictions)
    print("-"*20 + f" scoring " + "-"*20)
    print(f"{'model':<15} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
    print(f'{model_name:<16} {mae:>10.2f} {rmse:>10.2f} {r2:>8.2f}')


