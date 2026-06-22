# column transformer pipeline for preprocessing
from sklearn.compose import ColumnTransformer   
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
from sklearn.model_selection import GridSearchCV, GroupKFold, TimeSeriesSplit
import pandas as pd

# for Bayesian optimization
try:
    from skopt import BayesSearchCV
except ImportError as e:
    raise ImportError(
        "Bayesian optimization requires scikit-optimize. "
        "Install it with: uv add scikit-optimize"
    ) from e


# Instead of OneHotEncoder use simple One-hot encoding the categorical features and ensure the same columns in train and test sets
def one_hot_encode_and_align(train_df, test_df, categorical_cols):
    train_encoded = pd.get_dummies(train_df, columns=categorical_cols)
    test_encoded = pd.get_dummies(test_df, columns=categorical_cols)
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    train_encoded.columns = [str(c) for c in train_encoded.columns]
    test_encoded.columns = [str(c) for c in test_encoded.columns]

    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    train_encoded = train_encoded.astype("float32")
    test_encoded = test_encoded.astype("float32")

    return train_encoded, test_encoded

# load train and test data from the database
def clean_column_names(df):
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df

def init_preprocessor(in_df):
    # identify numeric and categorical columns
    numeric_features = in_df.select_dtypes(include=['float64', 'float32']).columns
    categorical_features = in_df.select_dtypes(include=['int64', 'int32']).columns
    # numeric transformer
    numeric_transformer = Pipeline(steps=[ 
        ('scaler', StandardScaler()) 
    ])  
    # categorical transformer
    categorical_transformer = Pipeline(steps=[  
        ('onehot', OneHotEncoder(handle_unknown='ignore')) 
    ])  

    # combine transformers into a column transformer
    preprocessor = ColumnTransformer(   
        transformers=[ 
            ('num', numeric_transformer, numeric_features), 
            ('cat', categorical_transformer, categorical_features) 
        ])
    return preprocessor


# tune the model using GridSearchCV
def tune_model_grid(model_pipeline, 
                    in_param_grid, 
                    in_features_train, 
                    in_target_train,
                    scoring='neg_mean_absolute_error'):
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(estimator=model_pipeline, 
                               param_grid=in_param_grid, 
                               cv=tscv,
                               scoring=scoring, 
                               n_jobs=-1)
    grid_search.fit(in_features_train, in_target_train)
    #print(f'Best parameters: {grid_search.best_params_}')
    return grid_search.best_estimator_, grid_search.best_params_


# tune the model using Bayesian optimization
def tune_model_bayesian(model_pipeline, 
                        in_param_bayes, 
                        in_features_train, 
                        in_target_train,
                        scoring='neg_mean_absolute_error'):
    tscv = TimeSeriesSplit(n_splits=5)
    bayes_search = BayesSearchCV(estimator=model_pipeline, 
                                 search_spaces=in_param_bayes, 
                                 cv=tscv,
                                 scoring=scoring, 
                                 n_jobs=-1)
    bayes_search.fit(in_features_train, in_target_train)
    #print(f'Best parameters: {bayes_search.best_params_}')
    return bayes_search.best_estimator_, bayes_search.best_params_


# save trained model to pickle file
def save_model_to_pickle(model_pipeline, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model_pipeline, f)

# load trained model from pickle file
def load_model_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        model_pipeline = pickle.load(f)
    return model_pipeline

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# learn curve for training and testing data
def plot_learning_curve_timeseries(model_pipeline, model_name, features_train, target_train):
    tscv = TimeSeriesSplit(n_splits=3)
    train_sizes, train_scores, test_scores = learning_curve(model_pipeline, 
                                                            features_train, 
                                                            target_train, 
                                                            cv=tscv, 
                                                            scoring='neg_mean_absolute_error', n_jobs=-1)
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure(figsize=(6, 3))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='orange', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='steelblue', label='Cross-validation score')
    plt.title(f'Learning Curve of {model_name}')
    plt.xlabel('Training examples')
    plt.ylabel('MAE')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

import numpy as np
from sklearn.model_selection import GroupKFold

# learn curve for training and testing data with GroupKFold to ensure that all data from the same unit is in the same fold
def plot_learning_curve_grouped(model, model_name, X_train, y_train, df_train):
    groups = (
        df_train["dataset_id"].astype(str)
        + "_"
        + df_train["unit_nr"].astype(str)
    )

    cv = GroupKFold(n_splits=5)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        groups=groups,
        cv=cv,
        scoring="neg_mean_absolute_error",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1,
    )

    train_mae = -train_scores.mean(axis=1)
    val_mae = -val_scores.mean(axis=1)

    plt.figure(figsize=(7, 4))
    plt.plot(train_sizes, train_mae, "o-", label="Training MAE")
    plt.plot(train_sizes, val_mae, "o-", label="Validation MAE")
    plt.title(f"Learning Curve - {model_name}")
    plt.xlabel("Training samples")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)
    plt.show()