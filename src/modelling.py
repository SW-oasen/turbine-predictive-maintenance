# column transformer pipeline for preprocessing
from sklearn.compose import ColumnTransformer   
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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

