from database import load_data_from_database
from modeling import prepare_features, get_models
from evaluation import compare_models
from train_predict import load_model_from_pickle

df_train, df_test, df_summary, df_test_rul, df_test_last_cycle = load_data_from_database()

X_train, y_train, X_test = prepare_features(df_train, df_test)

models = load_model_from_pickle('models.pkl')
results = compare_models(models, X_train, y_train, X_test, df_test, df_test_rul)

print(results)