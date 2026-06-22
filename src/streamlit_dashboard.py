import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import seaborn as sns

from database import *
from feature_engineering import *
from modeling import *
from evaluation import *
from config import PROJECT_ROOT, MODEL_PATH

st.set_page_config(
    page_title="Turbofan RUL Prediction",
    layout="wide",
)

st.title("Turbofan Predictive Maintenance")
st.caption("Remaining Useful Life prediction using NASA CMAPSS turbofan engine data")

selected_sensor_index = [4, 5, 6, 7, 9, 11, 12, 13, 14, 14, 15]
@st.cache_data
def load_data():
    return prepare_data_for_modeling(selected_sensor_index)

@st.cache_data
def load_results():
    return pd.DataFrame(
        [
            ["LightGBM", 18.74, 26.00, 0.74],
            ["XGBoost", 18.69, 25.96, 0.74],
            ["Tuned LightGBM", 18.30, 25.54, 0.75],
            ["Tuned XGBoost", 18.07, 25.45, 0.75],
            ["XGBoost + RUL cap 150", 18.30, 24.64, 0.77],
        ],
        columns=["Model", "MAE", "RMSE", "R²"],
    )


@st.cache_data
def load_rul_cap_results():
    return pd.DataFrame(
        [
            [75, 30.24, 42.73, 0.30],
            [100, 22.01, 31.64, 0.62],
            [125, 18.69, 25.96, 0.74],
            [150, 18.30, 24.64, 0.77],
            [175, 19.27, 25.59, 0.75],
            [200, 20.48, 27.21, 0.72],
            [250, 22.15, 30.22, 0.65],
            [None, 23.58, 33.09, 0.58],
        ],
        columns=["max_rul", "MAE", "RMSE", "R²"],
    )

@st.cache_data
def load_models():
    best_lgbm = load_model_from_pickle(MODEL_PATH / "best_lgbm_model.pkl")
    best_xgb = load_model_from_pickle(MODEL_PATH / "best_xgb_model.pkl")
    return best_lgbm, best_xgb


feature_train, target_train, feature_test, target_test, df_test, df_test_rul, df_summary = load_data()
best_lgbm, best_xgb = load_models()
predictions_xgb = best_xgb.predict(feature_test)  
eval_df_xgb = create_eval_dataframe_from_predictions(predictions_xgb, df_test, df_test_rul)

df_overview = df_summary.groupby(["dataset_id", 'split']).agg(
    total_units=pd.NamedAgg(column="unit_nr", aggfunc="nunique"),
    max_cycle=pd.NamedAgg(column="cycles_count", aggfunc="max"),
    average_cycles=pd.NamedAgg(column="cycles_count", aggfunc="mean"),
).reset_index()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model Performance",
        "RUL Cap Experiment",
        "Feature Importance",
        "Engine Explorer",
    ],
)


if page == "Overview":
    st.header("Project Overview")

    st.write(
        """
        This project predicts the Remaining Useful Life (RUL) of turbofan engines
        based on operational settings and sensor measurements from the NASA CMAPSS dataset.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Datasets", "FD001–FD004")
    col2.metric("Best MAE", "18.07 cycles")
    col3.metric("Best R²", "0.77")

    st.subheader("Dataset Summary")
    st.dataframe(df_overview, use_container_width=True)


elif page == "Model Performance":
    st.header("Model Performance")

    results = load_results()
    st.dataframe(results, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(results["Model"], results["MAE"], marker="o", label="MAE", color="steelblue")
    ax.plot(results["Model"], results["RMSE"], marker="o", label="RMSE", color="lightgreen")
    ax.set_ylabel("MAE / RMSE")
    ax.set_title("Model Performance Comparison")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="lower left")
    y_max = max(results["MAE"].max(), results["RMSE"].max()) * 1.1
    ax.set_ylim(0, y_max)

    ax_twin = ax.twinx()
    ax_twin.plot(results["Model"], results["R²"], marker="o", color="orange", label="R²")
    ax_twin.set_ylabel("R²")
    ax_twin.legend(loc="lower right")
    y_twin_max = results["R²"].max() * 1.1
    ax_twin.set_ylim(0, y_twin_max)

    st.pyplot(fig)


elif page == "RUL Cap Experiment":
    st.header("RUL Capping Experiment")

    cap_results = load_rul_cap_results()
    st.dataframe(cap_results, use_container_width=True)

    plot_df = cap_results.copy()
    plot_df["max_rul_label"] = plot_df["max_rul"].apply(
        lambda x: "None" if pd.isna(x) else str(int(x))
)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(plot_df["max_rul_label"], plot_df["MAE"], marker="o", label="MAE", color="steelblue")
    ax.plot(plot_df["max_rul_label"], plot_df["RMSE"], marker="o", label="RMSE", color="green")
    ax.set_xlabel("RUL cap")
    ax.set_ylabel("MAE / RMSE")
    ax.legend(loc="lower left")
    y_max = max(plot_df["MAE"].max(), plot_df["RMSE"].max()) * 1.1
    ax.set_ylim(0, y_max)

    ax_twin = ax.twinx()
    ax_twin.plot(plot_df["max_rul_label"], plot_df["R²"], marker="o", label="R²", color="orange")
    ax_twin.set_xlabel("RUL cap")
    ax_twin.set_ylabel("R²")
    ax_twin.legend(loc="lower right")
    y_twin_max = plot_df["R²"].max() * 1.1
    ax_twin.set_ylim(0, y_twin_max)
    fig.suptitle("Impact of RUL Capping on Model Performance", fontsize=14)

    st.pyplot(fig)

    st.info(
        "The best observed cap was max_rul = 150. "
        "It reduces noise from the early life phase while preserving enough degradation information."
    )


elif page == "Feature Importance":
    st.header("Feature Importance")

    lgbm_importance_path = PROJECT_ROOT / "models" / "feature_importance_lgbm.csv"
    xgb_importance_path = PROJECT_ROOT / "models" / "feature_importance_xgb.csv"

    df_feature_importance_lgbm = pd.read_csv(lgbm_importance_path)
    df_feature_importance_xgb = pd.read_csv(xgb_importance_path)

    if lgbm_importance_path.exists() and xgb_importance_path.exists():
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sns.barplot(x=df_feature_importance_lgbm["importance"], y=df_feature_importance_lgbm["feature"], ax=axes[0], color="steelblue")
        axes[0].set_title("Tuned LightGBM model")
        sns.barplot(x=df_feature_importance_xgb["importance"], y=df_feature_importance_xgb["feature"], ax=axes[1], color="lightgreen")
        axes[1].set_title("Tuned XGBoost model") 
        axes[1].set_ylabel("")
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)
    else:
        st.warning(
            "No feature importance file found. "
            "Save your feature importance as reports/feature_importance.csv "
            "with columns: feature, importance."
        )

elif page == "Engine Explorer":
    st.header("Engine Health Explorer")

    # eval: dataset_id, unit_nr, true_RUL, predicted_RUL, abs_error
    eval_df_xgb["abs_error"] = (eval_df_xgb["true_RUL"] - eval_df_xgb["predicted_RUL"]).abs()

    selected_dataset = st.selectbox(
        "Dataset",
        sorted(eval_df_xgb["dataset_id"].unique())
    )

    dataset_eval = eval_df_xgb[eval_df_xgb["dataset_id"] == selected_dataset].copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Engines", len(dataset_eval))
    col2.metric("MAE", f"{dataset_eval['abs_error'].mean():.2f}")
    col3.metric("Worst error", f"{dataset_eval['abs_error'].max():.2f}")
    col4.metric(
        "Critical engines (RUL < 30)",
        int((dataset_eval["true_RUL"] < 30).sum())
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(dataset_eval["true_RUL"], dataset_eval["predicted_RUL"], alpha=0.7)

    min_val = min(dataset_eval["true_RUL"].min(), dataset_eval["predicted_RUL"].min())
    max_val = max(dataset_eval["true_RUL"].max(), dataset_eval["predicted_RUL"].max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--",color="silver")

    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(f"True vs Predicted RUL for Dataset {selected_dataset}")
    #ax.grid(True)

    st.pyplot(fig)

    view_mode = st.radio(
        "Show engines",
        ["Most critical engines", "Best predictions", "Worst predictions"],
        horizontal=True,
    )

    if view_mode == "Worst predictions":
        table_df = dataset_eval.sort_values("abs_error", ascending=False).head(10)
    elif view_mode == "Most critical engines":
        table_df = dataset_eval.sort_values("true_RUL", ascending=True).head(10)
    else:
        table_df = dataset_eval.sort_values("abs_error", ascending=True).head(10)

    display_df = table_df[[
        "dataset_id", "unit_nr", "true_RUL", "predicted_RUL", "abs_error"
    ]].copy()

    display_df["predicted_RUL"] = display_df["predicted_RUL"].round(1)
    display_df["abs_error"] = display_df["abs_error"].round(1)

    st.dataframe(display_df, use_container_width=True)
             
    engine_options = (
        table_df["dataset_id"].astype(str)
        + " / Unit "
        + table_df["unit_nr"].astype(str)
    )

    selected_engine_label = st.selectbox(
        "Inspect one highlighted engine",
        engine_options
    )

    selected_row = table_df.iloc[list(engine_options).index(selected_engine_label)]
    selected_unit = selected_row["unit_nr"]

    unit_df = df_test[
        (df_test["dataset_id"] == selected_dataset)
        & (df_test["unit_nr"] == selected_unit)
    ].sort_values("time_cycles")

    st.subheader(f"{selected_dataset} — Unit {selected_unit}")

    col1, col2, col3 = st.columns(3)
    col1.metric("True RUL", f"{selected_row['true_RUL']:.0f}")
    col2.metric("Predicted RUL", f"{selected_row['predicted_RUL']:.1f}")
    col3.metric("Absolute Error", f"{selected_row['abs_error']:.1f}")

    important_sensors = ["sensor6", "sensor11", "sensor13", "sensor14",  "sensor15"]

    available_sensors = [s for s in important_sensors if s in unit_df.columns]

    fig, ax = plt.subplots(figsize=(8, 4))

    for sensor in available_sensors:
        values = unit_df[sensor]
        normalized = (values - values.mean()) / values.std()
        ax.plot(unit_df["time_cycles"], normalized, label=sensor)

    ax.set_xlabel("Time cycles")
    ax.set_ylabel("Standardized sensor value")
    ax.set_title("Key Sensor Trends (standardized - Z-score)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)