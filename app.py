"""Streamlit interface: load saved models and evaluate uploaded CSV data."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
TARGET = "diagnosis"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
    "SVM": "svm.pkl",
}

st.set_page_config(page_title="AIML Classification Evaluation", page_icon="📊", layout="wide")
st.title("Machine Learning Classification Model Evaluation")
st.write("Upload a CSV containing the 30 feature columns. Include `diagnosis` to calculate evaluation metrics.")
st.caption("Dataset: Breast Cancer Wisconsin (Diagnostic). Target: 0 = malignant, 1 = benign.")

missing = [file for file in MODEL_FILES.values() if not (MODEL_DIR / file).exists()]
if missing:
    st.error("Saved models are missing. Run `python train_models.py` before starting the app.")
    st.stop()

selected_model = st.selectbox("Select a trained classifier", list(MODEL_FILES))
uploaded = st.file_uploader("Upload test data in CSV format", type=["csv"])

@st.cache_resource
def load_saved_model(model_path):
    return joblib.load(model_path)

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        st.subheader("Data preview")
        st.dataframe(data.head())

        model = load_saved_model(MODEL_DIR / MODEL_FILES[selected_model])
        expected_columns = list(model.feature_names_in_)

        if TARGET in data.columns:
            y_true = data[TARGET].astype(int)
            features = data.drop(columns=[TARGET])
        else:
            y_true = None
            features = data.copy()

        missing_columns = [column for column in expected_columns if column not in features.columns]
        extra_columns = [column for column in features.columns if column not in expected_columns]
        if missing_columns:
            st.error("Missing required columns: " + ", ".join(missing_columns))
            st.stop()
        if extra_columns:
            st.warning("Extra columns will be ignored: " + ", ".join(extra_columns))
        features = features[expected_columns]

        predictions = model.predict(features)
        output = features.copy()
        output["predicted_diagnosis"] = predictions
        output["predicted_label"] = output["predicted_diagnosis"].map({0: "malignant", 1: "benign"})

        st.subheader("Predictions")
        st.dataframe(output)

        if y_true is not None:
            if not set(y_true.unique()).issubset({0, 1}):
                st.error("The diagnosis target must contain only 0 or 1.")
                st.stop()

            probability = model.predict_proba(features)[:, list(model.classes_).index(0)]
            actual_malignant = (y_true == 0).astype(int)
            predicted_malignant = (predictions == 0).astype(int)
            metric_values = {
                "Accuracy": accuracy_score(y_true, predictions),
                "AUC": roc_auc_score(actual_malignant, probability),
                "Precision": precision_score(actual_malignant, predicted_malignant, zero_division=0),
                "Recall": recall_score(actual_malignant, predicted_malignant, zero_division=0),
                "F1": f1_score(actual_malignant, predicted_malignant, zero_division=0),
                "MCC": matthews_corrcoef(y_true, predictions),
            }

            st.subheader("Evaluation metrics")
            metric_cols = st.columns(3)
            for column, (label, value) in zip(metric_cols * 2, metric_values.items()):
                column.metric(label, f"{value:.4f}")

            st.subheader("Confusion matrix")
            matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
            st.write("Rows = actual; columns = predicted. Class order: malignant, benign.")
            st.dataframe(pd.DataFrame(
                matrix,
                index=["Actual malignant", "Actual benign"],
                columns=["Predicted malignant", "Predicted benign"]
            ))

            st.subheader("Classification report")
            report = classification_report(
                y_true, predictions, labels=[0, 1],
                target_names=["malignant", "benign"],
                zero_division=0, output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4))
        else:
            st.info("No `diagnosis` column was supplied. Predictions are available, but metrics require true labels.")

        st.download_button(
            "Download predictions CSV",
            output.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv"
        )
    except Exception as error:
        st.error(f"Unable to process this CSV: {error}")
else:
    st.info("You can test the app with the included `test_data.csv` file.")
