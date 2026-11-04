# BITS AIML – Machine Learning Assignment 2

**Student:** KEERTHANA K  
**BITS ID:** 2026AJ05404

## Problem statement
Implement and compare five classification algorithms on one public binary classification dataset: Logistic Regression, Decision Tree, K-Nearest Neighbours (KNN), Gaussian Naive Bayes and Support Vector Machine (SVM). Evaluate each model using Accuracy, AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC).

## Dataset description
The UCI Breast Cancer Wisconsin (Diagnostic) dataset contains 569 observations and 30 numerical input features. The target is `diagnosis`, where 0 = malignant and 1 = benign. Dataset source: [UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). The training script uses the equivalent dataset bundled with scikit-learn for reproducible execution.

## Links
- GitHub repository: https://github.com/2026aj05404-dot/ML_Assignment
- Live Streamlit application: https://mlassignment-2026aj05404.streamlit.app/

## Methodology
An 80:20 stratified train/test split is used with `random_state=42`; the test set contains 114 records. StandardScaler is included in the pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. Decision Tree is used without scaling. Complete pipelines are saved so preprocessing is applied consistently at prediction time.

Precision, Recall, F1 and AUC treat malignant (class 0) as the positive class. Accuracy and MCC use the original class labels.

## Model comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.8478 | 0.9286 | 0.8864 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9512 | 0.9286 | 0.9398 | 0.9054 |
| Gaussian Naive Bayes | 0.9298 | 0.9868 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| SVM | 0.9825 | 0.9950 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |

## Observations and overall winner
Logistic Regression and SVM achieved the highest accuracy (0.9825), and tied on Precision, Recall, F1 and MCC. Logistic Regression achieved a slightly higher AUC (0.9954) than SVM (0.9950), so Logistic Regression is the overall winner on this test split when AUC is used as the tie-breaker. KNN also performed strongly. Gaussian Naive Bayes and Decision Tree scored lower on this test split. These results are specific to this split and do not guarantee performance on other data.

## Repository contents
- `train_models.py` – trains and evaluates the five classifiers.
- `app.py` – Streamlit interface that loads saved models and evaluates uploaded CSV data without retraining.
- `models/` (or saved model files in the repository root) – five saved trained pipelines.
- `test_data.csv` – labelled held-out test sample for testing the app.
- `artifacts/model_metrics.csv` – model comparison metrics.
- `artifacts/dataset_info.txt` – dataset and target information.
- `requirements.txt` – Python dependencies.

## Run locally
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Streamlit application
The app accepts a CSV containing the 30 feature columns. Include the `diagnosis` column (0 or 1) to calculate all six metrics, a confusion matrix and a classification report. Without the target column, it displays predictions only. The deployed application has been tested with `test_data.csv`; model selection, CSV upload and metric display are working.

## Virtual Lab evidence
The BITS Virtual Lab execution screenshot is not included in this project package and must be captured separately during a genuine execution in the BITS Virtual Lab, as required by the assignment.
