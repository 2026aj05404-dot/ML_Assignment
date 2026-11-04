# BITS AIML – Machine Learning Assignment 2

**Student:** KEERTHANA K  
**BITS ID:** 2026AJ05404

## Problem statement
Compare five classification algorithms on the Breast Cancer Wisconsin (Diagnostic) dataset: Logistic Regression, Decision Tree, K-Nearest Neighbours (KNN), Gaussian Naive Bayes and Support Vector Machine (SVM).

## Dataset description
The dataset contains 569 observations and 30 numerical features. The target is `diagnosis`, where 0 = malignant and 1 = benign. Dataset source: [UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). The training script uses the equivalent dataset bundled with scikit-learn for repeatable execution.

## Repository contents
- `train_models.py` – trains and evaluates the five classifiers.
- `app.py` – Streamlit application; loads saved models and evaluates uploaded CSV files.
- `models/` – saved trained pipelines.
- `test_data.csv` – labelled held-out sample for testing the app.
- `artifacts/model_metrics.csv` – metrics produced by the training script.
- `requirements.txt` – Python dependencies.

## Methodology
An 80:20 stratified train/test split is used with `random_state=42`. StandardScaler is included in the pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. Decision Tree is used without scaling. Complete pipelines are saved so preprocessing is applied consistently at prediction time.

Precision, Recall, F1 and AUC treat malignant (class 0) as the positive class. Accuracy and MCC use the original class labels.

## Model comparison
Run `python train_models.py` to generate `artifacts/model_metrics.csv`. Use the results from your own execution in your submitted report. Explain which classifier performed best and why, based on the metrics.

## Run locally
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Streamlit app
The app accepts a CSV containing the 30 feature columns. Include the `diagnosis` column (0 or 1) to calculate Accuracy, AUC, Precision, Recall, F1, MCC, a confusion matrix and a classification report. Without the target column, it displays predictions only. The app loads saved models and does not retrain them.

## Submission links
- GitHub repository: https://github.com/2026aj05404-dot/ML_Assignment
- Streamlit Community Cloud: add the deployed app URL after deployment.

## Academic note
Run the project yourself in BITS Virtual Lab, inspect the outputs, and write your own observations. Add your genuine BITS Lab screenshot separately as required by the assignment.