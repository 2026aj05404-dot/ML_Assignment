"""Train and evaluate five classifiers for the BITS AIML Assignment 2."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

# Public Breast Cancer Wisconsin (Diagnostic) dataset.
# Target mapping in scikit-learn: 0 = malignant, 1 = benign.
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target.astype(int).rename("diagnosis")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Gaussian Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
    ])
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, MODEL_DIR / filename)

    prediction = model.predict(X_test)
    malignant_probability = model.predict_proba(X_test)[:, list(model.classes_).index(0)]
    actual_malignant = (y_test == 0).astype(int)
    predicted_malignant = (prediction == 0).astype(int)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, prediction),
        "AUC": roc_auc_score(actual_malignant, malignant_probability),
        "Precision": precision_score(actual_malignant, predicted_malignant, zero_division=0),
        "Recall": recall_score(actual_malignant, predicted_malignant, zero_division=0),
        "F1": f1_score(actual_malignant, predicted_malignant, zero_division=0),
        "MCC": matthews_corrcoef(y_test, prediction)
    })

test_data = X_test.copy()
test_data["diagnosis"] = y_test
test_data.to_csv(ROOT / "test_data.csv", index=False)
pd.DataFrame(results).to_csv(ARTIFACT_DIR / "model_metrics.csv", index=False)

print("Finished training all five models.")
print(pd.DataFrame(results).round(4).to_string(index=False))
