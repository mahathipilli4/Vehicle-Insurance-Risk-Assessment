
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

folder = Path(__file__).resolve().parent

# Load train and test datasets
train_df = pd.read_csv(folder / "X_train.csv")
test_df = pd.read_csv(folder / "X_test.csv")

X_train = train_df.drop(columns=["claim_filed"])
y_train = train_df["claim_filed"]

X_test = test_df.drop(columns=["claim_filed"])
y_test = test_df["claim_filed"]

# Logistic Regression with feature scaling
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Performance metrics
print("\n===== LOGISTIC REGRESSION PERFORMANCE =====")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Claim (0)", "Claim Filed (1)"],
    zero_division=0
))

# Save the trained model
joblib.dump(model, folder / "vehicle_insurance_model.pkl")

print("\nLogistic Regression model saved successfully!")
