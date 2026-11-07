
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. PROJECT FILE PATHS
# ==========================================

folder = Path(__file__).resolve().parent

model_path = folder / "vehicle_insurance_model.pkl"
test_path = folder / "X_test.csv"

# Check whether required files exist
if not model_path.exists():
    raise FileNotFoundError(
        "vehicle_insurance_model.pkl not found. "
        "Run modeltrain.py first."
    )

if not test_path.exists():
    raise FileNotFoundError(
        "X_test.csv not found. Run train.py first."
    )

# ==========================================
# 2. LOAD MODEL AND TEST DATA
# ==========================================

model = joblib.load(model_path)
test_df = pd.read_csv(test_path)

features = [
    "driver_age",
    "past_accidents",
    "speeding_tickets",
    "annual_mileage"
]

target = "claim_filed"

missing_columns = [
    column for column in features + [target]
    if column not in test_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in X_test.csv: {missing_columns}"
    )

X_test = test_df[features]
y_test = test_df[target]

# ==========================================
# 3. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 4. CALCULATE PERFORMANCE METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

# Calculate ROC-AUC when possible
roc_auc = None

try:
    if (
        hasattr(model, "predict_proba")
        and y_test.nunique() == 2
        and 1 in model.classes_
    ):
        classes = list(model.classes_)
        probabilities = model.predict_proba(X_test)[
            :, classes.index(1)
        ]

        roc_auc = roc_auc_score(y_test, probabilities)

    elif (
        hasattr(model, "decision_function")
        and y_test.nunique() == 2
    ):
        scores = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, scores)

except (ValueError, AttributeError):
    roc_auc = None

# ==========================================
# 5. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

# ==========================================
# 6. CLASSIFICATION REPORT
# ==========================================

report_text = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Claim (0)", "Claim Filed (1)"],
    zero_division=0
)

report_dict = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Claim (0)", "Claim Filed (1)"],
    output_dict=True,
    zero_division=0
)

# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 48)
print(" VEHICLE INSURANCE MODEL PERFORMANCE")
print("=" * 48)

print(f"\nTest Samples : {len(y_test)}")
print(f"Accuracy     : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1-Score     : {f1:.4f}")

if roc_auc is not None:
    print(f"ROC-AUC      : {roc_auc:.4f}")
else:
    print("ROC-AUC      : Not available")

print("\nConfusion Matrix")
print("Rows = Actual [0, 1]")
print("Columns = Predicted [0, 1]")
print(cm)

print("\nClassification Report")
print(report_text)

# ==========================================
# 8. SAVE MODEL METRICS
# ==========================================

metrics_df = pd.DataFrame([{
    "test_samples": len(y_test),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc
}])

metrics_df.to_csv(
    folder / "model_metrics.csv",
    index=False
)

# ==========================================
# 9. SAVE CONFUSION MATRIX
# ==========================================

confusion_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

confusion_df.to_csv(
    folder / "confusion_matrix.csv",
    index=True
)

# ==========================================
# 10. SAVE CLASSIFICATION REPORT
# ==========================================

report_df = pd.DataFrame(report_dict).transpose()

report_df.index.name = "Class"

report_df.to_csv(
    folder / "classification_report.csv",
    index=True
)

# ==========================================
# 11. COMPLETION MESSAGE
# ==========================================

print("\nSaved model_metrics.csv")
print("Saved confusion_matrix.csv")
print("Saved classification_report.csv")

print("\nModel evaluation completed successfully!")
