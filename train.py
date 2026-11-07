
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

folder = Path(__file__).resolve().parent

# Load the prepared dataset
df = pd.read_csv(folder / "vehicle_insurance_training.csv")

# Select input features and target
X = df[
    [
        "driver_age",
        "past_accidents",
        "speeding_tickets",
        "annual_mileage"
    ]
]

y = df["claim_filed"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# Save split datasets
X_train.assign(claim_filed=y_train).to_csv(
    folder / "X_train.csv", index=False
)

X_test.assign(claim_filed=y_test).to_csv(
    folder / "X_test.csv", index=False
)

print("\nTrain-test split completed successfully!")
print("Saved X_train.csv and X_test.csv")
