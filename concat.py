
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent

df = pd.read_csv(folder / "car_insurance_claim.csv")

features = [
    "driver_age",
    "past_accidents",
    "speeding_tickets",
    "annual_mileage",
    "claim_filed"
]

df = df[features].copy()

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["claim_filed"].value_counts())

print("\nAverage features by claim:")
print(
    df.groupby("claim_filed")[
        ["driver_age", "past_accidents",
         "speeding_tickets", "annual_mileage"]
    ].mean().round(2)
)

df = df.dropna(subset=features)

df.to_csv(
    folder / "vehicle_insurance_training.csv",
    index=False
)

print("\nSaved: vehicle_insurance_training.csv")
