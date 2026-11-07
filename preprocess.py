
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent
input_file = folder / "combined_vehicle_insurance.csv"
output_file = folder / "preprocessed_vehicle_insurance.csv"

df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)

required_columns = [
    "driver_age",
    "past_accidents",
    "speeding_tickets",
    "annual_mileage",
    "claim_filed",
    "data_source"
]

df = df[required_columns].copy()

numeric_columns = [
    "driver_age",
    "past_accidents",
    "speeding_tickets",
    "annual_mileage",
    "claim_filed"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nMissing values before cleaning:")
print(df.isnull().sum())

df = df.dropna(subset=["claim_filed"]).copy()
df = df[df["claim_filed"].isin([0, 1])].copy()
df["claim_filed"] = df["claim_filed"].astype(int)

duplicates = df.duplicated().sum()
print("\nDuplicate records found:", duplicates)

df = df.drop_duplicates().copy()

print("\nMissing feature values:")
print(df[numeric_columns[:-1]].isnull().sum())

print("\nFinal dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["claim_filed"].value_counts())

df.to_csv(output_file, index=False)

print("\nSaved:", output_file.name)
