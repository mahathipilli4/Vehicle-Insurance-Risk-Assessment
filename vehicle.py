import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# =======================================================
# VEHICLE INSURANCE RISK ASSESSMENT
# =======================================================

print("=" * 55)
print("       VEHICLE INSURANCE RISK ASSESSMENT")
print("=" * 55)


# 1. LOAD BOTH ORIGINAL DATASETS

df1 = pd.read_csv("car_insurance_claim.csv")
df2 = pd.read_csv("Car_insurance_new.csv")

print("\n1. ORIGINAL DATASET DETAILS")
print("-" * 40)
print("Old Dataset: car_insurance_claim.csv")
print("Rows:", df1.shape[0])
print("Columns:", df1.shape[1])

print("\nNew Dataset: Car_insurance_new.csv")
print("Rows:", df2.shape[0])
print("Columns:", df2.shape[1])


# 2. STANDARDIZE COMMON FEATURES

print("\n2. STANDARDIZING COMMON FEATURES")
print("-" * 40)

age_mapping = {
    "16-25": 20.5,
    "26-39": 32.5,
    "40-64": 52.0,
    "65+": 70.0
}

df2["AGE"] = df2["AGE"].map(age_mapping)

common_columns = [
    "driver_age",
    "past_accidents",
    "speeding_tickets",
    "annual_mileage",
    "claim_filed"
]

data1 = df1[common_columns].copy()

data2 = df2.rename(columns={
    "AGE": "driver_age",
    "PAST_ACCIDENTS": "past_accidents",
    "SPEEDING_VIOLATIONS": "speeding_tickets",
    "ANNUAL_MILEAGE": "annual_mileage",
    "OUTCOME": "claim_filed"
})[common_columns].copy()

print("Common features selected:")
print(common_columns)


# 3. CONCATENATE BOTH DATASETS

data1["data_source"] = "Old Dataset"
data2["data_source"] = "New Kaggle Dataset"

print("\n3. CONCATENATING DATASETS")
print("-" * 40)

combined_df = pd.concat(
    [data1, data2],
    ignore_index=True
)

print("Concatenation completed successfully!")
print("Old dataset records:", len(data1))
print("New dataset records:", len(data2))
print("Total records:", len(combined_df))
print("Total columns:", len(combined_df.columns))
print("Combined dataset shape:", combined_df.shape)


# 4. MISSING VALUES

print("\n4. MISSING VALUES BEFORE PREPROCESSING")
print("-" * 40)
print(combined_df.isnull().sum())
print("\nTotal missing values:", combined_df.isnull().sum().sum())


# 5. COMBINED DATASET PREVIEW

print("\n5. COMBINED DATASET PREVIEW")
print("-" * 40)
print(combined_df.head(10).to_string(index=False))


# 6. RECORDS FROM EACH DATASET

print("\n6. RECORDS FROM EACH DATASET")
print("-" * 40)
print(combined_df["data_source"].value_counts())


# 7. SAVE COMBINED DATASET

combined_df.to_csv("combined_vehicle_insurance.csv", index=False)

print("\nCombined dataset saved:")
print("combined_vehicle_insurance.csv")
print("\n========== CONCATENATION COMPLETED ==========")


# MODEL TRAINING — NO TRAINING/EVALUATION OUTPUT

feature_columns = [
    "driver_age",
    "past_accidents",
    "speeding_tickets",
    "annual_mileage"
]

target_column = "claim_filed"

for col in feature_columns + [target_column]:
    combined_df[col] = pd.to_numeric(
        combined_df[col],
        errors="coerce"
    )

combined_df = combined_df.dropna(subset=[target_column])
combined_df = combined_df[
    combined_df[target_column].isin([0, 1])
].copy()

combined_df[target_column] = combined_df[target_column].astype(int)

X = combined_df[feature_columns]
y = combined_df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)


# 8. ML PREDICTION WITH RISK LEVEL

print("\n8. VEHICLE INSURANCE CLAIM PREDICTION")
print("-" * 40)

try:
    driver_age = float(input("Enter driver age: "))
    past_accidents = int(input("Enter number of past accidents: "))
    speeding_tickets = int(input("Enter number of speeding tickets: "))
    annual_mileage = float(input("Enter annual mileage: "))
    if not 18 <= driver_age <= 100:
        raise ValueError("Driver age must be between 18 and 100.")
    if past_accidents < 0 or speeding_tickets < 0:
        raise ValueError("Accident and ticket counts cannot be negative.")
    if annual_mileage < 0:
        raise ValueError("Annual mileage cannot be negative.")
    new_data = pd.DataFrame([{
        "driver_age": driver_age,
        "past_accidents": past_accidents,
        "speeding_tickets": speeding_tickets,
        "annual_mileage": annual_mileage
    }])

    prediction = model.predict(new_data)[0]
    probabilities = model.predict_proba(new_data)[0]
    claim_index = list(model.classes_).index(1)
    claim_probability = probabilities[claim_index] * 100
    print("\nPREDICTION RESULT")
    if prediction == 1:
        print("Prediction: Claim Filed")
    else:
        print("Prediction: No Claim Filed")

    print(f"Estimated probability of claim: {claim_probability:.2f}%")
   # Risk classification based on predicted claim probability
    if claim_probability >= 50:
        print("Risk Level: HIGH RISK")
    else:
        print("Risk Level: LOW RISK")
except ValueError as error:
    print("Invalid input:", error)
# 9. UNDERWRITING ASSESSMENT
print("\n9. UNDERWRITING ASSESSMENT")
print("-" * 40)
owner = input(
    "Is the vehicle owner the policyholder? (yes/no): "
).strip().lower()
driver = input(
    "Who was driving? (owner/friend/family): "
).strip().lower()
license_valid = input(
    "Does the actual driver have a valid license? (yes/no): "
).strip().lower()
policy_type = input(
    "Policy type (comprehensive/third_party): "
).strip().lower()
vehicle_damaged = input(
    "Was the insured vehicle damaged? (yes/no): "
).strip().lower()
driver_injured = input(
    "Was the driver injured? (yes/no): "
).strip().lower()
third_party_injured = input(
    "Was a third party injured? (yes/no): "
).strip().lower()
# 10. UNDERWRITING RULES
print("\n10. UNDERWRITING REVIEW")
print("-" * 40)
valid_yes_no = ["yes", "no"]
valid_drivers = ["owner", "friend", "family"]
valid_policies = ["comprehensive", "third_party"]

if owner not in valid_yes_no:
    print("Invalid owner input.")

elif driver not in valid_drivers:
    print("Invalid driver input.")

elif license_valid not in valid_yes_no:
    print("Invalid license input.")

elif policy_type not in valid_policies:
    print("Invalid policy type.")

elif any(value not in valid_yes_no for value in [
    vehicle_damaged,
    driver_injured,
    third_party_injured
]):
    print("Invalid accident details.")

else:
    manual_review = False

    if owner == "no":
        print("Verify policyholder and vehicle ownership details.")
        manual_review = True

    if driver != "owner":
        print("Actual driver differs from the owner.")
        print("Check permitted-driver conditions and policy terms.")
        manual_review = True

    if license_valid == "no":
        print("Valid driving licence not confirmed.")
        print("Review applicable policy conditions.")
        manual_review = True

    if policy_type == "third_party" and vehicle_damaged == "yes":
        print("Own vehicle damage is generally not covered")
        print("under third-party-only insurance.")
        print("Check the actual policy wording.")
        manual_review = True

    if driver_injured == "yes":
        print("Driver injury reported.")
        print("Check applicable personal accident or medical coverage.")
        manual_review = True

    if third_party_injured == "yes":
        print("Third-party injury reported.")
        print("Review liability coverage and policy conditions.")
        manual_review = True

    if manual_review:
        print("\nUnderwriting Result: MANUAL REVIEW REQUIRED")
    else:
        print(
            "\nUnderwriting Result: "
            "FURTHER POLICY VERIFICATION REQUIRED"
        )
print("\n========== PROCESS COMPLETED ==========")