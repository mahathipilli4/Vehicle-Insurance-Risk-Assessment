import joblib
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent
model = joblib.load(folder / "vehicle_insurance_model.pkl")

print("=== Vehicle Insurance Risk Assessment ===")

try:
    driver_age = int(input("Enter driver age: "))
    past_accidents = int(input("Enter number of past accidents: "))
    speeding_tickets = int(input("Enter number of speeding tickets: "))
    annual_mileage = float(input("Enter annual mileage: "))

    if not 18 <= driver_age <= 100:
        raise ValueError("Driver age must be between 18 and 100.")

    if past_accidents < 0 or speeding_tickets < 0 or annual_mileage < 0:
        raise ValueError("Accidents, tickets, and mileage cannot be negative.")

    customer = pd.DataFrame([{
        "driver_age": driver_age,
        "past_accidents": past_accidents,
        "speeding_tickets": speeding_tickets,
        "annual_mileage": annual_mileage
    }])

    prediction = int(model.predict(customer)[0])
    probabilities = model.predict_proba(customer)[0]
    classes = list(model.classes_)
    claim_probability = probabilities[classes.index(1)] * 100

    print("\n=== Assessment Result ===")
    print("Predicted class:", prediction)
    print(
        "Prediction:",
        "Claim Filed (1)" if prediction == 1 else "No Claim (0)"
    )
    print(f"Model-estimated claim probability: {claim_probability:.2f}%")

    print("\nNote: This is an educational ML estimate,")
    print("not a guarantee or an actual insurance decision.")

except ValueError as error:
    print("Invalid input:", error)