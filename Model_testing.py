import joblib
import pandas as pd
import numpy as np

# Load saved model
model = joblib.load("churn_model.pkl")

# Load preprocessing objects
scaler = joblib.load("scaler.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
contract_encoder = joblib.load("contract_encoder.pkl")
payment_encoder = joblib.load("payment_encoder.pkl")


def predict_churn(age, gender, tenure, monthly_charges,
                  contract, payment_method, total_charges):

    customer = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "TotalCharges": total_charges
    }])

    # Scale numerical features
    numerical_scaled = scaler.transform(
        customer[["Age", "Tenure", "MonthlyCharges", "TotalCharges"]]
    )

    # Encode categorical features
    gender_encoded = gender_encoder.transform(
        customer[["Gender"]]
    )

    contract_encoded = contract_encoder.transform(
        customer[["Contract"]]
    )

    payment_encoded = payment_encoder.transform(
        customer[["PaymentMethod"]]
    )

    # Combine all features
    X_new = np.hstack([
        numerical_scaled,
        gender_encoded,
        contract_encoded,
        payment_encoded
    ])

    # Convert to DataFrame with feature names
    X_new = pd.DataFrame(
        X_new,
        columns=model.feature_names_in_
    )

    # Prediction
    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0][1]

    return prediction, probability


# Test customer
prediction, probability = predict_churn(
    35,
    "Male",
    8,
    75.50,
    "Month-to-month",
    "Credit card",
    604.00
)

print("\n--- Customer Churn Prediction ---")

if prediction == 1:
    print("Prediction: Customer will churn")
else:
    print("Prediction: Customer will not churn")

print(f"Churn probability: {probability:.2%}")