import joblib
import numpy as np
import pandas as pd

model = joblib.load("deployment/fraud_model.pkl")
scaler = joblib.load("deployment/scaler.pkl")
feature_columns = joblib.load("deployment/feature_columns.pkl")

def predict_fraud(amount, hour, v_features, threshold=0.3):
    amount_log = np.log1p(amount)
    data = v_features + [amount_log, hour]

    input_df = pd.DataFrame([data], columns=feature_columns)
    input_df[["Amount_log", "Hour"]] = scaler.transform(input_df[["Amount_log", "Hour"]])

    prob = model.predict_proba(input_df)[0][1]
    is_fraud = prob >= threshold

    return {
        "fraud_probability": round(float(prob), 4),
        "prediction": "FRAUD" if is_fraud else "LEGITIMATE",
        "action": "Block & Alert" if is_fraud else "Approve"
    }

if __name__ == "__main__":
    example_v = [0.0] * 28
    result = predict_fraud(amount=150.0, hour=14, v_features=example_v)
    print("Example Prediction:")
    print(result)
