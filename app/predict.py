import joblib
import pandas as pd

from feature_engineering import process_data

model = joblib.load('../models/fraud_model.pkl')

def predict_transaction(transaction):

    df = pd.DataFrame([transaction])

    df.columns = df.columns.str.lower()

    df = process_data(df)

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return {
        "fraud": int(prediction),
        "risk_score": float(probability)
    }