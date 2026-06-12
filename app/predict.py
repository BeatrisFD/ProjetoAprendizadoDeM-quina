import joblib
import pandas as pd

from feature_engineering import process_data


def predict_transaction(transaction, model_path):

    model = joblib.load(model_path)

    df = pd.DataFrame([transaction])

    df.columns = df.columns.str.lower()

    df = process_data(df)

    expected_columns = model.feature_names_in_

    df = df[expected_columns]

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return {
        "fraud": int(prediction),
        "risk_score": float(probability)
    }