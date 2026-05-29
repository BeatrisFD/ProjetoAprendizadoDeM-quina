from fastapi import FastAPI
from predict import predict_transaction
from database import transactions

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Fraud Detection API Online"
    }

@app.post("/predict")
def predict(data: dict):

    result = predict_transaction(data)

    data["prediction"] = result

    transactions.insert_one(data)

    return result