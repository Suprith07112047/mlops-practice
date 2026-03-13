from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Diabetes Prediction API")

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input schema (match dataset column order!)
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict(data: PatientData):
    # Convert input to DataFrame with correct column names
    features = pd.DataFrame([[data.Pregnancies, data.Glucose, data.BloodPressure,
                              data.SkinThickness, data.Insulin, data.BMI,
                              data.DiabetesPedigreeFunction, data.Age]],
                            columns=["Pregnancies","Glucose","BloodPressure","SkinThickness",
                                     "Insulin","BMI","DiabetesPedigreeFunction","Age"])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]

    return {"diabetic": bool(prediction)}
