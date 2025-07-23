# Import Libraries
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

# Declare FastAPI
app = FastAPI()

# Get Pickle File
with open("Obesity Level Prediction Classification Model Deployment in Machine Learning with Python Pickle File.pkl", "rb") as f:
    model = pickle.load(f)

class ObesityInput(BaseModel):
    Gender: int = 0
    Age: int = 23
    Height: float = 1.50
    Weight: float = 50.0
    family_history_with_overweight: int = 1
    FAVC: int = 1
    FCVC: float = 2.5
    NCP: float = 3.0
    CAEC: int = 1
    SMOKE: int = 0
    CH2O: float = 2.0
    SCC: int = 0
    FAF: float = 1.5
    TUE: float = 1.0
    CALC: int = 1
    MTRANS: int = 1

class PredictionResponse(BaseModel):
    prediction: int

# Make Endpoint
@app.post("/predict", 
          response_model = PredictionResponse)

# Predict
def predict(input_data: ObesityInput):
    try:
        BMI = input_data.Weight / (input_data.Height ** 2)
        features = [[
            input_data.Gender,
            input_data.Age,
            input_data.Height,
            input_data.Weight,
            input_data.family_history_with_overweight,
            input_data.FAVC,
            input_data.FCVC,
            input_data.NCP,
            input_data.CAEC,
            input_data.SMOKE,
            input_data.CH2O,
            input_data.SCC,
            input_data.FAF,
            input_data.TUE,
            input_data.CALC,
            input_data.MTRANS,
            BMI
        ]]

        prediction = model.predict(features)
        return PredictionResponse(prediction = int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code = 500, 
                            detail = f"Prediction failed: {str(e)}")

# Run FastAPI
if __name__ == "__main__":
    uvicorn.run(app, 
                host = "127.0.0.1", 
                port = 7000)