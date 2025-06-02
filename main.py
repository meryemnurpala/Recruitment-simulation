from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# API nesnesi
app = FastAPI()

# Model ve scaler yükleniyor
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# Giriş verisi formatı
class CandidateInput(BaseModel):
    experience: float
    score: float

@app.get("/")
def read_root():
    return {"message": "API is working!"}


# Ana endpoint
@app.post("/predict")
def predict(candidate: CandidateInput):
    data = np.array([[candidate.experience, candidate.score]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    result = "HIRED ✅" if prediction == 1 else "NOT HIRED ❌"
    return {"prediction": result}
