from fastapi import FastAPI
from pydantic import BaseModel
import torch
from your_model_code import MenstrualHealthPlatform  # your model class

app = FastAPI()

# Initialize model
platform = MenstrualHealthPlatform()
platform.load_model("menstrual_model.pth")

# Define input structure
class UserData(BaseModel):
    bbt: float
    flow: str
    lh: str
    cycle_day: int
    age: int
    bmi: float

@app.post("/predict")
def predict(data: UserData):
    """Handles menstrual cycle predictions."""
    result = platform.process_daily_data(data.dict())
    return result
