from fastapi import FastAPI
from pydantic import BaseModel
from getPrediction import get_predictions

app = FastAPI()

class InputData(BaseModel):
    b64imgstr: str

@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

@app.post('/predict')
async def predict(input: InputData):
    b64imgstring = input.b64imgstr
    # Errors
    if (len(b64imgstring) == 0):
        return {"error": "No image provided"}

    # Get the predictions
    preds = get_predictions(b64imgstring)
    if (preds["fire"]):
        confidence = preds["confidence"][0].item()
        confidence = round(confidence * 100, 3)
        return {
            "fire": True,
            "confidence": confidence,
            "image": preds["image"]
        }
    else :
        return {
            "fire": False,
            "image": preds["image"]
        }
    


