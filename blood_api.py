import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("blood_api")

# Define predict function
@app.post("/predict")
def predict(Recency, Frequency, Monetary, Time):
    data = pd.DataFrame([[Recency, Frequency, Monetary, Time]])
    data.columns = ["Recency", "Frequency", "Monetary", "Time"]
    predictions = predict_model(model, data=data)
    return {"prediction": list(predictions["Label"])}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
