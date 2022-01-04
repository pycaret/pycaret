import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("boston_api")

# Define predict function
@app.post("/predict")
def predict(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat):
    data = pd.DataFrame(
        [[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat]]
    )
    data.columns = [
        "crim",
        "zn",
        "indus",
        "chas",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "black",
        "lstat",
    ]
    predictions = predict_model(model, data=data)
    return {"prediction": list(predictions["Label"])}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
