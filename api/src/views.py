from fastapi import FastAPI
from fastapi.responses import JSONResponse
from models import PredictionRequest,PredictionInput,DataInput
import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


# base model ;)

with open("recurrence_model.pkl", "rb") as file:
    model = pickle.load(file)

model_ = None


app = FastAPI()




@app.post("/predict/")
async def predict(request: PredictionRequest):

    """
    Return prediction based on age, tumor_size, smoking status.
    """
    input_data = np.array([[request.age, request.tumor_size, request.smoking_status]])
    prediction = model.predict(input_data)
    result = "High Risk of Recurrence" if prediction[0] == 1 else "Low Risk of Recurrence"


    return JSONResponse(content={"prediction": result})





@app.post("/train/")
async def train_model(data: DataInput):
    """
    Train the model with the given data and save it.
    """
    X = pd.DataFrame(data.X)
    y = pd.Series(data.y)


    new_model = LinearRegression()
    new_model.fit(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump(new_model, f)

    global model_
    model_ = new_model  

    return {"message": "Model trained successfully!"}

@app.post("/predict2/")
async def predict_from_trained(data: PredictionInput):
    """
    Predict using the most recently trained model.
    """
    if model_ is None:
        return JSONResponse(content={"error": "Model not trained!"}, status_code=400)
    
    predictions = model_.predict(data.X)
    
    return {"predictions": predictions.tolist()}
