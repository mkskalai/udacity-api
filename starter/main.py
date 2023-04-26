import os
import sys
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from joblib import load
from starter.ml.data import process_data
from starter.ml.model import inference


dir_path = os.path.dirname(os.path.realpath(__file__))

model = load(os.path.join(dir_path, "../starter/model/random_forest.joblib"))
encoder = load(os.path.join(dir_path, "../starter/model/encoder.joblib"))
lb = load(os.path.join(dir_path, "../starter/model/lb.joblib"))

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

app = FastAPI()

class Sample(BaseModel):
    age: str = Field(example="39") 
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example="77516")
    education: str = Field(example="Bachelors")
    education_num: int = Field(alias="education-num", example="13")
    marital_status: str = Field(alias="marital-status", example="Never-married")
    occupation: str = Field(example="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(alias="capital-gain", example="2174")
    capital_loss: int = Field(alias="capital-loss", example="0")
    hours_per_week: int = Field(alias="hours-per-week", example="40")
    native_country: str = Field(alias="native-country", example="United-States")

@app.get("/")
async def welcome():
    return "Hello adventurer!"

@app.post("/predict/")
async def make_prediction(sample: Sample):
    X = pd.DataFrame(sample.dict(by_alias=True), index=[0])
    X, _, _, _ = process_data(X, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb)
    y = inference(model, X)
    return {"Prediction": int(y)}
