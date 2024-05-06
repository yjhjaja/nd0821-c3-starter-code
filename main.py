from fastapi  import FastAPI
from pydantic import BaseModel, Field
import os, pickle, sys
import numpy  as np
import pandas as pd

path_proj = '/Users/junhui.yang/Library/CloudStorage/Dropbox/udacity/nd0821-c3-starter-code-master/'
sys.path.insert(1, path_proj + 'ml/')
from model import inference
from data  import process_data
        
app = FastAPI()

model   = pickle.load(open(path_proj + 'model/model.pkl'  , 'rb'))
encoder = pickle.load(open(path_proj + 'model/encoder.pkl', 'rb'))
lb      = pickle.load(open(path_proj + 'model/lb.pkl'     , 'rb'))

class InputData(BaseModel):
    # Example: first row of census.csv
    age:            int = Field(example = 39)
    workclass:      str = Field(example = 'State-gov')
    fnlgt:          int = Field(example = 77516)
    education:      str = Field(example = 'Bachelors')
    education_num:  int = Field(example = 13)
    marital_status: str = Field(example = 'Never-married')
    occupation:     str = Field(example = 'Adm-clerical')
    relationship:   str = Field(example = 'Not-in-family')
    race:           str = Field(example = 'White')
    sex:            str = Field(example = 'Male')
    capital_gain:   int = Field(example = 2174)
    capital_loss:   int = Field(example = 0)
    hours_per_week: int = Field(example = 40)
    native_country: str = Field(example = 'United-States')

@app.get('/')
async def greet():
    return 'Hello, welcome!'

@app.post('/inference')
async def perform_inference(data: InputData):
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
    example = {key.replace('_', '-'): [value] for key, value in data.__dict__.items()}
    input_data = pd.DataFrame.from_dict(example)
    X_test, _, _, _ = process_data(input_data, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb)
    pred = inference(model, X_test)[0]
    return str(pred)
