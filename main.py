# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 21:39:06 2024

@author: Rajin
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelInput(BaseModel):
    MDVP_Fo_Hz: float
    MDVP_Flo_Hz: float
    MDVP_Shimmer: float
    Shimmer_APQ5 : float
    MDVP_APQ : float
    HNR: float
    spread1: float
    spread2: float
    PPE: float

# loading the saved model
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

@app.post('/parkinsons_detection')
def parkinsons_detection(input_parameters: ModelInput):
    input_data = [
        input_parameters.MDVP_Fo_Hz,
        input_parameters.MDVP_Flo_Hz,
        input_parameters.MDVP_Shimmer,
        input_parameters.Shimmer_APQ5,
        input_parameters.MDVP_APQ,
        input_parameters.HNR,
        input_parameters.spread1,
        input_parameters.spread2,
        input_parameters.PPE
    ]
   
    input_data_array = np.asarray(input_data)
    input_data_reshaped = input_data_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)

    detection = parkinsons_model.predict(std_data)
    
    if detection[0] == 0:
        return "The person doesn't have Parkinson's Disease"
    else:
        return "The person has Parkinson's Disease"



