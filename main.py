# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 21:39:06 2024

@author: Rajin
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class modelInput(BaseModel):
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

@app.post('/parkinsons_detection')
def parkinsons_detection(input_parameters: modelInput):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    mdvp_fo = input_dictionary['MDVP_Fo_Hz']
    mdvp_flo = input_dictionary['MDVP_Flo_Hz']
    mdvp_shimmer = input_dictionary['MDVP_Shimmer']
    shimmer_apq5 = input_dictionary['Shimmer_APQ5']
    mdvp_apq = input_dictionary['MDVP_APQ']
    hnr = input_dictionary['HNR']
    sp1 = input_dictionary['spread1']
    sp2 = input_dictionary['spread2']
    ppe = input_dictionary['PPE']
    
   
    input_list = [mdvp_fo, mdvp_flo, mdvp_shimmer, shimmer_apq5, mdvp_apq, hnr, sp1, sp2, ppe]

    detection = parkinsons_model.predict([input_list])
    
    if detection[0] == 0:
        return "The person doesn't have Parkinson's Disease"
    else:
        return "The person has Parkinson's Disease"
