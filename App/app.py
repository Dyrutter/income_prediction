from fastapi import FastAPI, HTTPException
import joblib, numpy, json
from pydantic import BaseModel, Field 
from fastapi.encoders import jsonable_encoder
import pandas as pd
from App.data import process_data

class Item(BaseModel):
    """
    Define inputs and types
    Field is used to convert hyphens to underscores
    """
    age: int
    workclass: object
    fnlgt: int
    education: object
    education_num: int = Field(alias="education-num") #column names have hypthens
    marital_status: object = Field(alias="marital-status")
    occupation: object
    relationship: object
    race: object
    sex: object
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: object = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True #Compensate for hyphens
        #Extra
        schema_extra = {
                        "example": {
                                    'age':50,
                                    'workclass':"Private", 
                                    'fnlgt':234721,
                                    'education':"Doctorate",
                                    'education_num':16,
                                    'marital_status':"Separated",
                                    'occupation':"Exec-managerial",
                                    'relationship':"Not-in-family",
                                    'race':"Black",
                                    'sex':"Female",
                                    'capital_gain':0,
                                    'capital_loss':0,
                                    'hours_per_week':50,
                                    'native_country':"United-States"
                                    }
                        }      
# Load models, they were stored in the prior directory
cv_rfc = joblib.load('App/rfc_model.pkl')
lrc = joblib.load('App/logistic_model.pkl')

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route, return json of message
@app.get("/")
async def root():
    return {"Message": "This is a Salary Prediction Model API"}

# Define the route to the sample predictor
@app.post("/predict_salary") 
async def predict_salary(sample: Item): 
    """
    Input a instance of raw data
    Return value is a salary prediction
    """
    # Extra
    data = {'age': sample.age,
            'workclass': sample.workclass,
            'fnlgt': sample.fnlgt,
            'education': sample.education,
            'education-num': sample.education_num,
            'marital-status': sample.marital_status,
            'occupation': sample.occupation,
            'relationship': sample.relationship,
            'race': sample.race,
            'sex': sample.sex,
            'capital-gain': sample.capital_gain,
            'capital-loss': sample.capital_loss,
            'hours-per-week': sample.hours_per_week,
            'native-country': sample.native_country}
    if(not(sample)): 
        raise HTTPException(status_code=400, 
                            detail="Please Provide a valid sample")
    # jsonable_encoder converts BaseModel object to json
    #label_dict = jsonable_encoder()
    sample = jsonable_encoder(sample)
    person = pd.DataFrame(sample, index=[0]) 
    #person = pd.DataFrame(data, index=[0]) 
    person = process_data(person) # Format data for model
    prediction = cv_rfc.predict(person) # Predict on created df
    salary = {}
    salary_cat = prediction.tolist()
    salary['salary'] = salary_cat[0]
    return salary
    """
    pred = ''
    if salary_cat[0] == 0:
        pred = ">50k"
        salary['salary'] = pred
        salary = json.dumps(salary)
        return salary
    elif salary_cat[0] == 1:
        pred = "<=50k"
        salary['salary'] = pred
        salary = json.dumps(salary)
        return salary
    """

