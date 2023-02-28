from fastapi import FastAPI, HTTPException
import joblib
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
    # column names have hypthens
    education_num: int = Field(alias="education-num")
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
        """
        Compensate for hyphenated column names
        Provide schema for inputs
        """
        allow_population_by_field_name = True  # Compensate for hyphens
        schema_extra = {
            "example": {
                'age': 50,
                'workclass': "Private",
                'fnlgt': 234721,
                'education': "Doctorate",
                'education_num': 16,
                'marital_status': "Separated",
                'occupation': "Exec-managerial",
                'relationship': "Not-in-family",
                'race': "Black",
                'sex': "Female",
                'capital_gain': 0,
                'capital_loss': 0,
                'hours_per_week': 50,
                'native_country': "United-States"
            }
        }


# Load models, they were stored in the prior directory
cv_rfc = joblib.load('App/rfc_model.pkl')
lrc = joblib.load('App/logistic_model.pkl')

# Initialize an instance of FastAPI
app = FastAPI()


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
    if (not (sample)):
        raise HTTPException(status_code=400,
                            detail="Please Provide a valid sample")
    # jsonable_encoder converts BaseModel object to json
    sample = jsonable_encoder(sample)
    person = pd.DataFrame(sample, index=[0])
    person = process_data(person)  # Format data for model
    prediction = cv_rfc.predict(person)  # Predict on created df
    salary = {}
    # Prediction converted from np to list, expected less inputs error thrown
    salary_cat = prediction.tolist()
    salary['salary'] = salary_cat[0]
    return salary
