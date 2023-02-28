from fastapi import status
import requests
import json


def get_salary():
    data = {
        "age": 26,
        "workclass": "private",
        "fnlgt": 189765,
        "education": "Bachelors",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 592,
        "capital-loss": 88,
        "hours-per-week": 60,
        "native-country": "United-States"}
    request = requests.post(
        'https://salary-prediction-pnnc.onrender.com/predict_salary',
        auth=(
            'usr',
            'pass'),
        data=json.dumps(data))
    assert request.status_code == status.HTTP_200_OK
    assert request.json() == {"salary": ">50K"}
    print(f"Input demographic sample is: {data}")
    print(f"JSON salary prediction is: {request.json()}")


get_salary()
