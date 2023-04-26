import requests
import json


def post(data):
    r = requests.post("https://udacity-api.onrender.com/predict", json.dumps(data))
    return r.status_code, r.json()["Prediction"]

queries = [
    {
        "age": 39,
        "workclass": "Without-pay",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 12,
        "native-country": "United-States",
    },
    {
        "age": 52,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Masters",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15000,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States",
    }
]

for data in queries:
    status_code, prediction = post(data)
    print("data:", data)
    print("status code:", status_code)
    print("Prediciton:", prediction)

    print("=================")
