import os
import sys
import json
from fastapi.testclient import TestClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../starter")))
from main import app

client = TestClient(app)

def test_api_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Hello adventurer!"

def test_api_post_0():
    r = client.post(
        "/predict/42", 
        data = json.dumps({
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
        }),
    )
    assert r.status_code == 200
    assert r.json() == {"Prediction": 0}


def test_api_post_1():
    r = client.post(
        "/predict/24", 
        data = json.dumps({
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
        }),
    )
    assert r.status_code == 200
    assert r.json() == {"Prediction": 1}
