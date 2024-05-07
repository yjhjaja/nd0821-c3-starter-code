from fastapi.testclient import TestClient
import json, os, sys

dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(1, dir)
from main import app
client = TestClient(app)

def test_get():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == 'Hello, welcome!'

def test_post_1():
    data = {"age": 20,
            "workclass": "Private",
            "fnlgt": 112847,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Never-married",
            "occupation": "Farming-fishing",
            "relationship": "Own-child",
            "race": "Asian-Pac-Islander",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
    r = client.post("/inference", data = json.dumps(data))
    assert r.status_code == 200
    assert r.json()=='0'

def test_post_2():
    data = {"age": 40,
            "workclass": "Private",
            "fnlgt": 100000,
            "education": "Doctorate",
            "education_num": 20,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
    r = client.post("/inference", data = json.dumps(data))
    assert r.status_code == 200
    assert r.json()=='1'

def test_post_3():
    data = {}
    r = client.post("/inference", json = json.dumps(data))
    assert r.status_code == 422
