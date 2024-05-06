from fastapi.testclient import TestClient
import json, sys

path_proj = '/Users/junhui.yang/Library/CloudStorage/Dropbox/udacity/nd0821-c3-starter-code-master/'
sys.path.insert(1, path_proj)
from main import app
client = TestClient(app)

def test_get():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == 'Hello, welcome!'

def test_post_1():
    data = {"age": 40,
            "workclass": "Self-emp-inc",
            "fnlgt": 300000,
            "education": "HS-grad",
            "education_num": 6,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capital_gain": 10000,
            "capital_loss": 10,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
    r = client.post("/inference", data = json.dumps(data))
    assert r.status_code == 200
    assert r.json()=='0' or r.json()=='1'

def test_post_2():
    data = {}
    r = client.post("/inference", json = json.dumps(data))
    assert r.status_code == 422