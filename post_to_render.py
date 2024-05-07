import json
import requests

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

response = requests.post("https://render-deployment-example-20u2.onrender.com/inference", data=json.dumps(data))

print(response.status_code)
print(response.json())