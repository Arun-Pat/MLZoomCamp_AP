import requests
import json

participant1 = {
    "a1_score": 1, 
    "a2_score": 1, 
    "a3_score": 1, 
    "a4_score": 1, 
    "a5_score": 0, 
    "a6_score": 1, 
    "a7_score": 1, 
    "a8_score": 1, 
    "a9_score": 1, 
    "a10_score": 0, 
    "age" : 17,
    "gender": "m" , 
    "ethnicity" : "asian", 
    "jaundice" : "yes", 
    "autism": "yes",  
    "country_of_res" : "bahamas", 
    "relation" : "health_care_professional"
}

# This is sample of NON Autistic patient
participant2 = {
    "a1_score": 1, 
    "a2_score": 1, 
    "a3_score": 0, 
    "a4_score": 1, 
    "a5_score": 0, 
    "a6_score": 0, 
    "a7_score": 1, 
    "a8_score": 1, 
    "a9_score": 0, 
    "a10_score": 1, 
    "age" : 35,
    "gender": "f" , 
    "ethnicity" : "white-European", 
    "jaundice" : "no", 
    "autism": "yes",  
    "country_of_res" : "united_states", 
    "relation" : "self"
}
host = 'https://predict-cs4g.onrender.com'
# host = 'asd-app-env.eba-tjc29bbs.ap-south-1.elasticbeanstalk.com'
url = f'{host}/predict'
# url = "http://localhost:9696/predict"

response = requests.post(url, json=participant1)
result = response.json()

print(json.dumps(result, indent=2))
