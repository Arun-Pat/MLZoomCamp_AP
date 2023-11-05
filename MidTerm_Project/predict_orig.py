
import pickle
import xgboost as xgb

input_file = 'model_A.bin'


with open(input_file, 'rb') as f_in: 
    dv1, model1 = pickle.load(f_in)

# This is sample of Autistic patient
participant1 = {
    'a1_score': 1, 
    'a2_score': 1, 
    'a3_score': 1, 
    'a4_score': 1, 
    'a5_score': 0, 
    'a6_score': 1, 
    'a7_score': 1, 
    'a8_score': 1, 
    'a9_score': 1, 
    'a10_score': 0, 
    'age' : 17,
    'gender': 'm' , 
    'ethnicity' : 'asian', 
    'jaundice' : 'yes', 
    'autism': 'yes',  
    'country_of_res' : 'bahamas', 
    'relation' : 'health_care_professional'
}

# This is sample of NON Autistic patient
participant2 = {
    'a1_score': 1, 
    'a2_score': 1, 
    'a3_score': 0, 
    'a4_score': 1, 
    'a5_score': 0, 
    'a6_score': 0, 
    'a7_score': 1, 
    'a8_score': 1, 
    'a9_score': 0, 
    'a10_score': 1, 
    'age' : 35,
    'gender': 'f' , 
    'ethnicity' : 'white-European', 
    'jaundice' : 'no', 
    'autism': 'yes',  
    'country_of_res' : 'united_states', 
    'relation' : 'self'
}


for p in [participant1,participant2 ]: 
    features = list(dv1.get_feature_names_out())
    X_sample = dv1.transform([p])

    dtest_sample = xgb.DMatrix(X_sample, feature_names=features)

    y_pred = model1.predict(dtest_sample)
    print(y_pred)

