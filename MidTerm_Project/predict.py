import pickle
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify


input_file = 'model_A.bin'


with open(input_file, 'rb') as f_in: 
    dv1, model1 = pickle.load(f_in)

app = Flask('autism')

@app.route('/predict', methods=['POST'])
def predict():
    participant = request.get_json()
    features = list(dv1.get_feature_names_out())
    X_sample = dv1.transform([participant])
    dtest_sample = xgb.DMatrix(X_sample, feature_names=features)

    y_pred = model1.predict(dtest_sample)
    asd = y_pred >= 0.5
    result = {
            'asd_probability': float(y_pred),
            'class/asd': bool(asd) 
    } 

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)