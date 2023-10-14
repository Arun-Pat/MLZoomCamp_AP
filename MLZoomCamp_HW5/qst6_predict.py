import pickle
import numpy as np
from flask import Flask, request, jsonify

model_file = 'model1.bin'
dv_file ='dv.bin'

def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)
    

def predict_single(client, dv, model):
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

dv = load('dv.bin')
model = load('model2.bin')

app = Flask('credit_score')


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    y_pred = predict_single(client, dv, model)
    get_card = y_pred >= 0.5

    result = {
        'get_card_probability': float(y_pred),
        'get_card': bool(get_card)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

