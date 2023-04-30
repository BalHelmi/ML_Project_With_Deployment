from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open('RandomForestModel.pkl', 'rb'))



@app.route('/',  methods=['GET'])
def predict():
    request = pd.read_csv('X_test.csv')
    prediction = model.predict(request)
    prediction = np.array2string(prediction)
    return "The prediction for the request is : " + prediction

@app.route('/health', methods=['GET'])
def health():
    return 'The service is up and running!'

if __name__ == '__main__':
    app.run(debug=True)