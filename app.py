from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import pickle as pkl
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load data
df = pd.read_csv("Cleaned_Car_data.csv")
companys = sorted(df["company"].unique().tolist())
model_names = sorted(df["name"].unique().tolist())
year = sorted(df["year"].unique().tolist(), reverse=True)
fuel_type = sorted(df["fuel_type"].unique().tolist())

@app.route('/data', methods=["POST"])
def data():
    if request.method == "POST":
        try:
            json_data = {
                "companys": companys,
                "model_names": model_names,
                "year": year,
                "fuel_type": fuel_type
            }
            return jsonify(json_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Method not allowed'}), 405

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get JSON data from body
        data = request.json
        
        # Extract data
        company = data.get("cname")
        model = data.get("model")
        year = int(data.get("year"))
        fuel_type = data.get("fuel_type")
        kms_driven = int(data.get("kms_driven"))

        if None in [company, model, year, fuel_type, kms_driven]:
            return jsonify({'error': 'Invalid input'}), 400

        try:
            # Load the model
            pkl_model = pkl.load(open("model.pkl", "rb"))
        except FileNotFoundError:
            return jsonify({'error': 'Model file not found'}), 500

        # Prepare input data for prediction
        input_data = pd.DataFrame([[model, company, year, kms_driven, fuel_type]],
                                  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        
        # Predict
        pre = pkl_model.predict(input_data)
        return jsonify({"predict": np.round(pre[0], 2).tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

