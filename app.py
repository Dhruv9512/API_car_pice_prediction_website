from flask import Flask,jsonify,request
from flask_cors import CORS
import pandas as pd
import pickle as pkl
import numpy as np
import os
app = Flask(__name__)
CORS(app)

df = pd.read_csv("Cleaned_Car_data.csv")
companys = sorted(df["company"].unique().tolist())
model_names = sorted(df["name"].unique().tolist())
year = sorted(df["year"].unique().tolist(),reverse=True)
fuel_type = sorted(df["fuel_type"].unique().tolist())


@app.route('/data', methods=["POST"])
def data():
    if request.method == "POST":
        try:
            json_data = [
                {
                    "companys":companys,
                    "model_names":model_names,
                    "year":year,
                    "fuel_type":fuel_type
                }
            ]
            return jsonify(json_data)
        except Exception as e:
            return jsonify({'error': str(e)})

    else:
        return jsonify({'error': 'Method not allowed'}), 405
    


@app.route('/predict', methods=["POST"])
def predict():

    try:
        # get json data from body
        data = request.json

        # store it 
        company = data.get("cname") 
        model = data.get("model")
        year = int(data.get("year"))
        fuel_type = data.get("fuel_type")
        kms_driven = int(data.get("kms_driven"))

        try:
            pkl_model = pkl.load(open("model.pkl", "rb"))
        except FileNotFoundError:
            return jsonify({'error': 'Model file not found'}), 500

        # predict
        input_data = pd.DataFrame([[model,company,year,kms_driven,fuel_type]],columns=['name', 'company', 'year',  'kms_driven' ,'fuel_type'])
        pre =  pkl_model.predict(input_data)
        return jsonify([{
            "predict": np.round(pre[0],2).tolist()
        }])
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500



if __name__ == "__main__":
    DEBUG_MODE = os.getenv('DEBUG_MODE') == 'True'
    app.run(debug=DEBUG_MODE)