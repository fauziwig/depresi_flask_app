from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from flask_cors import CORS

# Load the trained model
model = load('./depression_prediction_model.joblib')

#initialize the flask app
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]

        print(f"prediction: {prediction}")

        return jsonify({"depression": int(prediction)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/coba', methods=['POST'])
def coba():
    try:
        dummy = request.json
        # Menampilkan data yang diterima di konsol server
        print(f"Data yang diterima dari user: {dummy}")

        return jsonify({"dummy": dummy}), 200

    except Exception as e:
        error_message = f"Terjadi error pada /coba: {str(e)}"
        print(error_message) # Juga menampilkan error di konsol
        return jsonify({"error": error_message}), 500


@app.route('/')
def home():
    return "Welcome to the Stroke Prediction API"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)