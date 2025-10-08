from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("iris_model.joblib")

@app.route('/')
def home():
    return "🌼 Iris Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({
        "prediction": int(prediction)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
