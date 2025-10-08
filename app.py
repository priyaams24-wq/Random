from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("iris_model.joblib")

# Mapping prediction labels
species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from UI
        features = [
            float(request.form['sepal_length']),
            float(request.form['sepal_width']),
            float(request.form['petal_length']),
            float(request.form['petal_width'])
        ]

        prediction = model.predict([features])[0]
        predicted_class = species[prediction]

        return render_template('index.html', prediction_text=f"Predicted Iris Species: 🌸 {predicted_class}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


# API endpoint for JSON POST (optional)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction), "species": species[int(prediction)]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
