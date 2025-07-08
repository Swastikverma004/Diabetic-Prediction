from flask import Flask, request, jsonify
import numpy as np
import sys, scipy, xgboost, joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('diabetes_model_v2.joblib')  # Save your model first with joblib.dump()

@app.route('/')
def home():
    return app.send_static_file('project.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Prepare input data in same format as training
    features = np.array([
        data['pregnancies'],
        data['glucose'],
        data['bloodPressure'],
        data['skinThickness'],
        data['insulin'],
        data['bmi'],
        data['dpf'],
        data['age']
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    # Return results
    return jsonify({
        'prediction': int(prediction[0]),
        'confidence': float(probability[0][prediction[0]])
    })

if __name__ == '__main__':
    app.run(debug=True)