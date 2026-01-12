from flask import Flask, request, render_template
import pickle
import numpy as np
import os


app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    # The form sends strings, we convert them to floats
    age_years = float(request.form['age'])
    gender = float(request.form['gender'])
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    ap_hi = float(request.form['ap_hi'])
    ap_lo = float(request.form['ap_lo'])
    cholesterol = float(request.form['cholesterol'])
    gluc = float(request.form['gluc'])
    smoke = float(request.form['smoke'])
    alco = float(request.form['alco'])
    active = float(request.form['active'])

    # Convert Age (Years) to Days because the model was trained on Days
    age_days = age_years * 365

    # Organize features in the EXACT order the model expects
    # [age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]
    features = [np.array([age_days, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active])]
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)
    
    # Generate Output
    if prediction[0] == 1:
        result_text = "HIGH RISK DETECTED"
        result_color = "#dc3545" # Red
        advice = "Please consult a cardiologist."
    else:
        result_text = "NORMAL (Low Risk)"
        result_color = "#28a745" # Green
        advice = "Keep up the healthy lifestyle!"

    return render_template('index.html', 
                           prediction_text=result_text, 
                           text_color=result_color,
                           advice_text=advice)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
