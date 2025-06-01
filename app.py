from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)  # ✅ Corrected from _name_

# Load model and label encoder
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        income = float(request.form['annual_income'])
        amount = float(request.form['purchase_amount'])
        freq = int(request.form['purchase_frequency'])

        # Combine inputs
        features = np.array([[age, income, amount, freq]])

        # Predict
        prediction = model.predict(features)[0]
        region = label_encoder.inverse_transform([prediction])[0]

        return render_template('index.html', result=f"Predicted Region: {region}")
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':  # ✅ Corrected from _name_ and _main_
    app.run(debug=True)
