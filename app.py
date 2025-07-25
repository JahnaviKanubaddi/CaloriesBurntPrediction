from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scaler
rf_model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load dataset for column reference
df = pd.read_csv("exercise.csv")
X = df.drop(columns=["Calories"])

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    height = int(request.form['height'])
    weight = int(request.form['weight'])
    duration = int(request.form['duration'])
    body_temp = float(request.form['body_temp'])
    heart_rate = int(request.form['heart_rate'])

    user_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'Duration': [duration],
        'Body_Temp': [body_temp],
        'Heart_Rate': [heart_rate],
        'User_ID': [0]  # Default
    })

    user_data = user_data.reindex(columns=X.columns, fill_value=0)
    user_data_scaled = scaler.transform(user_data)
    predicted_calories = rf_model.predict(user_data_scaled)

    return render_template("result.html", calories=predicted_calories[0])

if __name__ == "__main__":
    app.run(debug=True)
