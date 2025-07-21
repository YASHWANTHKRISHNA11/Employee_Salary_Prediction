from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load regression model
model = joblib.load("salary_regressor_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            input_data = {
                'age': int(request.form['age']),
                'workclass': int(request.form['workclass']),
                'fnlwgt': int(request.form['fnlwgt']),
                'education': int(request.form['education']),
                'educational-num': int(request.form['educational_num']),
                'marital-status': int(request.form['marital_status']),
                'occupation': int(request.form['occupation']),
                'relationship': int(request.form['relationship']),
                'race': int(request.form['race']),
                'gender': int(request.form['gender']),
                'capital-gain': int(request.form['capital_gain']),
                'capital-loss': int(request.form['capital_loss']),
                'hours-per-week': int(request.form['hours_per_week']),
                'native-country': int(request.form['native_country']),
            }

            input_df = pd.DataFrame([input_data])

            prediction = model.predict(input_df)[0]
            salary_amount = round(prediction)

            return render_template('result.html', prediction=salary_amount)
        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
