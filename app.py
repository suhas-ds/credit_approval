from flask import Flask,render_template,url_for,request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	
    filename = 'trained_model/model.pkl'
    model = pickle.load(open(filename, 'rb'))

    if request.method == 'POST':

        # Featch Value for Age
        age = float(request.form['Age'])

        # Fetch value for  Total Debt
        debt = float(request.form['Total_Debt'])

        # Fetch value for Total Professional Experience
        experience = float(request.form['Tot_Experience'])

        # Fetch value for Income
        income = float(request.form['Income'])


        # Fetch value for Credit Score
        cre_score = float(request.form['Credit_Score'])

        
        # Featch value for Prior Default
        default = request.form['Default'] 
        if default == "No":
            default = int(0.0)
        if default == "Yes":
            default = int(1.0)

        # 'PriorDefault', 'YearsEmployed', 'CreditScore', 'Debt', 'Income','Age'
        to_predict_list = [default,experience,cre_score,debt,income,age]
        print(to_predict_list)
        X_nparray = np.array(to_predict_list, dtype=np.float32).reshape(1, 6)
        
        print(X_nparray)
        prediction = model.predict(X_nparray)
        prediction_value = prediction[0]
        print(prediction_value)

        
        if int(prediction_value) == 1:
            status = "Congratulations! Your Credit Card Application has been Approved!"
        if int(prediction_value) == 0:
            status = "Your Credit Card Application has been Rejected"

        return render_template('index.html',prediction = status)

@app.errorhandler(500)
def internal_error(error):
    return "500: Something went wrong"

@app.errorhandler(404)
def not_found(error):
    return "404: Page not found",404


if __name__ == '__main__':
	app.run(debug=True)