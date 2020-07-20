from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


'''
# prediction function 
def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1, 6) 
    loaded_model = joblib.load(open("Random_Forrest_Credit_Approval.pkl", "rb")) 
    result = loaded_model.predict(to_predict) 
    return result[0]

@app.route('/predict', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)         
        if int(result)== 1: 
            prediction ='Approved'
        else: 
            prediction ='Not Approved'            
        return render_template("result.html", prediction = prediction) 
'''

@app.route('/predict',methods=['POST'])
def predict():
	#Alternative Usage of Saved Model
    model = pickle.load(open("model.pkl", "rb"))

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


        #newX1 = [gender, age, debt, b_cust, experience, default, e_status, cre_score, income] + m_status + e_level + citizen + d_licence + ethnicity
        # 'PriorDefault', 'YearsEmployed', 'CreditScore', 'Debt', 'Income','Age'
        to_predict_list = [default,experience,cre_score,debt,income,age]
        print(to_predict_list)
        X_nparray = np.array(to_predict_list, dtype=np.float32).reshape(1, 6)
        # X_nparray = np.asarray(to_predict_list, dtype=np.int)
        print(X_nparray)
        prediction = model.predict(X_nparray)
        status = prediction[0]
        print(status)
        if int(status) == 1:
            return render_template('result.html',prediction = "Congratulations! Your Credit Card Application has been Approved!")
        if int(status) == 0:
            return render_template('result.html',prediction = "Your Credit Card Application has been Rejected")

    

'''
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
'''


if __name__ == '__main__':
	app.run(debug=True)