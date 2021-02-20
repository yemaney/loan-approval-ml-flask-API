from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

# import Flask and jsonify
from flask import Flask, jsonify, request, render_template
# import Resource, Api and reqparser
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy
import pickle

app = Flask(__name__)
api = Api(app)


@app.route('/')
def welcome():
    return render_template('form.html')

model = pickle.load( open( "model.p", "rb" ) )

@app.route('/result', methods=['POST'])
def result():
    Gender = request.form.get("var_1", type=str)
    Married = request.form.get("var_2", type=str)
    Dependents = request.form.get("var_3", type=str)
    Education = request.form.get("var_4", type=str)
    Self_Employed = request.form.get("var_5", type=str)
    ApplicantIncome = request.form.get("var_6", type=int)
    CoapplicantIncome = request.form.get("var_7", type=int)
    LoanAmount = request.form.get("var_8", type=int)
    Loan_Amount_Term = request.form.get("var_9", type=int)
    Credit_History = request.form.get("var_10", type=int)
    Property_Area = request.form.get("var_11", type=str)
    
    entry = dict(Gender=Gender,Married=Married,Dependents=Dependents,
                Education=Education,Self_Employed=Self_Employed,
                ApplicantIncome=ApplicantIncome,CoapplicantIncome=CoapplicantIncome,
                LoanAmount=LoanAmount,Loan_Amount_Term=Loan_Amount_Term,
                Credit_History=Credit_History,
                Property_Area=Property_Area)
    
    df = pd.DataFrame(entry.values(), index=entry.keys()).transpose()
    # getting predictions from our model.
        # it is much simpler because we used pipelines during development
    res = model.predict(df)
        # we cannot send numpt array as a result
    
    if res.tolist() == ['Y']:
        entry = 'Approved'
    else:
        entry = 'Denied'
#     entry = res.tolist() 
    return render_template('result.html', entry=entry)

# 'Based on the information provided, we believe you are eligible for the loan'
# 'Unfortunately you may not be eligible for a loan'

# class Scoring(Resource):
#     def post(self):
#         json_data = request.get_json()
#         df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
#         # getting predictions from our model.
#         # it is much simpler because we used pipelines during development
#         res = model.predict(df)
#         # we cannot send numpt array as a result
#         return res.tolist() 


# # assign endpoint
# api.add_resource(Scoring, '/scoring')

if __name__ == '__main__':
    app.run( debug=True)