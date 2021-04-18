# imports
from flask import Flask, request, render_template
from flask_restful import Api
import pandas as pd
import pickle

model = pickle.load( open( "model.p", "rb" ) )
preprocess = pickle.load( open( "preprocess.p", "rb" ) )

app = Flask(__name__)
api = Api(app)


@app.route('/')
def welcome():
    return render_template('form.html')


@app.route('/result', methods=['POST'])
def result():
    """Takes in the inforation sumbmitted in the form.html file as an input.
    Then transforms those inputs into a dataframe that can be fed to the model
    predictor.
    """
    # collect values that were entered on the form html page
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
    
    # create a dictionary from all the values collected
    entry = dict(Gender=Gender,Married=Married,Dependents=Dependents,
                Education=Education,Self_Employed=Self_Employed,
                ApplicantIncome=ApplicantIncome,CoapplicantIncome=CoapplicantIncome,
                LoanAmount=LoanAmount,Loan_Amount_Term=Loan_Amount_Term,
                Credit_History=Credit_History,
                Property_Area=Property_Area)
    
    # turn dictionary into a datframe
    df = pd.DataFrame(entry.values(), index=entry.keys()).transpose()
    data = preprocess.transform(df) # preprocess dataframe 
    # getting predictions from our model.
    res = model.predict(data).tolist()[0] # get the models predictions
    
    # make output verbose depending on machine prediction
    if res == 1:
        entry = 'Approved'
    else:
        entry = 'Denied'
    return render_template('result.html', entry=entry)

if __name__ == '__main__':
    app.run( debug=True)