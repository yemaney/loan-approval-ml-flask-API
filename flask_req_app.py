# imports
from flask import Flask, request
from flask_restful import Resource, Api
import pandas as pd
import pickle

app = Flask(__name__)
api = Api(app)

# importing pickle files created an saved in the modeling notebook.
model = pickle.load( open( "model.p", "rb" ) )
preprocess = pickle.load( open( "preprocess.p", "rb" ) )


class Prediction(Resource):
    def post(self):
        """Takes json format input of a user profile. Uses [model] to predict
        if a person qualifies for a loan.

        Returns:
            [string]: [short string detailing if the persons profile is expected to be approved for a loan]
        """

        json_data = request.get_json() # gets the data 
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose() # convert data into a dataframe
        data = preprocess.transform(df) # preprocess dataframe 

        res = model.predict(data).tolist()[0] # get the models predictions
        
        
        if res == 1:
            string = 'You may be eligible for a loan'
        else:
            string = 'You are probably not eligible for a loan'
        
        return string 


# assign endpoint
api.add_resource(Prediction, '/Prediction')

if __name__ == '__main__':
    app.run(debug=True)