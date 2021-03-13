# imports
from flask import Flask, request
from flask_restful import Resource, Api
import pandas as pd
import pickle

app = Flask(__name__)
api = Api(app)


model = pickle.load( open( "model.p", "rb" ) )


class Scoring(Resource):
    def post(self):
        """Function takes in a json format api request as data.
        the data must be formatted in the correct order illustrated in the notebook.

        Returns:
            list: list will have a single entry, either a 'Y' or 'N', depending on the 
            models predictions
        """
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        # getting predictions from our model.
        # it is much simpler because we used pipelines during development
        res = model.predict(df)
        # we cannot send numpt array as a result
        return res.tolist() 


# assign endpoint
api.add_resource(Scoring, '/scoring')

if __name__ == '__main__':
    app.run(debug=True)