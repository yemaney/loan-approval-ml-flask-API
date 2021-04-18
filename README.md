# Loan Approval ML Flask API Project

## Motivation:
Banks get many customers looking for loans everyday. This gives them access to information that can be leveraged to create machine learning solutions. 

- Make more better and more accurate loan decisions
- Find connections, correlations, and biases in their decision-making process
- Stream Line and standardize Application Process
- Reduce customer support overhead and improve customer communication

## This Project Has Three Parts:
1. Data Exploration Notebook: 
    - Use matplotlib and seaborn to visualize data, find connections and a little story telling
2. Modeling Notebook:
    - Create a Pipeline to preprocess the data
    - Test multiple classification algorithms
    - Saving the best model and preprocessor with pickle library
3. Deployment:
    1. Flask request app.
        - simple script run through the terminal and can be accessed through the modeling notebook
        - uses the pickled files to make a prediciton and return the prediction
    2. Flask GUI app.
        - Script run through the terminal, but is accessed through a web browser
        - Renders a GUI interface with the pages:
            1. Home page:
                - takes inputs for variables present in the data
                - button for submitting the information
            2. Result page:
                - takes information submitted in the home page and uses the pickled model to make a prediction
                - returns the prediction
