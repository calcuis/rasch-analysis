import numpy as np
from sklearn.linear_model import LinearRegression

def perform_rasch_analysis(data):
    # Create the LinearRegression object
    regressor = LinearRegression()

    # Extract the predictors and outcomes from the data
    X = data[:, :-1]
    y = data[:, -1]

    # Fit the predictors and outcomes to the regressor
    regressor.fit(X, y)

    # Get the coefficients and store them in the results dictionary
    results = {"coefficients": regressor.coef_}

    # Return the results
    return results
