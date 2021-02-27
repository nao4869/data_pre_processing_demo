import numpy
import pandas as pd
import matplotlib.pyplot as plot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


def main():
    x, y = retrieveMatricsValuesFromData()
    # buildLinearRegressionModel(x, y)
    buildPolynomialRegressionModel(x, y)


def retrieveMatricsValuesFromData() -> numpy.ndarray:
    # read the dataset from CSV file
    dataset = pd.read_csv("Position_Salaries.csv")

    # matrics of feature - : will take all the rows, -1 means index of last column
    x = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    return x, y


# Training the linear regression model on the whole dataset
def buildLinearRegressionModel(x, y):
    linear_regressor = LinearRegression()
    fitted_model = linear_regressor.fit(x, y)
    
    # visualizing the linear regression results
    plot.scatter(x, y, color = 'red')
    plot.plot(x, linear_regressor.predict(x), color = 'blue')
    
    plot.title('Linear Regression model')
    plot.xlabel('Position Level')
    plot.ylabel('Salary')
    plot.show()


# Training the linear regression model on the whole dataset
def buildPolynomialRegressionModel(x, y):
    polynomial_regressor = PolynomialFeatures(degree=10)
    x_polynomial_feature = polynomial_regressor.fit_transform(x)

    linear_regressor = LinearRegression()
    linear_regressor.fit(x_polynomial_feature, y)
    
    # visualizing the linear regression results
    plot.scatter(x, y, color = 'red')
    plot.plot(x, linear_regressor.predict(x_polynomial_feature), color = 'blue')
    
    plot.title('Polynomial Regression model')
    plot.xlabel('Position Level')
    plot.ylabel('Salary')
    plot.show()


main()
