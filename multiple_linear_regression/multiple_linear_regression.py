import numpy
import pandas as pd
import matplotlib.pyplot as plot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def main():
    rd_to_state, profit = retrieveMatricsValuesFromData()
    encoded_data = encodeCategoricalData(rd_to_state)
    x_train, x_test, y_train, y_test = splitDataSet(encoded_data, profit)
    buildMultipleLinearRegressionModel(x_train, x_test, y_train, y_test)


def retrieveMatricsValuesFromData() -> numpy.ndarray:
    # read the dataset from CSV file
    dataset = pd.read_csv("50_Startups.csv")

    # matrics of feature - : will take all the rows, -1 means index of last column
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    return x, y


def encodeCategoricalData(x: numpy.ndarray):
    # Encoding categorical data
    columnTransformer = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(), [3])], remainder='passthrough')

    encoded_data = numpy.array(columnTransformer.fit_transform(x))
    return encoded_data


def splitDataSet(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    # Splitting the dataset into Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=0
    )
    return x_train, x_test, y_train, y_test


def buildMultipleLinearRegressionModel(x_train, x_test, y_train, y_test):
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # predict the test set results
    y_predicted = regressor.predict(x_test)
    numpy.set_printoptions(precision=2)
    print(numpy.concatenate((y_predicted.reshape(len(y_predicted),1), y_test.reshape(len(y_test),1)),1))

main()
