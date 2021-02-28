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
    x_train, x_test, y_train, y_test = splitDataSet(rd_to_state, profit)
    print(encoded_data)


def retrieveMatricsValuesFromData() -> numpy.ndarray:
    # read the dataset from CSV file
    dataset = pd.read_csv("50_Startups.csv")

    # matrics of feature - : will take all the rows, -1 means index of last column
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    return x, y

# Encode the state string to numerical value
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


main()
