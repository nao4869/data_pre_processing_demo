import numpy
import pandas as pd
import matplotlib.pyplot as plot

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():
    x, y = retrieveMatricsValuesFromData()
    x_train, x_test, y_train, y_test = splitDataSet(x, y)
    buildLogisticRegressionModel(x_train, x_test, y_train, y_test)


def retrieveMatricsValuesFromData() -> numpy.ndarray:
    # read the dataset from CSV file
    dataset = pd.read_csv("Social_Network_Ads.csv")

    # matrics of feature - : will take all the rows, -1 means index of last column
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return x, y


def splitDataSet(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    # Splitting the dataset into Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=0
    )
    return x_train, x_test, y_train, y_test


def buildLogisticRegressionModel(x_train, x_test, y_train, y_test):
    standardScaler = StandardScaler()
    x_training_set = standardScaler.fit_transform(x_train)

    # Standarize the test data set
    standarized_x_test_set = standardScaler.transform(x_test)

    # Training the logistic regression model on the training set
    classifier = LogisticRegression(random_state=0)

    # Calcaulate weight of logistic regression model
    classifier.fit(x_train, y_train)
    
    probaility = classifier.predict_proba(standarized_x_test_set)
    print(probaility)


main()
