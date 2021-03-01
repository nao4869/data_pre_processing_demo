import numpy
import pandas as pd
import matplotlib.pyplot as plot

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, accuracy_score


def main():
    x, y = retrieveMatricsValuesFromData()
    x_train, x_test, y_train, y_test = splitDataSet(x, y)
    training_set_predicted_purchased = buildLogisticRegressionModel(
        x_train, x_test, y_train, y_test)
    displayActualAndPredictedResults(y_test, training_set_predicted_purchased)
    buildConfusionMatrix(y_test, training_set_predicted_purchased)


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
    standarized_x_training_set = standardScaler.fit_transform(x_train)

    # Standarize the test data set
    standarized_x_test_set = standardScaler.transform(x_test)

    # Training the logistic regression model on the training set
    classifier = LogisticRegression(random_state=0)

    # Calcaulate weight of logistic regression model
    classifier.fit(standarized_x_training_set, y_train)

    # Predicting the purchased rate result from training set age, EstimatedSalary
    expected_prediction = classifier.predict(
        standardScaler.transform(x_test))

    # get prediction score for whole test set
    prediction_score = classifier.score(standarized_x_test_set, y_test)
    # print(prediction_score)

    return expected_prediction


def displayActualAndPredictedResults(test_set_purchased, training_set_predicted_purchased):
    # comparing actual results and predicted results
    actual_and_predicted_results = numpy.concatenate((training_set_predicted_purchased.reshape(
        len(training_set_predicted_purchased), 1), test_set_purchased.reshape(len(test_set_purchased), 1)), 1)
    print(actual_and_predicted_results)


def buildConfusionMatrix(test_set_purchased, training_set_predicted_purchased):
    confusion_matrix_output = confusion_matrix(
        test_set_purchased, training_set_predicted_purchased)
    print(confusion_matrix_output)


main()
