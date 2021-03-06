import numpy
import pandas as pd
import matplotlib.pyplot as plot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    x, y = retrieveMatricsValuesFromData()
    x_train, x_test, y_train, y_test = splitDataSet(x, y)
    buildSimpleLinearRegressionModel(
        x_train,
        y_train,
        x_test,
        y_test
    )
    # print(x_test)


def retrieveMatricsValuesFromData() -> numpy.ndarray:
    # read the dataset from CSV file
    dataset = pd.read_csv("Salary_Data.csv")

    # matrics of feature - : will take all the rows, -1 means index of last column
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    return x, y


def splitDataSet(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    # Splitting the dataset into Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=0
    )
    return x_train, x_test, y_train, y_test


# Training the SLR model on the training set
def buildSimpleLinearRegressionModel(x_training_set: numpy.ndarray, y_training_set: numpy.ndarray, x_test_set: numpy.ndarray, y_test_set: numpy.ndarray) -> numpy.ndarray:
    regressor = LinearRegression()
    result = regressor.fit(
        x_training_set,
        y_training_set
    )

    # Predicting the test set result
    y_predicted_salary = regressor.predict(x_test_set)

    testCase = input()

    if (testCase == 'training'):
        # Visualizing training set results
        plot.scatter(
            x_training_set,
            y_training_set,
            color='red'
        )
        plot.plot(
            x_training_set,
            regressor.predict(x_training_set),
            color='blue'
        )
        plot.title('Salary vs Experience (Training set)')
        plot.xlabel('Years of experience')
        plot.ylabel('Salary')
        plot.show()
    else:
        # Visualizing test set results
        plot.scatter(
            x_test_set,
            y_test_set,
            color='red'
        )
        plot.plot(
            x_training_set,
            regressor.predict(x_training_set),
            color='blue'
        )
        plot.title('Salary vs Experience (Test set)')
        plot.xlabel('Years of experience')
        plot.ylabel('Salary')
        plot.show()


main()
