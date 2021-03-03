import numpy
import pandas as pd
import matplotlib.pyplot as plot

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier


def main():
    x, y = retrieveMatricsValuesFromData()
    x_train, x_test, y_train, y_test = splitDataSet(x, y)
    training_set_predicted_purchased = buildKernelSupportVectorMachineModel(
        x_train, x_test, y_train, y_test)
    
    displayActualAndPredictedResults(y_test, training_set_predicted_purchased)
    buildConfusionMatrix(y_test, training_set_predicted_purchased)

    # visualizeTrainingSet(x_train, y_train)
    # visualizeTestSetResult(x_test, y_test)


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


def buildKernelSupportVectorMachineModel(x_train, x_test, y_train, y_test):
    standardScaler = StandardScaler()
    standarized_x_training_set = standardScaler.fit_transform(x_train)

    # Standarize the test data set
    standarized_x_test_set = standardScaler.transform(x_test)

    # Create SVM instance
    classifier = SVC(kernel='rbf', random_state=0)

    # Calcaulate weight of SVM model
    classifier.fit(standarized_x_training_set, y_train)

    # Predicting the purchased rate result from training set age, EstimatedSalary
    expected_prediction = classifier.predict(
        standardScaler.transform(x_test))

    # get prediction score for whole test set
    prediction_score = classifier.score(standarized_x_test_set, y_test)

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

    predicted_score = accuracy_score(
        test_set_purchased, training_set_predicted_purchased)
    print(predicted_score)


def visualizeTrainingSet(x_train, y_train):
    standardScaler = StandardScaler()
    standarized_x_training_set = standardScaler.fit_transform(x_train)

    # Create KNN instance
    classifier = KNeighborsClassifier(n_neighbors=5)

    classifier.fit(standarized_x_training_set, y_train)
    x_set, y_set = standardScaler.inverse_transform(
        standarized_x_training_set), y_train

    X1, X2 = numpy.meshgrid(numpy.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=0.25),
                            numpy.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=0.25))
    plot.contourf(X1, X2, classifier.predict(standardScaler.transform(numpy.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                  alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plot.xlim(X1.min(), X1.max())
    plot.ylim(X2.min(), X2.max())
    for i, j in enumerate(numpy.unique(y_set)):
        plot.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                     c=ListedColormap(('red', 'green'))(i), label=j)
    plot.title('Kernel SVM (Training set)')
    plot.xlabel('Age')
    plot.ylabel('Estimated Salary')
    plot.legend()
    plot.show()


def visualizeTestSetResult(x_test, y_test):
    standardScaler = StandardScaler()
    # Standarize the test data set
    standarized_x_test_set = standardScaler.fit_transform(x_test)

    # Create KNN instance
    classifier = KNeighborsClassifier(n_neighbors=30)

    classifier.fit(standarized_x_test_set, y_test)
    x_set, y_set = standardScaler.inverse_transform(
        standarized_x_test_set), y_test

    X1, X2 = numpy.meshgrid(numpy.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=0.25),
                            numpy.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=0.25))
    plot.contourf(X1, X2, classifier.predict(standardScaler.transform(numpy.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                  alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plot.xlim(X1.min(), X1.max())
    plot.ylim(X2.min(), X2.max())
    for i, j in enumerate(numpy.unique(y_set)):
        plot.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                     c=ListedColormap(('red', 'green'))(i), label=j)
    plot.title('Kernel SVM (Test set)')
    plot.xlabel('Age')
    plot.ylabel('Estimated Salary')
    plot.legend()
    plot.show()


main()
