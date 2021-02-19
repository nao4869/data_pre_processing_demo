import numpy
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    x, y = retrieveMatricsValuesFromData()
    print(x)
    print(y)


def retrieveMatricsValuesFromData() -> numpy.ndarray:
    # read the dataset from CSV file
    dataset = pd.read_csv("Data.csv")

    # matrics of feature - : will take all the rows, -1 means index of last column
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # calculate the mean of age and salary for missing data
    imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
    imputer.fit(x[:, 1:3])
    x[:, 1:3] = imputer.transform(x[:, 1:3])

    # Encoding categorical data
    columnTransformer = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(), [0])], remainder='passthrough')

    x = numpy.array(columnTransformer.fit_transform(x))

    # Encoding the dependent variable
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(y)

    return x, y


def splitDataSet(x: numpy.ndarray, y: numpy.ndarray):
    # Splitting the dataset into Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=1,
    )


def calculateFeatureScalling(x_train, x_test):
    # Feature Scalling
    scaller = StandardScaler()
    x_train[:, 3:] = scaller.fit_transform(x_train[:, 3:])
    x_test[:, 3:] = scaller.transform(x_test[:, 3:])


main()
