import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# dependent variable - 従属変数 - ある要因によって影響された結果として表れる変数のことを目的変数あるいは従属変数

# read the dataset from CSV file
dataset = pd.read_csv("Data.csv")

# matrics of feature - : will take all the rows, -1 means index of last column
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding categorical data
columnTransformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [0])], remainder='passthrough')

x = np.array(columnTransformer.fit_transform(x))

# Encoding the dependent variable
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(y)

# Splitting the dataset into Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=1,
)

# Feature Scalling
scaller = StandardScaler()
x_train[:, 3:] = scaller.fit_transform(x_train[:, 3:])
x_test[:, 3:] = scaller.transform(x_test[:, 3:])

# print(x)
# print(y)

print(x_train)
print(x_test)
# print(y_train)
# print(y_test)
