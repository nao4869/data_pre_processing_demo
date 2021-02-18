import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

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
        ('encoder', OneHotEncoder(), [0])
    ],
    remainder='passthrough',
)

x = np.array(columnTransformer.fit_transform(x))

# Encoding the dependent variable
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(y)

print(x)
print(y)
