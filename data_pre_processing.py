import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# dependent variable - 従属変数 - ある要因によって影響された結果として表れる変数のことを目的変数あるいは従属変数
# something that we wanna predict

# read the dataset from CSV file
dataset = pd.read_csv("Data.csv")

# matrics of feature - : will take all the rows, -1 means index of last column
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)