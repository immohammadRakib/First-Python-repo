import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = np.array([[1], [2], [3], [4], [5]]) # পড়ার ঘণ্টা
y = np.array([40, 50, 60, 70, 80])      # প্রাপ্ত নম্বর

Model = LinearRegression()
Model.fit(X,y)

prediction = Model.predict([[6]])
print(f"৬ ঘণ্টা পড়লে স্কোর হতে পারে: {prediction[0]}")
