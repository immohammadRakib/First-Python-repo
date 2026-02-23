import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = [[60, 160], [85, 170], [45, 150], [90, 180]]
y = [0, 1, 0, 1] 

clf = LogisticRegression()
clf.fit(X,y)

newData = [[70, 165]]
prediction = clf.predict(newData)


if prediction[0] == 0:
    print("ফলাফল: সুস্থ")
else:
    print("ফলাফল: অসুস্থ")