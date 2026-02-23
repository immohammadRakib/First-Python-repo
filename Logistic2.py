import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

X = np.array([[22, 20000], [25, 25000], [47, 80000], [52, 110000], [36, 60000], 
              [29, 35000], [48, 95000], [33, 50000], [42, 85000], [20, 15000]])
y = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 0]) 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(f"মডেলের একুরেসি: {accuracy_score(y_test, y_pred) * 100}%")


new_person = sc.transform([[33, 160000]])
prediction = classifier.predict(new_person)

print("গাড়ি কিনবে" if prediction[0] == 1 else "গাড়ি কিনবে না")