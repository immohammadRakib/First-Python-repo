import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = np.array([[1000], [1500], [5000], [2000], [3000], [4000]])
y = np.array([10, 20, 30, 50, 60, 80])

# X = np.array([[1000], [1500], [2000], [3000], [4000]]) # বাড়ির সাইজ
# y = np.array([20, 30, 40, 60, 80]) # দাম (লাখ টাকায়)

# X_train, y_train, X_test, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print(f"২০০০ স্কয়ার ফিট বাড়ির দাম কম্পিউটার বলছে: {prediction[0]} লাখ")
print(f"আসল দাম ছিল: {y_test[0]} লাখ")