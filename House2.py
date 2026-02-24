import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

data = {
    'Size': [1500, 2000, 1200, 2500, 1800, 1350, 2200, 2800, 1600, 1900],
    'Rooms': [3, 4, 2, 5, 3, 2, 4, 5, 3, 3],
    'Age': [10, 5, 15, 2, 8, 12, 4, 1, 9, 6],
    'Price': [450000, 600000, 350000, 800000, 520000, 400000, 650000, 900000, 480000, 550000]
}

df = pd.DataFrame(data)

X = df[['Size', 'Rooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

new_house = [[1700,3,5]]
predicted_price = model.predict(new_house)

print(predicted_price)