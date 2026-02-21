import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1,1)
marks = np.array([12, 22, 28, 42, 48, 62, 70, 82, 88, 95])


# plt.scatter(hours, marks, color='red')
# plt.xlabel('Exam Hours')
# plt.ylabel('Exam Marks')
# plt.title('Actual Data Point')
# plt.show()

model = LinearRegression()
model.fit(hours, marks)

# plt.scatter(hours, marks, color='green')
# plt.plot(hours, model.predict(hours), color='blue')
# plt.title('Linear Regression Line')
# plt.show()

new_hours = np.array([[0]])
predicted_mark = model.predict(new_hours)
print(f"যদি সাবিহা ১২ ঘণ্টা পড়ে, তবে সে সম্ভাব্য মার্কস পাবে: {predicted_mark[0]:.2f}")