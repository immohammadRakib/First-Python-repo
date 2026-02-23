import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([[20, 30], [22, 35], [25, 40], [45, 80], [50, 85], [48, 90], [60, 20], [65, 25], [70, 15]])

kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)