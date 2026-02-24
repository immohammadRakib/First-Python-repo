import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# ডাটা (বয়স এবং খরচ)
X = np.array([[20, 30], [22, 35], [25, 40], [45, 80], [50, 85], [48, 90], [60, 20], [65, 25], [70, 15]])

# ৩টি গ্রুপে ভাগ করা
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X) # কোন পয়েন্ট কোন গ্রুপে (০, ১, বা ২)

# গ্রাফে দেখানো
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis') # ডাটা পয়েন্ট

# সেন্টারগুলো লাল রঙে দেখানো
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X') 

plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.title('K-Means Clustering with Centers')
plt.show()
