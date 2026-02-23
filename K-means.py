import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([
    [10, 20], [12, 25], [11, 18],  # গ্রুপ ১ (কম বয়স, কম খরচ)
    [40, 80], [45, 85], [42, 78],  # গ্রুপ ২ (মাঝারি বয়স, বেশি খরচ)
    [70, 30], [75, 35], [72, 28]   # গ্রুপ ৩ (বেশি বয়স, কম খরচ)
])

kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(X)
y_kmeans = kmeans.predict(X)

print(f"গ্রুপগুলো হলো: {y_kmeans}")

# ৫. ক্লাস্টার সেন্টার (প্রতিটি গ্রুপের মেইন পয়েন্ট বা কেন্দ্র)
centers = kmeans.cluster_centers_
print(f"সেন্টারগুলো: {centers}")

