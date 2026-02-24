import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# ১. অনেকগুলো ডামি ডাটা তৈরি করা
X = np.random.rand(100, 2) * 100 

# ২. WCSS (Error) মাপার জন্য একটি লিস্ট তৈরি
wcss = []

# ৩. ১ থেকে ১০টি ক্লাস্টার নিয়ে লুপ চালিয়ে দেখা
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    # kmeans.inertia_ আমাদের সেই WCSS বা এরর ভ্যালু দেয়
    wcss.append(kmeans.inertia_)

# ৪. গ্রাফ প্লট করা
plt.plot(range(1, 11), wcss, marker='o', color='blue')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS (Error)')
plt.show()
