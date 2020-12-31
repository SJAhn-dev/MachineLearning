#!/usr/bin/env python
# coding: utf-8

# In[70]:


from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)


kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.title('N_clusters 10', pad=10)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, edgecolors='k')
plt.show()

X_varied, y_varied = make_blobs(n_samples = 200, cluster_std = [4.0, 2.0, 4.0], random_state = 170)
y_pred = KMeans(n_clusters = 5, random_state = 0).fit_predict(X_varied)
plt.title('std = 4.0, 2.0, 4.0 / clusters = 5')
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred, s=60, edgecolors='k')


# In[ ]:




