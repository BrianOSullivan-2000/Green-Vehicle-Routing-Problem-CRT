
# In[1]

# Import some packages
import numpy as np
import random as random
import matplotlib.pyplot as plt
import scipy.spatial

# Make N random points
N = 100
points = np.random.rand(N, 2) * 100

# Compute Euclidean distance between all points
distances = scipy.spatial.distance.cdist(points, points)

# In[2]

# Plot
plt.scatter(points[:, 0], points[:, 1], s=5)
plt.title("Random Points!")
plt.show()
