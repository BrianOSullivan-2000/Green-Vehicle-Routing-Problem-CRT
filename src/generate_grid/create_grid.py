import src.generate_grid.grid as grid
import random
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
from scipy.spatial import cKDTree, distance_matrix
import seaborn as sns

# In[1]

# epoints and vpoints are random sets of points to simulate
# elevation data and graph vertices, respectively

epoints = []
vx, vy = [], []

# Generate 100 epoints at random locations with elevations from 0 to 20
for _ in range(100):

    x, y = random.randrange(1000), random.randrange(1000)
    z = np.ceil(random.random() * 20)
    epoints.append([x, y, z])

# Generate 100 vertices at random locations
for _ in range(100):

    x, y = random.randrange(1000), random.randrange(1000)
    vx.append(x)
    vy.append(y)

# Convert to numpy arrays for Grid class
epoints = np.array(epoints, dtype='int')
vpoints = np.vstack((np.sort(vx), vy)).T


# In[2]

# Create 1000*1000 grid
dublin = grid.Grid(1000, 1)

# Add vertices and elevations to grid
dublin.add_vertices(vpoints)
dublin.add_elevation_points(epoints)

# Interpolate elevation across grid using epoints as input data
dublin.create_interpolation(epoints)

# Create dataframe of grid for pandas/geopandas operations
dublin.create_df()

# Read in WLTP driving cycle
dublin.read_driving_cycle("data/WLTP.csv", h=4)

# Create distance, gradient and speed matrices between all vertices
dublin.compute_distance()
dublin.compute_gradient()
dublin.compute_speed_profile()

# Compute cost matrix with MEET methodology
# Each value represents CO2 emitted along path in kg
dublin.compute_cost(method="meet")
c_meet = dublin.cost_matrix


# In[3]

# For -6% gradient COPERT returns clearly incorrect negative emissions
# Check for negative values in copert cost_matrix

#dublin.compute_cost(method="copert")
#c_copert = dublin.cost_matrix
#x = c_copert[c_copert < 0].values.flatten()
#len(x[~np.isnan(x)]) / len(x)

# Approx. 3.5% incorrect outputs using randomly simulated data


# In[4]

# Plotting tool for graph

# Creat figure and elevation dataframe
fig, ax = plt.subplots(figsize=(11, 9))
df = pd.DataFrame(dublin.elevation)


# Generate elevation heatmap with overlay of vertices
ax = sns.heatmap(data=df, cmap="terrain", vmin=0, vmax=30)
ax.invert_yaxis()
ax = sns.scatterplot(data=dublin.df[dublin.df['is_vertice'] != 0], x='x', y='y', color="0")
ax.set_xticks(range(0, len(dublin.xx), int(len(dublin.xx)/20)))
ax.set_xticklabels(range(0, len(dublin.xx), int(len(dublin.xx)/20)))
ax.set_yticks(range(0, len(dublin.xx), int(len(dublin.xx)/20)))
ax.set_yticklabels(range(0, len(dublin.xx), int(len(dublin.xx)/20)))
plt.show()


# In[5]

# quick indicator of slope impact for MEET method (multiplicative factor)
# init available gradients, read in gradient coefficients, let average velocity = 100km/hr
grads = [-6, -4, -2, 2, 4, 6]
MEETdf = pd.read_csv("data\MEET_Slope_Correction_Coefficients_Light_Diesel_CO2.csv")
v = 100

# Find gradient correction factor at each gradient, output results
for g in grads:
    cf = MEETdf[MEETdf['Slope (%)'] == g].loc[:, "A6":"A0"].values[0, :]
    print(cf[0]*v**6 + cf[1]*v**5 + cf[2]*v**4 + cf[3]*v**3 + cf[4]*v**2 + cf[5]*v + cf[6])
