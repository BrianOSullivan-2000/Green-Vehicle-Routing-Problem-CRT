import src.generate_grid.grid as grid
import random
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
from scipy.spatial import cKDTree, distance_matrix
import math


# In[1]

# Bounding box for Dublin
lon_b = (-6.475924, -6.084280)
lat_b = (53.229386, 53.459765)

# Round to 4 decimals points otherwise not enough memory for grid
# TODO: look into 5 or 6 decimal points
lon_b = (-6.4759, -6.0843)
lat_b = (53.2294, 53.4598)

# Step size
h = 0.0001

# Make the Grid
dublin = grid.Grid(lon_b=lon_b, lat_b=lat_b, h=h, load=True)


# In[2]

# Scat site testing data
ds = pd.read_pickle("data/scats_sites_with_elev.pkl")
ds = ds.loc[:, "Lat":"Elev"]

ds = ds.sample(100)

# Clean data up a little
ds = ds.drop(ds[ds['Long']==0].index)
ds = ds.drop(ds[ds['Lat']==0].index)
ds = ds.drop(ds[ds['Long']>-6.0843].index)
ds = ds.drop(ds[ds['Long']<-6.4759].index)
ds = ds.drop(ds[ds['Lat']>53.4598].index)
ds = ds.drop(ds[ds['Lat']<53.2294].index)
ds = ds[["Long", "Lat", "Elev"]]

# round values to grid values
# TODO: inverstigate if ok
epoints = np.round(ds.to_numpy(), 4)

# add points to grid
dublin.add_elevation_points(epoints)
dublin.add_vertices(epoints)
dublin.create_interpolation(epoints)

# create df for grid
dublin.create_df()


# compute matrices for various edges
dublin.compute_distance()
dublin.compute_gradient()
dublin.read_driving_cycle("data/WLTP.csv", h=4)
dublin.compute_speed_profile()
dublin.compute_cost()

dublin.cost_matrix


# In[1]

import src.generate_tsplib.generate_tsplib as tsp

nodes = dublin.df[dublin.df['is_vertice'] != 0].values
nodes = nodes[:, 0:2]
edge_weights = dublin.cost_matrix.to_numpy()


tsp.generate_tsplib(filename="sample", instance_name="instance_01", capacity=100, edge_weight_type="EXPLICIT", edge_weight_format="FULL_MATRIX",
                    nodes=nodes, demand=np.ones(len(nodes)), depot_index=[0], edge_weights=edge_weights)
