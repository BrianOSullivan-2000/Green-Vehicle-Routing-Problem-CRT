import src.generate_grid.grid as grid
import random
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
from scipy.spatial import cKDTree, distance_matrix
import math
import networkx as nx
import momepy
import itertools
import osmnx as ox
import utm


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
dublin = grid.Grid(lon_b=lon_b, lat_b=lat_b, h=h)

# Scat site testing data
ds = pd.read_pickle("data/scats_sites_with_elev.pkl")
ds = ds.loc[:, "Lat":"Elev"]

# Vertices
vdf = pd.read_pickle("data/distance_matrices/sparse_n200.pkl")

vpoints = list(vdf.columns)


vpoints = np.round(np.array(vpoints), 4)


# Clean data up a little
ds = ds.drop(ds[ds['Long']==0].index)
ds = ds.drop(ds[ds['Lat']==0].index)
ds = ds.drop(ds[ds['Long']>-6.08442].index)
ds = ds.drop(ds[ds['Long']<-6.47592].index)
ds = ds.drop(ds[ds['Lat']>53.45976].index)
ds = ds.drop(ds[ds['Lat']<53.22938].index)
ds = ds[["Long", "Lat", "Elev"]]


# round values to grid values
# TODO: investigate if ok
epoints = np.round(ds.to_numpy(), 4)

# add points to grid
dublin.add_elevation_points(epoints)
dublin.add_vertices(vpoints)
#dublin.create_interpolation(epoints)

# create df for grid
dublin.create_df()

# compute matrices for various edges
dublin.compute_distance(mode="OSM", filename="data/distance_matrices/sparse_n200.pkl")
dublin.compute_gradient()
dublin.read_driving_cycle("data/WLTP.csv", h=4)
dublin.compute_speed_profile(filename="data/speed_matrices/sparse_n200.pkl")
dublin.create_geometries("data/geom_matrices/sparse_n200.pkl")
dublin.compute_cost(method="COPERT with meet")

np.set_printoptions(suppress=True)

dublin.cost_matrix.values.flatten()[dublin.cost_matrix.values.flatten()!=0]



# In[1]

import src.generate_tsplib.generate_tsplib as tsp

nodes = dublin.df[dublin.df['is_vertice'] != 0].values
nodes = nodes[:, 0:2]
edge_weights = dublin.cost_matrix.to_numpy()


tsp.generate_tsplib(filename="instances/sample_n200", instance_name="instance", capacity=100, edge_weight_type="EXPLICIT", edge_weight_format="FULL_MATRIX",
                    nodes=nodes, demand=np.ones(len(nodes)), depot_index=[0], edge_weights=edge_weights)


# In[1]

df = dublin.df.iloc[::500, :]

# Make dataframe
names = {'Elevation':df['elevation'], 'longitude':df['x'], 'latitude':df['y']}
df = pd.DataFrame(data = names)

# point geometry
geometry = gpd.points_from_xy(df['longitude'], df['latitude'])
map_df = pd.DataFrame(data = names)

gdf = gpd.GeoDataFrame(map_df, columns=['Elevation'], geometry=geometry, crs={'init' : 'epsg:4326'})
gdf = gdf.set_crs(epsg=4326, allow_override=True)


geometry_v = gpd.points_from_xy(vpoints[:, 0], vpoints[:, 1])
gdf_v = gpd.GeoDataFrame(data=pd.DataFrame(vpoints), geometry=geometry_v, crs={'init' : 'epsg:4326'})
gdf_v = gdf_v.set_crs(epsg=4326, allow_override=True)

# Dublin shapefile (NOT IN GITHUB, go to https://www.townlands.ie/page/download/ to access)
dub_df = gpd.read_file("../Brians_Lab/data/counties.shp")
#dub_plot = gpd.read_file("../Brians_Lab/data/civil_parishes.shp")
# Both GeoDataFrames need to have same projection for plotting
dub_df = dub_df.set_crs(epsg=4326)
dub_df = dub_df[dub_df["NAME_TAG"]=="Dublin"]

gdf = gdf[geometry.within(dub_df.geometry.values[0])]

# In[1]


fig, ax = plt.subplots(1, 1, figsize=(10,10))

# Plot elevations
gdf.plot(ax=ax, column='Elevation', cmap='terrain', vmin = -60, vmax = 200,
         marker=',', markersize=15, legend=True, alpha=0.7, zorder=1)

# Add county border
dub_df.plot(ax=ax, color="none", edgecolor="k", alpha=0.5, zorder=2)


# Plot vertices
gdf_v.plot(ax=ax, color="k", marker=',', markersize=1, zorder=5)


# Bounds for limits
lon_b = (-6.4759, -6.0843)
lat_b = (53.2294, 53.4598)

# Plot
plt.xlim(lon_b)
plt.ylim(lat_b)

plt.show()
