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

# Scat site testing data
ds = pd.read_pickle("data/scats_sites_with_elev.pkl")
ds = ds.loc[:, "Lat":"Elev"]

# Vertices
vdf = pd.read_json("data/distance_matrices/n20.json")

# Read osm graph to get coordinates
G = nx.read_gpickle("../Brians_Lab/data/dublin_graph.gpickle/dublin_graph.gpickle")
nodes, edges, W = momepy.nx_to_gdf(G, spatial_weights=True)

nodes = nodes[nodes["osmid"].isin(vdf.index)]
vpoints = np.around(nodes.loc[:, 'x':'y'].to_numpy(), 4)


# In[1]


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
epoints
# add points to grid
dublin.add_elevation_points(epoints)
dublin.add_vertices(vpoints)
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


tsp.generate_tsplib(filename="instances/sample_n20", instance_name="instance", capacity=100, edge_weight_type="EXPLICIT", edge_weight_format="FULL_MATRIX",
                    nodes=nodes, demand=np.ones(len(nodes)), depot_index=[0], edge_weights=edge_weights)


# In[1]


sample_ids = vdf.index
# every node pair combination
pairs = np.array(list(itertools.combinations(sample_ids, 2)))

paths = []
count = 0

for pair in pairs:

    # get path length, add to matrix, count
    paths.append(nx.shortest_path(G, pair[0], pair[1], weight="length"))
    count += 1

    # impromptu progress bar
    if count % 5 == 0:
        print(count // 5)


# In[1]

df = dublin.df.iloc[::200, :]

# Make dataframe
names = {'Elevation':df['elevation'], 'longitude':df['x'], 'latitude':df['y']}
df = pd.DataFrame(data = names)

# point geometry
geometry = gpd.points_from_xy(df['longitude'], df['latitude'])
map_df = pd.DataFrame(data = names)

gdf = gpd.GeoDataFrame(map_df, columns=['Elevation'], geometry=geometry, crs={'init' : 'epsg:4326'})
gdf = gdf.set_crs(epsg=4326, allow_override=True)


# In[1]


geometry_v = gpd.points_from_xy(vpoints[:, 0], vpoints[:, 1])
gdf_v = gpd.GeoDataFrame(data=pd.DataFrame(vpoints), geometry=geometry_v, crs={'init' : 'epsg:4326'})

# Dublin shapefile (NOT IN GITHUB, go to https://www.townlands.ie/page/download/ to access)
dub_df = gpd.read_file("../Brians_Lab/data/counties.shp")
dub_plot = gpd.read_file("../Brians_Lab/data/civil_parishes.shp")
# Both GeoDataFrames need to have same projection for plotting
dub_df = dub_df.set_crs(epsg=4326)
dub_df = dub_df[dub_df["NAME_TAG"]=="Dublin"]


gdf = gdf[geometry.within(dub_df.geometry.unary_union)]


# In[1]

# Prepare paths for plotting
steps = []

# Add each step
for path in paths:
    for i in range(len(path)-1):
        steps.append([path[i], path[i+1]])

# Find corresponding edge for each node pair
steps = np.array(steps)
path_edges = edges[(edges['u'].isin(steps[: ,0])) & (edges['v'].isin(steps[: ,1]))]


# In[1]

# Clean road network data
edges2 = edges

# Speed limits and road types
speeds = ["50", "30", "80", "60", "40", "100", "120", "20"]
types = np.unique(edges['highway'].values)
types = types[[4,5,7,8,9,10,11,12,14,15]]

# Pandas filtering
edges2 = edges2[edges2['maxspeed'].isin(speeds)]
edges2 = edges2[edges2['highway'].isin(types)]


# In[1]

fig, ax = plt.subplots(1, 1, figsize=(10,10))

#ox.plot_graph_routes(G, paths, ax=ax, bgcolor='#ffffff', edgecolor="#111111", edge_alpha=0.5,
#                     node_alpha=0, show=False, ig_dest_size=5, route_linewidth=2, node_size=0)

# Plot elevations
gdf.plot(ax=ax, column='Elevation', cmap='terrain', vmin = -60, vmax = 200,
         marker=',', markersize=10, legend=True, alpha=0.7, zorder=1)

# Add county border
dub_df.plot(ax=ax, color="none", edgecolor="k", alpha=0.5, zorder=2)


# Plot road network and paths
edges2.plot(ax=ax, alpha=0.2, color="k", linewidth=0.5, zorder=3)
path_edges.plot(ax=ax, color="crimson", linewidth=1.5, zorder=4)

# Plot vertices
gdf_v.plot(ax=ax, color="k", marker=',', markersize=20, zorder=5)


# Bounds for limits
lon_b = (-6.4759, -6.0843)
lat_b = (53.2294, 53.4598)

# Plot
plt.xlim(lon_b)
plt.ylim(lat_b)
plt.savefig("data/figures/n20.jpeg")
plt.show()
