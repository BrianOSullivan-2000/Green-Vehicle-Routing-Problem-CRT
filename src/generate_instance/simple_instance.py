import src.generate_grid.grid as grid
import random
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, distance_matrix
import math
import networkx as nx
import momepy
import itertools
import osmnx as ox
import utm
from src.elevation_call.create_evel_query_file import create_elev_query_file
from src.elevation_call.read_elev_query_result import read_elev_res


# Bounding box for Dublin
# Round to 4 decimals points otherwise not enough memory for grid
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
vdf = pd.read_pickle("data/distance_matrices/sparse_n20.pkl")
vpoints = list(vdf.columns)

vpoints = np.round(np.array(vpoints), 4)


# We can get the elevations directly from open elevation instead of interpolating (TODO: ask others how to do this)
vdf = pd.DataFrame(vpoints, columns=['longitude', 'latitude'])
#vdf.to_pickle("data/instance_elevs/n20/n20_lat_long.pkl")
#create_elev_query_file("data/instance_elevs/n20/n20_lat_long.pkl", "data/instance_elevs/n20/n20_to_query.json")


# Clean data up a little
ds = ds.drop(ds[ds['Long']==0].index)
ds = ds.drop(ds[ds['Lat']==0].index)
ds = ds.drop(ds[ds['Long']>-6.08442].index)
ds = ds.drop(ds[ds['Long']<-6.47592].index)
ds = ds.drop(ds[ds['Lat']>53.45976].index)
ds = ds.drop(ds[ds['Lat']<53.22938].index)
ds = ds[["Long", "Lat", "Elev"]]


# round values to grid values
epoints = np.round(ds.to_numpy(), 4)

# add points to grid
dublin.add_elevation_points(epoints)
dublin.add_vertices(vpoints)
dublin.create_interpolation(epoints, k=5, p=4)

# create df for grid
dublin.create_df()

# compute matrices for various edges
dublin.compute_distance(mode="OSM", filename="data/distance_matrices/sparse_n20.pkl")
dublin.compute_gradient()
dublin.read_driving_cycle("data/WLTP.csv", h=4, hbefa_filename="data/HBEFA_Driving_Cycles.pkl")
dublin.compute_speed_profile(filename="data/speed_matrices/sparse_n20.pkl")
dublin.create_geometries("data/geom_matrices/sparse_n20.pkl")

dublin.compute_traffic(filename="data/traffic_matrices/weekday_peak.pkl")
dublin.read_highways(filename="data/highway_matrices/sparse_n20.pkl")
dublin.compute_level_of_service()

dublin.read_weather(filename="data/weather_matrices/2016-01-28_4pm_tp.pkl")
dublin.compute_weather_correction()
dublin.read_skin_temp(filename="data/weather_matrices/skt.pkl")

dublin.compute_cost(method="copert with meet")
np.set_printoptions(suppress=True)

dublin.cost_matrix


# In[1]


import src.generate_tsplib.generate_tsplib as tsp

nodes = dublin.df[dublin.df['is_vertice'] != 0].values
nodes = nodes[:, 0:2]
edge_weights = dublin.cost_matrix.to_numpy()


tsp.generate_tsplib(filename="instances/sample_n20", instance_name="instance", capacity=100, edge_weight_type="EXPLICIT", edge_weight_format="FULL_MATRIX",
                    nodes=nodes, demand=np.ones(len(nodes)), depot_index=[0], edge_weights=edge_weights)


# In[1]

lon_b = (-6.33, -6.19)
lat_b = (53.315, 53.37)

df = dublin.df[(dublin.df['x'] > lon_b[0]) & (dublin.df['x'] < lon_b[1])]
df = df[(df['y'] > lat_b[0]) & (df['y'] < lat_b[1])]

df = df.iloc[::10]

# Make dataframe
names = {'Elevation':df['elevation'], 'longitude':df['x'], 'latitude':df['y']}
df = pd.DataFrame(data = names)

# point geometry
geometry = gpd.points_from_xy(df['longitude'], df['latitude'])
map_df = pd.DataFrame(data = names)

gdf = gpd.GeoDataFrame(map_df, columns=['Elevation'], geometry=geometry, crs={'init' : 'epsg:4326'})
gdf = gdf.set_crs(epsg=4326, allow_override=True)


geometry_v = gpd.points_from_xy(vpoints[:, 0], vpoints[:, 1])
gdf_v = gpd.GeoDataFrame(data=dublin.df[dublin.df['is_vertice'] != 0], geometry=geometry_v, crs={'init' : 'epsg:4326'})
gdf_v = gdf_v.set_crs(epsg=4326, allow_override=True)

# Dublin shapefile (NOT IN GITHUB, go to https://www.townlands.ie/page/download/ to access)
dub_df = gpd.read_file("../Brians_Lab/data/counties.shp")
#dub_plot = gpd.read_file("../Brians_Lab/data/civil_parishes.shp")
# Both GeoDataFrames need to have same projection for plotting
dub_df = dub_df.set_crs(epsg=4326)
dub_df = dub_df[dub_df["NAME_TAG"]=="Dublin"]


line_geom = dublin.geom_matrix.values.flatten()

costs = dublin.cost_matrix.values.flatten()[dublin.cost_matrix.values.flatten() != 0]
line_geoms = dublin.geom_matrix.values.flatten()[dublin.geom_matrix.values.flatten() != 0]

line_gdf = gpd.GeoDataFrame(data=costs, geometry=line_geoms, crs={'init' : 'epsg:4326'})

# In[1]


fig, ax = plt.subplots(1, 1, figsize=(10,10))

# Get max and min elevations for plotting elevations colourmap
node_elevs = df['Elevation'].values

# Plot elevations
gdf.plot(ax=ax, column='Elevation', cmap='terrain', vmin = min(node_elevs), vmax = max(node_elevs),
         marker=',', markersize=10, legend=True, alpha=0.7, zorder=1)

# Plot rainfall
#dublin.weather.plot(ax=ax, column='Precipitation', cmap='Blues', legend=True,
            #vmin=0, vmax=np.max(dublin.weather['Precipitation']))

# Plot traffic
#dublin.traffic.plot(ax=ax, column='Traffic', cmap='rainbow', legend=True,
            #vmin=0, vmax=np.max(dublin.traffic['Traffic']))

# Add sparse border
dub_df.plot(ax=ax, color="none", edgecolor="k", alpha=0.5, zorder=2)

# Plot vertices
gdf_v.plot(ax=ax, color="k", marker=',', markersize=5, zorder=5)

line_gdf.plot(ax=ax, alpha=1, color="k", linewidth=1, zorder=6)

# Can label vertices of points
#for x, y, label in zip(gdf_v.geometry.x, gdf_v.geometry.y, gdf_v.is_vertice):
    #ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")

# Bounds for limits
#lon_b = (-6.35, -6.15)
#lat_b = (53.25, 53.4)

lon_b = (-6.33, -6.19)
lat_b = (53.315, 53.37)

# Plot
plt.xlim(lon_b)
plt.ylim(lat_b)

plt.show()
