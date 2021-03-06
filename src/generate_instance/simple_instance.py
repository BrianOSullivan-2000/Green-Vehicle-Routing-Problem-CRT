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
import src.generate_tsplib.generate_tsplib as tsp


# In[1]


# Extra Elevation points
#ds1 = pd.read_pickle("data/elevation_matrices/coords1.pkl")
#ds2 = pd.read_pickle("data/elevation_matrices/coords2.pkl")
#ds3 = pd.read_pickle("data/elevation_matrices/coords3.pkl")
#ds1 = ds1.append(ds2)
#ds1 = ds1.append(ds2)
#ds1.columns = ["Long", "Lat", "Elev"]

# Scat site testing data
ds = pd.read_pickle("data/scats_sites_with_elev.pkl")
ds = ds.loc[:, "Lat":"Elev"]
#ds = ds.append(ds1)

# Clean data up a little
ds = ds.drop(ds[ds['Long']==0].index)
ds = ds.drop(ds[ds['Lat']==0].index)
ds = ds.drop(ds[ds['Long']>-6].index)
ds = ds.drop(ds[ds['Long']<-6.5].index)
ds = ds.drop(ds[ds['Lat']>53.5].index)
ds = ds.drop(ds[ds['Lat']<53.1].index)
ds = ds[["Long", "Lat", "Elev"]]

# round values to grid values
epoints = np.round(ds.to_numpy(), 4)

# Bounding box for Dublin
# Round to 4 decimals points otherwise not enough memory for grid
lon_b = (-6.5, -6)
lat_b = (53.1, 53.5)

# Step size
h = 0.0001

domain, n, depot, traffic, rain = "dublin_south", "50", "centre", "weekday_offpeak", "mild"

def create_instance(domain, n, depot, traffic, rain):

    # Make the Grid
    dublin = grid.Grid(lon_b=lon_b, lat_b=lat_b, h=h)

    v_file = "{}/{}_n{}.pkl".format(domain, depot, n)

    # add points to grid
    dublin.add_elevation_points(epoints, filename="data/elevation_matrices/{}".format(v_file))
    dublin.create_interpolation(k=6, p=2)

    # Vertices
    vdf = pd.read_pickle("data/distance_matrices/{}".format(v_file))
    vpoints = list(vdf.columns)
    vpoints = np.round(np.array(vpoints), 4)

    # We can get the elevations directly from open elevation instead of interpolating (TODO: ask others how to do this)
    vdf = pd.DataFrame(vpoints, columns=['longitude', 'latitude'])
    #vdf.to_pickle("data/instance_elevs/n20/n20_lat_long.pkl")
    #create_elev_query_file("data/instance_elevs/n20/n20_lat_long.pkl", "data/instance_elevs/n20/n20_to_query.json")
    dublin.add_vertices(vpoints)

    # create df for grid
    dublin.create_df()

    # compute matrices for various edges
    dublin.compute_distance(mode="OSM", filename="data/distance_matrices/{}".format(v_file))
    dublin.compute_gradient()
    dublin.read_driving_cycle("data/WLTP.csv", h=4, hbefa_filename="data/HBEFA_Driving_Cycles.pkl")
    dublin.compute_speed_profile(filename="data/speed_matrices/{}".format(v_file))
    dublin.create_geometries("data/geom_matrices/{}".format(v_file))

    dublin.compute_traffic(filename="data/traffic_matrices/{}.pkl".format(traffic))
    dublin.read_highways(filename="data/highway_matrices/{}".format(v_file))
    dublin.compute_level_of_service()

    dublin.read_weather(filename="data/weather_matrices/{}.pkl".format(rain))
    dublin.compute_weather_correction()
    dublin.read_skin_temp(filename="data/weather_matrices/Skin_temperature_averages.pkl")

    dublin.compute_cost(method="copert with meet")
    np.set_printoptions(suppress=True)

    return dublin


# In[1]


ns = ["20", "50", "100", "200", "500", "1000"]
depots = ["centre", "corner"]
traffics = ["weekday_offpeak", "weekday_peak", "weekend_peak"]
rains = ["heavy", "mild", "low"]

for domain in ["dublin_centre", "m50", "dublin_south"]:
    for n in ns:
        for depot in depots:
            for traffic in traffics:
                for rain in rains:


                    if domain == "dublin_south" and n == "1000":
                        print("Ignore this combo")

                    else:
                        dublin = create_instance(domain, n, depot, traffic, rain)

                        nodes = dublin.df[dublin.df['is_vertice'] != 0].values

                        nodes = nodes[:, 0:2]
                        edge_weights = dublin.cost_matrix.to_numpy()

                        if traffic == "weekday_offpeak":
                            t = "wdo"
                        elif traffic == "weekday_peak":
                            t = "wdp"
                        elif traffic == "weekend_peak":
                            t = "wep"

                        filename = "instances/{}/{}_rainfall/{}/{}_n{}".format(domain, rain, traffic, depot, n)

                        d = "m50" if domain == "m50" else "DS"
                        instance_name = "{}_{}_{}_{}_n{}".format(d, depot[0:2], rain[0], t, n)

                        print(instance_name)

                        a1 = dublin.cost_matrix.values.flatten()[dublin.cost_matrix.values.flatten() != 0].shape[0]
                        a2 = dublin.distance_matrix.values.flatten()[dublin.distance_matrix.values.flatten() != 0].shape[0]

                        comment = "Generated by Finucane, Fulcher, O'Sullivan, and Seth (2021)"

                        if a1 != a2:
                            print("{} has a bug")

                        tsp.generate_tsplib(filename=filename, comment=comment, instance_name=instance_name, capacity=100,
                                            edge_weight_type="EXPLICIT", edge_weight_format="SPARSE_MATRIX", nodes=nodes,
                                            demand=np.random.randint(1,4,len(nodes)), depot_index=[0], edge_weights=edge_weights)


# In[1]


#lon_b = (-6.33, -6.19)
#lat_b = (53.315, 53.37)

df = dublin.df[(dublin.df['x'] > lon_b[0]) & (dublin.df['x'] < lon_b[1])]
df = df[(df['y'] > lat_b[0]) & (df['y'] < lat_b[1])]

df = df.sample(frac=0.02)
#df = df.iloc[::50]

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
gdf_vv = gdf_v[gdf_v['is_vertice'] == 1]
gdf_v = gdf_v[gdf_v['is_vertice'] != 1]

# Dublin shapefile (NOT IN GITHUB, go to https://www.townlands.ie/page/download/ to access)
dub_df = gpd.read_file("../Brians_Lab/data/counties.shp")
#dub_plot = gpd.read_file("../Brians_Lab/data/civil_parishes.shp")
# Both GeoDataFrames need to have same projection for plotting
dub_df = dub_df.set_crs(epsg=4326)
dub_df = dub_df[dub_df["NAME_TAG"]=="Dublin"]

gdf = gpd.sjoin(gdf, dub_df).iloc[:, 0:2]

line_geom = dublin.geom_matrix.values.flatten()

costs = dublin.cost_matrix.values.flatten()[dublin.cost_matrix.values.flatten() != 0]
line_geoms = dublin.geom_matrix.values.flatten()[dublin.geom_matrix.values.flatten() != 0]


costs = np.triu(dublin.cost_matrix.values)
line_geoms = np.triu(dublin.geom_matrix.values)

costs = costs[costs != 0][49:]
line_geoms = line_geoms[line_geoms != 0][49:]

line_gdf = gpd.GeoDataFrame(data=costs, geometry=line_geoms, crs={'init' : 'epsg:4326'})


# In[1]


import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('terrain')
new_cmap = truncate_colormap(cmap, 0.2, 1)

fig, ax = plt.subplots(1, 1, figsize=(10,10))

# Get max and min elevations for plotting elevations colourmap
node_elevs = df['Elevation'].values

# Plot elevations
#gdf.plot(ax=ax, column='Elevation', cmap=new_cmap, norm=colors.PowerNorm(gamma=0.6),
#         marker=',', markersize=15, legend=True, legend_kwds={"label":"Elevation (m above sea-level)"}, alpha=0.7, zorder=1)

# Plot rainfall
#dublin.weather.plot(ax=ax, column='Precipitation', cmap='Blues', legend=True,
            #vmin=0, vmax=np.max(dublin.weather['Precipitation']))

# Plot traffic
#dublin.traffic.plot(ax=ax, column='Traffic', cmap='rainbow', legend=True,
            #vmin=0, vmax=np.max(dublin.traffic['Traffic']))

# Add sparse border
dub_df.plot(ax=ax, color="c", edgecolor="k", alpha=0.2, zorder=2)

# Plot vertices
gdf_v.plot(ax=ax, color="k", alpha=1, marker=',', markersize=20, zorder=6)
gdf_vv.plot(ax=ax, color="k", alpha=1, marker='*', markersize=200, zorder=6)
# Plot edges
line_gdf.plot(ax=ax, alpha=1, color="r", linewidth=0.4, zorder=5, legend=True)

# Can label vertices of points
#for x, y, label in zip(gdf_v.geometry.x, gdf_v.geometry.y, gdf_v.is_vertice):
    #ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")

# Bounds for limits
lon_bb = (-6.40, -6.08)
lat_bb = (53.225, 53.33)

#lon_b = (-6.33, -6.19)
#lat_b = (53.315, 53.37)

# Plot
plt.xlim(lon_bb)
plt.ylim(lat_bb)

plt.legend(labels = ["Vertex", "Depot", "Edge"])
plt.xlabel("Longitude", fontsize=15)
plt.ylabel("Latitude", fontsize=15)
plt.title("South Dublin, n100 Instance ", size=20)
plt.savefig("data/figures/Dublin_South_n100.jpeg", dpi=300)
plt.show()
