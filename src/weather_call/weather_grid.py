
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import pandas as pd
import geopandas as gpd
import utm
from scipy.spatial import cKDTree, distance_matrix
from shapely.geometry import Polygon


# Open the ERA5 dataset, just get precipitation
ds = xr.open_dataset("data/ERA5_JanFeb.nc")
df = ds.to_dataframe()
df1 = df['tp'].dropna()
df1 = df1.reset_index()

# Lets go with my birthday
time_idx = df1['time'].iloc[249568]
gdf = df1[df1['time'] == time_idx]
gdf.to_pickle("data/weather_matrices/2016-01-28_4pm.pkl")

# Convert to mm
gdf.loc[:, 'tp'] = gdf.loc[:, 'tp'] * 1000


# Bounds
lon_b = (-6.7, -6.0)
lat_b = (53.0, 53.7)
h = 0.005

# Quick interpolation of weather data, get original points and grid to interpolate over
lons, lats, prec = gdf['longitude'], gdf['latitude'], gdf['tp']

x, y = np.arange(lon_b[0], lon_b[1], h), np.arange(lat_b[0], lat_b[1], h)
yy, xx = np.meshgrid(y, x)

# Convert both input points and gridpoints to correct grid using UTM projection
obs_raw = np.asarray(utm.from_latlon(np.asarray(lons), np.asarray(lats))[0:2])
obs = np.stack((obs_raw[0], obs_raw[1]), axis=1)
grid_obs_raw = np.asarray(utm.from_latlon(xx.ravel(), yy.ravel())[0:2])
grid_obs = np.stack((grid_obs_raw[0], grid_obs_raw[1]), axis=1)


# IDW2 interpolation
tree = cKDTree(np.array(obs))
d, inds = tree.query(np.array(grid_obs), k=7)
w = 1.0 / d**2
weighted_averages = np.sum(w * prec.values[inds], axis=1) / np.sum(w, axis=1)

# Set elevation to each gridpoint
tp = np.reshape(weighted_averages, (len(x), len(y)))


i_geometry = gpd.points_from_xy(xx.flatten(), yy.flatten())
i_names = {'Precipitation':tp.flatten(), 'longitude':xx.flatten(), 'latitude':yy.flatten()}
i_gdf = gpd.GeoDataFrame(pd.DataFrame(data=i_names), columns=['Precipitation'], geometry=i_geometry, crs={'init' : 'epsg:4326'})


# Get grid of midpoints
x_mid, y_mid = np.arange(x[0] - (h/2), x[-1] + 2*(h/2), h), np.arange(y[0] - (h/2), y[-1] + 2*(h/2), h)
x0, y0 = x_mid[0], y_mid[0]
x_step, y_step = x0, y0

recs = []
for i in range(len(x)):

    y_step = y0

    for j in range(len(y)):

        recs.append([(x_step, y_step), (x_step+h, y_step), (x_step+h, y_step+h), (x_step, y_step+h)])
        y_step += h

    x_step += h


grid_geom = pd.Series(recs).apply(lambda x: Polygon(x))
i_gdf['geometry'] = grid_geom


# Bin values according to rainfall ranges
# Two different options, m50 study by De Courcy et al
# or London study by Tsapakis et al
m50_bins = np.array((0, 0.0005, 0.5, 4, 50))
london_bins = np.array((0, 0.0005, 0.2, 6, 50))

m50_vals = np.array((1, 1-0.025, 1-0.053, 1-0.155))
london_vals = np.array((1, 1-0.021, 1-0.038, 1-0.06))

# Going with london metrics for now
i_gdf['Rain_Type'] = np.digitize(i_gdf['Precipitation'], london_bins)
i_gdf['Rain_Type'] = np.array([london_vals[idx] for idx in i_gdf['Rain_Type']])


# In[1]


geom_df = pd.read_pickle("data/geom_matrices/county_n20.pkl")

inter_recs = i_gdf[i_gdf.crosses(geom_df.iloc[0, 1])]

if not (inter_recs['Rain_Type']==inter_recs['Rain_Type'].iloc[0]).all():

    total_len = geom_df.iloc[0, 1].length
    len_weights = []

    for rec in inter_recs['geometry']:
        len_weights.append(rec.intersection(geom_df.iloc[0, 1]).length / total_len)

    net_weight = np.sum(np.array(len_weights) * inter_recs['Rain_Type'])

else:
    net_weight = inter_recs['Rain_Type'].iloc[0]


# In[1]


# Raw data
geometry = gpd.points_from_xy(gdf['longitude'], gdf['latitude'])
names = {'Precipitation':gdf['tp'], 'longitude':gdf['longitude'], 'latitude':gdf['latitude']}
gdf = gpd.GeoDataFrame(pd.DataFrame(data=names), columns=['Precipitation'], geometry=geometry, crs={'init' : 'epsg:4326'})


# Get the map overlay of Dublin
dub_df = gpd.read_file("../Brians_Lab/data/counties.shp")
dub_df = dub_df.set_crs(epsg=4326)
dub_df = dub_df[dub_df["NAME_TAG"]=="Dublin"]

gg = gpd.points_from_xy([-6.246406, -6.319458],  [53.428524, 53.353687])
ggdf = gpd.GeoDataFrame(data={'tp':[0,0]}, geometry=gg, crs={'init' : 'epsg:4326'})

#i_gdf = i_gdf.iloc[::50, :]

#gdf = gdf[gdf.within(dub_df.geometry.values[0])]
#i_gdf = i_gdf[i_gdf.within(dub_df.geometry.values[0])]


# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10,10))
dub_df.plot(ax=ax, color='none', edgecolor="k", alpha=1, zorder=3)

# Can plot rainfall by mm or by factor
i_gdf.plot(ax=ax, column='Precipitation', cmap='Blues', legend=True,
            vmin=0, vmax=np.max(i_gdf['Precipitation']))

#i_gdf.plot(ax=ax, column='Rain_Type', cmap='Blues', legend=True,
#            vmin=0, vmax=np.max(i_gdf['Rain_Type']))


gdf.plot(ax=ax, column='Precipitation',  cmap='Blues', marker=',', markersize=5,
         vmin=0, vmax=np.max(i_gdf['Precipitation']), alpha=1, zorder=2)

ggdf.plot(ax=ax, color='k', markersize=10, zorder=3)

# Bounds for limits
#lon_b = (-6.4759, -6.0843)
#lat_b = (53.2294, 53.4598)

#plt.xlim(lon_b)
#plt.ylim(lat_b)
plt.show()
