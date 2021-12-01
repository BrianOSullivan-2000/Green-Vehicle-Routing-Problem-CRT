
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import pandas as pd
import geopandas as gpd
import utm
from scipy.spatial import cKDTree, distance_matrix

# Open the ERA5 dataset, just get precipitation
ds = xr.open_dataset("data/ERA5_JanFeb.nc")
df = ds.to_dataframe()
df1 = df['tp'].dropna()
df1 = df1.reset_index()

# Lets go with my birthday
time_idx = df1['time'].iloc[249568]
gdf = df1[df1['time'] == time_idx]

# Convert to mm
gdf.loc[:, 'tp'] = gdf.loc[:, 'tp'] * 1000


# In[1]


# Bounds
lon_b = (-6.7, -6.0)
lat_b = (53.0, 53.7)
h = 0.01

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
w = 1.0 / d**3
weighted_averages = np.sum(w * prec.values[inds], axis=1) / np.sum(w, axis=1)

# Set elevation to each gridpoint
tp = np.reshape(weighted_averages, (len(x), len(y)))


# In[1]

# Get gdfs ready for plotting
# Interpolated data

i_geometry = gpd.points_from_xy(xx.flatten(), yy.flatten())
i_names = {'Precipitation':tp.flatten(), 'longitude':xx.flatten(), 'latitude':yy.flatten()}
i_gdf = gpd.GeoDataFrame(pd.DataFrame(data=i_names), columns=['Precipitation'], geometry=i_geometry, crs={'init' : 'epsg:4326'})

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

gdf = gdf[gdf.within(dub_df.geometry.values[0])]
i_gdf = i_gdf[i_gdf.within(dub_df.geometry.values[0])]


# In[1]


# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10,10))
dub_df.plot(ax=ax, color='none', edgecolor="k", alpha=1, zorder=3)

i_gdf.plot(ax=ax, column='Precipitation', cmap='Blues', vmin=0,
         marker=',', markersize=500, legend=True, alpha=1, zorder=1)

gdf.plot(ax=ax, color="k", marker=',', markersize=5, alpha=1, zorder=2)

ggdf.plot(ax=ax, markersize=10, zorder=3)
# Bounds for limits
#lon_b = (-6.4759, -6.0843)
#lat_b = (53.2294, 53.4598)

plt.xlim(lon_b)
plt.ylim(lat_b)
plt.show()
