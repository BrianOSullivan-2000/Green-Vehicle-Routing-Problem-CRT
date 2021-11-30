import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# Plot for plotting instances
# TODO: Turn into function which can be called in other files

# In[1]

ds = pd.read_pickle("data/scats_sites_with_elev.pkl")

# Filter data
ds = ds.drop(ds[ds['Long']==0].index)
ds = ds.drop(ds[ds['Lat']==0].index)
ds = ds.drop(ds[ds['Long']>-6.0843].index)
ds = ds.drop(ds[ds['Long']<-6.4759].index)
ds = ds.drop(ds[ds['Lat']>53.4598].index)
ds = ds.drop(ds[ds['Lat']<53.2294].index)
df1 = ds[["Long", "Lat", "Elev"]]


# Make dataframe
names = {'Elevation':df1['Elev'], 'longitude':df1['Long'], 'latitude':df1['Lat']}
df1 = pd.DataFrame(data = names)

# point geometry
geometry1 = gpd.points_from_xy(df1['longitude'], df1['latitude'])
# data
names1 = {'Elevation':df1['Elevation'], 'longitude':df1['longitude'], 'latitude':df1['latitude']}

# map DataFrame
map_df1 = pd.DataFrame(data = names1)

# geopandas DataFrame
gdf1 = gpd.GeoDataFrame(map_df1, columns=['Elevation'], geometry=geometry1, crs={'init' : 'epsg:4326'})


# Dublin shapefile (NOT IN GITHUB, go to https://www.townlands.ie/page/download/ to access)
dub_df = gpd.read_file("../Brians_Lab/data/townlands.shp")
# Both GeoDataFrames need to have same projection for plotting
dub_df = dub_df.set_crs(epsg=4326)

# In[2]

# Plotting function for GeoDataFrames
def map_plot(gdf, variable, markersize):

    # Plot formatting
    plt.rcParams["font.serif"] = "Times New Roman"
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    # Dublin map overlay
    #dub_df.plot(ax=ax, color="white", edgecolor="black", alpha=1)

    # Plot data
    gdf.plot(ax=ax, column=variable, cmap='terrain', vmin = 0, vmax = 150,
             marker=',', markersize=markersize, legend=True, alpha=1)

    gdf.plot(ax=ax, color="k", marker=',', markersize=5)

# In[3]

# Bounds for limits
lon_b = (-6.4759, -6.0843)
lat_b = (53.2294, 53.4598)

# Plot
map_plot(gdf1, 'Elevation', 2)
plt.xlim(lon_b)
plt.ylim(lat_b)
plt.show()


# In[4]


import xarray as xr

ds = xr.open_dataset("../Brians_Lab/data/ERA5_Test.nc")
df = ds.to_dataframe()
df1 = df.dropna(thresh=3)
df1 = df1.reset_index()


# In[5]

df2=df1
df2 = df2.drop(['u10','v10', 'd2m', 't2m', 'skt', 'snowc', 'sf', 'sp'], axis=1)

df2 = df2.drop(df2[df2['longitude']<lon_b[0]].index)
df2 = df2.drop(df2[df2['longitude']>lon_b[1]].index)
df2 = df2.drop(df2[df2['latitude']<lat_b[0]].index)
df2 = df2.drop(df2[df2['latitude']>lat_b[1]].index)

geometry = gpd.points_from_xy(df2['longitude'], df2['latitude'])

names2 = {'Precipitation':df2['tp']-273.15, 'longitude':df2['longitude'], 'latitude':df2['latitude']}
map_df2 = pd.DataFrame(data = names2)

gdf2 = gpd.GeoDataFrame(map_df2, columns=['Precipitation'], geometry=geometry, crs={'init' : 'epsg:4326'})

map_plot(gdf2, "Precipitation", 2)


bins = np.array((0, 0.05, 0.5, 4))

arr = np.digitize(df2['tp'].values, bins)

len(arr[arr==1]) /arr.shape[0]
len(arr[arr==2]) /arr.shape[0]
len(arr[arr==3]) /arr.shape[0]
len(arr[arr==4]) /arr.shape[0]

np.min(df2['tp'].values)
