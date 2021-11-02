import xarray as xr
import netCDF4 as nc
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np


# In[1]


# Bounding box for Dublin
lon_b = (-6.475924, -6.084280)
lat_b = (53.229386, 53.459765)

# Read in points from data folder
ds = pd.read_pickle("data/scats_sites_with_elev.pkl")

# Filter data
df = ds.loc[:, "Lat":"Elev"]
df1 = df.dropna(thresh=3)
df1 = df1.drop(df1[df1.Elev==0].index)

# Shapefile to be used as overlay
dub_df = gpd.read_file("data/dublin/townlands.shp")

# Both GeoDataFrames need to have same projection for plotting
dub_df = dub_df.set_crs(epsg=4326)


# In[2]


# Create GeoDataFrame with point geometry
geometry = gpd.points_from_xy(df1['Long'], df1['Lat'])
names = {'Elevation':df1['Elev'], 'longitude':df1['Long'], 'latitude':df1['Lat']}
map_df = pd.DataFrame(data = names)
gdf = gpd.GeoDataFrame(map_df, columns=['Elevation'], geometry=geometry, crs={'init' : 'epsg:4326'})


# In[3]

# Plotting function for GeoDataFrames
def map_plot(gdf, variable, markersize):

    # Plot formatting
    plt.rcParams["font.serif"] = "Times New Roman"
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    # Dublin map overlay
    dub_df.plot(ax=ax, color="white", edgecolor="black", alpha=0.2)

    # Plot data
    gdf.plot(ax=ax, column=variable, cmap='rainbow', vmin = 0, vmax = 150, marker=',', markersize=markersize, legend=True)


# In[4]

# Final output
map_plot(gdf, 'Elevation', 5)
plt.xlim(lon_b)
plt.ylim(lat_b)
plt.show()
