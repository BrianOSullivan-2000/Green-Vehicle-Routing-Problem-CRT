# this script plots a topographic map of dublin

import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# import dublin location-elev data
# .001 lat and long step size
elev_df_001 = pd.read_pickle("data/elevation_call_for_plots/dub_grid_elev_001res.pkl")

# get geometry
geometry = gpd.points_from_xy(elev_df_001['longitude'], elev_df_001['latitude'])

# create geodf
gdf = gpd.GeoDataFrame(elev_df_001, columns=['elevation'], geometry=geometry, crs={'init': 'epsg:4326'})
gdf = gdf.set_crs(epsg=4326, allow_override=True)
gdf["elevation"] = gdf["elevation"].astype(int).copy()  # needed for plot

# import counties shapefile
dub_df = gpd.read_file("./data/counties/counties.shp")
# match projection
dub_df = dub_df.set_crs(epsg=4326)
# filter to dublin
dub_df = dub_df[dub_df["NAME_TAG"] == "Dublin"]


# define new colour map truncation
# for clearer values around zero
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(
        n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap


# generate new colour map
cmap = plt.get_cmap('terrain')
new_cmap = truncate_colormap(cmap, 0.2, 1)

# plot dublin grid
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# plot elevations
gdf.clip(mask=dub_df).plot(ax=ax, column='elevation',  cmap=new_cmap, marker=',', markersize=5,
                           alpha=1, zorder=2, legend=True, legend_kwds={'label': "m above sea level"},
                           norm=colors.PowerNorm(gamma=0.5))

# plot dublin county boundary
dub_df.plot(ax=ax, color='none', edgecolor="k", alpha=0.2, zorder=3)

# plot params
plt.xlim((-6.5, -6.05))
plt.ylim((53.20, 53.5))
plt.title("Topography of Dublin", fontsize=15)
plt.xlabel("Longitude ($^\circ$)")
plt.ylabel("Latitude ($^\circ$)")
