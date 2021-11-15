# this script plots traffic as a kde on dublin

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import geoplot as gplt
import geoplot.crs as gcrs

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# combine jan and feb data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load jan and feb data (processed already)
jan_traffic_data = pd.read_pickle(".\\data\\scats_jan2020_processed_data.pkl")
feb_traffic_data = pd.read_pickle(".\\data\\scats_feb2020_processed_data.pkl")

# combine into full df
full_traffic_data = pd.concat([jan_traffic_data, feb_traffic_data], ignore_index=True)

# normalise traffic volume data to range [0, 1]
norm_traffic_val = (full_traffic_data["Sum_Volume"] - min(full_traffic_data["Sum_Volume"])) /\
                   (max(full_traffic_data["Sum_Volume"]) - min(full_traffic_data["Sum_Volume"]))

# assign norm-ed col
full_traffic_data["Norm_Vol_WD"] = norm_traffic_val

# load traffic site location data
sites_geodf = pd.read_pickle(".\\data\\valid_scats_sites_geom.pkl")
sites_geodf["SiteID"] = sites_geodf["SiteID"].astype("int64")

# match traffic site geom data to traffic data


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# mapping set-up
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# import shape file of ireland electoral districts
eds_map = gpd.read_file(".\\data\\ireland_eds_map_data\\ireland_eds.shp")

# filter shape file to just dublin electoral regions
dub_eds = eds_map[eds_map["CO_NAME"] == "Dublin"]

# plot just scats sites
fig, ax = plt.subplots(figsize=(10, 10))
plt.grid(True)
gplt.pointplot(sites_geodf, hue='Elev', legend=True)
dub_eds.to_crs(epsg=4326).plot(ax=ax, color="lightgrey")
gplt.kdeplot(sites_geodf, cmap='Reds', shade=True, clip=dub_eds, ax=ax)
plt.title("Map of Dublin with SCATS Traffic Measurement Sites Marked")
sites_geodf.plot(ax=ax, markersize=10,  column="Elev", legend=True)
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# all traffic data together


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# specific scenarios
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# isolate weekday hour of 9am
wd_9 = full_traffic_data[(full_traffic_data["Day_Type"] == "WD") & (full_traffic_data["Hour_in_Day"] == "09")]

ax = gplt.polyplot(dub_eds, zorder=1, edgecolor="lightgrey")
gplt.kdeplot(sites_geodf, cmap='Reds', shade=True, shade_lowest=False, clip=dub_eds, ax=ax)

# TODO fix issue with clip
# TODO wrap into function which takes day and time args

