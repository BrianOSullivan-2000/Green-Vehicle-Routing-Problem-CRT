# this script plots traffic as a kde on dublin

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import geoplot as gplt
import geoplot.crs as gcrs
import numpy as np
import utm
import scipy.interpolate as interp

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# combine jan and feb data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load jan and feb data (processed already)
jan_traffic_data = pd.read_pickle(".\\data\\scats_jan2020_processed_data.pkl")
feb_traffic_data = pd.read_pickle(".\\data\\scats_feb2020_processed_data.pkl")

# remove abnormal dates from jan
jan_traffic_data = jan_traffic_data[~(jan_traffic_data["Day_in_Month"].isin([1, 2, 3]))]

# combine into full df
full_traffic_data = pd.concat([jan_traffic_data, feb_traffic_data], ignore_index=True)

# normalise traffic volume data to range [0, 1]
norm_traffic_val = (full_traffic_data["All_Detector_Vol"] - min(full_traffic_data["All_Detector_Vol"])) /\
                   (max(full_traffic_data["All_Detector_Vol"]) - min(full_traffic_data["All_Detector_Vol"]))

# assign norm-ed col
full_traffic_data["Norm_Vol"] = norm_traffic_val

# load traffic site location data
sites_geodf = pd.read_pickle(".\\data\\valid_scats_sites_geom.pkl")
sites_geodf["SiteID"] = sites_geodf["SiteID"].astype("int64")


# match traffic site elev data to traffic data
def site_elev(row):
    return sites_geodf[sites_geodf["SiteID"] == row["Site"]]["Elev"].iloc[0]


# match traffic site geom data to traffic data
def site_geom(row):
    return sites_geodf[sites_geodf["SiteID"] == row["Site"]]["geometry"].iloc[0]


# assign matching elev
full_traffic_data["Elev"] = full_traffic_data.apply(site_elev, axis=1)

# assign matching geometry
full_traffic_data["geometry"] = full_traffic_data.apply(site_geom, axis=1)


# attempting interpolation
# isolate latitudes and longitudes
lats = full_traffic_data["geometry"].apply(lambda p: p.y)
lons = full_traffic_data["geometry"].apply(lambda p: p.x)

# isolate variable of interest
data = full_traffic_data["All_Detector_Vol"][0:10]

# convert to utm coords - eastings
eastings = utm.from_latlon(np.asarray(lats[0:10]), np.asarray(lons[0:10]))[0]

# convert to utm coords - northings
northings = utm.from_latlon(np.asarray(lats[0:10]), np.asarray(lons[0:10]))[1]

# make grid
gridx, gridy = np.meshgrid(eastings, northings)

# test following example from https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.interpolate.RBFInterpolator.html#scipy.interpolate.RBFInterpolator
obs_raw = np.asarray(utm.from_latlon(np.asarray(lats[0:100]), np.asarray(lons[0:100]))[0:2])
obs = np.stack((obs_raw[0], obs_raw[1]), axis=1)
vals = full_traffic_data["All_Detector_Vol"][0:100]
min1 = min(obs_raw[0])
max1 = max(obs_raw[0])
min2 = min(obs_raw[1])
max2 = max(obs_raw[1])
x_grid = np.mgrid[min1:max1:5j, min2:max2:5j]
x_flat = x_grid.reshape(2, -1).T  # reshape
y_flat = interp.RBFInterpolator(obs, vals)(x_flat)  # interpolate the values for that grid
y_grid = y_flat.reshape(5, 5)  # reshape as wanted
fig, ax = plt.subplots()
ax.pcolormesh(*x_grid, y_grid,shading='gouraud')
p = ax.scatter(*obs.T, c=vals, s=50, ec='k', vmin=-0.25, vmax=0.25)
fig.colorbar(p)
plt.show()

# TODO do this for means
# TODO test for sanity
# TODO try interpolate a value not a plot
# TODO plot better - dublin map?









# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # mapping set-up
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# # import shape file of ireland electoral districts
# eds_map = gpd.read_file(".\\data\\ireland_eds_map_data\\ireland_eds.shp")
#
# # filter shape file to just dublin electoral regions
# dub_eds = eds_map[eds_map["CO_NAME"] == "Dublin"]
#
# # all traffic data together


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# specific scenarios
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# # isolate weekday hour of 9am
# wd_9 = full_traffic_data[(full_traffic_data["Day_Type"] == "WD") & (full_traffic_data["Hour_in_Day"] == "09")]
#
# ax = gplt.polyplot(dub_eds, zorder=1, edgecolor="lightgrey", projection=gcrs.EuroPP())
# gplt.kdeplot(sites_geodf, cmap='Reds', shade=True, shade_lowest=False, ax=ax, projection=gcrs.EuroPP(), clip=dub_eds)

# TODO fix issue with clip
# TODO wrap into function which takes day and time args

