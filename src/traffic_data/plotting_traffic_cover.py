# this script plots traffic as a kde on dublin

# import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
# import geoplot as gplt
# import geoplot.crs as gcrs
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

#full_traffic_data["Elev"] = np.zeros(full_traffic_data.shape[0])
#full_traffic_data["lon"] = np.zeros(full_traffic_data.shape[0])
#full_traffic_data["lat"] = np.zeros(full_traffic_data.shape[0])
#full_traffic_data["geometry"] = np.zeros(full_traffic_data.shape[0])


#for id in sites_geodf['SiteID'].values:

#    elev = sites_geodf.loc[sites_geodf['SiteID'] == id, "Elev"]
#    geometry = sites_geodf.loc[sites_geodf['SiteID'] == id, "geometry"]
#    x, y = geometry.x, geometry.y

#    full_traffic_data.loc[full_traffic_data["Site"] == id, 'Elev'] = elev
#    full_traffic_data.loc[full_traffic_data["Site"] == id, 'lon'] = x
#    full_traffic_data.loc[full_traffic_data["Site"] == id, 'lat'] = y


# assign matching elev
full_traffic_data["Elev"] = full_traffic_data.apply(site_elev, axis=1)

# assign matching geometry
full_traffic_data["geometry"] = full_traffic_data.apply(site_geom, axis=1)

# TODO test for sanity
# TODO plot better - dublin map?

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# interpolation function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def traffic_interpolator(day_type, hour_in_day, num_neighbours):
    # isolate relevant data
    specific_data = full_traffic_data[(full_traffic_data["Day_Type"] == day_type) \
                                      & (full_traffic_data["Hour_in_Day"] == hour_in_day)].copy

    # take mean of each site for these
    # throws a SettingWithCopyWarning but is ok
    specific_data["Mean_Detector_Vol_hr"] = specific_data.groupby(["Site"])["All_Detector_Vol"].transform('mean')

    # remove duplicates of sites
    # so left with just mean of each site at this hour and day type
    specific_data = specific_data.drop_duplicates(subset=['Site'])

    # drop the normed vol col as now using the mean value
    specific_data = specific_data.drop(columns="All_Detector_Vol")

    # isolate latitudes and longitudes
    lats = specific_data["geometry"].apply(lambda q: q.y)
    lons = specific_data["geometry"].apply(lambda q: q.x)

    # convert locations to utm and correct format
    obs_raw = np.asarray(utm.from_latlon(np.asarray(lats), np.asarray(lons))[0:2])
    obs = np.stack((obs_raw[0], obs_raw[1]), axis=1)

    # isolate value for each point
    vals = wd_0900["Mean_Detector_Vol_hr"]

    # make interpolator
    # grbf kernel with epsilon=1 shape parameter
    interpolator = interp.RBFInterpolator(obs, vals, neighbors=num_neighbours, kernel="gaussian", epsilon=1)

    return interpolator


our_interpolator = traffic_interpolator("WD", 9, 5)

y_flat = our_interpolator(x_flat)  # interpolate the values for that grid
y_grid = y_flat.reshape(5, 5)  # reshape as wanted
fig, ax = plt.subplots()
ax.pcolormesh(*x_grid, y_grid, shading='gouraud')
p = ax.scatter(*obs.T, c=vals, s=50, ec='k')
fig.colorbar(p)

our_interpolator([ [ 692899.19445525, 5923903.80728167]])



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
