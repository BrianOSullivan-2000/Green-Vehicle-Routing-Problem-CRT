# this script plots traffic as a kde on dublin

# import geopandas as gpd
import pandas as pd
# import matplotlib.pyplot as plt
# import geoplot as gplt
# import geoplot.crs as gcrs
import numpy as np
import utm
import scipy.interpolate as interp
from scipy.spatial import cKDTree

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
#full_traffic_data["Elev"] = full_traffic_data.apply(site_elev, axis=1)

# assign matching geometry
#full_traffic_data["geometry"] = full_traffic_data.apply(site_geom, axis=1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# interpolation function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
day_type, hour_in_day, num_neighbours = "WD", 9, 5

def traffic_interpolator(day_type, hour_in_day, num_neighbours):
    # isolate relevant data
    specific_data = full_traffic_data[(full_traffic_data["Day_Type"] == day_type)
                                      & (full_traffic_data["Hour_in_Day"] == hour_in_day)].copy()

    # take mean of each site for these
    # throws a SettingWithCopyWarning but is ok
    specific_data["Mean_Detector_Vol_hr"] = specific_data.groupby(["Site"])["All_Detector_Vol"].transform('mean')

    # filter out values greater than 10 and less than 3000
    #specific_data = specific_data[specific_data["Mean_Detector_Vol_hr"] < 3000]
    #specific_data = specific_data[specific_data["Mean_Detector_Vol_hr"] > 10]

    # remove duplicates of sites
    # so left with just mean of each site at this hour and day type
    specific_data = specific_data.drop_duplicates(subset=['Site'])

    # drop the normed vol col as now using the mean value
    specific_data = specific_data.drop(columns="All_Detector_Vol")

    # NOTE to KATE: Brian here, I was having issues using the site_elev, site_geom
    # functions earlier on in the file. I believe it's simply because the raw traffic
    # data is too large with 1,000,000+ points, it would take around 8 hours to
    # assign elevations and geometries to the entire dataframe. Instead I put the
    # functions inside the interpolator, since we only need the elevations and geometries
    # for the rows we are concerned with

    # assign matching elev and geometry
    specific_data["Elev"] = specific_data.apply(site_elev, axis=1)
    specific_data["geometry"] = specific_data.apply(site_geom, axis=1)

    # isolate latitudes and longitudes
    lats = specific_data["geometry"].apply(lambda q: q.y)
    lons = specific_data["geometry"].apply(lambda q: q.x)

    # convert locations to utm and correct format
    obs_raw = np.asarray(utm.from_latlon(np.asarray(lats), np.asarray(lons))[0:2])
    obs = np.stack((obs_raw[0], obs_raw[1]), axis=1)

    # isolate value for each point
    vals = specific_data["Mean_Detector_Vol_hr"]

    # make interpolator
    # grbf kernel with epsilon=1 shape parameter
    interpolator = interp.RBFInterpolator(obs, vals, neighbors=num_neighbours, kernel="gaussian", epsilon=1)

    return interpolator


def write_traffic_file(filename, day_type, hour_in_day, num_neighbours, lon_b, lat_b, h):

    # Load in interpolator
    our_interpolator = traffic_interpolator(day_type, hour_in_day, num_neighbours)

    # Grid that will be interpolated on to
    x, y = np.arange(lon_b[0], lon_b[1], h), np.arange(lat_b[0], lat_b[1], h)
    x, y = np.round(x, 4), np.round(y, 4)
    yy, xx = np.meshgrid(y, x)

    # Convert to utm format
    grid_obs_raw = np.asarray(utm.from_latlon(yy.ravel(), xx.ravel())[0:2])
    grid_obs = np.stack((grid_obs_raw[0], grid_obs_raw[1]), axis=1)

    # Compute interpolated traffic levels
    traffic = our_interpolator(grid_obs)

    # Express data as dataframe for writing
    data = {"longitude": xx.flatten(), "latitude": yy.flatten(), "traffic": traffic}
    traffic_df = pd.DataFrame(data=data)

    # Interpolated data is slightly too large for github, cut it in half
    #traffic_df = traffic_df.iloc[::2, :]

    # Save to data folder
    traffic_df.to_pickle("data/traffic_matrices/{}".format(filename))


# Bounding box and stepsize for grid
lon_b = (-6.42, -6.10)
lat_b = (53.25, 53.45)
h = 0.001

# Save a default test file
traffic = write_traffic_file("weekend_peak.pkl", "WE", 14, 5, lon_b, lat_b, h)


# In[1]

import geopandas as gpd
import matplotlib.pyplot as plt

# Plotting code, read df
df = pd.read_pickle("data/traffic_matrices/weekday_peak.pkl")

# Convert df to geodatafram with point geometry
geometry = gpd.points_from_xy(df['longitude'], df['latitude'])
names = {'traffic':df['traffic'], 'longitude':df['longitude'], 'latitude':df['latitude']}
gdf = gpd.GeoDataFrame(pd.DataFrame(data=names), columns=['traffic'], geometry=geometry, crs={'init' : 'epsg:4326'})

# Get the map overlay of Dublin
dub_df = gpd.read_file("../Brians_Lab/data/counties.shp")
dub_df = dub_df.set_crs(epsg=4326)
dub_df = dub_df[dub_df["NAME_TAG"]=="Dublin"]

dub_df = dub_df.to_crs('epsg:4326')
gdf = gdf.to_crs('epsg:4326')
# Make GeoDataFrame smaller for performance
#gdf = gdf.iloc[::10, :]

gdf = gpd.sjoin(gdf, dub_df).iloc[:, 0:2]


# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10,10))
dub_df.plot(ax=ax, color='none', edgecolor="k", alpha=1, zorder=3)

gdf.plot(ax=ax, column='traffic',  cmap='plasma', marker=',', markersize=5,
             alpha=1, zorder=2, legend=True, vmin=0, vmax=5000)
# Plot
plt.xlim(lon_b)
plt.ylim(lat_b)

plt.title("Mean Hourly Site Vehicle Count, Weekday 9am", size=15)
# plt.savefig("data/figures/Dublin_traffic_fig.jpeg", dpi=300)
plt.show()
