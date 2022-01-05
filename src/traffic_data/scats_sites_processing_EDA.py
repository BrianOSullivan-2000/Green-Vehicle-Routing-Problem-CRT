# this script is to process and explore the scats sites
# and save an updated scats site file with only valid locations and geometries
# working directory assumed project root

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# import the scats site data
site_data = pd.read_pickle(".\\data\\scats_sites_with_elev.pkl")

# ~~~~~~~~~~~~~~~~~~~~~~~~~
# sanity check data
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# check for zero elevations
# these would be locations in water
zero_elev_index = site_data["Elev"][site_data["Elev"] == 0].index

len(zero_elev_index)  # 52

# checkout real-world location of 0 elev readings
site_data.iloc[zero_elev_index.tolist()]["Site_Description_Cap"]

# many locations are beside water - quays etc
# keep for traffic purposes but not for elevation purposes

# check for strange longitudes
min(site_data["Long"])  # sensible
max(site_data["Long"])  # not sensible, some 0 values
len(site_data[site_data["Long"] == 0])  # 42
max(site_data["Long"][site_data["Long"] != 0])  # sensible

# get index of the 0 longitude values
zero_long_index = site_data["Long"][site_data["Long"] == 0].index

# check for strange latitudes
min(site_data["Lat"])  # not sensible, some 0 values
max(site_data["Lat"])  # sensible
len(site_data[site_data["Lat"] == 0])  # 42
max(site_data["Lat"][site_data["Lat"] != 0])  # sensible

# get index of the 0 latitude values
zero_lat_index = site_data["Lat"][site_data["Lat"] == 0].index

# check if the zero vals for long and lat are the same sites
zero_lat_index == zero_long_index  # yes the same

# remove sites with no location data
# keeping elevation=0 sites that are not within lat/long=0
# because can still be used for traffic analyis
valid_sites = site_data.drop(zero_lat_index.tolist())

# ~~~~~~~~~~~~~~~~~~~~~~~~~
# plot sites
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# import shape file of ireland electoral districts
eds_map = gpd.read_file(".\\data\\ireland_eds_map_data\\ireland_eds.shp")

# filter shape file to just dublin electoral regions
dub_eds = eds_map[eds_map["CO_NAME"] == "Dublin"]

# get point of each valid site
geometry = gpd.points_from_xy(valid_sites["Long"], valid_sites["Lat"], crs="EPSG:4326")

# make a geodf of the sites for geo plotting
sites_geo_df = gpd.GeoDataFrame(valid_sites, geometry=geometry, crs="EPSG:4326")

# plot sites data on map of dublin
plt.rcParams["font.serif"] = "Times New Roman"
fig, ax = plt.subplots(1, 1, figsize=(7,7))
# plt.grid(True)
dub_eds.to_crs(epsg=4326).plot(ax=ax, color="darkseagreen")
sites_geo_df.plot(ax=ax, markersize=5, color="k")
plt.xlim((-6.4, -6.03))
plt.ylim((53.2, 53.44))
plt.title("Map of Dublin Showing SCATS Traffic Measurement Sites", fontsize=15)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig("scats_sites.png", bbox_inches="tight")

# save valid sites as plain df
valid_sites.to_pickle(".\\data\\valid_scats_sites.pkl")

# save geometry df
sites_geo_df.to_pickle(".\\data\\valid_scats_sites_geom.pkl")
