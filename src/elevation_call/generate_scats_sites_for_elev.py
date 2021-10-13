# script to generate the required position data
# ie latitude and longitude
# for use in getting elevation data
# from open-elevation API
# working directory assumed project root

import pandas as pd

# load data
site_data = pd.read_csv(".\\data\\scats_sites_post.csv")

# get rid of final row as NA
site_data = site_data.drop([895])

# create df of lat and long position data in cols
site_data = site_data.drop(columns=["_id", "SiteID", "Site_Description_Cap", "Site_Description_Lower", "Region"])

# update column names
site_data = site_data.rename(columns={"Lat": "latitude", "Long": "longitude"})

# save to pickle
site_data.to_pickle(".\\data\\lat_long_to_query.pkl")