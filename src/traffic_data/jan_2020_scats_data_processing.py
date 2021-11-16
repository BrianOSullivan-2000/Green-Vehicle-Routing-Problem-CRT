# this script is to process the scats hourly data from jan 2020
# to transform raw data to be usable in analysis
# working directory assumed project root
# NOTE: raw data file too large for github so not uploaded - zip file used instead

import pandas as pd
from zipfile import ZipFile

# # uncomment if needed
# # extract scats_detector_volume_202001 from zip file
# # will use csv file in this scrip to load
# with ZipFile(".\\data\\scats_detector_volume_202001.zip", "r") as f:
#     f.printdir()
#     f.extractall(".\\data")

# load raw data
# NOTE: raw data file too large for github so not uploaded - zip file uploaded instead
jan_traffic = pd.read_csv(".\\data\\scats_detector_volume_202001.csv")

# isolate variables of interest
jan_traffic = jan_traffic[["End_Time", "Site", "Sum_Volume", "Avg_Volume"]]

# check for missing data
jan_traffic.isnull().values.any()  # False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# remove invalid sites (which have no location)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# load data on valid sites
valid_sites = pd.read_pickle(".\\data\\valid_scats_sites.pkl")

# isolate site ids for valid sites
valid_site_ids = valid_sites["SiteID"]

# keep only values from sites with valid ids
jan_traffic_data = jan_traffic[jan_traffic["Site"].isin(valid_site_ids)]

# ~~~~~~~~~~~~~~~~~~~~~~~
# process data
# ~~~~~~~~~~~~~~~~~~~~~~~

# want to split End_Time into day and hour of day
# day of month
jan_traffic["Day_in_Month"] = jan_traffic["End_Time"].str[8:10].astype("int64")

# hour of day
jan_traffic["Hour_in_Day"] = jan_traffic["End_Time"].str[11:13].astype("int64")

# Hour_in_Day is "time that one hour count period finishes"
# so want to change "00" to hour "24" of the previous day
# -1 to recorded day for hour 0
# and change hour 0 to hour 24
# will give SettingWithCopyWarning but is fine
jan_traffic["Day_in_Month"].loc[jan_traffic["Hour_in_Day"] == 0] = \
    jan_traffic["Day_in_Month"][jan_traffic["Hour_in_Day"] == 0] - 1

jan_traffic["Hour_in_Day"] = jan_traffic["Hour_in_Day"].replace([0], 24)

# create replacement dict for day in week
day_in_week_map = {6: 1, 13: 1, 20: 1, 27: 1,
                   7: 2, 14: 2, 21: 2, 28: 2,
                   1: 3, 8: 3, 15: 3, 22: 3, 29: 3,
                   2: 4, 9: 4, 16: 4, 23: 4, 30: 4,
                   3: 5, 10: 5, 17: 5, 24: 5, 31: 5,
                   4: 6, 11: 6, 18: 6, 25: 6,
                   5: 7, 12: 7, 19: 7, 26: 7}

# add variable specifying day of week
jan_traffic["Day_in_Week"] = jan_traffic["Day_in_Month"].map(day_in_week_map)

# create map to map day in week to weekday or weekend
weekday_weekend_map = {1: "WD",
                       2: "WD",
                       3: "WD",
                       4: "WD",
                       5: "WD",
                       6: "WE",
                       7: "WE"}

# add variable specifying weekend or weekday
jan_traffic["Day_Type"] = jan_traffic["Day_in_Week"].map(weekday_weekend_map)

# re-check for NAN
jan_traffic.isnull().values.any()  # False

# save to pickle
jan_traffic.to_pickle(".\\data\\scats_jan2020_processed_data.pkl")
