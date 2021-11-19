# this script is to process the scats hourly data from feb 2020
# to transform raw data to be usable in analysis
# working directory assumed project root
# NOTE: raw data file too large for github so not uploaded - zip file used instead

import pandas as pd

# load raw data
# NOTE: raw data file too large for github so not uploaded - zip file uploaded instead
feb_traffic_data = pd.read_csv(".\\data\\scats_detector_volume_202002.csv")

# isolate variables of interest
feb_traffic_data = feb_traffic_data[["End_Time", "Site", "Sum_Volume"]]

# check for missing data
feb_traffic_data.isnull().values.any()  # False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# remove invalid sites (which have no location)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# load data on valid sites
valid_sites = pd.read_pickle(".\\data\\valid_scats_sites.pkl")

# isolate site ids for valid sites
valid_site_ids = valid_sites["SiteID"]

# keep only values from sites with valid ids
feb_traffic_data = feb_traffic_data[feb_traffic_data["Site"].isin(valid_site_ids)]

# ~~~~~~~~~~~~~~~~~~~~~~~
# process data
# ~~~~~~~~~~~~~~~~~~~~~~~

# want to combine all detectors from indiv site into one measurement
# as detectors are not geographically differentiated from e/o

# get sum total of measurements from all detectors per site per time
# assign to column
feb_traffic_data["All_Detector_Vol"] = feb_traffic_data.groupby(['End_Time', 'Site'])['Sum_Volume'].transform('sum')

# remove duplicates of site and time combo
# so left with just total of all detectors for hour at site
feb_traffic_data = feb_traffic_data.drop_duplicates(subset=['End_Time', 'Site'])

# remove sum_volume column as no longer relevant
feb_traffic_data = feb_traffic_data.drop(columns="Sum_Volume")

# want to split End_Time into day and hour of day
# day of month
feb_traffic_data["Day_in_Month"] = feb_traffic_data["End_Time"].str[8:10].astype("int64")

# hour of day
feb_traffic_data["Hour_in_Day"] = feb_traffic_data["End_Time"].str[11:13].astype("int64")

# Hour_in_Day is "time that one hour count period finishes"
# so want to change "00" to hour "24" of the previous day
# -1 to recorded day for hour 0
# and change hour 0 to hour 24
# will give SettingWithCopyWarning but is fine
feb_traffic_data["Day_in_Month"].loc[feb_traffic_data["Hour_in_Day"] == 0] = \
    feb_traffic_data["Day_in_Month"].loc[feb_traffic_data["Hour_in_Day"] == 0] - 1
feb_traffic_data["Hour_in_Day"] = feb_traffic_data["Hour_in_Day"].replace([0], 24)

# replace day 0 with day 31
# as measurement at hour 0 of feb 1 is actually hour 24 for 31 jan
feb_traffic_data["Day_in_Month"] = feb_traffic_data["Day_in_Month"].replace([0], 31)

# create replacement dict for day in week
# 31 refers to 31 jan = friday
day_in_week_map = {6: 4, 13: 4, 20: 4, 27: 4,
                   7: 5, 14: 5, 21: 5, 28: 5,
                   1: 6, 8: 6, 15: 6, 22: 6, 29: 6,
                   2: 7, 9: 7, 16: 7, 23: 7,
                   3: 1, 10: 1, 17: 1, 24: 1,
                   4: 2, 11: 2, 18: 2, 25: 2,
                   5: 3, 12: 3, 19: 3, 26: 3,
                   31: 5}

# add variable specifying day of week
feb_traffic_data["Day_in_Week"] = feb_traffic_data["Day_in_Month"].map(day_in_week_map)

# create map to map day in week to weekday or weekend
weekday_weekend_map = {1: "WD",
                       2: "WD",
                       3: "WD",
                       4: "WD",
                       5: "WD",
                       6: "WE",
                       7: "WE"}

# add variable specifying weekend or weekday
feb_traffic_data["Day_Type"] = feb_traffic_data["Day_in_Week"].map(weekday_weekend_map)

# re-check for NAN
feb_traffic_data.isnull().values.any()  # False

# save to pickle
feb_traffic_data.to_pickle(".\\data\\scats_feb2020_processed_data.pkl")
