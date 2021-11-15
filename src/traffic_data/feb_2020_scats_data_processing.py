# this script is to process the scats hourly data from feb 2020
# to transform raw data to be usable in analysis
# working directory assumed project root
# NOTE: raw data file too large for github so not uploaded - zip file used instead

import pandas as pd

# load raw data
# NOTE: raw data file too large for github so not uploaded - zip file uploaded instead
feb_traffic = pd.read_csv(".\\data\\scats_detector_volume_202002.csv")

# isolate variables of interest
feb_traffic = feb_traffic[["End_Time", "Site", "Sum_Volume", "Avg_Volume"]]

# check for missing data
feb_traffic.isnull().values.any()  # False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# remove invalid sites (which have no location)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# load data on valid sites
valid_sites = pd.read_pickle(".\\data\\valid_scats_sites.pkl")

# isolate site ids for valid sites
valid_site_ids = valid_sites["SiteID"]

# keep only values from sites with valid ids
feb_traffic_data = feb_traffic[feb_traffic["Site"].isin(valid_site_ids)]

# ~~~~~~~~~~~~~~~~~~~~~~~
# process data
# ~~~~~~~~~~~~~~~~~~~~~~~

# want to split End_Time into day and hour of day
# day of month
feb_traffic["Day_in_Month"] = feb_traffic["End_Time"].str[8:10]

# hour of day
feb_traffic["Hour_in_Day"] = feb_traffic["End_Time"].str[11:13]

# Hour_in_Day is "time that one hour count period finishes"
# so want to change "00" to hour "24"
feb_traffic["Hour_in_Day"] = feb_traffic["Hour_in_Day"].replace(['00'], '24')

# create replacement dict for day in week
day_in_week_map = {"06": "04", "13": "04", "20": "04", "27": "04",
                   "07": "05", "14": "05", "21": "05", "28": "05",
                   "01": "06", "08": "06", "15": "06", "22": "06", "29": "06",
                   "02": "07", "09": "07", "16": "07", "23": "07",
                   "03": "01", "10": "01", "17": "01", "24": "01",
                   "04": "02", "11": "02", "18": "02", "25": "02",
                   "05": "03", "12": "03", "19": "03", "26": "03"}

# add variable specifying day of week
feb_traffic["Day_in_Week"] = feb_traffic["Day_in_Month"].map(day_in_week_map)

# create map to map day in week to weekday or weekend
weekday_weekend_map = {"01": "WD",
                       "02": "WD",
                       "03": "WD",
                       "04": "WD",
                       "05": "WD",
                       "06": "WE",
                       "07": "WE"}

# add variable specifying weekend or weekday
feb_traffic["Day_Type"] = feb_traffic["Day_in_Week"].map(weekday_weekend_map)

# re-check for NAN
feb_traffic.isnull().values.any()  # False

# save to pickle
feb_traffic.to_pickle(".\\data\\scats_feb2020_processed_data.pkl")
