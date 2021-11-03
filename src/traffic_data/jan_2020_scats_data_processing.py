# this script is to process the scats hourly data from jan 2020
# to transform raw data to be usable in analysis
# working directory assumed project root
# NOTE: raw data file too large for github so not uploaded - zip file used instead

import pandas as pd
from zipfile import ZipFile

# extract scats_detector_volume_202001 from zip file
# will use csv file in this scrip to load
with ZipFile(".\\data\\scats_detector_volume_202001.zip", "r") as f:
    f.printdir()
    f.extractall(".\\data")

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
jan_traffic["Day_in_Month"] = jan_traffic["End_Time"].str[8:10]

# hour of day
jan_traffic["Hour_in_Day"] = jan_traffic["End_Time"].str[11:13]

# Hour_in_Day is "time that one hour count period finishes"
# so want to change "00" to hour "24"
jan_traffic["Hour_in_Day"] = jan_traffic["Hour_in_Day"].replace(['00'],'24')

# create replacement dict for day in week
day_in_week_map = {"06": "01", "13": "01", "20": "01", "27": "01",
                   "07": "02", "14": "02", "21": "02", "28": "02",
                   "01": "03", "08": "03", "15": "03", "22": "03", "29": "03",
                   "02": "04", "09": "04", "16": "04", "23": "04", "30": "04",
                   "03": "05", "10": "05", "17": "05", "24": "05", "31": "05",
                   "04": "06", "11": "06", "18": "06", "25": "06",
                   "05": "07", "12": "07", "19": "07", "26": "07"}

# add variable specifying day of week
jan_traffic["Day_in_Week"] = jan_traffic["Day_in_Month"].map(day_in_week_map)

# create map to map day in week to weekday or weekend
weekday_weekend_map = {"01": "WD",
                       "02": "WD",
                       "03": "WD",
                       "04": "WD",
                       "05": "WD",
                       "06": "WE",
                       "07": "WE"}

# add variable specifying weekend or weekday
jan_traffic["Day_Type"] = jan_traffic["Day_in_Week"].map(weekday_weekend_map)

# re-check for NAN
jan_traffic.isnull().values.any()  # False

# save to pickle
jan_traffic.to_pickle(".\\data\\scats_jan2020_processed_data.pkl")
