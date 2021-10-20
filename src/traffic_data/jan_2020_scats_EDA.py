# this script is to explore the scats hourly data from jan 2020
# to look at traffic distributions by day and over certain time periods
# working directory assumed project root

from zipfile import ZipFile
import pandas as pd

# unzip processed scats data for jan 2020
with ZipFile(".\\data\\scats_jan2020_processed_data.zip", "r") as f:
    f.printdir()
    f.extractall(".\\data")

# load pickle with processed scats data for jan 2020
jan_traffic_data = pd.read_pickle(".\\data\\scats_jan2020_processed_data.pkl")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# distributions of traffic for different hours
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# look at summary stats by hour
summary_by_hr = jan_traffic_data.groupby("Hour_in_Day")["Sum_Volume"].describe()

# summary stats by hour and by weekday/weekend
summary_by_hr_wdwe = jan_traffic_data.groupby(["Hour_in_Day", "Day_Type"])["Sum_Volume"].describe()

# TODO: update with further stats if needed
# TODO: make table for report
# TODO: split by sites?
