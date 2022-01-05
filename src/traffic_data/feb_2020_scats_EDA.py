# this script is to explore the scats hourly data from feb 2020
# to look at traffic distributions by day and over certain time periods
# working directory assumed project root

import pandas as pd
import matplotlib.pyplot as plt

# load pickle with processed scats data for feb 2020
feb_traffic_data = pd.read_pickle(".\\data\\scats_feb2020_processed_data.pkl")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# distributions of traffic for different hours
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# look at summary stats by hour
summary_by_hr = feb_traffic_data.groupby("Hour_in_Day")["All_Detector_Vol"].describe()

# summary stats by hour and by weekday/weekend
summary_by_hr_wdwe = feb_traffic_data.groupby(["Hour_in_Day", "Day_Type"])["All_Detector_Vol"].describe()
summary_by_hr_wdwe = summary_by_hr_wdwe.reset_index(level="Day_Type")
summary_by_hr_wdwe.index = summary_by_hr_wdwe.index.astype(int)

# summary stats by hour and by day
summary_by_hr_d = feb_traffic_data.groupby(["Hour_in_Day", "Day_in_Week"])["All_Detector_Vol"].describe()
summary_by_hr_d = summary_by_hr_d.reset_index(level="Day_in_Week")
summary_by_hr_d.index = summary_by_hr_d.index.astype(int)

# plot of mean counts by WE/WD per hour
fig, ax = plt.subplots(1, 1, figsize=(7,7))
plt.rcParams["font.serif"] = "Times New Roman"
summary_by_hr_wdwe.groupby("Day_Type")["mean"].plot()
plt.xlim(1, 24)
plt.xlabel("Hour in Day")
plt.ylabel("Mean Site Vehicle Count")
plt.title("February 2020: Mean Site Vehicle Count by Hour in Day", fontsize=15)
plt.grid(True, alpha=0.3)
plt.legend(title="Day Type")

# plot of std of counts by WE/WD per hour
# fig, ax = plt.subplots()
# summary_by_hr_wdwe.groupby("Day_Type")["std"].plot()
# plt.xlim(0, 25)
# plt.xlabel("Hour in Day")
# plt.ylabel("Std of Vehicle Count")
# plt.title("Std of Vehicle Count by Hour of Day")
# plt.grid(True)
# plt.legend(title="Day Type")

# plot of mean counts by day of week per hour
fig, ax = plt.subplots()
summary_by_hr_d.groupby("Day_in_Week")["mean"].plot()
plt.xlim(0, 25)
plt.xlabel("Hour in Day")
plt.ylabel("Mean Site Vehicle Count")
plt.title("February 2020: Mean Site Vehicle Count by Hour in Day")
plt.grid(True)
plt.legend(title="Day in Week")

# weekday and weekend seem to have more significant difference
# so get relative traffic volumes for these cases separately

# weekday
wd_feb_traffic = feb_traffic_data[feb_traffic_data["Day_Type"] == "WD"].copy()

# range normalise b/t 1 and 0
wd_norm_traffic_val = (wd_feb_traffic["All_Detector_Vol"] - min(wd_feb_traffic["All_Detector_Vol"])) /\
                      (max(wd_feb_traffic["All_Detector_Vol"]) - min(wd_feb_traffic["All_Detector_Vol"]))

wd_feb_traffic["Norm_Vol_WD"] = wd_norm_traffic_val

# weekend
we_feb_traffic = feb_traffic_data[feb_traffic_data["Day_Type"] == "WE"].copy()

# range normalise b/t 1 and 0
we_feb_traffic["Norm_Traffic_Val"] = (we_feb_traffic["All_Detector_Vol"] - min(we_feb_traffic["All_Detector_Vol"])) /\
                                     (max(we_feb_traffic["All_Detector_Vol"]) - min(we_feb_traffic["All_Detector_Vol"]))

# TODO: update with further stats if needed
# TODO: make table for report
# TODO: split by sites?
