# this script is to explore the scats hourly data from jan 2020
# to look at traffic distributions by day and over certain time periods
# working directory assumed project root

import pandas as pd
import matplotlib.pyplot as plt

# load pickle with processed scats data for jan 2020
jan_traffic_data = pd.read_pickle(".\\data\\scats_jan2020_processed_data.pkl")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# distributions of traffic for different hours
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# look at summary stats by hour
summary_by_hr = jan_traffic_data.groupby("Hour_in_Day")["All_Detector_Vol"].describe()

# summary stats by hour and by weekday/weekend
summary_by_hr_wdwe = jan_traffic_data.groupby(["Hour_in_Day", "Day_Type"])["All_Detector_Vol"].describe()
summary_by_hr_wdwe = summary_by_hr_wdwe.reset_index(level="Day_Type")
summary_by_hr_wdwe.index = summary_by_hr_wdwe.index.astype(int)

# summary stats by hour and by day
summary_by_hr_d = jan_traffic_data.groupby(["Hour_in_Day", "Day_in_Week"])["All_Detector_Vol"].describe()
summary_by_hr_d = summary_by_hr_d.reset_index(level="Day_in_Week")
summary_by_hr_d.index = summary_by_hr_d.index.astype(int)

# plot of mean counts by WE/WD per hour
fig, ax = plt.subplots()
summary_by_hr_wdwe.groupby("Day_Type")["mean"].plot()
plt.xlim(0, 25)
plt.xlabel("Hour in Day")
plt.ylabel("Mean Site Vehicle Count")
plt.title("January 2020: Mean Site Vehicle Count by Hour in Day")
plt.grid(True)
plt.legend(title="Day Type")

# # plot of std of counts by WE/WD per hour
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
plt.title("January 2020: Mean Site Vehicle Count by Hour in Day")
plt.grid(True)
plt.legend(title="Day in Week")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# investigate the dip in Wed-Thu-Fri comp to Mon-Tue
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# checkout hypothesis that jan 1-3 are dragging down averages
with pd.option_context('display.max_rows', None):
    print(jan_traffic_data[["All_Detector_Vol", "Hour_in_Day", "Day_in_Month"]][
              jan_traffic_data["Day_in_Week"] == 3].groupby(["Hour_in_Day", "Day_in_Month"]).describe())

# wednesdays
# summarise by different wednesdays
wed_data = jan_traffic_data[["All_Detector_Vol", "Hour_in_Day", "Day_in_Month"]][jan_traffic_data["Day_in_Week"] == 3]
wed_data_summary = wed_data.groupby(["Hour_in_Day", "Day_in_Month"])["All_Detector_Vol"].describe()
wed_data_summary = wed_data_summary.reset_index(level="Day_in_Month")
wed_data_summary.index = wed_data_summary.index.astype(int)

# thursdays
# summarise by different thursdays
thu_data = jan_traffic_data[["All_Detector_Vol", "Hour_in_Day", "Day_in_Month"]][jan_traffic_data["Day_in_Week"] == 4]
thu_data_summary = thu_data.groupby(["Hour_in_Day", "Day_in_Month"])["All_Detector_Vol"].describe()
thu_data_summary = thu_data_summary.reset_index(level="Day_in_Month")
thu_data_summary.index = thu_data_summary.index.astype(int)

# fridays
# summarise by different fridays
fri_data = jan_traffic_data[["All_Detector_Vol", "Hour_in_Day", "Day_in_Month"]][jan_traffic_data["Day_in_Week"] == 5]
fri_data_summary = fri_data.groupby(["Hour_in_Day", "Day_in_Month"])["All_Detector_Vol"].describe()
fri_data_summary = fri_data_summary.reset_index(level="Day_in_Month")
fri_data_summary.index = fri_data_summary.index.astype(int)

# sat
# summarise by different saturdays
sat_data = jan_traffic_data[["All_Detector_Vol", "Hour_in_Day", "Day_in_Month"]][jan_traffic_data["Day_in_Week"] == 6]
sat_data_summary = sat_data.groupby(["Hour_in_Day", "Day_in_Month"])["All_Detector_Vol"].describe()
sat_data_summary = sat_data_summary.reset_index(level="Day_in_Month")
sat_data_summary.index = sat_data_summary.index.astype(int)

# sun
# summarise by different sundays
sun_data = jan_traffic_data[["All_Detector_Vol", "Hour_in_Day", "Day_in_Month"]][jan_traffic_data["Day_in_Week"] == 7]
sun_data_summary = sun_data.groupby(["Hour_in_Day", "Day_in_Month"])["All_Detector_Vol"].describe()
sun_data_summary = sun_data_summary.reset_index(level="Day_in_Month")
sun_data_summary.index = sun_data_summary.index.astype(int)

# plot wednesdays
fig, ax = plt.subplots()
wed_data_summary.groupby("Day_in_Month")["mean"].plot()
plt.xlim(0, 25)
plt.xlabel("Hour in Day")
plt.ylabel("Mean Site Vehicle Count")
plt.title("Wednesdays in January: Mean Site Vehicle Count by Hour in Day")
plt.grid(True)
plt.legend(title="Day in Month")

# plot thursdays
fig, ax = plt.subplots()
thu_data_summary.groupby("Day_in_Month")["mean"].plot()
plt.xlim(0, 25)
plt.xlabel("Hour in Day")
plt.ylabel("Mean Site Vehicle Count")
plt.title("Thursdays in January: Mean Site Vehicle Count by Hour in Day")
plt.grid(True)
plt.legend(title="Day in Month")

# plot fridays
fig, ax = plt.subplots()
fri_data_summary.groupby("Day_in_Month")["mean"].plot()
plt.xlim(0, 25)
plt.xlabel("Hour in Day")
plt.ylabel("Mean Site Vehicle Count")
plt.title("Fridays in January: Mean Site Vehicle Count by Hour in Day")
plt.grid(True)
plt.legend(title="Day in Month")

# plot saturdays
fig, ax = plt.subplots()
sat_data_summary.groupby("Day_in_Month")["mean"].plot()
plt.xlim(0, 25)
plt.xlabel("Hour in Day")
plt.ylabel("Mean Site Vehicle Count")
plt.title("Saturdays in January: Mean Site Vehicle Count by Hour in Day")
plt.grid(True)
plt.legend(title="Day in Month")

# plot sundays
fig, ax = plt.subplots()
sun_data_summary.groupby("Day_in_Month")["mean"].plot()
plt.xlim(0, 25)
plt.xlabel("Hour in Day")
plt.ylabel("Mean Site Vehicle Count")
plt.title("Sundays in January: Mean Site Vehicle Count by Hour in Day")
plt.grid(True)
plt.legend(title="Day in Month")

# plotting for overleaf
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
wed_data_summary.groupby("Day_in_Month")["mean"].plot(ax=ax1)
thu_data_summary.groupby("Day_in_Month")["mean"].plot(ax=ax2)
fri_data_summary.groupby("Day_in_Month")["mean"].plot(ax=ax3)
sat_data_summary.groupby("Day_in_Month")["mean"].plot(ax=ax4)
ax1.title.set_text("Wednesdays in January")
ax2.title.set_text("Thursdays in January")
ax3.title.set_text("Fridays in January")
ax4.title.set_text("Saturdays in January")
ax3.xaxis.label.set_visible(False)
ax4.xaxis.label.set_visible(False)
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)
ax1.legend(title="Day in Month", loc="upper left")
ax2.legend(title="Day in Month", loc="upper left")
ax3.legend(title="Day in Month", loc="upper left")
ax4.legend(title="Day in Month", loc="upper left")
fig.suptitle('Mean Site Vehicle Count by Hour in Day')
fig.supxlabel('Hour in Day')
fig.supylabel('Mean Site Vehicle Count')

# so want to remove three dates as are inconsistent
# wed jan 1
# thu jan 2
# fri jan 3
# done in plotting_traffic_cover script

# weekday and weekend seem to have more significant difference
# so get relative traffic volumes for these cases separately

# weekday
wd_jan_traffic = jan_traffic_data[jan_traffic_data["Day_Type"] == "WD"].copy()

# range normalise b/t 1 and 0
wd_norm_traffic_val = (wd_jan_traffic["All_Detector_Vol"] - min(wd_jan_traffic["All_Detector_Vol"])) /\
                      (max(wd_jan_traffic["All_Detector_Vol"]) - min(wd_jan_traffic["All_Detector_Vol"]))

wd_jan_traffic["Norm_Vol_WD"] = wd_norm_traffic_val

# weekend
we_jan_traffic = jan_traffic_data[jan_traffic_data["Day_Type"] == "WE"].copy()

# range normalise b/t 1 and 0
we_jan_traffic["Norm_Traffic_Val"] = (we_jan_traffic["All_Detector_Vol"] - min(we_jan_traffic["All_Detector_Vol"])) /\
                                     (max(we_jan_traffic["All_Detector_Vol"]) - min(we_jan_traffic["All_Detector_Vol"]))

# TODO: update with further stats if needed
# TODO: make table for report
# TODO: split by sites?
