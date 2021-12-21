# import geopandas as gpd
import pandas as pd
# import matplotlib.pyplot as plt
# import geoplot as gplt
# import geoplot.crs as gcrs
import numpy as np
import utm
import scipy.interpolate as interp
import random

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
def traffic_interpolator(specific_data, num_neighbours, eps):

    # take mean of each site for these
    # throws a SettingWithCopyWarning but is ok
    specific_data["Mean_Detector_Vol_hr"] = specific_data.groupby(["Site"])["All_Detector_Vol"].transform('mean')

    # remove duplicates of sites
    # so left with just mean of each site at this hour and day type
    specific_data = specific_data.copy().drop_duplicates(subset=['Site'])

    # drop the normed vol col as now using the mean value
    specific_data = specific_data.drop(columns="All_Detector_Vol")

    # assign matching elev and geometry
    specific_data["Elev"] = specific_data.apply(site_elev, axis=1)
    specific_data["geometry"] = specific_data.apply(site_geom, axis=1)

    # isolate latitudes and longitudes
    lats = specific_data["geometry"].apply(lambda q: q.y).copy()
    lons = specific_data["geometry"].apply(lambda q: q.x).copy()

    # convert locations to utm and correct format
    obs_raw = np.asarray(utm.from_latlon(np.asarray(lats), np.asarray(lons))[0:2])
    obs = np.stack((obs_raw[0], obs_raw[1]), axis=1)

    # isolate value for each point
    vals = specific_data["Mean_Detector_Vol_hr"]

    # make interpolator
    # grbf kernel with epsilon=1 shape parameter
    interpolator = interp.RBFInterpolator(obs, vals, neighbors=num_neighbours, kernel="gaussian", epsilon=eps)

    return interpolator

# set up values to check
eps_vals = [0.5, 1, 2]
num_neighbour_vals = [3, 5, 10]
times = [6, 9, 12, 15, 18, 21]

# set up training and testing sites
K = 10  # number of folds
all_sites = np.asarray(full_traffic_data["Site"].unique())
num_sites = len(full_traffic_data["Site"].unique())
n_train = int(num_sites*0.85)
train_sites = np.random.choice(all_sites, size=n_train, replace=False)
test_sites = all_sites[~np.isin(all_sites, train_sites)]

# split into training and testing data
train_data = full_traffic_data[np.isin(full_traffic_data["Site"], train_sites)].copy()
test_data = full_traffic_data[np.isin(full_traffic_data["Site"], test_sites)].copy()

mse_storage = np.zeros((K, len(eps_vals), len(num_neighbour_vals)))
folds = np.repeat(np.arange(0, K, 1), np.ceil(n_train/K))
random.shuffle(folds)  # mutates in place
folds = folds[0:n_train]

for k in range(K):
    # split into train and val fold sites
    train_fold_sites = train_sites[folds != k].copy()
    val_fold_sites = train_sites[folds == k].copy()

    # separate out train and val fold data
    train_folds = train_data[np.isin(train_data["Site"], train_fold_sites)].copy()
    val_folds = train_data[np.isin(train_data["Site"], val_fold_sites)].copy()

    for eps_ind, i in enumerate(eps_vals):
        print(i)
        for neigh_ind, j in enumerate(num_neighbour_vals):
            mse_storage_j = []
            for t in times:
                # filter data to specific scenario
                filtered_data = train_folds[(train_folds["Hour_in_Day"] == t) &
                                            (train_folds["Day_Type"] == "WD")]

                # create interpolator on training fold
                train_interp = traffic_interpolator(filtered_data, j, i)

                # now check performance with val fold
                # get true values
                val_folds["Mean_Detector_Vol_hr"] = val_folds.groupby(["Site"])["All_Detector_Vol"].transform('mean')

                # remove duplicates of sites
                # so left with just mean of each site at this hour and day type
                val_folds = val_folds.drop_duplicates(subset=['Site'])

                # assign matching elev and geometry
                val_folds["Elev"] = val_folds.apply(site_elev, axis=1)
                val_folds["geometry"] = val_folds.apply(site_geom, axis=1)

                # isolate latitudes and longitudes
                lats_val = val_folds["geometry"].apply(lambda q: q.y).copy()
                lons_val = val_folds["geometry"].apply(lambda q: q.x).copy()

                # convert locations to utm and correct format
                obs_raw_val = np.asarray(utm.from_latlon(np.asarray(lats_val), np.asarray(lons_val))[0:2])
                obs_val = np.stack((obs_raw_val[0], obs_raw_val[1]), axis=1)

                # interpolate values for val fold
                val_interp_results = train_interp(obs_val)

                # true values
                true = val_folds["Mean_Detector_Vol_hr"].to_numpy()

                # calc mse
                mse = (np.sum((true - val_interp_results)**2)) / len(true)

                # store mse for this time
                mse_storage_j.append(mse)

            # store mean mse for this number of neighbours and eps vals
            mse_storage[k, eps_ind, neigh_ind] = np.mean(mse_storage_j)
