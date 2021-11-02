# script to link elevation data from open-elevation api
# with scats traffic record sites
# coalesce all information into one dataframe
# working directory assumed project root

import pandas as pd

# read in scats site data
# drop last entry as NA
scats_data = pd.read_csv(".\\data\\scats_sites_post.csv")
scats_data = scats_data.drop([895])

# read in scraped elevation data
elev_data = pd.read_pickle(".\\data\\post_and_elev.pkl")

# ensure lats and longs match for both scats and elev
# then can simply append the elevation column to scats_data
# if true can continue
(scats_data["Lat"] == elev_data["latitude"]).sum() == len(elev_data)
(scats_data["Long"] == elev_data["longitude"]).sum() == len(elev_data)

# append elev column to full scats site df
scats_data["Elev"] = elev_data["elevation"]

# save scats site data with elev as pickle file
scats_data.to_pickle(".\\data\\scats_sites_with_elev.pkl")
